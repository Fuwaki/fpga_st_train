//! Training supervision: pseudo ground truth on disk and/or an online teacher.

use std::sync::mpsc::Receiver;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use image::RgbImage;

use crate::backend::TrainDevice;
use crate::config::{Config, TeacherKind};
use crate::data::{load_rgb, resize_short_edge, rgb_batch_nchw, rgb_to_tensor};
use crate::teacher::weights;
use crate::teacher::{DistillTeacher, OnnxTeacher, Stylizer};
use crate::{InferBackend, TrainBackend};

/// Optional online teacher; pseudo-GT comes from the dataset batch.
pub struct Supervision {
    teacher: Option<LiveTeacher>,
    /// Style-packet teachers, in student style order. Empty for a one-teacher run.
    packets: Vec<PacketTeacher>,
}

struct PacketTeacher {
    name: String,
    teacher: OnnxTeacher,
}

/// One style a sample image should be compared against.
pub struct StyleView<'a> {
    pub name: &'a str,
    pub teacher: &'a dyn DistillTeacher,
}

enum LiveTeacher {
    /// Session thread can accept the next batch while CUDA trains.
    Onnx(OnnxTeacher),
    Device(Box<dyn DistillTeacher>),
}

impl Supervision {
    /// Build supervision from run config (loads the configured teacher when enabled).
    pub fn try_from_config(config: &Config, device: &TrainDevice) -> Result<Self> {
        if !config.teacher.styles.is_empty() {
            let started = Instant::now();
            let mut packets = Vec::with_capacity(config.teacher.styles.len());
            for style in &config.teacher.styles {
                let kind = TeacherKind::parse(&style.kind)?;
                packets.push(PacketTeacher {
                    name: style.name.clone(),
                    teacher: OnnxTeacher::open(kind, &style.model)
                        .with_context(|| format!("style '{}'", style.name))?,
                });
            }
            log::info!(
                "{} style teachers loaded in {:.2} s",
                packets.len(),
                started.elapsed().as_secs_f32()
            );
            return Ok(Self {
                teacher: None,
                packets,
            });
        }
        let teacher = if config.teacher.enabled {
            Some(load_teacher(config, device)?)
        } else {
            None
        };
        Ok(Self {
            teacher,
            packets: Vec::new(),
        })
    }

    /// More than one style packet shares the step.
    pub fn is_multi(&self) -> bool {
        self.packets.len() > 1
    }

    pub fn n_styles(&self) -> usize {
        if self.packets.is_empty() {
            1
        } else {
            self.packets.len()
        }
    }

    pub fn style_name(&self, index: usize) -> &str {
        self.packets
            .get(index)
            .map(|packet| packet.name.as_str())
            .unwrap_or("")
    }

    /// Teachers for sample images. A one-teacher run is a single unnamed entry.
    pub fn sample_styles(&self) -> Vec<StyleView<'_>> {
        if !self.packets.is_empty() {
            return self
                .packets
                .iter()
                .map(|packet| StyleView {
                    name: packet.name.as_str(),
                    teacher: &packet.teacher,
                })
                .collect();
        }
        match self.teacher() {
            Some(teacher) => vec![StyleView { name: "", teacher }],
            None => Vec::new(),
        }
    }

    /// Run one packet teacher. Content is RGB `[0, 1]`; the student quantises itself.
    pub fn stylize_style(
        &self,
        index: usize,
        content: Tensor<TrainBackend, 4>,
    ) -> Result<Tensor<TrainBackend, 4>> {
        let packet = self
            .packets
            .get(index)
            .with_context(|| format!("style index {index} is outside the teacher list"))?;
        let out = packet.teacher.stylize(content.inner())?;
        Ok(Tensor::<TrainBackend, 4>::from_inner(out))
    }

    pub fn teacher(&self) -> Option<&dyn DistillTeacher> {
        if let Some(packet) = self.packets.first() {
            return Some(&packet.teacher);
        }
        match &self.teacher {
            Some(LiveTeacher::Onnx(teacher)) => Some(teacher),
            Some(LiveTeacher::Device(teacher)) => Some(teacher.as_ref()),
            None => None,
        }
    }

    /// WhiteBox and rain-princess run off the training GPU, so the next batch
    /// can be queued behind the student backward. Several styles run in order.
    pub fn pipelines_on_host(&self) -> bool {
        if self.packets.len() > 1 {
            return false;
        }
        if self.packets.len() == 1 {
            return true;
        }
        matches!(self.teacher, Some(LiveTeacher::Onnx(_)))
    }

    /// Queue one host batch. `None` means this teacher stays on the training thread.
    pub fn submit_host_batch(
        &self,
        images: &[RgbImage],
    ) -> Option<Result<Receiver<Result<Vec<f32>>>>> {
        let LiveTeacher::Onnx(teacher) = self.teacher.as_ref()? else {
            return None;
        };
        let (nchw, h, w) = rgb_batch_nchw(images);
        Some(teacher.submit_nchw(nchw, images.len(), h, w))
    }

    /// Target image and optional teacher feature maps for one training batch.
    pub fn resolve_batch(
        &self,
        content: Tensor<TrainBackend, 4>,
        provided_target: Option<Tensor<TrainBackend, 4>>,
        need_teacher_feats: bool,
    ) -> Result<(Tensor<TrainBackend, 4>, Vec<Tensor<TrainBackend, 4>>)> {
        let (teacher_out, teacher_feats): (
            Option<Tensor<InferBackend, 4>>,
            Vec<Tensor<InferBackend, 4>>,
        ) = match self.teacher() {
            Some(teacher) if provided_target.is_none() || need_teacher_feats => {
                // Feature distillation needs the decoder intermediates. For the normal pixel
                // and gradient losses, however, constructing and retaining those maps is
                // unnecessary overhead; use the plain stylize path instead.
                if need_teacher_feats {
                    if !teacher.has_features() {
                        bail!(
                            "loss.feature > 0 requires a MicroAST teacher (teacher.kind = \"microast\")"
                        );
                    }
                    let (out, feats) = teacher.features(content.clone().inner())?;
                    (Some(out), feats)
                } else {
                    (Some(teacher.stylize(content.inner())?), Vec::new())
                }
            }
            _ => (None, Vec::new()),
        };

        let target = match (provided_target, teacher_out) {
            (Some(target), _) => target,
            (None, Some(out)) => Tensor::<TrainBackend, 4>::from_inner(out),
            (None, None) => bail!(
                "some samples have no pseudo ground truth and the teacher is disabled -- \
                 generate targets with the 'stylize' subcommand, or set teacher.enabled = true"
            ),
        };
        let teacher_feats: Vec<Tensor<TrainBackend, 4>> = teacher_feats
            .into_iter()
            .map(Tensor::<TrainBackend, 4>::from_inner)
            .collect();
        Ok((target, teacher_feats))
    }
}

fn load_teacher(config: &Config, device: &TrainDevice) -> Result<LiveTeacher> {
    let started = Instant::now();
    let teacher = match config.teacher.kind()? {
        TeacherKind::MicroAst => {
            let model = weights::load(&config.teacher.weights_dir, device)
                .context("failed to load the MicroAST teacher")?;
            let img = load_rgb(&config.teacher.style)
                .with_context(|| format!("cannot load style {}", config.teacher.style.display()))?;
            let img = resize_short_edge(
                &img,
                config.teacher.style_size as u32,
                image::imageops::FilterType::Triangle,
            );
            let style = rgb_to_tensor::<InferBackend>(&img, device);
            log::info!("style branch precomputed (fixed style, alpha fixed at 1.0)");
            LiveTeacher::Device(Box::new(Stylizer::new(model, style)))
        }
        kind @ (TeacherKind::RainPrincess | TeacherKind::Whitebox | TeacherKind::Fns) => {
            let model = config
                .teacher
                .model
                .as_deref()
                .context("teacher.model is required for an ONNX teacher")?;
            LiveTeacher::Onnx(OnnxTeacher::open(kind, model)?)
        }
    };
    log::info!("teacher loaded in {:.2} s", started.elapsed().as_secs_f32());
    Ok(teacher)
}
