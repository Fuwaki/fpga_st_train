//! Training supervision: pseudo ground truth on disk and/or an online teacher.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;

use crate::backend::TrainDevice;
use crate::config::Config;
use crate::data::{load_rgb, resize_short_edge, rgb_to_tensor};
use crate::teacher::Stylizer;
use crate::teacher::weights;
use crate::{InferBackend, TrainBackend};

/// Optional online teacher; pseudo-GT comes from the dataset batch.
pub struct Supervision {
    teacher: Option<Stylizer>,
}

impl Supervision {
    /// Build supervision from run config (loads teacher weights when enabled).
    pub fn try_from_config(config: &Config, device: &TrainDevice) -> Result<Self> {
        let teacher = if config.teacher.enabled {
            let started = Instant::now();
            let model = weights::load(&config.teacher.weights_dir, device)
                .context("failed to load the MicroAST teacher")?;
            log::info!("teacher loaded in {:.2} s", started.elapsed().as_secs_f32());
            let img = load_rgb(&config.teacher.style)
                .with_context(|| format!("cannot load style {}", config.teacher.style.display()))?;
            let img = resize_short_edge(
                &img,
                config.teacher.style_size as u32,
                image::imageops::FilterType::Triangle,
            );
            let style = rgb_to_tensor::<InferBackend>(&img, device);
            log::info!("style branch precomputed (fixed style, alpha fixed at 1.0)");
            Some(Stylizer::new(model, style))
        } else {
            None
        };
        Ok(Self { teacher })
    }

    pub fn teacher(&self) -> Option<&Stylizer> {
        self.teacher.as_ref()
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
        ) = match self.teacher.as_ref() {
            Some(teacher) if provided_target.is_none() || need_teacher_feats => {
                let (out, feats) = teacher.features(content.clone().inner());
                (
                    Some(out),
                    if need_teacher_feats {
                        feats
                    } else {
                        Vec::new()
                    },
                )
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
