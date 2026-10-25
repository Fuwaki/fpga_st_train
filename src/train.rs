//! The training loop.
//!
//! Deliberately hand written rather than built on burn::train::Learner: style transfer
//! needs two inputs per sample, several loss terms and per-term logging, which reads more
//! clearly here than through TrainStep. It also means your model only has to implement
//! [crate::student::StudentModel] -- no TrainStep, no Learner plumbing.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::backend::{InferBackend, TrainBackend, TrainDevice, select_device};
use crate::config::{Config, OptimConfig, Schedule};
use crate::data::train_source::TrainContentSource;
use crate::data::{
    image_paths, load_rgb, overfit_frame, resize_long_edge, rgb_batch_to_tensor, rgb_to_tensor,
    save_rgb, tensor_to_rgb,
};
use crate::eval;
use crate::losses;
use crate::quant::Quantizable;
use crate::run_log::{RunLog, RunProvenance, SampleMetrics, TrainStepMetrics, ValStepMetrics};
use crate::student::StudentModel;
use crate::supervision::Supervision;
use crate::teacher::DistillTeacher;

/// Learning-rate schedule wrapper.
///
/// Burn's schedulers are distinct types with a shared trait; the trait is not object safe
/// (it has a generic associated method), so they are unified in a small enum.
enum LrSchedule {
    Constant(burn::lr_scheduler::constant::ConstantLr),
    Exponential(burn::lr_scheduler::exponential::ExponentialLrScheduler),
    Linear(burn::lr_scheduler::linear::LinearLrScheduler),
}

impl LrSchedule {
    fn build(config: &OptimConfig) -> Result<Self> {
        let lr = config.learning_rate;
        let bad = |e: String| anyhow::anyhow!("invalid learning-rate schedule: {e}");
        Ok(match &config.schedule {
            Schedule::Constant => Self::Constant(burn::lr_scheduler::constant::ConstantLr::new(lr)),
            Schedule::Exponential { gamma } => Self::Exponential(
                burn::lr_scheduler::exponential::ExponentialLrSchedulerConfig::new(lr, *gamma)
                    .init()
                    .map_err(bad)?,
            ),
            Schedule::Linear {
                final_lr,
                num_iters,
            } => Self::Linear(
                burn::lr_scheduler::linear::LinearLrSchedulerConfig::new(lr, *final_lr, *num_iters)
                    .init()
                    .map_err(bad)?,
            ),
        })
    }

    fn next(&mut self) -> f64 {
        use burn::lr_scheduler::LrScheduler;
        match self {
            Self::Constant(s) => s.step(),
            Self::Exponential(s) => s.step(),
            Self::Linear(s) => s.step(),
        }
    }
}

/// Save a model to disk as a named messagepack file.
pub fn save_checkpoint<B: Backend, M: Module<B>>(model: &M, path: PathBuf) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path.clone(), &recorder)
        .map_err(|e| anyhow::anyhow!("cannot save {}: {e}", path.display()))
}

/// One batch of content and its supervision, built by sampling the dataset.
pub(crate) struct Batch {
    pub(crate) content: Tensor<TrainBackend, 4>,
    target: Option<Tensor<TrainBackend, 4>>,
}

struct CpuBatch {
    contents: Vec<image::RgbImage>,
    targets: Option<Vec<image::RgbImage>>,
}

fn prepare_cpu_batch(
    dataset: &TrainContentSource,
    crop: u32,
    batch: usize,
    rng: &mut StdRng,
) -> Result<CpuBatch> {
    // Draw seeds in the training thread, then perform decode/resize/crop in parallel. This
    // keeps the sample sequence deterministic regardless of Rayon scheduling.
    let seeds: Vec<u64> = (0..batch).map(|_| rand::Rng::random(rng)).collect();
    let samples: Vec<(image::RgbImage, Option<image::RgbImage>)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut sample_rng = StdRng::seed_from_u64(seed);
            dataset.sample_crops(crop, &mut sample_rng)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut contents = Vec::with_capacity(batch);
    let mut targets = Vec::with_capacity(batch);
    let mut complete = true;
    for (content, target) in samples {
        contents.push(content);
        match target {
            Some(target) => targets.push(target),
            None => complete = false,
        }
    }
    Ok(CpuBatch {
        contents,
        targets: complete.then_some(targets),
    })
}

/// Next content batch, with the ONNX teacher possibly already running on Vega.
struct HostAhead {
    cpu: CpuBatch,
    pending: Option<mpsc::Receiver<Result<Vec<f32>>>>,
    ready: Option<Vec<f32>>,
}

impl HostAhead {
    /// Wait for Vega if a batch was submitted. Safe to call twice.
    fn finish(&mut self) -> Result<()> {
        if let Some(rx) = self.pending.take() {
            self.ready = Some(rx.recv().context("ONNX teacher thread stopped")??);
        }
        Ok(())
    }
}

fn launch_host_ahead(
    rx: &mpsc::Receiver<Result<CpuBatch>>,
    supervision: &Supervision,
) -> Result<HostAhead> {
    let cpu = rx
        .recv()
        .context("training data prefetch thread stopped")??;
    let pending = if cpu.targets.is_none() {
        supervision.submit_host_batch(&cpu.contents).transpose()?
    } else {
        None
    };
    Ok(HostAhead {
        cpu,
        pending,
        ready: None,
    })
}

fn reclaim_teacher(
    ahead: &mut Option<HostAhead>,
    step: usize,
    iterations: usize,
    val_every: usize,
    sample_every: usize,
) -> Result<()> {
    let touches_session = (val_every > 0 && step % val_every == 0)
        || (sample_every > 0 && (step % sample_every == 0 || step == iterations));
    if touches_session {
        if let Some(pending) = ahead.as_mut() {
            pending.finish()?;
        }
    }
    Ok(())
}

fn host_target_tensor(
    buf: Vec<f32>,
    n: usize,
    h: usize,
    w: usize,
    device: &TrainDevice,
) -> Result<Tensor<TrainBackend, 4>> {
    let expect = n * 3 * h * w;
    if buf.len() != expect {
        bail!(
            "onnx teacher returned {} values, expected {expect}",
            buf.len()
        );
    }
    let inner = Tensor::<InferBackend, 4>::from_data(
        burn::tensor::TensorData::new(buf, [n, 3, h, w]),
        device,
    );
    Ok(Tensor::<TrainBackend, 4>::from_inner(inner))
}

fn upload_batch(batch: CpuBatch, device: &TrainDevice) -> Batch {
    Batch {
        content: rgb_batch_to_tensor::<TrainBackend>(&batch.contents, device),
        target: batch
            .targets
            .as_ref()
            .map(|targets| rgb_batch_to_tensor::<TrainBackend>(targets, device)),
    }
}

/// Teacher output for one step.
///
/// An overfit run feeds the same frame every step. The first step runs the
/// teacher and keeps the inner tensor; later steps wrap a fresh leaf around
/// that cache. Reusing the autodiff tensor itself would free its graph on the
/// first backward and leave the rest of the run with an empty gradient.
fn resolve_step_target(
    supervision: &crate::supervision::Supervision,
    content: Tensor<TrainBackend, 4>,
    provided: Option<Tensor<TrainBackend, 4>>,
    need_feats: bool,
    remember: bool,
    cached: &mut Option<Tensor<InferBackend, 4>>,
) -> Result<(Tensor<TrainBackend, 4>, Vec<Tensor<TrainBackend, 4>>)> {
    if remember {
        if let Some(saved) = cached {
            return Ok((
                Tensor::<TrainBackend, 4>::from_inner(saved.clone()),
                Vec::new(),
            ));
        }
    }
    let had_target = provided.is_some();
    let (target, feats) = supervision.resolve_batch(content, provided, need_feats)?;
    if remember && !had_target && !need_feats {
        let [_, _, h, w] = target.dims();
        log::info!("cached overfit teacher target ({h}x{w})");
        *cached = Some(target.clone().inner());
    }
    Ok((target, feats))
}

pub(crate) fn build_batch(
    dataset: &TrainContentSource,
    crop: u32,
    batch: usize,
    rng: &mut StdRng,
    device: &TrainDevice,
) -> Result<Batch> {
    Ok(upload_batch(
        prepare_cpu_batch(dataset, crop, batch, rng)?,
        device,
    ))
}

/// Load a checkpoint saved by [save_checkpoint] into a freshly built student.
pub fn load_checkpoint<M: Module<TrainBackend>>(
    model: M,
    path: &std::path::Path,
    device: &TrainDevice,
) -> Result<M> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(path.to_path_buf(), &recorder, device)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", path.display()))
}

/// Fixed seed for the held-out val batch (same crops on every `val_every` step).
const VAL_BATCH_SEED: u64 = 0x5641_4C5F9E37_79B9;

#[derive(Clone, Copy)]
struct LossWeights {
    pixel: f32,
    gradient: f32,
    flat: f32,
    edge: f32,
    contrast: f32,
    stroke: f32,
    texture: f32,
    feature: f32,
    tv: f32,
}

impl LossWeights {
    fn from_config(loss: &crate::config::LossConfig) -> Self {
        Self {
            pixel: loss.pixel,
            gradient: loss.gradient,
            flat: loss.flat,
            edge: loss.edge,
            contrast: loss.contrast,
            stroke: loss.stroke,
            texture: loss.texture,
            feature: loss.feature,
            tv: loss.tv,
        }
    }

    fn formula(&self) -> String {
        format!(
            "{}*L1 + {}*grad + {}*flat + {}*edge + {}*contrast + {}*stroke + {}*tex + {}*TV",
            self.pixel,
            self.gradient,
            self.flat,
            self.edge,
            self.contrast,
            self.stroke,
            self.texture,
            self.tv
        )
    }
}

struct LossParts<B: Backend> {
    total: Tensor<B, 1>,
    pixel: Tensor<B, 1>,
    gradient: Option<Tensor<B, 1>>,
    flat: Tensor<B, 1>,
    edge: Tensor<B, 1>,
    contrast: Tensor<B, 1>,
    stroke: Option<Tensor<B, 1>>,
    texture: Option<Tensor<B, 1>>,
    feature: Option<Tensor<B, 1>>,
    tv: Option<Tensor<B, 1>>,
}

impl<B: Backend> LossParts<B> {
    fn train_metrics(&self, lr: f64, steps_per_s: Option<f64>) -> TrainStepMetrics {
        TrainStepMetrics {
            loss: self.total.clone().into_scalar().to_f64(),
            lr,
            steps_per_s,
            l1: Some(self.pixel.clone().into_scalar().to_f64()),
            gradient: self
                .gradient
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            flat: Some(self.flat.clone().into_scalar().to_f64()),
            edge: Some(self.edge.clone().into_scalar().to_f64()),
            contrast: Some(self.contrast.clone().into_scalar().to_f64()),
            stroke: self
                .stroke
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            texture: self
                .texture
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            feature: self
                .feature
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            tv: self.tv.as_ref().map(|t| t.clone().into_scalar().to_f64()),
        }
    }

    fn val_metrics(&self) -> ValStepMetrics {
        ValStepMetrics {
            loss: self.total.clone().into_scalar().to_f64(),
            l1: Some(self.pixel.clone().into_scalar().to_f64()),
            gradient: self
                .gradient
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            flat: Some(self.flat.clone().into_scalar().to_f64()),
            edge: Some(self.edge.clone().into_scalar().to_f64()),
            contrast: Some(self.contrast.clone().into_scalar().to_f64()),
            stroke: self
                .stroke
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            texture: self
                .texture
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            feature: self
                .feature
                .as_ref()
                .map(|t| t.clone().into_scalar().to_f64()),
            tv: self.tv.as_ref().map(|t| t.clone().into_scalar().to_f64()),
        }
    }
}

fn forward_loss<B: Backend, M: StudentModel<B> + Quantizable<B>>(
    model: &M,
    content: Tensor<B, 4>,
    target: Tensor<B, 4>,
    teacher_feats: &[Tensor<B, 4>],
    feature_scales: &[f32],
    weights: &LossWeights,
    quant_scheme: Option<&crate::quant::QuantScheme>,
) -> LossParts<B> {
    let (output, student_feats) = match quant_scheme {
        Some(scheme) => (model.forward_quantised(content, scheme), Vec::new()),
        None => model.intermediate(content),
    };

    let pixel = losses::l1(output.clone(), target.clone());
    let mut total = pixel.clone().mul_scalar(weights.pixel);
    let mut gradient = None;
    let mut stroke = None;
    let mut texture = None;
    let mut feature = None;
    let mut tv = None;

    if weights.feature > 0.0
        && !student_feats.is_empty()
        && student_feats.len() == teacher_feats.len()
    {
        let stages = student_feats.len() as f32;
        let mut acc: Option<Tensor<B, 1>> = None;
        for ((student, teacher), scale) in student_feats
            .iter()
            .zip(teacher_feats)
            .zip(feature_scales.iter())
        {
            let stage = (student.clone() - teacher.clone())
                .abs()
                .mean()
                .div_scalar(*scale);
            acc = Some(match acc {
                None => stage,
                Some(prev) => prev + stage,
            });
        }
        if let Some(acc) = acc {
            let term = acc.div_scalar(stages);
            feature = Some(term.clone());
            total = total + term.mul_scalar(weights.feature);
        }
    }

    if weights.gradient > 0.0 {
        let term = losses::gradient_l1(output.clone(), target.clone());
        gradient = Some(term.clone());
        total = total + term.mul_scalar(weights.gradient);
    }
    if weights.stroke > 0.0 {
        let term = losses::stroke_l1(output.clone(), target.clone());
        stroke = Some(term.clone());
        total = total + term.mul_scalar(weights.stroke);
    }
    if weights.texture > 0.0 {
        let term = losses::texture_energy(output.clone(), target.clone());
        texture = Some(term.clone());
        total = total + term.mul_scalar(weights.texture);
    }
    let flat = losses::flat(output.clone(), target.clone());
    if weights.flat > 0.0 {
        total = total + flat.clone().mul_scalar(weights.flat);
    }
    let edge = losses::edge(output.clone(), target.clone());
    if weights.edge > 0.0 {
        total = total + edge.clone().mul_scalar(weights.edge);
    }
    let contrast = losses::contrast(output.clone(), target.clone());
    if weights.contrast > 0.0 {
        total = total + contrast.clone().mul_scalar(weights.contrast);
    }
    if weights.tv > 0.0 {
        let term = losses::total_variation(output);
        tv = Some(term.clone());
        total = total + term.mul_scalar(weights.tv);
    }

    LossParts {
        total,
        pixel,
        gradient,
        flat,
        edge,
        contrast,
        stroke,
        texture,
        feature,
        tv,
    }
}

struct Slot<M, Opt> {
    config: Config,
    /// `None` only while the optimiser step is moving the module.
    model: Option<M>,
    optimiser: Opt,
    schedule: LrSchedule,
    run_log: RunLog,
    sample_images: Vec<(String, image::RgbImage)>,
    last_train_metrics: Option<TrainStepMetrics>,
    quant_scheme: Option<crate::quant::QuantScheme>,
    weights: LossWeights,
}

/// One student inside a cohort. The teacher batch is shared; each slot owns its graph.
trait CohortSlot {
    fn train_step(
        &mut self,
        content: Tensor<TrainBackend, 4>,
        target: Tensor<TrainBackend, 4>,
        teacher_feats: &[Tensor<TrainBackend, 4>],
        feature_scales: &[f32],
        cohort: bool,
        step: usize,
        iterations: usize,
        log_this: bool,
        started: &Instant,
    ) -> Result<()>;

    fn validate_step(
        &mut self,
        content: Tensor<InferBackend, 4>,
        target: Tensor<InferBackend, 4>,
        teacher_feats: &[Tensor<InferBackend, 4>],
        feature_scales: &[f32],
        cohort: bool,
        step: usize,
    ) -> Result<()>;

    fn checkpoint_and_sample(
        &mut self,
        step: usize,
        iterations: usize,
        checkpoint_every: usize,
        sample_every: usize,
        teacher: Option<&dyn DistillTeacher>,
        device: &TrainDevice,
    ) -> Result<()>;

    fn finish(&mut self, iterations: usize, wall_s: f64, cohort: bool) -> Result<()>;
}

impl<M, Opt> CohortSlot for Slot<M, Opt>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<InferBackend> + Quantizable<InferBackend>,
    Opt: Optimizer<M, TrainBackend>,
{
    fn train_step(
        &mut self,
        content: Tensor<TrainBackend, 4>,
        target: Tensor<TrainBackend, 4>,
        teacher_feats: &[Tensor<TrainBackend, 4>],
        feature_scales: &[f32],
        cohort: bool,
        step: usize,
        iterations: usize,
        log_this: bool,
        started: &Instant,
    ) -> Result<()> {
        let weights = self.weights;
        let parts = forward_loss(
            self.model.as_ref().expect("student"),
            content,
            target,
            teacher_feats,
            feature_scales,
            &weights,
            self.quant_scheme.as_ref(),
        );
        let grads = GradientsParams::from_grads(
            parts.total.clone().backward(),
            self.model.as_ref().expect("student"),
        );
        let lr = self.schedule.next();
        let model = self.model.take().expect("student");
        self.model = Some(self.optimiser.step(lr, model, grads));
        if log_this {
            let elapsed = started.elapsed().as_secs_f64();
            let metrics = parts.train_metrics(lr, Some(step as f64 / elapsed.max(1e-9)));
            let prefix = if cohort {
                format!("[{}] ", self.config.run.name)
            } else {
                String::new()
            };
            println!(
                "{prefix}step {step:>7}/{:<7} loss {:>9.5}  l1 {:.5}  grad {:.5}  flat {:.5}  edge {:.5}  contrast {:.5}  stroke {:.5}  tex {:.5}  feat {:.5}  tv {:.5}  lr {:.3e}  {:.2} step/s  {:.1} s",
                iterations,
                metrics.loss,
                metrics.l1.unwrap_or(0.0),
                metrics.gradient.unwrap_or(0.0),
                metrics.flat.unwrap_or(0.0),
                metrics.edge.unwrap_or(0.0),
                metrics.contrast.unwrap_or(0.0),
                metrics.stroke.unwrap_or(0.0),
                metrics.texture.unwrap_or(0.0),
                metrics.feature.unwrap_or(0.0),
                metrics.tv.unwrap_or(0.0),
                metrics.lr,
                metrics.steps_per_s.unwrap_or(0.0),
                elapsed,
            );
            self.last_train_metrics = Some(metrics.clone());
            self.run_log.train_step(step, metrics)?;
        }
        Ok(())
    }

    fn validate_step(
        &mut self,
        content: Tensor<InferBackend, 4>,
        target: Tensor<InferBackend, 4>,
        teacher_feats: &[Tensor<InferBackend, 4>],
        feature_scales: &[f32],
        cohort: bool,
        step: usize,
    ) -> Result<()> {
        let weights = self.weights;
        let valid = self.model.as_ref().expect("student").valid();
        let val_parts = forward_loss(
            &valid,
            content,
            target,
            teacher_feats,
            feature_scales,
            &weights,
            self.quant_scheme.as_ref(),
        );
        let val_metrics = val_parts.val_metrics();
        let prefix = if cohort {
            format!("[{}] ", self.config.run.name)
        } else {
            String::new()
        };
        println!(
            "{prefix}  val {step:>7} loss {:>9.5}  l1 {:.5}",
            val_metrics.loss,
            val_metrics.l1.unwrap_or(0.0),
        );
        self.run_log.val_step(step, val_metrics)?;
        Ok(())
    }

    fn checkpoint_and_sample(
        &mut self,
        step: usize,
        iterations: usize,
        checkpoint_every: usize,
        sample_every: usize,
        teacher: Option<&dyn DistillTeacher>,
        device: &TrainDevice,
    ) -> Result<()> {
        if checkpoint_every > 0 && step % checkpoint_every == 0 {
            save_checkpoint(
                self.model.as_ref().expect("student"),
                self.config
                    .run
                    .output_dir
                    .join(format!("step_{step:06}.mpk")),
            )?;
        }
        if sample_every > 0 && (step % sample_every == 0 || step == iterations) {
            for (name, img) in &self.sample_images {
                match sample(
                    self.model.as_ref().expect("student"),
                    img,
                    name,
                    teacher,
                    &self.config.run.output_dir,
                    step,
                    device,
                    &self.config.run.name,
                ) {
                    Ok(metrics) => self.run_log.sample_eval(step, name, metrics)?,
                    Err(e) => log::warn!("could not write sample '{name}' at step {step}: {e}"),
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self, iterations: usize, wall_s: f64, cohort: bool) -> Result<()> {
        let final_path = self.config.run.output_dir.join("model.mpk");
        save_checkpoint(self.model.as_ref().expect("student"), final_path.clone())?;
        self.run_log
            .run_end(iterations, wall_s, self.last_train_metrics.clone())?;
        let prefix = if cohort {
            format!("[{}] ", self.config.run.name)
        } else {
            String::new()
        };
        println!("{prefix}done in {wall_s:.1} s -> {}", final_path.display());
        Ok(())
    }
}

fn shared_recipe(config: &Config) -> serde_json::Value {
    serde_json::json!({
        "data": config.data,
        "teacher": config.teacher,
        "optim": config.optim,
        "quant": config.quant,
        "arch": config.student.arch,
        "device": config.run.device,
        "log_every": config.run.log_every,
        "val_every": config.run.val_every,
        "val_batch_size": config.run.val_batch_size,
        "checkpoint_every": config.run.checkpoint_every,
        "sample_every": config.run.sample_every,
        "sample_limit": config.run.sample_limit,
        "sample_size": config.run.sample_size,
        "sample_content": config.run.sample_content,
    })
}

/// Run training. `build_student` constructs the concrete model (see [crate::registry]).
pub fn run<M, F>(config: Config, provenance: RunProvenance, build_student: F) -> Result<()>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule:
        StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend> + Quantizable<InferBackend>,
    F: Fn(&crate::config::StudentConfig, &TrainDevice) -> Result<M>,
{
    run_all(vec![config], vec![provenance], build_student)
}

/// One teacher forward per step, then a backward pass for each student.
///
/// Every config must share the data, teacher, optimiser, and step budget.
/// Loss weights and `run.seed` may differ. The seed draws that student's
/// initial weights; crops follow the first config's seed. Each student keeps
/// its own formula, Adam state, log, and output directory.
pub fn run_all<M, F>(
    configs: Vec<Config>,
    provenances: Vec<RunProvenance>,
    build_student: F,
) -> Result<()>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule:
        StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend> + Quantizable<InferBackend>,
    F: Fn(&crate::config::StudentConfig, &TrainDevice) -> Result<M>,
{
    // Keep CPU preprocessing parallel without saturating the host; this workload uses eight
    // dedicated Rayon workers regardless of the machine's logical core count.
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .map_err(|e| anyhow::anyhow!("cannot configure data preparation thread pool: {e}"))?;

    if configs.is_empty() {
        bail!("cohort is empty");
    }
    if configs.len() != provenances.len() {
        bail!(
            "cohort has {} configs and {} provenances",
            configs.len(),
            provenances.len()
        );
    }
    let recipe = shared_recipe(&configs[0]);
    let mut seen_dirs = Vec::new();
    for config in &configs {
        config.validate()?;
        if shared_recipe(config) != recipe {
            bail!(
                "cohort member '{}' does not share data, teacher, or step budget with '{}'",
                config.run.name,
                configs[0].run.name
            );
        }
        if seen_dirs.contains(&config.run.output_dir) {
            bail!(
                "cohort output_dir {} is used more than once",
                config.run.output_dir.display()
            );
        }
        seen_dirs.push(config.run.output_dir.clone());
    }

    let config0 = configs[0].clone();
    let device = select_device(&config0.run.device);
    let mut rng = StdRng::seed_from_u64(config0.run.seed);

    let dataset = Arc::new(TrainContentSource::build(&config0.data)?);
    let crop = config0.data.crop as u32;
    let batch = config0.data.batch_size;
    let supervision = Supervision::try_from_config(&config0, &device)?;

    let feature_weight = configs
        .iter()
        .map(|c| c.loss.feature)
        .fold(0.0f32, f32::max);
    if feature_weight > 0.0 {
        match supervision.teacher() {
            None => bail!(
                "loss.feature > 0 requires teacher.enabled = true (feature distillation is teacher-guided)"
            ),
            Some(teacher) if !teacher.has_features() => {
                bail!("loss.feature > 0 requires a MicroAST teacher (teacher.kind = \"microast\")")
            }
            Some(_) => {}
        }
    }
    if config0.loss.perceptual > 0.0 || config0.loss.style > 0.0 {
        log::warn!(
            "perceptual/style weights are set but the VGG terms are not implemented yet; \
             they are currently ignored"
        );
    }
    if configs.len() > 1 {
        let names = configs
            .iter()
            .map(|c| c.run.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "cohort of {}: one teacher forward per step, then each student backward ({names}); crops use seed {}",
            configs.len(),
            config0.run.seed,
        );
    }

    let mut slots = Vec::with_capacity(configs.len());
    for (config, provenance) in configs.into_iter().zip(provenances) {
        TrainBackend::seed(&device, config.run.seed);
        let run_log = RunLog::open(&config.run.output_dir, &config, &provenance)?;
        log::info!("metrics -> {}", run_log.path().display());
        crate::config::ModelCard::of(&config.student)
            .with_meta("run", serde_json::Value::String(config.run.name.clone()))
            .with_meta(
                "style",
                serde_json::Value::String(config.teacher.style_label()),
            )
            .with_meta(
                "iterations",
                serde_json::Value::Number(config.optim.iterations.into()),
            )
            .save(&config.run.output_dir)?;

        let mut model: M = build_student(&config.student, &device)?;
        if let Some(path) = &config.run.init_from {
            model = load_checkpoint(model, path, &device)?;
            log::info!("warm start from {}", path.display());
        }
        model.after_load();
        log::info!("student: {}", model.describe());

        let optimiser = AdamConfig::new()
            .with_beta_1(config.optim.beta_1)
            .with_beta_2(config.optim.beta_2)
            .with_epsilon(config.optim.epsilon)
            .init::<TrainBackend, M>();
        let schedule = LrSchedule::build(&config.optim)?;

        let prefix = if slots.capacity() > 1 {
            format!("[{}] ", config.run.name)
        } else {
            String::new()
        };
        let weights = LossWeights::from_config(&config.loss);
        println!(
            "{prefix}training '{}' for {} steps: batch {batch}, crop {crop}, device {}, init seed {}, loss = {}",
            config.run.name,
            config.optim.iterations,
            config.run.device,
            config.run.seed,
            weights.formula(),
        );

        let quant_scheme = if config.quant.enabled {
            let path = config.quant.calibration.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "quant.enabled = true requires quant.calibration (run the 'calibrate' subcommand first)"
                )
            })?;
            let calibration = crate::quant::Calibration::load(path)?;
            calibration.validate_for(&model)?;
            let n_act = model.quant_points().len();
            println!(
                "{prefix}low-bit QAT: {} ({} activation points, image scale {:.5}, largest scale {:.4})",
                calibration.bits.label(),
                n_act,
                calibration.scales[0],
                calibration.scales.iter().cloned().fold(0.0f32, f32::max)
            );
            Some(calibration.scheme())
        } else {
            None
        };

        let sample_images = load_sample_images(&config, &dataset)?;
        slots.push(Slot {
            config,
            model: Some(model),
            optimiser,
            schedule,
            run_log,
            sample_images,
            last_train_metrics: None,
            quant_scheme,
            weights,
        });
    }
    TrainBackend::seed(&device, config0.run.seed);
    let feature_scales: Vec<f32> = if feature_weight > 0.0 {
        let teacher = supervision.teacher().expect("checked above");
        let probe = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let probe = probe.content.inner();
        let (_, feats) = teacher.features(probe)?;
        let scales: Vec<f32> = feats
            .iter()
            .map(|f| (f.clone().abs().mean().into_scalar().to_f64() as f32).max(1e-3))
            .collect();
        log::info!("feature distillation scales (per stage): {scales:?}");
        scales
    } else {
        Vec::new()
    };
    if feature_weight > 0.0 {
        println!("feature distillation: weight {feature_weight} over the teacher decoder stages");
    }

    let cohort = slots.len() > 1;
    let iterations = slots[0].config.optim.iterations;
    let log_every = slots[0].config.run.log_every;
    let val_every = slots[0].config.run.val_every;
    let checkpoint_every = slots[0].config.run.checkpoint_every;
    let sample_every = slots[0].config.run.sample_every;
    let val_batch_size = slots[0].config.run.val_batch_size.max(1);
    let mut val_rng = StdRng::seed_from_u64(slots[0].config.run.seed ^ VAL_BATCH_SEED);
    let started = Instant::now();
    let remember_target = slots[0].config.data.overfit_long_edge.is_some();
    let mut cached_target: Option<Tensor<InferBackend, 4>> = None;

    // Keep two CPU-prepared batches queued so decoding/cropping for the next step overlaps
    // teacher and student GPU work. The bounded queue limits host memory usage.
    let (batch_tx, batch_rx) = mpsc::sync_channel::<Result<CpuBatch>>(2);
    let producer_dataset = Arc::clone(&dataset);
    let producer = thread::Builder::new()
        .name("training-data-prefetch".into())
        .spawn(move || {
            let mut train_rng = rng;
            for _ in 0..iterations {
                let batch = prepare_cpu_batch(&producer_dataset, crop, batch, &mut train_rng);
                if batch_tx.send(batch).is_err() {
                    break;
                }
            }
        })
        .context("start training data prefetch thread")?;

    let overlap_teacher =
        supervision.pipelines_on_host() && !remember_target && feature_weight == 0.0;
    if overlap_teacher {
        println!("onnx teacher runs one batch ahead of the student backward");
    }
    let mut ahead: Option<HostAhead> = None;

    for step in 1..=iterations {
        let (cpu_batch, host_target) = if let Some(mut pending) = ahead.take() {
            pending.finish()?;
            (pending.cpu, pending.ready)
        } else {
            (
                batch_rx
                    .recv()
                    .context("training data prefetch thread stopped")??,
                None,
            )
        };
        let batch_data = upload_batch(cpu_batch, &device);
        let content = batch_data.content;
        let (target, teacher_feats) = if let Some(buf) = host_target {
            let [n, _, h, w] = content.dims();
            (host_target_tensor(buf, n, h, w, &device)?, Vec::new())
        } else {
            resolve_step_target(
                &supervision,
                content.clone(),
                batch_data.target,
                feature_weight > 0.0,
                remember_target,
                &mut cached_target,
            )?
        };
        // Materialise the teacher so its activations drop back into the CUDA
        // pool as free slices. Do not return those pages to the driver: a new
        // 487 MiB page does not allocate on this T10, and the student reuses
        // the slices the teacher already reserved.
        let _ = target.clone().sum().into_scalar();
        let log_this = log_every > 0 && (step % log_every == 0 || step == iterations);
        if overlap_teacher && step < iterations {
            ahead = Some(launch_host_ahead(&batch_rx, &supervision)?);
        }

        for slot in &mut slots {
            let weights = slot.weights;
            let parts = forward_loss(
                slot.model.as_ref().expect("student"),
                content.clone(),
                target.clone(),
                &teacher_feats,
                &feature_scales,
                &weights,
                slot.quant_scheme.as_ref(),
            );
            let grads = GradientsParams::from_grads(
                parts.total.clone().backward(),
                slot.model.as_ref().expect("student"),
            );
            let lr = slot.schedule.next();
            let model = slot.model.take().expect("student");
            slot.model = Some(slot.optimiser.step(lr, model, grads));
            if log_this {
                let elapsed = started.elapsed().as_secs_f64();
                let metrics = parts.train_metrics(lr, Some(step as f64 / elapsed.max(1e-9)));
                let prefix = if cohort {
                    format!("[{}] ", slot.config.run.name)
                } else {
                    String::new()
                };
                println!(
                    "{prefix}step {step:>7}/{:<7} loss {:>9.5}  l1 {:.5}  grad {:.5}  flat {:.5}  edge {:.5}  contrast {:.5}  stroke {:.5}  tex {:.5}  feat {:.5}  tv {:.5}  lr {:.3e}  {:.2} step/s  {:.1} s",
                    iterations,
                    metrics.loss,
                    metrics.l1.unwrap_or(0.0),
                    metrics.gradient.unwrap_or(0.0),
                    metrics.flat.unwrap_or(0.0),
                    metrics.edge.unwrap_or(0.0),
                    metrics.contrast.unwrap_or(0.0),
                    metrics.stroke.unwrap_or(0.0),
                    metrics.texture.unwrap_or(0.0),
                    metrics.feature.unwrap_or(0.0),
                    metrics.tv.unwrap_or(0.0),
                    metrics.lr,
                    metrics.steps_per_s.unwrap_or(0.0),
                    elapsed,
                );
                slot.last_train_metrics = Some(metrics.clone());
                slot.run_log.train_step(step, metrics)?;
            }
        }

        reclaim_teacher(&mut ahead, step, iterations, val_every, sample_every)?;
        if val_every > 0 && step % val_every == 0 {
            let val_data = build_batch(&dataset, crop, val_batch_size, &mut val_rng, &device)?;
            let (val_target, val_teacher_feats) = supervision.resolve_batch(
                val_data.content.clone(),
                val_data.target,
                feature_weight > 0.0,
            )?;
            let val_teacher_inner: Vec<Tensor<InferBackend, 4>> = val_teacher_feats
                .iter()
                .map(|t| t.clone().inner())
                .collect();
            for slot in &mut slots {
                let weights = slot.weights;
                let valid = slot.model.as_ref().expect("student").valid();
                let val_parts = forward_loss(
                    &valid,
                    val_data.content.clone().inner(),
                    val_target.clone().inner(),
                    &val_teacher_inner,
                    &feature_scales,
                    &weights,
                    slot.quant_scheme.as_ref(),
                );
                let val_metrics = val_parts.val_metrics();
                let prefix = if cohort {
                    format!("[{}] ", slot.config.run.name)
                } else {
                    String::new()
                };
                println!(
                    "{prefix}  val {step:>7} loss {:>9.5}  l1 {:.5}",
                    val_metrics.loss,
                    val_metrics.l1.unwrap_or(0.0),
                );
                slot.run_log.val_step(step, val_metrics)?;
            }
        }

        for slot in &mut slots {
            if checkpoint_every > 0 && step % checkpoint_every == 0 {
                save_checkpoint(
                    slot.model.as_ref().expect("student"),
                    slot.config
                        .run
                        .output_dir
                        .join(format!("step_{step:06}.mpk")),
                )?;
            }
            // The final step is always sampled, so a 2-step smoke still writes
            // target_*.png when sample_every is the full-run interval.
            if sample_every > 0 && (step % sample_every == 0 || step == iterations) {
                for (name, img) in &slot.sample_images {
                    match sample(
                        slot.model.as_ref().expect("student"),
                        img,
                        name,
                        supervision.teacher(),
                        &slot.config.run.output_dir,
                        step,
                        &device,
                        &slot.config.run.name,
                    ) {
                        Ok(metrics) => slot.run_log.sample_eval(step, name, metrics)?,
                        Err(e) => {
                            log::warn!("could not write sample '{name}' at step {step}: {e}");
                        }
                    }
                }
            }
        }
    }

    producer
        .join()
        .map_err(|_| anyhow::anyhow!("training data prefetch thread panicked"))?;

    let wall_s = started.elapsed().as_secs_f64();
    for slot in &mut slots {
        let final_path = slot.config.run.output_dir.join("model.mpk");
        save_checkpoint(slot.model.as_ref().expect("student"), final_path.clone())?;
        slot.run_log
            .run_end(iterations, wall_s, slot.last_train_metrics.clone())?;
        let prefix = if cohort {
            format!("[{}] ", slot.config.run.name)
        } else {
            String::new()
        };
        println!("{prefix}done in {wall_s:.1} s -> {}", final_path.display());
    }
    Ok(())
}

fn recipe_without_arch(config: &Config) -> serde_json::Value {
    let mut value = shared_recipe(config);
    value.as_object_mut().expect("recipe object").remove("arch");
    value
}

fn boxed_slot<M, F>(
    config: Config,
    provenance: RunProvenance,
    device: &TrainDevice,
    dataset: &TrainContentSource,
    cohort: bool,
    build: F,
) -> Result<Box<dyn CohortSlot>>
where
    M: StudentModel<TrainBackend>
        + Quantizable<TrainBackend>
        + AutodiffModule<TrainBackend>
        + 'static,
    M::InnerModule: StudentModel<InferBackend> + Quantizable<InferBackend>,
    F: Fn(&crate::config::StudentConfig, &TrainDevice) -> Result<M>,
{
    let run_log = RunLog::open(&config.run.output_dir, &config, &provenance)?;
    log::info!("metrics -> {}", run_log.path().display());
    crate::config::ModelCard::of(&config.student)
        .with_meta("run", serde_json::Value::String(config.run.name.clone()))
        .with_meta(
            "style",
            serde_json::Value::String(config.teacher.style_label()),
        )
        .with_meta(
            "iterations",
            serde_json::Value::Number(config.optim.iterations.into()),
        )
        .save(&config.run.output_dir)?;

    TrainBackend::seed(device, config.run.seed);
    let mut model = build(&config.student, device)?;
    if let Some(path) = &config.run.init_from {
        model = load_checkpoint(model, path, device)?;
        log::info!("warm start from {}", path.display());
    }
    model.after_load();
    log::info!("student: {}", model.describe());

    let optimiser = AdamConfig::new()
        .with_beta_1(config.optim.beta_1)
        .with_beta_2(config.optim.beta_2)
        .with_epsilon(config.optim.epsilon)
        .init::<TrainBackend, M>();
    let schedule = LrSchedule::build(&config.optim)?;
    let prefix = if cohort {
        format!("[{}] ", config.run.name)
    } else {
        String::new()
    };
    let weights = LossWeights::from_config(&config.loss);
    println!(
        "{prefix}training '{}' for {} steps: batch {}, crop {}, device {}, init seed {}, loss = {}",
        config.run.name,
        config.optim.iterations,
        config.data.batch_size,
        config.data.crop,
        config.run.device,
        config.run.seed,
        weights.formula(),
    );

    let quant_scheme = if config.quant.enabled {
        let path = config.quant.calibration.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "quant.enabled = true requires quant.calibration (run the 'calibrate' subcommand first)"
            )
        })?;
        let calibration = crate::quant::Calibration::load(path)?;
        calibration.validate_for(&model)?;
        Some(calibration.scheme())
    } else {
        None
    };
    let sample_images = load_sample_images(&config, dataset)?;
    Ok(Box::new(Slot {
        config,
        model: Some(model),
        optimiser,
        schedule,
        run_log,
        sample_images,
        last_train_metrics: None,
        quant_scheme,
        weights,
    }))
}

fn build_tiny(
    config: &crate::config::StudentConfig,
    device: &TrainDevice,
) -> Result<crate::student::TinyDwStudent<TrainBackend>> {
    use crate::student::plugin::{parse_kind, parse_site};
    use crate::student::tiny_dw::{parse_color, parse_encoder, parse_lut_bits};
    let plugin = parse_kind(config.str_option("plugin", "none")).map_err(anyhow::Error::msg)?;
    let site = parse_site(config.str_option("plugin_site", "h4")).map_err(anyhow::Error::msg)?;
    let color = parse_color(config.str_option("color", "none")).map_err(anyhow::Error::msg)?;
    let encoder = parse_encoder(config.str_option("encoder", "sep")).map_err(anyhow::Error::msg)?;
    let lut_bits =
        parse_lut_bits(config.usize_option("lut_bits", 8)).map_err(anyhow::Error::msg)?;
    let trunk_h8 = crate::student::tiny_dw::parse_trunk_h8(config.str_option("trunk_at", "h4"))
        .map_err(anyhow::Error::msg)?;
    Ok(crate::student::TinyDwStudent::with_encoder(
        config.usize_option("base", 32),
        config.usize_option("res_blocks", 4),
        config.usize_option("mid", 32),
        config.usize_option("fine", 16),
        plugin,
        site,
        color,
        encoder,
        config.usize_option("lut_hidden", 4),
        lut_bits,
        trunk_h8,
        device,
    ))
}

fn build_guided(
    config: &crate::config::StudentConfig,
    device: &TrainDevice,
) -> Result<crate::student::guided::GuidedStudent<TrainBackend>> {
    let spec =
        crate::student::guided::GuideSpec::from_config(config).map_err(anyhow::Error::msg)?;
    crate::student::guided::print_cost(spec);
    Ok(crate::student::guided::GuidedStudent::new(spec, device))
}

/// One teacher forward per step for students whose graphs differ (`tiny_dw` beside
/// `tiny_dw_guided`). Data, teacher, and step budget still have to match.
/// Loss weights and `run.seed` may differ. The seed draws that student's
/// initial weights; crops follow the first config's seed.
pub fn run_mixed(configs: Vec<Config>, provenances: Vec<RunProvenance>) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .map_err(|e| anyhow::anyhow!("cannot configure data preparation thread pool: {e}"))?;

    if configs.is_empty() {
        bail!("cohort is empty");
    }
    if configs.len() != provenances.len() {
        bail!(
            "cohort has {} configs and {} provenances",
            configs.len(),
            provenances.len()
        );
    }
    let recipe = recipe_without_arch(&configs[0]);
    let mut seen_dirs = Vec::new();
    for config in &configs {
        config.validate()?;
        if recipe_without_arch(config) != recipe {
            bail!(
                "cohort member '{}' does not share data, teacher, or step budget with '{}'",
                config.run.name,
                configs[0].run.name
            );
        }
        if seen_dirs.contains(&config.run.output_dir) {
            bail!(
                "cohort output_dir {} is used more than once",
                config.run.output_dir.display()
            );
        }
        seen_dirs.push(config.run.output_dir.clone());
    }

    let config0 = configs[0].clone();
    let device = select_device(&config0.run.device);
    let mut rng = StdRng::seed_from_u64(config0.run.seed);
    let dataset = Arc::new(TrainContentSource::build(&config0.data)?);
    let crop = config0.data.crop as u32;
    let batch = config0.data.batch_size;
    let supervision = Supervision::try_from_config(&config0, &device)?;
    let feature_weight = configs
        .iter()
        .map(|c| c.loss.feature)
        .fold(0.0f32, f32::max);
    if feature_weight > 0.0 {
        match supervision.teacher() {
            None => bail!(
                "loss.feature > 0 requires teacher.enabled = true (feature distillation is teacher-guided)"
            ),
            Some(teacher) if !teacher.has_features() => {
                bail!("loss.feature > 0 requires a MicroAST teacher (teacher.kind = \"microast\")")
            }
            Some(_) => {}
        }
    }

    let names = configs
        .iter()
        .map(|c| format!("{}:{}", c.run.name, c.student.arch))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "cohort of {}: one teacher forward per step, then each student backward ({names}); crops use seed {}",
        configs.len(),
        config0.run.seed,
    );

    let cohort = configs.len() > 1;
    let mut slots: Vec<Box<dyn CohortSlot>> = Vec::with_capacity(configs.len());
    for (config, provenance) in configs.into_iter().zip(provenances) {
        let slot = match config.student.arch.as_str() {
            "tiny_dw" => boxed_slot(config, provenance, &device, &dataset, cohort, build_tiny)?,
            "tiny_dw_guided" => {
                boxed_slot(config, provenance, &device, &dataset, cohort, build_guided)?
            }
            other => bail!("mixed cohort does not support student.arch {other:?}"),
        };
        slots.push(slot);
    }
    TrainBackend::seed(&device, config0.run.seed);

    let feature_scales: Vec<f32> = if feature_weight > 0.0 {
        let teacher = supervision.teacher().expect("checked above");
        let probe = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let (_, feats) = teacher.features(probe.content.inner())?;
        let scales: Vec<f32> = feats
            .iter()
            .map(|f| (f.clone().abs().mean().into_scalar().to_f64() as f32).max(1e-3))
            .collect();
        log::info!("feature distillation scales (per stage): {scales:?}");
        scales
    } else {
        Vec::new()
    };

    let iterations = config0.optim.iterations;
    let log_every = config0.run.log_every;
    let val_every = config0.run.val_every;
    let checkpoint_every = config0.run.checkpoint_every;
    let sample_every = config0.run.sample_every;
    let val_batch_size = config0.run.val_batch_size.max(1);
    let mut val_rng = StdRng::seed_from_u64(config0.run.seed ^ VAL_BATCH_SEED);
    let started = Instant::now();
    let remember_target = config0.data.overfit_long_edge.is_some();
    let mut cached_target: Option<Tensor<InferBackend, 4>> = None;

    let (batch_tx, batch_rx) = mpsc::sync_channel::<Result<CpuBatch>>(2);
    let producer_dataset = Arc::clone(&dataset);
    let producer = thread::Builder::new()
        .name("training-data-prefetch".into())
        .spawn(move || {
            let mut train_rng = rng;
            for _ in 0..iterations {
                let batch = prepare_cpu_batch(&producer_dataset, crop, batch, &mut train_rng);
                if batch_tx.send(batch).is_err() {
                    break;
                }
            }
        })
        .context("start training data prefetch thread")?;

    let overlap_teacher =
        supervision.pipelines_on_host() && !remember_target && feature_weight == 0.0;
    if overlap_teacher {
        println!("onnx teacher runs one batch ahead of the student backward");
    }
    let mut ahead: Option<HostAhead> = None;

    for step in 1..=iterations {
        let (cpu_batch, host_target) = if let Some(mut pending) = ahead.take() {
            pending.finish()?;
            (pending.cpu, pending.ready)
        } else {
            (
                batch_rx
                    .recv()
                    .context("training data prefetch thread stopped")??,
                None,
            )
        };
        let batch_data = upload_batch(cpu_batch, &device);
        let content = batch_data.content;
        let (target, teacher_feats) = if let Some(buf) = host_target {
            let [n, _, h, w] = content.dims();
            (host_target_tensor(buf, n, h, w, &device)?, Vec::new())
        } else {
            resolve_step_target(
                &supervision,
                content.clone(),
                batch_data.target,
                feature_weight > 0.0,
                remember_target,
                &mut cached_target,
            )?
        };
        let _ = target.clone().sum().into_scalar();
        let log_this = log_every > 0 && (step % log_every == 0 || step == iterations);
        if overlap_teacher && step < iterations {
            ahead = Some(launch_host_ahead(&batch_rx, &supervision)?);
        }

        for slot in &mut slots {
            slot.train_step(
                content.clone(),
                target.clone(),
                &teacher_feats,
                &feature_scales,
                cohort,
                step,
                iterations,
                log_this,
                &started,
            )?;
        }

        reclaim_teacher(&mut ahead, step, iterations, val_every, sample_every)?;
        if val_every > 0 && step % val_every == 0 {
            let val_data = build_batch(&dataset, crop, val_batch_size, &mut val_rng, &device)?;
            let (val_target, val_teacher_feats) = supervision.resolve_batch(
                val_data.content.clone(),
                val_data.target,
                feature_weight > 0.0,
            )?;
            let val_teacher_inner: Vec<Tensor<InferBackend, 4>> = val_teacher_feats
                .iter()
                .map(|t| t.clone().inner())
                .collect();
            for slot in &mut slots {
                slot.validate_step(
                    val_data.content.clone().inner(),
                    val_target.clone().inner(),
                    &val_teacher_inner,
                    &feature_scales,
                    cohort,
                    step,
                )?;
            }
        }

        for slot in &mut slots {
            slot.checkpoint_and_sample(
                step,
                iterations,
                checkpoint_every,
                sample_every,
                supervision.teacher(),
                &device,
            )?;
        }
    }

    producer
        .join()
        .map_err(|_| anyhow::anyhow!("training data prefetch thread panicked"))?;
    let wall_s = started.elapsed().as_secs_f64();
    for slot in &mut slots {
        slot.finish(iterations, wall_s, cohort)?;
    }
    Ok(())
}

fn load_sample_images(
    config: &Config,
    dataset: &TrainContentSource,
) -> Result<Vec<(String, image::RgbImage)>> {
    let paths = match &config.run.sample_content {
        Some(p) => image_paths(p)?,
        None => match dataset.first_content_path() {
            Some(p) => vec![p],
            None => bail!("run.sample_content must be set when training from data.hf"),
        },
    };
    let limit = if config.run.sample_limit == 0 {
        paths.len()
    } else {
        config.run.sample_limit
    };
    let mut out = Vec::new();
    for path in paths.into_iter().take(limit) {
        let img = load_rgb(&path)?;
        let img = match config.data.overfit_long_edge {
            Some(edge) => overfit_frame(&img, edge as u32),
            None => resize_long_edge(
                &img,
                config.run.sample_size as u32,
                image::imageops::FilterType::Lanczos3,
            ),
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("sample")
            .to_owned();
        save_rgb(
            &img,
            config.run.output_dir.join(format!("content_{name}.png")),
        )?;
        out.push((name, img));
    }
    Ok(out)
}

/// `LOW` when the upsampled paint carries less of the output layout than this.
/// The two collapsed `whitebox_look` arms scored 0.00; a healthy `b_h4` 0.26–0.47.
const LOW_PAINT_SPATIAL: f64 = 0.1;

/// `LOW` when any output channel's slope on the teacher is below this.
/// A healthy `b_h4` bottoms out at 0.67. The `whitebox_look` colour casts sat at 0.14–0.40,
/// and one of them kept a live paint, so only the slope catches it.
const LOW_CHANNEL_SLOPE: f64 = 0.5;

/// Run the validation (non-autodiff) model on one image and save the result next to the
/// teacher's own output, so the log carries an objective quality signal and not just the
/// training loss.
fn sample<M>(
    model: &M,
    image: &image::RgbImage,
    name: &str,
    teacher: Option<&dyn DistillTeacher>,
    output_dir: &std::path::Path,
    step: usize,
    device: &TrainDevice,
    run_name: &str,
) -> Result<SampleMetrics>
where
    M: AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend>,
{
    let valid = model.valid();
    // Same contract as infer: stride-2 down and nearest up only round-trip when both
    // edges are multiples of 4. The H/2 skip adds those two tensors, so a 270-wide
    // sample (enc → 135, trunk up → 136) panics. Pad, then crop the metrics back.
    let (w, h) = image.dimensions();
    let padded = crate::data::pad_to_multiple(image, 4);
    let content = rgb_to_tensor::<InferBackend>(&padded, device);
    let styled = valid.stylize(content.clone());
    let mut out = tensor_to_rgb(styled)?;
    if (w, h) != padded.dimensions() {
        out = image::imageops::crop_imm(&out, 0, 0, w, h).to_image();
    }
    let path = output_dir.join(format!("sample_{step:06}_{name}.png"));
    save_rgb(&out, &path)?;

    let lapvar = eval::lapvar(&out);
    let checkerboard = eval::checkerboard(&out);
    let saturation = eval::saturation(&out);
    let mut metrics = SampleMetrics {
        lapvar,
        checkerboard,
        saturation,
        psnr_vs_teacher: None,
        lapvar_teacher: None,
        lapvar_ratio: None,
        energy_4_8: None,
        energy_8_16: None,
        slope_r: None,
        slope_g: None,
        slope_b: None,
        paint_spatial: None,
        in_dead: None,
    };
    let mut line = format!(
        "  [{run_name}] sample {step:>7} {name:<12} lapvar {:.5}  checker {:.3e}  sat {:.3}",
        lapvar, checkerboard, saturation,
    );
    if let Some(signal) = valid.sample_signal(content.clone()) {
        metrics.paint_spatial = Some(signal.paint_spatial as f64);
        metrics.in_dead = Some(signal.in_dead as f64);
        line.push_str(&format!(
            "  paint {:.3}  in_dead {:.2}",
            signal.paint_spatial, signal.in_dead,
        ));
    }
    let mut low = metrics.paint_spatial.is_some_and(|v| v < LOW_PAINT_SPATIAL);
    if let Some(teacher) = teacher {
        let mut target = tensor_to_rgb(teacher.stylize(content)?)?;
        if (w, h) != padded.dimensions() {
            target = image::imageops::crop_imm(&target, 0, 0, w, h).to_image();
        }
        save_rgb(
            &target,
            output_dir.join(format!("target_{step:06}_{name}.png")),
        )?;
        let psnr = eval::psnr(&out, &target)?;
        let lapvar_t = eval::lapvar(&target);
        let lapvar_ratio = lapvar / lapvar_t.max(1e-12);
        let (energy_4_8, energy_8_16) = eval::energy_ratio(&out, &target);
        metrics.psnr_vs_teacher = Some(psnr);
        metrics.lapvar_teacher = Some(lapvar_t);
        metrics.lapvar_ratio = Some(lapvar_ratio);
        metrics.energy_4_8 = Some(energy_4_8);
        metrics.energy_8_16 = Some(energy_8_16);
        let slopes = eval::channel_slopes(&out, &target)?;
        metrics.slope_r = Some(slopes[0]);
        metrics.slope_g = Some(slopes[1]);
        metrics.slope_b = Some(slopes[2]);
        if slopes.iter().any(|s| *s < LOW_CHANNEL_SLOPE) {
            low = true;
        }
        line.push_str(&format!(
            "  psnr {:>5.2} dB  lapvar_ratio {:.2}  e4-8 {:.2}  e8-16 {:.2}  slope {:.2}/{:.2}/{:.2}",
            psnr, lapvar_ratio, energy_4_8, energy_8_16, slopes[0], slopes[1], slopes[2],
        ));
    }
    if low {
        line.push_str("  LOW");
    }
    println!("{line}");
    Ok(metrics)
}
