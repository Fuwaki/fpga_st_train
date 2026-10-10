//! The training loop.
//!
//! Deliberately hand written rather than built on burn::train::Learner: style transfer
//! needs two inputs per sample, several loss terms and per-term logging, which reads more
//! clearly here than through TrainStep. It also means your model only has to implement
//! [crate::student::StudentModel] -- no TrainStep, no Learner plumbing.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::{InferBackend, TrainBackend, TrainDevice, select_device};
use crate::config::{Config, OptimConfig, Schedule};
use crate::data::dataset::{PairDataset, load_pair_cropped};
use crate::data::{
    image_paths, load_rgb, resize_long_edge, rgb_to_tensor, save_rgb,
    tensor_to_rgb,
};
use crate::eval;
use crate::losses;
use crate::run_log::{RunLog, RunProvenance, SampleMetrics, TrainStepMetrics, ValStepMetrics};
use crate::quant::Quantizable;
use crate::student::StudentModel;
use crate::supervision::Supervision;
use crate::teacher::Stylizer;

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

pub(crate) fn build_batch(
    dataset: &PairDataset,
    crop: u32,
    batch: usize,
    rng: &mut StdRng,
    device: &TrainDevice,
) -> Result<Batch> {
    let mut contents = Vec::with_capacity(batch);
    let mut targets = Vec::with_capacity(batch);
    let mut complete = true;
    for _ in 0..batch {
        let pair = dataset.sample(rng);
        let (content_img, target_img) = load_pair_cropped(pair, crop, rng)?;
        contents.push(rgb_to_tensor::<TrainBackend>(&content_img, device));
        match target_img {
            Some(t) => targets.push(rgb_to_tensor::<TrainBackend>(&t, device)),
            None => complete = false,
        }
    }
    Ok(Batch {
        content: Tensor::cat(contents, 0),
        target: if complete {
            Some(Tensor::cat(targets, 0))
        } else {
            None
        },
    })
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
    stroke: f32,
    feature: f32,
    tv: f32,
}

struct LossParts<B: Backend> {
    total: Tensor<B, 1>,
    pixel: Tensor<B, 1>,
    gradient: Option<Tensor<B, 1>>,
    stroke: Option<Tensor<B, 1>>,
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
            gradient: self.gradient.as_ref().map(|t| t.clone().into_scalar().to_f64()),
            stroke: self.stroke.as_ref().map(|t| t.clone().into_scalar().to_f64()),
            feature: self.feature.as_ref().map(|t| t.clone().into_scalar().to_f64()),
            tv: self.tv.as_ref().map(|t| t.clone().into_scalar().to_f64()),
        }
    }

    fn val_metrics(&self) -> ValStepMetrics {
        ValStepMetrics {
            loss: self.total.clone().into_scalar().to_f64(),
            l1: Some(self.pixel.clone().into_scalar().to_f64()),
            gradient: self.gradient.as_ref().map(|t| t.clone().into_scalar().to_f64()),
            stroke: self.stroke.as_ref().map(|t| t.clone().into_scalar().to_f64()),
            feature: self.feature.as_ref().map(|t| t.clone().into_scalar().to_f64()),
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
        let term = losses::stroke_l1(output.clone(), target);
        stroke = Some(term.clone());
        total = total + term.mul_scalar(weights.stroke);
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
        stroke,
        feature,
        tv,
    }
}

/// Run training. `build_student` constructs the concrete model (see [crate::registry]).
pub fn run<M, F>(config: Config, provenance: RunProvenance, build_student: F) -> Result<()>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule:
        StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend> + Quantizable<InferBackend>,
    F: FnOnce(&crate::config::StudentConfig, &TrainDevice) -> Result<M>,
{
    config.validate()?;
    let device = select_device(&config.run.device);

    let mut run_log = RunLog::open(&config.run.output_dir, &config, &provenance)?;
    log::info!("metrics -> {}", run_log.path().display());

    // Make the run directory self-describing: the inference tool rebuilds the architecture from
    // this card instead of being handed the training config it came from.
    crate::config::ModelCard::of(&config.student)
        .with_meta("run", serde_json::Value::String(config.run.name.clone()))
        .with_meta(
            "style",
            serde_json::Value::String(config.teacher.style.display().to_string()),
        )
        .with_meta(
            "iterations",
            serde_json::Value::Number(config.optim.iterations.into()),
        )
        .save(&config.run.output_dir)?;

    let mut rng = StdRng::seed_from_u64(config.run.seed);

    // ---- data ------------------------------------------------------------------------
    let dataset = PairDataset::build(
        &config.data.content_dir,
        config.data.pseudo_gt_dir.as_deref(),
        config.data.limit,
    )?;
    let crop = config.data.crop as u32;
    let batch = config.data.batch_size;

    let supervision = Supervision::try_from_config(&config, &device)?;

    // ---- model / optimiser -----------------------------------------------------------
    let mut model: M = build_student(&config.student, &device)?;
    if let Some(path) = &config.run.init_from {
        model = load_checkpoint(model, path, &device)?;
        log::info!("warm start from {}", path.display());
    }
    model.after_load();
    log::info!("student: {}", model.describe());

    let mut optimiser = AdamConfig::new()
        .with_beta_1(config.optim.beta_1)
        .with_beta_2(config.optim.beta_2)
        .with_epsilon(config.optim.epsilon)
        .init::<TrainBackend, M>();
    let mut schedule = LrSchedule::build(&config.optim)?;

    let loss_weights = LossWeights {
        pixel: config.loss.pixel,
        gradient: config.loss.gradient,
        stroke: config.loss.stroke,
        feature: config.loss.feature,
        tv: config.loss.tv,
    };
    let feature_weight = loss_weights.feature;
    if feature_weight > 0.0 && supervision.teacher().is_none() {
        bail!(
            "loss.feature > 0 requires teacher.enabled = true (feature distillation is teacher-guided)"
        );
    }
    if config.loss.perceptual > 0.0 || config.loss.style > 0.0 {
        log::warn!(
            "perceptual/style weights are set but the VGG terms are not implemented yet; \
             they are currently ignored"
        );
    }

    println!(
        "training '{}' for {} steps: batch {batch}, crop {crop}, device {}, loss = {}*L1 + {}*gradL1 + {}*stroke + {}*TV",
        config.run.name,
        config.optim.iterations,
        config.run.device,
        loss_weights.pixel,
        loss_weights.gradient,
        loss_weights.stroke,
        loss_weights.tv,
    );
    if feature_weight > 0.0 {
        println!("feature distillation: weight {feature_weight} over the teacher decoder stages");
    }

    // Feature-distillation normalisation: each stage's L1 is divided by that stage's mean
    // absolute activation, so the three stages contribute comparably regardless of scale.
    // Measured once up front, which also keeps the training step free of device syncs.
    let feature_scales: Vec<f32> = if feature_weight > 0.0 {
        let teacher = supervision.teacher().expect("checked above");
        let probe = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let probe = probe.content.inner();
        let (_, feats) = teacher.features(probe);
        let scales: Vec<f32> = feats
            .iter()
            .map(|f| (f.clone().abs().mean().into_scalar().to_f64() as f32).max(1e-3))
            .collect();
        log::info!("feature distillation scales (per stage): {scales:?}");
        scales
    } else {
        Vec::new()
    };

    // ---- int8 quantisation-aware training ---------------------------------------------
    // The activation scales are calibrated once, on the fp32 (or already fine-tuned) model,
    // and then held fixed: the grid the weights must adapt to is the one the accelerator will
    // actually use.
    let quant_scheme: Option<crate::quant::QuantScheme> = if config.quant.enabled {
        let path = config.quant.calibration.as_ref().ok_or_else(|| {
            anyhow::anyhow!("quant.enabled = true requires quant.calibration (run the 'calibrate' subcommand first)")
        })?;
        let calibration = crate::quant::Calibration::load(path)?;
        calibration.validate_for(&model)?;
        let n_act = model.quant_points().len();
        println!(
            "low-bit QAT: {} ({} activation points, image scale {:.5}, largest scale {:.4})",
            calibration.bits.label(),
            n_act,
            calibration.scales[0],
            calibration.scales.iter().cloned().fold(0.0f32, f32::max)
        );
        Some(calibration.scheme())
    } else {
        None
    };

    // Progress samples: a fixed, held-out image (or folder of images) is much more
    // informative than the first training image, which is also being fitted.
    let sample_images: Vec<(String, image::RgbImage)> = {
        let paths = match &config.run.sample_content {
            Some(p) => image_paths(p)?,
            None => vec![dataset.get(0).content.clone()],
        };
        let mut out = Vec::new();
        for path in paths.into_iter().take(3) {
            let img = load_rgb(&path)?;
            let img = resize_long_edge(
                &img,
                config.run.sample_size as u32,
                image::imageops::FilterType::Lanczos3,
            );
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
        out
    };

    let val_batch_size = config.run.val_batch_size.max(1);
    let mut val_rng = StdRng::seed_from_u64(config.run.seed ^ VAL_BATCH_SEED);

    // ---- loop ------------------------------------------------------------------------
    let started = Instant::now();
    let mut last_train_metrics: Option<TrainStepMetrics> = None;

    for step in 1..=config.optim.iterations {
        let batch_data = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let content = batch_data.content;
        let (target, teacher_feats) = supervision.resolve_batch(
            content.clone(),
            batch_data.target,
            feature_weight > 0.0,
        )?;

        let parts = forward_loss(
            &model,
            content,
            target,
            &teacher_feats,
            &feature_scales,
            &loss_weights,
            quant_scheme.as_ref(),
        );
        let loss = parts.total.clone();

        // Avoid per-step `into_scalar` GPU→CPU sync (hurts Gen1/x4 PCIe badly).
        let log_this = config.run.log_every > 0 && step % config.run.log_every == 0;

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = schedule.next();
        model = optimiser.step(lr, model, grads);

        if log_this {
            let elapsed = started.elapsed().as_secs_f64();
            let metrics = parts.train_metrics(lr, Some(step as f64 / elapsed.max(1e-9)));
            last_train_metrics = Some(metrics.clone());
            println!(
                "step {step:>7}/{:<7} loss {:>9.5}  l1 {:.5}  grad {:.5}  stroke {:.5}  feat {:.5}  tv {:.5}  lr {:.3e}  {:.2} step/s  {:.1} s",
                config.optim.iterations,
                metrics.loss,
                metrics.l1.unwrap_or(0.0),
                metrics.gradient.unwrap_or(0.0),
                metrics.stroke.unwrap_or(0.0),
                metrics.feature.unwrap_or(0.0),
                metrics.tv.unwrap_or(0.0),
                metrics.lr,
                metrics.steps_per_s.unwrap_or(0.0),
                elapsed,
            );
            run_log.train_step(step, metrics)?;
        }

        let val_this = config.run.val_every > 0 && step % config.run.val_every == 0;
        if val_this {
            let val_data = build_batch(&dataset, crop, val_batch_size, &mut val_rng, &device)?;
            let (val_target, val_teacher_feats) = supervision.resolve_batch(
                val_data.content.clone(),
                val_data.target,
                feature_weight > 0.0,
            )?;
            let valid = model.valid();
            let val_teacher_inner: Vec<Tensor<InferBackend, 4>> = val_teacher_feats
                .iter()
                .map(|t| t.clone().inner())
                .collect();
            let val_parts = forward_loss(
                &valid,
                val_data.content.inner(),
                val_target.inner(),
                &val_teacher_inner,
                &feature_scales,
                &loss_weights,
                quant_scheme.as_ref(),
            );
            let val_metrics = val_parts.val_metrics();
            println!(
                "  val {step:>7} loss {:>9.5}  l1 {:.5}",
                val_metrics.loss,
                val_metrics.l1.unwrap_or(0.0),
            );
            run_log.val_step(step, val_metrics)?;
        }
        if config.run.checkpoint_every > 0 && step % config.run.checkpoint_every == 0 {
            save_checkpoint(
                &model,
                config.run.output_dir.join(format!("step_{step:06}.mpk")),
            )?;
        }
        if config.run.sample_every > 0 && step % config.run.sample_every == 0 {
            for (name, img) in &sample_images {
                match sample(
                    &model,
                    img,
                    name,
                    supervision.teacher(),
                    &config.run.output_dir,
                    step,
                    &device,
                ) {
                    Ok(metrics) => run_log.sample_eval(step, name, metrics)?,
                    Err(e) => {
                        log::warn!("could not write sample '{name}' at step {step}: {e}");
                    }
                }
            }
        }
    }

    // The canonical name: a run directory plus its model card is a runnable model directory,
    // so predict/infer/gallery scripts can point straight at runs/<tag>.
    let final_path = config.run.output_dir.join("model.mpk");
    save_checkpoint(&model, final_path.clone())?;
    let wall_s = started.elapsed().as_secs_f64();
    run_log.run_end(config.optim.iterations, wall_s, last_train_metrics)?;
    println!(
        "done in {:.1} s -> {}",
        wall_s,
        final_path.display()
    );
    Ok(())
}

/// Run the validation (non-autodiff) model on one image and save the result next to the
/// teacher's own output, so the log carries an objective quality signal and not just the
/// training loss.
fn sample<M>(
    model: &M,
    image: &image::RgbImage,
    name: &str,
    teacher: Option<&Stylizer>,
    output_dir: &std::path::Path,
    step: usize,
    device: &TrainDevice,
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
    };
    let mut line = format!(
        "  sample {step:>7} {name:<12} lapvar {:.5}  checker {:.3e}  sat {:.3}",
        lapvar,
        checkerboard,
        saturation,
    );
    if let Some(teacher) = teacher {
        let mut target = tensor_to_rgb(teacher.stylize(content))?;
        if (w, h) != padded.dimensions() {
            target = image::imageops::crop_imm(&target, 0, 0, w, h).to_image();
        }
        save_rgb(
            &target,
            output_dir.join(format!("target_{step:06}_{name}.png")),
        )?;
        let psnr = eval::psnr(&out, &target)?;
        let lapvar_t = eval::lapvar(&target);
        metrics.psnr_vs_teacher = Some(psnr);
        metrics.lapvar_teacher = Some(lapvar_t);
        line.push_str(&format!(
            "  psnr_vs_teacher {:>5.2} dB  lapvar_t {:.5}",
            psnr,
            lapvar_t,
        ));
    }
    println!("{line}");
    Ok(metrics)
}
