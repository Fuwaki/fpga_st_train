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
    image_paths, load_rgb, resize_long_edge, resize_short_edge, rgb_to_tensor, save_rgb,
    tensor_to_rgb,
};
use crate::eval;
use crate::losses;
use crate::student::StudentModel;
use crate::teacher::Stylizer;
use crate::teacher::weights;

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
struct Batch {
    content: Tensor<TrainBackend, 4>,
    target: Option<Tensor<TrainBackend, 4>>,
}

fn build_batch(
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

/// Everything the calibration pass needs, independent of how the model was trained.
///
/// Calibration only looks at activation ranges, so it needs real content crops and nothing
/// else: no optimiser, no teacher, not even the training config.
#[derive(Debug, Clone)]
pub struct CalibrationJob {
    /// Folder of content images.
    pub content_dir: PathBuf,
    /// Random crop size fed to the network. Must be a multiple of 4.
    pub crop: usize,
    /// Images per batch.
    pub batch_size: usize,
    /// RNG seed, so calibrating the same model twice gives identical scales.
    pub seed: u64,
    /// How many batches to measure.
    pub batches: usize,
    /// Multiplier applied to the observed maxima before dividing by the code maximum.
    /// 1.0 keeps every observed activation representable.
    pub safety: f32,
}

impl Default for CalibrationJob {
    fn default() -> Self {
        Self {
            content_dir: PathBuf::from("data/content"),
            crop: 192,
            batch_size: 4,
            seed: 0,
            batches: 20,
            safety: 1.0,
        }
    }
}

/// Measure the activation range at every quantised point of a trained student.
///
/// Returns symmetric scales (absmax * safety / code_max) for the requested bit widths. This is
/// the whole of the post-training recipe: the ranges depend only on the weights, so a handful
/// of content batches is enough.
pub fn calibrate<M, F>(
    job: &CalibrationJob,
    arch: &str,
    build_student: F,
    device: &TrainDevice,
    checkpoint: &std::path::Path,
    bits: crate::quant::Bits,
) -> Result<crate::quant::Calibration>
where
    M: StudentModel<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend>,
    F: FnOnce(&TrainDevice) -> Result<M>,
{
    let mut rng = StdRng::seed_from_u64(job.seed);
    let dataset = PairDataset::build(&job.content_dir, None, None)?;

    let model = load_checkpoint(build_student(device)?, checkpoint, device)?;
    let points = model.quant_points();
    if points.is_empty() {
        bail!("this student has no quantised points (StudentModel::quant_points is empty)");
    }
    // Inference only: skip the autodiff machinery entirely.
    let model = model.valid();
    log::info!(
        "calibrating {} quantised points for {} on {} batches of {}x{} crops",
        points.len(),
        bits.label(),
        job.batches,
        job.batch_size,
        job.crop
    );

    let mut absmax = vec![0.0f32; points.len()];
    for step in 1..=job.batches {
        let batch = build_batch(&dataset, job.crop as u32, job.batch_size, &mut rng, device)?;
        model.observe_activations(batch.content.valid(), &mut |point, value| {
            if value > absmax[point] {
                absmax[point] = value;
            }
        });
        if step % 5 == 0 || step == job.batches {
            log::info!("calibrated {step}/{} batches", job.batches);
        }
    }

    let scales: Vec<f32> = absmax
        .iter()
        .map(|m| (m * job.safety) / bits.act_max())
        .map(|s| s.max(1e-5))
        .collect();
    Ok(crate::quant::Calibration {
        arch: arch.to_owned(),
        points,
        scales,
        absmax,
        safety: job.safety,
        batches: job.batches,
        bits,
    })
}

/// Run training. build_student constructs the concrete model (see main.rs).
pub fn run<M, F>(config: Config, build_student: F) -> Result<()>
where
    M: StudentModel<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend>,
    F: FnOnce(&crate::config::StudentConfig, &TrainDevice) -> Result<M>,
{
    config.validate()?;
    let device = select_device(&config.run.device);

    std::fs::create_dir_all(&config.run.output_dir)
        .with_context(|| format!("cannot create {}", config.run.output_dir.display()))?;
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

    // ---- teacher ---------------------------------------------------------------------
    // The style branch is a constant for a fixed style, so it is precomputed once here
    // instead of being recomputed on every step (see teacher::stylizer).
    let teacher: Option<Stylizer> = if config.teacher.enabled {
        let started = Instant::now();
        let model = weights::load(&config.teacher.weights_dir, &device)
            .context("failed to load the MicroAST teacher")?;
        log::info!("teacher loaded in {:.2} s", started.elapsed().as_secs_f32());
        let img = load_rgb(&config.teacher.style)
            .with_context(|| format!("cannot load style {}", config.teacher.style.display()))?;
        let img = resize_short_edge(
            &img,
            config.teacher.style_size as u32,
            image::imageops::FilterType::Triangle,
        );
        let style = rgb_to_tensor::<InferBackend>(&img, &device);
        log::info!("style branch precomputed (fixed style, alpha fixed at 1.0)");
        Some(Stylizer::new(model, style))
    } else {
        None
    };

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

    let pixel_weight = config.loss.pixel;
    let gradient_weight = config.loss.gradient;
    let feature_weight = config.loss.feature;
    let tv_weight = config.loss.tv;
    if feature_weight > 0.0 && teacher.is_none() {
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
        "training '{}' for {} steps: batch {batch}, crop {crop}, device {}, loss = {}*L1 + {}*gradL1 + {}*TV",
        config.run.name,
        config.optim.iterations,
        config.run.device,
        pixel_weight,
        gradient_weight,
        tv_weight,
    );
    if feature_weight > 0.0 {
        println!("feature distillation: weight {feature_weight} over the teacher decoder stages");
    }

    // Feature-distillation normalisation: each stage's L1 is divided by that stage's mean
    // absolute activation, so the three stages contribute comparably regardless of scale.
    // Measured once up front, which also keeps the training step free of device syncs.
    let feature_scales: Vec<f32> = if feature_weight > 0.0 {
        let teacher = teacher.as_ref().expect("checked above");
        let probe = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let probe = Tensor::<InferBackend, 4>::from_data(probe.content.into_data(), &device);
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
        let expected = model.quant_points();
        if expected.is_empty() {
            bail!("this student has no quantised points (StudentModel::quant_points is empty)");
        }
        if calibration.scales.len() != expected.len() {
            bail!(
                "{} has {} activation scales but the model has {} quantised points",
                path.display(),
                calibration.scales.len(),
                expected.len()
            );
        }
        println!(
            "low-bit QAT: {} ({} activation points, image scale {:.5}, largest scale {:.4})",
            calibration.bits.label(),
            expected.len(),
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

    // ---- loop ------------------------------------------------------------------------
    let started = Instant::now();
    let mut window_loss = 0.0f64;
    let mut window_pixel = 0.0f64;
    let mut window_gradient = 0.0f64;
    let mut window_feature = 0.0f64;
    let mut window_tv = 0.0f64;
    let mut window_count = 0usize;

    for step in 1..=config.optim.iterations {
        let batch_data = build_batch(&dataset, crop, batch, &mut rng, &device)?;
        let content = batch_data.content;
        let provided_target = batch_data.target;

        // A single teacher pass per step supplies both the pseudo ground truth and (when
        // feature distillation is on) the intermediate features the student is matched to.
        let teacher_input: Option<Tensor<InferBackend, 4>> = if teacher.is_some() {
            Some(Tensor::<InferBackend, 4>::from_data(
                content.clone().into_data(),
                &device,
            ))
        } else {
            None
        };
        let (teacher_out, teacher_feats): (
            Option<Tensor<InferBackend, 4>>,
            Vec<Tensor<InferBackend, 4>>,
        ) = match (&teacher, &teacher_input) {
            (Some(teacher), Some(input)) if provided_target.is_none() || feature_weight > 0.0 => {
                let (out, feats) = teacher.features(input.clone());
                (
                    Some(out),
                    if feature_weight > 0.0 {
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
            (None, Some(out)) => Tensor::<TrainBackend, 4>::from_data(out.into_data(), &device),
            (None, None) => bail!(
                "some samples have no pseudo ground truth and the teacher is disabled -- \
                 generate targets with the 'stylize' subcommand, or set teacher.enabled = true"
            ),
        };
        let teacher_feats: Vec<Tensor<TrainBackend, 4>> = teacher_feats
            .into_iter()
            .map(|f| Tensor::<TrainBackend, 4>::from_data(f.into_data(), &device))
            .collect();

        let (output, student_feats) = match &quant_scheme {
            Some(scheme) => (model.forward_quantised(content, scheme), Vec::new()),
            None => model.intermediate(content),
        };
        let pixel = losses::l1(output.clone(), target.clone());
        let pixel_value = pixel.clone().into_scalar().to_f64();
        let mut loss = pixel.mul_scalar(pixel_weight);

        // Feature-level distillation: match the student's decoder stages to the teacher's,
        // which is where the brush texture actually lives (plain output L1 cannot ask for it
        // -- the L1 optimum is the conditional mean, i.e. a blurred picture).
        let mut feature_value = 0.0f64;
        if feature_weight > 0.0
            && !student_feats.is_empty()
            && student_feats.len() == teacher_feats.len()
        {
            let stages = student_feats.len() as f32;
            let mut total: Option<Tensor<TrainBackend, 1>> = None;
            for ((student, teacher), scale) in student_feats
                .into_iter()
                .zip(teacher_feats)
                .zip(feature_scales.iter())
            {
                let stage = (student - teacher).abs().mean().div_scalar(*scale);
                total = Some(match total {
                    None => stage,
                    Some(acc) => acc + stage,
                });
            }
            if let Some(total) = total {
                let term = total.div_scalar(stages);
                feature_value = term.clone().into_scalar().to_f64();
                loss = loss + term.mul_scalar(feature_weight);
            }
        }

        let mut gradient_value = 0.0f64;
        if gradient_weight > 0.0 {
            let term = losses::gradient_l1(output.clone(), target);
            gradient_value = term.clone().into_scalar().to_f64();
            loss = loss + term.mul_scalar(gradient_weight);
        }
        let mut tv_value = 0.0f64;
        if tv_weight > 0.0 {
            let term = losses::total_variation(output);
            tv_value = term.clone().into_scalar().to_f64();
            loss = loss + term.mul_scalar(tv_weight);
        }
        let loss_value = loss.clone().into_scalar().to_f64();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = schedule.next();
        model = optimiser.step(lr, model, grads);

        window_loss += loss_value;
        window_pixel += pixel_value;
        window_gradient += gradient_value;
        window_feature += feature_value;
        window_tv += tv_value;
        window_count += 1;

        if config.run.log_every > 0 && step % config.run.log_every == 0 {
            let elapsed = started.elapsed().as_secs_f64();
            let n = window_count.max(1) as f64;
            println!(
                "step {step:>7}/{:<7} loss {:>9.5}  l1 {:.5}  grad {:.5}  feat {:.5}  tv {:.5}  lr {:.3e}  {:.2} step/s  {:.1} s",
                config.optim.iterations,
                window_loss / n,
                window_pixel / n,
                window_gradient / n,
                window_feature / n,
                window_tv / n,
                lr,
                step as f64 / elapsed.max(1e-9),
                elapsed,
            );
            window_loss = 0.0;
            window_pixel = 0.0;
            window_gradient = 0.0;
            window_feature = 0.0;
            window_tv = 0.0;
            window_count = 0;
        }
        if config.run.checkpoint_every > 0 && step % config.run.checkpoint_every == 0 {
            save_checkpoint(
                &model,
                config.run.output_dir.join(format!("step_{step:06}.mpk")),
            )?;
        }
        if config.run.sample_every > 0 && step % config.run.sample_every == 0 {
            for (name, img) in &sample_images {
                if let Err(e) = sample(
                    &model,
                    img,
                    name,
                    teacher.as_ref(),
                    &config.run.output_dir,
                    step,
                    &device,
                ) {
                    log::warn!("could not write sample '{name}' at step {step}: {e}");
                }
            }
        }
    }

    // The canonical name: a run directory plus its model card is a runnable model directory,
    // so predict/infer/gallery scripts can point straight at runs/<tag>.
    let final_path = config.run.output_dir.join("model.mpk");
    save_checkpoint(&model, final_path.clone())?;
    println!(
        "done in {:.1} s -> {}",
        started.elapsed().as_secs_f32(),
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
) -> Result<()>
where
    M: AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<<TrainBackend as AutodiffBackend>::InnerBackend>,
{
    let valid = model.valid();
    let content = rgb_to_tensor::<InferBackend>(image, device);
    let styled = valid.stylize(content.clone());
    let out = tensor_to_rgb(styled)?;
    let path = output_dir.join(format!("sample_{step:06}_{name}.png"));
    save_rgb(&out, &path)?;

    let mut line = format!(
        "  sample {step:>7} {name:<12} lapvar {:.5}  checker {:.3e}  sat {:.3}",
        eval::lapvar(&out),
        eval::checkerboard(&out),
        eval::saturation(&out),
    );
    if let Some(teacher) = teacher {
        let target = tensor_to_rgb(teacher.stylize(content))?;
        save_rgb(
            &target,
            output_dir.join(format!("target_{step:06}_{name}.png")),
        )?;
        line.push_str(&format!(
            "  psnr_vs_teacher {:>5.2} dB  lapvar_t {:.5}",
            eval::psnr(&out, &target)?,
            eval::lapvar(&target),
        ));
    }
    println!("{line}");
    Ok(())
}
