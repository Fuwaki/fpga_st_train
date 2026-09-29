//! Command line entry point.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::activation::relu;
use clap::{Args, Parser, Subcommand};
use image::imageops::FilterType;

use fpga_st_train::backend::TrainDevice;
use fpga_st_train::config::{Config, ModelCard, StudentConfig};
use fpga_st_train::student::{TinyDwStudent, TinyStudent};
use fpga_st_train::teacher::Teacher;
use fpga_st_train::teacher::microast::MicroAst;
use fpga_st_train::teacher::weights;
use fpga_st_train::{InferBackend, TrainBackend, data, eval};

#[derive(Parser, Debug)]
#[command(
    name = "fpga_st_train",
    version,
    about = "FPGA style transfer: MicroAST teacher inference and distillation training"
)]
struct Cli {
    /// Compute device: "default", "d0" (discrete GPU 0, the RX 580), "i0", "cpu".
    #[arg(long, global = true, default_value = "default")]
    device: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Stylise content images with the MicroAST teacher (single pair or whole folders).
    Stylize(StylizeArgs),
    /// Generate a configuration template.
    InitConfig(InitConfigArgs),
    /// Train the student.
    Train(TrainArgs),
    /// Run a delivered model directory over content images (the deployment entry point).
    Infer(InferArgs),
    /// Measure the activation ranges needed for int8 quantisation.
    Calibrate(CalibrateArgs),
    /// Write the int8 weights and the accelerator manifest.
    ExportInt8(ExportInt8Args),
    /// Report image-space quality metrics for a folder of images.
    Metrics(MetricsArgs),
    /// Dev: push raw f32 tensors through the teacher, for numerical parity checks.
    Parity(ParityArgs),
    /// Dev: run individual tensor ops and print statistics, to compare backends.
    Probe(ProbeArgs),
}

#[derive(Args, Debug)]
struct ProbeArgs {
    /// Write the report here instead of stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct StylizeArgs {
    /// Directory holding the four MicroAST .pth.tar checkpoints.
    #[arg(long)]
    teacher: PathBuf,
    /// Content image file or directory.
    #[arg(long)]
    content: PathBuf,
    /// Style image file or directory.
    #[arg(long)]
    style: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Resize the content so its long edge is at most this (0 = leave alone).
    #[arg(long, default_value_t = 0)]
    content_size: u32,
    /// Center-crop the content to this square size (0 = no crop).
    #[arg(long, default_value_t = 0)]
    crop: u32,
    /// Resize the style so its short edge is this.
    #[arg(long, default_value_t = 256)]
    style_size: u32,
    /// Stylisation strength. The released checkpoints only behave at 1.0.
    #[arg(long, default_value_t = 1.0)]
    alpha: f32,
    /// Output image extension.
    #[arg(long, default_value = "png")]
    ext: String,
}

#[derive(Args, Debug)]
struct InitConfigArgs {
    #[arg(long, default_value = "config.json")]
    out: PathBuf,
}

#[derive(Args, Debug)]
struct TrainArgs {
    /// Run configuration JSON (see the init-config subcommand).
    #[arg(long)]
    config: PathBuf,
    /// Override optim.iterations.
    #[arg(long)]
    iterations: Option<usize>,
    /// Override run.output_dir.
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct InferArgs {
    /// Model directory. See the infer subcommand help for the files it may contain.
    #[arg(long)]
    model: PathBuf,
    /// Content image file or directory.
    #[arg(long)]
    content: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// auto | fp32 | sim | int. `auto` runs the export when the directory has one.
    #[arg(long, default_value = "auto")]
    mode: String,
    /// Calibration JSON for `--mode sim`. Defaults to <model>/calibration.json.
    #[arg(long)]
    calibration: Option<PathBuf>,
    /// Resize the content so its long edge is at most this (0 = leave alone).
    #[arg(long, default_value_t = 0)]
    size: u32,
    /// Output image extension.
    #[arg(long, default_value = "png")]
    ext: String,
}

#[derive(Args, Debug)]
struct CalibrateArgs {
    /// Model directory (model.json + model.mpk).
    #[arg(long)]
    model: PathBuf,
    /// Folder of content images used to measure the activation ranges.
    #[arg(long)]
    content: PathBuf,
    /// Checkpoint to calibrate. Defaults to <model>/model.mpk.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Random crop size fed to the network (a multiple of 4).
    #[arg(long, default_value_t = 192)]
    crop: usize,
    /// Images per batch.
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    /// Number of content batches to measure the activation ranges on.
    #[arg(long, default_value_t = 20)]
    batches: usize,
    /// Multiplier applied to the observed maxima before dividing by the code maximum.
    /// 1.0 keeps every observed activation representable.
    #[arg(long, default_value_t = 1.0)]
    safety: f32,
    /// Seed for the content sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Bit widths, for example w8a8 or w4a8.
    #[arg(long, default_value = "w8a8")]
    bits: String,
    /// Where to write the calibration JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args, Debug)]
struct ExportInt8Args {
    /// Model directory (model.json + model.mpk).
    #[arg(long)]
    model: PathBuf,
    /// Checkpoint to export. Defaults to <model>/model.mpk.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Calibration JSON written by the calibrate subcommand.
    #[arg(long)]
    calibration: PathBuf,
    /// Output directory for model.json, weights.bin, manifest.json and calibration.json.
    #[arg(long)]
    out: PathBuf,
}

/// Measure the per-point activation ranges of a trained student.
fn calibrate(args: CalibrateArgs, device_spec: &str) -> Result<()> {
    let bits = fpga_st_train::quant::Bits::parse(&args.bits).map_err(|e| anyhow::anyhow!("{e}"))?;
    let card = ModelCard::load(&args.model)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.model.join("model.mpk"));
    let device = fpga_st_train::backend::select_device(device_spec);
    let job = fpga_st_train::train::CalibrationJob {
        content_dir: args.content.clone(),
        crop: args.crop,
        batch_size: args.batch_size,
        seed: args.seed,
        batches: args.batches,
        safety: args.safety,
    };

    let calibration = match card.arch.as_str() {
        "tiny" => fpga_st_train::train::calibrate::<TinyStudent<TrainBackend>, _>(
            &job,
            &card.arch,
            |device| Ok(tiny_with_options(&card.student(), device)),
            &device,
            &checkpoint,
            bits,
        )?,
        "tiny_dw" => fpga_st_train::train::calibrate::<TinyDwStudent<TrainBackend>, _>(
            &job,
            &card.arch,
            |device| Ok(tiny_dw_with_options(&card.student(), device)),
            &device,
            &checkpoint,
            bits,
        )?,
        other => bail!("unknown student arch {other:?}; register it in calibrate (main.rs)"),
    };
    calibration.save(&args.out)?;

    println!(
        "{} -- {:<14} {:>12} {:>12}",
        calibration.bits.label(),
        "point",
        "absmax",
        "scale"
    );
    for ((name, absmax), scale) in calibration
        .points
        .iter()
        .zip(&calibration.absmax)
        .zip(&calibration.scales)
    {
        println!("{name:<14} {absmax:>12.5} {scale:>12.6}");
    }
    println!(
        "{} quantised points over {} batches -> {}",
        calibration.scales.len(),
        calibration.batches,
        args.out.display()
    );
    Ok(())
}

/// Write the integer weights plus the manifest the accelerator reads.
///
/// The output directory becomes a complete model directory: alongside weights.bin and
/// manifest.json it gets model.json and calibration.json, so it can be run with
/// out-of-the-box (mode int) and re-simulated (mode sim) without the training run it came from.
fn export_int8(args: ExportInt8Args, device_spec: &str) -> Result<()> {
    use burn::module::AutodiffModule;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

    let card = ModelCard::load(&args.model)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.model.join("model.mpk"));
    let device = fpga_st_train::backend::select_device(device_spec);
    let calibration = fpga_st_train::quant::Calibration::load(&args.calibration)?;

    match card.arch.as_str() {
        "tiny" => {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            let model = tiny_with_options::<TrainBackend>(&card.student(), &device)
                .load_file(checkpoint.clone(), &recorder, &device)
                .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", checkpoint.display()))?
                .valid();
            let manifest = fpga_st_train::export::export_int8(&model, &calibration, &args.out)?;
            card.with_meta("bits", serde_json::Value::String(manifest.bits.label()))
                .with_meta(
                    "weight_packing",
                    serde_json::Value::String(manifest.weight_packing.clone()),
                )
                .with_meta("weight_bytes", serde_json::json!(manifest.weight_bytes))
                .with_meta(
                    "calibration_batches",
                    serde_json::json!(calibration.batches),
                )
                .with_meta("calibration_safety", serde_json::json!(calibration.safety))
                .save(&args.out)?;
            calibration.save(&args.out.join("calibration.json"))?;
            println!(
                "{} layers, {} {} weights ({:.1} KiB, {}) -> {}",
                manifest.layers.len(),
                manifest.weight_bytes,
                manifest.bits.label(),
                manifest.weight_bytes as f32 / 1024.0,
                manifest.weight_packing,
                args.out.display()
            );
            for layer in &manifest.layers {
                println!(
                    "  {:<9} {:>3}->{:<3} k{}x{} s{} pad{}  in {:.5}  out {:.5}  worst requant {:.5}",
                    layer.name,
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel[0],
                    layer.kernel[1],
                    layer.stride[0],
                    layer.reflect_pad,
                    layer.in_scale,
                    layer.out_scale,
                    layer.requant.iter().cloned().fold(0.0f32, f32::max),
                );
            }
        }
        "tiny_dw" => {
            bail!(
                "int8 export for arch tiny_dw is not implemented yet; train and infer in fp32 mode work"
            );
        }
        other => bail!("unknown student arch {other:?}; register it in export_int8 (main.rs)"),
    }
    Ok(())
}

#[derive(Args, Debug)]
struct MetricsArgs {
    /// Directory (or single file) of stylised images.
    #[arg(long)]
    images: PathBuf,
    /// Content image to compare against, for PSNR. Optional.
    #[arg(long)]
    content: Option<PathBuf>,
    /// Side-by-side sheet: "content,output1,output2" -> one PNG. Optional.
    #[arg(long)]
    sheet: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ParityArgs {
    #[arg(long)]
    teacher: PathBuf,
    /// Raw f32 tensor [1,3,H,W] written by data::save_f32.
    #[arg(long)]
    content_raw: PathBuf,
    #[arg(long)]
    style_raw: PathBuf,
    #[arg(long)]
    out_raw: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    alpha: f32,
}

// ===========================================================================
// The student registry
// ---------------------------------------------------------------------------
// One architecture exists: the FPGA-oriented fixed-style student in
// src/student/tiny.rs. To add another, put it under src/student/, implement
// StudentModel<B> for it (plus the optional quantisation hooks), and add one
// arm to each of the matches below.
//
// Note the *generic* impl over B: the training loop needs the trait for both
// the autodiff backend (training) and the inner backend (periodic samples).
// ===========================================================================

/// The FPGA-oriented fixed-style student (see src/student/tiny.rs).
fn tiny_with_options<B: Backend>(config: &StudentConfig, device: &B::Device) -> TinyStudent<B> {
    TinyStudent::new(
        config.usize_option("base", 32),
        config.usize_option("res_blocks", 4),
        config.usize_option("mid", 32),
        config.usize_option("fine", 16),
        device,
    )
}

fn build_tiny(config: &StudentConfig, device: &TrainDevice) -> Result<TinyStudent<TrainBackend>> {
    Ok(tiny_with_options(config, device))
}

fn tiny_dw_with_options<B: Backend>(
    config: &StudentConfig,
    device: &B::Device,
) -> TinyDwStudent<B> {
    TinyDwStudent::new(
        config.usize_option("base", 32),
        config.usize_option("res_blocks", 4),
        config.usize_option("mid", 32),
        config.usize_option("fine", 16),
        device,
    )
}

fn build_tiny_dw(
    config: &StudentConfig,
    device: &TrainDevice,
) -> Result<TinyDwStudent<TrainBackend>> {
    Ok(tiny_dw_with_options(config, device))
}

/// Build a student from a model card.
enum LoadedStudent {
    Tiny(TinyStudent<InferBackend>),
    TinyDw(TinyDwStudent<InferBackend>),
}

impl LoadedStudent {
    fn describe(&self) -> String {
        use fpga_st_train::student::StudentModel;
        match self {
            Self::Tiny(model) => format!("fp32 {}", model.describe()),
            Self::TinyDw(model) => format!("fp32 {}", model.describe()),
        }
    }

    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Tensor<InferBackend, 4> {
        use fpga_st_train::student::StudentModel;
        match self {
            Self::Tiny(model) => model.stylize(content),
            Self::TinyDw(model) => model.stylize(content),
        }
    }

    fn forward_quantised(
        &self,
        content: Tensor<InferBackend, 4>,
        scheme: &fpga_st_train::quant::QuantScheme,
    ) -> Tensor<InferBackend, 4> {
        use fpga_st_train::student::StudentModel;
        match self {
            Self::Tiny(model) => model.forward_quantised(content, scheme),
            Self::TinyDw(model) => model.forward_quantised(content, scheme),
        }
    }
}

/// The three ways a delivered model can be run.
enum Engine {
    /// Plain fp32 weights.
    Fp32(LoadedStudent),
    /// fp32 weights with low-bit fake quantisation, i.e. exactly what quantisation-aware
    /// training optimised. This is a *model* of the hardware, not the hardware.
    Sim(LoadedStudent, fpga_st_train::quant::QuantScheme),
    /// The exported integer graph, run by the bit-exact reference interpreter. This is what
    /// the FPGA executes, and it needs nothing but manifest.json and weights.bin.
    Fixed(fpga_st_train::intref::IntModel),
}

impl Engine {
    fn describe(&self) -> String {
        match self {
            Engine::Fp32(model) => model.describe(),
            Engine::Sim(model, scheme) => {
                format!(
                    "{} simulation of {}",
                    scheme.bits.label(),
                    model.describe().trim_start_matches("fp32 ")
                )
            }
            Engine::Fixed(model) => format!("integer reference {}", model.describe()),
        }
    }

    /// Run one content image. The image is the only thing the three modes share.
    fn run(&self, image: &image::RgbImage, device: &TrainDevice) -> Result<image::RgbImage> {
        match self {
            Engine::Fp32(model) => {
                let content = data::rgb_to_tensor::<InferBackend>(image, device);
                data::tensor_to_rgb(model.stylize(content))
            }
            Engine::Sim(model, scheme) => {
                let content = data::rgb_to_tensor::<InferBackend>(image, device);
                data::tensor_to_rgb(model.forward_quantised(content, scheme))
            }
            Engine::Fixed(model) => model.forward_image(image),
        }
    }
}

/// Run a delivered model directory over content images. This is the deployment path: it uses
/// only the student, no teacher.
///
/// The directory is self-describing, so the training config is not needed:
///
/// | file             | modes         | content                                   |
/// |------------------|---------------|-------------------------------------------|
/// | model.json       | fp32, sim     | architecture and its options              |
/// | model.mpk        | fp32, sim     | fp32 weights                              |
/// | calibration.json | sim           | activation scales used during calibration |
/// | manifest.json    | int           | the integer graph and per-layer constants |
/// | weights.bin      | int           | packed weight codes                       |
///
/// fp32 is the reference. sim differs from it only by quantisation. int differs from sim only
/// by the accumulation order, so int-versus-fp32 is the honest end-to-end number for the
/// accelerator, and sim-versus-int is the check that the export matches the training code.
fn infer(args: InferArgs, device_spec: &str) -> Result<()> {
    let dir = args.model.clone();
    let mpk = dir.join("model.mpk");
    let manifest = dir.join("manifest.json");

    let mode = match args.mode.as_str() {
        "auto" => {
            if manifest.is_file() {
                "int"
            } else if mpk.is_file() {
                "fp32"
            } else {
                bail!(
                    "{} holds neither manifest.json (an export) nor model.mpk (fp32 weights)",
                    dir.display()
                )
            }
        }
        known @ ("fp32" | "sim" | "int") => known,
        other => bail!("unknown --mode {other:?}: use auto, fp32, sim or int"),
    };

    let device = fpga_st_train::backend::select_device(device_spec);
    let engine = match mode {
        "int" => {
            if !manifest.is_file() {
                bail!("--mode int needs {}", manifest.display());
            }
            Engine::Fixed(fpga_st_train::intref::IntModel::load(&dir)?)
        }
        _ => {
            use burn::module::AutodiffModule;
            use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

            if !mpk.is_file() {
                bail!("--mode {mode} needs {}", mpk.display());
            }
            let card = ModelCard::load(&dir)?;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            let loaded = match card.arch.as_str() {
                "tiny" => {
                    let model = LoadedStudent::Tiny(
                        tiny_with_options::<TrainBackend>(&card.student(), &device)
                            .load_file(mpk.clone(), &recorder, &device)
                            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", mpk.display()))?
                            .valid(),
                    );
                    model
                }
                "tiny_dw" => {
                    let model = LoadedStudent::TinyDw(
                        tiny_dw_with_options::<TrainBackend>(&card.student(), &device)
                            .load_file(mpk.clone(), &recorder, &device)
                            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", mpk.display()))?
                            .valid(),
                    );
                    model
                }
                other => bail!("unknown student arch {other:?}; register it in infer (main.rs)"),
            };
            if mode == "sim" {
                let path = args
                    .calibration
                    .clone()
                    .unwrap_or_else(|| dir.join("calibration.json"));
                if !path.is_file() {
                    bail!(
                        "--mode sim needs the activation scales used for calibration; \
                         {} does not exist (pass --calibration)",
                        path.display()
                    );
                }
                let calibration = fpga_st_train::quant::Calibration::load(&path)?;
                Engine::Sim(loaded, calibration.scheme())
            } else {
                Engine::Fp32(loaded)
            }
        }
    };

    let contents = data::image_paths(&args.content)?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("cannot create {}", args.out.display()))?;
    log::info!("mode {mode}: {}", engine.describe());
    log::info!("{} content image(s)", contents.len());

    let mut total = 0.0f32;
    for path in &contents {
        let img = data::load_rgb(path)?;
        let img = if args.size > 0 {
            data::resize_long_edge(&img, args.size, FilterType::Lanczos3)
        } else {
            img
        };
        // The network downsamples by 4 and upsamples by 4, so a size that is not a multiple of
        // 4 comes back one or two pixels short. Pad to a multiple of 4, then crop back.
        let (w, h) = img.dimensions();
        let padded = data::pad_to_multiple(&img, 4);

        let step = Instant::now();
        let output = engine.run(&padded, &device)?;
        // The wgpu queue is asynchronous: the readback inside tensor_to_rgb is what blocks, so
        // the timer must end after it (see docs/backend.md).
        let millis = step.elapsed().as_secs_f32() * 1e3;
        total += millis;

        let output = if padded.dimensions() == (w, h) {
            output
        } else {
            image::imageops::crop_imm(&output, 0, 0, w, h).to_image()
        };
        let name = format!("{}.{}", stem(path), args.ext);
        data::save_rgb(&output, args.out.join(&name))?;
        println!(
            "{name}  {}x{}  {millis:.1} ms  {}",
            output.width(),
            output.height(),
            eval::describe(&output)
        );
    }
    println!("total {total:.0} ms for {} image(s)", contents.len());
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Command::Stylize(args) => stylize(args, &cli.device),
        Command::InitConfig(args) => {
            Config::write_default(&args.out)?;
            println!("wrote {}", args.out.display());
            Ok(())
        }
        Command::Train(args) => train(args, &cli.device),
        Command::Infer(args) => infer(args, &cli.device),
        Command::Calibrate(args) => calibrate(args, &cli.device),
        Command::ExportInt8(args) => export_int8(args, &cli.device),
        Command::Metrics(args) => metrics(args),
        Command::Parity(args) => parity(args, &cli.device),
        Command::Probe(args) => probe(args, &cli.device),
    }
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_owned()
}

fn stylize(args: StylizeArgs, device_spec: &str) -> Result<()> {
    let device = fpga_st_train::backend::select_device(device_spec);

    let started = Instant::now();
    let model: MicroAst<InferBackend> = weights::load(&args.teacher, &device)?;
    log::info!("teacher loaded in {:.2} s", started.elapsed().as_secs_f32());

    let contents = data::image_paths(&args.content)?;
    let styles = data::image_paths(&args.style)?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("cannot create {}", args.out.display()))?;
    log::info!(
        "{} content image(s) x {} style image(s)",
        contents.len(),
        styles.len()
    );

    let mut total = 0.0f32;
    for style_path in &styles {
        let style_img = data::resize_short_edge(
            &data::load_rgb(style_path)?,
            args.style_size,
            FilterType::Triangle,
        );
        let style = data::rgb_to_tensor::<InferBackend>(&style_img, &device);

        for content_path in &contents {
            let mut img = data::load_rgb(content_path)?;
            if args.content_size > 0 {
                img = data::resize_long_edge(&img, args.content_size, FilterType::Lanczos3);
            }
            if args.crop > 0 {
                img = data::center_crop(&img, args.crop, args.crop);
            }
            let content = data::rgb_to_tensor::<InferBackend>(&img, &device);

            let step = Instant::now();
            let output = model.stylize(content, style.clone(), args.alpha);
            // stylize() only records ops on the GPU command stream (wgpu queues async);
            // the into_data() call inside tensor_to_rgb() is what actually blocks on the
            // result, so timing has to end after readback or every measurement here reads
            // near-zero.
            let output = data::tensor_to_rgb(output)?;
            let millis = step.elapsed().as_secs_f32() * 1e3;
            total += millis;
            let name = format!("{}__{}.{}", stem(content_path), stem(style_path), args.ext);
            data::save_rgb(&output, args.out.join(&name))?;
            println!(
                "{name}  {}x{}  {millis:.0} ms  {}",
                output.width(),
                output.height(),
                eval::describe(&output)
            );
        }
    }
    println!("total {total:.0} ms");
    Ok(())
}

fn train(args: TrainArgs, device_spec: &str) -> Result<()> {
    let mut config = Config::load(&args.config)?;
    let base = args
        .config
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    config.rebase(&base);
    if let Some(iterations) = args.iterations {
        config.optim.iterations = iterations;
    }
    if let Some(dir) = args.output_dir {
        config.run.output_dir = dir;
    }
    if device_spec != "default" {
        config.run.device = device_spec.to_owned();
    }
    let arch = config.student.arch.clone();
    match arch.as_str() {
        "tiny" => fpga_st_train::train::run(config, build_tiny),
        "tiny_dw" => fpga_st_train::train::run(config, build_tiny_dw),
        other => bail!("unknown student arch {other:?}; register it in train (main.rs)"),
    }
}

fn metrics(args: MetricsArgs) -> Result<()> {
    let paths = data::image_paths(&args.images)?;
    let reference = match &args.content {
        Some(p) => Some(data::load_rgb(p)?),
        None => None,
    };
    for path in &paths {
        let img = data::load_rgb(path)?;
        let mut line = format!("{:<44} {}", stem(path), eval::describe(&img));
        if let Some(reference) = &reference {
            let resized =
                data::resize_exact(reference, img.width(), img.height(), FilterType::Triangle);
            line.push_str(&format!(
                "  psnr_vs_content {:.2} dB",
                eval::psnr(&img, &resized)?
            ));
        }
        println!("{line}");
    }

    if let Some(sheet_path) = &args.sheet {
        let mut panes: Vec<(String, image::RgbImage)> = Vec::new();
        // Labels are kept on the caller side; the sheet itself is pixels only.
        if let Some(reference) = &reference {
            let first = &paths[0];
            let target = data::load_rgb(first)?.dimensions();
            panes.push((
                "content".into(),
                data::resize_exact(reference, target.0, target.1, FilterType::Triangle),
            ));
        }
        for path in &paths {
            panes.push((stem(path), data::load_rgb(path)?));
        }
        let out = eval::sheet(&panes, 6);
        data::save_rgb(&out, sheet_path)?;
        println!("sheet -> {}", sheet_path.display());
    }
    Ok(())
}

/// Deterministic input, so two backends can be compared op by op.
fn probe_input(shape: [usize; 4]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect()
}

/// Report a few statistics plus a positional fingerprint of a tensor.
fn stats<B: Backend>(label: &str, t: Tensor<B, 4>, out: &mut String) {
    let shape = t.dims();
    let sum = t.clone().sum().into_scalar().to_f64();
    let mean = t.clone().mean().into_scalar().to_f64();
    let flat = t.into_data().to_vec::<f32>().unwrap_or_default();
    let mx = flat.iter().cloned().fold(f32::MIN, f32::max);
    let mn = flat.iter().cloned().fold(f32::MAX, f32::min);
    let sample: Vec<String> = [0usize, 1, flat.len() / 2, flat.len().saturating_sub(1)]
        .iter()
        .map(|&i| format!("{:.6}", flat.get(i).copied().unwrap_or(f32::NAN)))
        .collect();
    out.push_str(&format!(
        "{label:<34} shape {shape:?} sum {sum:.6} mean {mean:.6} min {mn:.6} max {mx:.6} sample [{}]\n",
        sample.join(", ")
    ));
}

fn probe(args: ProbeArgs, device_spec: &str) -> Result<()> {
    use burn::nn::PaddingConfig2d;
    use burn::nn::conv::Conv2dConfig;
    use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

    let device = fpga_st_train::backend::select_device(device_spec);
    let mut report = String::new();
    report.push_str(&format!(
        "backend {}\n",
        fpga_st_train::backend::BACKEND_NAME
    ));

    let x = Tensor::<InferBackend, 4>::from_data(
        TensorData::new(probe_input([1, 3, 32, 32]), [1, 3, 32, 32]),
        &device,
    );
    stats("00 input", x.clone(), &mut report);

    let padded = x.clone().pad([(1, 1), (1, 1)], PadMode::Reflect);
    stats("01 reflect pad 1", padded.clone(), &mut report);

    // Fixed, deterministic parameters: conv2d initialises randomly by default, so two
    // processes would otherwise build two different models and the comparison is void.
    let mk = |cin: usize, cout: usize, k: usize, s: usize, g: usize, bias: bool| {
        let mut conv = Conv2dConfig::new([cin, cout], [k, k])
            .with_stride([s, s])
            .with_groups(g)
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(bias)
            .init::<InferBackend>(&device);
        let w_shape = [cout, cin / g, k, k];
        let n: usize = w_shape.iter().product();
        let w: Vec<f32> = (0..n)
            .map(|i| ((i * 29 % 97) as f32 / 97.0 - 0.5) * 0.2)
            .collect();
        conv.weight = burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(w, w_shape),
            &device,
        ));
        if bias {
            let b: Vec<f32> = (0..cout)
                .map(|i| ((i * 13 % 41) as f32 / 41.0 - 0.5) * 0.1)
                .collect();
            conv.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(b, [cout]),
                &device,
            )));
        }
        conv
    };

    // The encoder chain, exactly as MicroAST runs it, so a divergence is attributable.
    let c1 = mk(3, 16, 9, 1, 1, true);
    let p4 = x.clone().pad([(4, 4), (4, 4)], PadMode::Reflect);
    let h1 = relu(c1.forward(p4));
    stats("02 relu(conv 3->16 k9 g1)", h1.clone(), &mut report);

    let c2 = mk(16, 32, 3, 2, 16, true);
    let h2 = relu(c2.forward(h1.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    stats("03 relu(conv 16->32 k3 s2 g16)", h2.clone(), &mut report);

    let c3 = mk(32, 32, 1, 1, 1, true);
    let h3 = relu(c3.forward(h2));
    stats("04 relu(conv 32->32 k1 g1)", h3.clone(), &mut report);

    let c4 = mk(32, 64, 3, 2, 32, true);
    let h4 = relu(c4.forward(h3.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    stats("05 relu(conv 32->64 k3 s2 g32)", h4.clone(), &mut report);

    let c5 = mk(64, 64, 1, 1, 1, true);
    let h5 = relu(c5.forward(h4));
    stats("06 relu(conv 64->64 k1 g1)", h5.clone(), &mut report);

    let c6 = mk(64, 64, 3, 1, 64, true);
    let h6 = c6.forward(h5.pad([(1, 1), (1, 1)], PadMode::Reflect));
    stats("07 conv 64->64 k3 g64", h6.clone(), &mut report);

    // Residual block (two 64-channel convs, the shape the decoder uses)
    let c7 = mk(64, 64, 3, 1, 1, true);
    let r1 = relu(c6.forward(h6.pad([(1, 1), (1, 1)], PadMode::Reflect)));
    let r2 = c7.forward(r1.clone().pad([(1, 1), (1, 1)], PadMode::Reflect));
    stats("08 residual out (64ch)", r2, &mut report);

    let y = Tensor::<InferBackend, 4>::from_data(
        TensorData::new(probe_input([1, 4, 8, 8]), [1, 4, 8, 8]),
        &device,
    );
    let up = burn::tensor::module::interpolate(
        y.clone(),
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    stats("08 bilinear 2x align=false", up, &mut report);

    let pool = y.clone().mean_dim(2).mean_dim(3);
    stats("09 mean_dim(2).mean_dim(3)", pool, &mut report);

    let (m, s) = fpga_st_train::teacher::microast::calc_mean_std(y.clone());
    stats("10 adain mean", m, &mut report);
    stats("11 adain std", s, &mut report);

    match &args.out {
        Some(path) => std::fs::write(path, &report)?,
        None => print!("{report}"),
    }
    Ok(())
}

fn tensor_from_raw<B: Backend>(path: &Path, device: &B::Device) -> Result<Tensor<B, 4>> {
    let (shape, values) = data::load_f32(path)?;
    if shape.len() != 4 {
        bail!("{}: expected a 4D shape, got {shape:?}", path.display());
    }
    let data = TensorData::new(values, [shape[0], shape[1], shape[2], shape[3]]);
    Ok(Tensor::from_data(data, device))
}

fn parity(args: ParityArgs, device_spec: &str) -> Result<()> {
    let device = fpga_st_train::backend::select_device(device_spec);
    let model: MicroAst<InferBackend> = weights::load(&args.teacher, &device)?;
    let content = tensor_from_raw::<InferBackend>(&args.content_raw, &device)?;
    let style = tensor_from_raw::<InferBackend>(&args.style_raw, &device)?;
    let output = model.stylize(content, style, args.alpha);
    let (shape, values) = data::tensor_to_f32(output)?;
    data::save_f32(&args.out_raw, &shape, &values)?;
    println!("wrote {} with shape {shape:?}", args.out_raw.display());
    Ok(())
}
