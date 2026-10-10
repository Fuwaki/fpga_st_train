//! Clap definitions for the `fpga_st_train` binary.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "fpga_st_train",
    version,
    about = "FPGA style transfer: MicroAST teacher inference and distillation training"
)]
pub struct Cli {
    /// Compute device: "default", "d0" (discrete GPU 0, the RX 580), "i0", "cpu".
    #[arg(long, global = true, default_value = "default")]
    pub device: String,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
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
pub struct ProbeArgs {
    /// Write the report here instead of stdout.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct StylizeArgs {
    #[arg(long)]
    pub teacher: PathBuf,
    #[arg(long)]
    pub content: PathBuf,
    #[arg(long)]
    pub style: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub content_size: u32,
    #[arg(long, default_value_t = 0)]
    pub crop: u32,
    #[arg(long, default_value_t = 256)]
    pub style_size: u32,
    #[arg(long, default_value_t = 1.0)]
    pub alpha: f32,
    #[arg(long, default_value = "png")]
    pub ext: String,
}

#[derive(Args, Debug)]
pub struct InitConfigArgs {
    #[arg(long, default_value = "config.json")]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub iterations: Option<usize>,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct InferArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub content: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "auto")]
    pub mode: String,
    #[arg(long)]
    pub calibration: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub size: u32,
    #[arg(long, default_value = "png")]
    pub ext: String,
}

#[derive(Args, Debug)]
pub struct CalibrateArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub content: PathBuf,
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 192)]
    pub crop: usize,
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 20)]
    pub batches: usize,
    #[arg(long, default_value_t = 1.0)]
    pub safety: f32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value = "w8a8")]
    pub bits: String,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct ExportInt8Args {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    #[arg(long)]
    pub calibration: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct MetricsArgs {
    #[arg(long)]
    pub images: PathBuf,
    #[arg(long)]
    pub content: Option<PathBuf>,
    #[arg(long)]
    pub sheet: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct ParityArgs {
    #[arg(long)]
    pub teacher: PathBuf,
    #[arg(long)]
    pub content_raw: PathBuf,
    #[arg(long)]
    pub style_raw: PathBuf,
    #[arg(long)]
    pub out_raw: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    pub alpha: f32,
}
