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
    /// Compute device. CUDA: "d0" is nvidia-smi index 0 (this machine's V100). Vulkan/wgpu: "d0"/"d1" are discrete adapters, "i0" integrated, "cpu" lavapipe.
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
    /// Copy a guided checkpoint into a narrower student, keeping the strongest channels.
    Prune(PruneArgs),
    /// Fold a 3-plane guided checkpoint onto luma plus the 135° chroma axis.
    FoldYc(FoldYcArgs),
    /// Duplicate the trunk norm and H/2 FiLM into one packet per style.
    PackStyles(PackStylesArgs),
    /// Extract one style as a single-style checkpoint for the integer export.
    SplitStyle(SplitStyleArgs),
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
    /// Directory with the four MicroAST `*_iter_160000.pth.tar` files. Overrides
    /// `teacher.weights_dir` from the config (env `TEACHER_WEIGHTS_DIR` also works).
    #[arg(long)]
    pub teacher_weights: Option<PathBuf>,
    /// Another student trained on the same teacher forward. Repeat for a cohort of 2 or 3.
    /// Data, loss, teacher, seed, and step budget must match `--config`.
    #[arg(long = "with")]
    pub with: Vec<PathBuf>,
}

#[derive(Args, Debug)]
pub struct PruneArgs {
    /// Directory with `model.json` and `model.mpk` from a trained guided student.
    #[arg(long)]
    pub from: PathBuf,
    /// Checkpoint inside `--from`. Defaults to `model.mpk`.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// Where to write the narrow `model.mpk`, `model.json`, and `prune.json`.
    #[arg(long)]
    pub out: PathBuf,
    /// Images used to rank channels. Defaults to `content_*.png` in `--from`.
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// Reflect-pad content up to this width before scoring. 640 is the deployment frame.
    #[arg(long, default_value_t = 640)]
    pub width: usize,
    /// Reflect-pad content up to this height before scoring.
    #[arg(long, default_value_t = 480)]
    pub height: usize,
    #[arg(long, default_value_t = 16)]
    pub base: usize,
    #[arg(long, default_value_t = 80)]
    pub trunk: usize,
    #[arg(long, default_value_t = 24)]
    pub up_h4: usize,
    #[arg(long, default_value_t = 40)]
    pub up_h2: usize,
    #[arg(long, default_value_t = 10)]
    pub h2_guide: usize,
    #[arg(long, default_value_t = 8)]
    pub guide: usize,
    /// Residual blocks to keep. `0` drops the block and copies every other slice.
    #[arg(long, default_value_t = 1)]
    pub res_blocks: usize,
}

#[derive(Args, Debug)]
pub struct FoldYcArgs {
    /// Directory with `model.json` and `model.mpk` from a 3-plane guided student.
    #[arg(long)]
    pub from: PathBuf,
    /// Checkpoint inside `--from`. Defaults to `model.mpk`.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// Where to write the 2-plane `model.mpk` and `model.json`.
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct PackStylesArgs {
    /// Directory with `model.json` and `model.mpk` from a guided student.
    #[arg(long)]
    pub from: PathBuf,
    /// Checkpoint inside `--from`. Defaults to `model.mpk`.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// Where to write the multi-style `model.mpk` and `model.json`.
    #[arg(long)]
    pub out: PathBuf,
    /// Style slots. Existing slots keep their packets; new ones copy style 0.
    #[arg(long, default_value_t = 2)]
    pub styles: usize,
    /// Add the per-style guide affine (`packet_guide`), initialised to the identity.
    #[arg(long)]
    pub packet_guide: bool,
}

#[derive(Args, Debug)]
pub struct SplitStyleArgs {
    /// Directory with a multi-style guided `model.json` and `model.mpk`.
    #[arg(long)]
    pub from: PathBuf,
    /// Checkpoint inside `--from`. Defaults to `model.mpk`.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// Style to extract.
    #[arg(long)]
    pub style: usize,
    /// Where to write the single-style `model.mpk` and `model.json`.
    #[arg(long)]
    pub out: PathBuf,
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
    /// Style packet. `0` is the trunk norm and H/2 FiLM stored in the original fields.
    #[arg(long, default_value_t = 0)]
    pub style: usize,
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
    #[arg(long)]
    pub frame_width: Option<usize>,
    #[arg(long)]
    pub frame_height: Option<usize>,
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
