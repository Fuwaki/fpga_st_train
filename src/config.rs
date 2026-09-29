//! Run configuration. Loaded from a JSON file by the train and eval subcommands.
//!
//! Generate a fully commented starting point with:
//!
//!     fpga_st_train config --out config.json
//!
//! Every field has a default, so a partial file is valid.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Top level configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub run: RunConfig,
    pub data: DataConfig,
    pub teacher: TeacherConfig,
    pub loss: LossConfig,
    pub optim: OptimConfig,
    pub student: StudentConfig,
    /// int8 quantisation-aware training. Disabled unless requested.
    #[serde(default)]
    pub quant: QuantConfig,
}

/// int8 quantisation-aware fine-tuning.
///
/// When enabled, training runs the fake-quantised forward pass (see `crate::quant`) instead of
/// the fp32 one, so the weights adapt to the int8 grid. The activation scales come from the
/// `calibrate` subcommand and are held fixed during fine-tuning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantConfig {
    /// Run the fake-quantised forward pass during training.
    #[serde(default)]
    pub enabled: bool,
    /// Calibration file written by the `calibrate` subcommand.
    #[serde(default)]
    pub calibration: Option<PathBuf>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        let config: Self = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse config {}", path.display()))?;
        Ok(config)
    }

    /// Write the default configuration, as a starting point to edit.
    pub fn write_default(path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = serde_json::to_string_pretty(&Self::default())?;
        std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))
    }

    /// Resolve a relative path against the directory holding the config file.
    pub fn rebase(&mut self, base: &Path) {
        let fix = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        };
        fix(&mut self.run.output_dir);
        fix(&mut self.data.content_dir);
        if let Some(p) = self.run.sample_content.as_mut() {
            fix(p);
        }
        if let Some(p) = self.data.pseudo_gt_dir.as_mut() {
            fix(p);
        }
        fix(&mut self.teacher.weights_dir);
        fix(&mut self.teacher.style);
        if let Some(p) = self.loss.vgg_weights.as_mut() {
            fix(p);
        }
        if let Some(p) = self.quant.calibration.as_mut() {
            fix(p);
        }
        if let Some(p) = self.run.init_from.as_mut() {
            fix(p);
        }
    }

    /// Fail early on obviously wrong configurations, before any model is built.
    pub fn validate(&self) -> Result<()> {
        if !self.data.content_dir.is_dir() {
            bail!(
                "data.content_dir is not a directory: {}",
                self.data.content_dir.display()
            );
        }
        if self.data.crop == 0 || !self.data.crop.is_multiple_of(4) {
            bail!(
                "data.crop must be a positive multiple of 4, got {}",
                self.data.crop
            );
        }
        if self.data.batch_size == 0 {
            bail!("data.batch_size must be positive");
        }
        if self.optim.iterations == 0 {
            bail!("optim.iterations must be positive");
        }
        let student_is_identity =
            self.loss.pixel == 0.0 && self.loss.perceptual == 0.0 && self.loss.style == 0.0;
        if student_is_identity {
            bail!("every loss term is zero -- the model would learn nothing");
        }
        Ok(())
    }
}

/// Where things go and how noisy to be.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    pub name: String,
    pub output_dir: PathBuf,
    pub seed: u64,
    /// Compute device: "default", "d0" (the RX 580), "i0", "cpu".
    /// The --device command line flag overrides this.
    pub device: String,
    pub log_every: usize,
    pub checkpoint_every: usize,
    pub sample_every: usize,
    /// Image or folder used to render progress samples. Falls back to the first
    /// training content image when unset.
    pub sample_content: Option<PathBuf>,
    /// Long edge the sample content is resized to before stylisation.
    pub sample_size: usize,
    /// Checkpoint to warm start from, e.g. the fp32 model before int8 fine-tuning.
    #[serde(default)]
    pub init_from: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            name: "run".into(),
            output_dir: PathBuf::from("runs/train"),
            seed: 0,
            device: "default".into(),
            log_every: 20,
            checkpoint_every: 500,
            sample_every: 250,
            sample_content: None,
            sample_size: 384,
            init_from: None,
        }
    }
}

/// Training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    /// Folder of content images.
    pub content_dir: PathBuf,
    /// Folder of teacher outputs. Files are matched to content by file stem.
    /// When absent the teacher is run online for every batch.
    pub pseudo_gt_dir: Option<PathBuf>,
    /// Random crop size used for training. Must be a multiple of 4.
    pub crop: usize,
    pub batch_size: usize,
    /// Cap on the number of content images used (None = all).
    pub limit: Option<usize>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            content_dir: PathBuf::from("data/content"),
            pseudo_gt_dir: None,
            crop: 256,
            batch_size: 4,
            limit: None,
        }
    }
}

/// The MicroAST teacher.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TeacherConfig {
    /// Directory holding the four *_iter_160000.pth.tar checkpoints.
    pub weights_dir: PathBuf,
    /// Style image(s) the teacher stylises with.
    pub style: PathBuf,
    /// Style images are resized so their short edge is this.
    pub style_size: usize,
    /// Stylisation strength. The released checkpoints only behave at 1.0.
    pub alpha: f32,
    /// Set false when training purely against pre-generated pseudo ground truth.
    pub enabled: bool,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            weights_dir: PathBuf::from("models"),
            style: PathBuf::from("style.jpg"),
            style_size: 256,
            alpha: 1.0,
            enabled: false,
        }
    }
}

/// Loss term weights.
///
/// pixel and tv need nothing but the images. perceptual and style need VGG16 weights
/// (see the README); they are skipped when vgg_weights is None.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LossConfig {
    /// L1 between the student output and the target.
    pub pixel: f32,
    /// L1 between the finite-difference gradients of output and target.
    pub gradient: f32,
    /// Feature-level distillation: per-layer L1 between the student's and the teacher's
    /// decoder feature maps, normalised by each layer's mean magnitude. 0 disables it.
    pub feature: f32,
    /// Total-variation regulariser on the student output.
    pub tv: f32,
    /// VGG relu3_3 content distance to the target.
    pub perceptual: f32,
    /// VGG Gram style distance to the style images.
    pub style: f32,
    /// Path to a torchvision VGG16 state dict (vgg16-397923af.pth).
    pub vgg_weights: Option<PathBuf>,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            pixel: 1.0,
            gradient: 0.0,
            feature: 0.0,
            tv: 1e-4,
            perceptual: 0.0,
            style: 0.0,
            vgg_weights: None,
        }
    }
}

/// Optimiser and schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimConfig {
    pub learning_rate: f64,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    /// Number of optimisation steps.
    pub iterations: usize,
    pub schedule: Schedule,
    /// Optional gradient-norm clipping threshold.
    pub grad_clip: Option<f32>,
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-5,
            iterations: 4000,
            schedule: Schedule::default(),
            grad_clip: None,
        }
    }
}

/// Learning-rate schedule.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schedule {
    /// Hold the learning rate constant.
    #[default]
    Constant,
    /// Multiply by gamma every step.
    Exponential { gamma: f64 },
    /// Decay linearly to final_lr over num_iters.
    Linear { final_lr: f64, num_iters: usize },
}

/// Student model selection and free-form options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StudentConfig {
    /// Which architecture to build. Consumed by build_student in main.rs.
    pub arch: String,
    /// Architecture-specific options, so new models need no config changes.
    pub options: serde_json::Value,
}

impl Default for StudentConfig {
    fn default() -> Self {
        Self {
            arch: "tiny".into(),
            options: serde_json::json!({ "base": 32, "res_blocks": 4 }),
        }
    }
}

impl StudentConfig {
    /// Read an integer option, falling back to a default.
    pub fn usize_option(&self, key: &str, default: usize) -> usize {
        self.options
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    /// Read a float option, falling back to a default.
    pub fn f32_option(&self, key: &str, default: f32) -> f32 {
        self.options
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(default)
    }

    /// Read a string option, falling back to a default.
    pub fn str_option<'s>(&'s self, key: &str, default: &'s str) -> &'s str {
        self.options
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(default)
    }
}

/// A self-describing model directory.
///
/// Every training run writes one of these next to its checkpoints, and every export writes one
/// next to the quantised artefacts, so the inference tool never needs the training config that
/// produced the model. That is what makes a delivered directory runnable on its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Which architecture to build. Consumed by build_student in main.rs.
    pub arch: String,
    /// Architecture-specific options, exactly as they appear in the training config.
    #[serde(default)]
    pub options: serde_json::Value,
    /// Free-form metadata (style, run name, bit widths, provenance) for humans and for the
    /// deliverable listing. Never read by the model builder.
    #[serde(default)]
    pub meta: serde_json::Map<String, serde_json::Value>,
}

impl ModelCard {
    /// File name inside a model directory.
    pub const FILE: &'static str = "model.json";

    /// Describe the student a training config asks for.
    pub fn of(student: &StudentConfig) -> Self {
        Self {
            arch: student.arch.clone(),
            options: student.options.clone(),
            meta: serde_json::Map::new(),
        }
    }

    /// Attach one metadata entry; builder style so a card reads as one expression.
    pub fn with_meta(mut self, key: &str, value: serde_json::Value) -> Self {
        self.meta.insert(key.to_owned(), value);
        self
    }

    /// Rebuild the student config this card describes.
    pub fn student(&self) -> StudentConfig {
        StudentConfig {
            arch: self.arch.clone(),
            options: self.options.clone(),
        }
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join(Self::FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
    }

    pub fn save(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let path = dir.join(Self::FILE);
        std::fs::write(&path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("cannot write {}", path.display()))
    }
}
