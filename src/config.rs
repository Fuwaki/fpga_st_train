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
        if let Some(p) = self.teacher.model.as_mut() {
            fix(p);
        }
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
        if self.data.hf.is_none() {
            let path = &self.data.content_dir;
            let ok = if self.data.overfit_long_edge.is_some() {
                path.is_file() || path.is_dir()
            } else {
                path.is_dir()
            };
            if !ok {
                bail!(
                    "data.content_dir is not usable: {} (or set data.hf for Hugging Face)",
                    path.display()
                );
            }
        }
        if let Some(edge) = self.data.overfit_long_edge {
            if edge == 0 || !edge.is_multiple_of(8) {
                bail!("data.overfit_long_edge must be a positive multiple of 8, got {edge}");
            }
            if self.data.hf.is_some() {
                bail!("data.overfit_long_edge cannot be combined with data.hf");
            }
        } else if self.data.crop == 0 || !self.data.crop.is_multiple_of(4) {
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
        let no_signal = self.loss.pixel == 0.0
            && self.loss.gradient == 0.0
            && self.loss.flat == 0.0
            && self.loss.edge == 0.0
            && self.loss.contrast == 0.0
            && self.loss.stroke == 0.0
            && self.loss.line == 0.0
            && self.loss.texture == 0.0
            && self.loss.feature == 0.0
            && self.loss.tv == 0.0
            && self.loss.perceptual == 0.0
            && self.loss.style == 0.0;
        if no_signal {
            bail!("every loss term is zero -- the model would learn nothing");
        }
        if self.teacher.enabled {
            match self.teacher.kind()? {
                TeacherKind::MicroAst => {
                    let ckpt = self
                        .teacher
                        .weights_dir
                        .join("content_encoder_iter_160000.pth.tar");
                    if !ckpt.is_file() {
                        let dir = display_path(&self.teacher.weights_dir);
                        bail!(
                            "MicroAST teacher weights not found at {dir}\n  \
                             expected file: content_encoder_iter_160000.pth.tar (and three siblings)\n  \
                             fix: put checkpoints there, or pass --teacher-weights / set TEACHER_WEIGHTS_DIR"
                        );
                    }
                }
                TeacherKind::RainPrincess | TeacherKind::Whitebox => {
                    let Some(model) = &self.teacher.model else {
                        bail!(
                            "teacher.kind = {:?} requires teacher.model",
                            self.teacher.kind
                        );
                    };
                    if !model.is_file() {
                        bail!("teacher model not found: {}", display_path(model));
                    }
                }
            }
        }
        Ok(())
    }
}

fn default_sample_limit() -> usize {
    3
}

fn display_path(path: &Path) -> String {
    path.canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

/// Where things go and how noisy to be.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    pub name: String,
    pub output_dir: PathBuf,
    pub seed: u64,
    /// Compute device. On the CUDA build, `d0` is nvidia-smi GPU 0.
    /// The --device command line flag overrides this when it is not `default`.
    pub device: String,
    pub log_every: usize,
    /// Fixed-batch validation loss on `model.valid()` (0 = disabled).
    pub val_every: usize,
    /// Content images in the fixed val batch (same random crops every time).
    pub val_batch_size: usize,
    pub checkpoint_every: usize,
    pub sample_every: usize,
    /// How many images from `sample_content` to render. `0` means every image
    /// in the folder. Existing configs omit this and keep the old limit of 3.
    #[serde(default = "default_sample_limit")]
    pub sample_limit: usize,
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
            val_every: 0,
            val_batch_size: 4,
            checkpoint_every: 500,
            sample_every: 250,
            sample_limit: default_sample_limit(),
            sample_content: None,
            sample_size: 384,
            init_from: None,
        }
    }
}

/// Hugging Face training images (`hf-hub` + parquet shards, cached under `~/.cache/huggingface`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HfDataConfig {
    /// e.g. `detection-datasets/coco`
    pub repo: String,
    /// e.g. `train`
    pub split: String,
    /// Git revision on the dataset repo.
    pub revision: String,
    /// Cap parquet shards (sorted `data/{split}-*.parquet`). `None` = all shards.
    pub max_shards: Option<usize>,
    /// Rows in the full split. Skips the datasets-server `/size` call when set,
    /// so a later trial can keep the same sampling pool if that host is down.
    pub num_rows: Option<usize>,
}

impl Default for HfDataConfig {
    fn default() -> Self {
        Self {
            repo: "detection-datasets/coco".into(),
            split: "train".into(),
            revision: "main".into(),
            max_shards: None,
            num_rows: None,
        }
    }
}

/// Training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    /// Folder of content images (ignored when `hf` is set).
    pub content_dir: PathBuf,
    /// Load content from Hugging Face instead of `content_dir`.
    pub hf: Option<HfDataConfig>,
    /// Folder of teacher outputs. Files are matched to content by file stem.
    /// When absent the teacher is run online for every batch.
    pub pseudo_gt_dir: Option<PathBuf>,
    /// Random crop size used for training. Must be a multiple of 4.
    /// Ignored when `overfit_long_edge` is set.
    pub crop: usize,
    /// Fit one content image, resized so its long edge is this value and both
    /// edges are multiples of 8. No random crop. `content_dir` is that image
    /// (or a folder containing only it). None keeps the normal crop pipeline.
    pub overfit_long_edge: Option<usize>,
    pub batch_size: usize,
    /// Cap on the number of content images used (None = all).
    pub limit: Option<usize>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            content_dir: PathBuf::from("data/content"),
            hf: None,
            pseudo_gt_dir: None,
            crop: 256,
            overfit_long_edge: None,
            batch_size: 4,
            limit: None,
        }
    }
}

/// Which online teacher produces the distillation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeacherKind {
    /// Two-input MicroAST. Needs `weights_dir` and `style`.
    MicroAst,
    /// Johnson fast neural style, rain-princess ONNX. Needs `model`.
    RainPrincess,
    /// White-box cartoonization ONNX. Needs `model`.
    Whitebox,
}

impl TeacherKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "microast" => Ok(Self::MicroAst),
            "rain_princess" => Ok(Self::RainPrincess),
            "whitebox" => Ok(Self::Whitebox),
            other => {
                bail!("unknown teacher.kind {other:?}; expected microast, rain_princess, whitebox")
            }
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::MicroAst => "microast",
            Self::RainPrincess => "rain_princess",
            Self::Whitebox => "whitebox",
        }
    }
}

/// Online distillation teacher.
///
/// `kind` defaults to `microast`, so existing configs keep loading the four
/// MicroAST checkpoints. `rain_princess` and `whitebox` read an ONNX file from `model`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TeacherConfig {
    /// `microast`, `rain_princess`, or `whitebox`.
    pub kind: String,
    /// ONNX file. Required when `kind` is not `microast`.
    pub model: Option<PathBuf>,
    /// Directory holding the four *_iter_160000.pth.tar checkpoints.
    pub weights_dir: PathBuf,
    /// Style image the MicroAST teacher stylises with.
    pub style: PathBuf,
    /// Style images are resized so their short edge is this.
    pub style_size: usize,
    /// Stylisation strength. The released MicroAST checkpoints only behave at 1.0.
    pub alpha: f32,
    /// Set false when training purely against pre-generated pseudo ground truth.
    pub enabled: bool,
}

impl TeacherConfig {
    pub fn kind(&self) -> Result<TeacherKind> {
        TeacherKind::parse(&self.kind)
    }

    /// Label stored on the model card. MicroAST records the style image; ONNX records the file.
    pub fn style_label(&self) -> String {
        match self.kind().unwrap_or(TeacherKind::MicroAst) {
            TeacherKind::MicroAst => self.style.display().to_string(),
            TeacherKind::RainPrincess | TeacherKind::Whitebox => self
                .model
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| self.kind.clone()),
        }
    }
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            kind: TeacherKind::MicroAst.name().to_owned(),
            model: None,
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
    /// Extra student gradient where the teacher is flat. 0 disables it.
    #[serde(default)]
    pub flat: f32,
    /// Missing student gradient on teacher edges (3×3 tolerance). 0 disables it.
    #[serde(default)]
    pub edge: f32,
    /// 16×16 window mean and variance. 0 disables it.
    #[serde(default)]
    pub contrast: f32,
    /// Band-pass L1 at 3–9 px and 9–25 px. Asks for brush strokes without the
    /// 1 px conditional-mean trap of `gradient`. 0 disables it.
    pub stroke: f32,
    /// Signed gradient L1 weighted by the teacher's gradient magnitude: the
    /// teacher's lines, in the teacher's direction. 0 disables it.
    #[serde(default)]
    pub line: f32,
    /// Local band amplitude (4–8 px and 8–16 px). Position inside the window is free.
    /// 0 disables it.
    pub texture: f32,
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
            flat: 0.0,
            edge: 0.0,
            contrast: 0.0,
            stroke: 0.0,
            line: 0.0,
            texture: 0.0,
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
    /// Which architecture to build. Dispatched in [crate::registry].
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
    /// Which architecture to build. Dispatched in [crate::registry].
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
