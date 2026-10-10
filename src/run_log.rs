//! Append-only JSONL training log (`metrics.jsonl` in the run directory).
//!
//! Each line is one event. All step-aligned records share the same `step` field. Metric
//! fields are sparse: only present when that quantity was measured at that step.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::Config;

pub const METRICS_FILE: &str = "metrics.jsonl";
pub const CONFIG_SNAPSHOT: &str = "config.json";

/// Optional provenance passed from the CLI into [crate::train::run].
#[derive(Debug, Clone, Default)]
pub struct RunProvenance {
    /// Path to the JSON file the user passed to `train --config`.
    pub config_path: Option<PathBuf>,
}

/// Handles `metrics.jsonl` and the frozen config snapshot for one run.
pub struct RunLog {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl RunLog {
    pub fn open(output_dir: &Path, config: &Config, provenance: &RunProvenance) -> Result<Self> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("cannot create {}", output_dir.display()))?;

        let config_path = output_dir.join(CONFIG_SNAPSHOT);
        let text = serde_json::to_string_pretty(config)?;
        std::fs::write(&config_path, text)
            .with_context(|| format!("cannot write {}", config_path.display()))?;

        let path = output_dir.join(METRICS_FILE);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("cannot open {}", path.display()))?;

        let mut log = Self {
            writer: BufWriter::new(file),
            path,
        };

        let mut start = Map::new();
        start.insert("kind".into(), Value::String("run_start".into()));
        start.insert("step".into(), Value::Number(0.into()));
        start.insert("run".into(), Value::String(config.run.name.clone()));
        if let Some(p) = &provenance.config_path {
            start.insert(
                "config_source".into(),
                Value::String(p.display().to_string()),
            );
        }
        start.insert(
            "hyperparams".into(),
            serde_json::to_value(config).context("serialise config for run_start")?,
        );
        log.write_map(&start)?;

        Ok(log)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn train_step(&mut self, step: usize, metrics: TrainStepMetrics) -> Result<()> {
        let mut m = Map::new();
        m.insert("kind".into(), Value::String("train".into()));
        m.insert("step".into(), json_usize(step));
        m.insert("split".into(), Value::String("train".into()));
        metrics.insert_into(&mut m);
        self.write_map(&m)
    }

    pub fn val_step(&mut self, step: usize, metrics: ValStepMetrics) -> Result<()> {
        let mut m = Map::new();
        m.insert("kind".into(), Value::String("val".into()));
        m.insert("step".into(), json_usize(step));
        m.insert("split".into(), Value::String("val".into()));
        metrics.insert_into(&mut m);
        self.write_map(&m)
    }

    pub fn sample_eval(&mut self, step: usize, name: &str, metrics: SampleMetrics) -> Result<()> {
        let mut m = Map::new();
        m.insert("kind".into(), Value::String("sample".into()));
        m.insert("step".into(), json_usize(step));
        m.insert("sample".into(), Value::String(name.to_owned()));
        metrics.insert_into(&mut m);
        self.write_map(&m)
    }

    pub fn run_end(&mut self, step: usize, wall_s: f64, final_train: Option<TrainStepMetrics>) -> Result<()> {
        let mut m = Map::new();
        m.insert("kind".into(), Value::String("run_end".into()));
        m.insert("step".into(), json_usize(step));
        m.insert("wall_s".into(), json_f64(wall_s));
        if let Some(t) = final_train {
            t.insert_into(&mut m);
        }
        self.write_map(&m)?;
        self.writer.flush().context("flush metrics.jsonl")?;
        Ok(())
    }

    fn write_map(&mut self, map: &Map<String, Value>) -> Result<()> {
        let line = serde_json::to_string(map).context("serialise metrics line")?;
        writeln!(self.writer, "{line}").context("append metrics.jsonl")?;
        Ok(())
    }
}

fn json_usize(v: usize) -> Value {
    Value::Number(serde_json::Number::from(v))
}

fn json_f64(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn insert_opt_f64(map: &mut Map<String, Value>, key: &str, v: Option<f64>) {
    if let Some(v) = v {
        map.insert(key.into(), json_f64(v));
    }
}

/// Training loss terms logged on `run.log_every` (sparse fields).
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrainStepMetrics {
    pub loss: f64,
    pub lr: f64,
    pub steps_per_s: Option<f64>,
    pub l1: Option<f64>,
    pub gradient: Option<f64>,
    pub stroke: Option<f64>,
    pub feature: Option<f64>,
    pub tv: Option<f64>,
}

impl TrainStepMetrics {
    fn insert_into(&self, map: &mut Map<String, Value>) {
        map.insert("loss".into(), json_f64(self.loss));
        map.insert("lr".into(), json_f64(self.lr));
        insert_opt_f64(map, "steps_per_s", self.steps_per_s);
        insert_opt_f64(map, "l1", self.l1);
        insert_opt_f64(map, "gradient", self.gradient);
        insert_opt_f64(map, "stroke", self.stroke);
        insert_opt_f64(map, "feature", self.feature);
        insert_opt_f64(map, "tv", self.tv);
    }
}

/// Fixed-batch validation loss (same terms as training, `model.valid()`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct ValStepMetrics {
    pub loss: f64,
    pub l1: Option<f64>,
    pub gradient: Option<f64>,
    pub stroke: Option<f64>,
    pub feature: Option<f64>,
    pub tv: Option<f64>,
}

impl ValStepMetrics {
    fn insert_into(&self, map: &mut Map<String, Value>) {
        map.insert("val_loss".into(), json_f64(self.loss));
        insert_opt_f64(map, "val_l1", self.l1);
        insert_opt_f64(map, "val_gradient", self.gradient);
        insert_opt_f64(map, "val_stroke", self.stroke);
        insert_opt_f64(map, "val_feature", self.feature);
        insert_opt_f64(map, "val_tv", self.tv);
    }
}

/// Image-space metrics from progress samples (`run.sample_every`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct SampleMetrics {
    pub lapvar: f64,
    pub checkerboard: f64,
    pub saturation: f64,
    pub psnr_vs_teacher: Option<f64>,
    pub lapvar_teacher: Option<f64>,
}

impl SampleMetrics {
    fn insert_into(&self, map: &mut Map<String, Value>) {
        map.insert("lapvar".into(), json_f64(self.lapvar));
        map.insert("checkerboard".into(), json_f64(self.checkerboard));
        map.insert("saturation".into(), json_f64(self.saturation));
        insert_opt_f64(map, "psnr_vs_teacher", self.psnr_vs_teacher);
        insert_opt_f64(map, "lapvar_teacher", self.lapvar_teacher);
    }
}
