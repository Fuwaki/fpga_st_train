//! Pluggable student architectures.
//!
//! Add a module under `registry/`, implement the four hooks (`train`, `calibrate`,
//! `export_int8`, `load_fp32`), and append one row to [TABLE].

mod common;
mod tiny;
mod tiny_dw;

use std::path::Path;

use anyhow::Result;

use crate::backend::TrainDevice;
use crate::config::{Config, ModelCard};
use crate::infer_student::StudentHandle;
use crate::run_log::RunProvenance;
use crate::quant::CalibrationJob;

type TrainFn = fn(Config, RunProvenance) -> Result<()>;
type CalibrateFn = fn(
    &CalibrationJob,
    &ModelCard,
    &TrainDevice,
    &Path,
    crate::quant::Bits,
) -> Result<crate::quant::Calibration>;
type ExportFn = fn(
    &ModelCard,
    &Path,
    &crate::quant::Calibration,
    &Path,
    &TrainDevice,
) -> Result<crate::quant::QuantManifest>;
type LoadFn = fn(&ModelCard, &Path, &TrainDevice) -> Result<StudentHandle>;

struct ArchEntry {
    name: &'static str,
    train: TrainFn,
    calibrate: CalibrateFn,
    export_int8: ExportFn,
    load_fp32: LoadFn,
}

const TABLE: &[ArchEntry] = &[
    ArchEntry {
        name: tiny::ID,
        train: tiny::train,
        calibrate: tiny::calibrate,
        export_int8: tiny::export_int8,
        load_fp32: tiny::load_fp32,
    },
    ArchEntry {
        name: tiny_dw::ID,
        train: tiny_dw::train,
        calibrate: tiny_dw::calibrate,
        export_int8: tiny_dw::export_int8,
        load_fp32: tiny_dw::load_fp32,
    },
];

fn lookup(name: &str) -> Result<&'static ArchEntry> {
    TABLE
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| {
            let known = TABLE
                .iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::anyhow!("unknown student arch {name:?}; known: {known}")
        })
}

/// All registered `student.arch` strings.
pub fn arch_names() -> Vec<&'static str> {
    TABLE.iter().map(|e| e.name).collect()
}

pub fn train(config: Config, provenance: RunProvenance) -> Result<()> {
    (lookup(&config.student.arch)?.train)(config, provenance)
}

pub fn calibrate_student(
    job: &CalibrationJob,
    card: &ModelCard,
    device: &TrainDevice,
    checkpoint: &Path,
    bits: crate::quant::Bits,
) -> Result<crate::quant::Calibration> {
    (lookup(&card.arch)?.calibrate)(job, card, device, checkpoint, bits)
}

pub fn export_int8_from_card(
    card: &ModelCard,
    checkpoint: &Path,
    calibration: &crate::quant::Calibration,
    out: &Path,
    device: &TrainDevice,
) -> Result<crate::quant::QuantManifest> {
    (lookup(&card.arch)?.export_int8)(card, checkpoint, calibration, out, device)
}

pub fn load_student_fp32(
    card: &ModelCard,
    checkpoint: &Path,
    device: &TrainDevice,
) -> Result<StudentHandle> {
    (lookup(&card.arch)?.load_fp32)(card, checkpoint, device)
}

/// Re-export builders for tests and one-off scripts.
pub use tiny::build as build_tiny;
pub use tiny_dw::build as build_tiny_dw;
