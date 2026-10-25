//! `arch = "tiny_dw_guided"` registration.
//!
//! Fp32 trial only. The graph is not in the integer export yet: calibration and
//! `export-int8` fail on purpose so a checkpoint cannot be packaged by accident.

use std::path::Path;

use anyhow::{Result, bail};
use burn::prelude::*;

use crate::TrainBackend;
use crate::backend::TrainDevice;
use crate::config::{Config, ModelCard, StudentConfig};
use crate::infer_student::StudentHandle;
use crate::quant::CalibrationJob;
use crate::run_log::RunProvenance;
use crate::student::guided::{GuideSpec, GuidedStudent, print_cost};
use crate::train::run as run_training;

pub const ID: &str = "tiny_dw_guided";

pub fn build<B: Backend>(config: &StudentConfig, device: &B::Device) -> GuidedStudent<B> {
    let spec = GuideSpec::from_config(config).unwrap_or_else(|err| panic!("{err}"));
    GuidedStudent::new(spec, device)
}

fn build_train(
    config: &StudentConfig,
    device: &TrainDevice,
) -> Result<GuidedStudent<TrainBackend>> {
    let spec = GuideSpec::from_config(config).map_err(anyhow::Error::msg)?;
    print_cost(spec);
    Ok(GuidedStudent::new(spec, device))
}

pub fn train(config: Config, provenance: RunProvenance) -> Result<()> {
    run_training(config, provenance, build_train)
}

pub fn train_cohort(configs: Vec<Config>, provenances: Vec<RunProvenance>) -> Result<()> {
    crate::train::run_all(configs, provenances, build_train)
}

pub fn calibrate(
    _job: &CalibrationJob,
    _card: &ModelCard,
    _device: &TrainDevice,
    _checkpoint: &Path,
    _bits: crate::quant::Bits,
) -> Result<crate::quant::Calibration> {
    bail!("tiny_dw_guided is an fp32 architecture trial; calibration is not implemented")
}

pub fn export_int8(
    _card: &ModelCard,
    _checkpoint: &Path,
    _calibration: &crate::quant::Calibration,
    _out: &Path,
    _device: &TrainDevice,
) -> Result<crate::quant::QuantManifest> {
    bail!("tiny_dw_guided is an fp32 architecture trial; integer export is not implemented")
}

pub fn load_fp32(
    card: &ModelCard,
    checkpoint: &Path,
    device: &TrainDevice,
) -> Result<StudentHandle> {
    let student = card.student();
    super::common::load_fp32::<GuidedStudent<TrainBackend>, _>(&student, build, checkpoint, device)
}
