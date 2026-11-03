//! `arch = "tiny_dw_guided"` registration.
//!
//! The white-box line (H/4 trunk, H/2 FiLM, full-resolution guide, InstanceNorm or
//! LayerNorm) calibrates and exports. Other guided topologies still fail export.

use std::path::Path;

use anyhow::Result;
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
    print_cost(&spec);
    Ok(GuidedStudent::new(spec, device))
}

pub fn train(config: Config, provenance: RunProvenance) -> Result<()> {
    run_training(config, provenance, build_train)
}

pub fn train_cohort(configs: Vec<Config>, provenances: Vec<RunProvenance>) -> Result<()> {
    crate::train::run_all(configs, provenances, build_train)
}

pub fn calibrate(
    job: &CalibrationJob,
    card: &ModelCard,
    device: &TrainDevice,
    checkpoint: &Path,
    bits: crate::quant::Bits,
) -> Result<crate::quant::Calibration> {
    let student = card.student();
    super::common::invoke_calibrate::<GuidedStudent<TrainBackend>, _>(
        job,
        ID,
        |dev| Ok(build(&student, dev)),
        device,
        checkpoint,
        bits,
    )
}

pub fn export_int8(
    card: &ModelCard,
    checkpoint: &Path,
    calibration: &crate::quant::Calibration,
    out: &Path,
    device: &TrainDevice,
) -> Result<crate::quant::QuantManifest> {
    let student = card.student();
    super::common::export_int8::<GuidedStudent<TrainBackend>, _>(
        &student,
        build,
        checkpoint,
        calibration,
        out,
        device,
    )
}

pub fn load_fp32(
    card: &ModelCard,
    checkpoint: &Path,
    device: &TrainDevice,
) -> Result<StudentHandle> {
    let student = card.student();
    super::common::load_fp32::<GuidedStudent<TrainBackend>, _>(&student, build, checkpoint, device)
}
