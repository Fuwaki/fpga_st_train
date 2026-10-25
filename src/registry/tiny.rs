//! `arch = "tiny"` registration.

use std::path::Path;

use anyhow::Result;
use burn::prelude::*;

use crate::TrainBackend;
use crate::backend::TrainDevice;
use crate::config::{Config, ModelCard, StudentConfig};
use crate::infer_student::StudentHandle;
use crate::quant::CalibrationJob;
use crate::run_log::RunProvenance;
use crate::student::TinyStudent;
use crate::train::run as run_training;

pub const ID: &str = "tiny";

pub fn build<B: Backend>(config: &StudentConfig, device: &B::Device) -> TinyStudent<B> {
    TinyStudent::new(
        config.usize_option("base", 32),
        config.usize_option("res_blocks", 4),
        config.usize_option("mid", 32),
        config.usize_option("fine", 16),
        device,
    )
}

fn build_train(config: &StudentConfig, device: &TrainDevice) -> Result<TinyStudent<TrainBackend>> {
    Ok(build(config, device))
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
    super::common::invoke_calibrate::<TinyStudent<TrainBackend>, _>(
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
    super::common::export_int8::<TinyStudent<TrainBackend>, _>(
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
    super::common::load_fp32::<TinyStudent<TrainBackend>, _>(&student, build, checkpoint, device)
}
