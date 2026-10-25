//! `arch = "tiny_dw"` registration.

use std::path::Path;

use anyhow::Result;
use burn::prelude::*;

use crate::TrainBackend;
use crate::backend::TrainDevice;
use crate::config::{Config, ModelCard, StudentConfig};
use crate::infer_student::StudentHandle;
use crate::quant::CalibrationJob;
use crate::run_log::RunProvenance;
use crate::student::TinyDwStudent;
use crate::student::plugin;
use crate::train::run as run_training;

pub const ID: &str = "tiny_dw";

pub fn build<B: Backend>(config: &StudentConfig, device: &B::Device) -> TinyDwStudent<B> {
    let plugin = plugin::parse_kind(config.str_option("plugin", "none"))
        .unwrap_or_else(|err| panic!("{err}"));
    let site = plugin::parse_site(config.str_option("plugin_site", "h4"))
        .unwrap_or_else(|err| panic!("{err}"));
    let color = crate::student::tiny_dw::parse_color(config.str_option("color", "none"))
        .unwrap_or_else(|err| panic!("{err}"));
    let encoder = crate::student::tiny_dw::parse_encoder(config.str_option("encoder", "sep"))
        .unwrap_or_else(|err| panic!("{err}"));
    let lut_bits = crate::student::tiny_dw::parse_lut_bits(config.usize_option("lut_bits", 8))
        .unwrap_or_else(|err| panic!("{err}"));
    let trunk_h8 = crate::student::tiny_dw::parse_trunk_h8(config.str_option("trunk_at", "h4"))
        .unwrap_or_else(|err| panic!("{err}"));
    TinyDwStudent::with_encoder(
        config.usize_option("base", 32),
        config.usize_option("res_blocks", 4),
        config.usize_option("mid", 32),
        config.usize_option("fine", 16),
        plugin,
        site,
        color,
        encoder,
        config.usize_option("lut_hidden", 4),
        lut_bits,
        trunk_h8,
        device,
    )
}

fn build_train(
    config: &StudentConfig,
    device: &TrainDevice,
) -> Result<TinyDwStudent<TrainBackend>> {
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
    super::common::invoke_calibrate::<TinyDwStudent<TrainBackend>, _>(
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
    super::common::export_int8::<TinyDwStudent<TrainBackend>, _>(
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
    super::common::load_fp32::<TinyDwStudent<TrainBackend>, _>(&student, build, checkpoint, device)
}
