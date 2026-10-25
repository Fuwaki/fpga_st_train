//! Shared hooks used by every registered student architecture.

use std::path::Path;

use crate::backend::TrainDevice;
use crate::config::StudentConfig;
use crate::export::Exportable;
use crate::infer_student::StudentHandle;
use crate::quant::CalibrationJob;
use crate::quant::{Quantizable, calibrate_activation_ranges};
use crate::student::StudentModel;
use crate::{InferBackend, TrainBackend};
use anyhow::Result;
use burn::module::AutodiffModule;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

pub(crate) fn invoke_calibrate<M, F>(
    job: &CalibrationJob,
    arch: &str,
    build: F,
    device: &TrainDevice,
    checkpoint: &Path,
    bits: crate::quant::Bits,
) -> Result<crate::quant::Calibration>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: Quantizable<InferBackend>,
    F: FnOnce(&TrainDevice) -> Result<M>,
{
    calibrate_activation_ranges(job, arch, build, device, checkpoint, bits)
}

pub(crate) fn export_int8<M, F>(
    student: &StudentConfig,
    build: F,
    checkpoint: &Path,
    calibration: &crate::quant::Calibration,
    out: &Path,
    device: &TrainDevice,
) -> Result<crate::quant::QuantManifest>
where
    M: StudentModel<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: Exportable,
    F: Fn(&StudentConfig, &TrainDevice) -> M,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = build(student, device)
        .load_file(checkpoint.to_path_buf(), &recorder, device)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", checkpoint.display()))?
        .valid();
    crate::export::export_int8(&model, calibration, out)
}

pub(crate) fn load_fp32<M, F>(
    student: &StudentConfig,
    build: F,
    checkpoint: &Path,
    device: &TrainDevice,
) -> Result<StudentHandle>
where
    M: StudentModel<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: StudentModel<InferBackend> + Quantizable<InferBackend> + 'static,
    F: Fn(&StudentConfig, &TrainDevice) -> M,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = build(student, device)
        .load_file(checkpoint.to_path_buf(), &recorder, device)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", checkpoint.display()))?
        .valid();
    Ok(StudentHandle::new(model))
}
