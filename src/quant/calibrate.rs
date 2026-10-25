//! Post-training activation range measurement.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::TrainBackend;
use crate::backend::TrainDevice;
use crate::data::dataset::PairDataset;
use crate::data::train_source::TrainContentSource;
use crate::quant::{Bits, Calibration, Quantizable};
use crate::student::StudentModel;
use crate::train::{build_batch, load_checkpoint};

/// Inputs for the calibration pass (independent of how the model was trained).
#[derive(Debug, Clone)]
pub struct CalibrationJob {
    pub content_dir: PathBuf,
    pub crop: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub batches: usize,
    pub safety: f32,
}

impl Default for CalibrationJob {
    fn default() -> Self {
        Self {
            content_dir: PathBuf::from("data/content"),
            crop: 192,
            batch_size: 4,
            seed: 0,
            batches: 20,
            safety: 1.0,
        }
    }
}

/// Measure activation ranges and build a [Calibration] file for export / QAT.
pub fn calibrate_activation_ranges<M, F>(
    job: &CalibrationJob,
    arch: &str,
    build_student: F,
    device: &TrainDevice,
    checkpoint: &Path,
    bits: Bits,
) -> Result<Calibration>
where
    M: StudentModel<TrainBackend> + Quantizable<TrainBackend> + AutodiffModule<TrainBackend>,
    M::InnerModule: Quantizable<<TrainBackend as AutodiffBackend>::InnerBackend>,
    F: FnOnce(&TrainDevice) -> Result<M>,
{
    let mut rng = StdRng::seed_from_u64(job.seed);
    let dataset = TrainContentSource::Disk(PairDataset::build(&job.content_dir, None, None)?);

    let model = load_checkpoint(build_student(device)?, checkpoint, device)?;
    let points = model.quant_points();
    if points.is_empty() {
        bail!("this student has no quantised points (Quantizable::quant_points is empty)");
    }
    let model = model.valid();
    log::info!(
        "calibrating {} quantised points for {} on {} batches of {}x{} crops",
        points.len(),
        bits.label(),
        job.batches,
        job.batch_size,
        job.crop
    );

    let mut absmax = vec![0.0f32; points.len()];
    for step in 1..=job.batches {
        let batch = build_batch(&dataset, job.crop as u32, job.batch_size, &mut rng, device)?;
        model.observe_activations(batch.content.valid(), &mut |point, value| {
            if value > absmax[point] {
                absmax[point] = value;
            }
        });
        if step % 5 == 0 || step == job.batches {
            log::info!("calibrated {step}/{} batches", job.batches);
        }
    }

    let scales: Vec<f32> = absmax
        .iter()
        .map(|m| (m * job.safety) / bits.act_max())
        .map(|s| s.max(1e-5))
        .collect();
    Ok(Calibration {
        arch: arch.to_owned(),
        points,
        scales,
        absmax,
        safety: job.safety,
        batches: job.batches,
        bits,
    })
}
