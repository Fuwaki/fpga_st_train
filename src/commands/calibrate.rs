use anyhow::{Result, anyhow};

use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::registry;
use fpga_st_train::quant::CalibrationJob;

use crate::cli::CalibrateArgs;

pub fn run(args: CalibrateArgs, device_spec: &str) -> Result<()> {
    let bits = fpga_st_train::quant::Bits::parse(&args.bits).map_err(|e| anyhow!("{e}"))?;
    let card = ModelCard::load(&args.model)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.model.join("model.mpk"));
    let device = select_device(device_spec);
    let job = CalibrationJob {
        content_dir: args.content.clone(),
        crop: args.crop,
        batch_size: args.batch_size,
        seed: args.seed,
        batches: args.batches,
        safety: args.safety,
    };

    let calibration = registry::calibrate_student(&job, &card, &device, &checkpoint, bits)?;
    calibration.save(&args.out)?;

    println!(
        "{} -- {:<14} {:>12} {:>12}",
        calibration.bits.label(),
        "point",
        "absmax",
        "scale"
    );
    for ((name, absmax), scale) in calibration
        .points
        .iter()
        .zip(&calibration.absmax)
        .zip(&calibration.scales)
    {
        println!("{name:<14} {absmax:>12.5} {scale:>12.6}");
    }
    println!(
        "{} quantised points over {} batches -> {}",
        calibration.scales.len(),
        calibration.batches,
        args.out.display()
    );
    Ok(())
}
