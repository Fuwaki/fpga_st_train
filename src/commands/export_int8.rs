use anyhow::Result;

use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::quant::Calibration;
use fpga_st_train::registry;

use crate::cli::ExportInt8Args;

pub fn run(args: ExportInt8Args, device_spec: &str) -> Result<()> {
    let card = ModelCard::load(&args.model)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.model.join("model.mpk"));
    let device = select_device(device_spec);
    let calibration = Calibration::load(&args.calibration)?;

    let manifest =
        registry::export_int8_from_card(&card, &checkpoint, &calibration, &args.out, &device)?;

    card.with_meta("bits", serde_json::Value::String(manifest.bits.label()))
        .with_meta(
            "weight_packing",
            serde_json::Value::String(manifest.weight_packing.clone()),
        )
        .with_meta("weight_bytes", serde_json::json!(manifest.weight_bytes))
        .with_meta(
            "calibration_batches",
            serde_json::json!(calibration.batches),
        )
        .with_meta("calibration_safety", serde_json::json!(calibration.safety))
        .save(&args.out)?;
    calibration.save(&args.out.join("calibration.json"))?;

    println!(
        "{} layers, {} {} weights ({:.1} KiB, {}) -> {}",
        manifest.layers.len(),
        manifest.weight_bytes,
        manifest.bits.label(),
        manifest.weight_bytes as f32 / 1024.0,
        manifest.weight_packing,
        args.out.display()
    );
    for layer in &manifest.layers {
        println!(
            "  {:<12} {:>3}->{:<3} g{:>2} k{}x{} s{} pad{}  in {:.5}  out {:.5}  worst requant {:.5}",
            layer.name,
            layer.in_channels,
            layer.out_channels,
            layer.groups,
            layer.kernel[0],
            layer.kernel[1],
            layer.stride[0],
            layer.reflect_pad,
            layer.in_scale,
            layer.out_scale,
            layer.requant.iter().cloned().fold(0.0f32, f32::max),
        );
    }
    Ok(())
}
