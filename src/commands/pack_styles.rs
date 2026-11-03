use std::fs;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use serde_json::json;

use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::registry::build_guided;
use fpga_st_train::train::save_checkpoint;
use fpga_st_train::{InferBackend, StudentModel};

use crate::cli::{PackStylesArgs, SplitStyleArgs};

pub fn run(args: PackStylesArgs, device_spec: &str) -> Result<()> {
    if args.styles < 2 {
        bail!("--styles must be at least 2, got {}", args.styles);
    }
    let device = select_device(device_spec);
    let card = ModelCard::load(&args.from)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.from.join("model.mpk"));
    let model = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&card.student(), &device),
        &checkpoint,
        &device,
    )?;
    println!("{}", model.describe());
    let packed = model
        .repack_styles(args.styles, args.packet_guide || model.spec().packet_guide, &device)
        .map_err(anyhow::Error::msg)?;
    println!("{}", packed.describe());
    println!("params {} -> {}", model.num_params(), packed.num_params());

    fs::create_dir_all(&args.out)?;
    let card = packet_card(&card, args.styles, packed.spec().packet_guide)?
        .with_meta("packed_from", json!(checkpoint.display().to_string()));
    card.save(&args.out)?;
    let mpk = args.out.join("model.mpk");
    save_checkpoint(&packed, mpk.clone())?;

    let reloaded = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&card.student(), &device),
        &mpk,
        &device,
    )?;
    if reloaded.num_params() != packed.num_params() {
        bail!(
            "reloaded checkpoint has {} parameters, pack wrote {}",
            reloaded.num_params(),
            packed.num_params()
        );
    }
    // A ramp, so the guide sees edges. Old slots must match the source style,
    // new slots must match style 0: the identity guide affine changes nothing.
    let ramp = Tensor::<InferBackend, 1, Int>::arange(0..3 * 16 * 16, &device)
        .float()
        .div_scalar((3 * 16 * 16) as f32)
        .reshape([1, 3, 16, 16]);
    let source_styles = model.style_count();
    for style in 0..args.styles {
        let got = styled(&reloaded, &ramp, style)?;
        let want = styled(&model, &ramp, if style < source_styles { style } else { 0 })?;
        if got.iter().any(|value| !value.is_finite()) {
            bail!("packed style {style} produced a non-finite value");
        }
        let drift = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if drift > 1e-5 {
            bail!("packed style {style} drifts {drift} from its source before training");
        }
    }
    println!(
        "reloaded {} parameters; every style matches its source on a ramp frame",
        reloaded.num_params()
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

/// One style out of a packed checkpoint, guide affine folded, ready for `export-int8`.
pub fn split(args: SplitStyleArgs, device_spec: &str) -> Result<()> {
    let device = select_device(device_spec);
    let card = ModelCard::load(&args.from)?;
    let checkpoint = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| args.from.join("model.mpk"));
    let model = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&card.student(), &device),
        &checkpoint,
        &device,
    )?;
    println!("{}", model.describe());
    let single = model
        .split_style(args.style, &device)
        .map_err(anyhow::Error::msg)?;
    println!("{}", single.describe());

    fs::create_dir_all(&args.out)?;
    let mut options = card.options.clone();
    let obj = options
        .as_object_mut()
        .context("student options in model.json must be an object")?;
    obj.remove("styles");
    obj.remove("packet_guide");
    let card = ModelCard {
        arch: card.arch.clone(),
        options,
        meta: card.meta.clone(),
    }
    .with_meta("split_from", json!(checkpoint.display().to_string()))
    .with_meta("split_style", json!(args.style));
    card.save(&args.out)?;
    let mpk = args.out.join("model.mpk");
    save_checkpoint(&single, mpk.clone())?;

    let reloaded = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&card.student(), &device),
        &mpk,
        &device,
    )?;
    let ramp = Tensor::<InferBackend, 1, Int>::arange(0..3 * 16 * 16, &device)
        .float()
        .div_scalar((3 * 16 * 16) as f32)
        .reshape([1, 3, 16, 16]);
    let got = styled(&reloaded, &ramp, 0)?;
    let want = styled(&model, &ramp, args.style)?;
    let drift = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if drift > 1e-4 {
        bail!("split style {} drifts {drift} from the packed forward", args.style);
    }
    println!(
        "reloaded {} parameters; matches packed style {} to {drift:.2e} on a ramp frame",
        reloaded.num_params(),
        args.style
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

fn styled<M: StudentModel<InferBackend>>(
    model: &M,
    frame: &Tensor<InferBackend, 4>,
    style: usize,
) -> Result<Vec<f32>> {
    model
        .stylize_style(frame.clone(), style)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("style {style} forward: {err}"))
}

fn packet_card(card: &ModelCard, styles: usize, packet_guide: bool) -> Result<ModelCard> {
    let mut options = card.options.clone();
    let obj = options
        .as_object_mut()
        .context("student options in model.json must be an object")?;
    obj.insert("styles".to_owned(), json!(styles));
    if packet_guide {
        obj.insert("packet_guide".to_owned(), json!(1));
    }
    Ok(ModelCard {
        arch: card.arch.clone(),
        options,
        meta: card.meta.clone(),
    })
}
