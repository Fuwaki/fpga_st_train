use std::fs;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use serde_json::json;

use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::registry::build_guided;
use fpga_st_train::train::save_checkpoint;
use fpga_st_train::{InferBackend, StudentModel};

use crate::cli::PackStylesArgs;

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
        .expand_styles(args.styles, &device)
        .map_err(anyhow::Error::msg)?;
    println!("{}", packed.describe());
    println!("params {} -> {}", model.num_params(), packed.num_params());

    fs::create_dir_all(&args.out)?;
    let card = packet_card(&card, args.styles)?
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
    let blank = Tensor::<InferBackend, 4>::zeros([1, 3, 8, 8], &device);
    let first = reloaded
        .stylize_style(blank.clone(), 0)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("style 0 forward: {err}"))?;
    let second = reloaded
        .stylize_style(blank, 1)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("style 1 forward: {err}"))?;
    if first.iter().any(|value| !value.is_finite()) || second.iter().any(|value| !value.is_finite())
    {
        bail!("packed forward produced a non-finite value");
    }
    let drift = first
        .iter()
        .zip(&second)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if drift > 1e-5 {
        bail!("style 0 and style 1 differ by {drift} before training");
    }
    println!(
        "reloaded {} parameters; both styles match on a blank frame",
        reloaded.num_params()
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

fn packet_card(card: &ModelCard, styles: usize) -> Result<ModelCard> {
    let mut options = card.options.clone();
    let obj = options
        .as_object_mut()
        .context("student options in model.json must be an object")?;
    obj.insert("styles".to_owned(), json!(styles));
    Ok(ModelCard {
        arch: card.arch.clone(),
        options,
        meta: card.meta.clone(),
    })
}
