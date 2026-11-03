use std::fs;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use serde_json::json;

use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::registry::build_guided;
use fpga_st_train::train::save_checkpoint;
use fpga_st_train::{InferBackend, StudentModel};

use crate::cli::FoldYcArgs;

pub fn run(args: FoldYcArgs, device_spec: &str) -> Result<()> {
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
    let source_params = model.num_params();
    println!("{}", model.describe());
    let yc = model.fold_yc(&device).map_err(anyhow::Error::msg)?;
    println!("{}", yc.describe());
    println!(
        "params {source_params} -> {}   macs {:.1} M @ 640x480",
        yc.num_params(),
        yc.spec().cost(480, 640).macs() as f64 / 1e6,
    );

    fs::create_dir_all(&args.out)?;
    let card = yc_card(&card)?.with_meta("folded_from", json!(checkpoint.display().to_string()));
    card.save(&args.out)?;
    let mpk = args.out.join("model.mpk");
    save_checkpoint(&yc, mpk.clone())?;

    let reloaded = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&card.student(), &device),
        &mpk,
        &device,
    )?;
    if reloaded.num_params() != yc.num_params() {
        bail!(
            "reloaded checkpoint has {} parameters, fold wrote {}",
            reloaded.num_params(),
            yc.num_params()
        );
    }
    let blank = Tensor::<InferBackend, 4>::zeros([1, 3, 8, 8], &device);
    let values = reloaded
        .stylize(blank)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("folded forward: {err}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("folded forward produced a non-finite value");
    }
    println!(
        "reloaded {} parameters and a finite forward",
        reloaded.num_params()
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

fn yc_card(card: &ModelCard) -> Result<ModelCard> {
    let mut options = card.options.clone();
    let obj = options
        .as_object_mut()
        .context("student options in model.json must be an object")?;
    obj.insert("input".to_owned(), json!("yc"));
    Ok(ModelCard {
        arch: card.arch.clone(),
        options,
        meta: card.meta.clone(),
    })
}
