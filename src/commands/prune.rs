use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::ops::PadMode;
use serde_json::json;

use fpga_st_train::InferBackend;
use fpga_st_train::StudentModel;
use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::data::{image_paths, load_rgb, rgb_to_tensor};
use fpga_st_train::registry::build_guided;
use fpga_st_train::student::guided::PruneWidths;
use fpga_st_train::train::save_checkpoint;

use crate::cli::PruneArgs;

pub fn run(args: PruneArgs, device_spec: &str) -> Result<()> {
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

    let images = content_images(&args)?;
    println!(
        "ranking channels on {} image(s), reflect-padded to {}x{}",
        images.len(),
        args.width,
        args.height
    );
    let mut batch = Vec::with_capacity(images.len());
    for path in &images {
        let image = load_rgb(path).with_context(|| format!("cannot read {}", path.display()))?;
        let tensor = rgb_to_tensor::<InferBackend>(&image, &device);
        batch.push(pad_frame(tensor, args.height, args.width)?);
    }

    let widths = PruneWidths {
        base: args.base,
        trunk: args.trunk,
        up_h4: args.up_h4,
        up_h2: args.up_h2,
        h2_guide: args.h2_guide,
        guide: args.guide,
        res_blocks: args.res_blocks,
    };
    let (narrow, report) = model
        .prune(&batch, widths, args.height, args.width, &device)
        .map_err(anyhow::Error::msg)?;
    println!("{}", narrow.describe());
    println!(
        "params {} -> {}   macs {:.1} -> {:.1} M @ {}x{}",
        report.source_params,
        report.params,
        report.source_macs as f64 / 1e6,
        report.macs as f64 / 1e6,
        args.width,
        args.height,
    );
    for (name, fraction) in &report.retained {
        let idx = &report.kept[name];
        println!(
            "  {name:<12} keep {:>3}  retained {fraction:.3}  idx {}",
            idx.len(),
            idx.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    fs::create_dir_all(&args.out)?;
    let narrow_card = narrow_card(&card, widths)?
        .with_meta("pruned_from", json!(checkpoint.display().to_string()))
        .with_meta("source_params", json!(report.source_params));
    narrow_card.save(&args.out)?;
    let mpk = args.out.join("model.mpk");
    save_checkpoint(&narrow, mpk.clone())?;
    let record = json!({
        "source": checkpoint.display().to_string(),
        "images": images.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "frame": {"width": args.width, "height": args.height},
        "widths": {
            "base": widths.base,
            "trunk": widths.trunk,
            "up_h4": widths.up_h4,
            "up_h2": widths.up_h2,
            "h2_guide": widths.h2_guide,
            "guide": widths.guide,
            "res_blocks": widths.res_blocks,
        },
        "kept": report.kept,
        "retained": report.retained,
        "params": report.params,
        "source_params": report.source_params,
        "macs": report.macs,
        "source_macs": report.source_macs,
    });
    fs::write(
        args.out.join("prune.json"),
        serde_json::to_string_pretty(&record)? + "\n",
    )?;

    let reloaded = fpga_st_train::train::load_module(
        build_guided::<InferBackend>(&narrow_card.student(), &device),
        &mpk,
        &device,
    )?;
    if reloaded.num_params() != narrow.num_params() {
        bail!(
            "reloaded checkpoint has {} parameters, prune wrote {}",
            reloaded.num_params(),
            narrow.num_params()
        );
    }
    let styled = reloaded.stylize(batch[0].clone());
    let values = styled
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("pruned forward: {err}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("pruned forward produced a non-finite value");
    }
    println!(
        "reloaded {} parameters and a finite forward",
        reloaded.num_params()
    );
    println!("wrote {}", args.out.display());
    Ok(())
}

fn narrow_card(card: &ModelCard, widths: PruneWidths) -> Result<ModelCard> {
    let mut options = card.options.clone();
    let obj = options
        .as_object_mut()
        .context("student options in model.json must be an object")?;
    for (key, value) in [
        ("base", widths.base),
        ("trunk", widths.trunk),
        ("up_h4", widths.up_h4),
        ("up_h2", widths.up_h2),
        ("h2_guide", widths.h2_guide),
        ("guide", widths.guide),
        ("res_blocks", widths.res_blocks),
    ] {
        obj.insert(key.to_owned(), json!(value));
    }
    Ok(ModelCard {
        arch: card.arch.clone(),
        options,
        meta: card.meta.clone(),
    })
}

fn content_images(args: &PruneArgs) -> Result<Vec<PathBuf>> {
    let paths = if let Some(content) = &args.content {
        image_paths(content)?
    } else {
        image_paths(&args.from)?
            .into_iter()
            .filter(|path| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem.starts_with("content_"))
            })
            .collect()
    };
    if paths.is_empty() {
        bail!(
            "no content images in {} (pass --content)",
            args.content
                .as_deref()
                .unwrap_or(args.from.as_path())
                .display()
        );
    }
    Ok(paths)
}

fn pad_frame<B: Backend>(image: Tensor<B, 4>, height: usize, width: usize) -> Result<Tensor<B, 4>> {
    let [_, _, h, w] = image.dims();
    if h > height || w > width {
        bail!("image is {w}x{h}, larger than the score frame {width}x{height}");
    }
    let pad_h = height - h;
    let pad_w = width - w;
    if pad_h == 0 && pad_w == 0 {
        return Ok(image);
    }
    Ok(image.pad([(0, pad_h), (0, pad_w)], PadMode::Reflect))
}
