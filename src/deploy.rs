//! Deployment inference: fp32, quant simulation, and integer reference paths.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;

use crate::backend::{TrainDevice, select_device};
use crate::config::ModelCard;
use crate::data;
use crate::eval;
use crate::infer_student::{InferStudent, StudentHandle};
use crate::intref::IntModel;
use crate::quant::{Calibration, QuantScheme};

/// The three ways a delivered model directory can be run.
pub enum Engine {
    /// Plain fp32 weights.
    Fp32(StudentHandle),
    /// fp32 with fake quantisation (training-time QAT grid).
    Sim(StudentHandle, QuantScheme),
    /// Bit-exact integer reference (what the FPGA runs).
    Fixed(IntModel),
}

impl Engine {
    pub fn describe(&self) -> String {
        match self {
            Engine::Fp32(model) => model.describe(),
            Engine::Sim(model, scheme) => {
                format!(
                    "{} simulation of {}",
                    scheme.bits.label(),
                    model.describe().trim_start_matches("fp32 ")
                )
            }
            Engine::Fixed(model) => format!("integer reference {}", model.describe()),
        }
    }

    /// Stylise one content image.
    pub fn run(&self, image: &image::RgbImage, device: &TrainDevice) -> Result<image::RgbImage> {
        match self {
            Engine::Fp32(model) => {
                let content = data::rgb_to_tensor::<crate::InferBackend>(image, device);
                data::tensor_to_rgb(model.stylize(content))
            }
            Engine::Sim(model, scheme) => {
                let content = data::rgb_to_tensor::<crate::InferBackend>(image, device);
                data::tensor_to_rgb(model.forward_quantised(content, scheme))
            }
            Engine::Fixed(model) => model.forward_image(image),
        }
    }
}

/// Resolve `auto` / `fp32` / `sim` / `int` for a model directory.
pub fn resolve_engine(
    model_dir: &Path,
    mode: &str,
    calibration: Option<PathBuf>,
    device_spec: &str,
) -> Result<(String, Engine)> {
    let mpk = model_dir.join("model.mpk");
    let manifest = model_dir.join("manifest.json");

    let mode = match mode {
        "auto" => {
            if manifest.is_file() {
                "int"
            } else if mpk.is_file() {
                "fp32"
            } else {
                bail!(
                    "{} holds neither manifest.json (an export) nor model.mpk (fp32 weights)",
                    model_dir.display()
                )
            }
        }
        known @ ("fp32" | "sim" | "int") => known,
        other => bail!("unknown mode {other:?}: use auto, fp32, sim or int"),
    };

    let device = select_device(device_spec);
    let engine = match mode {
        "int" => {
            if !manifest.is_file() {
                bail!("mode int needs {}", manifest.display());
            }
            Engine::Fixed(IntModel::load(model_dir)?)
        }
        _ => {
            if !mpk.is_file() {
                bail!("mode {mode} needs {}", mpk.display());
            }
            let card = ModelCard::load(model_dir)?;
            let loaded = crate::registry::load_student_fp32(&card, &mpk, &device)?;
            if mode == "sim" {
                let path = calibration.unwrap_or_else(|| model_dir.join("calibration.json"));
                if !path.is_file() {
                    bail!(
                        "mode sim needs calibration scales; {} does not exist",
                        path.display()
                    );
                }
                let calibration = Calibration::load(&path)?;
                Engine::Sim(loaded, calibration.scheme())
            } else {
                Engine::Fp32(loaded)
            }
        }
    };

    Ok((mode.to_owned(), engine))
}

/// Run a resolved engine over content paths and write images to `out`.
pub fn stylise_contents(
    engine: &Engine,
    mode: &str,
    device: &TrainDevice,
    content: &Path,
    out: &Path,
    long_edge_max: u32,
    ext: &str,
) -> Result<()> {
    let contents = data::image_paths(content)?;
    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    log::info!("mode {mode}: {}", engine.describe());
    log::info!("{} content image(s)", contents.len());

    let mut total = 0.0f32;
    for path in &contents {
        let img = data::load_rgb(path)?;
        let img = if long_edge_max > 0 {
            data::resize_long_edge(&img, long_edge_max, FilterType::Lanczos3)
        } else {
            img
        };
        let (w, h) = img.dimensions();
        let padded = data::pad_to_multiple(&img, 4);

        let step = Instant::now();
        let output = engine.run(&padded, device)?;
        let millis = step.elapsed().as_secs_f32() * 1e3;
        total += millis;

        let output = if padded.dimensions() == (w, h) {
            output
        } else {
            image::imageops::crop_imm(&output, 0, 0, w, h).to_image()
        };
        let name = format!("{}.{}", data::path_stem(path), ext);
        data::save_rgb(&output, out.join(&name))?;
        println!(
            "{name}  {}x{}  {millis:.1} ms  {}",
            output.width(),
            output.height(),
            eval::describe(&output)
        );
    }
    println!("total {total:.0} ms for {} image(s)", contents.len());
    Ok(())
}
