//! Dump every stage of a trained `tiny_dw_guided` student, two path ablations,
//! and its effective receptive field, for offline analysis.
//!
//! ```text
//! cargo run --example guided_probe --no-default-features --features ndarray -- \
//!   experiments/tiny_dw_whitebox_line/trials/d_line2 <out dir> [checkpoint.mpk]
//! ```
//!
//! Per sample image `<out>/<image>/`:
//!
//! - `<stage>.npy` -- each tap of [`GuidedStudent::trace`], NCHW f32
//! - `rgb_mean_paint.npy`, `rgb_mean_guide.npy` -- the output with that path
//!   replaced by its per-channel spatial mean
//! - `erf.npy` -- `|∂ luma(out[p]) / ∂ in|`, summed over RGB and averaged over a
//!   grid of output pixels `p`, cropped to a 129×129 window centred on `p`

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use fpga_st_train::InferBackend;
use fpga_st_train::backend::select_device;
use fpga_st_train::config::ModelCard;
use fpga_st_train::data::{load_rgb, rgb_to_tensor};
use fpga_st_train::registry::build_guided;
use fpga_st_train::student::guided::{Ablate, GuidedStudent};

type Grad = Autodiff<InferBackend>;

const IMAGES: [&str; 6] = ["animal", "cat", "city", "landscape", "portrait", "water"];
const ERF_HALF: usize = 64;

fn main() {
    let mut args = std::env::args().skip(1);
    let trial = PathBuf::from(args.next().expect("trial dir"));
    let out_dir = PathBuf::from(args.next().expect("output dir"));
    let checkpoint = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| trial.join("model.mpk"));
    let device = select_device("d0");
    let card = ModelCard::load(&trial).expect("model.json");
    let student = card.student();
    let model: GuidedStudent<Grad> = build_guided::<Grad>(&student, &device)
        .load_file(
            &checkpoint,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .expect("checkpoint");
    let plain = model.valid();

    for name in IMAGES {
        let path = trial.join(format!("content_{name}.png"));
        if !path.exists() {
            continue;
        }
        let image = load_rgb(&path).expect(name);
        let dir = out_dir.join(name);
        fs::create_dir_all(&dir).expect("output dir");
        let x = rgb_to_tensor::<InferBackend>(&image, &device);

        plain.trace(x.clone(), Ablate::None, &mut |stage, map| {
            write_npy(&dir.join(format!("{stage}.npy")), map);
        });
        let mean_paint = plain.trace(x.clone(), Ablate::MeanPaint, &mut |_, _| {});
        write_npy(&dir.join("rgb_mean_paint.npy"), &mean_paint);
        let mean_guide = plain.trace(x.clone(), Ablate::MeanGuide, &mut |_, _| {});
        write_npy(&dir.join("rgb_mean_guide.npy"), &mean_guide);

        let erf = effective_field(&model, &image, &device);
        write_npy(&dir.join("erf.npy"), &erf);
        println!("probed {name}");
    }
}

/// Mean input-gradient magnitude around a 4×4 grid of interior output pixels.
fn effective_field(
    model: &GuidedStudent<Grad>,
    image: &image::RgbImage,
    device: &fpga_st_train::InferDevice,
) -> Tensor<InferBackend, 4> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let side = 2 * ERF_HALF + 1;
    let mut acc = Tensor::<InferBackend, 4>::zeros([1, 1, side, side], device);
    let mut count = 0;
    for gy in 0..4 {
        for gx in 0..4 {
            let py = ERF_HALF + (h - 2 * ERF_HALF - 1) * gy / 3;
            let px = ERF_HALF + (w - 2 * ERF_HALF - 1) * gx / 3;
            let x = rgb_to_tensor::<Grad>(image, device).require_grad();
            let y = model.trace(x.clone(), Ablate::None, &mut |_, _| {});
            let pixel = y.slice(s![.., .., py..py + 1, px..px + 1]);
            let luma =
                Tensor::<Grad, 1>::from_floats([0.299, 0.587, 0.114], device).reshape([1, 3, 1, 1]);
            let grads = (pixel * luma).sum().backward();
            let g = x.grad(&grads).expect("input grad").abs().sum_dim(1);
            acc = acc
                + g.slice(s![
                    ..,
                    ..,
                    py - ERF_HALF..py + ERF_HALF + 1,
                    px - ERF_HALF..px + ERF_HALF + 1
                ]);
            count += 1;
        }
    }
    acc.div_scalar(count as f32)
}

fn write_npy<B: Backend>(path: &Path, map: &Tensor<B, 4>) {
    let dims = map.dims();
    let data = map.clone().into_data().convert::<f32>();
    let values = data.to_vec::<f32>().expect("f32 map");
    let shape = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape},), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut file = fs::File::create(path).expect("npy file");
    file.write_all(b"\x93NUMPY\x01\x00").expect("npy magic");
    file.write_all(&(header.len() as u16).to_le_bytes())
        .expect("npy header len");
    file.write_all(header.as_bytes()).expect("npy header");
    for v in values {
        file.write_all(&v.to_le_bytes()).expect("npy data");
    }
}
