//! Dump one ONNX teacher forward as raw little-endian f32, NCHW RGB in `[0, 1]`.
//!
//! ```text
//! onnx_dump <rain_princess|whitebox> <model.onnx> <image> <side> <input.f32> <output.f32>
//! ```
//!
//! The input file is the exact tensor the session sees, so a CPU ONNX Runtime
//! run can be compared without a second resize.

use std::path::Path;

use anyhow::{Context, Result, bail};
use fpga_st_train::config::TeacherKind;
use fpga_st_train::data::load_rgb;
use fpga_st_train::teacher::OnnxTeacher;
use image::imageops::FilterType;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let kind = args.next().context("kind")?;
    let model = args.next().context("model")?;
    let image = args.next().context("image")?;
    let side: u32 = args.next().context("side")?.parse().context("side")?;
    let input_out = args.next().context("input.f32")?;
    let output_out = args.next().context("output.f32")?;

    let kind = TeacherKind::parse(&kind)?;
    let img = load_rgb(Path::new(&image))?;
    let img = image::imageops::resize(&img, side, side, FilterType::Lanczos3);
    let (w, h) = img.dimensions();
    let plane = (w as usize) * (h as usize);
    let raw = img.as_raw();
    let mut nchw = vec![0f32; 3 * plane];
    for i in 0..plane {
        let o = i * 3;
        nchw[i] = raw[o] as f32 / 255.0;
        nchw[plane + i] = raw[o + 1] as f32 / 255.0;
        nchw[2 * plane + i] = raw[o + 2] as f32 / 255.0;
    }
    write_f32(&input_out, &nchw)?;

    let teacher = OnnxTeacher::open(kind, Path::new(&model))?;
    let out = teacher.run_nchw(&nchw, 1, h as usize, w as usize)?;
    if out.len() != nchw.len() {
        bail!("output length {} != input length {}", out.len(), nchw.len());
    }
    write_f32(&output_out, &out)?;
    println!(
        "wrote {input_out} and {output_out}  1x3x{h}x{w}  device {}",
        teacher.device_label()
    );
    Ok(())
}

fn write_f32(path: &str, data: &[f32]) -> Result<()> {
    let bytes = bytemuck_bytes(data);
    std::fs::write(path, bytes).with_context(|| format!("cannot write {path}"))
}

fn bytemuck_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}
