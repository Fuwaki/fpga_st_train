//! Image, tensor and dataset IO.

pub mod dataset;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use image::{RgbImage, imageops::FilterType};

/// Extensions we treat as images.
pub const IMAGE_EXTENSIONS: [&str; 7] = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

/// True if the path looks like an image we can read.
pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Sorted list of image files directly inside a directory, or the single path itself.
pub fn image_paths(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        bail!("{} is neither a file nor a directory", path.display());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(path)
        .with_context(|| format!("cannot list {}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    out.sort();
    if out.is_empty() {
        bail!("no images found in {}", path.display());
    }
    Ok(out)
}

pub fn load_rgb(path: impl AsRef<Path>) -> Result<RgbImage> {
    let path = path.as_ref();
    let img = image::open(path).with_context(|| format!("cannot open image {}", path.display()))?;
    Ok(img.to_rgb8())
}

pub fn save_rgb(img: &RgbImage, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    img.save(path)
        .with_context(|| format!("cannot write image {}", path.display()))
}

fn scaled(w: u32, h: u32, factor: f64) -> (u32, u32) {
    (
        (w as f64 * factor).round().max(1.0) as u32,
        (h as f64 * factor).round().max(1.0) as u32,
    )
}

/// Resize so the short edge equals size, keeping the aspect ratio.
///
/// This mirrors PIL's Image.resize as used by torchvision transforms.Resize.
pub fn resize_short_edge(img: &RgbImage, size: u32, filter: FilterType) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let short = w.min(h);
    if short == size {
        return img.clone();
    }
    let (nw, nh) = scaled(w, h, size as f64 / short as f64);
    image::imageops::resize(img, nw, nh, filter)
}

/// Resize so the long edge equals size, keeping the aspect ratio.
pub fn resize_long_edge(img: &RgbImage, size: u32, filter: FilterType) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    if long == size {
        return img.clone();
    }
    let (nw, nh) = scaled(w, h, size as f64 / long as f64);
    image::imageops::resize(img, nw, nh, filter)
}

/// Resize to an exact size, ignoring the aspect ratio.
pub fn resize_exact(img: &RgbImage, w: u32, h: u32, filter: FilterType) -> RgbImage {
    if img.width() == w && img.height() == h {
        return img.clone();
    }
    image::imageops::resize(img, w, h, filter)
}

/// Center-crop to at most w x h.
pub fn center_crop(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let (iw, ih) = (img.width(), img.height());
    let cw = w.min(iw);
    let ch = h.min(ih);
    let x = (iw - cw) / 2;
    let y = (ih - ch) / 2;
    image::imageops::crop_imm(img, x, y, cw, ch).to_image()
}

/// Reflect-pad an image so both edges become a multiple of `m`.
///
/// The student downsamples by 4 and upsamples by 4; on an odd-sized image that pair is not
/// exactly reversible (ceil arithmetic adds a pixel). Padding to a multiple of `m` first,
/// then cropping the result back, keeps output size equal to input size.
pub fn pad_to_multiple(img: &RgbImage, m: u32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let pw = (m - w % m) % m;
    let ph = (m - h % m) % m;
    if pw == 0 && ph == 0 {
        return img.clone();
    }
    let sx = |x: u32| {
        if x < w {
            x
        } else {
            (2 * w).saturating_sub(2 + x)
        }
    };
    let sy = |y: u32| {
        if y < h {
            y
        } else {
            (2 * h).saturating_sub(2 + y)
        }
    };
    let mut out = RgbImage::new(w + pw, h + ph);
    for y in 0..h + ph {
        for x in 0..w + pw {
            out.put_pixel(x, y, *img.get_pixel(sx(x).min(w - 1), sy(y).min(h - 1)));
        }
    }
    out
}

/// RGB8 to [1, 3, H, W] f32 in [0, 1].
pub fn rgb_to_tensor<B: Backend>(img: &RgbImage, device: &B::Device) -> Tensor<B, 4> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = Vec::with_capacity(3 * h * w);
    for channel in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data.push(img.get_pixel(x as u32, y as u32)[channel] as f32 / 255.0);
            }
        }
    }
    Tensor::from_data(TensorData::new(data, [1, 3, h, w]), device)
}

/// [1, 3, H, W] f32 to RGB8, clamping to [0, 1].
pub fn tensor_to_rgb<B: Backend>(tensor: Tensor<B, 4>) -> Result<RgbImage> {
    let [n, c, h, w] = tensor.dims();
    if n != 1 || c != 3 {
        bail!("expected a [1, 3, H, W] tensor, got [{n}, {c}, {h}, {w}]");
    }
    let data: Vec<f32> = tensor
        .clamp(0.0, 1.0)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("cannot read tensor data: {e}"))?;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let px = [
                (data[i] * 255.0).round() as u8,
                (data[h * w + i] * 255.0).round() as u8,
                (data[2 * h * w + i] * 255.0).round() as u8,
            ];
            img.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    Ok(img)
}

/// Write a flat f32 buffer with a shape header (used for parity checks vs PyTorch).
pub fn save_f32(path: impl AsRef<Path>, shape: &[usize], data: &[f32]) -> Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    let mut file =
        std::fs::File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let header = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    writeln!(file, "{header}")?;
    for v in data {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Read a flat f32 buffer written by save_f32.
pub fn load_f32(path: impl AsRef<Path>) -> Result<(Vec<usize>, Vec<f32>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    let newline = bytes
        .iter()
        .position(|b| *b == b'\n')
        .with_context(|| format!("{} has no header line", path.display()))?;
    let header = std::str::from_utf8(&bytes[..newline])?.trim().to_owned();
    let shape: Vec<usize> = header
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("bad shape header {header:?}"))?;
    let body = &bytes[newline + 1..];
    let data: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((shape, data))
}

/// Tensor to flat f32 buffer, for parity checks.
pub fn tensor_to_f32<B: Backend>(tensor: Tensor<B, 4>) -> Result<(Vec<usize>, Vec<f32>)> {
    let shape = tensor.dims().to_vec();
    let data = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("cannot read tensor data: {e}"))?;
    Ok((shape, data))
}
