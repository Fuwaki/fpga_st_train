//! Evaluation helpers: image-space quality metrics and side-by-side sheets.
//!
//! These are the metrics that need no extra weights. Gram/perceptual numbers require VGG
//! and are the documented next step.
//!
//! Definitions follow the Python pipeline in style-transfer/scripts/eval_style.py so the
//! numbers are directly comparable:
//!
//! - lapvar: mean squared Laplacian, i.e. stroke/texture energy. Higher = more brushwork.
//! - checkerboard: energy on the 2px (Nyquist) period. This is the artefact signature of
//!   nearest-neighbour upsampling combined with instance norm -- the failure mode the
//!   Vangogh student showed in evaluation.

use anyhow::Result;
use image::{Rgb, RgbImage};

/// Luma plane in [0, 1].
fn luma(img: &RgbImage) -> Vec<f64> {
    img.pixels()
        .map(|p| (0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64) / 255.0)
        .collect()
}

/// Mean squared Laplacian (stroke energy).
pub fn lapvar(img: &RgbImage) -> f64 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let a = luma(img);
    let at = |x: usize, y: usize| a[y * w + x];
    let mut acc = 0.0;
    let mut n = 0.0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let d1 = at(x - 1, y) - 2.0 * at(x, y) + at(x + 1, y);
            let d2 = at(x, y - 1) - 2.0 * at(x, y) + at(x, y + 1);
            acc += d1 * d1 + d2 * d2;
            n += 1.0;
        }
    }
    if n == 0.0 { 0.0 } else { acc / n }
}

/// Mean and standard deviation of luma.
pub fn contrast(img: &RgbImage) -> (f64, f64) {
    let a = luma(img);
    let mean = a.iter().sum::<f64>() / a.len() as f64;
    let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / a.len() as f64;
    (mean, var.sqrt())
}

/// Mean per-pixel saturation, (max channel - min channel) in [0, 1].
pub fn saturation(img: &RgbImage) -> f64 {
    let n = img.pixels().count().max(1) as f64;
    img.pixels()
        .map(|p| {
            let mx = p[0].max(p[1]).max(p[2]) as f64;
            let mn = p[0].min(p[1]).min(p[2]) as f64;
            (mx - mn) / 255.0
        })
        .sum::<f64>()
        / n
}

/// Peak signal-to-noise ratio between two same-sized images, on [0, 1].
pub fn psnr(a: &RgbImage, b: &RgbImage) -> Result<f64> {
    anyhow::ensure!(
        a.dimensions() == b.dimensions(),
        "psnr needs equal sizes, got {:?} and {:?}",
        a.dimensions(),
        b.dimensions()
    );
    let n = (a.width() as f64) * (a.height() as f64) * 3.0;
    let mse = a
        .pixels()
        .zip(b.pixels())
        .map(|(p, q)| {
            let d0 = (p[0] as f64 - q[0] as f64) / 255.0;
            let d1 = (p[1] as f64 - q[1] as f64) / 255.0;
            let d2 = (p[2] as f64 - q[2] as f64) / 255.0;
            d0 * d0 + d1 * d1 + d2 * d2
        })
        .sum::<f64>()
        / n;
    Ok(10.0 * (1.0 / mse.max(1e-12)).log10())
}

/// Nyquist-period (2px) energy fraction: the checkerboard artefact score.
///
/// Estimated from the alternating row/column sums, which is the Nyquist bin of a naive
/// DFT and avoids pulling in an FFT dependency.
pub fn checkerboard(img: &RgbImage) -> f64 {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 4 || h < 4 {
        return 0.0;
    }
    let a = luma(img);
    let total: f64 = a.iter().map(|v| v * v).sum::<f64>() + 1e-12;
    let mut alt_x = 0.0;
    let mut alt_y = 0.0;
    for y in 0..h {
        for x in 0..w {
            let v = a[y * w + x];
            if x % 2 == 0 {
                alt_x += v;
            } else {
                alt_x -= v;
            }
            if y % 2 == 0 {
                alt_y += v;
            } else {
                alt_y -= v;
            }
        }
    }
    ((alt_x * alt_x + alt_y * alt_y) / (w as f64 * h as f64)) / total
}

/// Human-readable metric line.
pub fn describe(img: &RgbImage) -> String {
    let (mean, std) = contrast(img);
    format!(
        "lapvar {:.5}  checker {:.3e}  luma {:.3}+-{:.3}  sat {:.3}",
        lapvar(img),
        checkerboard(img),
        mean,
        std,
        saturation(img),
    )
}

/// Build a one-row contact sheet: content | a | b | ...
///
/// Pane labels are carried by the caller; this only lays out pixels.
pub fn sheet(panes: &[(String, RgbImage)], gap: u32) -> RgbImage {
    let height = panes.iter().map(|(_, i)| i.height()).max().unwrap_or(1);
    let width: u32 =
        panes.iter().map(|(_, i)| i.width()).sum::<u32>() + gap * (panes.len() as u32 + 1);
    let mut out = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));
    let mut x = gap;
    for (_, img) in panes {
        let y = (height - img.height()) / 2;
        for (px, py, pixel) in img.enumerate_pixels() {
            let dst_x = x + px;
            let dst_y = y + py;
            if dst_x < width && dst_y < height {
                out.put_pixel(dst_x, dst_y, *pixel);
            }
        }
        x += img.width() + gap;
    }
    out
}
