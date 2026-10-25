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

/// Mean-square energy of the 4–8 px and 8–16 px bands, averaged over RGB.
///
/// Kernels match `losses::texture_energy`: blur(5)−blur(9) and blur(9)−blur(17).
pub fn band_energy(img: &RgbImage) -> (f64, f64) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 17 || h < 17 {
        return (0.0, 0.0);
    }
    let planes = rgb_planes(img);
    let mut fine = 0.0;
    let mut coarse = 0.0;
    for plane in &planes {
        fine += band_ms(plane, w, h, 5, 9);
        coarse += band_ms(plane, w, h, 9, 17);
    }
    let n = planes.len() as f64;
    (fine / n, coarse / n)
}

/// Student energy divided by teacher energy. 1 means the band matches.
pub fn energy_ratio(student: &RgbImage, teacher: &RgbImage) -> (f64, f64) {
    let (sf, sc) = band_energy(student);
    let (tf, tc) = band_energy(teacher);
    (sf / tf.max(1e-12), sc / tc.max(1e-12))
}

fn rgb_planes(img: &RgbImage) -> [Vec<f64>; 3] {
    let n = img.width() as usize * img.height() as usize;
    let mut planes = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (i, pixel) in img.pixels().enumerate() {
        planes[0][i] = pixel[0] as f64 / 255.0;
        planes[1][i] = pixel[1] as f64 / 255.0;
        planes[2][i] = pixel[2] as f64 / 255.0;
    }
    planes
}

fn reflect_index(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = i.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}

fn box_blur_plane(src: &[f64], w: usize, h: usize, k: usize) -> Vec<f64> {
    let pad = k as isize / 2;
    let mut out = vec![0.0; w * h];
    let inv = 1.0 / (k * k) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dy in 0..k as isize {
                let yy = reflect_index(y as isize + dy - pad, h);
                for dx in 0..k as isize {
                    let xx = reflect_index(x as isize + dx - pad, w);
                    acc += src[yy * w + xx];
                }
            }
            out[y * w + x] = acc * inv;
        }
    }
    out
}

fn band_ms(src: &[f64], w: usize, h: usize, fine: usize, coarse: usize) -> f64 {
    let a = box_blur_plane(src, w, h, fine);
    let b = box_blur_plane(src, w, h, coarse);
    let mut acc = 0.0;
    for (p, q) in a.iter().zip(b.iter()) {
        let d = p - q;
        acc += d * d;
    }
    acc / (w * h) as f64
}

/// Per-channel OLS slope of `student` on `teacher` (`cov(s, t) / var(t)`).
///
/// A channel the student has flattened scores near 0. A match scores near 1.
pub fn channel_slopes(student: &RgbImage, teacher: &RgbImage) -> Result<[f64; 3]> {
    anyhow::ensure!(
        student.dimensions() == teacher.dimensions(),
        "channel slopes need equal sizes, got {:?} and {:?}",
        student.dimensions(),
        teacher.dimensions()
    );
    let n = (student.width() as usize) * (student.height() as usize);
    if n == 0 {
        return Ok([0.0; 3]);
    }
    let mut slopes = [0.0; 3];
    for c in 0..3 {
        let mut mean_s = 0.0;
        let mut mean_t = 0.0;
        for (p, q) in student.pixels().zip(teacher.pixels()) {
            mean_s += p[c] as f64;
            mean_t += q[c] as f64;
        }
        mean_s /= n as f64 * 255.0;
        mean_t /= n as f64 * 255.0;
        let mut cov = 0.0;
        let mut var_t = 0.0;
        for (p, q) in student.pixels().zip(teacher.pixels()) {
            let s = p[c] as f64 / 255.0 - mean_s;
            let t = q[c] as f64 / 255.0 - mean_t;
            cov += s * t;
            var_t += t * t;
        }
        slopes[c] = if var_t < 1e-12 { 0.0 } else { cov / var_t };
    }
    Ok(slopes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_has_no_band_energy_and_a_copy_matches() {
        let mut flat = RgbImage::new(64, 64);
        for pixel in flat.pixels_mut() {
            *pixel = Rgb([40, 80, 120]);
        }
        let (fine, coarse) = band_energy(&flat);
        assert!(fine < 1e-8, "fine {fine}");
        assert!(coarse < 1e-8, "coarse {coarse}");

        let mut wave = RgbImage::new(64, 64);
        for (x, y, pixel) in wave.enumerate_pixels_mut() {
            let v = (6.283185 * (x as f64) / 6.0).sin();
            let u = (128.0 + 80.0 * v).round() as u8;
            *pixel = Rgb([u, u, u]);
            let _ = y;
        }
        let (wave_fine, _) = band_energy(&wave);
        assert!(
            wave_fine > fine + 1e-4,
            "a 6 px wave ({wave_fine}) should carry 4–8 px energy"
        );
        let (ratio_fine, ratio_coarse) = energy_ratio(&wave, &wave);
        assert!((ratio_fine - 1.0).abs() < 1e-6);
        assert!((ratio_coarse - 1.0).abs() < 1e-6);

        let slopes = channel_slopes(&wave, &wave).unwrap();
        assert!(slopes.iter().all(|s| (*s - 1.0).abs() < 1e-6));
        let flat_slopes = channel_slopes(&flat, &wave).unwrap();
        assert!(flat_slopes.iter().all(|s| s.abs() < 1e-6));
    }
}
