//! A bit-exact integer interpreter for an exported model.
//!
//! This is the reference implementation of what manifest.json plus weights.bin describe:
//! integer convolutions with an int32 accumulator, per-output-channel requantisation, and the
//! unsigned uint8 image at the end. It contains no floating point arithmetic other than the
//! final per-channel requantisation multiply, which is exactly what a fixed-point datapath
//! does with a fractional multiplier.
//!
//! Two things use it:
//!
//! - the infer subcommand, so a delivered w8a8/ or w4a8/ directory can be run without the
//!   fp32 checkpoint at all -- the directory is self-contained;
//! - validation. Comparing this output against the fake-quantised burn forward pass (which
//!   uses dequantised float convolutions) is the end-to-end check that the manifest, the
//!   weight packing and the layer scales all agree with the training code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use rayon::prelude::*;

use crate::quant::{QuantLayer, QuantManifest};

/// One activation tensor as integer codes, laid out [C][H][W].
#[derive(Clone, Debug)]
struct Act {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<i32>,
}

/// A loaded exported model, ready to run.
pub struct IntModel {
    pub manifest: QuantManifest,
    /// Unpacked weight codes, indexed by QuantLayer::weight_offset.
    weights: Vec<i8>,
    /// Where the manifest and weights were read from.
    pub dir: PathBuf,
}

impl IntModel {
    /// Load manifest.json and weights.bin from a directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("cannot read {}", manifest_path.display()))?;
        let manifest: QuantManifest = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", manifest_path.display()))?;
        if manifest.points.len() != manifest.point_scales.len() {
            bail!(
                "manifest has {} points but {} scales",
                manifest.points.len(),
                manifest.point_scales.len()
            );
        }
        let weights = manifest.load_weights(&dir)?;
        Ok(Self {
            manifest,
            weights,
            dir,
        })
    }

    /// One-line description for logs.
    pub fn describe(&self) -> String {
        let m = &self.manifest;
        format!(
            "{} {} ({} layers, {} weight codes = {} KiB as {})",
            m.arch,
            m.bits.label(),
            m.layers.len(),
            self.weights.len(),
            m.weight_bytes as f32 / 1024.0,
            m.weight_packing,
        )
    }

    /// Run the exported graph on one RGB image.
    pub fn forward_image(&self, img: &RgbImage) -> Result<RgbImage> {
        let (data, h, w) = rgb_to_chw(img);
        let out = self.forward_chw(&data, h, w)?;
        Ok(chw_to_rgb(&out, h, w))
    }

    /// Run the exported graph on a [3][H][W] f32 image in [0, 1].
    pub fn forward_chw(&self, input: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let m = &self.manifest;
        if input.len() != 3 * height * width {
            bail!(
                "expected {} input values, got {}",
                3 * height * width,
                input.len()
            );
        }
        let by_name: HashMap<&str, &QuantLayer> =
            m.layers.iter().map(|l| (l.name.as_str(), l)).collect();

        let point_count = m.points.len();
        let mut points: Vec<Option<Act>> = (0..point_count).map(|_| None).collect();
        let mut image: Option<Act> = None;

        // The input image is quantised at point 0 with the calibrated scale.
        let s0 = m.point_scales[0];
        let qmax = m.act_qmax;
        let mut data = vec![0i32; 3 * height * width];
        for (dst, src) in data.iter_mut().zip(input) {
            *dst = round_clamp(*src / s0, -qmax, qmax);
        }
        points[0] = Some(Act {
            c: 3,
            h: height,
            w: width,
            data,
        });

        for (index, op) in m.graph.iter().enumerate() {
            match op.op.as_str() {
                "conv" => {
                    let name = op
                        .layer
                        .as_deref()
                        .with_context(|| format!("graph[{index}]: conv without a layer"))?;
                    let layer = by_name
                        .get(name)
                        .with_context(|| format!("graph[{index}]: unknown layer {name}"))?;
                    let src = points
                        .get(layer.in_point)
                        .and_then(|p| p.as_ref())
                        .with_context(|| {
                            format!(
                                "graph[{index}]: point {} is not live for {name}",
                                layer.in_point
                            )
                        })?;
                    let out = self.conv(layer, src);
                    if layer.out_unsigned {
                        image = Some(out);
                    } else if layer.out_point < point_count {
                        points[layer.out_point] = Some(out);
                    } else {
                        bail!(
                            "graph[{index}]: {name} writes point {} out of range",
                            layer.out_point
                        );
                    }
                }
                // The activation range of a conv2d+relu layer is already [0, qmax]; the
                // separate relu op is for the hardware scheduler and is a no-op here.
                "relu" => {}
                "add" => {
                    let point = op
                        .point
                        .with_context(|| format!("graph[{index}]: add without a point"))?;
                    let source = op
                        .source_point
                        .with_context(|| format!("graph[{index}]: add without a source_point"))?;
                    let branch =
                        points
                            .get(source)
                            .and_then(|p| p.as_ref())
                            .with_context(|| {
                                format!("graph[{index}]: source point {source} is not live")
                            })?;
                    let trunk = points
                        .get(point)
                        .and_then(|p| p.as_ref())
                        .with_context(|| format!("graph[{index}]: point {point} is not live"))?;
                    // Both operands land on the same grid before the add: the convolution
                    // result already sits on it, the branch input is requantised onto it.
                    let factor = m.point_scales[source] / m.point_scales[point];
                    let qmax = m.act_qmax;
                    let mut data = vec![0i32; trunk.data.len()];
                    for ((dst, a), b) in data.iter_mut().zip(&branch.data).zip(&trunk.data) {
                        *dst = round_clamp(*a as f32 * factor, -qmax, qmax) + *b;
                    }
                    points[point] = Some(Act {
                        c: trunk.c,
                        h: trunk.h,
                        w: trunk.w,
                        data,
                    });
                }
                "upsample_nearest2x" => {
                    let point = op
                        .point
                        .with_context(|| format!("graph[{index}]: upsample without a point"))?;
                    let src = points
                        .get(point)
                        .and_then(|p| p.as_ref())
                        .with_context(|| format!("graph[{index}]: point {point} is not live"))?;
                    points[point] = Some(upsample2x(src));
                }
                "quantize_u8_output" => {
                    let codes = image.take().with_context(|| {
                        format!("graph[{index}]: no image produced before the output op")
                    })?;
                    let scale = op.scale.unwrap_or(1.0 / 255.0);
                    let out: Vec<f32> = codes.data.iter().map(|c| *c as f32 * scale).collect();
                    return Ok(out);
                }
                other => bail!("graph[{index}]: unknown op {other}"),
            }
        }
        bail!("the graph never reached a quantize_u8_output step")
    }

    /// One integer convolution, the whole datapath: reflect pad, int32 accumulate, requant.
    fn conv(&self, layer: &QuantLayer, input: &Act) -> Act {
        let pad = layer.reflect_pad;
        let (kh, kw) = (layer.kernel[0], layer.kernel[1]);
        let (sh, sw) = (layer.stride[0], layer.stride[1]);
        let padded = reflect_pad(input, pad);
        let out_h = (padded.h - kh) / sh + 1;
        let out_w = (padded.w - kw) / sw + 1;
        let ic_n = layer.in_channels;
        let oc_n = layer.out_channels;
        // The requantisation clamp is the quantiser's saturation limit, and it is what
        // implements the ReLU for a conv2d+relu layer: rounding a symmetric grid that contains
        // zero commutes with max(., 0), so clamping to [0, qmax] is exactly relu-then-quantise.
        let (lo, hi) = if layer.out_unsigned {
            (0.0, self.manifest.output_qmax)
        } else if layer.out_relu {
            (0.0, self.manifest.act_qmax)
        } else {
            (-self.manifest.act_qmax, self.manifest.act_qmax)
        };
        let phw = padded.h * padded.w;
        let plane = out_h * out_w;
        let mut out = vec![0i32; oc_n * plane];

        out.par_chunks_mut(plane).enumerate().for_each(|(oc, dst)| {
            let base = layer.weight_offset + oc * ic_n * kh * kw;
            let requant = layer.requant[oc];
            let bias = layer.bias_int32[oc] as i64;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = bias;
                    for ic in 0..ic_n {
                        let src = ic * phw;
                        let wgt = base + ic * kh * kw;
                        for ky in 0..kh {
                            let row = (oy * sh + ky) * padded.w + ox * sw;
                            for kx in 0..kw {
                                let a = padded.data[src + row + kx] as i64;
                                let b = self.weights[wgt + ky * kw + kx] as i64;
                                acc += a * b;
                            }
                        }
                    }
                    dst[oy * out_w + ox] = round_clamp(acc as f32 * requant, lo, hi);
                }
            }
        });

        Act {
            c: oc_n,
            h: out_h,
            w: out_w,
            data: out,
        }
    }
}

/// Round half away from zero and clamp, matching the backends round() and the requant clamp.
fn round_clamp(value: f32, lo: f32, hi: f32) -> i32 {
    value.round().clamp(lo, hi) as i32
}

/// Reflection padding, as PyTorch and burn define it: the edge is not repeated.
fn reflect_pad(a: &Act, pad: usize) -> Act {
    if pad == 0 {
        return a.clone();
    }
    let h = a.h + 2 * pad;
    let w = a.w + 2 * pad;
    let mut data = vec![0i32; a.c * h * w];
    for c in 0..a.c {
        for y in 0..h {
            let sy = reflect_index(y as isize - pad as isize, a.h);
            for x in 0..w {
                let sx = reflect_index(x as isize - pad as isize, a.w);
                data[(c * h + y) * w + x] = a.data[(c * a.h + sy) * a.w + sx];
            }
        }
    }
    Act { c: a.c, h, w, data }
}

fn reflect_index(index: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as isize;
    let mut i = index;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        } else {
            i = 2 * (n - 1) - i;
        }
    }
    i as usize
}

/// Nearest-neighbour 2x upsampling: each code is duplicated into a 2x2 block.
fn upsample2x(a: &Act) -> Act {
    let h = a.h * 2;
    let w = a.w * 2;
    let mut data = vec![0i32; a.c * h * w];
    for c in 0..a.c {
        for y in 0..h {
            for x in 0..w {
                data[(c * h + y) * w + x] = a.data[(c * a.h + y / 2) * a.w + x / 2];
            }
        }
    }
    Act { c: a.c, h, w, data }
}

/// RGB8 to a [3][H][W] f32 plane in [0, 1].
pub fn rgb_to_chw(img: &RgbImage) -> (Vec<f32>, usize, usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data[(c * h + y) * w + x] = img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
            }
        }
    }
    (data, h, w)
}

/// The inverse of [rgb_to_chw], clamping to [0, 1].
pub fn chw_to_rgb(data: &[f32], h: usize, w: usize) -> RgbImage {
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let mut px = [0u8; 3];
            for (c, slot) in px.iter_mut().enumerate() {
                *slot = (data[(c * h + y) * w + x].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            img.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    img
}
