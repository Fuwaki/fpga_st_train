//! Static int8 quantisation for the student: range calibration, fake-quantised (QAT)
//! forward passes, and the export a fixed-point accelerator consumes.
//!
//! # Scheme
//!
//! - **Activations**: symmetric int8, one scale per *tensor*. `x_q = clamp(round(x/s), -127, 127)`;
//!   the dequantised value is `x_q * s`. No zero point, so the datapath needs no offset adder.
//! - **Weights**: symmetric int8, one scale per *output channel*. This is what keeps a 3x3
//!   convolution accurate without per-element scaling: the whole filter row shares one
//!   multiplier.
//! - **Bias**: kept in int32 at export (`round(b / (s_x * s_w))`), so the accumulator stays
//!   integer until the requantisation step.
//! - **Accumulator**: int32. The widest case here is 3x3x64 = 576 products of 127*127, i.e.
//!   ~9.3e6, far below 2^31.
//! - **Residual add**: both operands are requantised onto the *same* grid (the scale of the
//!   next consumer) before the add, as an integer accelerator must.
//!
//! Rounding uses a straight-through estimator, so the very same code path runs in fp32
//! training (gradients flow as if nothing happened) and as the integer simulation used for
//! evaluation. There is deliberately no separate "quantised model" implementation to drift
//! out of sync with the trained one.

use burn::prelude::*;
use serde::{Deserialize, Serialize};

/// Bit widths of one quantisation scheme.
///
/// - **internal activations are symmetric**: `acts` bits give the range
///   `[-2^(acts-1)+1, 2^(acts-1)-1]` (127 for 8 bits) with zero point 0, so no datapath adder
///   needs an offset.
/// - **weights are symmetric**: `weights` bits give the same signed range (127 for 8 bits,
///   7 for 4, 1 for 2 -- ternary).
/// - **the output image is unsigned**: the head's result *is* an image, so it is the one
///   tensor that uses the whole `[0, 2^act_bits - 1]` codebook (`out_unsigned` in the export).
///   That costs nothing and avoids throwing away half the range of a value that can never be
///   negative.
///
/// Making *every* activation unsigned (every quantised point here is post-ReLU) is a tempting
/// further step and is not usable yet; the attempt is written up in
/// `runs/note/v3-int8-qat.md`. This struct describes the validated symmetric path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bits {
    /// Bits per activation.
    pub acts: u8,
    /// Bits per weight.
    pub weights: u8,
}

impl Default for Bits {
    fn default() -> Self {
        Self {
            acts: 8,
            weights: 8,
        }
    }
}

impl Bits {
    pub fn new(acts: u8, weights: u8) -> Self {
        Self { acts, weights }
    }

    /// Largest magnitude representable with a symmetric `bits`-bit code.
    pub fn max_value(bits: u8) -> f32 {
        debug_assert!((2..=16).contains(&bits), "unsupported bit width {bits}");
        ((1i32 << (bits - 1)) - 1) as f32
    }

    /// Largest code of an unsigned `bits`-bit tensor: `2^bits - 1`.
    pub fn unsigned_max(bits: u8) -> f32 {
        debug_assert!((2..=16).contains(&bits), "unsupported bit width {bits}");
        ((1i32 << bits) - 1) as f32
    }

    /// Largest magnitude for an internal activation: symmetric, so `2^(acts-1) - 1`.
    ///
    /// The output image is the one exception and uses [Self::unsigned_max]; see [Self].
    pub fn act_max(&self) -> f32 {
        Self::max_value(self.acts)
    }

    /// Largest magnitude for a weight: symmetric, so `2^(weights-1) - 1`.
    pub fn weight_max(&self) -> f32 {
        Self::max_value(self.weights)
    }

    /// e.g. `W4A8`.
    pub fn label(&self) -> String {
        format!("W{}A{}", self.weights, self.acts)
    }

    /// Parse a label such as `W4A8` or `w4a8` (the leading W is optional).
    ///
    /// The letter order is the same as [Self::label], so a label round-trips.
    pub fn parse(text: &str) -> Result<Self, String> {
        let lowered = text.trim().to_ascii_lowercase();
        let body = lowered.strip_prefix('w').unwrap_or(&lowered);
        let (weights, acts) = body
            .split_once('a')
            .ok_or_else(|| format!("expected W<weights>A<acts> such as W4A8, got {text:?}"))?;
        let bad =
            |what: &str, value: &str| format!("{value:?} is not a valid {what} width in {text:?}");
        let weights: u8 = weights.parse().map_err(|_| bad("weight", weights))?;
        let acts: u8 = acts.parse().map_err(|_| bad("activation", acts))?;
        if !(2..=16).contains(&weights) {
            return Err(format!(
                "weight width {weights} in {text:?} is outside 2..=16"
            ));
        }
        if !(2..=16).contains(&acts) {
            return Err(format!(
                "activation width {acts} in {text:?} is outside 2..=16"
            ));
        }
        Ok(Self { acts, weights })
    }
}

/// Everything the fake-quantised forward pass needs: bit widths plus activation scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantScheme {
    #[serde(default)]
    pub bits: Bits,
    /// One symmetric activation scale per quantised point.
    pub acts: Vec<f32>,
}

impl QuantScheme {
    pub fn act_max(&self) -> f32 {
        self.bits.act_max()
    }

    pub fn weight_max(&self) -> f32 {
        self.bits.weight_max()
    }
}

/// `round(x)` with an identity backward pass (straight-through estimator).
///
/// Forward: `x + round(x) - x = round(x)`. Backward: `dx + 0 - 0 = dx`, because both the
/// rounded value and the copy of `x` inside the correction are detached.
pub fn ste_round<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let rounded = x.clone().round();
    x.clone() + (rounded.detach() - x.detach())
}

/// Symmetric fake quantisation of one activation onto an explicit scale.
pub fn fake_quant_act<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    scale: f32,
    qmax: f32,
) -> Tensor<B, D> {
    let inverse = 1.0 / scale.max(1e-12);
    ste_round(x.mul_scalar(inverse))
        .clamp(-qmax, qmax)
        .mul_scalar(scale)
}

/// The final image the accelerator emits: uint8, scale 1/255, zero point 0.
///
/// This *is* part of the simulated forward pass, so that the simulation and the exported
/// manifest describe the same graph. Without it the simulation would quietly skip the output
/// rounding and the [0, 1] clamp and would overstate how close the deployed model is to its
/// fp32 twin.
///
/// Semantics, in hardware terms: the head's int32 accumulator is multiplied by
/// `requant[c]` and rounded to an integer code in `[0, 255]`; the image value is that code
/// times 1/255. Clamping happens *after* the multiply, exactly as the datapath does it.
pub fn fake_quant_output<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    ste_round(x.clamp(0.0, 1.0).mul_scalar(255.0)).div_scalar(255.0)
}

/// Symmetric fake quantisation of a convolution weight, one scale per output channel.
pub fn fake_quant_weight<B: Backend>(weight: Tensor<B, 4>, qmax: f32) -> Tensor<B, 4> {
    let [out_ch, in_ch, kh, kw] = weight.dims();
    // The scale is derived from the current weights but must not receive gradient.
    let amax = weight
        .clone()
        .abs()
        .reshape([out_ch, in_ch * kh * kw])
        .max_dim(1)
        .detach();
    let scale = amax
        .div_scalar(qmax)
        .clamp_min(1e-8)
        .reshape([out_ch, 1, 1, 1]);
    let codes = ste_round(weight.div(scale.clone())).clamp(-qmax, qmax);
    codes * scale
}

/// Integer codes and per-channel scales for one weight tensor, computed on the host.
///
/// Returns `(codes, scales)` with `codes` laid out as `[out_ch][in_ch][kh][kw]` exactly like
/// the fp32 weight, ready to be written out for the accelerator.
pub fn quantise_weights_host(weight: &[f32], out_ch: usize, qmax: f32) -> (Vec<i8>, Vec<f32>) {
    let per_channel = weight.len() / out_ch;
    let mut codes = vec![0i8; weight.len()];
    let mut scales = vec![0.0f32; out_ch];
    for channel in 0..out_ch {
        let slice = &weight[channel * per_channel..(channel + 1) * per_channel];
        let amax = slice.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = (amax / qmax).max(1e-8);
        scales[channel] = scale;
        for (dst, src) in codes[channel * per_channel..(channel + 1) * per_channel]
            .iter_mut()
            .zip(slice)
        {
            *dst = (src / scale).round().clamp(-qmax, qmax) as i8;
        }
    }
    (codes, scales)
}

/// Activation ranges measured by the `calibrate` subcommand, one entry per quantised point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// Identifies the model layout these scales belong to.
    pub arch: String,
    /// Human-readable name of every quantised point, same order as `scales`.
    pub points: Vec<String>,
    /// Symmetric int8 activation scales (`absmax / 127`).
    pub scales: Vec<f32>,
    /// Raw observed absolute maxima, before the safety factor.
    pub absmax: Vec<f32>,
    /// Multiplier applied to `absmax` before dividing by the code maximum.
    pub safety: f32,
    /// Number of content batches the ranges were measured on.
    pub batches: usize,
    /// Bit widths these scales were calibrated for.
    #[serde(default)]
    pub bits: Bits,
}

impl Calibration {
    /// Check that scales match the model's quantised activation layout.
    pub fn validate_for<B: Backend, M: crate::quant::Quantizable<B>>(
        &self,
        model: &M,
    ) -> anyhow::Result<()> {
        let points = model.quant_points();
        if points.is_empty() {
            anyhow::bail!("model does not implement quantisation (empty quant_points)");
        }
        if points.len() != self.scales.len() {
            anyhow::bail!(
                "calibration has {} scales but the model has {} quantised points",
                self.scales.len(),
                points.len()
            );
        }
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("cannot parse {}: {e}", path.display()))
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// The scales as a plain slice, for the forward pass.
    pub fn as_slice(&self) -> &[f32] {
        &self.scales
    }

    /// Everything the fake-quantised forward pass needs.
    pub fn scheme(&self) -> QuantScheme {
        QuantScheme {
            bits: self.bits,
            acts: self.scales.clone(),
        }
    }
}

fn default_groups() -> usize {
    1
}

/// One convolution of the exported integer graph, in execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantLayer {
    /// Stable name, e.g. `enc.0` or `res.2.c1`.
    pub name: String,
    /// `conv2d` or `conv2d+relu`.
    pub op: String,
    pub in_channels: usize,
    pub out_channels: usize,
    /// 1 for a dense convolution; `in_channels` for depthwise.
    #[serde(default = "default_groups")]
    pub groups: usize,
    pub act_bits: u8,
    pub weight_bits: u8,
    /// Always 0: the code range is symmetric about zero (or `[0, 2^bits-1]` when
    /// `out_unsigned`, which is still zero-point-free in the sense that no offset is added).
    pub act_zero_point: i32,
    /// True when a ReLU sits between the accumulator and the requantisation, so the output
    /// codes are clamped to `[0, act_qmax]` instead of `[-act_qmax, act_qmax]`.
    ///
    /// Every layer but the residual `c2` convolutions and the head sets this.
    pub out_relu: bool,
    /// True when this layer's *output* codes are unsigned `[0, 2^act_bits - 1]` instead of
    /// symmetric `[-2^(act_bits-1)+1, 2^(act_bits-1)-1]`.
    ///
    /// Only the head sets this: its result is the image, so clamping it symmetrically would
    /// clip everything brighter than 127/255. Takes precedence over [Self::out_relu], which is
    /// false for the head anyway.
    pub out_unsigned: bool,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    /// Reflection padding applied before the (valid) convolution.
    pub reflect_pad: usize,
    /// Index into `points` of this layer's quantised input.
    pub in_point: usize,
    /// Index into `points` of the activation this layer feeds.
    pub out_point: usize,
    pub in_scale: f32,
    pub out_scale: f32,
    /// Per-output-channel dequantisation scale of the weight.
    pub weight_scale: Vec<f32>,
    /// fp32 bias, for reference; `bias_int32` is what the hardware uses.
    pub bias: Vec<f32>,
    /// `round(bias / (in_scale * weight_scale))`, the value added to the int32 accumulator.
    pub bias_int32: Vec<i32>,
    /// `in_scale * weight_scale[c] / out_scale`: multiply the accumulator by this, round and
    /// clamp to int8 to get the output activation codes.
    pub requant: Vec<f32>,
    /// Offset of this layer's codes inside the *unpacked* code stream.
    ///
    /// Use [Self::weight_byte_offset] to address `weights.bin`: at 4 bits two codes share a
    /// byte, so the two offsets differ.
    pub weight_offset: usize,
    /// Number of weight codes in this layer.
    pub weight_count: usize,
    /// Offset of this layer's codes inside `weights.bin`, in bytes.
    pub weight_byte_offset: usize,
    /// Bytes this layer occupies in `weights.bin`:
    /// `weight_count * weight_bits / 8`.
    pub weight_bytes: usize,
}

/// A step of the exported graph.
///
/// Convolutions appear here too, so the whole datapath is one ordered list. Non-convolution
/// steps carry the extra operand they need and nothing else.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantOp {
    /// One of conv, relu, add, upsample_nearest2x, quantize_u8_output.
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point: Option<usize>,
    /// For an add: the second operand's quantised point (the residual branch input).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_point: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
}

/// Everything the accelerator needs, alongside `weights.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantManifest {
    pub format: String,
    pub arch: String,
    /// Bit widths of the exported graph.
    pub bits: Bits,
    /// Largest magnitude of an internal activation code (symmetric, zero point 0).
    pub act_qmax: f32,
    /// Largest activation code of the unsigned output image (`2^act_bits - 1`).
    pub output_qmax: f32,
    /// Largest weight magnitude (weights are signed).
    pub weight_qmax: f32,
    pub notes: Vec<String>,
    /// Name of every quantised activation point. `points[0]` is the input image.
    pub points: Vec<String>,
    pub point_scales: Vec<f32>,
    /// Execution order: conv / relu / upsample / add.
    pub graph: Vec<QuantOp>,
    pub layers: Vec<QuantLayer>,
    pub weight_file: String,
    /// How codes are packed in `weights.bin`.
    pub weight_packing: String,
    /// Size of `weights.bin` in bytes.
    pub weight_bytes: usize,
}

/// Pack weight codes into bytes.
///
/// 8-bit schemes store one code per byte (two's complement). Narrower schemes pack
/// `8 / bits` codes per byte, least-significant code first: for 4 bits the first code of a
/// pair lives in the low nibble, the second in the high nibble, and each nibble is two's
/// complement (`0xF` is -1). That is the layout an FPGA BRAM read gives you for free.
pub fn pack_weight_codes(codes: &[i8], bits: u8) -> Vec<u8> {
    match bits {
        8 => codes.iter().map(|code| *code as u8).collect(),
        4 => codes
            .chunks(2)
            .map(|pair| {
                let low = (pair[0] as u8) & 0x0f;
                let high = pair.get(1).map(|c| ((*c as u8) & 0x0f) << 4).unwrap_or(0);
                low | high
            })
            .collect(),
        other => panic!("no packing defined for {other}-bit weights"),
    }
}

impl QuantManifest {
    pub fn save(&self, dir: &std::path::Path, weights: &[i8]) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let bytes = pack_weight_codes(weights, self.bits.weights);
        std::fs::write(dir.join(&self.weight_file), &bytes)?;
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(self)?,
        )?;
        Ok(())
    }

    /// Read back `weights.bin` next to this manifest.
    ///
    /// This is the inverse of [QuantManifest::save] and is what the reference interpreter
    /// (see `crate::intref`) and any FPGA bring-up script should use.
    pub fn load_weights(&self, dir: &std::path::Path) -> anyhow::Result<Vec<i8>> {
        let raw = std::fs::read(dir.join(&self.weight_file)).map_err(|e| {
            anyhow::anyhow!("cannot read {}: {e}", dir.join(&self.weight_file).display())
        })?;
        let per_byte = (8 / self.bits.weights) as usize;
        let total: usize = self.layers.iter().map(|l| l.weight_count).sum();
        let mut codes = vec![0i8; total];
        for layer in &self.layers {
            let start = layer.weight_byte_offset;
            let end = start + layer.weight_bytes;
            if end > raw.len() {
                anyhow::bail!(
                    "{} is {} bytes but layer {} claims bytes {}..{}",
                    self.weight_file,
                    raw.len(),
                    layer.name,
                    start,
                    end
                );
            }
            // Codes are laid out in the order the layer weights were written, so walking the
            // bytes and emitting low code first reconstructs the original stream.
            let mut decoded = 0usize;
            for byte in &raw[start..end] {
                for k in 0..per_byte {
                    let shift = self.bits.weights as u32 * k as u32;
                    let mask = (1u16 << self.bits.weights) - 1;
                    let value = ((*byte as u16) >> shift) & mask;
                    let sign_bit = 1u16 << (self.bits.weights - 1);
                    let signed = if value & sign_bit != 0 {
                        (value as i32) - (1i32 << self.bits.weights)
                    } else {
                        value as i32
                    };
                    if layer.weight_offset + decoded >= codes.len() {
                        anyhow::bail!(
                            "layer {} decodes past the end of the code stream",
                            layer.name
                        );
                    }
                    codes[layer.weight_offset + decoded] = signed as i8;
                    decoded += 1;
                }
            }
            if decoded != layer.weight_count {
                anyhow::bail!(
                    "layer {}: decoded {} codes but weight_count is {}",
                    layer.name,
                    decoded,
                    layer.weight_count
                );
            }
        }
        Ok(codes)
    }
}
