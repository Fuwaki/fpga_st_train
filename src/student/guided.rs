//! Low-resolution trunk with a full-resolution RGB guide.
//!
//! The separable SOTA (`tiny_dw`, `film_both`) spends its content skip at H/2: 64 channels
//! that have to sit in a delay buffer until the decoder climbs back. Style does not need
//! that map. Colour is global (InstanceNorm, FiLM), brush texture is a mid-scale grid, and
//! the only full-resolution fact that still matters is where the edges are.
//!
//! This student throws the content away first. A 2×2 average pool feeds the stem, two
//! stride-2 stages land the trunk at H/8, and three nearest upsamples climb back. The H/2
//! skip is gone. At full resolution the original RGB is read again through a 3×3 (3→`guide`
//! channels) and added to the upsampled features, the same split HDRnet uses: the low-res
//! net decides the paint, the full-res image decides where it sits.
//!
//! `trunk_at = h4` keeps `enc1` but drops its stride, so the residual and InstanceNorm
//! stay on the H/4 map and `dec_h4` does not upsample first. Default `h8` is the graph above.
//!
//! `full = 0` keeps only that guide and one 3×3 head at full resolution, and spends the
//! width at H/2. `full > 0` inserts one more separable 3×3 of that width before a 1×1 RGB
//! head. Integer export covers the white-box line: H/4 trunk, H/2 FiLM, full-resolution
//! guide, and either InstanceNorm or LayerNorm. Shuffle, a full-resolution stack, and the
//! other guide mixes stay fp32.

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::{avg_pool2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

use super::plugin::{Film, Plugin, PluginKind};
use super::tiny::{Conv, ConvInit};
use super::tiny_dw::{PwConv, ResBlockDw, SepConv};
use super::{SampleSignal, StudentModel};
use crate::config::StudentConfig;
use crate::export::{self, ConvRef, ExportStep, Exportable};
use crate::quant::{self, QuantScheme};

#[path = "guided_prune.rs"]
mod prune;

pub use prune::{PruneReport, PruneWidths};

/// Gain on the convolutions that feed the trunk InstanceNorm.
///
/// `Default` keeps Burn's `1/√3`. `Sqrt2` is the ReLU fan-in gain. Decoder
/// layers, the guides, and FiLM stay on the default either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreInGain {
    Default,
    Sqrt2,
}

impl PreInGain {
    fn conv_init(self) -> ConvInit {
        match self {
            PreInGain::Default => ConvInit::Default,
            PreInGain::Sqrt2 => ConvInit::Relu,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PreInGain::Default => "default",
            PreInGain::Sqrt2 => "sqrt2",
        }
    }
}

/// Widths for one guided student. `full = 0` is the H/2-heavy decoder.
#[derive(Clone, Debug)]
pub struct GuideSpec {
    pub base: usize,
    pub trunk: usize,
    pub res_blocks: usize,
    pub up_h4: usize,
    pub up_h2: usize,
    pub guide: usize,
    pub full: usize,
    /// 1 keeps a single full-resolution separable layer. 2 stacks a second one.
    pub full_layers: usize,
    /// Pixel-shuffle in place of nearest on the trunk (H/8 and H/4).
    /// Identity weights reproduce nearest.
    pub shuffle: bool,
    /// Pixel-shuffle on the H/2 → full step only. Follows `shuffle` when the
    /// config omits `upsample_guide`.
    pub shuffle_guide: bool,
    pub film_h2: bool,
    /// FiLM on the full-resolution guide features, before the texture layers.
    pub film_guide: bool,
    pub film_full: bool,
    /// ReLU after the full-resolution RGB guide, before it is added.
    /// Off leaves that tensor signed.
    pub guide_relu: bool,
    /// Channels of a 3×3 on the pooled RGB, added to the H/2 decoder output.
    /// Zero disables it. A 1×1 aligns the width when it differs from `up_h2`.
    pub h2_guide: usize,
    /// Kernel of the full-resolution RGB guide. 3 or 5.
    pub guide_kernel: usize,
    /// `paint + paint ⊙ guide` instead of `paint + guide`.
    pub guide_gate: bool,
    /// `paint + guide + paint ⊙ g`. `g` is a zero-init 1×1 on the paint, so
    /// step 0 matches `add`. The guide tensor is not the multiplier.
    pub guide_resid: bool,
    /// Residual and InstanceNorm stay at H/4. `enc1` stride is 1, and
    /// `dec_h4` reads that map directly. Default is the H/8 trunk.
    pub trunk_h4: bool,
    /// Epsilon inside the trunk InstanceNorm. `1e-5` matches every other norm.
    pub in_eps: f32,
    /// Trunk norm kind: "in" (InstanceNorm, default) or "ln" (LayerNorm).
    pub trunk_norm_kind: String,
    /// Init gain for stem, enc0, enc1, and the residual blocks.
    pub pre_in_gain: PreInGain,
    /// Channels of the content image. `3` is RGB. `2` is BT.601 luma plus the
    /// 135° chroma axis, `C = (Cr − Cb) / √2`. The conversion happens inside
    /// the forward when a 3-channel tensor is given, so the teacher still sees RGB.
    pub input_planes: usize,
    /// `none` or `rgb565`. The 565 step is inside the student forward, so the
    /// teacher still receives the unquantised RGB batch.
    pub input_quant: String,
    /// Style packets. `1` is the single trunk norm and H/2 FiLM. Each extra
    /// style copies that pair and leaves every convolution shared.
    pub styles: usize,
    /// Each style, style 0 included, carries a per-channel `exp(log_gain)`, `bias`
    /// on the full-resolution guide before its ReLU. The gain stays positive so the
    /// fold into the guide kernel keeps one set of int8 weight codes for every style.
    pub packet_guide: bool,
}

/// Inverse of [`RGB_FROM_YCBCR`]. Rows are Y, Cb, Cr. Same numbers as
/// `numpy.linalg.inv` of the matrix in `scripts/fold_stem_2ch.py`.
pub(crate) const YCBCR_FROM_RGB: [[f64; 3]; 3] = [
    [0.298_999_944_348, 0.587_000_125_992, 0.113_999_929_66],
    [-0.168_735_860_241, -0.331_264_179_454, 0.500_000_039_695],
    [0.500_000_039_695, -0.418_687_679_024, -0.081_312_360_671],
];

/// `rgb = M @ [Y, Cb, Cr]` with centered chroma. Columns are Y, Cb, Cr.
pub(crate) const RGB_FROM_YCBCR: [[f64; 3]; 3] = [
    [1.0, 0.0, 1.402],
    [1.0, -0.344_136, -0.714_136],
    [1.0, 1.772, 0.0],
];

fn coeff(value: f64) -> f32 {
    value as f32
}

/// `C = (Cr − Cb) / √2`, the 135° chroma axis.
fn rgb_to_yc<B: Backend>(rgb: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, _, h, w] = rgb.dims();
    let r = rgb.clone().slice([0..n, 0..1, 0..h, 0..w]);
    let g = rgb.clone().slice([0..n, 1..2, 0..h, 0..w]);
    let b = rgb.slice([0..n, 2..3, 0..h, 0..w]);
    let row = |i: usize| {
        r.clone() * coeff(YCBCR_FROM_RGB[i][0])
            + g.clone() * coeff(YCBCR_FROM_RGB[i][1])
            + b.clone() * coeff(YCBCR_FROM_RGB[i][2])
    };
    let y = row(0);
    let cb = row(1);
    let cr = row(2);
    let c = (cr - cb) * std::f32::consts::FRAC_1_SQRT_2;
    Tensor::cat(vec![y, c], 1)
}

/// Quantise RGB in `[0, 1]` onto 5/6/5 bits. R and B use 32 codes, G uses 64.
fn quantize_rgb565<B: Backend>(rgb: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = rgb.dims();
    if c != 3 {
        return rgb;
    }
    let rgb = rgb.clamp(0.0, 1.0);
    let channel =
        |tensor: Tensor<B, 4>, levels: f32| tensor.mul_scalar(levels).round().div_scalar(levels);
    let r = channel(rgb.clone().slice([0..n, 0..1, 0..h, 0..w]), 31.0);
    let g = channel(rgb.clone().slice([0..n, 1..2, 0..h, 0..w]), 63.0);
    let b = channel(rgb.slice([0..n, 2..3, 0..h, 0..w]), 31.0);
    Tensor::cat(vec![r, g, b], 1)
}

fn parse_upsample(name: &str, value: &str) -> Result<bool, String> {
    match value {
        "nearest" => Ok(false),
        "shuffle" => Ok(true),
        other => Err(format!("{name} must be nearest or shuffle, got {other}")),
    }
}

impl GuideSpec {
    pub fn from_config(config: &StudentConfig) -> Result<Self, String> {
        let shuffle = parse_upsample("upsample", config.str_option("upsample", "nearest"))?;
        let shuffle_guide = match config.str_option("upsample_guide", "") {
            "" => shuffle,
            other => parse_upsample("upsample_guide", other)?,
        };
        let (guide_gate, guide_resid) = match config.str_option("guide_mix", "add") {
            "add" => (false, false),
            "gate" => (true, false),
            "resid" => (false, true),
            other => {
                return Err(format!(
                    "guide_mix must be add, gate, or resid, got {other}"
                ));
            }
        };
        let trunk_h4 = match config.str_option("trunk_at", "h8") {
            "h8" => false,
            "h4" => true,
            other => {
                return Err(format!(
                    "tiny_dw_guided trunk_at must be h8 or h4, got {other}"
                ));
            }
        };
        let spec = Self {
            base: config.usize_option("base", 32),
            trunk: config.usize_option("trunk", 64),
            res_blocks: config.usize_option("res_blocks", 1),
            up_h4: config.usize_option("up_h4", 32),
            up_h2: config.usize_option("up_h2", 24),
            guide: config.usize_option("guide", 8),
            full: config.usize_option("full", 16),
            full_layers: config.usize_option("full_layers", 1),
            shuffle,
            shuffle_guide,
            film_h2: config.usize_option("film_h2", 1) != 0,
            film_guide: config.usize_option("film_guide", 0) != 0,
            film_full: config.usize_option("film_full", 1) != 0,
            guide_relu: config.usize_option("guide_relu", 1) != 0,
            h2_guide: config.usize_option("h2_guide", 0),
            guide_kernel: config.usize_option("guide_kernel", 3),
            guide_gate,
            guide_resid,
            trunk_h4,
            in_eps: config.f32_option("in_eps", 1e-5),
            trunk_norm_kind: config.str_option("trunk_norm_kind", "in").to_string(),
            pre_in_gain: match config.str_option("pre_in_gain", "default") {
                "default" => PreInGain::Default,
                "sqrt2" => PreInGain::Sqrt2,
                other => {
                    return Err(format!(
                        "tiny_dw_guided pre_in_gain must be default or sqrt2, got {other}"
                    ));
                }
            },
            input_planes: match config.str_option("input", "rgb") {
                "rgb" => 3,
                "yc" => 2,
                other => {
                    return Err(format!(
                        "tiny_dw_guided input must be rgb or yc, got {other}"
                    ));
                }
            },
            input_quant: config.str_option("input_quant", "none").to_string(),
            styles: config.usize_option("styles", 1),
            packet_guide: config.usize_option("packet_guide", 0) != 0,
        };
        spec.check()?;
        Ok(spec)
    }

    fn check(&self) -> Result<(), String> {
        for (name, value) in [
            ("base", self.base),
            ("trunk", self.trunk),
            ("up_h4", self.up_h4),
            ("up_h2", self.up_h2),
            ("guide", self.guide),
        ] {
            if value == 0 {
                return Err(format!("tiny_dw_guided {name} must be > 0"));
            }
        }
        if self.full_layers != 1 && self.full_layers != 2 {
            return Err(format!(
                "tiny_dw_guided full_layers must be 1 or 2, got {}",
                self.full_layers
            ));
        }
        if self.full == 0 && self.full_layers == 2 {
            return Err("full_layers = 2 requires full > 0".to_owned());
        }
        if self.full == 0 && self.film_full {
            return Err("film_full requires full > 0".to_owned());
        }
        if self.guide_kernel != 3 && self.guide_kernel != 5 {
            return Err(format!(
                "tiny_dw_guided guide_kernel must be 3 or 5, got {}",
                self.guide_kernel
            ));
        }
        if self.guide_gate && self.guide_resid {
            return Err("guide_gate and guide_resid are different mixes".to_owned());
        }
        if !(self.in_eps.is_finite() && self.in_eps > 0.0) {
            return Err(format!(
                "tiny_dw_guided in_eps must be positive, got {}",
                self.in_eps
            ));
        }
        if self.trunk_norm_kind != "in" && self.trunk_norm_kind != "ln" {
            return Err(format!(
                "tiny_dw_guided trunk_norm_kind must be 'in' or 'ln', got {}",
                self.trunk_norm_kind
            ));
        }
        if self.input_planes != 2 && self.input_planes != 3 {
            return Err(format!(
                "tiny_dw_guided input_planes must be 2 or 3, got {}",
                self.input_planes
            ));
        }
        if self.input_quant != "none" && self.input_quant != "rgb565" {
            return Err(format!(
                "tiny_dw_guided input_quant must be none or rgb565, got {}",
                self.input_quant
            ));
        }
        if self.input_quant == "rgb565" && self.input_planes != 3 {
            return Err("rgb565 quantisation needs a 3-channel RGB input".to_owned());
        }
        if self.styles == 0 {
            return Err("tiny_dw_guided styles must be >= 1".to_owned());
        }
        if self.styles > 1 && !self.film_h2 {
            return Err("style packets need the H/2 FiLM".to_owned());
        }
        Ok(())
    }

    /// MAC and A8 line-buffer account at one deployment size.
    ///
    /// Same MAC rule as `runs/note/v7-input-precision.md` §5: dense is `cin·cout·k²`
    /// per output pixel, depthwise is `cin·k²`. A 3×3 line buffer is two input rows.
    pub fn cost(&self, height: usize, width: usize) -> CostReport {
        let mut rows = Vec::new();
        let h2 = height / 2;
        let w2 = width / 2;
        let h4 = height / 4;
        let w4 = width / 4;
        let h8 = height / 8;
        let w8 = width / 8;
        // The pool is a handful of adds. It is not in the MAC total.
        push_dense(
            &mut rows,
            "stem",
            h2,
            w2,
            self.input_planes,
            self.base,
            3,
            width / 2,
        );
        push_sep(&mut rows, "enc0", h4, w4, self.base, self.trunk, width / 2);
        // enc1 reads the H/4 map either way. Stride 1 writes H/4; stride 2 writes H/8.
        let (enc1_h, enc1_w) = if self.trunk_h4 { (h4, w4) } else { (h8, w8) };
        push_sep(
            &mut rows,
            "enc1",
            enc1_h,
            enc1_w,
            self.trunk,
            self.trunk,
            width / 4,
        );
        let (res_h, res_w, res_row) = if self.trunk_h4 {
            (h4, w4, width / 4)
        } else {
            (h8, w8, width / 8)
        };
        for i in 0..self.res_blocks {
            push_sep(
                &mut rows,
                &format!("res{i}.s1"),
                res_h,
                res_w,
                self.trunk,
                self.trunk,
                res_row,
            );
            push_sep(
                &mut rows,
                &format!("res{i}.s2"),
                res_h,
                res_w,
                self.trunk,
                self.trunk,
                res_row,
            );
        }
        if self.shuffle && !self.trunk_h4 {
            push_shuffle(&mut rows, "sh_h8", h8, w8, self.trunk);
        }
        push_sep(
            &mut rows,
            "dec_h4",
            h4,
            w4,
            self.trunk,
            self.up_h4,
            width / 4,
        );
        if self.shuffle {
            push_shuffle(&mut rows, "sh_h4", h4, w4, self.up_h4);
        }
        push_sep(
            &mut rows,
            "dec_h2",
            h2,
            w2,
            self.up_h4,
            self.up_h2,
            width / 2,
        );
        if self.h2_guide > 0 {
            push_dense(
                &mut rows,
                "h2_guide",
                h2,
                w2,
                self.input_planes,
                self.h2_guide,
                3,
                width / 2,
            );
            if self.h2_guide != self.up_h2 {
                push_pw(
                    &mut rows,
                    "h2_guide_proj",
                    h2,
                    w2,
                    self.h2_guide,
                    self.up_h2,
                );
            }
        }
        if self.film_h2 {
            push_film(&mut rows, "film_h2", self.up_h2);
        }
        // Align to the guide width at H/2, then upsample. Cheaper than a full-res 1×1.
        push_pw(&mut rows, "to_guide", h2, w2, self.up_h2, self.guide);
        if self.shuffle_guide {
            push_shuffle(&mut rows, "sh_h2", h2, w2, self.guide);
        }
        push_dense(
            &mut rows,
            "guide",
            height,
            width,
            self.input_planes,
            self.guide,
            self.guide_kernel,
            width,
        );
        if self.guide_resid {
            push_pw(
                &mut rows,
                "guide_gain",
                height,
                width,
                self.guide,
                self.guide,
            );
        }
        if self.film_guide {
            push_film(&mut rows, "film_guide", self.guide);
        }
        if self.full > 0 {
            push_sep(
                &mut rows, "full", height, width, self.guide, self.full, width,
            );
            if self.film_full {
                push_film(&mut rows, "film_full", self.full);
            }
            if self.full_layers >= 2 {
                push_sep(
                    &mut rows, "full2", height, width, self.full, self.full, width,
                );
            }
            push_pw(&mut rows, "head", height, width, self.full, 3);
        } else {
            push_sep(&mut rows, "head", height, width, self.guide, 3, width);
        }
        CostReport { rows }
    }
}

/// One layer in the hardware account.
pub struct CostRow {
    pub name: String,
    pub macs: u64,
    /// Two A8 input rows of a 3×3. Zero for a 1×1.
    pub line_bytes: u64,
}

pub struct CostReport {
    pub rows: Vec<CostRow>,
}

impl CostReport {
    pub fn macs(&self) -> u64 {
        self.rows.iter().map(|row| row.macs).sum()
    }

    pub fn render(&self, title: &str, height: usize, width: usize) -> String {
        let mut out = format!("{title} @ {width}x{height}\n");
        for row in &self.rows {
            out.push_str(&format!(
                "  {:<10} {:>8.2} Mmac   line {:>7.1} KiB\n",
                row.name,
                row.macs as f64 / 1e6,
                row.line_bytes as f64 / 1024.0,
            ));
        }
        let full_line: u64 = self
            .rows
            .iter()
            .filter(|row| row.line_bytes > 0 && is_full_res(row, width))
            .map(|row| row.line_bytes)
            .sum();
        out.push_str(&format!(
            "  total {:>8.2} Mmac, full-res 3x3 line {:.1} KiB\n",
            self.macs() as f64 / 1e6,
            full_line as f64 / 1024.0,
        ));
        out
    }
}

fn is_full_res(row: &CostRow, _width: usize) -> bool {
    row.line_bytes > 0
        && matches!(
            row.name.as_str(),
            "guide" | "full" | "full2" | "head" | "up3.1" | "v0_head"
        )
}

fn push_dense(
    rows: &mut Vec<CostRow>,
    name: &str,
    oh: usize,
    ow: usize,
    cin: usize,
    cout: usize,
    k: usize,
    in_width: usize,
) {
    rows.push(CostRow {
        name: name.to_owned(),
        macs: oh as u64 * ow as u64 * cin as u64 * cout as u64 * k as u64 * k as u64,
        // A k×k kernel keeps k−1 input rows. k = 3 stays at two rows.
        line_bytes: if k >= 3 {
            (k as u64 - 1) * in_width as u64 * cin as u64
        } else {
            0
        },
    });
}

fn push_sep(
    rows: &mut Vec<CostRow>,
    name: &str,
    oh: usize,
    ow: usize,
    cin: usize,
    cout: usize,
    in_width: usize,
) {
    let dw = oh as u64 * ow as u64 * cin as u64 * 9;
    let pw = oh as u64 * ow as u64 * cin as u64 * cout as u64;
    rows.push(CostRow {
        name: name.to_owned(),
        macs: dw + pw,
        line_bytes: 2 * in_width as u64 * cin as u64,
    });
}

fn push_shuffle(rows: &mut Vec<CostRow>, name: &str, oh: usize, ow: usize, channels: usize) {
    // Grouped 1×1 writes four subpixel weights per channel, then a rearrange.
    rows.push(CostRow {
        name: name.to_owned(),
        macs: oh as u64 * ow as u64 * channels as u64 * 4,
        line_bytes: 0,
    });
}

fn push_pw(rows: &mut Vec<CostRow>, name: &str, oh: usize, ow: usize, cin: usize, cout: usize) {
    rows.push(CostRow {
        name: name.to_owned(),
        macs: oh as u64 * ow as u64 * cin as u64 * cout as u64,
        line_bytes: 0,
    });
}

fn push_film(rows: &mut Vec<CostRow>, name: &str, channels: usize) {
    let hidden = (channels / 4).max(1);
    // GAP is free at this scale. Three 1×1s on a 1×1 vector: reduce, γ, β.
    rows.push(CostRow {
        name: name.to_owned(),
        macs: (channels * hidden + 2 * hidden * channels) as u64,
        line_bytes: 0,
    });
}

/// `film_both` account, so a guided run can print the SOTA it is measured against.
pub fn v0_cost(height: usize, width: usize) -> CostReport {
    let mut rows = Vec::new();
    let h2 = height / 2;
    let w2 = width / 2;
    let h4 = height / 4;
    let w4 = width / 4;
    push_dense(&mut rows, "stem", height, width, 3, 32, 3, width);
    push_sep(&mut rows, "enc0", h2, w2, 32, 64, width);
    push_sep(&mut rows, "enc1", h4, w4, 64, 64, width / 2);
    push_sep(&mut rows, "res0.s1", h4, w4, 64, 64, width / 4);
    push_sep(&mut rows, "res0.s2", h4, w4, 64, 64, width / 4);
    push_sep(&mut rows, "up3.0", h2, w2, 64, 48, width / 2);
    push_film(&mut rows, "film_h2", 48);
    push_sep(&mut rows, "up3.1", height, width, 48, 24, width);
    push_film(&mut rows, "film_full", 24);
    push_sep(&mut rows, "v0_head", height, width, 24, 3, width);
    CostReport { rows }
}

/// One style besides the checkpoint-compatible slot. Style 0 stays in
/// `trunk_norm` and `film_h2` so an existing `model.mpk` still loads.
#[derive(Module, Debug)]
struct StylePacket<B: Backend> {
    trunk_norm: Plugin<B>,
    film_h2: Film<B>,
    guide_affine: Option<GuideAffine<B>>,
}

/// Per-channel `x · exp(log_gain) + bias` on the guide conv output, before its ReLU.
///
/// Folds into the guide kernel as `W·g`, `bias_conv·g + bias`. A positive `g` keeps
/// the per-channel int8 codes of `W` and only moves the requant multiplier; a negative
/// `bias` raises the threshold, which is how a style closes a guide channel.
#[derive(Module, Debug)]
pub struct GuideAffine<B: Backend> {
    log_gain: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> GuideAffine<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            log_gain: Param::from_tensor(Tensor::zeros([channels], device)),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = self.bias.dims()[0];
        let gain = self.log_gain.val().exp().reshape([1, c, 1, 1]);
        x * gain + self.bias.val().reshape([1, c, 1, 1])
    }
}

fn trunk_plugin<B: Backend>(spec: &GuideSpec, device: &B::Device) -> Plugin<B> {
    match spec.trunk_norm_kind.as_str() {
        "in" => Plugin::instance_norm(spec.trunk, spec.in_eps, device),
        "ln" => Plugin::layer_norm(spec.trunk, spec.in_eps, device),
        _ => unreachable!("trunk_norm_kind validated in check()"),
    }
}

/// Depthwise trunk at H/8, RGB guide at full resolution.
#[derive(Module, Debug)]
pub struct GuidedStudent<B: Backend> {
    stem: Conv<B>,
    enc0: SepConv<B>,
    enc1: SepConv<B>,
    res: Vec<ResBlockDw<B>>,
    trunk_norm: Plugin<B>,
    shuffle_h8: Option<Plugin<B>>,
    dec_h4: SepConv<B>,
    shuffle_h4: Option<Plugin<B>>,
    dec_h2: SepConv<B>,
    h2_guide: Option<Conv<B>>,
    h2_guide_proj: Option<PwConv<B>>,
    film_h2: Option<Film<B>>,
    to_guide: PwConv<B>,
    shuffle_guide: Option<Plugin<B>>,
    guide: Conv<B>,
    film_guide: Option<Film<B>>,
    full_sep: Option<SepConv<B>>,
    film_full: Option<Film<B>>,
    full_sep2: Option<SepConv<B>>,
    head_pw: Option<PwConv<B>>,
    head_sep: Option<SepConv<B>>,
    /// Zero-init 1×1 on the upsampled paint. Present only for `guide_mix = resid`.
    /// Built last so the rest of the graph keeps the additive init.
    guide_gain: Option<PwConv<B>>,
    /// Styles 1..N. Empty when `styles` is 1, so the parameter names do not move.
    style_packets: Vec<StylePacket<B>>,
    /// Style 0's guide affine. `None` unless `packet_guide`, so old names do not move.
    guide_affine: Option<GuideAffine<B>>,
    #[module(skip)]
    spec: GuideSpec,
}

impl<B: Backend> GuidedStudent<B> {
    pub fn new(spec: GuideSpec, device: &B::Device) -> Self {
        spec.check().unwrap_or_else(|err| panic!("{err}"));
        let film_h2 = spec.film_h2.then(|| Film::new(spec.up_h2, device));
        let shuffle_trunk = |channels: usize| {
            spec.shuffle
                .then(|| Plugin::new(PluginKind::Shuffle, channels, device))
        };
        let shuffle_guide = spec
            .shuffle_guide
            .then(|| Plugin::new(PluginKind::Shuffle, spec.guide, device));
        let (full_sep, film_full, full_sep2, head_pw, head_sep) = if spec.full > 0 {
            (
                Some(SepConv::new(spec.guide, spec.full, 3, 1, device)),
                spec.film_full.then(|| Film::new(spec.full, device)),
                (spec.full_layers >= 2).then(|| SepConv::new(spec.full, spec.full, 3, 1, device)),
                Some(PwConv::new(spec.full, 3, device)),
                None,
            )
        } else {
            (
                None,
                None,
                None,
                None,
                Some(SepConv::new(spec.guide, 3, 3, 1, device)),
            )
        };
        let pre_in = spec.pre_in_gain.conv_init();
        let cin = spec.input_planes;
        Self {
            stem: Conv::with_init(cin, spec.base, 3, 1, true, pre_in, device),
            enc0: SepConv::with_init(spec.base, spec.trunk, 3, 2, pre_in, device),
            enc1: SepConv::with_init(
                spec.trunk,
                spec.trunk,
                3,
                if spec.trunk_h4 { 1 } else { 2 },
                pre_in,
                device,
            ),
            res: (0..spec.res_blocks)
                .map(|_| ResBlockDw::with_init(spec.trunk, pre_in, device))
                .collect(),
            trunk_norm: trunk_plugin(&spec, device),
            shuffle_h8: if spec.trunk_h4 {
                None
            } else {
                shuffle_trunk(spec.trunk)
            },
            dec_h4: SepConv::new(spec.trunk, spec.up_h4, 3, 1, device),
            shuffle_h4: shuffle_trunk(spec.up_h4),
            dec_h2: SepConv::new(spec.up_h4, spec.up_h2, 3, 1, device),
            h2_guide: (spec.h2_guide > 0)
                .then(|| Conv::new(cin, spec.h2_guide, 3, 1, true, device)),
            h2_guide_proj: (spec.h2_guide > 0 && spec.h2_guide != spec.up_h2)
                .then(|| PwConv::new(spec.h2_guide, spec.up_h2, device)),
            film_h2,
            to_guide: PwConv::new(spec.up_h2, spec.guide, device),
            shuffle_guide,
            guide: Conv::new(cin, spec.guide, spec.guide_kernel, 1, true, device),
            film_guide: spec.film_guide.then(|| Film::new(spec.guide, device)),
            full_sep,
            film_full,
            full_sep2,
            head_pw,
            head_sep,
            guide_gain: spec.guide_resid.then(|| {
                let mut gain = PwConv::new(spec.guide, spec.guide, device);
                gain.zero(device);
                gain
            }),
            style_packets: (1..spec.styles)
                .map(|_| StylePacket {
                    trunk_norm: trunk_plugin(&spec, device),
                    film_h2: Film::new(spec.up_h2, device),
                    guide_affine: spec
                        .packet_guide
                        .then(|| GuideAffine::new(spec.guide, device)),
                })
                .collect(),
            guide_affine: spec
                .packet_guide
                .then(|| GuideAffine::new(spec.guide, device)),
            spec,
        }
    }

    pub fn spec(&self) -> &GuideSpec {
        &self.spec
    }

    /// RGB stays 3 channels. A 2-plane student turns RGB into luma and the 135° chroma axis.
    /// `rgb565` then snaps each channel onto the 5/6/5 codebook.
    fn student_input(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, channels, _, _] = content.dims();
        let content = match (self.spec.input_planes, channels) {
            (3, 3) | (2, 2) => content,
            (2, 3) => rgb_to_yc(content),
            (planes, got) => panic!("tiny_dw_guided input wants {planes} channels, got {got}"),
        };
        if self.spec.input_quant == "rgb565" {
            quantize_rgb565(content)
        } else {
            content
        }
    }

    fn forward(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        self.render(content, false, false).0
    }

    /// `mean_paint` replaces the upsampled paint with its per-channel spatial
    /// mean before the full-resolution guide is added. `probe_dead` records
    /// how many trunk channels are below `1e-5` variance just before
    /// InstanceNorm. The training forward leaves both off, so it does not sync.
    fn render(
        &self,
        content: Tensor<B, 4>,
        mean_paint: bool,
        probe_dead: bool,
    ) -> (Tensor<B, 4>, Option<f32>) {
        let ablate = if mean_paint {
            Ablate::MeanPaint
        } else {
            Ablate::None
        };
        self.render_tapped(content, ablate, probe_dead, 0, &mut |_, _| {})
    }

    /// Every stage of the forward, in order, for probes. `tap` sees each map
    /// once; the output is the same tensor [`Self::forward`] returns.
    pub fn trace(
        &self,
        content: Tensor<B, 4>,
        ablate: Ablate,
        tap: &mut dyn FnMut(&'static str, &Tensor<B, 4>),
    ) -> Tensor<B, 4> {
        self.render_tapped(content, ablate, false, 0, tap).0
    }

    fn render_tapped(
        &self,
        content: Tensor<B, 4>,
        ablate: Ablate,
        probe_dead: bool,
        style: usize,
        tap: &mut dyn FnMut(&'static str, &Tensor<B, 4>),
    ) -> (Tensor<B, 4>, Option<f32>) {
        let content = self.student_input(content);
        let [_, _, h0, w0] = content.dims();
        let x = pad_spatial(content, 8);
        let low = avg_pool2d(x.clone(), [2, 2], [2, 2], [0, 0], true, false);
        let low_for_guide = self.h2_guide.is_some().then(|| low.clone());
        let mut f = relu(self.stem.forward(low));
        tap("stem", &f);
        f = self.enc0.forward(f);
        tap("enc0", &f);
        f = self.enc1.forward(f);
        tap("enc1", &f);
        for (index, block) in self.res.iter().enumerate() {
            f = block.forward(f);
            tap(RES_TAPS.get(index).copied().unwrap_or("res"), &f);
        }
        let in_dead = probe_dead.then(|| channel_dead_frac(f.clone()));
        f = self.norm_at(f, style);
        tap("trunk_norm", &f);
        f = if self.spec.trunk_h4 {
            self.dec_h4.forward(f)
        } else {
            self.dec_h4.forward(self.up2(f, &self.shuffle_h8))
        };
        tap("dec_h4", &f);
        f = self.dec_h2.forward(self.up2(f, &self.shuffle_h4));
        tap("dec_h2", &f);
        if let Some(conv) = &self.h2_guide {
            let mut edge = relu(conv.forward(low_for_guide.expect("pooled rgb")));
            if let Some(proj) = &self.h2_guide_proj {
                edge = proj.forward(edge);
            }
            tap("h2_guide", &edge);
            f = f + edge;
        }
        if style == 0 {
            if let Some(film) = &self.film_h2 {
                f = film.forward(f);
                tap("film_h2", &f);
            }
        } else {
            f = self.style_packets[style - 1].film_h2.forward(f);
            tap("film_h2", &f);
        }
        // Project at H/2, then spread the 8-channel paint. The guide is what
        // breaks the 2×2 blocks back onto the original edges.
        f = relu(self.to_guide.forward(f));
        tap("paint", &f);
        f = self.up2(f, &self.shuffle_guide);
        let guide = self.guide.forward(x);
        let guide = match self.guide_affine_at(style) {
            Some(affine) => affine.forward(guide),
            None => guide,
        };
        let guide = if self.spec.guide_relu {
            relu(guide)
        } else {
            guide
        };
        tap("guide", &guide);
        let paint = if ablate == Ablate::MeanPaint {
            f.mean_dim(2).mean_dim(3)
        } else {
            f
        };
        let guide = if ablate == Ablate::MeanGuide {
            guide.mean_dim(2).mean_dim(3)
        } else {
            guide
        };
        let mut f = if let Some(gain) = &self.guide_gain {
            let g = gain.forward(paint.clone());
            paint.clone() + guide + paint * g
        } else if self.spec.guide_gate {
            paint.clone() + paint * guide
        } else {
            paint + guide
        };
        if let Some(film) = &self.film_guide {
            f = film.forward(f);
        }
        tap("mix", &f);
        let y = if let Some(full) = &self.full_sep {
            let mut f = full.forward(f);
            tap("full", &f);
            if let Some(film) = &self.film_full {
                f = film.forward(f);
            }
            if let Some(full2) = &self.full_sep2 {
                f = full2.forward(f);
            }
            self.head_pw.as_ref().expect("full-res head").forward(f)
        } else {
            self.head_sep
                .as_ref()
                .expect("full-res head")
                .forward_inner(f, false)
        };
        let y = crop_spatial(y, h0, w0);
        tap("rgb", &y);
        (y, in_dead)
    }

    fn guide_affine_at(&self, style: usize) -> Option<&GuideAffine<B>> {
        if style == 0 {
            self.guide_affine.as_ref()
        } else {
            self.style_packets[style - 1].guide_affine.as_ref()
        }
    }

    fn norm_at(&self, f: Tensor<B, 4>, style: usize) -> Tensor<B, 4> {
        if style == 0 {
            self.trunk_norm.forward(f)
        } else {
            self.style_packets[style - 1].trunk_norm.forward(f)
        }
    }

    fn up2(&self, x: Tensor<B, 4>, shuffle: &Option<Plugin<B>>) -> Tensor<B, 4> {
        match shuffle {
            Some(plugin) => plugin.shuffle_up(x),
            None => upsample2x(x),
        }
    }
}

impl<B: Backend> StudentModel<B> for GuidedStudent<B> {
    fn stylize(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(content)
    }

    fn style_count(&self) -> usize {
        self.spec.styles
    }

    fn stylize_style(&self, content: Tensor<B, 4>, style: usize) -> Tensor<B, 4> {
        if style >= self.spec.styles {
            panic!("style {style} is outside 0..{}", self.spec.styles);
        }
        self.render_tapped(content, Ablate::None, false, style, &mut |_, _| {})
            .0
    }

    fn describe(&self) -> String {
        let full = if self.spec.full == 0 {
            "head-only".to_owned()
        } else {
            format!("{}ch x{}", self.spec.full, self.spec.full_layers)
        };
        let up = match (self.spec.shuffle, self.spec.shuffle_guide) {
            (false, false) => "nearest",
            (false, true) => "nearest+shuffle_guide",
            (true, false) => "shuffle+nearest_guide",
            (true, true) => "shuffle",
        };
        let guide_act = if self.spec.guide_relu {
            "relu"
        } else {
            "linear"
        };
        let mix = if self.spec.guide_resid {
            "resid"
        } else if self.spec.guide_gate {
            "gate"
        } else {
            "add"
        };
        let trunk = if self.spec.trunk_h4 { "H/4" } else { "H/8" };
        let input = if self.spec.input_planes == 2 {
            "yc"
        } else {
            "rgb"
        };
        format!(
            "tiny_dw_guided pool=2 trunk={trunk} up={up} up_h4={} up_h2={} h2_guide={} guide={} k={} guide_act={guide_act} mix={mix} full={full} norm={} in_eps={:.0e} pre_in_gain={} input={input} quant={} styles={}{} ({} parameters)",
            self.spec.up_h4,
            self.spec.up_h2,
            self.spec.h2_guide,
            self.spec.guide,
            self.spec.guide_kernel,
            self.spec.trunk_norm_kind,
            self.spec.in_eps,
            self.spec.pre_in_gain.label(),
            self.spec.input_quant,
            self.spec.styles,
            if self.spec.packet_guide { " packet_guide" } else { "" },
            self.num_params()
        )
    }

    fn sample_signal(&self, content: Tensor<B, 4>) -> Option<SampleSignal> {
        let (y, in_dead) = self.render(content.clone(), false, true);
        let (y_flat, _) = self.render(content, true, false);
        Some(SampleSignal {
            paint_spatial: paint_spatial_contrib(y, y_flat),
            in_dead: in_dead.unwrap_or(0.0),
        })
    }
}

struct GuidedQuant {
    image: usize,
    pool: usize,
    enc0_dw: usize,
    enc0_pw: usize,
    enc1_dw: usize,
    enc1_pw: usize,
    res: Vec<[usize; 4]>,
    trunk_sum: usize,
    trunk_norm: usize,
    dec_h4_pw: usize,
    dec_h2_dw: usize,
    dec_h2_pw: usize,
    h2_pw: usize,
    h2_proj: usize,
    h2_add: usize,
    to_guide: usize,
    paint: usize,
    guide: usize,
    head_pw: usize,
    names: Vec<String>,
}

fn guided_quant(res_blocks: usize) -> GuidedQuant {
    let mut names = Vec::new();
    let mut push = |name: String| {
        let index = names.len();
        names.push(name);
        index
    };
    let image = push("image".into());
    let pool = push("pool".into());
    let enc0_dw = push("enc0.dw.in".into());
    let enc0_pw = push("enc0.pw.in".into());
    let enc1_dw = push("enc1.dw.in".into());
    let enc1_pw = push("enc1.pw.in".into());
    let mut res = Vec::new();
    for block in 0..res_blocks {
        res.push([
            push(format!("res{block}.s1.dw.in")),
            push(format!("res{block}.s1.pw.in")),
            push(format!("res{block}.s2.dw.in")),
            push(format!("res{block}.s2.pw.in")),
        ]);
    }
    let trunk_sum = push("trunk.sum".into());
    let trunk_norm = push("trunk.norm".into());
    let dec_h4_pw = push("dec_h4.pw.in".into());
    let dec_h2_dw = push("dec_h2.dw.in".into());
    let dec_h2_pw = push("dec_h2.pw.in".into());
    let h2_pw = push("h2_guide.pw.in".into());
    let h2_proj = push("h2_proj".into());
    let h2_add = push("h2_add".into());
    let to_guide = push("to_guide.in".into());
    let paint = push("paint".into());
    let guide = push("guide".into());
    let head_pw = push("head.pw.in".into());
    GuidedQuant {
        image,
        pool,
        enc0_dw,
        enc0_pw,
        enc1_dw,
        enc1_pw,
        res,
        trunk_sum,
        trunk_norm,
        dec_h4_pw,
        dec_h2_dw,
        dec_h2_pw,
        h2_pw,
        h2_proj,
        h2_add,
        to_guide,
        paint,
        guide,
        head_pw,
        names,
    }
}

fn qa<B: Backend, F: FnMut(usize, &Tensor<B, 4>)>(
    x: Tensor<B, 4>,
    scheme: Option<&QuantScheme>,
    point: usize,
    observe: &mut F,
) -> Tensor<B, 4> {
    observe(point, &x);
    match scheme {
        Some(scheme) => quant::fake_quant_act(x, scheme.acts[point], scheme.act_max()),
        None => x,
    }
}

fn wq<B: Backend>(weight: Tensor<B, 4>, scheme: Option<&QuantScheme>) -> Tensor<B, 4> {
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

fn on_scale<B: Backend>(
    x: Tensor<B, 4>,
    scheme: Option<&QuantScheme>,
    point: usize,
) -> Tensor<B, 4> {
    match scheme {
        Some(scheme) => quant::fake_quant_act(x, scheme.acts[point], scheme.act_max()),
        None => x,
    }
}

impl<B: Backend> GuidedStudent<B> {
    /// Why this topology is not in the integer manifest. `None` means it is.
    fn export_block_reason(&self) -> Option<String> {
        let spec = &self.spec;
        let reason = if spec.styles > 1 {
            "integer export is one style; the extra packets are not in the manifest"
        } else if spec.packet_guide {
            "the guide affine has to be folded into the guide kernel before export"
        } else if spec.shuffle || spec.shuffle_guide {
            "pixel-shuffle is not in the integer manifest"
        } else if !spec.trunk_h4 {
            "integer export keeps the trunk at H/4"
        } else if spec.full != 0 {
            "a full-resolution stack is not in the integer manifest"
        } else if spec.film_guide || spec.film_full {
            "only the H/2 FiLM is exported"
        } else if !spec.film_h2 {
            "integer export includes the H/2 FiLM"
        } else if spec.h2_guide == 0 || spec.h2_guide == spec.up_h2 {
            "integer export needs an H/2 guide with a 1x1 align"
        } else if !spec.guide_relu || spec.guide_gate || spec.guide_resid || spec.guide_kernel != 3
        {
            "integer export keeps the additive 3x3 ReLU guide"
        } else if !matches!(self.trunk_norm.kind(), PluginKind::In | PluginKind::Ln) {
            "the trunk norm has to be instance norm or layer norm"
        } else {
            return None;
        };
        Some(reason.to_owned())
    }

    fn quant_layout(&self) -> GuidedQuant {
        guided_quant(self.res.len())
    }

    fn forward_observed<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        scheme: Option<&QuantScheme>,
        mut observe: F,
    ) -> Tensor<B, 4> {
        if let Some(reason) = self.export_block_reason() {
            panic!("tiny_dw_guided quant forward: {reason}");
        }
        let content = self.student_input(content);
        let pts = self.quant_layout();
        let [_, _, h0, w0] = content.dims();
        let image = qa(pad_spatial(content, 8), scheme, pts.image, &mut observe);
        let pooled = avg_pool2d(image.clone(), [2, 2], [2, 2], [0, 0], true, false);
        let low = qa(pooled, scheme, pts.pool, &mut observe);

        let mut f = relu(
            self.stem
                .forward_with(low.clone(), wq(self.stem.weight(), scheme)),
        );
        f = sep_quant(
            &self.enc0,
            f,
            pts.enc0_dw,
            pts.enc0_pw,
            scheme,
            &mut observe,
            true,
        );
        f = sep_quant(
            &self.enc1,
            f,
            pts.enc1_dw,
            pts.enc1_pw,
            scheme,
            &mut observe,
            true,
        );
        for (index, (block, points)) in self.res.iter().zip(&pts.res).enumerate() {
            let sum_point = pts
                .res
                .get(index + 1)
                .map(|next| next[0])
                .unwrap_or(pts.trunk_sum);
            let input = qa(f, scheme, points[0], &mut observe);
            let mid = sep_quant(
                &block.s1,
                input.clone(),
                points[0],
                points[1],
                scheme,
                &mut observe,
                true,
            );
            let inner = sep_quant(
                &block.s2,
                mid,
                points[2],
                points[3],
                scheme,
                &mut observe,
                false,
            );
            let sum = on_scale(input, scheme, sum_point) + on_scale(inner, scheme, sum_point);
            if sum_point == pts.trunk_sum {
                observe(pts.trunk_sum, &sum);
            }
            f = sum;
        }
        if self.res.is_empty() {
            f = qa(f, scheme, pts.trunk_sum, &mut observe);
        }
        f = self.trunk_norm.forward(f);
        f = qa(f, scheme, pts.trunk_norm, &mut observe);
        f = sep_quant(
            &self.dec_h4,
            f,
            pts.trunk_norm,
            pts.dec_h4_pw,
            scheme,
            &mut observe,
            true,
        );
        f = upsample2x(f);
        f = sep_quant(
            &self.dec_h2,
            f,
            pts.dec_h2_dw,
            pts.dec_h2_pw,
            scheme,
            &mut observe,
            true,
        );

        let edge_conv = self.h2_guide.as_ref().expect("h2 guide");
        let mut edge = relu(edge_conv.forward_with(low, wq(edge_conv.weight(), scheme)));
        edge = qa(edge, scheme, pts.h2_pw, &mut observe);
        let proj = self.h2_guide_proj.as_ref().expect("h2 guide proj");
        let edge = proj.forward_with(edge, wq(proj.weight(), scheme));
        let edge_q = on_scale(edge, scheme, pts.h2_proj);
        observe(pts.h2_proj, &edge_q);
        let summed = on_scale(f, scheme, pts.h2_add) + on_scale(edge_q, scheme, pts.h2_add);
        observe(pts.h2_add, &summed);
        let film = self.film_h2.as_ref().expect("film_h2");
        f = film.forward(summed);
        f = qa(f, scheme, pts.to_guide, &mut observe);
        f = relu(
            self.to_guide
                .forward_with(f, wq(self.to_guide.weight(), scheme)),
        );
        f = upsample2x(f);
        let paint = on_scale(f, scheme, pts.paint);
        observe(pts.paint, &paint);

        let guide = relu(
            self.guide
                .forward_with(image, wq(self.guide.weight(), scheme)),
        );
        let guide = on_scale(guide, scheme, pts.guide);
        observe(pts.guide, &guide);
        let mixed = on_scale(paint, scheme, pts.paint) + on_scale(guide, scheme, pts.paint);
        let head = self.head_sep.as_ref().expect("head");
        let y = sep_quant(
            head,
            mixed,
            pts.paint,
            pts.head_pw,
            scheme,
            &mut observe,
            false,
        );
        let y = match scheme {
            Some(_) => quant::fake_quant_output(y),
            None => y,
        };
        crop_spatial(y, h0, w0)
    }
}

fn sep_quant<B: Backend, F: FnMut(usize, &Tensor<B, 4>)>(
    sep: &SepConv<B>,
    x: Tensor<B, 4>,
    dw_point: usize,
    pw_point: usize,
    scheme: Option<&QuantScheme>,
    observe: &mut F,
    out_relu: bool,
) -> Tensor<B, 4> {
    let x = qa(x, scheme, dw_point, observe);
    let dw = relu(sep.dw.forward_with(x, wq(sep.dw.weight(), scheme)));
    let dw = qa(dw, scheme, pw_point, observe);
    let y = sep.pw.forward_with(dw, wq(sep.pw.weight(), scheme));
    if out_relu { relu(y) } else { y }
}

impl<B: Backend> quant::Quantizable<B> for GuidedStudent<B> {
    fn quant_points(&self) -> Vec<String> {
        if self.export_block_reason().is_some() {
            return Vec::new();
        }
        self.quant_layout().names
    }

    fn forward_quantised(&self, content: Tensor<B, 4>, scheme: &QuantScheme) -> Tensor<B, 4> {
        self.forward_observed(content, Some(scheme), |_, _| {})
    }

    fn observe_activations(&self, content: Tensor<B, 4>, observe: &mut dyn FnMut(usize, f32)) {
        self.forward_observed(content, None, |point, tensor| {
            observe(
                point,
                tensor.clone().abs().max().into_scalar().to_f64() as f32,
            );
        });
    }
}

impl Exportable for GuidedStudent<crate::backend::InferBackend> {
    fn export_blocker(&self) -> Option<String> {
        self.export_block_reason()
    }

    fn export_notes(&self, bits: &crate::quant::Bits) -> Vec<String> {
        let mut notes = export::default_notes(bits);
        notes.push(
            "avg_pool2x: 2x2 average of the quantised image, then requantise onto the stem input. The full-resolution guide keeps reading the image."
                .to_owned(),
        );
        notes.push(
            "layer_norm: at each pixel, mean and variance over channels, then γ and β. instance_norm is the spatial reduction instead. Both requantise onto the next point."
                .to_owned(),
        );
        notes.push(format!("trunk norm: {}", self.spec.trunk_norm_kind));
        notes
    }

    fn export_plan(&self, scales: &[f32], point_count: usize) -> Vec<ExportStep<'_>> {
        let pts = self.quant_layout();
        let mut plan = Vec::new();
        plan.push(ExportStep::AvgPool2x {
            in_point: pts.image,
            out_point: pts.pool,
            out_scale: scales[pts.pool],
        });
        plan.push(conv_step(
            "stem".into(),
            ConvRef::Dense(&self.stem),
            pts.pool,
            pts.enc0_dw,
            true,
            scales[pts.enc0_dw],
            false,
        ));
        push_sep_export(
            &mut plan,
            "enc0",
            &self.enc0,
            pts.enc0_dw,
            pts.enc0_pw,
            pts.enc1_dw,
            scales,
            true,
        );
        push_sep_export(
            &mut plan,
            "enc1",
            &self.enc1,
            pts.enc1_dw,
            pts.enc1_pw,
            pts.res.first().map(|p| p[0]).unwrap_or(pts.trunk_sum),
            scales,
            true,
        );
        for (index, block) in self.res.iter().enumerate() {
            let points = pts.res[index];
            let sum_point = pts
                .res
                .get(index + 1)
                .map(|next| next[0])
                .unwrap_or(pts.trunk_sum);
            push_sep_export(
                &mut plan,
                &format!("res{index}.s1"),
                &block.s1,
                points[0],
                points[1],
                points[2],
                scales,
                true,
            );
            push_sep_export(
                &mut plan,
                &format!("res{index}.s2"),
                &block.s2,
                points[2],
                points[3],
                sum_point,
                scales,
                false,
            );
            plan.push(ExportStep::ResidualAdd {
                point: sum_point,
                source_point: points[0],
                scale: scales[sum_point],
            });
        }
        let (gamma, beta) = self
            .trunk_norm
            .host_norm()
            .expect("trunk norm")
            .expect("trunk norm affine");
        let norm_name = format!("trunk.{}", self.trunk_norm.kind().name());
        match self.trunk_norm.kind() {
            PluginKind::Ln => plan.push(ExportStep::LayerNorm {
                name: norm_name,
                in_point: pts.trunk_sum,
                out_point: pts.trunk_norm,
                channels: gamma.len(),
                gamma,
                beta,
                eps: self.spec.in_eps,
                out_scale: scales[pts.trunk_norm],
            }),
            PluginKind::In => plan.push(ExportStep::InstanceNorm {
                name: norm_name,
                in_point: pts.trunk_sum,
                out_point: pts.trunk_norm,
                channels: gamma.len(),
                gamma,
                beta,
                eps: self.spec.in_eps,
                out_scale: scales[pts.trunk_norm],
            }),
            other => panic!("trunk norm {other:?} is not exported"),
        }
        push_sep_export(
            &mut plan,
            "dec_h4",
            &self.dec_h4,
            pts.trunk_norm,
            pts.dec_h4_pw,
            pts.dec_h2_dw,
            scales,
            true,
        );
        plan.push(ExportStep::UpsampleNearest2x {
            point: pts.dec_h2_dw,
        });
        push_sep_export(
            &mut plan,
            "dec_h2",
            &self.dec_h2,
            pts.dec_h2_dw,
            pts.dec_h2_pw,
            pts.h2_add,
            scales,
            true,
        );
        let h2 = self.h2_guide.as_ref().expect("h2 guide");
        plan.push(conv_step(
            "h2_guide".into(),
            ConvRef::Dense(h2),
            pts.pool,
            pts.h2_pw,
            true,
            scales[pts.h2_pw],
            false,
        ));
        let proj = self.h2_guide_proj.as_ref().expect("h2 proj");
        plan.push(conv_step(
            "h2_proj".into(),
            ConvRef::Pointwise(proj),
            pts.h2_pw,
            pts.h2_proj,
            false,
            scales[pts.h2_proj],
            false,
        ));
        plan.push(ExportStep::ResidualAdd {
            point: pts.h2_add,
            source_point: pts.h2_proj,
            scale: scales[pts.h2_add],
        });
        let film = self.film_h2.as_ref().expect("film_h2");
        plan.push(ExportStep::Film {
            name: "film_h2".into(),
            film,
            channels: self.spec.up_h2,
            in_point: pts.h2_add,
            out_point: pts.to_guide,
            out_scale: scales[pts.to_guide],
        });
        plan.push(conv_step(
            "to_guide".into(),
            ConvRef::Pointwise(&self.to_guide),
            pts.to_guide,
            pts.paint,
            true,
            scales[pts.paint],
            false,
        ));
        plan.push(ExportStep::UpsampleNearest2x { point: pts.paint });
        plan.push(conv_step(
            "guide".into(),
            ConvRef::Dense(&self.guide),
            pts.image,
            pts.guide,
            true,
            scales[pts.guide],
            false,
        ));
        plan.push(ExportStep::ResidualAdd {
            point: pts.paint,
            source_point: pts.guide,
            scale: scales[pts.paint],
        });
        let head = self.head_sep.as_ref().expect("head");
        push_sep_export(
            &mut plan,
            "head",
            head,
            pts.paint,
            pts.head_pw,
            point_count,
            scales,
            false,
        );
        // The head's output scale is the image grid, not a calibrated activation.
        if let ExportStep::Conv {
            out_scale,
            out_unsigned,
            out_point,
            ..
        } = plan.last_mut().expect("head pointwise")
        {
            *out_scale = export::OUTPUT_SCALE;
            *out_unsigned = true;
            *out_point = point_count;
        }
        plan.push(ExportStep::OutputImage {
            scale: export::OUTPUT_SCALE,
        });
        plan
    }
}

fn conv_step<'a>(
    name: String,
    conv: ConvRef<'a>,
    in_point: usize,
    out_point: usize,
    relu: bool,
    out_scale: f32,
    out_unsigned: bool,
) -> ExportStep<'a> {
    ExportStep::Conv {
        name,
        conv,
        in_point,
        out_point,
        relu,
        out_scale,
        out_unsigned,
    }
}

fn push_sep_export<'a>(
    plan: &mut Vec<ExportStep<'a>>,
    name: &str,
    sep: &'a SepConv<crate::backend::InferBackend>,
    dw_in: usize,
    pw_in: usize,
    pw_out: usize,
    scales: &[f32],
    out_relu: bool,
) {
    plan.push(ExportStep::Conv {
        name: format!("{name}.dw"),
        conv: ConvRef::Depthwise(&sep.dw),
        in_point: dw_in,
        out_point: pw_in,
        relu: true,
        out_scale: scales[pw_in],
        out_unsigned: false,
    });
    let out_scale = if pw_out < scales.len() {
        scales[pw_out]
    } else {
        export::OUTPUT_SCALE
    };
    plan.push(ExportStep::Conv {
        name: format!("{name}.pw"),
        conv: ConvRef::Pointwise(&sep.pw),
        in_point: pw_in,
        out_point: pw_out,
        relu: out_relu,
        out_scale,
        out_unsigned: false,
    });
}

/// Which path [`GuidedStudent::trace`] replaces with its per-channel spatial mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ablate {
    None,
    /// The upsampled paint, just before the full-resolution guide is added.
    MeanPaint,
    /// The full-resolution RGB guide.
    MeanGuide,
}

const RES_TAPS: [&str; 4] = ["res0", "res1", "res2", "res3"];

/// Channels whose spatial variance is below this are the ones `eps = 1e-5`
/// used to pin. The threshold stays `1e-5` even when the norm itself uses
/// another epsilon, so the fraction is comparable across arms.
const IN_DEAD_VAR: f32 = 1e-5;

fn channel_dead_frac<B: Backend>(pre_in: Tensor<B, 4>) -> f32 {
    let mean = pre_in.clone().mean_dim(2).mean_dim(3);
    let var = (pre_in - mean).powf_scalar(2.0).mean_dim(2).mean_dim(3);
    let values = var
        .into_data()
        .to_vec::<f32>()
        .expect("pre-IN channel variance");
    if values.is_empty() {
        return 0.0;
    }
    let dead = values.iter().filter(|v| **v < IN_DEAD_VAR).count();
    dead as f32 / values.len() as f32
}

/// `1 - Var(y_flat) / Var(y)`, with each variance taken after subtracting the
/// per-channel spatial mean. A paint tensor that is already constant scores 0.
fn paint_spatial_contrib<B: Backend>(y: Tensor<B, 4>, y_flat: Tensor<B, 4>) -> f32 {
    let var_y = mean_spatial_var(y);
    if var_y < 1e-12 {
        return 0.0;
    }
    1.0 - mean_spatial_var(y_flat) / var_y
}

fn mean_spatial_var<B: Backend>(y: Tensor<B, 4>) -> f32 {
    let mean = y.clone().mean_dim(2).mean_dim(3);
    (y - mean).powf_scalar(2.0).mean().into_scalar().to_f64() as f32
}

fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

fn pad_spatial<B: Backend>(x: Tensor<B, 4>, multiple: usize) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    let hp = h.div_ceil(multiple) * multiple;
    let wp = w.div_ceil(multiple) * multiple;
    if hp == h && wp == w {
        x
    } else {
        x.pad([(0, hp - h), (0, wp - w)], PadMode::Reflect)
    }
}

fn crop_spatial<B: Backend>(y: Tensor<B, 4>, h: usize, w: usize) -> Tensor<B, 4> {
    let [_, _, hp, wp] = y.dims();
    if hp == h && wp == w {
        y
    } else {
        y.slice(s![.., .., 0..h, 0..w])
    }
}

/// Printed once at startup. V0 is the existing `film_both` graph, not a second model.
pub fn print_cost(spec: &GuideSpec) {
    let h = 480;
    let w = 640;
    println!("{}", v0_cost(h, w).render("v0 film_both", h, w));
    println!("{}", spec.cost(h, w).render("guided", h, w));
    println!(
        "v0 H/2 skip is 64 channels wide; an 8-row streaming FIFO at 320 px is 160 KiB. \
         guided has no skip FIFO: the guide re-reads the camera frame."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_full_res_macs_match_the_published_account() {
        let report = v0_cost(480, 640);
        let full: u64 = report
            .rows
            .iter()
            .filter(|row| row.name == "up3.1" || row.name == "v0_head")
            .map(|row| row.macs)
            .sum();
        // up3.1 (48→24 sep) + head (24→3 sep) at 640×480.
        assert_eq!(full, 575_078_400);
        let stem = report.rows.iter().find(|row| row.name == "stem").unwrap();
        assert_eq!(stem.macs, 265_420_800);
    }

    #[test]
    fn guided_stem_is_the_quarter_frame() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 24,
            guide: 8,
            full: 16,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: true,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let stem = report.rows.iter().find(|row| row.name == "stem").unwrap();
        // 320×240 × 3 × 32 × 9.
        assert_eq!(stem.macs, 66_355_200);
        let v0 = v0_cost(480, 640).macs();
        assert!(
            report.macs() * 2 < v0,
            "guided {} vs v0 {v0}",
            report.macs()
        );
    }

    #[test]
    fn v2_has_no_full_width_stack_beyond_guide_and_head() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let names: Vec<_> = report.rows.iter().map(|row| row.name.as_str()).collect();
        assert!(names.contains(&"guide"));
        assert!(names.contains(&"head"));
        assert!(!names.iter().any(|name| *name == "full"));
    }

    #[test]
    fn shuffle_two_layer_head_stays_under_two_fifths_of_v0() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 24,
            guide: 8,
            full: 16,
            full_layers: 2,
            shuffle: true,
            shuffle_guide: true,
            film_h2: true,
            film_guide: true,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let names: Vec<_> = report.rows.iter().map(|row| row.name.as_str()).collect();
        assert!(names.contains(&"sh_h8"));
        assert!(names.contains(&"full2"));
        assert!(names.contains(&"film_guide"));
        let v0 = v0_cost(480, 640).macs();
        assert!(
            report.macs() * 5 < v0 * 2,
            "v1s {} vs v0 {v0}",
            report.macs()
        );
    }

    #[test]
    fn shuffle_guide_touches_only_the_last_upsample() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: true,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let names: Vec<_> = report.rows.iter().map(|row| row.name.as_str()).collect();
        assert!(names.contains(&"sh_h2"));
        assert!(!names.contains(&"sh_h8"));
        assert!(!names.contains(&"sh_h4"));
        let sh = report.rows.iter().find(|row| row.name == "sh_h2").unwrap();
        // 320×240 × 8 channels × 4 subpixels.
        assert_eq!(sh.macs, 2_457_600);
    }

    #[test]
    fn h2_guide_is_a_half_res_dense_plus_a_pointwise_align() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 16,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let spatial = report
            .rows
            .iter()
            .find(|row| row.name == "h2_guide")
            .unwrap();
        // 320×240 × 3 × 16 × 9, two rows of the pooled RGB.
        assert_eq!(spatial.macs, 33_177_600);
        assert_eq!(spatial.line_bytes, 2 * 320 * 3);
        let proj = report
            .rows
            .iter()
            .find(|row| row.name == "h2_guide_proj")
            .unwrap();
        assert_eq!(proj.macs, 320 * 240 * 16 * 48);
        assert_eq!(proj.line_bytes, 0);
    }

    #[test]
    fn guide_kernel_five_keeps_four_input_rows() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 5,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let guide = report.rows.iter().find(|row| row.name == "guide").unwrap();
        // 640×480 × 3 × 8 × 25. Four rows, not two.
        assert_eq!(guide.macs, 184_320_000);
        assert_eq!(guide.line_bytes, 4 * 640 * 3);
    }

    #[test]
    fn quarter_trunk_keeps_the_residual_on_the_h4_grid() {
        let h8 = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let h4 = GuideSpec {
            trunk_h4: true,
            h2_guide: 16,
            ..h8.clone()
        };
        let h8_rows = h8.cost(480, 640);
        let h4_rows = h4.cost(480, 640);
        let macs = |report: &CostReport, prefix: &str| -> u64 {
            report
                .rows
                .iter()
                .filter(|row| row.name.starts_with(prefix))
                .map(|row| row.macs)
                .sum()
        };
        // 60×80 → 120×160 is four times the output pixels. Two residual seps.
        assert_eq!(macs(&h4_rows, "res0"), macs(&h8_rows, "res0") * 4);
        assert_eq!(macs(&h4_rows, "res0") - macs(&h8_rows, "res0"), 134_553_600);
        assert_eq!(macs(&h4_rows, "enc1") - macs(&h8_rows, "enc1"), 67_276_800);
        let res = h4_rows
            .rows
            .iter()
            .find(|row| row.name == "res0.s1")
            .unwrap();
        // One 3×3 row pair at H/4, 64 channels: 2 × 160 × 64.
        assert_eq!(res.line_bytes, 2 * 160 * 64);
        assert!(h4_rows.rows.iter().all(|row| row.name != "sh_h8"));
        assert!(h4_rows.rows.iter().any(|row| row.name == "h2_guide"));
    }

    #[test]
    fn resid_gain_is_a_full_res_pointwise_on_the_paint() {
        let spec = GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 0,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: true,
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let report = spec.cost(480, 640);
        let gain = report
            .rows
            .iter()
            .find(|row| row.name == "guide_gain")
            .unwrap();
        assert_eq!(gain.macs, 640 * 480 * 8 * 8);
        assert_eq!(gain.line_bytes, 0);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn h4_trunk_and_resid_gain_match_the_frame() {
        use burn::backend::NdArray;
        let device = Default::default();
        let h4 = GuideSpec {
            base: 8,
            trunk: 8,
            res_blocks: 1,
            up_h4: 8,
            up_h2: 8,
            guide: 4,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 4,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: true,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        };
        let resid = GuideSpec {
            trunk_h4: false,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: PreInGain::Default,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
            h2_guide: 0,
            guide_resid: true,
            ..h4
        };
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 / 17.0).collect();
        for spec in [h4.clone(), resid.clone()] {
            let model = GuidedStudent::<NdArray>::new(spec, &device);
            let x = Tensor::<NdArray, 4>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 3, 16, 16]),
                &device,
            );
            let y = model.stylize(x);
            assert_eq!(y.dims(), [1, 3, 16, 16]);
            let values = y.into_data().to_vec::<f32>().unwrap();
            assert!(values.iter().all(|v| v.is_finite()));
        }
        let model = GuidedStudent::<NdArray>::new(resid, &device);
        let gain = model.guide_gain.as_ref().expect("resid gain");
        let weight = gain.weight().into_data().to_vec::<f32>().unwrap();
        assert!(weight.iter().all(|v| *v == 0.0));
        let bias = gain
            .conv
            .bias
            .as_ref()
            .expect("bias")
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(bias.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn look_in_eps_stays_1e5_unless_configured() {
        let spec = GuideSpec::from_config(&StudentConfig::default()).expect("defaults");
        assert!((spec.in_eps - 1e-5).abs() < 1e-12);
        let config = StudentConfig {
            arch: "tiny_dw_guided".into(),
            options: serde_json::json!({ "in_eps": 1e-8, "full": 0, "film_full": 0 }),
        };
        let spec = GuideSpec::from_config(&config).expect("configured eps");
        assert!((spec.in_eps - 1e-8).abs() < 1e-16);
        assert_eq!(spec.pre_in_gain, PreInGain::Default);
    }

    #[cfg(feature = "ndarray")]
    fn h4_spec(gain: PreInGain) -> GuideSpec {
        GuideSpec {
            base: 32,
            trunk: 64,
            res_blocks: 1,
            up_h4: 32,
            up_h2: 48,
            guide: 8,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 16,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: true,
            in_eps: 1e-5,
            trunk_norm_kind: "in".to_string(),
            pre_in_gain: gain,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
            packet_guide: false,
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn sqrt2_gain_widens_only_the_pre_in_layers() {
        use burn::backend::NdArray;
        use burn::tensor::backend::Backend;
        type Cpu = NdArray<f32>;
        let device = Default::default();
        // Dense stem and the full-res guide share fan_in = 3·3·3. Depthwise is k².
        let dense_default = (1.0f32 / 27.0).sqrt();
        let dense_relu = (6.0f32 / 27.0).sqrt();
        let dw_default = (1.0f32 / 9.0).sqrt();
        let dw_relu = (6.0f32 / 9.0).sqrt();
        let max_abs = |w: Vec<f32>| w.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        Cpu::seed(&device, 0);
        let plain = GuidedStudent::<Cpu>::new(h4_spec(PreInGain::Default), &device);
        let stem = max_abs(plain.stem.weight().into_data().to_vec::<f32>().unwrap());
        assert!(
            stem <= dense_default * 1.001,
            "default stem bound {dense_default}, got {stem}"
        );

        Cpu::seed(&device, 0);
        let wide = GuidedStudent::<Cpu>::new(h4_spec(PreInGain::Sqrt2), &device);
        let stem = max_abs(wide.stem.weight().into_data().to_vec::<f32>().unwrap());
        let dw = max_abs(wide.enc0.dw.weight().into_data().to_vec::<f32>().unwrap());
        let guide = max_abs(wide.guide.weight().into_data().to_vec::<f32>().unwrap());
        let dec = max_abs(wide.dec_h4.dw.weight().into_data().to_vec::<f32>().unwrap());
        assert!(
            stem > dense_default,
            "sqrt2 stem should leave the default bound"
        );
        assert!(
            stem <= dense_relu * 1.001,
            "sqrt2 stem {stem} > {dense_relu}"
        );
        assert!(
            dw > dw_default,
            "sqrt2 depthwise should leave the default bound"
        );
        assert!(dw <= dw_relu * 1.001, "sqrt2 depthwise {dw} > {dw_relu}");
        assert!(
            guide <= dense_default * 1.001,
            "guide stays on the default gain, got {guide}"
        );
        assert!(
            dec <= dw_default * 1.001,
            "decoder stays on the default gain, got {dec}"
        );

        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 / 17.0).collect();
        let x = Tensor::<Cpu, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 16, 16]),
            &device,
        );
        let y_ref = wide.stylize(x.clone());
        let (y, _) = wide.render(x.clone(), false, false);
        let a = y_ref.into_data().to_vec::<f32>().unwrap();
        let b = y.into_data().to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-5));
        let signal = wide.sample_signal(x).expect("guided signal");
        assert!((0.0..=1.0).contains(&signal.in_dead));
        assert!(signal.paint_spatial.is_finite());
    }

    #[test]
    fn pre_in_gain_rejects_unknown_names() {
        let config = StudentConfig {
            arch: "tiny_dw_guided".into(),
            options: serde_json::json!({ "pre_in_gain": "he", "full": 0, "film_full": 0 }),
        };
        let err = GuideSpec::from_config(&config).expect_err("bad gain");
        assert!(err.contains("pre_in_gain"), "{err}");
    }

    #[test]
    fn trunk_norm_kind_selects_instance_or_layer_norm() {
        let spec = GuideSpec::from_config(&StudentConfig::default()).expect("defaults");
        assert_eq!(spec.trunk_norm_kind, "in");
        let ln = StudentConfig {
            arch: "tiny_dw_guided".into(),
            options: serde_json::json!({
                "trunk_norm_kind": "ln",
                "full": 0,
                "film_full": 0,
            }),
        };
        let spec = GuideSpec::from_config(&ln).expect("layer norm");
        assert_eq!(spec.trunk_norm_kind, "ln");
        let bad = StudentConfig {
            arch: "tiny_dw_guided".into(),
            options: serde_json::json!({
                "trunk_norm_kind": "bn",
                "full": 0,
                "film_full": 0,
            }),
        };
        let err = GuideSpec::from_config(&bad).expect_err("unknown norm");
        assert!(err.contains("trunk_norm_kind"), "{err}");
    }

    #[test]
    fn whitebox_ln_trial_configs_pass_the_guided_checks() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/tiny_dw_whitebox_ln/configs");
        let mut seen = 0;
        for entry in std::fs::read_dir(&dir).expect("study configs") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let config = crate::config::Config::load(&path).expect("parse");
            let spec = GuideSpec::from_config(&config.student).unwrap_or_else(|err| {
                panic!("{}: {err}", path.display());
            });
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let expect = if name == "j_ref_in" { "in" } else { "ln" };
            assert_eq!(spec.trunk_norm_kind, expect, "{name}");
            seen += 1;
        }
        assert_eq!(seen, 12);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn layer_norm_trunk_is_a_different_forward_from_instance_norm() {
        use burn::backend::NdArray;
        let device = Default::default();
        let inn = h4_spec(PreInGain::Default);
        let mut ln = h4_spec(PreInGain::Default);
        ln.trunk_norm_kind = "ln".to_string();
        let inn = GuidedStudent::<NdArray>::new(inn, &device);
        let ln = GuidedStudent::<NdArray>::new(ln, &device);
        assert_eq!(inn.trunk_norm.kind(), PluginKind::In);
        assert_eq!(ln.trunk_norm.kind(), PluginKind::Ln);
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 / 17.0).collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 16, 16]),
            &device,
        );
        let a = inn.stylize(x.clone()).into_data().to_vec::<f32>().unwrap();
        let b = ln.stylize(x).into_data().to_vec::<f32>().unwrap();
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
        let err = a
            .iter()
            .zip(b)
            .fold(0.0f32, |max, (p, q)| max.max((p - q).abs()));
        assert!(err > 1e-3, "ln and in agreed within {err}");
    }

    #[cfg(all(
        feature = "ndarray",
        not(feature = "cuda"),
        not(feature = "wgpu"),
        not(feature = "vulkan")
    ))]
    #[test]
    fn guided_manifest_matches_the_fake_quant_forward() {
        use crate::backend::InferBackend;
        use crate::intref::IntModel;
        use crate::quant::{Bits, Calibration, Quantizable};

        let device = Default::default();
        let n = 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 19) as f32 / 19.0).collect();
        for (kind, res_blocks) in [("in", 1), ("ln", 1), ("ln", 0)] {
            let mut spec = h4_spec(PreInGain::Default);
            spec.trunk_norm_kind = kind.to_string();
            spec.res_blocks = res_blocks;
            let model = GuidedStudent::<InferBackend>::new(spec, &device);
            let x = Tensor::<InferBackend, 4>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 3, 16, 16]),
                &device,
            );
            let points = model.quant_points();
            assert!(!points.is_empty(), "{kind}");
            let mut absmax = vec![0.0f32; points.len()];
            model.observe_activations(x.clone(), &mut |point, value| {
                absmax[point] = absmax[point].max(value);
            });
            let scales: Vec<f32> = absmax
                .iter()
                .map(|value| (value / 127.0).max(1.0e-3))
                .collect();
            let calibration = Calibration {
                arch: "tiny_dw_guided".into(),
                points: points.clone(),
                scales: scales.clone(),
                absmax,
                safety: 1.0,
                batches: 1,
                bits: Bits::new(8, 8),
            };
            let dir =
                std::env::temp_dir().join(format!("guided_manifest_parity_{kind}_{res_blocks}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let manifest = export::export_int8(&model, &calibration, &dir).expect("export");
            assert!(manifest.graph.iter().any(|op| op.op == "avg_pool2x"));
            let norm_op = if kind == "ln" {
                "layer_norm"
            } else {
                "instance_norm"
            };
            assert!(
                manifest.graph.iter().any(|op| op.op == norm_op),
                "{kind} {res_blocks}"
            );
            let sim = model
                .forward_quantised(x, &calibration.scheme())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let int = IntModel::load(&dir)
                .unwrap()
                .forward_chw(&data, 16, 16)
                .unwrap();
            assert_eq!(sim.len(), int.len());
            let mean = sim
                .iter()
                .zip(&int)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / sim.len() as f32;
            assert!(
                mean < 0.02,
                "{kind} {res_blocks} sim vs int mean abs {mean}"
            );
        }
    }

    fn small_packet_spec(extra: serde_json::Value) -> GuideSpec {
        let mut options = serde_json::json!({
            "base": 4,
            "trunk": 4,
            "res_blocks": 0,
            "up_h4": 4,
            "up_h2": 4,
            "guide": 2,
            "full": 0,
            "film_h2": 1,
            "film_full": 0,
            "h2_guide": 2,
            "trunk_norm_kind": "ln",
            "trunk_at": "h4",
        });
        if let (Some(dest), Some(src)) = (options.as_object_mut(), extra.as_object()) {
            for (key, value) in src {
                dest.insert(key.clone(), value.clone());
            }
        }
        GuideSpec::from_config(&StudentConfig {
            arch: "tiny_dw_guided".into(),
            options,
        })
        .expect("small spec")
    }

    #[test]
    fn rgb565_lands_on_the_565_codebook() {
        use burn::backend::NdArray;
        type Cpu = NdArray<f32>;
        let device = Default::default();
        let x = Tensor::<Cpu, 4>::from_data(
            burn::tensor::TensorData::new(vec![0.2f32, 0.8, 0.0], [1, 3, 1, 1]),
            &device,
        );
        let y = quantize_rgb565(x).into_data().to_vec::<f32>().unwrap();
        let expect = [
            (0.2f32 * 31.0).round() / 31.0,
            (0.8f32 * 63.0).round() / 63.0,
            0.0f32,
        ];
        for (got, want) in y.iter().zip(expect) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        assert!((y[0] - 0.2).abs() > 1e-3);
    }

    #[test]
    fn rgb565_rejects_a_two_plane_input() {
        let err = GuideSpec::from_config(&StudentConfig {
            arch: "tiny_dw_guided".into(),
            options: serde_json::json!({
                "input": "yc",
                "input_quant": "rgb565",
                "full": 0,
                "film_full": 0,
            }),
        })
        .expect_err("yc+565");
        assert!(err.contains("rgb565"), "{err}");
    }

    #[test]
    fn expand_styles_copies_the_packet_into_every_slot() {
        use burn::backend::NdArray;
        type Cpu = NdArray<f32>;
        let device = Default::default();
        let model = GuidedStudent::<Cpu>::new(small_packet_spec(serde_json::json!({})), &device);
        let wide = model.expand_styles(2, &device).expect("expand");
        assert!(wide.num_params() > model.num_params());
        assert_eq!(wide.style_count(), 2);
        let x = Tensor::<Cpu, 4>::zeros([1, 3, 8, 8], &device);
        let source = model
            .stylize(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let style0 = wide
            .stylize_style(x.clone(), 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let style1 = wide
            .stylize_style(x, 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for ((a, b), c) in source.iter().zip(&style0).zip(&style1) {
            assert!((a - b).abs() < 1e-5 && (b - c).abs() < 1e-5, "{a} {b} {c}");
        }
    }

    #[test]
    fn one_style_checkpoint_reloads() {
        use burn::backend::NdArray;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        type Cpu = NdArray<f32>;
        let device = Default::default();
        let spec = small_packet_spec(serde_json::json!({}));
        let model = GuidedStudent::<Cpu>::new(spec.clone(), &device);
        let x = Tensor::<Cpu, 4>::zeros([1, 3, 8, 8], &device);
        let before = model
            .stylize(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let path =
            std::env::temp_dir().join(format!("fpga_st_one_style_{}.mpk", std::process::id()));
        model
            .save_file(&path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
            .expect("save");
        let loaded = GuidedStudent::<Cpu>::new(spec, &device)
            .load_file(
                &path,
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .expect("load one-style checkpoint");
        let after = loaded.stylize(x).into_data().to_vec::<f32>().unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn packet_guide_starts_as_identity_and_moves_one_style() {
        use burn::backend::NdArray;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        type Cpu = NdArray<f32>;
        let device = Default::default();
        let model = GuidedStudent::<Cpu>::new(small_packet_spec(serde_json::json!({})), &device);
        let two = model.expand_styles(2, &device).expect("expand");
        let wide = two.repack_styles(3, true, &device).expect("repack");
        assert!(wide.spec.packet_guide);
        assert_eq!(wide.style_count(), 3);
        // One more packet, plus a 2-channel gain and bias for each of the 3 styles.
        let packet = two.num_params() - model.num_params();
        assert_eq!(wide.num_params(), two.num_params() + packet + 3 * 2 * 2);
        let ramp = Tensor::<Cpu, 1, Int>::arange(0..3 * 8 * 8, &device)
            .float()
            .div_scalar(192.0)
            .reshape([1, 3, 8, 8]);
        let out = |m: &GuidedStudent<Cpu>, style: usize| {
            m.stylize_style(ramp.clone(), style)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let source = out(&model, 0);
        for style in 0..3 {
            for (a, b) in source.iter().zip(&out(&wide, style)) {
                assert!((a - b).abs() < 1e-5, "style {style}: {a} vs {b}");
            }
        }
        // A bias on style 2 moves its guide map; styles 0 and 1 stay put. The tap is
        // checked rather than the RGB: a random head can gate the guide off entirely.
        let guide_tap = |m: &GuidedStudent<Cpu>, style: usize| {
            let mut seen = Vec::new();
            m.render_tapped(ramp.clone(), Ablate::None, false, style, &mut |name, t| {
                if name == "guide" {
                    seen = t.clone().into_data().to_vec::<f32>().unwrap();
                }
            });
            seen
        };
        let before = guide_tap(&wide, 2);
        let mut moved = wide;
        moved.style_packets[1].guide_affine.as_mut().unwrap().bias =
            Param::from_tensor(Tensor::from_floats([1.0, 1.0], &device));
        let after = guide_tap(&moved, 2);
        for (a, b) in before.iter().zip(&after) {
            // relu(x + 1) - relu(x) is in [0, 1].
            assert!(*b >= *a - 1e-6 && *b - *a <= 1.0 + 1e-5, "{a} -> {b}");
        }
        assert!(after.iter().sum::<f32>() > before.iter().sum::<f32>() + 1.0);
        let closed = out(&moved, 2);
        for style in 0..2 {
            for (a, b) in source.iter().zip(&out(&moved, style)) {
                assert!((a - b).abs() < 1e-5, "style {style} moved: {a} vs {b}");
            }
        }
        // The affine survives a save and reload under the packet_guide spec.
        let path =
            std::env::temp_dir().join(format!("fpga_st_packet_guide_{}.mpk", std::process::id()));
        let spec = moved.spec.clone();
        moved
            .save_file(&path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
            .expect("save");
        let loaded = GuidedStudent::<Cpu>::new(spec, &device)
            .load_file(
                &path,
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .expect("load packet_guide checkpoint");
        for (a, b) in closed.iter().zip(&out(&loaded, 2)) {
            assert!((a - b).abs() < 1e-5, "reload: {a} vs {b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn split_style_folds_the_guide_affine_into_one_style() {
        use burn::backend::NdArray;
        type Cpu = NdArray<f32>;
        let device = Default::default();
        let model = GuidedStudent::<Cpu>::new(small_packet_spec(serde_json::json!({})), &device);
        let mut wide = model.repack_styles(2, true, &device).expect("repack");
        let affine = wide.style_packets[0].guide_affine.as_mut().unwrap();
        affine.log_gain = Param::from_tensor(Tensor::from_floats([0.3, -0.2], &device));
        affine.bias = Param::from_tensor(Tensor::from_floats([-0.05, 0.1], &device));
        let ramp = Tensor::<Cpu, 1, Int>::arange(0..3 * 8 * 8, &device)
            .float()
            .div_scalar(192.0)
            .reshape([1, 3, 8, 8]);
        for style in 0..2 {
            let single = wide.split_style(style, &device).expect("split");
            assert_eq!(single.style_count(), 1);
            assert!(!single.spec.packet_guide);
            assert!(single.export_block_reason().is_none() || model.export_block_reason().is_some());
            let want = wide
                .stylize_style(ramp.clone(), style)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let got = single.stylize(ramp.clone()).into_data().to_vec::<f32>().unwrap();
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-5, "style {style}: {a} vs {b}");
            }
        }
        assert!(wide.split_style(2, &device).is_err());
    }
}
