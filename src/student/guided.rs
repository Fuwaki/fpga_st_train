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
//! head. Integer export is intentionally not implemented; this is an fp32 architecture trial.

use burn::module::Module;
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::{avg_pool2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

use super::plugin::{Film, Plugin, PluginKind};
use super::tiny::{Conv, ConvInit};
use super::tiny_dw::{PwConv, ResBlockDw, SepConv};
use super::{SampleSignal, StudentModel};
use crate::config::StudentConfig;
use crate::quant::{self, QuantScheme};

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
#[derive(Clone, Copy, Debug)]
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
    /// Init gain for stem, enc0, enc1, and the residual blocks.
    pub pre_in_gain: PreInGain,
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
            pre_in_gain: match config.str_option("pre_in_gain", "default") {
                "default" => PreInGain::Default,
                "sqrt2" => PreInGain::Sqrt2,
                other => {
                    return Err(format!(
                        "tiny_dw_guided pre_in_gain must be default or sqrt2, got {other}"
                    ));
                }
            },
        };
        spec.check()?;
        Ok(spec)
    }

    fn check(self) -> Result<(), String> {
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
        Ok(())
    }

    /// MAC and A8 line-buffer account at one deployment size.
    ///
    /// Same MAC rule as `runs/note/v7-input-precision.md` §5: dense is `cin·cout·k²`
    /// per output pixel, depthwise is `cin·k²`. A 3×3 line buffer is two input rows.
    pub fn cost(self, height: usize, width: usize) -> CostReport {
        let mut rows = Vec::new();
        let h2 = height / 2;
        let w2 = width / 2;
        let h4 = height / 4;
        let w4 = width / 4;
        let h8 = height / 8;
        let w8 = width / 8;
        // The pool is a handful of adds. It is not in the MAC total.
        push_dense(&mut rows, "stem", h2, w2, 3, self.base, 3, width / 2);
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
                3,
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
            3,
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
        Self {
            stem: Conv::with_init(3, spec.base, 3, 1, true, pre_in, device),
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
            trunk_norm: Plugin::instance_norm(spec.trunk, spec.in_eps, device),
            shuffle_h8: if spec.trunk_h4 {
                None
            } else {
                shuffle_trunk(spec.trunk)
            },
            dec_h4: SepConv::new(spec.trunk, spec.up_h4, 3, 1, device),
            shuffle_h4: shuffle_trunk(spec.up_h4),
            dec_h2: SepConv::new(spec.up_h4, spec.up_h2, 3, 1, device),
            h2_guide: (spec.h2_guide > 0).then(|| Conv::new(3, spec.h2_guide, 3, 1, true, device)),
            h2_guide_proj: (spec.h2_guide > 0 && spec.h2_guide != spec.up_h2)
                .then(|| PwConv::new(spec.h2_guide, spec.up_h2, device)),
            film_h2,
            to_guide: PwConv::new(spec.up_h2, spec.guide, device),
            shuffle_guide,
            guide: Conv::new(3, spec.guide, spec.guide_kernel, 1, true, device),
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
            spec,
        }
    }

    pub fn spec(&self) -> GuideSpec {
        self.spec
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
        self.render_tapped(content, ablate, probe_dead, &mut |_, _| {})
    }

    /// Every stage of the forward, in order, for probes. `tap` sees each map
    /// once; the output is the same tensor [`Self::forward`] returns.
    pub fn trace(
        &self,
        content: Tensor<B, 4>,
        ablate: Ablate,
        tap: &mut dyn FnMut(&'static str, &Tensor<B, 4>),
    ) -> Tensor<B, 4> {
        self.render_tapped(content, ablate, false, tap).0
    }

    fn render_tapped(
        &self,
        content: Tensor<B, 4>,
        ablate: Ablate,
        probe_dead: bool,
        tap: &mut dyn FnMut(&'static str, &Tensor<B, 4>),
    ) -> (Tensor<B, 4>, Option<f32>) {
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
        f = self.trunk_norm.forward(f);
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
        if let Some(film) = &self.film_h2 {
            f = film.forward(f);
            tap("film_h2", &f);
        }
        // Project at H/2, then spread the 8-channel paint. The guide is what
        // breaks the 2×2 blocks back onto the original edges.
        f = relu(self.to_guide.forward(f));
        tap("paint", &f);
        f = self.up2(f, &self.shuffle_guide);
        let guide = self.guide.forward(x);
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
        format!(
            "tiny_dw_guided pool=2 trunk={trunk} up={up} up_h4={} up_h2={} h2_guide={} guide={} k={} guide_act={guide_act} mix={mix} full={full} in_eps={:.0e} pre_in_gain={} ({} parameters)",
            self.spec.up_h4,
            self.spec.up_h2,
            self.spec.h2_guide,
            self.spec.guide,
            self.spec.guide_kernel,
            self.spec.in_eps,
            self.spec.pre_in_gain.label(),
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

impl<B: Backend> quant::Quantizable<B> for GuidedStudent<B> {
    fn quant_points(&self) -> Vec<String> {
        Vec::new()
    }

    fn forward_quantised(&self, content: Tensor<B, 4>, _scheme: &QuantScheme) -> Tensor<B, 4> {
        self.stylize(content)
    }

    fn observe_activations(&self, _content: Tensor<B, 4>, _observe: &mut dyn FnMut(usize, f32)) {}
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
pub fn print_cost(spec: GuideSpec) {
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
        };
        let h4 = GuideSpec {
            trunk_h4: true,
            h2_guide: 16,
            ..h8
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
            pre_in_gain: PreInGain::Default,
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
            pre_in_gain: PreInGain::Default,
        };
        let resid = GuideSpec {
            trunk_h4: false,
            in_eps: 1e-5,
            pre_in_gain: PreInGain::Default,
            h2_guide: 0,
            guide_resid: true,
            ..h4
        };
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 / 17.0).collect();
        for spec in [h4, resid] {
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
            pre_in_gain: gain,
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
}
