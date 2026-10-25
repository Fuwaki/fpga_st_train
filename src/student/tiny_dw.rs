//! MobileNet-style depthwise-separable student for FPGA-friendly deployment.
//!
//! Same topology as [super::tiny::TinyStudent] (encoder -> residual trunk -> nearest
//! upsampling decoder), but every 3x3 convolution is split into depthwise 3x3 + pointwise
//! 1x1. On hardware the depthwise stage is one MAC per channel per tap; the pointwise stage
//! is a small 1x1 mixer. Both map cleanly onto a line-buffer datapath without grouped or
//! transposed convolutions beyond depthwise.
//!
//! One additive skip, at 1/2 resolution only. The H/4 trunk is where the style is mixed,
//! and it is also where content edges die: two stride-2 layers leave the decoder a 4 px
//! grid, which is enough for a tint and not enough to place a stroke. `enc[0]` is already
//! 64 channels at H/2, the same shape as the upsampled trunk, so the skip is an add.
//! No concatenation, no extra convolution, no new parameters. A full-resolution skip
//! is left out: the stem is 32 channels at full resolution, twice the bytes of this
//! 64-channel H/2 map, and the brush strokes on the Van Gogh teacher correlate over
//! 4–8 px, which the H/2 map can still locate.

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

use super::StudentModel;
use super::conv::conv2d_valid;
use super::lut::{LutDwEncoder, LutEncoder};
use super::plugin::{Film, Plugin, PluginKind, PluginSite};
use super::tiny::{Conv, ConvInit};
use crate::quant;

/// ReflectionPad(k/2) + depthwise Conv2d (groups = in_channels).
#[derive(Module, Debug)]
pub struct DwConv<B: Backend> {
    pub conv: Conv2d<B>,
    pad: usize,
}

impl<B: Backend> DwConv<B> {
    pub fn new(channels: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        Self::with_init(channels, kernel, stride, ConvInit::Default, device)
    }

    pub fn with_init(
        channels: usize,
        kernel: usize,
        stride: usize,
        init: ConvInit,
        device: &B::Device,
    ) -> Self {
        let conv = init
            .apply(
                Conv2dConfig::new([channels, channels], [kernel, kernel])
                    .with_stride([stride, stride])
                    .with_groups(channels)
                    .with_padding(PaddingConfig2d::Valid)
                    .with_bias(true),
            )
            .init(device);
        Self {
            conv,
            pad: kernel / 2,
        }
    }

    pub fn reflect_pad(&self) -> usize {
        self.pad
    }

    pub fn weight(&self) -> Tensor<B, 4> {
        self.conv.weight.val()
    }

    pub fn forward_with(&self, x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = if self.pad > 0 {
            x.pad(
                [(self.pad, self.pad), (self.pad, self.pad)],
                PadMode::Reflect,
            )
        } else {
            x
        };
        let bias = self.conv.bias.as_ref().map(|b| b.val());
        conv2d_valid(
            x,
            weight,
            bias,
            self.conv.stride,
            self.conv.dilation,
            self.conv.groups,
        )
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with(x, self.weight())
    }
}

/// Pointwise 1x1 Conv2d.
#[derive(Module, Debug)]
pub struct PwConv<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> PwConv<B> {
    pub fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        Self::with_init(in_ch, out_ch, ConvInit::Default, device)
    }

    pub fn with_init(in_ch: usize, out_ch: usize, init: ConvInit, device: &B::Device) -> Self {
        let conv = init
            .apply(
                Conv2dConfig::new([in_ch, out_ch], [1, 1])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .with_bias(true),
            )
            .init(device);
        Self { conv }
    }

    pub fn weight(&self) -> Tensor<B, 4> {
        self.conv.weight.val()
    }

    pub fn forward_with(&self, x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
        let bias = self.conv.bias.as_ref().map(|b| b.val());
        conv2d_valid(
            x,
            weight,
            bias,
            self.conv.stride,
            self.conv.dilation,
            self.conv.groups,
        )
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with(x, self.weight())
    }

    pub(crate) fn zero(&mut self, device: &B::Device) {
        let [out, inn, kh, kw] = self.conv.weight.dims();
        self.conv.weight = Param::from_tensor(Tensor::<B, 4>::zeros([out, inn, kh, kw], device));
        if let Some(bias) = self.conv.bias.as_mut() {
            let n = bias.dims()[0];
            *bias = Param::from_tensor(Tensor::<B, 1>::zeros([n], device));
        }
    }
}

/// Depthwise 3x3 followed by pointwise 1x1 (MobileNet separable block).
#[derive(Module, Debug)]
pub struct SepConv<B: Backend> {
    pub dw: DwConv<B>,
    pub pw: PwConv<B>,
}

impl<B: Backend> SepConv<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        Self::with_init(in_ch, out_ch, kernel, stride, ConvInit::Default, device)
    }

    pub fn with_init(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        init: ConvInit,
        device: &B::Device,
    ) -> Self {
        Self {
            dw: DwConv::with_init(in_ch, kernel, stride, init, device),
            pw: PwConv::with_init(in_ch, out_ch, init, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_inner(x, true)
    }

    /// Depthwise + ReLU + pointwise. `out_relu` is false for the residual addend and the RGB head,
    /// matching the dense student: those outputs must stay signed.
    pub(crate) fn forward_inner(&self, x: Tensor<B, 4>, out_relu: bool) -> Tensor<B, 4> {
        let y = self.pw.forward(relu(self.dw.forward(x)));
        if out_relu { relu(y) } else { y }
    }

    fn forward_quant<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        x: Tensor<B, 4>,
        dw_point: usize,
        pw_point: usize,
        scheme: Option<&quant::QuantScheme>,
        observe: &mut F,
        wq_dw: Tensor<B, 4>,
        wq_pw: Tensor<B, 4>,
        out_relu: bool,
    ) -> Tensor<B, 4> {
        let h = qa(x, scheme, dw_point, observe);
        let h = relu(self.dw.forward_with(h, wq_dw));
        let h = qa(h, scheme, pw_point, observe);
        let y = self.pw.forward_with(h, wq_pw);
        if out_relu { relu(y) } else { y }
    }
}

fn qa<B: Backend, F: FnMut(usize, &Tensor<B, 4>)>(
    x: Tensor<B, 4>,
    scheme: Option<&quant::QuantScheme>,
    point: usize,
    observe: &mut F,
) -> Tensor<B, 4> {
    observe(point, &x);
    match scheme {
        Some(scheme) => quant::fake_quant_act(x, scheme.acts[point], scheme.act_max()),
        None => x,
    }
}

fn wq_dw<B: Backend>(conv: &DwConv<B>, scheme: Option<&quant::QuantScheme>) -> Tensor<B, 4> {
    let weight = conv.weight();
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

fn wq_pw<B: Backend>(conv: &PwConv<B>, scheme: Option<&quant::QuantScheme>) -> Tensor<B, 4> {
    let weight = conv.weight();
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

/// Two separable 3x3 blocks with a residual connection.
#[derive(Module, Debug)]
pub struct ResBlockDw<B: Backend> {
    pub s1: SepConv<B>,
    pub s2: SepConv<B>,
}

impl<B: Backend> ResBlockDw<B> {
    pub(crate) fn new(channels: usize, device: &B::Device) -> Self {
        Self::with_init(channels, ConvInit::Default, device)
    }

    pub(crate) fn with_init(channels: usize, init: ConvInit, device: &B::Device) -> Self {
        Self {
            s1: SepConv::with_init(channels, channels, 3, 1, init, device),
            s2: SepConv::with_init(channels, channels, 3, 1, init, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mid = self.s1.forward(x.clone());
        let inner = self.s2.forward_inner(mid, false);
        x + inner
    }
}

fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// Which same-width decoder 1×1 is a per-image FiLM instead of a pointwise mix.
///
/// The separable blocks, the nearest upsamples, and the RGB head stay. FiLM is a
/// global pool followed by per-channel γ, β, and it starts as the identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPath {
    None,
    /// Replace the extra H/2 1×1 (`mid` channels).
    FilmHalf,
    /// Replace the extra full-resolution 1×1 (`fine` channels).
    FilmFull,
    /// Replace both extra 1×1s.
    FilmBoth,
}

impl ColorPath {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FilmHalf => "film_half",
            Self::FilmFull => "film_full",
            Self::FilmBoth => "film_both",
        }
    }

    fn replaces_half(self) -> bool {
        matches!(self, Self::FilmHalf | Self::FilmBoth)
    }

    fn replaces_full(self) -> bool {
        matches!(self, Self::FilmFull | Self::FilmBoth)
    }
}

pub fn parse_color(name: &str) -> Result<ColorPath, String> {
    match name {
        "none" => Ok(ColorPath::None),
        "film_half" => Ok(ColorPath::FilmHalf),
        "film_full" => Ok(ColorPath::FilmFull),
        "film_both" => Ok(ColorPath::FilmBoth),
        other => Err(format!(
            "unknown tiny_dw color {other:?}; expected none, film_half, film_full, film_both"
        )),
    }
}

/// `h8` inserts one more stride-2 so the residual trunk is H/8. `h4` is the
/// shipped two-stage encoder.
pub fn parse_trunk_h8(name: &str) -> Result<bool, String> {
    match name {
        "h4" => Ok(false),
        "h8" => Ok(true),
        other => Err(format!("tiny_dw trunk_at must be h4 or h8, got {other}")),
    }
}

/// Which stack produces the H/2 skip and the H/4 trunk.
///
/// `Sep` is the depthwise stem the existing checkpoints were trained with.
/// `Lut` replaces that stem with three dense LUT-Conv stages.
/// `LutDw` keeps the dense stem and both pointwise mixers, and replaces only
/// the two stride-2 depthwise 3×3s with LUT-Conv.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    Sep,
    Lut,
    LutDw,
}

impl EncoderKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Sep => "sep",
            Self::Lut => "lut",
            Self::LutDw => "lut_dw",
        }
    }
}

pub fn parse_encoder(name: &str) -> Result<EncoderKind, String> {
    match name {
        "sep" | "conv" => Ok(EncoderKind::Sep),
        "lut" => Ok(EncoderKind::Lut),
        "lut_dw" => Ok(EncoderKind::LutDw),
        other => Err(format!(
            "unknown tiny_dw encoder {other:?}; expected sep, lut, or lut_dw"
        )),
    }
}

/// `0` is the float LUT surrogate. `2..=8` is the symmetric codebook width.
pub fn parse_lut_bits(bits: usize) -> Result<u8, String> {
    if bits == 0 || (2..=8).contains(&bits) {
        Ok(bits as u8)
    } else {
        Err(format!(
            "lut_bits {bits} is outside 0 or 2..=8 (0 keeps the float surrogate)"
        ))
    }
}

/// Depthwise-separable variant of [super::tiny::TinyStudent].
#[derive(Module, Debug)]
pub struct TinyDwStudent<B: Backend> {
    /// Dense 3x3 stem. A depthwise stem cannot mix RGB inside the kernel.
    /// Present for either encoder so a separable checkpoint still has this field.
    /// `encoder = lut` does not read it.
    pub stem: Conv<B>,
    pub enc: Vec<SepConv<B>>,
    /// Dense LUT replacement for `stem` and `enc`. `None` unless `encoder = lut`.
    lut_encoder: Option<LutEncoder<B>>,
    /// Depthwise LUT replacement for the two stride-2 spatial kernels.
    /// `None` unless `encoder = lut_dw`. The stem and the pointwise mixers stay live.
    lut_dw: Option<LutDwEncoder<B>>,
    pub res: Vec<ResBlockDw<B>>,
    pub up3: Vec<SepConv<B>>,
    /// Same-width decoder 1×1s that were not replaced by FiLM, half then full.
    pub up1: Vec<PwConv<B>>,
    film_half: Option<Film<B>>,
    film_full: Option<Film<B>>,
    pub head: SepConv<B>,
    plugin: Plugin<B>,
    #[module(skip)]
    plugin_site: PluginSite,
    #[module(skip)]
    color: ColorPath,
    #[module(skip)]
    encoder: EncoderKind,
    /// Third stride-2 after the H/4 map, so the residual mixes at H/8.
    /// A bare nearest lifts that map back to H/4 before the existing upsample
    /// meets the H/2 skip. Integer export stays on the two-stage trunk.
    #[module(skip)]
    trunk_h8: bool,
}

impl<B: Backend> TinyDwStudent<B> {
    /// Build from the `plugin` / `plugin_site` strings stored in a model card.
    pub fn from_named_plugin(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        plugin: &str,
        plugin_site: &str,
        device: &B::Device,
    ) -> Result<Self, String> {
        let kind = super::plugin::parse_kind(plugin)?;
        let site = super::plugin::parse_site(plugin_site)?;
        Ok(Self::with_plugin(
            base, res_blocks, mid, fine, kind, site, device,
        ))
    }

    pub fn new(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        device: &B::Device,
    ) -> Self {
        Self::with_plugin(
            base,
            res_blocks,
            mid,
            fine,
            PluginKind::None,
            PluginSite::H4,
            device,
        )
    }

    /// `plugin_site` selects the tensor a shape-preserving plugin sees.
    /// `Shuffle` replaces the H/4 nearest upsample and ignores the site.
    pub fn with_plugin(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        plugin: PluginKind,
        plugin_site: PluginSite,
        device: &B::Device,
    ) -> Self {
        Self::with_color(
            base,
            res_blocks,
            mid,
            fine,
            plugin,
            plugin_site,
            ColorPath::None,
            device,
        )
    }

    /// `color` replaces a same-width decoder 1×1 with FiLM. `None` is the ordinary decoder.
    pub fn with_color(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        plugin: PluginKind,
        plugin_site: PluginSite,
        color: ColorPath,
        device: &B::Device,
    ) -> Self {
        Self::with_encoder(
            base,
            res_blocks,
            mid,
            fine,
            plugin,
            plugin_site,
            color,
            EncoderKind::Sep,
            4,
            8,
            false,
            device,
        )
    }

    /// `encoder = lut` swaps the stem for LUT-Conv and keeps this decoder.
    /// `lut_hidden` is the MLP width inside each logical LUT. `lut_bits` is the
    /// codebook width; `0` leaves the surrogate in float.
    ///
    /// `trunk_h8` adds a third separable stride-2 after the usual H/4 map and
    /// one nearest upsample with no convolution before the H/2 skip. Only the
    /// separable encoder supports it.
    pub fn with_encoder(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        plugin: PluginKind,
        plugin_site: PluginSite,
        color: ColorPath,
        encoder: EncoderKind,
        lut_hidden: usize,
        lut_bits: u8,
        trunk_h8: bool,
        device: &B::Device,
    ) -> Self {
        let c1 = base;
        let c2 = base * 2;
        let plugin_channels = if plugin == PluginKind::Shuffle {
            c2
        } else {
            match plugin_site {
                PluginSite::H4 => c2,
                PluginSite::H2 => mid,
                PluginSite::Full => fine,
            }
        };
        if trunk_h8 && encoder != EncoderKind::Sep {
            panic!("tiny_dw trunk_at=h8 requires encoder sep; lut and lut_dw stay at H/4");
        }
        let stem = Conv::new(3, c1, 3, 1, true, device);
        let mut enc = vec![
            SepConv::new(c1, c2, 3, 2, device),
            SepConv::new(c2, c2, 3, 2, device),
        ];
        if trunk_h8 {
            enc.push(SepConv::new(c2, c2, 3, 2, device));
        }
        let res = (0..res_blocks)
            .map(|_| ResBlockDw::new(c2, device))
            .collect();
        let up3 = vec![
            SepConv::new(c2, mid, 3, 1, device),
            SepConv::new(mid, fine, 3, 1, device),
        ];
        let mut up1 = Vec::new();
        let film_half = if color.replaces_half() {
            Some(Film::new(mid, device))
        } else {
            up1.push(PwConv::new(mid, mid, device));
            None
        };
        let film_full = if color.replaces_full() {
            Some(Film::new(fine, device))
        } else {
            up1.push(PwConv::new(fine, fine, device));
            None
        };
        let head = SepConv::new(fine, 3, 3, 1, device);
        let lut_encoder = match encoder {
            EncoderKind::Lut => Some(LutEncoder::dw(base, lut_hidden, lut_bits, device)),
            EncoderKind::Sep | EncoderKind::LutDw => None,
        };
        let lut_dw = match encoder {
            EncoderKind::LutDw => Some(LutDwEncoder::new(base, lut_hidden, lut_bits, device)),
            EncoderKind::Sep | EncoderKind::Lut => None,
        };
        Self {
            stem,
            enc,
            lut_encoder,
            lut_dw,
            res,
            up3,
            up1,
            film_half,
            film_full,
            head,
            plugin: Plugin::new(plugin, plugin_channels, device),
            plugin_site,
            color,
            encoder,
            trunk_h8,
        }
    }

    /// Extra H/2 1×1, or FiLM when that slot was replaced. No ReLU after FiLM.
    fn half_color(&self, h: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.film_half {
            Some(film) => film.forward(h),
            None => relu(self.up1[0].forward(h)),
        }
    }

    /// Extra full-resolution 1×1, or FiLM when that slot was replaced.
    fn full_color(&self, h: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.film_full {
            Some(film) => film.forward(h),
            None => {
                // Half stays at index 0 when it is still a pointwise.
                let index = usize::from(self.film_half.is_none());
                relu(self.up1[index].forward(h))
            }
        }
    }

    fn half_color_quant<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        h: Tensor<B, 4>,
        scheme: Option<&quant::QuantScheme>,
        point: usize,
        observe: &mut F,
    ) -> Tensor<B, 4> {
        if let Some(film) = &self.film_half {
            observe(point, &h);
            return film.forward(h);
        }
        let h = qa(h, scheme, point, observe);
        relu(self.up1[0].forward_with(h, wq_pw(&self.up1[0], scheme)))
    }

    fn full_color_quant<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        h: Tensor<B, 4>,
        scheme: Option<&quant::QuantScheme>,
        point: usize,
        observe: &mut F,
    ) -> Tensor<B, 4> {
        if let Some(film) = &self.film_full {
            observe(point, &h);
            return film.forward(h);
        }
        let h = qa(h, scheme, point, observe);
        let index = usize::from(self.film_half.is_none());
        relu(self.up1[index].forward_with(h, wq_pw(&self.up1[index], scheme)))
    }

    fn at(&self, here: PluginSite, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.plugin_site == here && !self.plugin.is_shuffle() {
            self.plugin.forward(x)
        } else {
            x
        }
    }

    fn upsample_from_trunk(&self, h: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.plugin.is_shuffle() {
            self.plugin.shuffle_up(h)
        } else {
            upsample2x(h)
        }
    }

    /// Nearest steps from the residual trunk onto the H/2 skip.
    ///
    /// The default trunk is already H/4, so this is one upsample. `trunk_at = h8`
    /// inserts a bare nearest H/8 → H/4 first. No convolution sits between the
    /// two, so the ablation only moves where the residual mixing happens.
    fn upsample_trunk_to_skip(&self, h: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = if self.trunk_h8 { upsample2x(h) } else { h };
        self.upsample_from_trunk(h)
    }

    pub(crate) fn export_plugin_site(&self) -> PluginSite {
        self.plugin_site
    }

    pub(crate) fn export_plugin(&self) -> &Plugin<B> {
        &self.plugin
    }

    pub(crate) fn export_film_half(&self) -> Option<&Film<B>> {
        self.film_half.as_ref()
    }

    pub(crate) fn export_film_full(&self) -> Option<&Film<B>> {
        self.film_full.as_ref()
    }

    pub(crate) fn export_lut_dw(&self) -> Option<&LutDwEncoder<B>> {
        self.lut_dw.as_ref()
    }

    pub fn blocks_integer_export(&self) -> Option<String> {
        if self.trunk_h8 {
            return Some(
                "tiny_dw trunk_at=h8 is an fp32 ablation; integer export assumes the H/4 trunk"
                    .to_owned(),
            );
        }
        if self.encoder == EncoderKind::Lut {
            return Some(
                "tiny_dw encoder `lut` is not merged into the int8 decoder export yet; use `lut_dw` or `sep`"
                    .to_owned(),
            );
        }
        if self.encoder == EncoderKind::LutDw {
            let bits = self.lut_dw.as_ref().map(|lut| lut.bits()).unwrap_or(0);
            if bits < 2 {
                return Some(
                    "lut_dw needs lut_bits in 2..=8 for truth-table export; 0 keeps the float MLP surrogate"
                        .to_owned(),
                );
            }
        }
        match self.plugin.kind() {
            PluginKind::None | PluginKind::In => {}
            other => {
                return Some(format!(
                    "tiny_dw plugin `{}` is not in the integer export graph",
                    other.name()
                ));
            }
        }
        if self.plugin.is_shuffle() {
            return Some("pixel-shuffle upsample is not in the integer export graph".to_owned());
        }
        None
    }

    /// Stages that own activation quant points. The H/8 ablation's third
    /// stride-2 stays in float and is not one of them.
    fn counted_enc_stages(&self) -> usize {
        self.enc.len().min(2)
    }

    /// One quantised point per convolution input (dw + pw for each SepConv, plus both color slots).
    ///
    /// A FiLM slot still occupies its activation index. Integer export rejects that graph.
    pub fn quant_point_count(&self) -> usize {
        1 + 2 * self.counted_enc_stages() + 4 * self.res.len() + 2 * self.up3.len() + 2 + 2
    }

    pub fn quant_layout(&self) -> Vec<String> {
        let mut names = vec!["image".to_owned()];
        for i in 0..self.counted_enc_stages() {
            names.push(format!("enc.{i}.dw.in"));
            names.push(format!("enc.{i}.pw.in"));
        }
        for (r, _) in self.res.iter().enumerate() {
            names.push(format!("res.{r}.s1.dw.in"));
            names.push(format!("res.{r}.s1.pw.in"));
            names.push(format!("res.{r}.s2.dw.in"));
            names.push(format!("res.{r}.s2.pw.in"));
        }
        names.push("up3.0.dw.in".to_owned());
        names.push("up3.0.pw.in".to_owned());
        names.push("up1.0.in".to_owned());
        names.push("up3.1.dw.in".to_owned());
        names.push("up3.1.pw.in".to_owned());
        names.push("up1.1.in".to_owned());
        names.push("head.dw.in".to_owned());
        names.push("head.pw.in".to_owned());
        names
    }

    pub fn forward_core<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        scheme: Option<&quant::QuantScheme>,
        mut observe: F,
    ) -> Tensor<B, 4> {
        if self.encoder == EncoderKind::Lut || self.encoder == EncoderKind::LutDw {
            let (half_skip, h) = self.encode_with_observe(content, &mut observe);
            return self.finish_from_trunk(h, half_skip, 1 + 2 * self.enc.len(), scheme, observe);
        }
        let mut point = 0usize;
        let mut h = qa(content, scheme, point, &mut observe);
        point += 1;
        let stem_w = match scheme {
            Some(scheme) => quant::fake_quant_weight(self.stem.weight(), scheme.weight_max()),
            None => self.stem.weight(),
        };
        h = relu(self.stem.forward_with(h, stem_w));

        // enc[0] is the H/2 map. It is kept for the decoder skip; enc[1] is what
        // actually enters the H/4 bottleneck.
        h = self.enc[0].forward_quant(
            h,
            point,
            point + 1,
            scheme,
            &mut observe,
            wq_dw(&self.enc[0].dw, scheme),
            wq_pw(&self.enc[0].pw, scheme),
            true,
        );
        point += 2;
        let half_skip = h.clone();
        h = self.enc[1].forward_quant(
            h,
            point,
            point + 1,
            scheme,
            &mut observe,
            wq_dw(&self.enc[1].dw, scheme),
            wq_pw(&self.enc[1].pw, scheme),
            true,
        );
        point += 2;
        if self.trunk_h8 {
            // Float only. The integer layout still stops at the two exported stages.
            h = self.enc[2].forward(h);
        }

        self.finish_from_trunk(h, half_skip, point, scheme, observe)
    }

    /// Residual trunk and decoder, starting at the H/4 map.
    ///
    /// `point` is the first residual activation index. A LUT encoder skips the
    /// separable stem and lands on the same index, so the decoder's quantised
    /// points stay where the separable layout put them.
    fn finish_from_trunk<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        mut h: Tensor<B, 4>,
        half_skip: Tensor<B, 4>,
        point: usize,
        scheme: Option<&quant::QuantScheme>,
        mut observe: F,
    ) -> Tensor<B, 4> {
        let trunk_base = point;
        for (r, block) in self.res.iter().enumerate() {
            let p = trunk_base + 4 * r;
            let input = qa(h, scheme, p, &mut observe);
            let activated = relu(
                block
                    .s1
                    .dw
                    .forward_with(input.clone(), wq_dw(&block.s1.dw, scheme)),
            );
            let activated = qa(activated, scheme, p + 1, &mut observe);
            let mid = relu(
                block
                    .s1
                    .pw
                    .forward_with(activated, wq_pw(&block.s1.pw, scheme)),
            );
            let activated = qa(mid, scheme, p + 2, &mut observe);
            let activated = relu(
                block
                    .s2
                    .dw
                    .forward_with(activated, wq_dw(&block.s2.dw, scheme)),
            );
            let activated = qa(activated, scheme, p + 3, &mut observe);
            let inner = block
                .s2
                .pw
                .forward_with(activated, wq_pw(&block.s2.pw, scheme));
            let out_point = p + 4;
            h = match scheme {
                Some(scheme) => {
                    let out = scheme.acts[out_point];
                    let qmax = scheme.act_max();
                    quant::fake_quant_act(input, out, qmax)
                        + quant::fake_quant_act(inner, out, qmax)
                }
                None => input + inner,
            };
        }

        let dec_base = trunk_base + 4 * self.res.len();
        h = self.at(PluginSite::H4, h);
        let up = self.upsample_trunk_to_skip(h);
        // Same grid as the next convolution's input, matching the residual add:
        // both operands are requantised onto up3.0.dw.in, then that point observes the sum.
        h = match scheme {
            Some(scheme) => {
                let scale = scheme.acts[dec_base];
                let qmax = scheme.act_max();
                quant::fake_quant_act(up, scale, qmax)
                    + quant::fake_quant_act(half_skip, scale, qmax)
            }
            None => up + half_skip,
        };

        h = qa(h, scheme, dec_base, &mut observe);
        h = self.up3[0].forward_quant(
            h,
            dec_base,
            dec_base + 1,
            scheme,
            &mut observe,
            wq_dw(&self.up3[0].dw, scheme),
            wq_pw(&self.up3[0].pw, scheme),
            true,
        );
        h = self.half_color_quant(h, scheme, dec_base + 2, &mut observe);
        h = self.at(PluginSite::H2, h);

        h = upsample2x(h);

        h = qa(h, scheme, dec_base + 3, &mut observe);
        h = self.up3[1].forward_quant(
            h,
            dec_base + 3,
            dec_base + 4,
            scheme,
            &mut observe,
            wq_dw(&self.up3[1].dw, scheme),
            wq_pw(&self.up3[1].pw, scheme),
            true,
        );
        h = self.full_color_quant(h, scheme, dec_base + 5, &mut observe);
        h = self.at(PluginSite::Full, h);

        h = qa(h, scheme, dec_base + 6, &mut observe);
        let image = self.head.forward_quant(
            h,
            dec_base + 6,
            dec_base + 7,
            scheme,
            &mut observe,
            wq_dw(&self.head.dw, scheme),
            wq_pw(&self.head.pw, scheme),
            false,
        );
        match scheme {
            Some(_) => quant::fake_quant_output(image),
            None => image,
        }
    }

    /// `(H/2 skip, H/4 trunk)`. The LUT path centers the image and snaps both exits.
    fn encode_fp32(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        self.encode_with_observe(content, &mut |_, _| {})
    }

    fn encode_with_observe<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        observe: &mut F,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if let Some(encoder) = &self.lut_encoder {
            return encoder.forward_dw(content);
        }
        if self.lut_dw.is_some() {
            return self.forward_lut_dw(content, observe);
        }
        let mut h = relu(self.stem.forward(content));
        observe(1, &h);
        let after0_dw = relu(self.enc[0].dw.forward(h));
        observe(2, &after0_dw);
        h = relu(self.enc[0].pw.forward(after0_dw));
        let half_skip = h.clone();
        observe(3, &h);
        let after1_dw = relu(self.enc[1].dw.forward(h));
        observe(4, &after1_dw);
        h = relu(self.enc[1].pw.forward(after1_dw));
        if self.trunk_h8 {
            let after2_dw = relu(self.enc[2].dw.forward(h));
            h = relu(self.enc[2].pw.forward(after2_dw));
        }
        (half_skip, h)
    }

    /// Dense stem, then LUT depthwise + the existing pointwise at each stride-2 stage.
    ///
    /// Activation order matches [SepConv]: ReLU after the spatial kernel, pointwise,
    /// ReLU again. The skip is that H/2 tensor. It is not snapped back onto the LUT
    /// codebook, because the pointwise mixer is what the decoder already adds.
    fn forward_lut_dw<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        observe: &mut F,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let h = relu(self.stem.forward(content));
        observe(1, &h);
        let spatial = self
            .lut_dw
            .as_ref()
            .expect("encoder lut_dw")
            .forward_half(h);
        let after_dw = relu(spatial);
        observe(2, &after_dw);
        let h = relu(self.enc[0].pw.forward(after_dw));
        let skip = h.clone();
        observe(3, &h);
        let spatial = self
            .lut_dw
            .as_ref()
            .expect("encoder lut_dw")
            .forward_trunk(h);
        let after_dw = relu(spatial);
        observe(4, &after_dw);
        let h = relu(self.enc[1].pw.forward(after_dw));
        (skip, h)
    }

    fn intermediates(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let (half_skip, mut h) = self.encode_fp32(content);
        for block in &self.res {
            h = block.forward(h);
        }
        h = self.at(PluginSite::H4, h);
        let trunk = h.clone();

        h = self.upsample_trunk_to_skip(h) + half_skip;
        h = self.up3[0].forward(h);
        h = self.half_color(h);
        h = self.at(PluginSite::H2, h);
        let half = h.clone();

        h = upsample2x(h);
        h = self.up3[1].forward(h);
        h = self.full_color(h);
        h = self.at(PluginSite::Full, h);
        let full = h.clone();

        (self.head.forward_inner(h, false), vec![trunk, half, full])
    }

    /// H/4 maps around the residual trunk.
    ///
    /// `encoder` is the second stride-2 block, the trunk input. `trunk` is the
    /// four residual blocks with no plugin. `decoder_in` is what the first
    /// upsample reads, so an H/4 plugin such as InstanceNorm is included.
    pub fn trunk_boundary(&self, content: Tensor<B, 4>) -> TrunkBoundary<B> {
        let (_, mut h) = self.encode_fp32(content);
        let encoder = h.clone();
        for block in &self.res {
            h = block.forward(h);
        }
        let trunk = h.clone();
        let decoder_in = self.at(PluginSite::H4, h);
        TrunkBoundary {
            encoder,
            trunk,
            decoder_in,
        }
    }

    /// Decoder activations on one image, in forward order.
    ///
    /// Channel counts change, so redundancy is read by comparing the pairs that
    /// share a shape: the H/2 skip against the upsampled trunk, each depthwise
    /// against its input, and each extra pointwise against the separable block
    /// it follows.
    pub fn decoder_trace(&self, content: Tensor<B, 4>) -> DecoderTrace<B> {
        let (skip, mut h) = self.encode_fp32(content);
        for block in &self.res {
            h = block.forward(h);
        }
        h = self.at(PluginSite::H4, h);
        let up_trunk = self.upsample_trunk_to_skip(h);
        let fused = up_trunk.clone() + skip.clone();

        let half_dw = relu(self.up3[0].dw.forward(fused.clone()));
        let half_sep = self.up3[0].forward(fused.clone());
        let half_pw = self.at(PluginSite::H2, self.half_color(half_sep.clone()));

        let full_up = upsample2x(half_pw.clone());
        let full_dw = relu(self.up3[1].dw.forward(full_up.clone()));
        let full_sep = self.up3[1].forward(full_up.clone());
        let full_pw = self.at(PluginSite::Full, self.full_color(full_sep.clone()));
        let rgb = self.head.forward_inner(full_pw.clone(), false);

        DecoderTrace {
            up_trunk,
            skip,
            fused,
            half_dw,
            half_sep,
            half_pw,
            full_up,
            full_dw,
            full_sep,
            full_pw,
            rgb,
        }
    }
}

/// The three H/4 tensors that bracket the residual trunk and the H/4 plugin.
pub struct TrunkBoundary<B: Backend> {
    pub encoder: Tensor<B, 4>,
    pub trunk: Tensor<B, 4>,
    pub decoder_in: Tensor<B, 4>,
}

/// Decoder maps from the H/2 fusion through the RGB head.
pub struct DecoderTrace<B: Backend> {
    /// Nearest upsample of the H/4 trunk, including the H/4 plugin.
    pub up_trunk: Tensor<B, 4>,
    /// Encoder map at H/2, added to `up_trunk`.
    pub skip: Tensor<B, 4>,
    /// `up_trunk + skip`, the first decoder convolution's input.
    pub fused: Tensor<B, 4>,
    /// ReLU after the first decoder depthwise 3×3.
    pub half_dw: Tensor<B, 4>,
    /// First separable block, 64 → `mid`.
    pub half_sep: Tensor<B, 4>,
    /// Extra H/2 pointwise after that block.
    pub half_pw: Tensor<B, 4>,
    /// Nearest upsample of `half_pw` to full resolution.
    pub full_up: Tensor<B, 4>,
    /// ReLU after the full-resolution depthwise 3×3.
    pub full_dw: Tensor<B, 4>,
    /// Second separable block, `mid` → `fine`.
    pub full_sep: Tensor<B, 4>,
    /// Extra full-resolution pointwise, the head's input.
    pub full_pw: Tensor<B, 4>,
    pub rgb: Tensor<B, 4>,
}

impl<B: Backend> StudentModel<B> for TinyDwStudent<B> {
    fn describe(&self) -> String {
        let encoder = match self.encoder {
            EncoderKind::Lut => format!(
                "lut:b{}",
                self.lut_encoder.as_ref().map(|lut| lut.bits()).unwrap_or(0)
            ),
            EncoderKind::LutDw => format!(
                "lut_dw:b{}",
                self.lut_dw.as_ref().map(|lut| lut.bits()).unwrap_or(0)
            ),
            EncoderKind::Sep => self.encoder.name().to_owned(),
        };
        let trunk = if self.trunk_h8 { " trunk=H/8" } else { "" };
        format!(
            "tiny_dw encoder={encoder} plugin={} site={} color={}{trunk} ({} parameters)",
            self.plugin.kind().name(),
            self.plugin_site.name(),
            self.color.name(),
            self.num_params()
        )
    }

    fn stylize(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        self.intermediates(content).0
    }

    fn intermediate(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        self.intermediates(content)
    }
}

impl<B: Backend> quant::Quantizable<B> for TinyDwStudent<B> {
    fn forward_quantised(
        &self,
        content: Tensor<B, 4>,
        scheme: &quant::QuantScheme,
    ) -> Tensor<B, 4> {
        self.forward_core(content, Some(scheme), |_, _| {})
    }

    fn quant_points(&self) -> Vec<String> {
        self.quant_layout()
    }

    fn observe_activations(&self, content: Tensor<B, 4>, observe: &mut dyn FnMut(usize, f32)) {
        self.forward_core(content, None, |point, x| {
            observe(point, x.clone().abs().max().into_scalar().to_f64() as f32);
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tiny_dw_has_fewer_params_than_tiny() {
        fn conv3(in_ch: usize, out_ch: usize) -> usize {
            out_ch * in_ch * 9 + out_ch
        }
        fn sep3(in_ch: usize, out_ch: usize) -> usize {
            in_ch * 9 + in_ch + out_ch * in_ch + out_ch
        }
        fn pw1(in_ch: usize, out_ch: usize) -> usize {
            out_ch * in_ch + out_ch
        }

        let base = 32usize;
        let res_blocks = 4usize;
        let mid = 32usize;
        let fine = 16usize;
        let c2 = base * 2;

        let tiny = conv3(3, base)
            + conv3(base, c2)
            + conv3(c2, c2)
            + res_blocks * (conv3(c2, c2) + conv3(c2, c2))
            + conv3(c2, mid)
            + pw1(mid, mid)
            + conv3(mid, fine)
            + pw1(fine, fine)
            + conv3(fine, 3);

        let tiny_dw = conv3(3, base)
            + sep3(base, c2)
            + sep3(c2, c2)
            + res_blocks * (sep3(c2, c2) + sep3(c2, c2))
            + sep3(c2, mid)
            + pw1(mid, mid)
            + sep3(mid, fine)
            + pw1(fine, fine)
            + sep3(fine, 3);

        assert_eq!(tiny, 376_595);
        assert_eq!(tiny_dw, 51_635);
        assert!(tiny_dw < tiny);
    }

    #[test]
    fn parse_color_names() {
        assert_eq!(super::parse_color("none").unwrap(), super::ColorPath::None);
        assert_eq!(
            super::parse_color("film_both").unwrap(),
            super::ColorPath::FilmBoth
        );
        assert!(super::parse_color("grid").is_err());
        assert_eq!(
            super::parse_encoder("sep").unwrap(),
            super::EncoderKind::Sep
        );
        assert_eq!(
            super::parse_encoder("lut").unwrap(),
            super::EncoderKind::Lut
        );
        assert_eq!(
            super::parse_encoder("lut_dw").unwrap(),
            super::EncoderKind::LutDw
        );
        assert!(super::parse_encoder("dense").is_err());
        assert_eq!(super::parse_lut_bits(8).unwrap(), 8);
        assert!(super::parse_lut_bits(1).is_err());
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod forward_sanity_gpu {
    use super::*;
    use crate::backend::{InferBackend, select_device};

    #[test]
    fn tiny_dw_forward_is_not_constant_on_gpu() {
        let device = select_device("d1");
        let model = TinyDwStudent::<InferBackend>::new(32, 4, 32, 16, &device);
        let n = 1 * 3 * 64 * 64;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let x = Tensor::<InferBackend, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 64, 64]),
            &device,
        );
        let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        eprintln!(
            "gpu d1 out mean={mean:.6} min={mn:.6} max={mx:.6} span={:.6}",
            mx - mn
        );
        assert!(mx - mn > 0.01, "GPU output is constant: min={mn} max={mx}");
    }

    #[test]
    fn tiny_dw_forward_large_resolution_on_gpu() {
        let device = select_device("d1");
        let model = TinyDwStudent::<InferBackend>::new(32, 4, 32, 16, &device);
        for (h, w) in [(192, 192), (288, 384), (512, 512)] {
            let n = 1 * 3 * h * w;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
                .collect();
            let x = Tensor::<InferBackend, 4>::from_data(
                burn::tensor::TensorData::new(data, [1, 3, h, w]),
                &device,
            );
            let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
            let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = out.iter().sum::<f32>() / out.len() as f32;
            eprintln!(
                "gpu {h}x{w} mean={mean:.6} min={mn:.6} max={mx:.6} span={:.6}",
                mx - mn
            );
            assert!(mx - mn > 0.01, "{h}x{w} output is constant");
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod forward_sanity_cpu {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn every_plugin_keeps_the_image_shape_on_cpu() {
        use super::super::plugin::{PluginKind, PluginSite};
        let device = Default::default();
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();
        let kinds = [
            PluginKind::None,
            PluginKind::Affine,
            PluginKind::In,
            PluginKind::Ln,
            PluginKind::Se,
            PluginKind::Film,
            PluginKind::Eca,
            PluginKind::Grn,
            PluginKind::Ca,
            PluginKind::Frelu,
            PluginKind::Dw7,
            PluginKind::SimAm,
            PluginKind::Shuffle,
            PluginKind::NarrowIn,
        ];
        for kind in kinds {
            for site in [PluginSite::H4, PluginSite::H2, PluginSite::Full] {
                let model = TinyDwStudent::<NdArray>::with_plugin(8, 1, 6, 4, kind, site, &device);
                let x = Tensor::<NdArray, 4>::from_data(
                    burn::tensor::TensorData::new(data.clone(), [1, 3, 16, 16]),
                    &device,
                );
                let out = model.stylize(x);
                assert_eq!(out.dims(), [1, 3, 16, 16], "{kind:?} {site:?}");
                let values = out.into_data().to_vec::<f32>().unwrap();
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{kind:?} {site:?} non-finite"
                );
            }
        }
        for kind in [PluginKind::In, PluginKind::NarrowIn] {
            let model =
                TinyDwStudent::<NdArray>::with_plugin(8, 0, 6, 4, kind, PluginSite::H4, &device);
            let x = Tensor::<NdArray, 4>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 3, 16, 16]),
                &device,
            );
            let out = model.stylize(x);
            assert_eq!(out.dims(), [1, 3, 16, 16], "res_blocks=0 {kind:?}");
            let values = out.into_data().to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|v| v.is_finite()),
                "res_blocks=0 {kind:?} non-finite"
            );
        }
    }

    #[test]
    fn tiny_dw_forward_is_not_constant_on_cpu() {
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::new(32, 4, 32, 16, &device);
        let n = 1 * 3 * 64 * 64;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 64, 64]),
            &device,
        );
        let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("cpu out min={mn} max={mx}");
        assert!(mx - mn > 0.01, "output is constant: min={mn} max={mx}");
    }

    #[test]
    fn film_color_keeps_the_image_shape_and_replaces_only_that_pointwise() {
        use super::super::plugin::{PluginKind, PluginSite};
        let device = Default::default();
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();
        let plain = TinyDwStudent::<NdArray>::with_color(
            32,
            1,
            48,
            24,
            PluginKind::None,
            PluginSite::H4,
            ColorPath::None,
            &device,
        );
        let half = TinyDwStudent::<NdArray>::with_color(
            32,
            1,
            48,
            24,
            PluginKind::None,
            PluginSite::H4,
            ColorPath::FilmHalf,
            &device,
        );
        let full = TinyDwStudent::<NdArray>::with_color(
            32,
            1,
            48,
            24,
            PluginKind::None,
            PluginSite::H4,
            ColorPath::FilmFull,
            &device,
        );
        let both = TinyDwStudent::<NdArray>::with_color(
            32,
            1,
            48,
            24,
            PluginKind::In,
            PluginSite::H4,
            ColorPath::FilmBoth,
            &device,
        );
        fn pw(channels: usize) -> usize {
            channels * channels + channels
        }
        fn film(channels: usize) -> usize {
            let hidden = (channels / 4).max(1);
            3 * channels * hidden + hidden + 2 * channels
        }
        assert_eq!(plain.num_params() - half.num_params(), pw(48) - film(48));
        assert_eq!(plain.num_params() - full.num_params(), pw(24) - film(24));
        assert!(plain.blocks_integer_export().is_none());
        assert!(half.blocks_integer_export().is_none());
        assert!(full.blocks_integer_export().is_none());
        assert!(both.blocks_integer_export().is_none());
        for model in [&half, &full, &both] {
            let x = Tensor::<NdArray, 4>::from_data(
                burn::tensor::TensorData::new(data.clone(), [1, 3, 16, 16]),
                &device,
            );
            let out = model.stylize(x);
            assert_eq!(out.dims(), [1, 3, 16, 16]);
            let values = out.into_data().to_vec::<f32>().unwrap();
            assert!(values.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn plain_decoder_checkpoint_still_loads() {
        use super::super::plugin::{PluginKind, PluginSite};
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let path =
            std::path::Path::new("experiments/tiny_dw_vangogh_trunk/trials/in_rb1/model.mpk");
        if !path.is_file() {
            return;
        }
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::with_plugin(
            32,
            1,
            48,
            24,
            PluginKind::In,
            PluginSite::H4,
            &device,
        );
        let loaded = model.load_file(
            path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        );
        assert!(loaded.is_ok(), "in_rb1 checkpoint: {:?}", loaded.err());
    }

    #[test]
    fn lut_encoder_docks_with_the_film_decoder() {
        use super::super::lut::LUT_CODE_SPAN;
        use super::super::plugin::{PluginKind, PluginSite};
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::with_encoder(
            8,
            1,
            6,
            4,
            PluginKind::In,
            PluginSite::H4,
            ColorPath::FilmBoth,
            EncoderKind::Lut,
            1,
            8,
            false,
            &device,
        );
        assert!(model.blocks_integer_export().is_some());
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 16, 16]),
            &device,
        );
        let trace = model.decoder_trace(x.clone());
        assert_eq!(trace.rgb.dims(), [1, 3, 16, 16]);
        let boundary = model.trunk_boundary(x.clone());
        assert_eq!(boundary.encoder.dims(), [1, 16, 4, 4]);
        for (name, tensor) in [("skip", trace.skip), ("trunk", boundary.encoder)] {
            let values = tensor.into_data().to_vec::<f32>().unwrap();
            assert!(values.iter().all(|v| v.is_finite()), "{name} is non-finite");
            let peak = values.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
            assert!(
                peak <= LUT_CODE_SPAN + 1e-3,
                "{name} peak {peak} leaves the decoder span"
            );
            let mean = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;
            assert!(mean > 1e-3, "{name} was crushed to {mean}");
        }
        let image = model.stylize(x).into_data().to_vec::<f32>().unwrap();
        assert!(image.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lut_dw_encoder_keeps_the_pointwise_and_docks() {
        use super::super::plugin::{PluginKind, PluginSite};
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::with_encoder(
            8,
            1,
            6,
            4,
            PluginKind::In,
            PluginSite::H4,
            ColorPath::FilmBoth,
            EncoderKind::LutDw,
            2,
            4,
            false,
            &device,
        );
        let tables = model.lut_dw.as_ref().unwrap().table_entries();
        // half: 8 ch * 9 taps * 15 codes. trunk: 16 ch * 9 * 15.
        assert_eq!(tables, 8 * 9 * 15 + 16 * 9 * 15);
        assert!(model.blocks_integer_export().is_none());
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 16, 16]),
            &device,
        );
        let trace = model.decoder_trace(x.clone());
        assert_eq!(trace.skip.dims(), [1, 16, 8, 8]);
        assert_eq!(trace.rgb.dims(), [1, 3, 16, 16]);
        let boundary = model.trunk_boundary(x);
        assert_eq!(boundary.encoder.dims(), [1, 16, 4, 4]);
        for (name, tensor) in [("skip", trace.skip), ("trunk", boundary.encoder)] {
            let values = tensor.into_data().to_vec::<f32>().unwrap();
            assert!(values.iter().all(|v| v.is_finite()), "{name} is non-finite");
        }
    }

    #[test]
    fn h8_trunk_lifts_back_onto_the_half_skip() {
        use super::super::plugin::{PluginKind, PluginSite};
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::with_encoder(
            8,
            1,
            6,
            4,
            PluginKind::In,
            PluginSite::H4,
            ColorPath::FilmBoth,
            EncoderKind::Sep,
            4,
            8,
            true,
            &device,
        );
        let h4 = TinyDwStudent::<NdArray>::with_encoder(
            8,
            1,
            6,
            4,
            PluginKind::In,
            PluginSite::H4,
            ColorPath::FilmBoth,
            EncoderKind::Sep,
            4,
            8,
            false,
            &device,
        );
        let msg = model.blocks_integer_export().expect("h8 export blocked");
        assert!(msg.contains("trunk_at=h8"), "{msg}");
        assert!(model.describe().contains("trunk=H/8"));
        assert_eq!(model.quant_point_count(), h4.quant_point_count());
        let n = 1 * 3 * 16 * 16;
        let data: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 16, 16]),
            &device,
        );
        let y = model.stylize(x.clone());
        assert_eq!(y.dims(), [1, 3, 16, 16]);
        let boundary = model.trunk_boundary(x.clone());
        assert_eq!(boundary.encoder.dims(), [1, 16, 2, 2]);
        let trace = model.decoder_trace(x);
        assert_eq!(trace.skip.dims(), [1, 16, 8, 8]);
        assert_eq!(trace.up_trunk.dims(), [1, 16, 8, 8]);
        let values = y.into_data().to_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
