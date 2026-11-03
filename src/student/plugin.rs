//! One optional block inserted into `tiny_dw`.
//!
//! Each plugin is a single mechanism at a single resolution. Parametric plugins that can
//! be the identity are initialised as the identity, so a trial that does not learn the
//! new weights matches the baseline at step 0. Instance norm, layer norm and SimAM change
//! the forward pass immediately: that change is the module.
//!
//! `in`, `ln`, and decoder FiLM are lowered to the integer manifest. Other plugins are not.

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::module::conv2d;
use burn::tensor::ops::PaddedConvOptions;

use super::tiny_dw::{DwConv, PwConv};

const EPS: f32 = 1e-5;

/// Which block to insert. `None` is the capacity-study decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginKind {
    None,
    /// Per-channel γ, β. Fixed-style FiLM with no conditioner.
    Affine,
    /// InstanceNorm over space, then γ, β. What AdaIN becomes when style is constant.
    In,
    /// LayerNorm over channels at each pixel, then γ, β. Streaming stand-in for IN.
    Ln,
    /// Squeeze-and-excitation, reduction 4. Scale starts at 1 (`2 * sigmoid`).
    Se,
    /// GAP → bottleneck → γ, β. Dynamic FiLM. Zero-init heads, so it starts as identity.
    Film,
    /// Efficient channel attention: a length-k circular conv on the GAP vector.
    Eca,
    /// ConvNeXt V2 global response normalisation. γ starts at 0, so it starts as identity.
    Grn,
    /// Coordinate attention. Pools H and W separately, so the gate can depend on position.
    Ca,
    /// FReLU: `max(x, DW3x3(x))`. The depthwise kernel starts as identity.
    Frelu,
    /// Residual 7×7 depthwise. Zero-init, so it starts as identity.
    Dw7,
    /// SimAM, parameter-free spatial-channel energy gate.
    SimAm,
    /// Grouped 1×1 expand + pixel-shuffle in place of the first nearest upsample.
    Shuffle,
    /// 1×1 down to C/4, InstanceNorm, 1×1 back. Replaces a residual trunk.
    NarrowIn,
}

impl PluginKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Affine => "affine",
            Self::In => "in",
            Self::Ln => "ln",
            Self::Se => "se",
            Self::Film => "film",
            Self::Eca => "eca",
            Self::Grn => "grn",
            Self::Ca => "ca",
            Self::Frelu => "frelu",
            Self::Dw7 => "dw7",
            Self::SimAm => "simam",
            Self::Shuffle => "shuffle",
            Self::NarrowIn => "narrow_in",
        }
    }

    pub fn blocks_integer_export(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Where a shape-preserving plugin is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginSite {
    /// 64-ch trunk, after the residual blocks, before the first upsample.
    H4,
    /// Decoder width `mid`, after the H/2 pointwise, before the second upsample.
    H2,
    /// Decoder width `fine`, after the full-resolution pointwise, before the head.
    Full,
}

impl PluginSite {
    pub fn name(self) -> &'static str {
        match self {
            Self::H4 => "h4",
            Self::H2 => "h2",
            Self::Full => "full",
        }
    }
}

pub fn parse_kind(name: &str) -> Result<PluginKind, String> {
    match name {
        "none" => Ok(PluginKind::None),
        "affine" => Ok(PluginKind::Affine),
        "in" => Ok(PluginKind::In),
        "ln" => Ok(PluginKind::Ln),
        "se" => Ok(PluginKind::Se),
        "film" => Ok(PluginKind::Film),
        "eca" => Ok(PluginKind::Eca),
        "grn" => Ok(PluginKind::Grn),
        "ca" => Ok(PluginKind::Ca),
        "frelu" => Ok(PluginKind::Frelu),
        "dw7" => Ok(PluginKind::Dw7),
        "simam" => Ok(PluginKind::SimAm),
        "shuffle" => Ok(PluginKind::Shuffle),
        "narrow_in" => Ok(PluginKind::NarrowIn),
        other => Err(format!(
            "unknown tiny_dw plugin {other:?}; expected none, affine, in, ln, se, film, eca, grn, ca, frelu, dw7, simam, shuffle, narrow_in"
        )),
    }
}

pub fn parse_site(name: &str) -> Result<PluginSite, String> {
    match name {
        "h4" => Ok(PluginSite::H4),
        "h2" => Ok(PluginSite::H2),
        "full" => Ok(PluginSite::Full),
        other => Err(format!(
            "unknown tiny_dw plugin_site {other:?}; expected h4, h2, full"
        )),
    }
}

fn bottleneck(channels: usize) -> usize {
    (channels / 4).max(1)
}

/// ECA kernel: odd integer nearest `|log2(C) + 1| / 2`, truncated the way the paper does.
fn eca_kernel(channels: usize) -> usize {
    let t = ((channels as f32).log2() + 1.0) / 2.0;
    let t = t.floor().max(1.0) as usize;
    if t % 2 == 0 { t + 1 } else { t }
}

fn broadcast_channel<B: Backend>(p: &Param<Tensor<B, 1>>) -> Tensor<B, 4> {
    let c = p.dims()[0];
    p.val().reshape([1, c, 1, 1])
}

fn gap<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    x.mean_dims(&[2, 3])
}

fn zero_conv<B: Backend>(conv: &mut Conv2d<B>, device: &B::Device) {
    let [out, inn, kh, kw] = conv.weight.dims();
    conv.weight = Param::from_tensor(Tensor::<B, 4>::zeros([out, inn, kh, kw], device));
    if let Some(bias) = conv.bias.as_mut() {
        let n = bias.dims()[0];
        *bias = Param::from_tensor(Tensor::<B, 1>::zeros([n], device));
    }
}

fn zero_pw<B: Backend>(pw: &mut PwConv<B>, device: &B::Device) {
    zero_conv(&mut pw.conv, device);
}

fn identity_dw3<B: Backend>(dw: &mut DwConv<B>, device: &B::Device) {
    let [c, inn, kh, kw] = dw.conv.weight.dims();
    debug_assert_eq!((inn, kh, kw), (1, 3, 3));
    let mut data = vec![0.0f32; c * 9];
    for ch in 0..c {
        data[ch * 9 + 4] = 1.0;
    }
    dw.conv.weight = Param::from_tensor(Tensor::<B, 4>::from_data(
        TensorData::new(data, [c, 1, 3, 3]),
        device,
    ));
    if let Some(bias) = dw.conv.bias.as_mut() {
        *bias = Param::from_tensor(Tensor::<B, 1>::zeros([c], device));
    }
}

fn conv2d_opts<B: Backend>(conv: &Conv2d<B>) -> PaddedConvOptions<2> {
    PaddedConvOptions::asymmetric(conv.stride, [0, 0], [0, 0], conv.dilation, conv.groups)
}

#[derive(Module, Debug)]
pub struct ChannelAffine<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> ChannelAffine<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x * broadcast_channel(&self.gamma) + broadcast_channel(&self.beta)
    }

    pub fn host_gamma_beta(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        let gamma = self
            .gamma
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let beta = self
            .beta
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| e.to_string())?;
        Ok((gamma, beta))
    }
}

#[derive(Module, Debug)]
pub struct NormAffine<B: Backend> {
    affine: ChannelAffine<B>,
    /// `true` reduces over H and W (instance norm). `false` reduces over channels (layer norm).
    #[module(skip)]
    spatial: bool,
    /// Added inside the square root. The default [`EPS`] keeps every existing graph.
    /// Guided's trunk passes its own epsilon, and the integer export uses that same value.
    #[module(skip)]
    eps: f32,
}

impl<B: Backend> NormAffine<B> {
    fn new(channels: usize, spatial: bool, device: &B::Device) -> Self {
        Self {
            affine: ChannelAffine::new(channels, device),
            spatial,
            eps: EPS,
        }
    }

    fn set_eps(&mut self, eps: f32) {
        self.eps = eps;
    }

    pub fn host_gamma_beta(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.affine.host_gamma_beta()
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let centered = if self.spatial {
            x.clone() - x.clone().mean_dims(&[2, 3])
        } else {
            x.clone() - x.clone().mean_dim(1)
        };
        let var = centered.clone().powf_scalar(2.0);
        let var = if self.spatial {
            var.mean_dims(&[2, 3])
        } else {
            var.mean_dim(1)
        };
        let hat = centered / (var + self.eps).sqrt();
        self.affine.forward(hat)
    }
}

#[derive(Module, Debug)]
pub struct SqueezeExcite<B: Backend> {
    reduce: PwConv<B>,
    expand: PwConv<B>,
}

impl<B: Backend> SqueezeExcite<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let hidden = bottleneck(channels);
        let mut expand = PwConv::new(hidden, channels, device);
        zero_pw(&mut expand, device);
        Self {
            reduce: PwConv::new(channels, hidden, device),
            expand,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = relu(self.reduce.forward(gap(x.clone())));
        // 2 * sigmoid(0) = 1, and `expand` is zero-initialised.
        let scale = sigmoid(self.expand.forward(z)) * 2.0;
        x * scale
    }
}

#[derive(Module, Debug)]
pub struct Film<B: Backend> {
    reduce: PwConv<B>,
    gamma: PwConv<B>,
    beta: PwConv<B>,
}

impl<B: Backend> Film<B> {
    pub(crate) fn new(channels: usize, device: &B::Device) -> Self {
        let hidden = bottleneck(channels);
        let mut gamma = PwConv::new(hidden, channels, device);
        let mut beta = PwConv::new(hidden, channels, device);
        zero_pw(&mut gamma, device);
        zero_pw(&mut beta, device);
        Self {
            reduce: PwConv::new(channels, hidden, device),
            gamma,
            beta,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = relu(self.reduce.forward(gap(x.clone())));
        let gamma = self.gamma.forward(z.clone()) + 1.0;
        let beta = self.beta.forward(z);
        x * gamma + beta
    }

    pub fn host_heads(&self) -> (&PwConv<B>, &PwConv<B>, &PwConv<B>) {
        (&self.reduce, &self.gamma, &self.beta)
    }
}

#[derive(Module, Debug)]
pub struct Eca<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Eca<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let k = eca_kernel(channels);
        Self {
            weight: Param::from_tensor(Tensor::zeros([k], device)),
            bias: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, _, _] = x.dims();
        let z = gap(x.clone()).reshape([n, c]);
        let y = circular_conv1d(z, self.weight.val(), self.bias.val());
        let scale = sigmoid(y.reshape([n, c, 1, 1])) * 2.0;
        x * scale
    }
}

fn circular_conv1d<B: Backend>(
    z: Tensor<B, 2>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let [n, c] = z.dims();
    let k = weight.dims()[0];
    let pad = k / 2;
    let left = z.clone().slice(s![0..n, c - pad..c]);
    let right = z.clone().slice(s![0..n, 0..pad]);
    let padded = Tensor::cat(vec![left, z, right], 1);
    let mut acc = Tensor::zeros([n, c], &padded.device());
    for t in 0..k {
        let window = padded.clone().slice(s![0..n, t..t + c]);
        let wt = weight.clone().slice(s![t..t + 1]).reshape([1, 1]);
        acc = acc + window * wt;
    }
    acc + bias.reshape([1, 1])
}

#[derive(Module, Debug)]
pub struct Grn<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> Grn<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::zeros([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let gx = (x.clone().powf_scalar(2.0).sum_dims(&[2, 3]) + EPS).sqrt();
        let nx = gx.clone() / (gx.mean_dim(1) + EPS);
        x.clone() * nx * broadcast_channel(&self.gamma) + broadcast_channel(&self.beta) + x
    }
}

#[derive(Module, Debug)]
pub struct CoordAttn<B: Backend> {
    reduce: PwConv<B>,
    height: PwConv<B>,
    width: PwConv<B>,
}

impl<B: Backend> CoordAttn<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let hidden = bottleneck(channels);
        let mut height = PwConv::new(hidden, channels, device);
        let mut width = PwConv::new(hidden, channels, device);
        zero_pw(&mut height, device);
        zero_pw(&mut width, device);
        Self {
            reduce: PwConv::new(channels, hidden, device),
            height,
            width,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = x.dims();
        let x_h = x.clone().mean_dim(3);
        let x_w = x.clone().mean_dim(2).swap_dims(2, 3);
        let y = relu(self.reduce.forward(Tensor::cat(vec![x_h, x_w], 2)));
        let y_h = y.clone().slice(s![.., .., 0..h, ..]);
        let y_w = y.slice(s![.., .., h..h + w, ..]).swap_dims(2, 3);
        let a_h = sigmoid(self.height.forward(y_h)) * 2.0;
        let a_w = sigmoid(self.width.forward(y_w)) * 2.0;
        x * a_h * a_w
    }
}

#[derive(Module, Debug)]
pub struct FreLu<B: Backend> {
    dw: DwConv<B>,
}

impl<B: Backend> FreLu<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let mut dw = DwConv::new(channels, 3, 1, device);
        identity_dw3(&mut dw, device);
        Self { dw }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.clone().max_pair(self.dw.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct DwResidual<B: Backend> {
    dw: DwConv<B>,
}

impl<B: Backend> DwResidual<B> {
    fn new(channels: usize, kernel: usize, device: &B::Device) -> Self {
        let mut dw = DwConv::new(channels, kernel, 1, device);
        zero_conv(&mut dw.conv, device);
        Self { dw }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.clone() + self.dw.forward(x)
    }
}

/// Grouped 1×1 (`groups = channels`) writes four copies of each channel, then pixel-shuffle
/// rearranges them onto the 2×2 neighbourhood. Identity weights reproduce nearest upsample.
#[derive(Module, Debug)]
pub struct ShuffleUp<B: Backend> {
    expand: Conv2d<B>,
}

impl<B: Backend> ShuffleUp<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let mut expand = Conv2dConfig::new([channels, channels * 4], [1, 1])
            .with_groups(channels)
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        let out = channels * 4;
        let mut data = vec![0.0f32; out];
        data.fill(1.0);
        expand.weight = Param::from_tensor(Tensor::<B, 4>::from_data(
            TensorData::new(data, [out, 1, 1, 1]),
            device,
        ));
        if let Some(bias) = expand.bias.as_mut() {
            *bias = Param::from_tensor(Tensor::<B, 1>::zeros([out], device));
        }
        Self { expand }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = conv2d(
            x,
            self.expand.weight.val(),
            self.expand.bias.as_ref().map(|b| b.val()),
            conv2d_opts(&self.expand),
        );
        pixel_shuffle2(y)
    }
}

/// PyTorch `PixelShuffle(2)`: channel `c * 4 + (h % 2) * 2 + (w % 2)` lands on that subpixel.
fn pixel_shuffle2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c4, h, w] = x.dims();
    let c = c4 / 4;
    let x = x.reshape([n, c, 2, 2, h, w]);
    let x = x.permute([0, 1, 4, 2, 5, 3]);
    x.reshape([n, c, h * 2, w * 2])
}

fn simam<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    let n = ((h * w).saturating_sub(1).max(1)) as f32;
    let center = x.clone() - x.clone().mean_dims(&[2, 3]);
    let square = center.powf_scalar(2.0);
    let y = square.clone() / (square.sum_dims(&[2, 3]) / n + 1e-4) * 0.25 + 0.5;
    x * sigmoid(y)
}

/// 1×1 to C/4, InstanceNorm on that narrow map, 1×1 back to C.
#[derive(Module, Debug)]
pub struct NarrowIn<B: Backend> {
    down: PwConv<B>,
    norm: NormAffine<B>,
    up: PwConv<B>,
}

impl<B: Backend> NarrowIn<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let inner = (channels / 4).max(1);
        Self {
            down: PwConv::new(channels, inner, device),
            norm: NormAffine::new(inner, true, device),
            up: PwConv::new(inner, channels, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let hidden = self.down.forward(x);
        let hidden = self.norm.forward(hidden);
        self.up.forward(hidden)
    }
}

/// The one live plugin. Unused arms stay `None` and contribute no parameters.
#[derive(Module, Debug)]
pub struct Plugin<B: Backend> {
    affine: Option<ChannelAffine<B>>,
    norm: Option<NormAffine<B>>,
    se: Option<SqueezeExcite<B>>,
    film: Option<Film<B>>,
    eca: Option<Eca<B>>,
    grn: Option<Grn<B>>,
    ca: Option<CoordAttn<B>>,
    frelu: Option<FreLu<B>>,
    dw7: Option<DwResidual<B>>,
    shuffle: Option<ShuffleUp<B>>,
    narrow_in: Option<NarrowIn<B>>,
    #[module(skip)]
    kind: PluginKind,
}

impl<B: Backend> Plugin<B> {
    pub fn new(kind: PluginKind, channels: usize, device: &B::Device) -> Self {
        let mut plugin = Self {
            affine: None,
            norm: None,
            se: None,
            film: None,
            eca: None,
            grn: None,
            ca: None,
            frelu: None,
            dw7: None,
            shuffle: None,
            narrow_in: None,
            kind,
        };
        match kind {
            PluginKind::None | PluginKind::SimAm => {}
            PluginKind::Affine => plugin.affine = Some(ChannelAffine::new(channels, device)),
            PluginKind::In => plugin.norm = Some(NormAffine::new(channels, true, device)),
            PluginKind::Ln => plugin.norm = Some(NormAffine::new(channels, false, device)),
            PluginKind::Se => plugin.se = Some(SqueezeExcite::new(channels, device)),
            PluginKind::Film => plugin.film = Some(Film::new(channels, device)),
            PluginKind::Eca => plugin.eca = Some(Eca::new(channels, device)),
            PluginKind::Grn => plugin.grn = Some(Grn::new(channels, device)),
            PluginKind::Ca => plugin.ca = Some(CoordAttn::new(channels, device)),
            PluginKind::Frelu => plugin.frelu = Some(FreLu::new(channels, device)),
            PluginKind::Dw7 => plugin.dw7 = Some(DwResidual::new(channels, 7, device)),
            PluginKind::Shuffle => plugin.shuffle = Some(ShuffleUp::new(channels, device)),
            PluginKind::NarrowIn => plugin.narrow_in = Some(NarrowIn::new(channels, device)),
        }
        plugin
    }

    /// Instance norm with a caller-chosen epsilon. `Plugin::new(In)` stays at [`EPS`].
    pub fn instance_norm(channels: usize, eps: f32, device: &B::Device) -> Self {
        let mut plugin = Self::new(PluginKind::In, channels, device);
        if let Some(norm) = plugin.norm.as_mut() {
            norm.set_eps(eps);
        }
        plugin
    }

    pub fn layer_norm(channels: usize, eps: f32, device: &B::Device) -> Self {
        let mut plugin = Self::new(PluginKind::Ln, channels, device);
        if let Some(norm) = plugin.norm.as_mut() {
            norm.set_eps(eps);
        }
        plugin
    }

    pub fn kind(&self) -> PluginKind {
        self.kind
    }

    pub fn is_shuffle(&self) -> bool {
        self.kind == PluginKind::Shuffle
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self.kind {
            PluginKind::None | PluginKind::Shuffle => x,
            PluginKind::Affine => self.affine.as_ref().expect("affine arm").forward(x),
            PluginKind::In | PluginKind::Ln => self.norm.as_ref().expect("norm arm").forward(x),
            PluginKind::Se => self.se.as_ref().expect("se arm").forward(x),
            PluginKind::Film => self.film.as_ref().expect("film arm").forward(x),
            PluginKind::Eca => self.eca.as_ref().expect("eca arm").forward(x),
            PluginKind::Grn => self.grn.as_ref().expect("grn arm").forward(x),
            PluginKind::Ca => self.ca.as_ref().expect("ca arm").forward(x),
            PluginKind::Frelu => self.frelu.as_ref().expect("frelu arm").forward(x),
            PluginKind::Dw7 => self.dw7.as_ref().expect("dw7 arm").forward(x),
            PluginKind::SimAm => simam(x),
            PluginKind::NarrowIn => self.narrow_in.as_ref().expect("narrow_in arm").forward(x),
        }
    }

    pub fn shuffle_up(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.shuffle.as_ref().expect("shuffle arm").forward(x)
    }

    /// H/4 InstanceNorm γ, β when `kind = in` and `site` matches at export time.
    pub fn host_instance_norm(&self) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        if self.kind != PluginKind::In {
            return None;
        }
        Some(self.norm.as_ref().expect("in arm").host_gamma_beta())
    }

    /// γ, β of a trunk InstanceNorm or LayerNorm.
    pub fn host_norm(&self) -> Option<Result<(Vec<f32>, Vec<f32>), String>> {
        match self.kind {
            PluginKind::In | PluginKind::Ln => {
                Some(self.norm.as_ref().expect("norm arm").host_gamma_beta())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eca_kernel_matches_the_paper_truncation() {
        assert_eq!(eca_kernel(64), 3);
        assert_eq!(eca_kernel(48), 3);
        assert_eq!(eca_kernel(24), 3);
    }

    #[test]
    fn parse_rejects_unknown_names() {
        assert!(parse_kind("cbam").is_err());
        assert!(parse_site("stem").is_err());
        assert_eq!(parse_kind("se").unwrap(), PluginKind::Se);
        assert_eq!(parse_site("full").unwrap(), PluginSite::Full);
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod identity_tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::module::interpolate;
    use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

    type B = NdArray;

    fn sample(channels: usize, h: usize, w: usize) -> Tensor<B, 4> {
        let n = channels * h * w;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 17) % 97) as f32 / 97.0 - 0.3)
            .collect();
        Tensor::from_data(
            TensorData::new(data, [1, channels, h, w]),
            &Default::default(),
        )
    }

    fn max_abs_diff(a: Tensor<B, 4>, b: Tensor<B, 4>) -> f32 {
        let d = (a - b).abs().into_data().to_vec::<f32>().unwrap();
        d.into_iter().fold(0.0f32, f32::max)
    }

    fn assert_identity(kind: PluginKind, channels: usize) {
        let device = Default::default();
        let x = sample(channels, 6, 8);
        let y = Plugin::<B>::new(kind, channels, &device).forward(x.clone());
        let err = max_abs_diff(y, x);
        assert!(err < 1e-4, "{kind:?} left identity by {err}");
    }

    #[test]
    fn identity_inits_reproduce_the_input() {
        for kind in [
            PluginKind::Affine,
            PluginKind::Se,
            PluginKind::Film,
            PluginKind::Eca,
            PluginKind::Grn,
            PluginKind::Ca,
            PluginKind::Frelu,
            PluginKind::Dw7,
        ] {
            assert_identity(kind, 8);
        }
    }

    #[test]
    fn norms_and_simam_stay_finite_and_are_not_the_identity() {
        let device = Default::default();
        let x = sample(8, 6, 8);
        for kind in [
            PluginKind::In,
            PluginKind::Ln,
            PluginKind::SimAm,
            PluginKind::NarrowIn,
        ] {
            let y = Plugin::<B>::new(kind, 8, &device).forward(x.clone());
            let values = y.clone().into_data().to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{kind:?} produced non-finite"
            );
            let err = max_abs_diff(y, x.clone());
            assert!(
                err > 1e-3,
                "{kind:?} should change the activation, diff {err}"
            );
        }
    }

    #[test]
    fn shuffle_identity_matches_nearest_upsample() {
        let device = Default::default();
        let x = sample(4, 3, 5);
        let plugin = Plugin::<B>::new(PluginKind::Shuffle, 4, &device);
        let shuffled = plugin.shuffle_up(x.clone());
        let [_, _, h, w] = x.dims();
        let nearest = interpolate(
            x,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
        let err = max_abs_diff(shuffled, nearest);
        assert!(err < 1e-4, "shuffle init diff {err}");
    }
}
