//! MicroAST (AAAI 2023, "Towards Super-Fast Ultra-Resolution Arbitrary Style Transfer").
//!
//! Ported 1:1 from the official PyTorch release (net_microAST.py). The inference path
//! (TestNet) is used: no VGG anywhere, just two micro encoders, a modulator and a decoder.
//!
//! # Fidelity notes
//!
//! Every numeric detail is matched against PyTorch:
//! - ReflectionPad2d(k/2) before every convolution (Burn has no reflect-padded conv,
//!   so padding is applied explicitly and the conv uses Valid padding).
//! - Grouped convolutions are used exactly as in the original (groups 16/32/64).
//! - Bilinear upsampling uses align_corners = false, matching nn.Upsample defaults.
//! - The variance in AdaIN uses the unbiased (n-1) estimator, matching torch.var.
//!
//! # Structure
//!
//! content -> ContentEncoder -> x = [x1 (64ch, /4), x2 (64ch, /4)]
//! style   -> StyleEncoder   -> s = [s1, s2]
//!                          -> Modulator -> w = [w1, w2], b = [b1, b2]  (64-dim each)
//!
//! Decoder:
//!   AdaIN(x2, s2) -> dec1 (residual, filter-modulated by w1,b1)
//!   AdaIN(   , s1) -> dec2 (residual, filter-modulated by w0,b0)
//!   -> upsample x2 -> convs -> upsample x2 -> convs -> RGB

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};

use crate::backend::InferBackend;
use crate::teacher::Teacher;

/// Epsilon used by MicroAST's calc_mean_std.
pub const ADAIN_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------------------

/// nn.ReflectionPad2d(k/2) followed by nn.Conv2d(...) -- MicroAST's ConvLayer.
#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    pub conv: Conv2d<B>,
    kernel_size: usize,
}

impl<B: Backend> ConvLayer<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_groups(groups)
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        Self { conv, kernel_size }
    }

    /// Half the kernel size: the ReflectionPad2d amount on each side.
    pub const fn pad(&self) -> usize {
        self.kernel_size / 2
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let p = self.pad();
        let x = if p > 0 {
            x.pad([(p, p), (p, p)], PadMode::Reflect)
        } else {
            x
        };
        self.conv.forward(x)
    }
}

/// Two 3x3 convs with a residual connection. Supports MicroAST's filter modulation.
#[derive(Module, Debug)]
pub struct ResidualLayer<B: Backend> {
    pub conv1: ConvLayer<B>,
    pub conv2: ConvLayer<B>,
}

impl<B: Backend> ResidualLayer<B> {
    pub fn new(channels: usize, kernel_size: usize, groups: usize, device: &B::Device) -> Self {
        Self {
            conv1: ConvLayer::new(channels, channels, kernel_size, 1, groups, device),
            conv2: ConvLayer::new(channels, channels, kernel_size, 1, groups, device),
        }
    }

    /// Plain residual path (MicroAST's filterMod = false).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let inner = relu(self.conv1.forward(x.clone()));
        x + self.conv2.forward(inner)
    }

    /// Style-modulated residual path (filterMod = true).
    ///
    /// weight and bias are [N, C, 1, 1] vectors predicted by the Modulator and
    /// broadcast over the spatial dimensions.
    pub fn forward_modulated(
        &self,
        x: Tensor<B, 4>,
        weight: Tensor<B, 4>,
        bias: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x1 = self.conv1.forward(x.clone());
        let x2 = weight.clone() * x1 + bias.clone() * x.clone();
        let x3 = relu(x2);
        let x4 = self.conv2.forward(x3.clone());
        let x5 = weight * x4 + bias * x3;
        x + x5
    }
}

// ---------------------------------------------------------------------------------------
// Encoder / Modulator / Decoder
// ---------------------------------------------------------------------------------------

/// Micro content/style encoder. Downsamples by 4 and returns two feature levels.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    /// The five strided/pointwise convs of enc1 (PyTorch indices 0, 2, 4, 6, 8).
    pub convs: Vec<ConvLayer<B>>,
    /// enc1[10]
    pub res: ResidualLayer<B>,
    /// enc2[0]
    pub res2: ResidualLayer<B>,
}

impl<B: Backend> Encoder<B> {
    /// Upstream builds the encoder with slim_factor = 1: 16 -> 32 -> 64 channels.
    pub fn new(device: &B::Device) -> Self {
        Self {
            convs: vec![
                ConvLayer::new(3, 16, 9, 1, 1, device),
                // grouped 3x3 stride-2: groups == in_channels, out == 2 * in
                ConvLayer::new(16, 32, 3, 2, 16, device),
                ConvLayer::new(32, 32, 1, 1, 1, device),
                ConvLayer::new(32, 64, 3, 2, 32, device),
                ConvLayer::new(64, 64, 1, 1, 1, device),
            ],
            res: ResidualLayer::new(64, 3, 1, device),
            res2: ResidualLayer::new(64, 3, 1, device),
        }
    }

    /// Returns [x1, x2]: both 64 channels at 1/4 resolution.
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        // Upstream applies ReLU after every one of the five convs, including the last.
        let mut h = x;
        for conv in &self.convs {
            h = relu(conv.forward(h));
        }
        let x1 = self.res.forward(h);
        let x2 = self.res2.forward(x1.clone());
        vec![x1, x2]
    }
}

/// Global-average-pool to 1x1 (nn.AdaptiveAvgPool2d((1, 1))).
fn global_pool<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    x.mean_dim(2).mean_dim(3)
}

/// Turns style features into per-channel scale/shift vectors for the decoder.
#[derive(Module, Debug)]
pub struct Modulator<B: Backend> {
    pub weight1: ConvLayer<B>,
    pub bias1: ConvLayer<B>,
    pub weight2: ConvLayer<B>,
    pub bias2: ConvLayer<B>,
}

impl<B: Backend> Modulator<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            weight1: ConvLayer::new(64, 64, 3, 1, 64, device),
            bias1: ConvLayer::new(64, 64, 3, 1, 64, device),
            weight2: ConvLayer::new(64, 64, 3, 1, 64, device),
            bias2: ConvLayer::new(64, 64, 3, 1, 64, device),
        }
    }

    /// Returns (weights, biases); each is [w from s1, w from s2].
    pub fn forward(&self, style_feats: &[Tensor<B, 4>]) -> (Vec<Tensor<B, 4>>, Vec<Tensor<B, 4>>) {
        let w1 = global_pool(self.weight1.forward(style_feats[0].clone()));
        let b1 = global_pool(self.bias1.forward(style_feats[0].clone()));
        let w2 = global_pool(self.weight2.forward(style_feats[1].clone()));
        let b2 = global_pool(self.bias2.forward(style_feats[1].clone()));
        (vec![w1, w2], vec![b1, b2])
    }
}

/// Micro decoder: two modulated residual blocks then two bilinear upsampling stages.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub dec1: ResidualLayer<B>,
    pub dec2: ResidualLayer<B>,
    /// dec3 convs (PyTorch indices 1, 3, 6, 8, 10).
    pub up: Vec<ConvLayer<B>>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            dec1: ResidualLayer::new(64, 3, 1, device),
            dec2: ResidualLayer::new(64, 3, 1, device),
            up: vec![
                ConvLayer::new(64, 32, 3, 1, 32, device),
                ConvLayer::new(32, 32, 1, 1, 1, device),
                ConvLayer::new(32, 16, 3, 1, 16, device),
                ConvLayer::new(16, 16, 1, 1, 1, device),
                ConvLayer::new(16, 3, 9, 1, 1, device),
            ],
        }
    }

    pub fn forward(
        &self,
        content_feats: &[Tensor<B, 4>],
        style_feats: &[Tensor<B, 4>],
        weights: &[Tensor<B, 4>],
        biases: &[Tensor<B, 4>],
        alpha: f32,
    ) -> Tensor<B, 4> {
        let x1 = blend(
            adaptive_instance_norm(content_feats[1].clone(), style_feats[1].clone()),
            content_feats[1].clone(),
            alpha,
        );
        let x2 = self
            .dec1
            .forward_modulated(x1, weights[1].clone(), biases[1].clone());
        let x3 = blend(
            adaptive_instance_norm(x2.clone(), style_feats[0].clone()),
            x2,
            alpha,
        );
        let x4 = self
            .dec2
            .forward_modulated(x3, weights[0].clone(), biases[0].clone());

        let [_, _, h, w] = x4.dims();
        let mut y = upsample2x(x4, h, w);
        y = relu(self.up[0].forward(y));
        y = relu(self.up[1].forward(y));
        let [_, _, h, w] = y.dims();
        y = upsample2x(y, h, w);
        y = relu(self.up[2].forward(y));
        y = relu(self.up[3].forward(y));
        self.up[4].forward(y)
    }
}

/// alpha * stylised + (1 - alpha) * content, matching the upstream blend.
fn blend<B: Backend>(stylised: Tensor<B, 4>, content: Tensor<B, 4>, alpha: f32) -> Tensor<B, 4> {
    stylised.mul_scalar(alpha) + content.mul_scalar(1.0 - alpha)
}

/// Bilinear 2x upsample with align_corners = false (PyTorch nn.Upsample default).
pub(crate) fn upsample2x<B: Backend>(x: Tensor<B, 4>, h: usize, w: usize) -> Tensor<B, 4> {
    let opts = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false);
    interpolate(x, [h * 2, w * 2], opts)
}

// ---------------------------------------------------------------------------------------
// AdaIN
// ---------------------------------------------------------------------------------------

/// Per-channel mean and std over the spatial dimensions.
///
/// Uses the unbiased (n-1) variance estimator, matching torch.Tensor.var defaults.
pub fn calc_mean_std<B: Backend>(x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_, _, h, w] = x.dims();
    let n = (h * w) as f32;
    let mean = x.clone().mean_dim(2).mean_dim(3);
    let centered = x - mean.clone();
    let var = (centered.clone() * centered)
        .mean_dim(2)
        .mean_dim(3)
        .mul_scalar(n / (n - 1.0));
    let std = (var.add_scalar(ADAIN_EPS)).sqrt();
    (mean, std)
}

/// adaptive_instance_normalization from the upstream function.py.
pub fn adaptive_instance_norm<B: Backend>(
    content: Tensor<B, 4>,
    style: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let (content_mean, content_std) = calc_mean_std(content.clone());
    let (style_mean, style_std) = calc_mean_std(style);
    ((content - content_mean) / content_std) * style_std + style_mean
}

// ---------------------------------------------------------------------------------------
// Full network
// ---------------------------------------------------------------------------------------

/// MicroAST inference network (the upstream TestNet: no VGG, no losses).
#[derive(Module, Debug)]
pub struct MicroAst<B: Backend> {
    pub content_encoder: Encoder<B>,
    pub style_encoder: Encoder<B>,
    pub modulator: Modulator<B>,
    pub decoder: Decoder<B>,
}

impl<B: Backend> MicroAst<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            content_encoder: Encoder::new(device),
            style_encoder: Encoder::new(device),
            modulator: Modulator::new(device),
            decoder: Decoder::new(device),
        }
    }

    /// The ensemble size of the style encoder + modulator, in MACs, is style-only work:
    /// it can be precomputed once per style and cached (see the README).
    pub fn forward(&self, content: Tensor<B, 4>, style: Tensor<B, 4>, alpha: f32) -> Tensor<B, 4> {
        let style_feats = self.style_encoder.forward(style);
        let (weights, biases) = self.modulator.forward(&style_feats);
        let content_feats = self.content_encoder.forward(content);
        self.decoder
            .forward(&content_feats, &style_feats, &weights, &biases, alpha)
    }
}

impl Teacher for MicroAst<InferBackend> {
    fn stylize(
        &self,
        content: Tensor<InferBackend, 4>,
        style: Tensor<InferBackend, 4>,
        alpha: f32,
    ) -> Tensor<InferBackend, 4> {
        self.forward(content, style, alpha)
    }
}
