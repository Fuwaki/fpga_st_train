//! A compact, FPGA-oriented student with one style baked into its weights.
//!
//! Design notes (the reasoning is written up in `runs/note/`):
//!
//! - The style is **fixed**, so the whole style branch of the teacher (style encoder +
//!   modulator + AdaIN statistics) collapses into constants. The student is therefore a
//!   plain fully-convolutional content->stylised mapping; no style input, no modulation.
//! - The layer inventory is deliberately the same vocabulary as the MicroAST teacher's
//!   decoder: reflection-padded 3x3 convolutions, ReLU, 1x1 pointwise convolutions and
//!   2x nearest-neighbour upsampling. Nothing here needs a transposed convolution, a
//!   dilation, a grouped convolution or attention, so every layer maps onto a
//!   line-buffer conv datapath.
//! - **No normalisation layer at any resolution.** The previous student here
//!   (16/32/64 + InstanceNorm after every convolution + nearest upsampling) produced a
//!   strong 2px checkerboard artefact (the evaluation in `teacher-eval/` measured
//!   checkerboard energy up to 0.19, versus 6.7e-7 for the teacher). InstanceNorm after
//!   a nearest-neighbour upsample re-normalises a piecewise-constant signal and amplifies
//!   exactly the Nyquist period. The teacher itself keeps its only normalisation (AdaIN)
//!   at 1/4 resolution, *before* any upsampling; this student does the same by having none.
//! - Reflection padding (not zero padding) everywhere, matching the teacher.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::{conv2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode, PaddedConvOptions};

use crate::quant;
use crate::student::StudentModel;

/// ReflectionPad(k/2) + Conv2d. The same trick the teacher uses, because Burn has no
/// reflect-padded convolution primitive.
#[derive(Module, Debug)]
pub struct Conv<B: Backend> {
    pub conv: Conv2d<B>,
    pad: usize,
}

impl<B: Backend> Conv<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(bias)
            .init(device);
        Self {
            conv,
            pad: kernel / 2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with(x, self.conv.weight.val())
    }

    /// Reflection padding applied before the (valid) convolution.
    pub fn reflect_pad(&self) -> usize {
        self.pad
    }

    /// The convolution kernel, as a plain tensor. Used by quantisation and by the export.
    pub fn weight(&self) -> Tensor<B, 4> {
        self.conv.weight.val()
    }

    /// The same convolution with an explicitly supplied kernel.
    ///
    /// The int8 path needs this: the fake-quantised weight is an activation-dependent tensor,
    /// not a parameter, so it cannot be written back into the module.
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
        conv2d(
            x,
            weight,
            bias,
            PaddedConvOptions::asymmetric(
                self.conv.stride,
                [0, 0],
                [0, 0],
                self.conv.dilation,
                self.conv.groups,
            ),
        )
    }
}

/// Observe (for calibration) and then optionally fake-quantise one activation.
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

/// The weight of one convolution, fake-quantised when the low-bit path is active.
fn wq<B: Backend>(conv: &Conv<B>, scheme: Option<&quant::QuantScheme>) -> Tensor<B, 4> {
    let weight = conv.weight();
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

/// Two 3x3 convolutions with a residual connection (the teacher's ResidualLayer).
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    pub c1: Conv<B>,
    pub c2: Conv<B>,
}

impl<B: Backend> ResBlock<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            c1: Conv::new(channels, channels, 3, 1, true, device),
            c2: Conv::new(channels, channels, 3, 1, true, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let inner = self.c2.forward(relu(self.c1.forward(x.clone())));
        x + inner
    }
}

/// Nearest-neighbour 2x upsample: on hardware this is a wire, not an arithmetic unit.
fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// The student: content [N,3,H,W] in [0,1] -> stylised [N,3,H,W].
///
/// The decoder intentionally mirrors the teacher's: the expensive 3x3 convolutions run
/// at reduced resolution, and the full-resolution tail is narrow (16 channels before the
/// RGB head). That is the shape MicroAST reaches state-of-the-art style with, and on this
/// hardware it is not a detail -- a single full-resolution 64 -> 32 3x3 convolution costs
/// 1.2 GMACs @256^2, more than the entire 1/4-resolution trunk.
#[derive(Module, Debug)]
pub struct TinyStudent<B: Backend> {
    /// 3 -> base (stride 1), base -> 2*base (stride 2), 2*base -> 2*base (stride 2).
    pub enc: Vec<Conv<B>>,
    /// Residual blocks at 1/4 resolution.
    pub res: Vec<ResBlock<B>>,
    /// 3x3 convolutions applied after each 2x upsample.
    pub up3: Vec<Conv<B>>,
    /// 1x1 convolutions refining the upsampled features.
    pub up1: Vec<Conv<B>>,
    /// Final 3x3 convolution to RGB.
    pub head: Conv<B>,
}

impl<B: Backend> TinyStudent<B> {
    /// Widths: base at full resolution, 2*base below it, then the two decoder widths.
    pub fn new(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        device: &B::Device,
    ) -> Self {
        let c1 = base;
        let c2 = base * 2;
        let enc = vec![
            Conv::new(3, c1, 3, 1, true, device),
            Conv::new(c1, c2, 3, 2, true, device),
            Conv::new(c2, c2, 3, 2, true, device),
        ];
        let res = (0..res_blocks).map(|_| ResBlock::new(c2, device)).collect();
        let up3 = vec![
            Conv::new(c2, mid, 3, 1, true, device),
            Conv::new(mid, fine, 3, 1, true, device),
        ];
        let up1 = vec![
            Conv::new(mid, mid, 1, 1, true, device),
            Conv::new(fine, fine, 1, 1, true, device),
        ];
        let head = Conv::new(fine, 3, 3, 1, true, device);
        Self {
            enc,
            res,
            up3,
            up1,
            head,
        }
    }
}

impl<B: Backend> TinyStudent<B> {
    /// Number of quantised activation points: one per convolution input, plus the image.
    pub fn quant_point_count(&self) -> usize {
        8 + 2 * self.res.len()
    }

    /// Names of the quantised points, in the order the network consumes them.
    ///
    /// Index `i` is the activation entering the `i`-th quantised operation; the index of a
    /// residual block's *output* grid is the index of the next block's first convolution, so a
    /// block never needs a grid of its own.
    pub fn quant_layout(&self) -> Vec<String> {
        let mut names = vec!["image".to_owned()];
        for i in 0..self.enc.len() - 1 {
            names.push(format!("enc.{}.in", i + 1));
        }
        for (r, _) in self.res.iter().enumerate() {
            names.push(format!("res.{r}.c1.in"));
            names.push(format!("res.{r}.c2.in"));
        }
        names.push("up3.0.in".to_owned());
        names.push("up1.0.in".to_owned());
        names.push("up3.1.in".to_owned());
        names.push("up1.1.in".to_owned());
        names.push("head.in".to_owned());
        names
    }

    /// Forward pass with optional low-bit fake quantisation of activations and weights.
    ///
    /// `scheme`: `None` runs the fp32 network; `Some(s)` quantises every convolution input
    /// with `s.acts[i]` (see [Self::quant_layout]) and every kernel per output channel.
    ///
    /// `observe` is called with `(point, tensor)` *before* that tensor is quantised, which is
    /// what the calibration pass uses to measure activation ranges.
    pub fn forward_core<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        scheme: Option<&quant::QuantScheme>,
        mut observe: F,
    ) -> Tensor<B, 4> {
        let base = 3 + 2 * self.res.len();

        let mut h = qa(content, scheme, 0, &mut observe);
        h = relu(self.enc[0].forward_with(h, wq(&self.enc[0], scheme)));
        h = qa(h, scheme, 1, &mut observe);
        h = relu(self.enc[1].forward_with(h, wq(&self.enc[1], scheme)));
        h = qa(h, scheme, 2, &mut observe);
        h = relu(self.enc[2].forward_with(h, wq(&self.enc[2], scheme)));

        for (r, block) in self.res.iter().enumerate() {
            let point = 3 + 2 * r;
            let input = qa(h, scheme, point, &mut observe);
            let activated = relu(block.c1.forward_with(input.clone(), wq(&block.c1, scheme)));
            let activated = qa(activated, scheme, point + 1, &mut observe);
            let inner = block.c2.forward_with(activated, wq(&block.c2, scheme));
            // The add needs both operands on one grid; that grid is the next consumer's input,
            // which is exactly the point the following qa() observes.
            h = match scheme {
                Some(scheme) => {
                    let out = scheme.acts[point + 2];
                    let qmax = scheme.act_max();
                    quant::fake_quant_act(input, out, qmax)
                        + quant::fake_quant_act(inner, out, qmax)
                }
                None => input + inner,
            };
        }

        h = upsample2x(h);

        h = qa(h, scheme, base, &mut observe);
        h = relu(self.up3[0].forward_with(h, wq(&self.up3[0], scheme)));
        h = qa(h, scheme, base + 1, &mut observe);
        h = relu(self.up1[0].forward_with(h, wq(&self.up1[0], scheme)));

        h = upsample2x(h);

        h = qa(h, scheme, base + 2, &mut observe);
        h = relu(self.up3[1].forward_with(h, wq(&self.up3[1], scheme)));
        h = qa(h, scheme, base + 3, &mut observe);
        h = relu(self.up1[1].forward_with(h, wq(&self.up1[1], scheme)));

        h = qa(h, scheme, base + 4, &mut observe);
        let image = self.head.forward_with(h, wq(&self.head, scheme));
        // The head is the one layer whose output is the image, so it is quantised to unsigned
        // uint8 on the 1/255 grid. This is part of the simulated forward pass, not an optional
        // extra: the export clamps the head exactly the same way, and a simulation that skipped
        // it would report a fidelity the accelerator cannot reach.
        match scheme {
            Some(_) => quant::fake_quant_output(image),
            None => image,
        }
    }

    /// Forward pass that also exposes the three stages matching the teacher's decoder
    /// (64 channels @1/4, 32 @1/2, 16 @1). See StudentModel::intermediate.
    pub fn intermediates(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut h = content.clone();
        for conv in &self.enc {
            h = relu(conv.forward(h));
        }
        for block in &self.res {
            h = block.forward(h);
        }
        let trunk = h.clone();

        h = upsample2x(h);
        h = relu(self.up3[0].forward(h));
        h = relu(self.up1[0].forward(h));
        let half = h.clone();

        h = upsample2x(h);
        h = relu(self.up3[1].forward(h));
        h = relu(self.up1[1].forward(h));
        let full = h.clone();

        (self.head.forward(h), vec![trunk, half, full])
    }
}

impl<B: Backend> StudentModel<B> for TinyStudent<B> {
    fn stylize(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        self.intermediates(content).0
    }

    fn intermediate(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        self.intermediates(content)
    }
}

impl<B: Backend> quant::Quantizable<B> for TinyStudent<B> {
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
