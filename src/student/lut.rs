//! Differentiable LUT-Conv, the training surrogate from HGQ-LUT.
//!
//! HGQ-LUT (arXiv:2604.22293) replaces a multiply-accumulate with one logical
//! LUT per scalar input. At train time each LUT is a tiny MLP, and the output
//! channel is the sum of those LUT outputs (Algorithm 1). A convolution is the
//! same layer after im2col. Heterogeneous bit-width quantizers and the EBOP
//! resource penalty are not here. [LutConv2d::to_manifest_layer] freezes each
//! trained map into the symmetric codebook the conv manifest already uses.
//!
//! The tiny_dw encoder does not call [LutEncoder::forward]. It calls
//! [LutEncoder::forward_dw], which centers the RGB image onto that codebook
//! and snaps both exits back onto the same real span. The separable decoder,
//! the H/4 InstanceNorm and the H/2 skip-add then see ordinary feature maps.

use anyhow::{Context, Result, bail};
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::TensorPrimitive;
use burn::tensor::activation::tanh;
use burn::tensor::ops::PadMode;
use burn::tensor::s;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use crate::quant::{Bits, LutLayer, fake_quant_act};

/// Negative slope used between LUT-Conv layers. Small enough that a dead hard
/// ReLU does not happen, and a shift on an FPGA.
pub const LUT_LEAKY_SLOPE: f32 = 0.05;

/// Real half-range of the symmetric codebook shared by the image and the
/// encoder exits. An 8-bit code then steps by `2/127`.
///
/// Pixels in `[0, 1]` are centered into `[-1, 1]`, inside this span, so the
/// paper's WRAP input quantizer and a saturating snap write the same code.
/// The H/2 skip is added to an InstanceNorm feature and is not normalised
/// itself, so both exits are snapped back into this span instead of being
/// handed over as an integer sum.
pub const LUT_CODE_SPAN: f32 = 2.0;

/// One-hidden-layer surrogate for a bank of 1-input logical LUTs.
///
/// `x` is `[N, S, n_in]`. Each pair `(i, j)` has its own map
/// `f_ij(x_i) = w_lin·x_i + w_out · tanh(x_i * w_hidden + b_hidden) + b_out`,
/// and `y_j = Σ_i f_ij(x_i)`.
#[derive(Module, Debug)]
pub struct LutDense<B: Backend> {
    /// `[n_in, n_out, hidden]`
    w_hidden: Param<Tensor<B, 3>>,
    b_hidden: Param<Tensor<B, 3>>,
    /// `[n_in, n_out, hidden]`
    w_out: Param<Tensor<B, 3>>,
    /// Linear term of each univariate map, `[n_in, n_out]`.
    ///
    /// Still one number per logical LUT: at export the linear piece and the tanh
    /// piece are the same truth table. Keeping it explicit lets `f(x) = sx` be
    /// represented exactly, which is the conv this layer has to replace. A pure
    /// tanh MLP only matches that line on a small interval.
    w_lin: Param<Tensor<B, 2>>,
    /// `[n_in, n_out]`
    b_out: Param<Tensor<B, 2>>,
}

impl<B: Backend> LutDense<B> {
    pub fn new(n_in: usize, n_out: usize, hidden: usize, device: &B::Device) -> Self {
        assert!(n_in > 0 && n_out > 0 && hidden > 0);
        let fan = (n_in as f64).sqrt();
        // Summed over n_in. A small positive total keeps a following ReLU live.
        // Equation 3 puts `b/N` inside each LUT so the sum of the biases is `b`.
        let bias = 0.05 / n_in as f64;
        Self {
            w_hidden: Param::from_tensor(Tensor::random(
                [n_in, n_out, hidden],
                Distribution::Normal(0.0, 0.75),
                device,
            )),
            b_hidden: Param::from_tensor(Tensor::zeros([n_in, n_out, hidden], device)),
            w_out: Param::from_tensor(Tensor::random(
                [n_in, n_out, hidden],
                Distribution::Normal(0.0, 0.1 / fan),
                device,
            )),
            w_lin: Param::from_tensor(Tensor::random(
                [n_in, n_out],
                Distribution::Normal(0.0, 1.0 / fan),
                device,
            )),
            b_out: Param::from_tensor(Tensor::ones([n_in, n_out], device) * bias),
        }
    }

    /// `[N, S, n_in] -> [N, S, n_out]`.
    ///
    /// `bits >= 2` saturates each logical LUT (Algorithm 1, `q_out`) before the
    /// sum. `bits == 0` is the float surrogate used when fitting a table.
    ///
    /// Training does not record this loop. Burn would keep every tap until the
    /// backward, and the H/2 stage at batch 32 / crop 192 is about 22 GB.
    /// The autodiff node saves the inputs and replays the loop.
    pub fn forward(&self, x: Tensor<B, 3>, bits: u8) -> Tensor<B, 3> {
        <LutDense<B> as DenseForward<B>>::apply(self, x, bits)
    }

    fn forward_eager(&self, x: Tensor<B, 3>, bits: u8) -> Tensor<B, 3> {
        dense_eager(
            x,
            self.w_lin.val(),
            self.b_out.val(),
            self.w_hidden.val(),
            self.b_hidden.val(),
            self.w_out.val(),
            bits,
        )
    }
}

/// LUT-Conv2d: reflection-padded im2col, then [`LutDense`].
///
/// `groups` stays 1, matching HGQ2. A 3×3 over `C` channels therefore has
/// `C * 9` logical inputs at every spatial site.
#[derive(Module, Debug)]
pub struct LutConv2d<B: Backend> {
    dense: LutDense<B>,
    kernel: usize,
    stride: usize,
    pad: usize,
}

impl<B: Backend> LutConv2d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        hidden: usize,
        device: &B::Device,
    ) -> Self {
        assert!(kernel > 0 && stride > 0);
        let n_in = in_channels * kernel * kernel;
        Self {
            dense: LutDense::new(n_in, out_channels, hidden, device),
            kernel,
            stride,
            pad,
        }
    }

    /// Float surrogate: no per-tap quantiser.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_bits(x, 0)
    }

    pub fn forward_bits(&self, x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4> {
        <LutConv2d<B> as ConvForward<B>>::apply(self, x, bits)
    }

    fn forward_bits_windows(&self, x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4> {
        conv_windows(
            x,
            self.dense.w_lin.val(),
            self.dense.b_out.val(),
            self.dense.w_hidden.val(),
            self.dense.b_hidden.val(),
            self.dense.w_out.val(),
            self.kernel,
            self.stride,
            self.pad,
            bits,
        )
    }

    /// Enumerate the symmetric input codebook and store one quantised output per code.
    ///
    /// `in_scale` is the real value of input code `+1`. The output scale is the
    /// smallest symmetric scale that holds every table entry. This fitted scale
    /// is the standalone manifest. [LutEncoder::forward_dw] trains against the
    /// fixed [LUT_CODE_SPAN] codebook instead, and does not divide by
    /// `next_in_scale`.
    pub fn to_manifest_layer(
        &self,
        name: &str,
        in_bits: u8,
        out_bits: u8,
        in_scale: f32,
        leaky_slope: f32,
    ) -> Result<LutLayer> {
        if !(2..=8).contains(&in_bits) || !(2..=8).contains(&out_bits) {
            bail!("LUT truth tables support 2 to 8 bits, got in {in_bits} out {out_bits}");
        }
        if !(in_scale.is_finite() && in_scale > 0.0) {
            bail!("LUT input scale must be finite and positive, got {in_scale}");
        }
        let weights = self.dense.host_weights()?;
        let in_qmax = Bits::max_value(in_bits) as i32;
        let out_qmax = Bits::max_value(out_bits) as i32;
        let table_len = (2 * in_qmax + 1) as usize;
        let mut samples = Vec::with_capacity(weights.n_in * weights.n_out * table_len);
        let mut max_abs = 0.0f32;
        for input in 0..weights.n_in {
            for output in 0..weights.n_out {
                for code in -in_qmax..=in_qmax {
                    let value = weights.eval(input, output, code as f32 * in_scale);
                    max_abs = max_abs.max(value.abs());
                    samples.push(value);
                }
            }
        }
        let out_scale = if max_abs == 0.0 {
            1.0
        } else {
            max_abs / out_qmax as f32
        };
        let tables = samples
            .into_iter()
            .map(|value| round_clamp(value / out_scale, out_qmax))
            .collect::<Vec<_>>();
        let kernel = self.kernel * self.kernel;
        let in_channels = weights.n_in / kernel;
        let span = weights.n_in as f32 * out_qmax as f32 * out_scale;
        let next_in_scale = if span == 0.0 {
            1.0
        } else {
            span / in_qmax as f32
        };
        let activation = if leaky_slope == 0.0 {
            "identity"
        } else {
            "leaky_relu"
        };
        Ok(LutLayer {
            name: name.to_owned(),
            in_channels,
            out_channels: weights.n_out,
            kernel: [self.kernel, self.kernel],
            stride: [self.stride, self.stride],
            reflect_pad: self.pad,
            fan_in: weights.n_in,
            in_bits,
            out_bits,
            in_qmax,
            out_qmax,
            in_scale,
            out_scale,
            next_in_scale,
            reduce: "sum".to_owned(),
            activation: activation.to_owned(),
            leaky_slope: (leaky_slope != 0.0).then_some(leaky_slope),
            table_len,
            tables,
            post_relu: false,
        })
    }

    /// Every table entry equals the quantised univariate map. A mismatch means the
    /// manifest layout and the MLP no longer describe the same function.
    pub fn matches_table(&self, layer: &LutLayer) -> Result<()> {
        let weights = self.dense.host_weights()?;
        if layer.tables.len() != weights.n_in * weights.n_out * layer.table_len {
            bail!(
                "{} table has {} entries, expected {} x {} x {}",
                layer.name,
                layer.tables.len(),
                weights.n_in,
                weights.n_out,
                layer.table_len
            );
        }
        for input in 0..weights.n_in {
            for output in 0..weights.n_out {
                for code in -layer.in_qmax..=layer.in_qmax {
                    let value = weights.eval(input, output, code as f32 * layer.in_scale);
                    let expected = round_clamp(value / layer.out_scale, layer.out_qmax);
                    let found = layer.code(input, output, code);
                    if found != expected {
                        bail!(
                            "{name} lut ({input}, {output}) code {code}: table {found}, mlp quantised to {expected}",
                            name = layer.name,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

struct HostLut {
    n_in: usize,
    n_out: usize,
    hidden: usize,
    w_hidden: Vec<f32>,
    b_hidden: Vec<f32>,
    w_out: Vec<f32>,
    w_lin: Vec<f32>,
    b_out: Vec<f32>,
}

impl HostLut {
    fn eval(&self, input: usize, output: usize, x: f32) -> f32 {
        let hidden = self.hidden;
        let base = (input * self.n_out + output) * hidden;
        let mut y =
            self.w_lin[input * self.n_out + output] * x + self.b_out[input * self.n_out + output];
        for d in 0..hidden {
            let z = x * self.w_hidden[base + d] + self.b_hidden[base + d];
            y += self.w_out[base + d] * z.tanh();
        }
        y
    }
}

impl<B: Backend> LutDense<B> {
    fn host_weights(&self) -> Result<HostLut> {
        let n_in = self.w_lin.val().dims()[0];
        let n_out = self.w_lin.val().dims()[1];
        let hidden = self.w_hidden.val().dims()[2];
        let w_lin = host_f32(self.w_lin.val())?;
        Ok(HostLut {
            n_in,
            n_out,
            hidden,
            w_hidden: host_f32(self.w_hidden.val())?,
            b_hidden: host_f32(self.b_hidden.val())?,
            w_out: host_f32(self.w_out.val())?,
            w_lin,
            b_out: host_f32(self.b_out.val())?,
        })
    }
}

fn host_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .context("LUT weight tensor is not f32")
}

fn round_clamp(value: f32, qmax: i32) -> i16 {
    let limit = qmax as f32;
    value.round().clamp(-limit, limit) as i16
}

/// Center an `[0, 1]` RGB batch onto the symmetric codebook.
///
/// Mid-gray lands on code 0. Values already inside the span are not wrapped;
/// the paper uses WRAP only to avoid an on-chip comparator, and WRAP and SAT
/// agree on this interval.
pub fn quantize_rgb<B: Backend>(x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4> {
    quantize_codes(x.mul_scalar(2.0).add_scalar(-1.0), bits)
}

fn quantize_codes<B: Backend, const D: usize>(x: Tensor<B, D>, bits: u8) -> Tensor<B, D> {
    if bits < 2 {
        return x;
    }
    let qmax = Bits::max_value(bits);
    fake_quant_act(x, LUT_CODE_SPAN / qmax, qmax)
}

/// LUT-Conv stack used as a stand-in for the student encoder.
/// Each layer is followed by a leaky ReLU. The negative slope is 0.05 so a
/// layer that steps below zero still sends a gradient; a hard ReLU there
/// stays dead for the rest of training.
#[derive(Module, Debug)]
pub struct LutEncoder<B: Backend> {
    pub layers: Vec<LutConv2d<B>>,
    /// `0` keeps the float surrogate. `2..=8` is the symmetric codebook width.
    #[module(skip)]
    bits: u8,
}

impl<B: Backend> LutEncoder<B> {
    /// `channels` is `[in, hidden..., out]`. One LUT-Conv per consecutive pair.
    /// Every layer shares `stride`. This is the stack the standalone manifest
    /// exports; the tiny_dw encoder uses [Self::dw] because the stem is stride 1
    /// and the next two stages are stride 2.
    pub fn new(
        channels: &[usize],
        kernel: usize,
        stride: usize,
        pad: usize,
        hidden: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            channels.len() >= 2,
            "encoder needs an input and an output width"
        );
        let layers = channels
            .windows(2)
            .map(|pair| LutConv2d::new(pair[0], pair[1], kernel, stride, pad, hidden, device))
            .collect();
        Self { layers, bits: 0 }
    }

    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// tiny_dw encoder: full-resolution `3 → base`, then two stride-2 stages
    /// to `base * 2`. The H/2 map is the decoder skip. The H/4 map enters the
    /// residual trunk.
    pub fn dw(base: usize, hidden: usize, bits: u8, device: &B::Device) -> Self {
        let wide = base * 2;
        Self {
            layers: vec![
                LutConv2d::new(3, base, 3, 1, 1, hidden, device),
                LutConv2d::new(base, wide, 3, 2, 1, hidden, device),
                LutConv2d::new(wide, wide, 3, 2, 1, hidden, device),
            ],
            bits,
        }
    }

    /// Float stack. No image centering and no exit snap.
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for layer in &self.layers {
            x = leaky_relu(layer.forward(x));
        }
        x
    }

    /// `(H/2 skip, H/4 trunk)`, both real tensors inside [LUT_CODE_SPAN].
    ///
    /// The sum of the logical LUTs is already the layer output: each bias is
    /// `b/N` and each linear map is initialised at `1/sqrt(N)`, which is how
    /// Eq. (3) recovers a dense layer. Dividing by the fan-in again, or by the
    /// export's worst-case `next_in_scale`, would shrink a typical code toward
    /// zero before the skip-add. The exit snap is a saturating quantiser onto
    /// the same codebook as the image, so a runaway sum cannot swamp the add.
    pub fn forward_dw(&self, image: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        assert_eq!(
            self.layers.len(),
            3,
            "the dw LUT encoder is a stem plus two stride-2 stages"
        );
        let mut x = quantize_rgb(image, self.bits);
        let mut skip = None;
        for (index, layer) in self.layers.iter().enumerate() {
            if index > 0 {
                x = quantize_codes(x, self.bits);
            }
            x = quantize_codes(leaky_relu(layer.forward_bits(x, self.bits)), self.bits);
            if index == 1 {
                skip = Some(x.clone());
            }
        }
        (skip.expect("the H/2 LUT stage is the decoder skip"), x)
    }
}

/// Depthwise 3×3 LUT-Conv: one univariate table per tap per channel.
///
/// Channels do not mix. A 4-bit table has 15 entries, so one tap fits in a
/// single FPGA logic cell and the nine taps of a kernel add in one clock.
/// The pointwise 1×1 that follows stays a multiply and maps onto DSP.
#[derive(Module, Debug)]
pub struct LutDepthwise2d<B: Backend> {
    pub channels: Vec<LutConv2d<B>>,
    #[module(skip)]
    taps: usize,
}

impl<B: Backend> LutDepthwise2d<B> {
    pub fn new(
        channels: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        hidden: usize,
        device: &B::Device,
    ) -> Self {
        assert!(channels > 0 && kernel > 0);
        Self {
            channels: (0..channels)
                .map(|_| LutConv2d::new(1, 1, kernel, stride, pad, hidden, device))
                .collect(),
            taps: kernel * kernel,
        }
    }

    /// Truth-table rows if this layer is frozen at `bits`. `bits < 2` has no table.
    pub fn table_entries(&self, bits: u8) -> usize {
        if bits < 2 {
            return 0;
        }
        let len = 2 * Bits::max_value(bits) as usize + 1;
        self.channels.len() * self.taps * len
    }

    pub fn forward_bits(&self, x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4> {
        let ys: Vec<_> = self
            .channels
            .iter()
            .enumerate()
            .map(|(index, conv)| {
                let xc = x.clone().slice(s![.., index..index + 1, .., ..]);
                conv.forward_bits(xc, bits)
            })
            .collect();
        cat_dim(ys, 1)
    }

    /// One depthwise LUT layer: `fan_in = channels * taps`, tables from each per-channel LUT-Conv.
    pub fn to_manifest_layer(
        &self,
        name: &str,
        in_bits: u8,
        out_bits: u8,
        in_scale: f32,
        leaky_slope: f32,
    ) -> Result<crate::quant::LutLayer> {
        if !(2..=8).contains(&in_bits) || !(2..=8).contains(&out_bits) {
            bail!("LUT truth tables support 2 to 8 bits, got in {in_bits} out {out_bits}");
        }
        let channels = self.channels.len();
        let taps = self.taps;
        let fan_in = channels * taps;
        let in_qmax = crate::quant::Bits::max_value(in_bits) as i32;
        let out_qmax = crate::quant::Bits::max_value(out_bits) as i32;
        let table_len = (2 * in_qmax + 1) as usize;
        let mut samples = Vec::with_capacity(fan_in * channels * table_len);
        let mut max_abs = 0.0f32;
        for oc in 0..channels {
            let weights = self.channels[oc].dense.host_weights()?;
            for tap in 0..taps {
                let fan = tap * channels + oc;
                for out_ch in 0..channels {
                    for code in -in_qmax..=in_qmax {
                        let value = if out_ch == oc {
                            weights.eval(tap, 0, code as f32 * in_scale)
                        } else {
                            0.0
                        };
                        max_abs = max_abs.max(value.abs());
                        samples.push(value);
                    }
                }
            }
        }
        let out_scale = if max_abs == 0.0 {
            1.0
        } else {
            max_abs / out_qmax as f32
        };
        let tables = samples
            .into_iter()
            .map(|value| round_clamp(value / out_scale, out_qmax))
            .collect::<Vec<_>>();
        let span = fan_in as f32 * out_qmax as f32 * out_scale;
        let next_in_scale = if span == 0.0 {
            1.0
        } else {
            span / in_qmax as f32
        };
        let activation = if leaky_slope == 0.0 {
            "identity"
        } else {
            "leaky_relu"
        };
        Ok(crate::quant::LutLayer {
            name: name.to_owned(),
            in_channels: channels,
            out_channels: channels,
            kernel: [3, 3],
            stride: [2, 2],
            reflect_pad: 1,
            fan_in,
            in_bits,
            out_bits,
            in_qmax,
            out_qmax,
            in_scale,
            out_scale,
            next_in_scale,
            reduce: "sum".to_owned(),
            activation: activation.to_owned(),
            leaky_slope: (leaky_slope != 0.0).then_some(leaky_slope),
            table_len,
            tables,
            post_relu: false,
        })
    }

    /// Same check as [LutConv2d::matches_table], for the merged depthwise layer.
    pub fn matches_table(&self, layer: &crate::quant::LutLayer, in_scale: f32) -> Result<()> {
        let channels = self.channels.len();
        let taps = self.taps;
        let in_qmax = layer.in_qmax;
        for oc in 0..channels {
            let weights = self.channels[oc].dense.host_weights()?;
            for tap in 0..taps {
                let fan = tap * channels + oc;
                for out_ch in 0..channels {
                    for code in -in_qmax..=in_qmax {
                        let value = if out_ch == oc {
                            weights.eval(tap, 0, code as f32 * in_scale)
                        } else {
                            0.0
                        };
                        let expected = round_clamp(value / layer.out_scale, layer.out_qmax);
                        let found = layer.code(fan, out_ch, code);
                        if found != expected {
                            bail!(
                                "{name} lut fan {fan} out {out_ch} code {code}: table {found}, mlp quantised to {expected}",
                                name = layer.name,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Encoder spatial path whose 3×3 depthwise layers are LUT-Conv.
///
/// The dense stem and both pointwise mixers stay outside this module, on the
/// separable student, so a checkpoint of that student still has those fields.
/// `half` is the stride-2 map at `base` channels. `trunk` is the next stride-2
/// map at `base * 2`.
#[derive(Module, Debug)]
pub struct LutDwEncoder<B: Backend> {
    pub half: LutDepthwise2d<B>,
    pub trunk: LutDepthwise2d<B>,
    #[module(skip)]
    bits: u8,
}

impl<B: Backend> LutDwEncoder<B> {
    pub fn new(base: usize, hidden: usize, bits: u8, device: &B::Device) -> Self {
        let wide = base * 2;
        Self {
            half: LutDepthwise2d::new(base, 3, 2, 1, hidden, device),
            trunk: LutDepthwise2d::new(wide, 3, 2, 1, hidden, device),
            bits,
        }
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn table_entries(&self) -> usize {
        self.half.table_entries(self.bits) + self.trunk.table_entries(self.bits)
    }

    /// Depthwise stage. The input is snapped onto the codebook the table will use.
    pub fn forward_half(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.half
            .forward_bits(quantize_codes(x, self.bits), self.bits)
    }

    pub fn forward_trunk(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.trunk
            .forward_bits(quantize_codes(x, self.bits), self.bits)
    }
}

trait ConvForward<B: Backend> {
    fn apply(conv: &LutConv2d<B>, x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4>;
}

impl<B: Backend> ConvForward<B> for LutConv2d<B> {
    default fn apply(conv: &LutConv2d<B>, x: Tensor<B, 4>, bits: u8) -> Tensor<B, 4> {
        conv.forward_bits_windows(x, bits)
    }
}

impl ConvForward<crate::backend::TrainBackend> for LutConv2d<crate::backend::TrainBackend> {
    fn apply(
        conv: &LutConv2d<crate::backend::TrainBackend>,
        x: Tensor<crate::backend::TrainBackend, 4>,
        bits: u8,
    ) -> Tensor<crate::backend::TrainBackend, 4> {
        let x = if conv.pad > 0 {
            x.pad(
                [(conv.pad, conv.pad), (conv.pad, conv.pad)],
                PadMode::Reflect,
            )
        } else {
            x
        };
        // Reflect padding stays in the autodiff graph so its backward
        // scatters into the mirrored pixels. The op itself sees pad 0.
        conv_forward_ad(conv, x, bits, 0)
    }
}

fn conv_windows<B: Backend>(
    x: Tensor<B, 4>,
    w_lin: Tensor<B, 2>,
    b_out: Tensor<B, 2>,
    w_hidden: Tensor<B, 3>,
    b_hidden: Tensor<B, 3>,
    w_out: Tensor<B, 3>,
    kernel: usize,
    stride: usize,
    pad: usize,
    bits: u8,
) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let x = if pad > 0 {
        x.pad([(pad, pad), (pad, pad)], PadMode::Reflect)
    } else {
        x
    };
    let [_, _, padded_h, padded_w] = x.dims();
    assert!(
        padded_h >= kernel && padded_w >= kernel,
        "im2col input {h}x{w} is smaller than the kernel"
    );
    let hout = (padded_h - kernel) / stride + 1;
    let wout = (padded_w - kernel) / stride + 1;
    let fan_in = c * kernel * kernel;
    let fan_out = w_lin.dims()[1];
    let mut bands = Vec::new();
    let mut oy = 0;
    while oy < hout {
        let rows = lut_window_rows(n, fan_in, fan_out, wout, hout - oy);
        let cols = im2col_window(&x, kernel, stride, oy, oy + rows, 0, wout);
        let y = dense_eager(
            cols,
            w_lin.clone(),
            b_out.clone(),
            w_hidden.clone(),
            b_hidden.clone(),
            w_out.clone(),
            bits,
        );
        let _ = y.clone().sum().into_scalar();
        let out_channels = y.dims()[2];
        bands.push(
            y.reshape([n, rows, wout, out_channels])
                .permute([0, 3, 1, 2]),
        );
        oy += rows;
    }
    if bands.len() == 1 {
        bands.pop().expect("band")
    } else {
        Tensor::cat(bands, 2)
    }
}

trait DenseForward<B: Backend> {
    fn apply(dense: &LutDense<B>, x: Tensor<B, 3>, bits: u8) -> Tensor<B, 3>;
}

impl<B: Backend> DenseForward<B> for LutDense<B> {
    default fn apply(dense: &LutDense<B>, x: Tensor<B, 3>, bits: u8) -> Tensor<B, 3> {
        dense.forward_eager(x, bits)
    }
}

impl DenseForward<crate::backend::TrainBackend> for LutDense<crate::backend::TrainBackend> {
    fn apply(
        dense: &LutDense<crate::backend::TrainBackend>,
        x: Tensor<crate::backend::TrainBackend, 3>,
        bits: u8,
    ) -> Tensor<crate::backend::TrainBackend, 3> {
        LUT_AD_FORWARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        dense_forward_ad(dense, x, bits)
    }
}

use std::sync::atomic::AtomicUsize;

static LUT_AD_FORWARDS: AtomicUsize = AtomicUsize::new(0);

fn dense_eager<B: Backend>(
    x: Tensor<B, 3>,
    w_lin: Tensor<B, 2>,
    b_out: Tensor<B, 2>,
    w_hidden: Tensor<B, 3>,
    b_hidden: Tensor<B, 3>,
    w_out: Tensor<B, 3>,
    bits: u8,
) -> Tensor<B, 3> {
    let [n, sites, n_in] = x.dims();
    let n_out = w_lin.dims()[1];
    let hidden = w_hidden.dims()[2];
    let mut acc = Tensor::zeros([n, sites, n_out], &x.device());
    let mut start = 0;
    let mut since_flush = 0usize;
    while start < n_in {
        // One input at a time makes a kernel launch per tap. A chunk of inputs
        // shares one `[N, S, K, C]` map, kept under the 4 MiB page so it does
        // not ask for the 122 MiB page that fails after the teacher forward.
        let k = lut_input_chunk(n, sites, n_out, n_in - start);
        let xi4 = x
            .clone()
            .slice(s![.., .., start..start + k])
            .unsqueeze_dim::<4>(3);
        let mut tap = xi4.clone()
            * w_lin
                .clone()
                .slice(s![start..start + k, ..])
                .reshape([1, 1, k, n_out]);
        for h in 0..hidden {
            let w_h = w_hidden
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            let b_h = b_hidden
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            let w_o = w_out
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            tap = tap + tanh(xi4.clone() * w_h + b_h) * w_o;
        }
        tap = tap
            + b_out
                .clone()
                .slice(s![start..start + k, ..])
                .reshape([1, 1, k, n_out]);
        acc = acc + quantize_codes(tap, bits).sum_dims_squeeze::<3, _>(&[2]);
        // Fusion keeps the whole loop until something reads it. Drain a full
        // page-sized chunk immediately, and drain unit chunks every few inputs.
        since_flush += k;
        if k > 1 || since_flush >= 8 || start + k == n_in {
            let _ = acc.clone().sum().into_scalar();
            since_flush = 0;
        }
        start += k;
    }
    acc
}

fn quantize_mask<B: Backend, const D: usize>(f: Tensor<B, D>, bits: u8) -> Tensor<B, D> {
    if bits < 2 {
        return Tensor::ones(f.shape(), &f.device());
    }
    let qmax = Bits::max_value(bits);
    let scale = LUT_CODE_SPAN / qmax;
    f.div_scalar(scale)
        .round()
        .abs()
        .lower_equal_elem(qmax)
        .float()
}

fn sum_batch_sites4<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 2> {
    x.sum_dims_squeeze::<2, _>(&[0, 1])
}

fn cat_dim<B: Backend, const D: usize>(mut parts: Vec<Tensor<B, D>>, dim: usize) -> Tensor<B, D> {
    if parts.len() == 1 {
        return parts.pop().expect("cat part");
    }
    Tensor::cat(parts, dim)
}

fn dense_backward_eager<B: Backend>(
    x: Tensor<B, 3>,
    w_lin: Tensor<B, 2>,
    b_out: Tensor<B, 2>,
    w_hidden: Tensor<B, 3>,
    b_hidden: Tensor<B, 3>,
    w_out: Tensor<B, 3>,
    dy: Tensor<B, 3>,
    bits: u8,
) -> (
    Tensor<B, 3>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
) {
    let [n, sites, n_in] = x.dims();
    let n_out = w_lin.dims()[1];
    let hidden = w_hidden.dims()[2];
    let mut dx_parts = Vec::new();
    let mut dw_lin_parts = Vec::new();
    let mut db_out_parts = Vec::new();
    let mut dw_h_parts = vec![Vec::new(); hidden];
    let mut db_h_parts = vec![Vec::new(); hidden];
    let mut dw_o_parts = vec![Vec::new(); hidden];
    let mut start = 0;
    let mut since_flush = 0usize;
    while start < n_in {
        let k = lut_input_chunk(n, sites, n_out, n_in - start);
        let xi4 = x
            .clone()
            .slice(s![.., .., start..start + k])
            .unsqueeze_dim::<4>(3);
        let w_lin_c = w_lin
            .clone()
            .slice(s![start..start + k, ..])
            .reshape([1, 1, k, n_out]);
        let mut tap = xi4.clone() * w_lin_c.clone();
        let mut acts = Vec::with_capacity(hidden);
        for h in 0..hidden {
            let w_h = w_hidden
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            let b_h = b_hidden
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            let w_o = w_out
                .clone()
                .slice(s![start..start + k, .., h..h + 1])
                .reshape([1, 1, k, n_out]);
            let ht = tanh(xi4.clone() * w_h.clone() + b_h);
            tap = tap + ht.clone() * w_o.clone();
            acts.push((w_h, w_o, ht));
        }
        tap = tap
            + b_out
                .clone()
                .slice(s![start..start + k, ..])
                .reshape([1, 1, k, n_out]);
        let dy_i = dy.clone().unsqueeze_dim::<4>(2) * quantize_mask(tap, bits);
        dw_lin_parts.push(sum_batch_sites4(dy_i.clone() * xi4.clone()));
        db_out_parts.push(sum_batch_sites4(dy_i.clone()));
        let mut dx_i = (dy_i.clone() * w_lin_c).sum_dims_squeeze::<3, _>(&[3]);
        for (h, (w_h, w_o, ht)) in acts.into_iter().enumerate() {
            let d_ht = dy_i.clone() * w_o;
            let d_u = d_ht * (ht.clone().powi_scalar(2).neg() + 1.0);
            dw_h_parts[h].push(sum_batch_sites4(d_u.clone() * xi4.clone()));
            db_h_parts[h].push(sum_batch_sites4(d_u.clone()));
            dw_o_parts[h].push(sum_batch_sites4(dy_i.clone() * ht));
            dx_i = dx_i + (d_u * w_h).sum_dims_squeeze::<3, _>(&[3]);
        }
        dx_parts.push(dx_i);
        since_flush += k;
        if k > 1 || since_flush >= 8 || start + k == n_in {
            let _ = dx_parts.last().unwrap().clone().sum().into_scalar();
            since_flush = 0;
        }
        start += k;
    }
    let stack = |parts: Vec<Vec<Tensor<B, 2>>>| {
        let planes: Vec<_> = parts
            .into_iter()
            .map(|rows| cat_dim(rows, 0).unsqueeze_dim::<3>(2))
            .collect();
        cat_dim(planes, 2)
    };
    (
        cat_dim(dx_parts, 2),
        cat_dim(dw_lin_parts, 0),
        cat_dim(db_out_parts, 0),
        stack(dw_h_parts),
        stack(db_h_parts),
        stack(dw_o_parts),
    )
}

#[derive(Clone, Debug)]
struct LutDenseState {
    bits: u8,
    x: burn_autodiff::NodeId,
    w_lin: burn_autodiff::NodeId,
    b_out: burn_autodiff::NodeId,
    w_hidden: burn_autodiff::NodeId,
    b_hidden: burn_autodiff::NodeId,
    w_out: burn_autodiff::NodeId,
}

fn dense_forward_ad<B: Backend, C: CheckpointStrategy>(
    dense: &LutDense<Autodiff<B, C>>,
    x: Tensor<Autodiff<B, C>, 3>,
    bits: u8,
) -> Tensor<Autodiff<B, C>, 3> {
    #[derive(Debug)]
    struct LutDenseOp;

    impl<B: Backend> Backward<B, 6> for LutDenseOp {
        type State = LutDenseState;

        fn backward(
            self,
            ops: Ops<Self::State, 6>,
            grads: &mut burn_autodiff::grads::Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let state = ops.state;
            let dy = grads.consume::<B>(&ops.node);
            let wrap3 = |prim| Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(prim));
            let wrap2 = |prim| Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(prim));
            let (dx, dw_lin, db_out, dw_h, db_h, dw_o) = dense_backward_eager(
                wrap3(checkpointer.retrieve_node_output(state.x)),
                wrap2(checkpointer.retrieve_node_output(state.w_lin)),
                wrap2(checkpointer.retrieve_node_output(state.b_out)),
                wrap3(checkpointer.retrieve_node_output(state.w_hidden)),
                wrap3(checkpointer.retrieve_node_output(state.b_hidden)),
                wrap3(checkpointer.retrieve_node_output(state.w_out)),
                wrap3(dy),
                state.bits,
            );
            let packed = [
                dx.into_primitive().tensor(),
                dw_lin.into_primitive().tensor(),
                db_out.into_primitive().tensor(),
                dw_h.into_primitive().tensor(),
                db_h.into_primitive().tensor(),
                dw_o.into_primitive().tensor(),
            ];
            for (parent, grad) in ops.parents.into_iter().zip(packed) {
                if let Some(node) = parent {
                    grads.register::<B>(node.id, grad);
                }
            }
        }
    }

    let x_ad = x.into_primitive().tensor();
    let w_lin_ad = dense.w_lin.val().into_primitive().tensor();
    let b_out_ad = dense.b_out.val().into_primitive().tensor();
    let w_hidden_ad = dense.w_hidden.val().into_primitive().tensor();
    let b_hidden_ad = dense.b_hidden.val().into_primitive().tensor();
    let w_out_ad = dense.w_out.val().into_primitive().tensor();
    let y = dense_eager(
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(x_ad.primitive.clone())),
        Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(w_lin_ad.primitive.clone())),
        Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(b_out_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(w_hidden_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(b_hidden_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(w_out_ad.primitive.clone())),
        bits,
    );
    let y_prim = y.into_primitive().tensor();

    let prepared = LutDenseOp
        .prepare::<C>([
            x_ad.node.clone(),
            w_lin_ad.node.clone(),
            b_out_ad.node.clone(),
            w_hidden_ad.node.clone(),
            b_hidden_ad.node.clone(),
            w_out_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();
    let ad = match prepared {
        OpsKind::Tracked(mut prep) => {
            let state = LutDenseState {
                bits,
                x: prep.checkpoint(&x_ad),
                w_lin: prep.checkpoint(&w_lin_ad),
                b_out: prep.checkpoint(&b_out_ad),
                w_hidden: prep.checkpoint(&w_hidden_ad),
                b_hidden: prep.checkpoint(&b_hidden_ad),
                w_out: prep.checkpoint(&w_out_ad),
            };
            prep.finish(state, y_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(y_prim),
    };
    Tensor::from_primitive(TensorPrimitive::Float(ad))
}

fn col2im_add<B: Backend>(
    mut acc: Tensor<B, 4>,
    cols: Tensor<B, 3>,
    channels: usize,
    kernel: usize,
    stride: usize,
    oy0: usize,
    rows: usize,
    wout: usize,
) -> Tensor<B, 4> {
    let n = acc.dims()[0];
    // Channels of one kernel tap land on distinct planes, so they are one
    // strided write. A sync per channel turned the H/2 backward into tens of
    // thousands of readbacks. Taps still overlap each other, so each tap is
    // read back before the next write.
    for ky in 0..kernel {
        for kx in 0..kernel {
            let base = (ky * kernel + kx) * channels;
            let patch = cols
                .clone()
                .slice(s![.., .., base..base + channels])
                .reshape([n, rows, wout, channels])
                .permute([0, 3, 1, 2]);
            let y0 = oy0 * stride + ky;
            let x0 = kx;
            let y1 = y0 + stride * rows;
            let x1 = x0 + stride * wout;
            let cur = acc.clone().slice(s![.., .., y0..y1;stride, x0..x1;stride]);
            acc = acc.slice_assign(s![.., .., y0..y1;stride, x0..x1;stride], cur + patch);
            let _ = acc.clone().sum().into_scalar();
        }
    }
    acc
}

fn conv_backward_eager<B: Backend>(
    x: Tensor<B, 4>,
    w_lin: Tensor<B, 2>,
    b_out: Tensor<B, 2>,
    w_hidden: Tensor<B, 3>,
    b_hidden: Tensor<B, 3>,
    w_out: Tensor<B, 3>,
    dy: Tensor<B, 4>,
    kernel: usize,
    stride: usize,
    pad: usize,
    bits: u8,
) -> (
    Tensor<B, 4>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
) {
    let [_, _, h, w] = x.dims();
    let x = if pad > 0 {
        x.pad([(pad, pad), (pad, pad)], PadMode::Reflect)
    } else {
        x
    };
    let [n, c, _, _] = x.dims();
    let fan_out = w_lin.dims()[1];
    let wout = dy.dims()[3];
    let hout = dy.dims()[2];
    let mut dx = Tensor::<B, 4>::zeros(x.shape(), &x.device());
    let mut dw_lin = Tensor::<B, 2>::zeros(w_lin.shape(), &w_lin.device());
    let mut db_out = Tensor::<B, 2>::zeros(b_out.shape(), &b_out.device());
    let mut dw_h = Tensor::<B, 3>::zeros(w_hidden.shape(), &w_hidden.device());
    let mut db_h = Tensor::<B, 3>::zeros(b_hidden.shape(), &b_hidden.device());
    let mut dw_o = Tensor::<B, 3>::zeros(w_out.shape(), &w_out.device());
    let mut oy = 0;
    while oy < hout {
        let rows = lut_window_rows(n, c * kernel * kernel, fan_out, wout, hout - oy);
        let cols = im2col_window(&x, kernel, stride, oy, oy + rows, 0, wout);
        let dy_win = dy
            .clone()
            .slice(s![.., .., oy..oy + rows, ..])
            .permute([0, 2, 3, 1])
            .reshape([n, rows * wout, fan_out]);
        let (dx_cols, dwl, dbo, dwh, dbh, dwo) = dense_backward_eager(
            cols,
            w_lin.clone(),
            b_out.clone(),
            w_hidden.clone(),
            b_hidden.clone(),
            w_out.clone(),
            dy_win,
            bits,
        );
        dx = col2im_add(dx, dx_cols, c, kernel, stride, oy, rows, wout);
        dw_lin = dw_lin + dwl;
        db_out = db_out + dbo;
        dw_h = dw_h + dwh;
        db_h = db_h + dbh;
        dw_o = dw_o + dwo;
        let _ = dx.clone().sum().into_scalar();
        oy += rows;
    }
    if pad > 0 {
        dx = dx.slice(s![.., .., pad..pad + h, pad..pad + w]);
    }
    (dx, dw_lin, db_out, dw_h, db_h, dw_o)
}

#[derive(Clone, Debug)]
struct LutConvState {
    bits: u8,
    kernel: usize,
    stride: usize,
    pad: usize,
    x: burn_autodiff::NodeId,
    w_lin: burn_autodiff::NodeId,
    b_out: burn_autodiff::NodeId,
    w_hidden: burn_autodiff::NodeId,
    b_hidden: burn_autodiff::NodeId,
    w_out: burn_autodiff::NodeId,
}

fn conv_forward_ad<B: Backend, C: CheckpointStrategy>(
    conv: &LutConv2d<Autodiff<B, C>>,
    x: Tensor<Autodiff<B, C>, 4>,
    bits: u8,
    pad: usize,
) -> Tensor<Autodiff<B, C>, 4> {
    #[derive(Debug)]
    struct LutConvOp;

    impl<B: Backend> Backward<B, 6> for LutConvOp {
        type State = LutConvState;

        fn backward(
            self,
            ops: Ops<Self::State, 6>,
            grads: &mut burn_autodiff::grads::Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let state = ops.state;
            let dy = grads.consume::<B>(&ops.node);
            let wrap4 = |prim| Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(prim));
            let wrap3 = |prim| Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(prim));
            let wrap2 = |prim| Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(prim));
            let (dx, dw_lin, db_out, dw_h, db_h, dw_o) = conv_backward_eager(
                wrap4(checkpointer.retrieve_node_output(state.x)),
                wrap2(checkpointer.retrieve_node_output(state.w_lin)),
                wrap2(checkpointer.retrieve_node_output(state.b_out)),
                wrap3(checkpointer.retrieve_node_output(state.w_hidden)),
                wrap3(checkpointer.retrieve_node_output(state.b_hidden)),
                wrap3(checkpointer.retrieve_node_output(state.w_out)),
                wrap4(dy),
                state.kernel,
                state.stride,
                state.pad,
                state.bits,
            );
            let packed = [
                dx.into_primitive().tensor(),
                dw_lin.into_primitive().tensor(),
                db_out.into_primitive().tensor(),
                dw_h.into_primitive().tensor(),
                db_h.into_primitive().tensor(),
                dw_o.into_primitive().tensor(),
            ];
            for (parent, grad) in ops.parents.into_iter().zip(packed) {
                if let Some(node) = parent {
                    grads.register::<B>(node.id, grad);
                }
            }
        }
    }

    let x_ad = x.into_primitive().tensor();
    let w_lin_ad = conv.dense.w_lin.val().into_primitive().tensor();
    let b_out_ad = conv.dense.b_out.val().into_primitive().tensor();
    let w_hidden_ad = conv.dense.w_hidden.val().into_primitive().tensor();
    let b_hidden_ad = conv.dense.b_hidden.val().into_primitive().tensor();
    let w_out_ad = conv.dense.w_out.val().into_primitive().tensor();
    let y = conv_windows(
        Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(x_ad.primitive.clone())),
        Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(w_lin_ad.primitive.clone())),
        Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(b_out_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(w_hidden_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(b_hidden_ad.primitive.clone())),
        Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(w_out_ad.primitive.clone())),
        conv.kernel,
        conv.stride,
        pad,
        bits,
    );
    let y_prim = y.into_primitive().tensor();
    let prepared = LutConvOp
        .prepare::<C>([
            x_ad.node.clone(),
            w_lin_ad.node.clone(),
            b_out_ad.node.clone(),
            w_hidden_ad.node.clone(),
            b_hidden_ad.node.clone(),
            w_out_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();
    let ad = match prepared {
        OpsKind::Tracked(mut prep) => {
            let state = LutConvState {
                bits,
                kernel: conv.kernel,
                stride: conv.stride,
                pad,
                x: prep.checkpoint(&x_ad),
                w_lin: prep.checkpoint(&w_lin_ad),
                b_out: prep.checkpoint(&b_out_ad),
                w_hidden: prep.checkpoint(&w_hidden_ad),
                b_hidden: prep.checkpoint(&b_hidden_ad),
                w_out: prep.checkpoint(&w_out_ad),
            };
            prep.finish(state, y_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(y_prim),
    };
    Tensor::from_primitive(TensorPrimitive::Float(ad))
}

fn leaky_relu<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let positive = x.clone().clamp(0.0, 1.0e4);
    let negative = x.clamp(-1.0e4, 0.0);
    positive + negative * LUT_LEAKY_SLOPE
}

/// How many logical inputs share one `[N, S, K, C]` map.
///
/// That map has to stay under 4 MiB. Above that, CubeCL asks for a 122 MiB
/// page, and a fresh page of that size does not allocate after the teacher
/// forward on this T10.
fn lut_input_chunk(batch: usize, sites: usize, n_out: usize, n_in_left: usize) -> usize {
    let denom = batch
        .saturating_mul(sites.max(1))
        .saturating_mul(n_out.max(1))
        .saturating_mul(4)
        .max(1);
    const CAP: usize = 3_800_000;
    (CAP / denom).clamp(1, n_in_left.max(1))
}

/// How many output rows fit in CubeCL's smallest page on this T10.
///
/// The runtime's pages are about 30 MiB, 122 MiB and 487 MiB. A buffer above
/// 4 MiB is placed on the 122 MiB page, and one above 32 MiB on the 487 MiB
/// page. After the teacher forward those two pages do not allocate, even
/// though the driver still reports free memory. Staying under 4 MiB uses the
/// 30 MiB page.
fn lut_window_rows(
    batch: usize,
    fan_in: usize,
    fan_out: usize,
    width: usize,
    rows_left: usize,
) -> usize {
    let width = width.max(1);
    let per_row = batch
        .saturating_mul(fan_in.max(fan_out))
        .saturating_mul(width)
        .saturating_mul(4)
        .max(1);
    const CAP: usize = 3_800_000;
    (CAP / per_row).clamp(1, rows_left.max(1))
}

/// `[N, C, H, W]` (already padded) → columns for one output window, `[N, sites, C*K*K]`.
///
/// Tap order matches the integer interpreter: `(ky * kw + kx) * in_channels + ic`.
fn im2col_window<B: Backend>(
    x: &Tensor<B, 4>,
    kernel: usize,
    stride: usize,
    oy0: usize,
    oy1: usize,
    ox0: usize,
    ox1: usize,
) -> Tensor<B, 3> {
    let [n, c, h, w] = x.dims();
    let hout = oy1 - oy0;
    let wout = ox1 - ox0;
    let mut taps = Vec::with_capacity(kernel * kernel);
    for ky in 0..kernel {
        for kx in 0..kernel {
            let y0 = oy0 * stride + ky;
            let x0 = ox0 * stride + kx;
            let y1 = (y0 + stride * hout).min(h);
            let x1 = (x0 + stride * wout).min(w);
            let patch = x.clone().slice(s![.., .., y0..y1;stride, x0..x1;stride]);
            let [pn, pc, ph, pw] = patch.dims();
            assert_eq!(
                [pn, pc, ph, pw],
                [n, c, hout, wout],
                "im2col tap {ky},{kx} produced {ph}x{pw}, expected {hout}x{wout}"
            );
            taps.push(patch);
        }
    }
    let cols = Tensor::cat(taps, 1);
    cols.permute([0, 2, 3, 1])
        .reshape([n, hout * wout, c * kernel * kernel])
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn centered_gray_is_code_zero_and_the_rails_have_opposite_sign() {
        let device = Default::default();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(vec![0.0, 0.5, 1.0], [1, 1, 1, 3]),
            &device,
        );
        let y = quantize_rgb(x, 8).into_data().to_vec::<f32>().unwrap();
        assert!(y[0] < -0.5, "black landed at {}", y[0]);
        assert!(y[1].abs() < 1e-5, "mid-gray landed at {}", y[1]);
        assert!(y[2] > 0.5, "white landed at {}", y[2]);
        assert!(y.iter().all(|v| v.abs() <= LUT_CODE_SPAN + 1e-4));
    }

    #[test]
    fn window_rows_stay_in_the_small_cuda_page() {
        // batch 32, crop 192: stem 192 wide, H/2 96, H/4 48.
        let stages = [(27, 32, 192), (288, 64, 96), (576, 64, 48)];
        for (fan_in, fan_out, width) in stages {
            let mut left = width;
            while left > 0 {
                let rows = lut_window_rows(32, fan_in, fan_out, width, left);
                let bytes = 32 * rows * width * fan_in.max(fan_out) * 4;
                assert!(
                    bytes <= 3_800_000,
                    "stage fan_in {fan_in} window {rows}x{width} is {bytes} bytes"
                );
                left -= rows;
            }
        }
    }

    #[test]
    fn depthwise_b4_tables_fit_one_logic_cell_per_tap() {
        let device = Default::default();
        let encoder = LutDwEncoder::<NdArray>::new(32, 4, 4, &device);
        // 4-bit codebook is 15 codes. 96 channels times 9 taps.
        assert_eq!(encoder.table_entries(), 96 * 9 * 15);
        assert!(encoder.table_entries() < 20_000);
    }

    #[test]
    fn input_chunks_stay_in_the_small_cuda_page() {
        // H/2, one output row: 4 inputs fill the page. The stem band is
        // already at the cap, so it stays one input at a time.
        let half = lut_input_chunk(32, 96, 64, 288);
        assert_eq!(half, 4);
        assert!(32 * 96 * half * 64 * 4 <= 3_800_000);
        assert_eq!(lut_input_chunk(32, 4 * 192, 32, 27), 1);
        let quarter = lut_input_chunk(32, 48, 64, 576);
        assert!(32 * 48 * quarter * 64 * 4 <= 3_800_000);
        assert!(quarter > 1);
    }

    #[test]
    fn autodiff_lut_matches_the_eager_graph() {
        type AD = crate::backend::TrainBackend;
        let device = Default::default();
        let dense = LutDense::<AD>::new(2, 1, 2, &device);
        let x = Tensor::<AD, 3>::from_floats([[[0.2, -0.4], [0.7, 0.1]]], &device).require_grad();
        let before = LUT_AD_FORWARDS.load(std::sync::atomic::Ordering::Relaxed);
        let y_op = dense.forward(x.clone(), 0);
        assert!(LUT_AD_FORWARDS.load(std::sync::atomic::Ordering::Relaxed) > before);
        let y_graph = dense_eager(
            x.clone(),
            dense.w_lin.val(),
            dense.b_out.val(),
            dense.w_hidden.val(),
            dense.b_hidden.val(),
            dense.w_out.val(),
            0,
        );
        let op = y_op.clone().into_data().to_vec::<f32>().unwrap();
        let graph = y_graph.clone().into_data().to_vec::<f32>().unwrap();
        for (a, b) in op.iter().zip(&graph) {
            assert!((a - b).abs() < 1e-5, "forward {a} vs {b}");
        }
        let g_op = y_op.sum().backward();
        let g_graph = y_graph.sum().backward();
        let w_op = dense
            .w_lin
            .grad(&g_op)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let w_graph = dense
            .w_lin
            .grad(&g_graph)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in w_op.iter().zip(&w_graph) {
            assert!((a - b).abs() < 1e-4, "w_lin grad {a} vs {b}");
        }
    }

    #[test]
    fn autodiff_conv_matches_the_eager_graph() {
        type AD = crate::backend::TrainBackend;
        let device = Default::default();
        let conv = LutConv2d::<AD>::new(2, 2, 3, 1, 1, 1, &device);
        let x = Tensor::<AD, 4>::ones([1, 2, 6, 6], &device).require_grad();
        let y_op = conv.forward_bits(x.clone(), 0);
        let y_graph = super::conv_windows(
            x.clone(),
            conv.dense.w_lin.val(),
            conv.dense.b_out.val(),
            conv.dense.w_hidden.val(),
            conv.dense.b_hidden.val(),
            conv.dense.w_out.val(),
            conv.kernel,
            conv.stride,
            conv.pad,
            0,
        );
        let op = y_op.clone().into_data().to_vec::<f32>().unwrap();
        let graph = y_graph.clone().into_data().to_vec::<f32>().unwrap();
        for (a, b) in op.iter().zip(&graph) {
            assert!((a - b).abs() < 1e-4, "conv forward {a} vs {b}");
        }
        let g_op = y_op.sum().backward();
        let g_graph = y_graph.sum().backward();
        let x_op = x.grad(&g_op).unwrap().into_data().to_vec::<f32>().unwrap();
        let x_graph = x
            .grad(&g_graph)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, (a, b)) in x_op.iter().zip(&x_graph).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "input grad[{i}] {a} vs {b}\nop {x_op:?}\ngraph {x_graph:?}"
            );
        }
    }
}
