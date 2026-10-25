//! Turn a trained student into the integer artefacts an accelerator consumes.
//!
//! Each model implements [Exportable]: its [Exportable::export_plan] is a literal transcription
//! of [crate::student::tiny::TinyStudent::forward_core] / the DW equivalent, one step per
//! operation. [export_int8] walks that plan once into the graph (what the hardware executes)
//! and once into the layer table (the constants each convolution needs).
//!
//! Everything a fixed-point accelerator needs is in manifest.json plus weights.bin;
//! [crate::intref] interprets exactly those two files.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::prelude::*;

use crate::backend::InferBackend;
use crate::quant::{
    Bits, Calibration, QuantLayer, QuantManifest, QuantOp, Quantizable, quantise_weights_host,
};
use crate::quant::{FilmHead, FilmLayer, InstanceNormLayer};
use crate::student::TinyDwStudent;
use crate::student::TinyStudent;
use crate::student::lut::{LUT_LEAKY_SLOPE, LutDepthwise2d, LutEncoder};
use crate::student::plugin::{Film, PluginKind, PluginSite};
use crate::student::tiny::Conv;
use crate::student::tiny_dw::{DwConv, PwConv};

/// The scale the final uint8 image is stored on. The head int32 accumulator is rescaled onto
/// this grid and clamped to [0, 255], so the last stage of the pipeline needs no extra logic:
/// the image byte *is* the activation code.
pub const OUTPUT_SCALE: f32 = 1.0 / 255.0;

/// A [Quantizable] model whose fake-quant graph can be lowered to manifest + weights.
pub trait Exportable: Quantizable<InferBackend> {
    /// The exact operation sequence of the quantised forward pass, with scales filled in.
    fn export_plan(&self, scales: &[f32], point_count: usize) -> Vec<ExportStep<'_>>;

    /// Extra lines appended to manifest.json `notes`.
    fn export_notes(&self, bits: &Bits) -> Vec<String> {
        default_notes(bits)
    }

    /// Why this graph cannot be lowered yet. `None` means the plan matches the forward.
    fn export_blocker(&self) -> Option<String> {
        None
    }
}

/// One convolution kernel referenced by an export step.
pub enum ConvRef<'a> {
    Dense(&'a Conv<InferBackend>),
    Depthwise(&'a DwConv<InferBackend>),
    Pointwise(&'a PwConv<InferBackend>),
}

/// One operation of the forward pass, in execution order.
pub enum ExportStep<'a> {
    Conv {
        name: String,
        conv: ConvRef<'a>,
        in_point: usize,
        out_point: usize,
        relu: bool,
        out_scale: f32,
        /// True only for the head: its codes are the unsigned image bytes.
        out_unsigned: bool,
    },
    /// out = requant(trunk) + requant(branch). `source_point` is requantised onto `point`.
    ResidualAdd {
        point: usize,
        source_point: usize,
        scale: f32,
    },
    /// Nearest-neighbour 2x upsampling: a wire on hardware, no arithmetic, no scale change.
    UpsampleNearest2x { point: usize },
    /// Emit the image: dequantise the head codes with this scale (they are unsigned bytes).
    OutputImage { scale: f32 },
    /// Depthwise LUT-Conv (truth table per tap); MLP weights are not stored.
    LutDw {
        name: String,
        lut: &'a LutDepthwise2d<InferBackend>,
        in_point: usize,
        out_point: usize,
        in_bits: u8,
        out_bits: u8,
        out_scale: f32,
        post_relu: bool,
    },
    /// Spatial InstanceNorm + γ, β on one quantised point (in-place).
    InstanceNorm {
        name: String,
        point: usize,
        channels: usize,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        out_scale: f32,
    },
    /// Global-pool FiLM replacing a same-width 1×1 colour slot.
    Film {
        name: String,
        film: &'a Film<InferBackend>,
        channels: usize,
        in_point: usize,
        out_point: usize,
        out_scale: f32,
    },
    /// Clamp one activation point to `[0, act_qmax]` (decoder ReLU after LUT).
    ReluPoint { point: usize },
}

impl ConvRef<'_> {
    fn weight_codes(&self, qmax: f32) -> Result<(Vec<i8>, Vec<f32>, Vec<f32>)> {
        let (weight, out_ch, bias) = match self {
            Self::Dense(c) => (
                tensor_to_vec(&c.weight())?,
                c.weight().dims()[0],
                read_bias(c.conv.bias.as_ref())?,
            ),
            Self::Depthwise(c) => (
                tensor_to_vec(&c.weight())?,
                c.weight().dims()[0],
                read_bias(c.conv.bias.as_ref())?,
            ),
            Self::Pointwise(c) => (
                tensor_to_vec(&c.weight())?,
                c.weight().dims()[0],
                read_bias(c.conv.bias.as_ref())?,
            ),
        };
        let (codes, scales) = quantise_weights_host(&weight, out_ch, qmax);
        Ok((codes, scales, bias))
    }

    fn layer_shape(&self) -> ([usize; 2], [usize; 2], usize, usize, usize) {
        match self {
            Self::Dense(c) => {
                let [out_ch, in_ch, kh, kw] = c.weight().dims();
                (c.conv.stride, [kh, kw], c.reflect_pad(), in_ch, out_ch)
            }
            Self::Depthwise(c) => {
                let [out_ch, _, kh, kw] = c.weight().dims();
                (c.conv.stride, [kh, kw], c.reflect_pad(), out_ch, out_ch)
            }
            Self::Pointwise(c) => {
                let [out_ch, in_ch, kh, kw] = c.weight().dims();
                (c.conv.stride, [kh, kw], 0, in_ch, out_ch)
            }
        }
    }

    fn groups(&self) -> usize {
        match self {
            Self::Dense(c) => c.conv.groups,
            Self::Depthwise(c) => c.conv.groups,
            Self::Pointwise(c) => c.conv.groups,
        }
    }

    fn op_name(&self, relu: bool) -> String {
        let base = match self {
            Self::Depthwise(_) => "depthwise_conv2d",
            Self::Dense(_) | Self::Pointwise(_) => "conv2d",
        };
        if relu {
            format!("{base}+relu")
        } else {
            base.to_owned()
        }
    }
}

fn tensor_to_vec(tensor: &Tensor<InferBackend, 4>) -> Result<Vec<f32>> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("cannot read a weight tensor: {e}"))
}

fn read_bias(bias: Option<&burn::module::Param<Tensor<InferBackend, 1>>>) -> Result<Vec<f32>> {
    match bias {
        Some(bias) => bias
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("cannot read a bias tensor: {e}")),
        None => Ok(Vec::new()),
    }
}

fn default_notes(bits: &Bits) -> Vec<String> {
    vec![
        format!(
            "activations: symmetric int{}, one scale per tensor, zero point 0",
            bits.acts
        ),
        format!(
            "weights: symmetric int{}, one scale per output channel, codes in [0-{}..{}] as two's complement",
            bits.weights,
            bits.weight_max() as i32,
            bits.weight_max() as i32
        ),
        "bias_int32[c]: added to the accumulator before requantisation".to_owned(),
        "requant[c]: multiply the int32 accumulator by this, round to nearest, then clamp: [-act_qmax, act_qmax] normally, [0, output_qmax] for the head".to_owned(),
        "reflect_pad: reflection padding applied before a valid convolution".to_owned(),
        "upsample_nearest2x: a wire, no arithmetic, no scale change".to_owned(),
        "add: both operands must be on the output point grid; the second convolution is, already, and source_point is requantised by point_scales[source_point] / point_scales[point]"
            .to_owned(),
        "quantize_u8_output: the image byte is the head unsigned code, scale 1/255".to_owned(),
    ]
}

impl Exportable for TinyStudent<InferBackend> {
    fn export_plan(&self, scales: &[f32], point_count: usize) -> Vec<ExportStep<'_>> {
        let base = 3 + 2 * self.res.len();
        let mut plan = Vec::new();

        plan.push(ExportStep::Conv {
            name: "enc.0".into(),
            conv: ConvRef::Dense(&self.enc[0]),
            in_point: 0,
            out_point: 1,
            relu: true,
            out_scale: scales[1],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "enc.1".into(),
            conv: ConvRef::Dense(&self.enc[1]),
            in_point: 1,
            out_point: 2,
            relu: true,
            out_scale: scales[2],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "enc.2".into(),
            conv: ConvRef::Dense(&self.enc[2]),
            in_point: 2,
            out_point: 3,
            relu: true,
            out_scale: scales[3],
            out_unsigned: false,
        });

        for (r, block) in self.res.iter().enumerate() {
            let point = 3 + 2 * r;
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.c1"),
                conv: ConvRef::Dense(&block.c1),
                in_point: point,
                out_point: point + 1,
                relu: true,
                out_scale: scales[point + 1],
                out_unsigned: false,
            });
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.c2"),
                conv: ConvRef::Dense(&block.c2),
                in_point: point + 1,
                out_point: point + 2,
                relu: false,
                out_scale: scales[point + 2],
                out_unsigned: false,
            });
            plan.push(ExportStep::ResidualAdd {
                point: point + 2,
                source_point: point,
                scale: scales[point + 2],
            });
        }

        plan.push(ExportStep::UpsampleNearest2x { point: base });
        plan.push(ExportStep::Conv {
            name: "up3.0".into(),
            conv: ConvRef::Dense(&self.up3[0]),
            in_point: base,
            out_point: base + 1,
            relu: true,
            out_scale: scales[base + 1],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "up1.0".into(),
            conv: ConvRef::Dense(&self.up1[0]),
            in_point: base + 1,
            out_point: base + 2,
            relu: true,
            out_scale: scales[base + 2],
            out_unsigned: false,
        });

        plan.push(ExportStep::UpsampleNearest2x { point: base + 2 });
        plan.push(ExportStep::Conv {
            name: "up3.1".into(),
            conv: ConvRef::Dense(&self.up3[1]),
            in_point: base + 2,
            out_point: base + 3,
            relu: true,
            out_scale: scales[base + 3],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "up1.1".into(),
            conv: ConvRef::Dense(&self.up1[1]),
            in_point: base + 3,
            out_point: base + 4,
            relu: true,
            out_scale: scales[base + 4],
            out_unsigned: false,
        });

        plan.push(ExportStep::Conv {
            name: "head".into(),
            conv: ConvRef::Dense(&self.head),
            in_point: base + 4,
            out_point: point_count,
            relu: false,
            out_scale: OUTPUT_SCALE,
            out_unsigned: true,
        });
        plan.push(ExportStep::OutputImage {
            scale: OUTPUT_SCALE,
        });
        plan
    }

    fn export_notes(&self, bits: &Bits) -> Vec<String> {
        let mut notes = default_notes(bits);
        notes.insert(
            3,
            format!(
                "accumulator: int32; the widest layer here is 576 products of {} * {}",
                bits.act_max() as i32,
                bits.weight_max() as i32
            ),
        );
        notes
    }
}

impl Exportable for TinyDwStudent<InferBackend> {
    fn export_blocker(&self) -> Option<String> {
        self.blocks_integer_export()
    }

    fn export_plan(&self, scales: &[f32], point_count: usize) -> Vec<ExportStep<'_>> {
        let mut plan = Vec::new();
        let mut point = 1usize;
        let lut_dw = self.export_lut_dw();
        let lut_bits = lut_dw.map(|lut| lut.bits()).unwrap_or(8);

        plan.push(ExportStep::Conv {
            name: "stem".into(),
            conv: ConvRef::Dense(&self.stem),
            in_point: 0,
            out_point: point,
            relu: true,
            out_scale: scales[point],
            out_unsigned: false,
        });

        let mut half_skip_point = 0usize;
        for (i, enc) in self.enc.iter().enumerate() {
            let dw_out = point + 1;
            if let Some(lut) = lut_dw {
                let spatial = if i == 0 { &lut.half } else { &lut.trunk };
                plan.push(ExportStep::LutDw {
                    name: format!("enc.{i}.dw"),
                    lut: spatial,
                    in_point: point,
                    out_point: dw_out,
                    in_bits: lut_bits,
                    out_bits: lut_bits,
                    out_scale: scales[dw_out],
                    post_relu: true,
                });
            } else {
                plan.push(ExportStep::Conv {
                    name: format!("enc.{i}.dw"),
                    conv: ConvRef::Depthwise(&enc.dw),
                    in_point: point,
                    out_point: dw_out,
                    relu: true,
                    out_scale: scales[dw_out],
                    out_unsigned: false,
                });
            }
            let pw_out = point + 2;
            plan.push(ExportStep::Conv {
                name: format!("enc.{i}.pw"),
                conv: ConvRef::Pointwise(&enc.pw),
                in_point: dw_out,
                out_point: pw_out,
                relu: true,
                out_scale: scales[pw_out],
                out_unsigned: false,
            });
            if i == 0 {
                half_skip_point = pw_out;
            }
            point = pw_out;
        }

        let trunk_base = point;
        for (r, block) in self.res.iter().enumerate() {
            let p = trunk_base + 4 * r;
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.s1.dw"),
                conv: ConvRef::Depthwise(&block.s1.dw),
                in_point: p,
                out_point: p + 1,
                relu: true,
                out_scale: scales[p + 1],
                out_unsigned: false,
            });
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.s1.pw"),
                conv: ConvRef::Pointwise(&block.s1.pw),
                in_point: p + 1,
                out_point: p + 2,
                relu: true,
                out_scale: scales[p + 2],
                out_unsigned: false,
            });
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.s2.dw"),
                conv: ConvRef::Depthwise(&block.s2.dw),
                in_point: p + 2,
                out_point: p + 3,
                relu: true,
                out_scale: scales[p + 3],
                out_unsigned: false,
            });
            plan.push(ExportStep::Conv {
                name: format!("res.{r}.s2.pw"),
                conv: ConvRef::Pointwise(&block.s2.pw),
                in_point: p + 3,
                out_point: p + 4,
                relu: false,
                out_scale: scales[p + 4],
                out_unsigned: false,
            });
            plan.push(ExportStep::ResidualAdd {
                point: p + 4,
                source_point: p,
                scale: scales[p + 4],
            });
        }

        let dec_base = trunk_base + 4 * self.res.len();
        if self.export_plugin_site() == PluginSite::H4
            && self.export_plugin().kind() == PluginKind::In
        {
            let (gamma, beta) = self
                .export_plugin()
                .host_instance_norm()
                .expect("plugin=in")
                .expect("instance norm γ/β");
            plan.push(ExportStep::InstanceNorm {
                name: "plugin.in_h4".into(),
                point: dec_base,
                channels: gamma.len(),
                gamma,
                beta,
                out_scale: scales[dec_base],
            });
        }
        plan.push(ExportStep::UpsampleNearest2x { point: dec_base });
        plan.push(ExportStep::ResidualAdd {
            point: dec_base,
            source_point: half_skip_point,
            scale: scales[dec_base],
        });
        plan.push(ExportStep::Conv {
            name: "up3.0.dw".into(),
            conv: ConvRef::Depthwise(&self.up3[0].dw),
            in_point: dec_base,
            out_point: dec_base + 1,
            relu: true,
            out_scale: scales[dec_base + 1],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "up3.0.pw".into(),
            conv: ConvRef::Pointwise(&self.up3[0].pw),
            in_point: dec_base + 1,
            out_point: dec_base + 2,
            relu: true,
            out_scale: scales[dec_base + 2],
            out_unsigned: false,
        });
        if let Some(film) = self.export_film_half() {
            let channels = film.host_heads().0.weight().dims()[1];
            plan.push(ExportStep::Film {
                name: "film.half".into(),
                film,
                channels,
                in_point: dec_base + 2,
                out_point: dec_base + 3,
                out_scale: scales[dec_base + 3],
            });
        } else {
            plan.push(ExportStep::Conv {
                name: "up1.0".into(),
                conv: ConvRef::Pointwise(&self.up1[0]),
                in_point: dec_base + 2,
                out_point: dec_base + 3,
                relu: true,
                out_scale: scales[dec_base + 3],
                out_unsigned: false,
            });
        }

        plan.push(ExportStep::UpsampleNearest2x {
            point: dec_base + 3,
        });
        plan.push(ExportStep::Conv {
            name: "up3.1.dw".into(),
            conv: ConvRef::Depthwise(&self.up3[1].dw),
            in_point: dec_base + 3,
            out_point: dec_base + 4,
            relu: true,
            out_scale: scales[dec_base + 4],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "up3.1.pw".into(),
            conv: ConvRef::Pointwise(&self.up3[1].pw),
            in_point: dec_base + 4,
            out_point: dec_base + 5,
            relu: true,
            out_scale: scales[dec_base + 5],
            out_unsigned: false,
        });
        if let Some(film) = self.export_film_full() {
            let channels = film.host_heads().0.weight().dims()[1];
            plan.push(ExportStep::Film {
                name: "film.full".into(),
                film,
                channels,
                in_point: dec_base + 5,
                out_point: dec_base + 6,
                out_scale: scales[dec_base + 6],
            });
        } else {
            plan.push(ExportStep::Conv {
                name: "up1.1".into(),
                conv: ConvRef::Pointwise(&self.up1[1]),
                in_point: dec_base + 5,
                out_point: dec_base + 6,
                relu: true,
                out_scale: scales[dec_base + 6],
                out_unsigned: false,
            });
        }

        plan.push(ExportStep::Conv {
            name: "head.dw".into(),
            conv: ConvRef::Depthwise(&self.head.dw),
            in_point: dec_base + 6,
            out_point: dec_base + 7,
            relu: true,
            out_scale: scales[dec_base + 7],
            out_unsigned: false,
        });
        plan.push(ExportStep::Conv {
            name: "head.pw".into(),
            conv: ConvRef::Pointwise(&self.head.pw),
            in_point: dec_base + 7,
            out_point: point_count,
            relu: false,
            out_scale: OUTPUT_SCALE,
            out_unsigned: true,
        });
        plan.push(ExportStep::OutputImage {
            scale: OUTPUT_SCALE,
        });
        plan
    }

    fn export_notes(&self, bits: &Bits) -> Vec<String> {
        let mut notes = default_notes(bits);
        notes.insert(
            3,
            format!(
                "accumulator: int32; depthwise layers accumulate {} products of {} * {}, pointwise layers up to 64 * {} * {}",
                9,
                bits.act_max() as i32,
                bits.weight_max() as i32,
                bits.act_max() as i32,
                bits.weight_max() as i32
            ),
        );
        notes.push(
            "depthwise_conv2d: groups == in_channels; each output channel uses only its own 3x3 kernel"
                .to_owned(),
        );
        notes
    }
}

/// Lowers a quantised forward graph to `manifest.json` + `weights.bin`.
pub struct Int8Exporter;

impl Int8Exporter {
    /// Write accelerator artefacts under `dir`.
    pub fn write<M: Exportable>(
        model: &M,
        calibration: &Calibration,
        dir: &Path,
    ) -> Result<QuantManifest> {
        export_int8(model, calibration, dir)
    }
}

const INSTANCE_NORM_EPS: f32 = 1e-5;

fn export_film_head(pw: &PwConv<InferBackend>, in_scale: f32, qmax: f32) -> Result<FilmHead> {
    let [out_ch, in_ch, _, _] = pw.weight().dims();
    let weight = tensor_to_vec(&pw.weight())?;
    let (codes, scales) = quantise_weights_host(&weight, out_ch, qmax);
    let bias = read_bias(pw.conv.bias.as_ref())?;
    let bias_int32: Vec<i32> = bias
        .iter()
        .zip(&scales)
        .map(|(b, w)| (b / (in_scale * w)).round() as i32)
        .collect();
    Ok(FilmHead {
        in_channels: in_ch,
        out_channels: out_ch,
        weight_scale: scales,
        weight_codes: codes,
        bias_int32,
    })
}

/// Write weights.bin and manifest.json for a model under dir.
pub fn export_int8<M: Exportable>(
    model: &M,
    calibration: &Calibration,
    dir: &Path,
) -> Result<QuantManifest> {
    if let Some(why) = model.export_blocker() {
        bail!("{why}");
    }
    let points = model.quant_points();
    if points.len() != calibration.scales.len() {
        bail!(
            "calibration has {} scales but this model has {} quantised points",
            calibration.scales.len(),
            points.len()
        );
    }
    let bits = calibration.bits;
    let steps = model.export_plan(&calibration.scales, points.len());

    let mut blob: Vec<i8> = Vec::new();
    let mut byte_cursor = 0usize;
    let mut layers: Vec<QuantLayer> = Vec::new();
    let mut graph: Vec<QuantOp> = Vec::new();
    let mut lut_layers: Vec<crate::quant::LutLayer> = Vec::new();
    let mut norm_layers: Vec<InstanceNormLayer> = Vec::new();
    let mut film_layers: Vec<FilmLayer> = Vec::new();
    let per_byte = (8 / bits.weights) as usize;

    for step in &steps {
        match step {
            ExportStep::Conv {
                name,
                conv,
                in_point,
                out_point,
                relu,
                out_scale,
                out_unsigned,
            } => {
                let (codes, weight_scale, bias) = conv.weight_codes(bits.weight_max())?;
                let (stride, kernel, reflect_pad, in_ch, out_ch) = conv.layer_shape();
                let in_scale = calibration.scales[*in_point];
                let groups = conv.groups();

                if codes.len() % per_byte != 0 {
                    bail!(
                        "layer {name}: {} weight codes is not a multiple of {per_byte}",
                        codes.len()
                    );
                }
                let offset = blob.len();
                let byte_offset = byte_cursor;
                blob.extend_from_slice(&codes);
                byte_cursor += codes.len() / per_byte;

                let bias_int32: Vec<i32> = bias
                    .iter()
                    .zip(&weight_scale)
                    .map(|(b, w)| (b / (in_scale * w)).round() as i32)
                    .collect();
                let requant: Vec<f32> = weight_scale
                    .iter()
                    .map(|w| in_scale * w / out_scale)
                    .collect();

                layers.push(QuantLayer {
                    name: name.clone(),
                    op: conv.op_name(*relu),
                    in_channels: in_ch,
                    out_channels: out_ch,
                    groups,
                    act_bits: bits.acts,
                    weight_bits: bits.weights,
                    act_zero_point: 0,
                    out_relu: *relu,
                    out_unsigned: *out_unsigned,
                    kernel,
                    stride,
                    reflect_pad,
                    in_point: *in_point,
                    out_point: *out_point,
                    in_scale,
                    out_scale: *out_scale,
                    weight_scale,
                    bias,
                    bias_int32,
                    requant,
                    weight_offset: offset,
                    weight_count: codes.len(),
                    weight_byte_offset: byte_offset,
                    weight_bytes: codes.len() / per_byte,
                });
                graph.push(QuantOp {
                    op: "conv".to_owned(),
                    layer: Some(name.clone()),
                    point: Some(*in_point),
                    source_point: None,
                    scale: Some(in_scale),
                });
                if *relu {
                    graph.push(QuantOp {
                        op: "relu".to_owned(),
                        layer: Some(name.clone()),
                        point: None,
                        source_point: None,
                        scale: None,
                    });
                }
            }
            ExportStep::ResidualAdd {
                point,
                source_point,
                scale,
            } => {
                graph.push(QuantOp {
                    op: "add".to_owned(),
                    layer: None,
                    point: Some(*point),
                    source_point: Some(*source_point),
                    scale: Some(*scale),
                });
            }
            ExportStep::UpsampleNearest2x { point } => {
                graph.push(QuantOp {
                    op: "upsample_nearest2x".to_owned(),
                    layer: None,
                    point: Some(*point),
                    source_point: None,
                    scale: None,
                });
            }
            ExportStep::OutputImage { scale } => {
                graph.push(QuantOp {
                    op: "quantize_u8_output".to_owned(),
                    layer: None,
                    point: None,
                    source_point: None,
                    scale: Some(*scale),
                });
            }
            ExportStep::LutDw {
                name,
                lut,
                in_point,
                out_point,
                in_bits,
                out_bits,
                out_scale,
                post_relu,
            } => {
                let in_scale = calibration.scales[*in_point];
                let mut layer =
                    lut.to_manifest_layer(name, *in_bits, *out_bits, in_scale, LUT_LEAKY_SLOPE)?;
                layer.next_in_scale = *out_scale;
                layer.post_relu = *post_relu;
                lut_layers.push(layer);
                graph.push(QuantOp {
                    op: "lut_conv2d".to_owned(),
                    layer: Some(name.clone()),
                    point: Some(*out_point),
                    source_point: Some(*in_point),
                    scale: Some(in_scale),
                });
                if *post_relu {
                    graph.push(QuantOp {
                        op: "relu_point".to_owned(),
                        layer: None,
                        point: Some(*out_point),
                        source_point: None,
                        scale: None,
                    });
                }
            }
            ExportStep::InstanceNorm {
                name,
                point,
                channels,
                gamma,
                beta,
                out_scale,
            } => {
                norm_layers.push(InstanceNormLayer {
                    name: name.clone(),
                    point: *point,
                    channels: *channels,
                    in_scale: calibration.scales[*point],
                    out_scale: *out_scale,
                    eps: INSTANCE_NORM_EPS,
                    gamma: gamma.clone(),
                    beta: beta.clone(),
                });
                graph.push(QuantOp {
                    op: "instance_norm".to_owned(),
                    layer: Some(name.clone()),
                    point: Some(*point),
                    source_point: None,
                    scale: Some(*out_scale),
                });
            }
            ExportStep::Film {
                name,
                film,
                channels,
                in_point,
                out_point,
                out_scale,
            } => {
                let in_scale = calibration.scales[*in_point];
                let hidden = (channels / 4).max(1);
                let (reduce, gamma, beta) = film.host_heads();
                film_layers.push(FilmLayer {
                    name: name.clone(),
                    channels: *channels,
                    hidden,
                    in_point: *in_point,
                    out_point: *out_point,
                    in_scale,
                    out_scale: *out_scale,
                    reduce: export_film_head(reduce, in_scale, bits.weight_max())?,
                    gamma: export_film_head(gamma, in_scale, bits.weight_max())?,
                    beta: export_film_head(beta, in_scale, bits.weight_max())?,
                });
                graph.push(QuantOp {
                    op: "film".to_owned(),
                    layer: Some(name.clone()),
                    point: Some(*out_point),
                    source_point: Some(*in_point),
                    scale: Some(*out_scale),
                });
            }
            ExportStep::ReluPoint { point } => {
                graph.push(QuantOp {
                    op: "relu_point".to_owned(),
                    layer: None,
                    point: Some(*point),
                    source_point: None,
                    scale: None,
                });
            }
        }
    }

    let weight_bytes = blob.len() * bits.weights as usize / 8;
    let manifest = QuantManifest {
        format: "fpga_st_train.int8.v1".to_owned(),
        arch: calibration.arch.clone(),
        bits,
        act_qmax: bits.act_max(),
        output_qmax: Bits::unsigned_max(bits.acts),
        weight_qmax: bits.weight_max(),
        notes: model.export_notes(&bits),
        points: points.clone(),
        point_scales: calibration.scales.clone(),
        graph,
        layers,
        weight_file: "weights.bin".to_owned(),
        weight_packing: format!("i{}-x{}", bits.weights, 8 / bits.weights),
        weight_bytes,
        lut_layers,
        norm_layers,
        film_layers,
    };
    manifest
        .save(dir, &blob)
        .with_context(|| format!("cannot write the export to {}", dir.display()))?;
    Ok(manifest)
}

/// Write a LUT-Conv stack into `manifest.json`.
///
/// Each logical LUT becomes one row of the symmetric codebook already used for
/// activations: input code `+1` means `input_scale` in real units, and a channel
/// is the integer sum of the looked-up codes times that layer's `out_scale`.
/// The MLP weights are not stored. `weights.bin` is empty.
///
/// The next layer's input scale is the previous layer's `next_in_scale`, which
/// fits the worst-case integer sum into the same signed range. The tiny_dw
/// training path does not use that scale: [LutEncoder::forward_dw] snaps the
/// real sum onto [crate::student::lut::LUT_CODE_SPAN] before the decoder.
pub fn export_lut_encoder<B: Backend>(
    encoder: &LutEncoder<B>,
    dir: &Path,
    in_bits: u8,
    out_bits: u8,
    input_scale: f32,
) -> Result<QuantManifest> {
    let mut lut_layers = Vec::with_capacity(encoder.layers.len());
    let mut points = vec!["input".to_owned()];
    let mut point_scales = vec![input_scale];
    let mut graph = Vec::with_capacity(encoder.layers.len());
    let mut in_scale = input_scale;
    for (index, conv) in encoder.layers.iter().enumerate() {
        let name = format!("enc.{index}");
        let layer = conv.to_manifest_layer(&name, in_bits, out_bits, in_scale, LUT_LEAKY_SLOPE)?;
        points.push(name.clone());
        point_scales.push(layer.next_in_scale);
        graph.push(QuantOp {
            op: "lut_conv2d".to_owned(),
            layer: Some(name),
            point: Some(index + 1),
            source_point: Some(index),
            scale: Some(in_scale),
        });
        in_scale = layer.next_in_scale;
        lut_layers.push(layer);
    }

    let manifest = QuantManifest {
        format: "fpga_st_train.lut.v1".to_owned(),
        arch: "lut_conv".to_owned(),
        bits: Bits::new(in_bits, out_bits),
        act_qmax: Bits::max_value(in_bits),
        output_qmax: Bits::max_value(out_bits),
        weight_qmax: Bits::max_value(out_bits),
        notes: vec![
            "lut_layers is the truth table of each scalar input. The MLP is not stored.".to_owned(),
            "index = (input * out_channels + output) * table_len + (code + in_qmax).".to_owned(),
            "a channel in real units is sum(codes) * out_scale, then leaky_relu when set."
                .to_owned(),
            "weights.bin is unused. The integer interpreter executes lut_conv2d from these tables."
                .to_owned(),
        ],
        points,
        point_scales,
        graph,
        layers: Vec::new(),
        weight_file: "weights.bin".to_owned(),
        weight_packing: "none".to_owned(),
        weight_bytes: 0,
        lut_layers,
        norm_layers: Vec::new(),
        film_layers: Vec::new(),
    };
    manifest
        .save(dir, &[])
        .with_context(|| format!("cannot write the LUT manifest to {}", dir.display()))?;
    Ok(manifest)
}
