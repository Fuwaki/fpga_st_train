//! Turn a trained student into the integer artefacts an accelerator consumes.
//!
//! The export walks the *same* operation order as
//! [crate::student::tiny::TinyStudent::forward_core], so the manifest and the simulation
//! cannot disagree about which scale belongs where. The plan built by [plan] is a literal
//! transcription of that forward pass, one variant per operation; [export_int8] then emits it
//! once into the graph (what the hardware executes) and once into the layer table (the
//! constants each convolution needs).
//!
//! Everything a fixed-point accelerator needs is in manifest.json plus weights.bin;
//! crate::intref interprets exactly those two files, so a third-party RTL implementation can
//! be checked against it byte for byte.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::backend::InferBackend;
use crate::quant::{Bits, Calibration, QuantLayer, QuantManifest, QuantOp, quantise_weights_host};
use crate::student::TinyStudent;
use crate::student::tiny::Conv;

/// The scale the final uint8 image is stored on. The head int32 accumulator is rescaled onto
/// this grid and clamped to [0, 255], so the last stage of the pipeline needs no extra logic:
/// the image byte *is* the activation code.
pub const OUTPUT_SCALE: f32 = 1.0 / 255.0;

fn weight_codes(conv: &Conv<InferBackend>, qmax: f32) -> Result<(Vec<i8>, Vec<f32>, Vec<f32>)> {
    let weight: Vec<f32> = conv
        .weight()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("cannot read a weight tensor: {e}"))?;
    let [out_ch, _, _, _] = conv.weight().dims();
    let (codes, scales) = quantise_weights_host(&weight, out_ch, qmax);
    let bias: Vec<f32> = match conv.conv.bias.as_ref() {
        Some(bias) => bias
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("cannot read a bias tensor: {e}"))?,
        None => Vec::new(),
    };
    Ok((codes, scales, bias))
}

/// One operation of the forward pass, in execution order.
enum Step<'a> {
    Conv {
        name: String,
        conv: &'a Conv<InferBackend>,
        /// Quantised point this layer reads (index into points / point_scales).
        in_point: usize,
        /// Quantised point this layer writes. The head writes the image itself, so its value
        /// is points.len().
        out_point: usize,
        relu: bool,
        out_scale: f32,
        /// True only for the head: its codes are the unsigned image bytes.
        out_unsigned: bool,
    },
    /// out = requant(block input) + requant(c2 result). The c2 convolution already lands on
    /// the grid of point; the block input has to be requantised onto it, which is what
    /// source_point identifies.
    ResidualAdd {
        point: usize,
        source_point: usize,
        scale: f32,
    },
    /// Nearest-neighbour 2x upsampling: a wire on hardware, no arithmetic, no scale change.
    /// The codes at this point are doubled in resolution in place, so the point index stays
    /// the same and only the tensor shape changes.
    UpsampleNearest2x { point: usize },
    /// Emit the image: dequantise the head codes with this scale (they are unsigned bytes).
    OutputImage { scale: f32 },
}

/// The exact operation sequence of forward_core, with every scale filled in.
fn plan<'a>(
    model: &'a TinyStudent<InferBackend>,
    scales: &[f32],
    point_count: usize,
) -> Vec<Step<'a>> {
    let base = 3 + 2 * model.res.len();
    let mut plan = Vec::new();

    plan.push(Step::Conv {
        name: "enc.0".into(),
        conv: &model.enc[0],
        in_point: 0,
        out_point: 1,
        relu: true,
        out_scale: scales[1],
        out_unsigned: false,
    });
    plan.push(Step::Conv {
        name: "enc.1".into(),
        conv: &model.enc[1],
        in_point: 1,
        out_point: 2,
        relu: true,
        out_scale: scales[2],
        out_unsigned: false,
    });
    plan.push(Step::Conv {
        name: "enc.2".into(),
        conv: &model.enc[2],
        in_point: 2,
        out_point: 3,
        relu: true,
        out_scale: scales[3],
        out_unsigned: false,
    });

    for (r, block) in model.res.iter().enumerate() {
        let point = 3 + 2 * r;
        plan.push(Step::Conv {
            name: format!("res.{r}.c1"),
            conv: &block.c1,
            in_point: point,
            out_point: point + 1,
            relu: true,
            out_scale: scales[point + 1],
            out_unsigned: false,
        });
        plan.push(Step::Conv {
            name: format!("res.{r}.c2"),
            conv: &block.c2,
            in_point: point + 1,
            out_point: point + 2,
            relu: false,
            out_scale: scales[point + 2],
            out_unsigned: false,
        });
        plan.push(Step::ResidualAdd {
            point: point + 2,
            source_point: point,
            scale: scales[point + 2],
        });
    }

    // The trunk is doubled before the first full-resolution convolution; the codes stay on
    // the same grid, which is why upsampling needs no requantisation anywhere.
    plan.push(Step::UpsampleNearest2x { point: base });
    plan.push(Step::Conv {
        name: "up3.0".into(),
        conv: &model.up3[0],
        in_point: base,
        out_point: base + 1,
        relu: true,
        out_scale: scales[base + 1],
        out_unsigned: false,
    });
    plan.push(Step::Conv {
        name: "up1.0".into(),
        conv: &model.up1[0],
        in_point: base + 1,
        out_point: base + 2,
        relu: true,
        out_scale: scales[base + 2],
        out_unsigned: false,
    });

    plan.push(Step::UpsampleNearest2x { point: base + 2 });
    plan.push(Step::Conv {
        name: "up3.1".into(),
        conv: &model.up3[1],
        in_point: base + 2,
        out_point: base + 3,
        relu: true,
        out_scale: scales[base + 3],
        out_unsigned: false,
    });
    plan.push(Step::Conv {
        name: "up1.1".into(),
        conv: &model.up1[1],
        in_point: base + 3,
        out_point: base + 4,
        relu: true,
        out_scale: scales[base + 4],
        out_unsigned: false,
    });

    plan.push(Step::Conv {
        name: "head".into(),
        conv: &model.head,
        in_point: base + 4,
        out_point: point_count,
        relu: false,
        out_scale: OUTPUT_SCALE,
        out_unsigned: true,
    });
    plan.push(Step::OutputImage {
        scale: OUTPUT_SCALE,
    });
    plan
}

/// Write weights.bin and manifest.json for a model under dir.
pub fn export_int8(
    model: &TinyStudent<InferBackend>,
    calibration: &Calibration,
    dir: &Path,
) -> Result<QuantManifest> {
    let points = model.quant_layout();
    if points.len() != calibration.scales.len() {
        bail!(
            "calibration has {} scales but this model has {} quantised points",
            calibration.scales.len(),
            points.len()
        );
    }
    let bits = calibration.bits;
    let steps = plan(model, &calibration.scales, points.len());

    let mut blob: Vec<i8> = Vec::new();
    let mut byte_cursor = 0usize;
    let mut layers: Vec<QuantLayer> = Vec::new();
    let mut graph: Vec<QuantOp> = Vec::new();
    let per_byte = (8 / bits.weights) as usize;

    for step in &steps {
        match step {
            Step::Conv {
                name,
                conv,
                in_point,
                out_point,
                relu,
                out_scale,
                out_unsigned,
            } => {
                let (codes, weight_scale, bias) = weight_codes(conv, bits.weight_max())?;
                let [out_ch, in_ch, kh, kw] = conv.weight().dims();
                let in_scale = calibration.scales[*in_point];

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
                    op: if *relu {
                        "conv2d+relu".to_owned()
                    } else {
                        "conv2d".to_owned()
                    },
                    in_channels: in_ch,
                    out_channels: out_ch,
                    act_bits: bits.acts,
                    weight_bits: bits.weights,
                    act_zero_point: 0,
                    out_relu: *relu,
                    out_unsigned: *out_unsigned,
                    kernel: [kh, kw],
                    stride: conv.conv.stride,
                    reflect_pad: conv.reflect_pad(),
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
            Step::ResidualAdd {
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
            Step::UpsampleNearest2x { point } => {
                graph.push(QuantOp {
                    op: "upsample_nearest2x".to_owned(),
                    layer: None,
                    point: Some(*point),
                    source_point: None,
                    scale: None,
                });
            }
            Step::OutputImage { scale } => {
                graph.push(QuantOp {
                    op: "quantize_u8_output".to_owned(),
                    layer: None,
                    point: None,
                    source_point: None,
                    scale: Some(*scale),
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
        notes: vec![
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
            format!(
                "accumulator: int32; the widest layer here is 576 products of {} * {}",
                bits.act_max() as i32,
                bits.weight_max() as i32
            ),
            "bias_int32[c]: added to the accumulator before requantisation".to_owned(),
            "requant[c]: multiply the int32 accumulator by this, round to nearest, then clamp: [-act_qmax, act_qmax] normally, [0, output_qmax] for the head".to_owned(),
            "reflect_pad: reflection padding applied before a valid convolution".to_owned(),
            "upsample_nearest2x: a wire, no arithmetic, no scale change".to_owned(),
            "add: both operands must be on the output point grid; the second convolution is, already, and source_point is requantised by point_scales[source_point] / point_scales[point]"
                .to_owned(),
            "quantize_u8_output: the image byte is the head unsigned code, scale 1/255".to_owned(),
        ],
        points: points.clone(),
        point_scales: calibration.scales.clone(),
        graph,
        layers,
        weight_file: "weights.bin".to_owned(),
        weight_packing: format!("i{}-x{}", bits.weights, 8 / bits.weights),
        weight_bytes,
    };
    manifest
        .save(dir, &blob)
        .with_context(|| format!("cannot write the export to {}", dir.display()))?;
    Ok(manifest)
}
