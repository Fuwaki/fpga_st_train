//! CPU channel-activity probe for a trained `tiny_dw_guided` checkpoint.
//!
//! One forward per content image on the ndarray backend. No training.
//!
//! ```text
//! cargo run --release --example channel_activity --no-default-features --features ndarray
//! ```

use std::collections::BTreeMap;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::Instant;

use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::relu;
use burn::tensor::module::{avg_pool2d, conv2d};
use burn::tensor::ops::{ConvOptions, PadMode};
use fpga_st_train::InferBackend;
use fpga_st_train::backend::{BACKEND_NAME, select_device};
use fpga_st_train::config::ModelCard;
use fpga_st_train::data::{load_rgb, rgb_to_tensor};
use fpga_st_train::registry::build_guided;
use fpga_st_train::student::StudentModel;
use fpga_st_train::student::guided::{Ablate, GuideSpec, GuidedStudent, PreInGain};

const ARMS: [&str; 2] = ["in", "ln"];
const IMAGES: [&str; 6] = ["animal", "cat", "city", "landscape", "portrait", "water"];

struct Weight {
    dims: Vec<usize>,
    data: Vec<f32>,
}

struct Collector {
    path: Vec<String>,
    weights: BTreeMap<String, Weight>,
}

impl ModuleVisitor<InferBackend> for Collector {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<InferBackend, D>>) {
        let tensor = param.val();
        let dims = tensor.dims();
        let data = tensor.into_data().to_vec::<f32>().expect("weight f32");
        self.weights.insert(
            self.path.join("."),
            Weight {
                dims: dims.as_slice().to_vec(),
                data,
            },
        );
    }
}

struct Stage {
    std: Vec<Vec<f64>>,
    abs: Vec<Vec<f64>>,
    zero: Vec<Vec<f64>>,
}

impl Stage {
    fn push(&mut self, std: Vec<f64>, abs: Vec<f64>, zero: Vec<f64>) {
        self.std.push(std);
        self.abs.push(abs);
        self.zero.push(zero);
    }
}

fn main() {
    assert!(
        BACKEND_NAME.contains("CPU") || BACKEND_NAME.contains("ndarray"),
        "this probe is CPU-only, built backend is {BACKEND_NAME}"
    );
    println!("backend {BACKEND_NAME}");
    let root = PathBuf::from("experiments/tiny_dw_whitebox_480p/trials");
    let device = select_device("cpu");
    print_budgets(&device);

    let mut trunk_std: BTreeMap<(&str, &str), Vec<f64>> = BTreeMap::new();
    for arm in ARMS {
        let trial = root.join(arm);
        let card = ModelCard::load(&trial).expect("model.json");
        let model: GuidedStudent<InferBackend> =
            build_guided::<InferBackend>(&card.student(), &device)
                .load_file(
                    trial.join("model.mpk"),
                    &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                    &device,
                )
                .expect("checkpoint");
        println!("\n=== {arm}  {} ===", model.describe());
        let mut collector = Collector {
            path: Vec::new(),
            weights: BTreeMap::new(),
        };
        model.visit(&mut collector);
        let weights = collector.weights;

        let h2_w = tensor4(require(&weights, "h2_guide.conv.weight"), &device);
        let h2_b = tensor1(require(&weights, "h2_guide.conv.bias"), &device);

        let mut stages: BTreeMap<String, Stage> = BTreeMap::new();
        for name in IMAGES {
            let path = trial.join(format!("content_{name}.png"));
            let image = load_rgb(&path).expect(name);
            let mut x = rgb_to_tensor::<InferBackend>(&image, &device);
            let [_, _, h, w] = x.dims();
            let pad_h = 480usize.saturating_sub(h);
            let pad_w = 640usize.saturating_sub(w);
            if pad_h > 0 || pad_w > 0 {
                x = x.pad([(0, pad_h), (0, pad_w)], PadMode::Reflect);
            }
            let started = Instant::now();
            model.trace(x.clone(), Ablate::None, &mut |stage, map| {
                let (std, abs, zero) = channel_stats(map);
                stages
                    .entry(stage.to_owned())
                    .or_insert_with(|| Stage {
                        std: Vec::new(),
                        abs: Vec::new(),
                        zero: Vec::new(),
                    })
                    .push(std, abs, zero);
            });
            let (std, abs, zero) = h2_relu_stats(&x, h2_w.clone(), h2_b.clone(), &device);
            stages
                .entry("h2_16".into())
                .or_insert_with(|| Stage {
                    std: Vec::new(),
                    abs: Vec::new(),
                    zero: Vec::new(),
                })
                .push(std, abs, zero);
            println!("  {name} {:.2}s", started.elapsed().as_secs_f64());
            let _ = stdout().flush();
        }

        println!(
            "  {:<12} {:>4} {:>4} {:>4} | {:>7} {:>7} {:>6} {:>4} {:>5} {:>7} {:>6}",
            "stage",
            "C",
            "keep",
            "drop",
            "std_bot",
            "abs_bot",
            "cv",
            "dead",
            "jac",
            "zero50",
            "off90"
        );
        for (name, stage) in &stages {
            let line = summarize(stage);
            println!(
                "  {:<12} {:>4} {:>4} {:>4} | {:>7.3} {:>7.3} {:>6.2} {:>4} {:>5.2} {:>7.2} {:>6}",
                name,
                line.channels,
                line.keep,
                line.drop,
                line.std_bot,
                line.abs_bot,
                line.cv,
                line.dead,
                line.jaccard,
                line.zero_med,
                line.off90,
            );
            let mean_std = mean_over_images(&stage.std);
            let order = order_ascending(&mean_std);
            let shown: Vec<usize> = if line.channels > 48 {
                order
                    .iter()
                    .take(8)
                    .copied()
                    .chain(order.iter().rev().take(8).copied())
                    .collect()
            } else {
                order.into_iter().take(line.drop).collect()
            };
            let label = if line.channels > 48 {
                "weakest8 .. strongest8"
            } else {
                "drop idx by mean spatial std"
            };
            println!(
                "    {label}: {}",
                shown
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if *name == "res0"
                || *name == "trunk_norm"
                || *name == "film_h2"
                || *name == "mix"
                || *name == "h2_16"
            {
                trunk_std.insert((arm, name_key(name)), mean_std);
            }
        }
        print_scales(&weights);
        print_coupled(&weights, &stages);
    }

    println!("\n=== in vs ln, overlap of the dropped set (lowest spatial std) ===");
    for stage in ["res0", "trunk_norm", "film_h2", "mix", "h2_16"] {
        let a = trunk_std.get(&("in", stage));
        let b = trunk_std.get(&("ln", stage));
        if let (Some(a), Some(b)) = (a, b) {
            let drop = a.len() - (a.len() * 2) / 3;
            let ov = jaccard_one(&bottom_set(a, drop), &bottom_set(b, drop));
            println!("  {stage:<12} drop {drop:>3}  jaccard {ov:.2}");
        }
    }
}

fn name_key(name: &str) -> &'static str {
    match name {
        "res0" => "res0",
        "trunk_norm" => "trunk_norm",
        "film_h2" => "film_h2",
        "mix" => "mix",
        "h2_16" => "h2_16",
        _ => "",
    }
}

struct Line {
    channels: usize,
    keep: usize,
    drop: usize,
    std_bot: f64,
    abs_bot: f64,
    cv: f64,
    dead: usize,
    jaccard: f64,
    zero_med: f64,
    off90: usize,
}

fn summarize(stage: &Stage) -> Line {
    let mean_std = mean_over_images(&stage.std);
    let mean_abs = mean_over_images(&stage.abs);
    let mean_zero = mean_over_images(&stage.zero);
    let channels = mean_std.len();
    let keep = (channels * 2) / 3;
    let drop = channels - keep;
    let mut zeros = mean_zero.clone();
    zeros.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let zero_med = zeros[channels / 2];
    let off90 = mean_zero.iter().filter(|z| **z > 0.90).count();
    let dead = mean_std.iter().filter(|s| s.powi(2) < 1e-5).count();
    Line {
        channels,
        keep,
        drop,
        std_bot: bottom_mass(&mean_std, drop),
        abs_bot: bottom_mass(&mean_abs, drop),
        cv: coeff_var(&mean_std),
        dead,
        jaccard: pairwise_jaccard(&stage.std, drop),
        zero_med,
        off90,
    }
}

fn print_budgets(device: &fpga_st_train::InferDevice) {
    let rows = [
        ("current", [32, 96, 32, 48, 16, 8]),
        ("x2/3", [21, 64, 21, 32, 10, 5]),
        ("align8", [24, 64, 24, 32, 8, 8]),
    ];
    println!("budget @ 640x480   params     Mmac   vs current");
    let mut base_params = 0usize;
    let mut base_macs = 0u64;
    for (i, (name, w)) in rows.iter().enumerate() {
        let spec = spec(*w);
        let params = GuidedStudent::<InferBackend>::new(spec.clone(), device).num_params();
        let macs = spec.cost(480, 640).macs();
        if i == 0 {
            base_params = params;
            base_macs = macs;
        }
        println!(
            "  {:<8} {:3}/{:3}/{:3}/{:3}/{:2}/{:1}  {:>8}  {:>7.1}   params {:>5.1}%  mac {:>5.1}%",
            name,
            w[0],
            w[1],
            w[2],
            w[3],
            w[4],
            w[5],
            params,
            macs as f64 / 1e6,
            100.0 * params as f64 / base_params as f64,
            100.0 * macs as f64 / base_macs as f64,
        );
    }
    println!("  widths are base/trunk/up_h4/up_h2/h2_guide/guide");
    println!(
        "  x2/3 keeps (n*2)/3 channels. align8 is the nearest multiple of 8, guide left at 8."
    );
}

fn spec(w: [usize; 6]) -> GuideSpec {
    GuideSpec {
        base: w[0],
        trunk: w[1],
        res_blocks: 1,
        up_h4: w[2],
        up_h2: w[3],
        guide: w[5],
        full: 0,
        full_layers: 1,
        shuffle: false,
        shuffle_guide: false,
        film_h2: true,
        film_guide: false,
        film_full: false,
        guide_relu: true,
        h2_guide: w[4],
        guide_kernel: 3,
        guide_gate: false,
        guide_resid: false,
        trunk_h4: true,
        in_eps: 1e-5,
        trunk_norm_kind: "ln".into(),
        pre_in_gain: PreInGain::Sqrt2,
        input_planes: 3,
        input_quant: "none".into(),
        styles: 1,
    }
}

fn require<'a>(weights: &'a BTreeMap<String, Weight>, key: &str) -> &'a Weight {
    weights.get(key).unwrap_or_else(|| {
        let known = weights.keys().cloned().collect::<Vec<_>>().join("\n  ");
        panic!("missing {key}\nknown:\n  {known}");
    })
}

fn print_scales(weights: &BTreeMap<String, Weight>) {
    let gamma = require(weights, "trunk_norm.norm.affine.gamma");
    let beta = require(weights, "trunk_norm.norm.affine.beta");
    let g: Vec<f64> = gamma.data.iter().map(|v| v.abs() as f64).collect();
    let b: Vec<f64> = beta.data.iter().map(|v| v.abs() as f64).collect();
    let drop = g.len() - (g.len() * 2) / 3;
    println!(
        "  norm |gamma| median {:.3}  cv {:.2}  bottom {drop} mass {:.3}  |beta| median {:.3}",
        median(&g),
        coeff_var(&g),
        bottom_mass(&g, drop),
        median(&b),
    );
}

fn print_coupled(weights: &BTreeMap<String, Weight>, stages: &BTreeMap<String, Stage>) {
    let pairs = [
        (
            "base",
            "stem",
            "enc0.pw.conv.weight",
            Some("enc0.dw.conv.weight"),
        ),
        (
            "trunk",
            "trunk_norm",
            "dec_h4.pw.conv.weight",
            Some("dec_h4.dw.conv.weight"),
        ),
        (
            "up_h4",
            "dec_h4",
            "dec_h2.pw.conv.weight",
            Some("dec_h2.dw.conv.weight"),
        ),
        ("up_h2", "film_h2", "to_guide.conv.weight", None),
        ("h2_16", "h2_16", "h2_guide_proj.conv.weight", None),
        (
            "guide",
            "mix",
            "head_sep.pw.conv.weight",
            Some("head_sep.dw.conv.weight"),
        ),
    ];
    println!("  coupled score = spatial std x depthwise L1 x next pointwise column L1");
    for (label, stage, key, dw_key) in pairs {
        let Some(stage) = stages.get(stage) else {
            println!("  {label:<8} missing activation {stage}");
            continue;
        };
        let Some(weight) = weights.get(key) else {
            println!("  {label:<8} missing weight {key}");
            continue;
        };
        let use_l1 = in_l1(weight);
        let std = mean_over_images(&stage.std);
        if use_l1.len() != std.len() {
            println!(
                "  {label:<8} width mismatch act {} vs {key} in {}",
                std.len(),
                use_l1.len()
            );
            continue;
        }
        let dw = dw_key.and_then(|name| weights.get(name)).map(out_l1);
        let score: Vec<f64> = std
            .iter()
            .zip(&use_l1)
            .enumerate()
            .map(|(i, (a, u))| {
                let depth = dw.as_ref().map(|d| d[i]).unwrap_or(1.0);
                a * u * depth
            })
            .collect();
        let drop = score.len() - (score.len() * 2) / 3;
        let use_bot = bottom_mass(&use_l1, drop);
        println!(
            "  {label:<8} next {key:<24} use_bot {use_bot:.3}  coupled_bot {:.3}  use_cv {:.2}",
            bottom_mass(&score, drop),
            coeff_var(&use_l1),
        );
        print_retention(label, &score);
    }
}

/// Channels kept so the removed tail holds at most `loss` of the coupled score.
fn print_retention(label: &str, score: &[f64]) {
    let mut order = order_ascending(score);
    order.reverse();
    let total: f64 = score.iter().sum();
    let mut acc = 0.0;
    let mut marks = [0usize; 4];
    let targets = [0.99, 0.98, 0.95, 0.90];
    let mut hit = [false; 4];
    for (kept, &idx) in order.iter().enumerate() {
        acc += score[idx];
        let frac = if total <= 1e-12 { 1.0 } else { acc / total };
        for (i, target) in targets.iter().enumerate() {
            if !hit[i] && frac + 1e-12 >= *target {
                marks[i] = kept + 1;
                hit[i] = true;
            }
        }
    }
    println!(
        "    keep for 99/98/95/90% of coupled score: {}/{}/{}/{} of {}",
        marks[0],
        marks[1],
        marks[2],
        marks[3],
        score.len()
    );
}

fn h2_relu_stats(
    image: &Tensor<InferBackend, 4>,
    weight: Tensor<InferBackend, 4>,
    bias: Tensor<InferBackend, 1>,
    device: &fpga_st_train::InferDevice,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let [_, _, h, w] = image.dims();
    let hp = h.div_ceil(8) * 8;
    let wp = w.div_ceil(8) * 8;
    let x = if hp == h && wp == w {
        image.clone()
    } else {
        image
            .clone()
            .pad([(0, hp - h), (0, wp - w)], PadMode::Reflect)
    };
    let low = avg_pool2d(x, [2, 2], [2, 2], [0, 0], true, false);
    let low = low.pad([(1, 1), (1, 1)], PadMode::Reflect);
    let y = relu(conv2d(
        low,
        weight,
        Some(bias),
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    ));
    let _ = device;
    channel_stats(&y)
}

fn channel_stats<B: Backend>(map: &Tensor<B, 4>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let [n, c, h, w] = map.dims();
    let values = map.clone().into_data().to_vec::<f32>().expect("activation");
    let spatial = n * h * w;
    let mut abs = vec![0.0f64; c];
    let mut sum = vec![0.0f64; c];
    let mut zero = vec![0.0f64; c];
    for bi in 0..n {
        for ci in 0..c {
            for pix in 0..(h * w) {
                let v = values[(bi * c + ci) * h * w + pix] as f64;
                sum[ci] += v;
                abs[ci] += v.abs();
                if v.abs() < 1e-6 {
                    zero[ci] += 1.0;
                }
            }
        }
    }
    let mut std = vec![0.0f64; c];
    for bi in 0..n {
        for ci in 0..c {
            let mean = sum[ci] / spatial as f64;
            for pix in 0..(h * w) {
                let v = values[(bi * c + ci) * h * w + pix] as f64;
                let d = v - mean;
                std[ci] += d * d;
            }
        }
    }
    for ci in 0..c {
        abs[ci] /= spatial as f64;
        std[ci] = (std[ci] / spatial as f64).sqrt();
        zero[ci] /= spatial as f64;
    }
    (std, abs, zero)
}

fn tensor4(weight: &Weight, device: &fpga_st_train::InferDevice) -> Tensor<InferBackend, 4> {
    let dims: [usize; 4] = weight.dims.clone().try_into().expect("rank 4");
    Tensor::from_data(TensorData::new(weight.data.clone(), dims), device)
}

fn tensor1(weight: &Weight, device: &fpga_st_train::InferDevice) -> Tensor<InferBackend, 1> {
    let dims: [usize; 1] = weight.dims.clone().try_into().expect("rank 1");
    Tensor::from_data(TensorData::new(weight.data.clone(), dims), device)
}

fn in_l1(weight: &Weight) -> Vec<f64> {
    let [out, inn, kh, kw]: [usize; 4] = weight.dims.clone().try_into().expect("conv weight");
    let k = kh * kw;
    let mut score = vec![0.0f64; inn];
    for o in 0..out {
        for i in 0..inn {
            let base = (o * inn + i) * k;
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += weight.data[base + t].abs() as f64;
            }
            score[i] += acc;
        }
    }
    score
}

fn out_l1(weight: &Weight) -> Vec<f64> {
    let [out, inn, kh, kw]: [usize; 4] = weight.dims.clone().try_into().expect("conv weight");
    let k = inn * kh * kw;
    let mut score = vec![0.0f64; out];
    for o in 0..out {
        let base = o * k;
        for t in 0..k {
            score[o] += weight.data[base + t].abs() as f64;
        }
    }
    score
}

fn mean_over_images(rows: &[Vec<f64>]) -> Vec<f64> {
    let c = rows[0].len();
    let mut acc = vec![0.0f64; c];
    for row in rows {
        for (i, v) in row.iter().enumerate() {
            acc[i] += v;
        }
    }
    let n = rows.len() as f64;
    for v in &mut acc {
        *v /= n;
    }
    acc
}

fn bottom_mass(scores: &[f64], drop: usize) -> f64 {
    let total: f64 = scores.iter().sum();
    if total <= 1e-12 || drop == 0 {
        return 0.0;
    }
    let mut order = scores.to_vec();
    order.sort_by(|a, b| a.partial_cmp(b).unwrap());
    order.iter().take(drop).sum::<f64>() / total
}

fn coeff_var(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    if mean <= 1e-12 {
        return 0.0;
    }
    let var = scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean
}

fn median(scores: &[f64]) -> f64 {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn order_ascending(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    idx
}

fn bottom_set(scores: &[f64], drop: usize) -> Vec<usize> {
    order_ascending(scores).into_iter().take(drop).collect()
}

fn pairwise_jaccard(rows: &[Vec<f64>], drop: usize) -> f64 {
    if rows.len() < 2 || drop == 0 {
        return 1.0;
    }
    let mut acc = 0.0;
    let mut n = 0.0;
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            acc += jaccard_one(&bottom_set(&rows[i], drop), &bottom_set(&rows[j], drop));
            n += 1.0;
        }
    }
    acc / n
}

fn jaccard_one(a: &[usize], b: &[usize]) -> f64 {
    let inter = a.iter().filter(|i| b.contains(i)).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}
