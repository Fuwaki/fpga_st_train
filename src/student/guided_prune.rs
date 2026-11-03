//! Slice a trained `tiny_dw_guided` checkpoint down to fewer channels.
//!
//! The kept index in each width is the one with the largest coupled score
//! (spatial standard deviation on the deployment frame, times how hard the next
//! layer reads that channel). Indices are then sorted, so surviving channels
//! stay in their original order. A fresh narrow student receives those slices;
//! Adam state is not copied.

use std::collections::{BTreeMap, BTreeSet};

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::avg_pool2d;

use super::{Ablate, GuidedStudent};

/// Output widths of a pruned guided student.
#[derive(Clone, Copy, Debug)]
pub struct PruneWidths {
    pub base: usize,
    pub trunk: usize,
    pub up_h4: usize,
    pub up_h2: usize,
    pub h2_guide: usize,
    pub guide: usize,
    /// Residual blocks kept. Blocks at this index and after are dropped, not sliced.
    pub res_blocks: usize,
}

/// Which source channels survived, and how much of each score they still hold.
#[derive(Clone, Debug)]
pub struct PruneReport {
    pub kept: BTreeMap<String, Vec<usize>>,
    pub retained: BTreeMap<String, f64>,
    pub source_params: usize,
    pub params: usize,
    pub source_macs: u64,
    pub macs: u64,
}

#[derive(Clone)]
struct Packed {
    dims: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Width {
    Base,
    Trunk,
    UpH4,
    UpH2,
    H2,
    Guide,
    Hidden,
    /// Every index of this axis. RGB inputs and the 3-channel head use it.
    All,
}

struct Keep {
    base: Vec<usize>,
    trunk: Vec<usize>,
    up_h4: Vec<usize>,
    up_h2: Vec<usize>,
    h2: Vec<usize>,
    guide: Vec<usize>,
    hidden: Vec<usize>,
}

impl Keep {
    fn indices(&self, width: Width) -> &[usize] {
        match width {
            Width::Base => &self.base,
            Width::Trunk => &self.trunk,
            Width::UpH4 => &self.up_h4,
            Width::UpH2 => &self.up_h2,
            Width::H2 => &self.h2,
            Width::Guide => &self.guide,
            Width::Hidden => &self.hidden,
            Width::All => &[],
        }
    }
}

impl<B: Backend> GuidedStudent<B> {
    /// Rank channels on `images` and copy the surviving weights into a narrower student.
    ///
    /// `images` are `[1, 3, H, W]` in `[0, 1]`. The score uses that spatial size, so pad
    /// to the deployment frame before calling. `mac_h` and `mac_w` are only the hardware
    /// account printed with the report.
    pub fn prune(
        &self,
        images: &[Tensor<B, 4>],
        widths: PruneWidths,
        mac_h: usize,
        mac_w: usize,
        device: &B::Device,
    ) -> Result<(Self, PruneReport), String> {
        self.ensure_whitebox_line()?;
        check_shrink("base", self.spec.base, widths.base)?;
        check_shrink("trunk", self.spec.trunk, widths.trunk)?;
        check_shrink("up_h4", self.spec.up_h4, widths.up_h4)?;
        check_shrink("up_h2", self.spec.up_h2, widths.up_h2)?;
        check_shrink("h2_guide", self.spec.h2_guide, widths.h2_guide)?;
        check_shrink("guide", self.spec.guide, widths.guide)?;
        if widths.res_blocks > self.spec.res_blocks {
            return Err(format!(
                "res_blocks prune keeps {}, source has {}",
                widths.res_blocks, self.spec.res_blocks
            ));
        }
        if images.is_empty() {
            return Err("prune needs at least one image".to_owned());
        }

        let scores = self.coupled_scores(images)?;
        let mut kept = BTreeMap::new();
        let mut retained = BTreeMap::new();
        let groups = [
            ("base", &scores.base, widths.base),
            ("trunk", &scores.trunk, widths.trunk),
            ("up_h4", &scores.up_h4, widths.up_h4),
            ("up_h2", &scores.up_h2, widths.up_h2),
            ("h2_guide", &scores.h2, widths.h2_guide),
            ("guide", &scores.guide, widths.guide),
        ];
        for (name, score, width) in groups {
            let idx = top_k(score, width)?;
            retained.insert(name.to_owned(), retained_mass(score, &idx));
            kept.insert(name.to_owned(), idx);
        }

        let weights = collect_weights(self);
        let (hidden, hidden_mass) = choose_hidden(
            weights
                .get("film_h2.reduce.conv.weight")
                .ok_or("film_h2.reduce.conv.weight")?,
            (widths.up_h2 / 4).max(1),
        )?;
        retained.insert("film_hidden".to_owned(), hidden_mass);
        kept.insert("film_hidden".to_owned(), hidden.clone());

        let keep = Keep {
            base: kept["base"].clone(),
            trunk: kept["trunk"].clone(),
            up_h4: kept["up_h4"].clone(),
            up_h2: kept["up_h2"].clone(),
            h2: kept["h2_guide"].clone(),
            guide: kept["guide"].clone(),
            hidden,
        };
        let sliced = drop_residual_blocks(slice_all(&weights, &keep)?, widths.res_blocks);

        let mut spec = self.spec.clone();
        spec.base = widths.base;
        spec.trunk = widths.trunk;
        spec.up_h4 = widths.up_h4;
        spec.up_h2 = widths.up_h2;
        spec.h2_guide = widths.h2_guide;
        spec.guide = widths.guide;
        spec.res_blocks = widths.res_blocks;
        spec.check()?;
        let narrow = GuidedStudent::new(spec, device);
        let narrow_paths = param_paths(&narrow);
        let source_paths: BTreeSet<String> = weights
            .keys()
            .filter(|path| !drops_residual(path, widths.res_blocks))
            .cloned()
            .collect();
        if narrow_paths != source_paths {
            return Err(format!(
                "pruned graph parameters differ from the source\n  only in source: {}\n  only in narrow: {}",
                set_diff(&source_paths, &narrow_paths),
                set_diff(&narrow_paths, &source_paths),
            ));
        }
        let mut installer = Installer {
            path: Vec::new(),
            slots: sliced,
        };
        let narrow = narrow.map(&mut installer);
        if !installer.slots.is_empty() {
            let left = installer
                .slots
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("prune left unused weights: {left}"));
        }

        let report = PruneReport {
            kept,
            retained,
            source_params: self.num_params(),
            params: narrow.num_params(),
            source_macs: self.spec.cost(mac_h, mac_w).macs(),
            macs: narrow.spec.cost(mac_h, mac_w).macs(),
        };
        Ok((narrow, report))
    }

    /// Copy every weight, and fold the three image convolutions onto luma plus the 135° axis.
    ///
    /// Float training feeds centered `C`, so the bias is left alone. The `+128` bias
    /// correction in `scripts/fold_stem_2ch.py` is only for the unsigned-byte export.
    pub fn fold_yc(&self, device: &B::Device) -> Result<Self, String> {
        if self.spec.input_planes != 3 {
            return Err(format!(
                "fold_yc expects a 3-plane student, got {}",
                self.spec.input_planes
            ));
        }
        let mut weights = collect_weights(self);
        for path in [
            "stem.conv.weight",
            "h2_guide.conv.weight",
            "guide.conv.weight",
        ] {
            if let Some(packed) = weights.remove(path) {
                weights.insert(path.to_owned(), fold_image_weight(&packed)?);
            }
        }
        let mut spec = self.spec.clone();
        spec.input_planes = 2;
        spec.check()?;
        let yc = GuidedStudent::new(spec, device);
        let yc_paths = param_paths(&yc);
        let source_paths: BTreeSet<String> = weights.keys().cloned().collect();
        if yc_paths != source_paths {
            return Err(format!(
                "folded graph parameters differ\n  only in source: {}\n  only in yc: {}",
                set_diff(&source_paths, &yc_paths),
                set_diff(&yc_paths, &source_paths),
            ));
        }
        let mut installer = Installer {
            path: Vec::new(),
            slots: weights,
        };
        let yc = yc.map(&mut installer);
        if !installer.slots.is_empty() {
            let left = installer
                .slots
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("fold left unused weights: {left}"));
        }
        Ok(yc)
    }

    /// Copy the trained trunk norm and H/2 FiLM into every style slot.
    ///
    /// Style 0 keeps the checkpoint names. Later styles start as that same packet,
    /// so one finetuned style is the shared initialisation and the others move away.
    pub fn expand_styles(&self, styles: usize, device: &B::Device) -> Result<Self, String> {
        if self.spec.styles != 1 {
            return Err(format!(
                "expand_styles starts from one style packet, got {}",
                self.spec.styles
            ));
        }
        if styles < 2 {
            return Err(format!(
                "expand_styles needs at least 2 styles, got {styles}"
            ));
        }
        if !self.spec.film_h2 {
            return Err("style packets need the H/2 FiLM".to_owned());
        }
        let mut spec = self.spec.clone();
        spec.styles = styles;
        spec.check()?;
        let wide = GuidedStudent::new(spec, device);
        let source = collect_weights(self);
        let mut weights = collect_weights(&wide);
        let mut filled = BTreeSet::new();
        for (path, packed) in &source {
            if !weights.contains_key(path) {
                return Err(format!(
                    "single-style weight {path} is not in the packet graph"
                ));
            }
            weights.insert(path.clone(), packed.clone());
            filled.insert(path.clone());
        }
        for slot in 0..styles - 1 {
            for (path, packed) in &source {
                let dest = if let Some(rest) = path.strip_prefix("trunk_norm.") {
                    format!("style_packets.{slot}.trunk_norm.{rest}")
                } else if let Some(rest) = path.strip_prefix("film_h2.") {
                    format!("style_packets.{slot}.film_h2.{rest}")
                } else {
                    continue;
                };
                if !weights.contains_key(&dest) {
                    return Err(format!("style packet has no parameter {dest}"));
                }
                weights.insert(dest.clone(), packed.clone());
                filled.insert(dest);
            }
        }
        let wide_paths = param_paths(&wide);
        if filled != wide_paths {
            return Err(format!(
                "style packet copy missed {}\n  unexpected {}",
                set_diff(&wide_paths, &filled),
                set_diff(&filled, &wide_paths),
            ));
        }
        let mut installer = Installer {
            path: Vec::new(),
            slots: weights,
        };
        let wide = wide.map(&mut installer);
        if !installer.slots.is_empty() {
            let left = installer
                .slots
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("expand_styles left unused weights: {left}"));
        }
        Ok(wide)
    }

    fn ensure_whitebox_line(&self) -> Result<(), String> {
        if self.spec.full != 0 {
            return Err("channel prune supports the full=0 head".to_owned());
        }
        if self.spec.h2_guide == 0 || !self.spec.film_h2 {
            return Err("channel prune expects the H/2 guide and FiLM".to_owned());
        }
        if self.spec.guide_gate
            || self.spec.guide_resid
            || self.spec.film_guide
            || self.spec.film_full
        {
            return Err("channel prune expects the additive guide without extra FiLM".to_owned());
        }
        if self.spec.shuffle || self.spec.shuffle_guide {
            return Err("channel prune expects nearest upsampling".to_owned());
        }
        Ok(())
    }

    fn coupled_scores(&self, images: &[Tensor<B, 4>]) -> Result<Scores, String> {
        let enc0_dw = out_l1(&self.enc0.dw.weight());
        let enc0_pw = in_l1(&self.enc0.pw.weight());
        let dec_h4_dw = out_l1(&self.dec_h4.dw.weight());
        let dec_h4_pw = in_l1(&self.dec_h4.pw.weight());
        let dec_h2_dw = out_l1(&self.dec_h2.dw.weight());
        let dec_h2_pw = in_l1(&self.dec_h2.pw.weight());
        let to_guide = in_l1(&self.to_guide.weight());
        let proj = self.h2_guide_proj.as_ref().ok_or("h2 guide projection")?;
        let h2_proj = in_l1(&proj.weight());
        let head = self.head_sep.as_ref().ok_or("full-resolution head")?;
        let head_dw = out_l1(&head.dw.weight());
        let head_pw = in_l1(&head.pw.weight());

        let mut stem = Vec::new();
        let mut trunk = Vec::new();
        let mut up_h4 = Vec::new();
        let mut up_h2 = Vec::new();
        let mut h2 = Vec::new();
        let mut mix = Vec::new();
        for (i, image) in images.iter().enumerate() {
            println!("  score {}/{}", i + 1, images.len());
            self.trace(image.clone(), Ablate::None, &mut |name, map| match name {
                "stem" => stem.push(channel_std(map)),
                "trunk_norm" => trunk.push(channel_std(map)),
                "dec_h4" => up_h4.push(channel_std(map)),
                "film_h2" => up_h2.push(channel_std(map)),
                "mix" => mix.push(channel_std(map)),
                _ => {}
            });
            h2.push(channel_std(&self.h2_features(image.clone())));
        }
        let stem = mean_rows(&stem)?;
        let trunk = mean_rows(&trunk)?;
        let up_h4 = mean_rows(&up_h4)?;
        let up_h2 = mean_rows(&up_h2)?;
        let h2 = mean_rows(&h2)?;
        let mix = mean_rows(&mix)?;
        Ok(Scores {
            base: product3(&stem, &enc0_dw, &enc0_pw)?,
            trunk: product3(&trunk, &dec_h4_dw, &dec_h4_pw)?,
            up_h4: product3(&up_h4, &dec_h2_dw, &dec_h2_pw)?,
            up_h2: product2(&up_h2, &to_guide)?,
            h2: product2(&h2, &h2_proj)?,
            guide: product3(&mix, &head_dw, &head_pw)?,
        })
    }

    /// ReLU features of the H/2 RGB guide, before the 1×1 that aligns it to `up_h2`.
    fn h2_features(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = super::pad_spatial(self.student_input(image), 8);
        let low = avg_pool2d(x, [2, 2], [2, 2], [0, 0], true, false);
        let conv = self.h2_guide.as_ref().expect("h2 guide");
        relu(conv.forward(low))
    }
}

struct Scores {
    base: Vec<f64>,
    trunk: Vec<f64>,
    up_h4: Vec<f64>,
    up_h2: Vec<f64>,
    h2: Vec<f64>,
    guide: Vec<f64>,
}

fn check_shrink(name: &str, from: usize, to: usize) -> Result<(), String> {
    if to == 0 || to > from {
        return Err(format!(
            "{name} prune width must be in 1..={from}, got {to}"
        ));
    }
    Ok(())
}

fn product2(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err(format!("score width {} != {}", a.len(), b.len()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
}

fn product3(a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>, String> {
    product2(&product2(a, b)?, c)
}

fn mean_rows(rows: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let Some(first) = rows.first() else {
        return Err("a scored activation was missing".to_owned());
    };
    let mut acc = vec![0.0f64; first.len()];
    for row in rows {
        if row.len() != acc.len() {
            return Err("activation width changed between images".to_owned());
        }
        for (slot, value) in acc.iter_mut().zip(row) {
            *slot += value;
        }
    }
    let n = rows.len() as f64;
    for value in &mut acc {
        *value /= n;
    }
    Ok(acc)
}

fn channel_std<B: Backend>(map: &Tensor<B, 4>) -> Vec<f64> {
    let [n, c, h, w] = map.dims();
    let values = map
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("activation f32");
    let spatial = n * h * w;
    let mut sum = vec![0.0f64; c];
    for bi in 0..n {
        for ci in 0..c {
            for pix in 0..(h * w) {
                sum[ci] += values[(bi * c + ci) * h * w + pix] as f64;
            }
        }
    }
    let mut var = vec![0.0f64; c];
    for bi in 0..n {
        for ci in 0..c {
            let mean = sum[ci] / spatial as f64;
            for pix in 0..(h * w) {
                let diff = values[(bi * c + ci) * h * w + pix] as f64 - mean;
                var[ci] += diff * diff;
            }
        }
    }
    var.iter().map(|v| (v / spatial as f64).sqrt()).collect()
}

fn out_l1<B: Backend>(weight: &Tensor<B, 4>) -> Vec<f64> {
    let packed = pack(weight);
    let [out, inn, kh, kw] = four(&packed.dims).expect("weight rank");
    let k = inn * kh * kw;
    let mut score = vec![0.0f64; out];
    for o in 0..out {
        let base = o * k;
        for tap in 0..k {
            score[o] += packed.data[base + tap].abs() as f64;
        }
    }
    score
}

fn in_l1<B: Backend>(weight: &Tensor<B, 4>) -> Vec<f64> {
    let packed = pack(weight);
    let [out, inn, kh, kw] = four(&packed.dims).expect("weight rank");
    let k = kh * kw;
    let mut score = vec![0.0f64; inn];
    for o in 0..out {
        for i in 0..inn {
            let base = (o * inn + i) * k;
            for tap in 0..k {
                score[i] += packed.data[base + tap].abs() as f64;
            }
        }
    }
    score
}

fn pack<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Packed {
    let dims = tensor.dims();
    Packed {
        dims: dims.as_slice().to_vec(),
        data: tensor
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("weight f32"),
    }
}

fn top_k(scores: &[f64], k: usize) -> Result<Vec<usize>, String> {
    if k == 0 || k > scores.len() {
        return Err(format!("keep {k} of {} channels", scores.len()));
    }
    if k == scores.len() {
        return Ok((0..k).collect());
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut keep: Vec<usize> = order.into_iter().take(k).collect();
    keep.sort_unstable();
    Ok(keep)
}

fn retained_mass(scores: &[f64], kept: &[usize]) -> f64 {
    let total: f64 = scores.iter().sum();
    if total <= 1e-12 {
        return 1.0;
    }
    kept.iter().map(|&i| scores[i]).sum::<f64>() / total
}

fn choose_hidden(reduce: &Packed, new_hidden: usize) -> Result<(Vec<usize>, f64), String> {
    let [old, inn, kh, kw] = four(&reduce.dims)?;
    let k = inn * kh * kw;
    let mut scores = vec![0.0f64; old];
    for o in 0..old {
        let base = o * k;
        for tap in 0..k {
            scores[o] += reduce.data[base + tap].abs() as f64;
        }
    }
    let idx = top_k(&scores, new_hidden)?;
    let mass = retained_mass(&scores, &idx);
    Ok((idx, mass))
}

struct Bag {
    path: Vec<String>,
    weights: BTreeMap<String, Packed>,
}

impl<B: Backend> ModuleVisitor<B> for Bag {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        let dims = tensor.dims();
        let data = tensor.into_data().to_vec::<f32>().expect("parameter f32");
        self.weights.insert(
            self.path.join("."),
            Packed {
                dims: dims.as_slice().to_vec(),
                data,
            },
        );
    }
}

fn collect_weights<B: Backend>(model: &GuidedStudent<B>) -> BTreeMap<String, Packed> {
    let mut bag = Bag {
        path: Vec::new(),
        weights: BTreeMap::new(),
    };
    model.visit(&mut bag);
    bag.weights
}

fn param_paths<B: Backend>(model: &GuidedStudent<B>) -> BTreeSet<String> {
    collect_weights(model).into_keys().collect()
}

struct Installer {
    path: Vec<String>,
    slots: BTreeMap<String, Packed>,
}

impl<B: Backend> ModuleMapper<B> for Installer {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let key = self.path.join(".");
        let packed = self
            .slots
            .remove(&key)
            .unwrap_or_else(|| panic!("prune has no slice for {key}"));
        if packed.dims.len() != D {
            panic!(
                "{key} rank {} does not match the parameter",
                packed.dims.len()
            );
        }
        let dims: [usize; D] = packed
            .dims
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("{key} rank"));
        let (_id, tensor, _mapper) = param.consume();
        let device = tensor.device();
        Param::from_tensor(Tensor::from_data(
            TensorData::new(packed.data, dims),
            &device,
        ))
    }
}

fn drops_residual(path: &str, keep_blocks: usize) -> bool {
    let Some(rest) = path.strip_prefix("res.") else {
        return false;
    };
    let Some(index) = rest.split('.').next() else {
        return false;
    };
    index
        .parse::<usize>()
        .is_ok_and(|block| block >= keep_blocks)
}

fn drop_residual_blocks(
    mut weights: BTreeMap<String, Packed>,
    keep_blocks: usize,
) -> BTreeMap<String, Packed> {
    weights.retain(|path, _| !drops_residual(path, keep_blocks));
    weights
}

fn fold_image_weight(packed: &Packed) -> Result<Packed, String> {
    let [out, cin, kh, kw] = four(&packed.dims).map_err(|err| format!("image kernel: {err}"))?;
    if cin != 3 {
        return Err(format!("image kernel has {cin} input channels"));
    }
    let theta = 135.0_f64.to_radians();
    let (u0, u1) = (theta.cos(), theta.sin());
    let m = super::RGB_FROM_YCBCR;
    let mut data = vec![0.0f32; out * 2 * kh * kw];
    for o in 0..out {
        for ky in 0..kh {
            for kx in 0..kw {
                for j in 0..2 {
                    let mut acc = 0.0f64;
                    for i in 0..3 {
                        let scale = if j == 0 {
                            m[i][0]
                        } else {
                            m[i][1] * u0 + m[i][2] * u1
                        };
                        let src = ((o * cin + i) * kh + ky) * kw + kx;
                        acc += packed.data[src] as f64 * scale;
                    }
                    data[((o * 2 + j) * kh + ky) * kw + kx] = acc as f32;
                }
            }
        }
    }
    Ok(Packed {
        dims: vec![out, 2, kh, kw],
        data,
    })
}

fn slice_all(
    weights: &BTreeMap<String, Packed>,
    keep: &Keep,
) -> Result<BTreeMap<String, Packed>, String> {
    let mut out = BTreeMap::new();
    for (path, packed) in weights {
        out.insert(path.clone(), slice_param(path, packed, keep)?);
    }
    Ok(out)
}

fn slice_param(path: &str, packed: &Packed, keep: &Keep) -> Result<Packed, String> {
    let (out, inn) = axes(path)?;
    if packed.dims.len() == 1 {
        let idx = axis_indices(out, packed.dims[0], keep)?;
        return slice_1d(packed, &idx);
    }
    let [channels_out, channels_in, _, _] =
        four(&packed.dims).map_err(|err| format!("{path}: {err}"))?;
    let out_idx = axis_indices(out, channels_out, keep)?;
    let in_idx = axis_indices(inn, channels_in, keep)?;
    slice_4d(packed, &out_idx, &in_idx).map_err(|err| format!("{path}: {err}"))
}

fn axes(path: &str) -> Result<(Width, Width), String> {
    if path.contains(".dw.") {
        return Ok((dw_width(path)?, Width::All));
    }
    if path.contains(".pw.") || path.contains(".conv.") {
        return pw_axes(path);
    }
    if path.starts_with("trunk_norm.norm.affine.") {
        return Ok((Width::Trunk, Width::All));
    }
    Err(format!("unclassified parameter {path}"))
}

fn dw_width(path: &str) -> Result<Width, String> {
    if path.starts_with("enc0.") {
        Ok(Width::Base)
    } else if path.starts_with("dec_h2.") {
        Ok(Width::UpH4)
    } else if path.starts_with("head_sep.") {
        Ok(Width::Guide)
    } else if path.starts_with("enc1.") || path.starts_with("res.") || path.starts_with("dec_h4.") {
        Ok(Width::Trunk)
    } else {
        Err(format!("unclassified depthwise {path}"))
    }
}

fn pw_axes(path: &str) -> Result<(Width, Width), String> {
    let pair = if path.starts_with("stem.conv.") {
        (Width::Base, Width::All)
    } else if path.starts_with("enc0.pw.") {
        (Width::Trunk, Width::Base)
    } else if path.starts_with("enc1.pw.") || (path.starts_with("res.") && path.contains(".pw.")) {
        (Width::Trunk, Width::Trunk)
    } else if path.starts_with("dec_h4.pw.") {
        (Width::UpH4, Width::Trunk)
    } else if path.starts_with("dec_h2.pw.") {
        (Width::UpH2, Width::UpH4)
    } else if path.starts_with("h2_guide.conv.") {
        (Width::H2, Width::All)
    } else if path.starts_with("h2_guide_proj.conv.") {
        (Width::UpH2, Width::H2)
    } else if path.starts_with("to_guide.conv.") {
        (Width::Guide, Width::UpH2)
    } else if path.starts_with("guide.conv.") {
        (Width::Guide, Width::All)
    } else if path.starts_with("head_sep.pw.") {
        (Width::All, Width::Guide)
    } else if path.starts_with("film_h2.reduce.conv.") {
        (Width::Hidden, Width::UpH2)
    } else if path.starts_with("film_h2.gamma.conv.") || path.starts_with("film_h2.beta.conv.") {
        (Width::UpH2, Width::Hidden)
    } else {
        return Err(format!("unclassified convolution {path}"));
    };
    Ok(pair)
}

fn axis_indices(width: Width, dim: usize, keep: &Keep) -> Result<Vec<usize>, String> {
    if matches!(width, Width::All) {
        return Ok((0..dim).collect());
    }
    let idx = keep.indices(width);
    if idx.len() > dim || idx.iter().any(|i| *i >= dim) {
        return Err(format!(
            "keep list of {} does not fit an axis of {dim}",
            idx.len()
        ));
    }
    Ok(idx.to_vec())
}

fn slice_1d(packed: &Packed, idx: &[usize]) -> Result<Packed, String> {
    if packed.dims.len() != 1 {
        return Err(format!("expected a vector, got {:?}", packed.dims));
    }
    let mut data = Vec::with_capacity(idx.len());
    for &i in idx {
        let Some(value) = packed.data.get(i) else {
            return Err(format!("index {i} outside {}", packed.data.len()));
        };
        data.push(*value);
    }
    Ok(Packed {
        dims: vec![idx.len()],
        data,
    })
}

fn slice_4d(packed: &Packed, out_idx: &[usize], in_idx: &[usize]) -> Result<Packed, String> {
    let [_, inn, kh, kw] = four(&packed.dims)?;
    let k = kh * kw;
    let mut data = Vec::with_capacity(out_idx.len() * in_idx.len() * k);
    for &o in out_idx {
        for &i in in_idx {
            let base = (o * inn + i) * k;
            let Some(tap) = packed.data.get(base..base + k) else {
                return Err(format!("tap at out {o} in {i} is outside the weight"));
            };
            data.extend_from_slice(tap);
        }
    }
    Ok(Packed {
        dims: vec![out_idx.len(), in_idx.len(), kh, kw],
        data,
    })
}

fn four(dims: &[usize]) -> Result<[usize; 4], String> {
    dims.try_into()
        .map_err(|_| format!("expected 4 dims, got {dims:?}"))
}

fn set_diff(left: &BTreeSet<String>, right: &BTreeSet<String>) -> String {
    left.difference(right)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_keeps_the_largest_channels_in_index_order() {
        let scores = [0.1, 0.9, 0.2, 0.8];
        assert_eq!(top_k(&scores, 2).unwrap(), vec![1, 3]);
    }

    #[test]
    fn keeping_every_channel_preserves_order() {
        let scores = [0.2, 0.9, 0.1];
        assert_eq!(top_k(&scores, 3).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn slice_4d_keeps_the_selected_output_and_input() {
        let packed = Packed {
            dims: vec![2, 2, 1, 1],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let sliced = slice_4d(&packed, &[1], &[0]).unwrap();
        assert_eq!(sliced.dims, vec![1, 1, 1, 1]);
        assert_eq!(sliced.data, vec![3.0]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn folded_yc_matches_the_projected_rgb_forward() {
        use burn::backend::NdArray;
        use burn::tensor::{Tensor, TensorData};

        use super::super::{GuideSpec, GuidedStudent, PreInGain};
        use crate::StudentModel;

        type Cpu = NdArray<f32>;
        let device = Default::default();
        let spec = GuideSpec {
            base: 4,
            trunk: 4,
            res_blocks: 1,
            up_h4: 4,
            up_h2: 4,
            guide: 2,
            full: 0,
            full_layers: 1,
            shuffle: false,
            shuffle_guide: false,
            film_h2: true,
            film_guide: false,
            film_full: false,
            guide_relu: true,
            h2_guide: 2,
            guide_kernel: 3,
            guide_gate: false,
            guide_resid: false,
            trunk_h4: true,
            in_eps: 1e-5,
            trunk_norm_kind: "ln".to_string(),
            pre_in_gain: PreInGain::Sqrt2,
            input_planes: 3,
            input_quant: "none".into(),
            styles: 1,
        };
        let rgb_model = GuidedStudent::<Cpu>::new(spec, &device);
        let yc_model = rgb_model.fold_yc(&device).expect("fold");
        assert_eq!(yc_model.spec().input_planes, 2);
        assert!(yc_model.num_params() < rgb_model.num_params());

        let (h, w) = (8usize, 8usize);
        let pixels = h * w;
        let mut rgb = vec![0.0f32; 3 * pixels];
        for i in 0..pixels {
            rgb[i] = (i % 5) as f32 / 5.0;
            rgb[pixels + i] = (i % 3) as f32 / 3.0;
            rgb[2 * pixels + i] = (i % 7) as f32 / 7.0;
        }
        let m = super::super::RGB_FROM_YCBCR;
        let f = super::super::YCBCR_FROM_RGB;
        let theta = 135.0_f64.to_radians();
        let (u0, u1) = (theta.cos(), theta.sin());
        let mut projected = vec![0.0f32; 3 * pixels];
        for i in 0..pixels {
            let (r, g, b) = (
                rgb[i] as f64,
                rgb[pixels + i] as f64,
                rgb[2 * pixels + i] as f64,
            );
            let y = f[0][0] * r + f[0][1] * g + f[0][2] * b;
            let cb = f[1][0] * r + f[1][1] * g + f[1][2] * b;
            let cr = f[2][0] * r + f[2][1] * g + f[2][2] * b;
            let c = (cr - cb) * std::f64::consts::FRAC_1_SQRT_2;
            let ycc = [y, u0 * c, u1 * c];
            for channel in 0..3 {
                let value =
                    m[channel][0] * ycc[0] + m[channel][1] * ycc[1] + m[channel][2] * ycc[2];
                projected[channel * pixels + i] = value as f32;
            }
        }
        let rgb_hat =
            Tensor::<Cpu, 4>::from_data(TensorData::new(projected, [1, 3, h, w]), &device);
        let original = Tensor::<Cpu, 4>::from_data(TensorData::new(rgb, [1, 3, h, w]), &device);
        let from_rgb = rgb_model
            .stylize(rgb_hat)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let from_yc = yc_model
            .stylize(original)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gap = from_rgb
            .iter()
            .zip(&from_yc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(gap < 1e-4, "folded forward drifted by {gap}");
    }
}
