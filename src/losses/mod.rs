//! Loss terms.
//!
//! The framework currently ships the terms that need no extra weights:
//!
//! - [pixel] -- L1/L2 between the student output and the teacher target
//! - [gradient_l1] -- pixel-aligned finite-difference L1
//! - [stroke_l1] -- band-pass L1 at the scales a small decoder can actually paint
//! - [texture_energy] -- phase-insensitive local amplitude of those bands
//! - [tv]    -- total-variation regulariser, which is what keeps the output smooth instead
//!   of speckled (the failure mode small style nets hit)
//!
//! Perceptual (VGG relu3_3) and Gram style terms are the natural next addition; they
//! slot in here and are enabled by LossConfig::perceptual / LossConfig::style once
//! vgg_weights is set.

use burn::prelude::*;
use burn::tensor::module::{avg_pool2d, max_pool2d};
use burn::tensor::ops::PadMode;

/// Mean absolute error.
pub fn l1<B: Backend>(prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    (prediction - target).abs().mean()
}

/// Mean squared error.
pub fn l2<B: Backend>(prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let diff = prediction - target;
    (diff.clone() * diff).mean()
}

/// L1 distance between finite-difference gradients ("gradient-domain L1").
///
/// Plain pixel L1 is minimised by the per-pixel conditional mean, so on brushy,
/// high-frequency targets (ink strokes, impasto) it systematically under-shoots the
/// texture and returns a blurred picture. Matching the *derivatives* as well removes that
/// bias: a wrong high-frequency detail is penalised in proportion to its amplitude rather
/// than being averaged away with everything else in the window.
pub fn gradient_l1<B: Backend>(prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let [_, _, h, w] = prediction.dims();
    let dx =
        |x: Tensor<B, 4>| x.clone().slice(s![.., .., .., 1..w]) - x.slice(s![.., .., .., 0..w - 1]);
    let dy =
        |x: Tensor<B, 4>| x.clone().slice(s![.., .., 1..h, ..]) - x.slice(s![.., .., 0..h - 1, ..]);
    let gx = (dx(prediction.clone()) - dx(target.clone())).abs().mean();
    let gy = (dy(prediction) - dy(target)).abs().mean();
    gx + gy
}

/// Per-channel gradient magnitude on the (H-1)×(W-1) interior.
///
/// `dx` and `dy` are the same forward differences as [`gradient_l1`], cropped to the
/// overlap so they can be combined. `sqrt(dx² + dy² + 1e-6)` is at least `1e-3`.
fn grad_mag<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    if h < 2 || w < 2 {
        return Tensor::zeros([n, c, 1, 1], &x.device());
    }
    let dx = x.clone().slice(s![.., .., 0..h - 1, 1..w])
        - x.clone().slice(s![.., .., 0..h - 1, 0..w - 1]);
    let dy = x.clone().slice(s![.., .., 1..h, 0..w - 1]) - x.slice(s![.., .., 0..h - 1, 0..w - 1]);
    (dx.clone() * dx + dy.clone() * dy + 1e-6).sqrt()
}

/// Extra student gradient where the teacher is flat.
///
/// `w = exp(−g(t) / mean g(t))` puts the weight on the teacher's quiet pixels.
/// `relu(g(s) − g(t))` only charges gradient the student has and the teacher does not,
/// so a student that is already flatter than the teacher pays nothing.
pub fn flat<B: Backend>(student: Tensor<B, 4>, teacher: Tensor<B, 4>) -> Tensor<B, 1> {
    let gs = grad_mag(student);
    let gt = grad_mag(teacher);
    // Host scalar: the weight depends only on the teacher, and a tensor divide
    // would not broadcast a rank-1 mean onto the map.
    let scale = (gt.clone().mean().into_scalar().to_f64() as f32).max(1e-6);
    let w = gt.clone().div_scalar(scale).neg().exp();
    let excess = (gs - gt).clamp_min(0.0);
    (w.clone() * excess).mean() / w.mean().clamp_min(1e-12)
}

/// Missing student gradient on teacher edges, weighted by the teacher's edge strength.
///
/// A 3×3 max-pool on `g(s)` accepts a stroke that landed one pixel to the side.
pub fn edge<B: Backend>(student: Tensor<B, 4>, teacher: Tensor<B, 4>) -> Tensor<B, 1> {
    let gs = grad_mag(student);
    let gt = grad_mag(teacher);
    let covered = max_pool2d(gs, [3, 3], [1, 1], [1, 1], [1, 1], false);
    let gap = (gt.clone() - covered).clamp_min(0.0);
    let scale = (gt.clone().mean().into_scalar().to_f64() as f32).max(1e-12);
    (gt * gap).mean().div_scalar(scale)
}

/// 16×16 non-overlapping window mean, plus a small variance term.
///
/// `mean|μ_s − μ_t| + 0.05 · mean|σ²_s − σ²_t| / max(mean σ²_t, 1e-4)`.
/// The variance scale is the teacher's own window variance, floored so a flat
/// teacher does not blow the term up. Windowing matches [`local_var`].
pub fn contrast<B: Backend>(student: Tensor<B, 4>, teacher: Tensor<B, 4>) -> Tensor<B, 1> {
    let (mean_s, var_s) = local_stats(student, 16);
    let (mean_t, var_t) = local_stats(teacher, 16);
    let tone = (mean_s - mean_t).abs().mean();
    let scale = (var_t.clone().mean().into_scalar().to_f64() as f32).max(1e-4);
    let spread = (var_s - var_t).abs().mean().div_scalar(scale);
    tone + spread.mul_scalar(0.05)
}

/// Odd-sized box blur. Reflect padding keeps a constant image constant, so a pure
/// tint has zero band-pass.
fn box_blur<B: Backend>(x: Tensor<B, 4>, kernel: usize) -> Tensor<B, 4> {
    debug_assert!(kernel >= 1 && kernel % 2 == 1);
    if kernel == 1 {
        return x;
    }
    let pad = kernel / 2;
    let x = x.pad([(pad, pad), (pad, pad)], PadMode::Reflect);
    avg_pool2d(x, [kernel, kernel], [1, 1], [0, 0], true, false)
}

/// L1 between box-filter band-passes. `fine` and `coarse` are odd kernel sizes.
fn bandpass_l1<B: Backend>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    fine: usize,
    coarse: usize,
) -> Tensor<B, 1> {
    let band = |x: Tensor<B, 4>| box_blur(x.clone(), fine) - box_blur(x, coarse);
    (band(prediction) - band(target)).abs().mean()
}

/// Band-pass L1 at the scales a depthwise H/4 trunk can actually move.
///
/// On a Van Gogh teacher crop the high-pass correlates for about 4–8 px, and the
/// trunk only places detail on a 4 px grid. Projecting the teacher onto that grid
/// (tone plus the 4 px block-mean of the residual) reduces the loss by:
///
/// - full-resolution L1: ~39%
/// - signed gradient L1: ~24% — mostly irreducible, which is why `vangogh_dw`
///   sat at grad 0.063 from step 100 to step 1000 while L1 fell 0.30 → 0.08
/// - 5–17 px band: ~66%
/// - 9–25 px band: ~78%
///
/// Gradient L1 is the term this student cannot pay down, and with weight 2 it is
/// the majority of the scalar loss, so Adam's second moment tracks a direction
/// that never shrinks. These two bands are the stroke body (not the global tint,
/// and not the 1 px residue) and a coarse student can reduce them. DC is left to
/// pixel L1: a pure tint has zero band-pass.
pub fn stroke_l1<B: Backend>(prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let dab = bandpass_l1(prediction.clone(), target.clone(), 5, 17);
    let swirl = bandpass_l1(prediction, target, 9, 25);
    dab + swirl
}

/// Local variance of a band, over non-overlapping windows.
///
/// A window wider than the band holds at least one stroke, so a phase slip barely
/// moves the number. The reduction is a reshape plus `mean_dims`, the same op the
/// instance-norm path already trains with. A 32-wide `avg_pool2d` is a different
/// kernel and is not used here.
fn local_stats<B: Backend>(x: Tensor<B, 4>, window: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [n, c, h, w] = x.dims();
    let h2 = (h / window) * window;
    let w2 = (w / window) * window;
    if h2 == 0 || w2 == 0 {
        let zeros = Tensor::zeros([n, c, 1, 1], &x.device());
        return (zeros.clone(), zeros);
    }
    let x = if h2 == h && w2 == w {
        x
    } else {
        x.slice(s![.., .., 0..h2, 0..w2])
    };
    let nh = h2 / window;
    let nw = w2 / window;
    let x = x.reshape([n, c, nh, window, nw, window]);
    let mean = x.clone().mean_dims(&[3, 5]);
    let mean_sq = (x.clone() * x).mean_dims(&[3, 5]);
    let var = (mean_sq - mean.clone() * mean.clone()).clamp_min(0.0);
    (mean.reshape([n, c, nh, nw]), var.reshape([n, c, nh, nw]))
}

fn local_var<B: Backend>(x: Tensor<B, 4>, window: usize) -> Tensor<B, 4> {
    local_stats(x, window).1
}

/// What one fully missing band adds to the loss. Same order as `stroke_l1`
/// on a real crop, so `loss.texture = 1` stays on that scale.
const MISSING_BAND: f32 = 0.05;

/// How far the student is below the teacher's band variance.
///
/// RMS (`sqrt(var + 1e-8)`) has derivative `1/(2 sqrt(var))`. On a quiet window
/// that is ~1e4, and every texture run drove the picture to black. Variance has
/// no such pole. Extra energy is ignored: stroke and pixel L1 already punish it,
/// and pushing energy down is what collapsed the weights. The teacher scale is
/// read before the student reduction so the sync does not land inside that graph.
fn local_deficit<B: Backend>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    window: usize,
) -> Tensor<B, 1> {
    let teacher = local_var(target, window);
    let scale = (teacher.clone().mean().into_scalar().to_f64() as f32).max(1e-4);
    let student = local_var(prediction, window);
    (teacher - student)
        .clamp_min(0.0)
        .mean()
        .mul_scalar(MISSING_BAND / scale)
}

/// Band energy, compared inside a window and not at a pixel.
///
/// 4–8 px uses a 16 px window (two cycles). 8–16 px uses a 32 px window.
/// Same box bands as the sample metric: blur(5)−blur(9) and blur(9)−blur(17).
/// A shifted stroke that keeps its width and density is cheap; a missing stroke is not.
pub fn texture_energy<B: Backend>(prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let band = |x: Tensor<B, 4>, fine: usize, coarse: usize| {
        box_blur(x.clone(), fine) - box_blur(x, coarse)
    };
    let fine = local_deficit(
        band(prediction.clone(), 5, 9),
        band(target.clone(), 5, 9),
        16,
    );
    let coarse = local_deficit(band(prediction, 9, 17), band(target, 9, 17), 32);
    fine + coarse
}

/// Total variation: mean absolute horizontal and vertical gradient.
///
/// This is the cheap regulariser that suppresses the 2px checkerboard artefact that
/// nearest-neighbour upsampling plus instance norm tends to produce.
pub fn total_variation<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 1> {
    let [_, _, h, w] = x.dims();
    let dx = (x.clone().slice(s![.., .., .., 1..w]) - x.clone().slice(s![.., .., .., 0..w - 1]))
        .abs()
        .mean();
    let dy = (x.clone().slice(s![.., .., 1..h, ..]) - x.clone().slice(s![.., .., 0..h - 1, ..]))
        .abs()
        .mean();
    dx + dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn image(data: Vec<f32>, h: usize, w: usize) -> Tensor<B, 4> {
        Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 1, h, w]),
            &Default::default(),
        )
    }

    fn scalar(loss: Tensor<B, 1>) -> f32 {
        loss.into_scalar().to_f64() as f32
    }

    fn constant(h: usize, w: usize, value: f32) -> Vec<f32> {
        vec![value; h * w]
    }

    fn checker(h: usize, w: usize, amp: f32) -> Vec<f32> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| if (x + y).is_multiple_of(2) { amp } else { -amp }))
            .collect()
    }

    /// Sine along x. `period` is the cycle length in pixels.
    fn sine(h: usize, w: usize, period: f32, phase: f32, amp: f32) -> Vec<f32> {
        (0..h * w)
            .map(|i| {
                let x = (i % w) as f32;
                amp * (std::f32::consts::TAU * (x + phase) / period).sin()
            })
            .collect()
    }

    #[test]
    fn stroke_ignores_tint_and_penalises_dabs_more_than_speckle() {
        let h = 64;
        let w = 64;
        let flat = image(constant(h, w, 0.0), h, w);
        let tint = image(constant(h, w, 0.4), h, w);
        let speckle = image(checker(h, w, 0.4), h, w);
        // Period 24 sits in the 9–25 px band. A 2 px slip is a small fraction of that cycle.
        let dab = image(sine(h, w, 24.0, 0.0, 0.4), h, w);
        let shifted = image(sine(h, w, 24.0, 2.0, 0.4), h, w);

        let tone = scalar(stroke_l1(tint, flat.clone()));
        let noise = scalar(stroke_l1(speckle, flat.clone()));
        let stroke = scalar(stroke_l1(dab.clone(), flat));
        let slip = scalar(stroke_l1(shifted, dab));

        assert!(
            tone < 1e-4,
            "a pure tint should be invisible to stroke_l1, got {tone}"
        );
        assert!(
            stroke > noise * 2.0,
            "a 24px wave ({stroke}) should cost more than 1px speckle ({noise})"
        );
        assert!(
            slip < stroke * 0.75,
            "a 2px slip ({slip}) should cost less than missing the wave ({stroke})"
        );
    }

    #[test]
    fn texture_energy_ignores_a_phase_slip() {
        let h = 64;
        let w = 64;
        let flat = image(constant(h, w, 0.2), h, w);
        let tint = image(constant(h, w, 0.6), h, w);
        // Period 6 sits in the 4–8 px band. A 2 px slip is a third of a cycle.
        let wave = image(sine(h, w, 6.0, 0.0, 0.4), h, w);
        let shifted = image(sine(h, w, 6.0, 2.0, 0.4), h, w);

        let tone = scalar(texture_energy(tint, flat.clone()));
        let miss = scalar(texture_energy(flat.clone(), wave.clone()));
        let extra = scalar(texture_energy(wave.clone(), flat));
        let slip = scalar(texture_energy(shifted, wave));

        assert!(
            tone < 1e-3,
            "a pure tint should be invisible to texture energy, got {tone}"
        );
        assert!(
            miss > 1e-3,
            "dropping a 6 px wave should cost something, got {miss}"
        );
        assert!(
            extra < 1e-3,
            "extra energy is not this term's job, got {extra}"
        );
        assert!(
            slip < miss * 0.25,
            "a 2 px slip ({slip}) should cost much less than missing the wave ({miss})"
        );
    }

    #[test]
    fn texture_energy_grad_stays_bounded_on_a_quiet_image() {
        use burn::backend::Autodiff;
        type AD = Autodiff<NdArray>;
        let h = 64;
        let w = 64;
        let device = Default::default();
        // A little speckle, not an exact constant: that is the window where
        // `1/(2 sqrt(var))` used to explode.
        let mut data = constant(h, w, 0.3);
        for (i, v) in data.iter_mut().enumerate() {
            *v += if i.is_multiple_of(2) { 0.01 } else { -0.01 };
        }
        let pred =
            Tensor::<AD, 4>::from_data(burn::tensor::TensorData::new(data, [1, 1, h, w]), &device)
                .require_grad();
        let target = Tensor::<AD, 4>::from_data(
            burn::tensor::TensorData::new(sine(h, w, 6.0, 0.0, 0.4), [1, 1, h, w]),
            &device,
        );
        let loss = texture_energy(pred.clone(), target);
        let grads = loss.backward();
        let g = pred.grad(&grads).expect("texture grad");
        let peak = g.abs().max().into_scalar().to_f64() as f32;
        assert!(peak < 1.0, "quiet-image texture grad peaked at {peak}");
    }

    #[test]
    fn texture_energy_ignores_brightness() {
        use burn::backend::Autodiff;
        type AD = Autodiff<NdArray>;
        let h = 64;
        let w = 64;
        let device = Default::default();
        let wave: Vec<f32> = sine(h, w, 6.0, 0.0, 0.4)
            .into_iter()
            .map(|v| v + 0.5)
            .collect();
        let target =
            Tensor::<AD, 4>::from_data(burn::tensor::TensorData::new(wave, [1, 1, h, w]), &device);
        let bright = Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(constant(h, w, 0.8), [1, 1, h, w]),
            &device,
        );
        let dark = Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(constant(h, w, 0.02), [1, 1, h, w]),
            &device,
        );
        let target_b = Tensor::<B, 4>::from_data(target.to_data(), &device);
        let bright_loss = scalar(texture_energy(bright, target_b.clone()));
        let dark_loss = scalar(texture_energy(dark, target_b));
        assert!(
            (bright_loss - dark_loss).abs() < 1e-4,
            "brightness leaked into texture loss: bright {bright_loss} dark {dark_loss}"
        );

        let pred = Tensor::<AD, 4>::from_data(
            burn::tensor::TensorData::new(constant(h, w, 0.5), [1, 1, h, w]),
            &device,
        )
        .require_grad();
        let loss = texture_energy(pred.clone(), target);
        let grads = loss.backward();
        let g = pred.grad(&grads).expect("grad");
        let mean = g.clone().mean().into_scalar().to_f64() as f32;
        let peak = g.abs().max().into_scalar().to_f64() as f32;
        assert!(
            mean.abs() < 1e-4 && peak < 1e-3,
            "flat tint should get no texture grad, mean {mean} peak {peak}"
        );
    }

    fn step_at(h: usize, w: usize, col: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; h * w];
        for y in 0..h {
            for x in col..w {
                data[y * w + x] = 1.0;
            }
        }
        data
    }

    #[test]
    fn look_flat_charges_texture_in_a_flat_region_only() {
        let h = 32;
        let w = 32;
        let quiet = image(constant(h, w, 0.4), h, w);
        let busy = image(step_at(h, w, w / 2), h, w);
        let extra = scalar(flat(busy.clone(), quiet.clone()));
        let already = scalar(flat(quiet, busy));
        assert!(
            extra > 1e-3,
            "texture on a flat teacher should cost, got {extra}"
        );
        assert!(
            already < 1e-4,
            "a student flatter than the teacher should be free, got {already}"
        );
    }

    #[test]
    fn look_edge_accepts_a_one_pixel_shift() {
        let h = 32;
        let w = 32;
        let teacher = image(step_at(h, w, 16), h, w);
        let aligned = scalar(edge(image(step_at(h, w, 16), h, w), teacher.clone()));
        let shifted = scalar(edge(image(step_at(h, w, 17), h, w), teacher.clone()));
        let missing = scalar(edge(image(constant(h, w, 0.2), h, w), teacher));
        assert!(
            aligned < 1e-3,
            "a matching edge should be free, got {aligned}"
        );
        assert!(
            shifted < missing * 0.25,
            "a 1 px shift ({shifted}) should cost much less than a missing edge ({missing})"
        );
        assert!(missing > 1e-2, "a missing edge should cost, got {missing}");
    }

    #[test]
    fn look_contrast_tracks_window_tone() {
        let h = 32;
        let w = 32;
        let teacher = image(constant(h, w, 0.4), h, w);
        let same = scalar(contrast(image(constant(h, w, 0.4), h, w), teacher.clone()));
        let lifted = scalar(contrast(image(constant(h, w, 0.6), h, w), teacher));
        assert!(same < 1e-4, "identical windows should match, got {same}");
        assert!(
            (lifted - 0.2).abs() < 1e-3,
            "a 0.2 mean shift should cost about 0.2, got {lifted}"
        );
    }
}
