//! Loss terms.
//!
//! The framework currently ships the terms that need no extra weights:
//!
//! - [pixel] -- L1/L2 between the student output and the teacher target
//! - [gradient_l1] -- pixel-aligned finite-difference L1
//! - [stroke_l1] -- band-pass L1 at the scales a small decoder can actually paint
//! - [tv]    -- total-variation regulariser, which is what keeps the output smooth instead
//!   of speckled (the failure mode small style nets hit)
//!
//! Perceptual (VGG relu3_3) and Gram style terms are the natural next addition; they
//! slot in here and are enabled by LossConfig::perceptual / LossConfig::style once
//! vgg_weights is set.

use burn::prelude::*;
use burn::tensor::module::avg_pool2d;
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
            .flat_map(|y| {
                (0..w).map(move |x| if (x + y).is_multiple_of(2) { amp } else { -amp })
            })
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

        assert!(tone < 1e-4, "a pure tint should be invisible to stroke_l1, got {tone}");
        assert!(
            stroke > noise * 2.0,
            "a 24px wave ({stroke}) should cost more than 1px speckle ({noise})"
        );
        assert!(
            slip < stroke * 0.75,
            "a 2px slip ({slip}) should cost less than missing the wave ({stroke})"
        );
    }
}
