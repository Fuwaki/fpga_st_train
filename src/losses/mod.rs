//! Loss terms.
//!
//! The framework currently ships the terms that need no extra weights:
//!
//! - [pixel] -- L1/L2 between the student output and the teacher target
//! - [tv]    -- total-variation regulariser, which is what keeps the output smooth instead
//!   of speckled (the failure mode small style nets hit)
//!
//! Perceptual (VGG relu3_3) and Gram style terms are the natural next addition; they
//! slot in here and are enabled by LossConfig::perceptual / LossConfig::style once
//! vgg_weights is set.

use burn::prelude::*;

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
