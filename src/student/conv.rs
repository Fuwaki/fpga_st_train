//! Valid 2-D convolution for the student layers, written so its backward is fast on CubeCL.
//!
//! Burn 0.21's CUDA `conv2d` backward is two orders of magnitude slower than its forward
//! when the channel count is small or the convolution is grouped. At batch 32 and a 192 crop
//! on the V100 (`fwd` / `fwd+bwd`, weight gradient included):
//!
//! | layer | burn | here |
//! |---|---|---|
//! | depthwise 3×3, 8 ch @192 | 0.85 / 236 ms | 3.0 / 10 ms |
//! | depthwise 3×3, 32 ch @96 | 0.93 / 155 ms | 3.0 / 10 ms |
//! | depthwise 3×3, 32 ch, stride 2 @96 | 0.56 / 42 ms | 1.4 / 5.8 ms |
//! | 1×1, 32→48 @96 | 1.2 / 43 ms | 1.0 / 4.6 ms |
//! | dense 3×3, 3→8 @192 | 0.64 / 59 ms | 3.4 / 14 ms |
//! | dense 3×3, 64→64 @48 | 6.3 / 21 ms | (left to burn) |
//!
//! That made one `b_h4` student 1.1 s per step, 0.92 s of it in backward.
//!
//! The rewrites are the same sums in a different order, and match `conv2d` to float
//! rounding (about 4e-7 on unit-scale inputs):
//!
//! - depthwise: k² shifted (and, for stride > 1, stepped) slices, each times its
//!   per-channel tap, summed
//! - 1×1: a batched matmul `[Cout, Cin] × [Cin, H·W]`
//! - dense with few input taps (`Cin·k² ≤ DENSE_SHIFT_MAX_TAPS`): the k² slices
//!   concatenated on channels, then that 1×1 matmul
//!
//! Anything else, and any dilation, goes to burn's `conv2d`.

use burn::prelude::*;
use burn::tensor::module::conv2d;
use burn::tensor::ops::PaddedConvOptions;

/// Above this many input taps (`Cin·k²`) a dense conv stays on burn's kernel. At 64→64
/// 3×3 that kernel already backs off to about 2.4× its forward.
const DENSE_SHIFT_MAX_TAPS: usize = 128;

/// Valid convolution, `weight` is `[Cout, Cin / groups, kh, kw]`. Padding is the caller's.
pub(crate) fn conv2d_valid<B: Backend>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor<B, 4> {
    let [_, cin, _, _] = x.dims();
    let [cout, cin_per_group, kh, kw] = weight.dims();
    let y = if dilation != [1, 1] {
        return conv2d(
            x,
            weight,
            bias,
            PaddedConvOptions::asymmetric(stride, [0, 0], [0, 0], dilation, groups),
        );
    } else if groups > 1 && groups == cin && cout == cin && cin_per_group == 1 {
        depthwise(x, weight, stride)
    } else if groups == 1 && kh == 1 && kw == 1 {
        pointwise(subsample(x, stride), weight)
    } else if groups == 1 && cin * kh * kw <= DENSE_SHIFT_MAX_TAPS {
        dense(x, weight, stride)
    } else {
        return conv2d(
            x,
            weight,
            bias,
            PaddedConvOptions::asymmetric(stride, [0, 0], [0, 0], dilation, groups),
        );
    };
    match bias {
        Some(b) => y + b.reshape([1, cout, 1, 1]),
        None => y,
    }
}

/// Output size of a valid convolution along one axis.
fn out_len(len: usize, k: usize, stride: usize) -> usize {
    (len - k) / stride + 1
}

/// The `[i, j]` tap of a valid kxk window: every output position's input pixel at that offset.
fn tap<B: Backend>(
    x: Tensor<B, 4>,
    i: usize,
    j: usize,
    out: [usize; 2],
    stride: [usize; 2],
) -> Tensor<B, 4> {
    let [oh, ow] = out;
    let [sh, sw] = stride;
    let h_end = i + sh * (oh - 1) + 1;
    let w_end = j + sw * (ow - 1) + 1;
    if stride == [1, 1] {
        x.slice(s![.., .., i..h_end, j..w_end])
    } else {
        x.slice(s![.., .., i..h_end;sh, j..w_end;sw])
    }
}

fn subsample<B: Backend>(x: Tensor<B, 4>, stride: [usize; 2]) -> Tensor<B, 4> {
    if stride == [1, 1] {
        return x;
    }
    let [_, _, h, w] = x.dims();
    tap(
        x,
        0,
        0,
        [out_len(h, 1, stride[0]), out_len(w, 1, stride[1])],
        stride,
    )
}

fn depthwise<B: Backend>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    stride: [usize; 2],
) -> Tensor<B, 4> {
    let [_, c, h, w] = x.dims();
    let [_, _, kh, kw] = weight.dims();
    let out = [out_len(h, kh, stride[0]), out_len(w, kw, stride[1])];
    let mut acc: Option<Tensor<B, 4>> = None;
    for i in 0..kh {
        for j in 0..kw {
            let coeff = weight
                .clone()
                .slice(s![.., .., i..i + 1, j..j + 1])
                .reshape([1, c, 1, 1]);
            let term = tap(x.clone(), i, j, out, stride) * coeff;
            acc = Some(match acc {
                None => term,
                Some(sum) => sum + term,
            });
        }
    }
    acc.expect("kernel has at least one tap")
}

fn pointwise<B: Backend>(x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, cin, h, w] = x.dims();
    let [cout, _, _, _] = weight.dims();
    let weight = weight.reshape([1, cout, cin]).expand([n, cout, cin]);
    weight
        .matmul(x.reshape([n, cin, h * w]))
        .reshape([n, cout, h, w])
}

fn dense<B: Backend>(x: Tensor<B, 4>, weight: Tensor<B, 4>, stride: [usize; 2]) -> Tensor<B, 4> {
    let [_, cin, h, w] = x.dims();
    let [cout, _, kh, kw] = weight.dims();
    let out = [out_len(h, kh, stride[0]), out_len(w, kw, stride[1])];
    let mut taps = Vec::with_capacity(kh * kw);
    for i in 0..kh {
        for j in 0..kw {
            taps.push(tap(x.clone(), i, j, out, stride));
        }
    }
    // Channels are tap-major, so the kernel goes `[Cout, kh, kw, Cin]` to match.
    let columns = Tensor::cat(taps, 1);
    let weight = weight
        .permute([0, 2, 3, 1])
        .reshape([cout, kh * kw * cin, 1, 1]);
    pointwise(columns, weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type T = NdArray;

    fn check(cin: usize, cout: usize, k: usize, stride: usize, groups: usize, size: usize) {
        let device = Default::default();
        let x = Tensor::<T, 4>::random([2, cin, size, size], Distribution::Default, &device);
        let w = Tensor::<T, 4>::random(
            [cout, cin / groups, k, k],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<T, 1>::random([cout], Distribution::Default, &device);
        let reference = conv2d(
            x.clone(),
            w.clone(),
            Some(b.clone()),
            PaddedConvOptions::asymmetric([stride, stride], [0, 0], [0, 0], [1, 1], groups),
        );
        let ours = conv2d_valid(x, w, Some(b), [stride, stride], [1, 1], groups);
        assert_eq!(
            reference.dims(),
            ours.dims(),
            "{cin}->{cout} k{k} s{stride} g{groups}"
        );
        let diff = (reference - ours).abs().max().into_scalar().to_f64();
        assert!(
            diff < 1e-4,
            "{cin}->{cout} k{k} s{stride} g{groups}: max diff {diff}"
        );
    }

    #[test]
    fn depthwise_matches_conv2d() {
        check(8, 8, 3, 1, 8, 11);
        check(6, 6, 3, 2, 6, 12);
        check(6, 6, 3, 2, 6, 11);
        check(4, 4, 5, 1, 4, 9);
    }

    #[test]
    fn pointwise_matches_conv2d() {
        check(5, 7, 1, 1, 1, 6);
        check(5, 7, 1, 2, 1, 7);
    }

    #[test]
    fn small_dense_matches_conv2d() {
        check(3, 8, 3, 1, 1, 10);
        check(3, 5, 7, 1, 1, 12);
        check(3, 4, 3, 2, 1, 11);
    }

    #[test]
    fn wide_dense_falls_back_and_still_matches() {
        check(16, 8, 3, 1, 1, 8);
    }
}
