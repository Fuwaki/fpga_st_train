//! Teacher networks.
//!
//! The teacher is only ever used to *produce supervision*: either by stylising a
//! content dataset into pseudo ground truth offline, or by supplying features during
//! training. It sits behind a small trait so another teacher can be dropped in
//! without touching the training loop.

pub mod microast;
pub mod stylizer;
pub mod weights;

pub use microast::MicroAst;
pub use stylizer::Stylizer;

use crate::backend::InferBackend;
use burn::tensor::Tensor;

/// A teacher that maps (content, style) to a stylised image.
///
/// Implementations run on the inference backend: no autodiff tape is built.
pub trait Teacher {
    /// Stylise content with style. Both are [N, 3, H, W] in [0, 1].
    ///
    /// alpha blends the stylisation. **The released MicroAST checkpoints only
    /// behave at alpha = 1.0** -- 0.5 collapses to a flat colour, see the README.
    fn stylize(
        &self,
        content: Tensor<InferBackend, 4>,
        style: Tensor<InferBackend, 4>,
        alpha: f32,
    ) -> Tensor<InferBackend, 4>;
}
