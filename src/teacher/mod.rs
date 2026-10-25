//! Teacher networks.
//!
//! The teacher is only ever used to *produce supervision*: either by stylising a
//! content dataset into pseudo ground truth offline, or by supplying features during
//! training. It sits behind a small trait so another teacher can be dropped in
//! without touching the training loop.

pub mod microast;
pub mod onnx;
pub mod stylizer;
pub mod weights;

pub use microast::MicroAst;
pub use onnx::OnnxTeacher;
pub use stylizer::Stylizer;

use anyhow::Result;
use burn::tensor::Tensor;

use crate::backend::InferBackend;

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

/// Online teacher used by the training loss.
///
/// Content and output are `[N, 3, H, W]` RGB in `[0, 1]`, the same layout
/// `rgb_batch_to_tensor` uploads. Feature maps are optional: MicroAST fills them,
/// an ONNX teacher leaves the vector empty.
pub trait DistillTeacher {
    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Result<Tensor<InferBackend, 4>>;

    /// Stylised image plus decoder feature maps. The default has no features.
    fn features(
        &self,
        content: Tensor<InferBackend, 4>,
    ) -> Result<(Tensor<InferBackend, 4>, Vec<Tensor<InferBackend, 4>>)> {
        Ok((self.stylize(content)?, Vec::new()))
    }

    /// Whether [Self::features] returns decoder maps the student can match.
    fn has_features(&self) -> bool {
        false
    }
}
