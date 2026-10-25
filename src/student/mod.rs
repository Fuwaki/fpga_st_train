//! The student model contract.
//!
//! # Integrating your own model
//!
//! There is exactly one integration point. Implement [StudentModel] for your network and
//! register the arch in [crate::registry]:
//!
//! 1. add your model to this module,
//! 2. implement [StudentModel] (one method: content in, stylised out),
//! 3. register the arch string in [crate::registry].
//!
//! Quantisation and integer export go through [crate::quant::Quantizable] and
//! [crate::export::Exportable] instead.

mod conv;
pub mod guided;
pub mod lut;
pub(crate) mod plugin;
pub mod tiny;
pub mod tiny_dw;

use burn::module::Module;
use burn::prelude::*;

pub use lut::{LutConv2d, LutDense, LutEncoder};
pub use tiny::TinyStudent;
pub use tiny_dw::TinyDwStudent;

/// Sample-time health of the low-resolution trunk. Absent for students that
/// have no paint / InstanceNorm split.
#[derive(Clone, Copy, Debug)]
pub struct SampleSignal {
    /// Share of output spatial variance that disappears when the upsampled
    /// paint is replaced by its per-channel spatial mean. 0 means the output
    /// does not move when the paint's layout changes.
    pub paint_spatial: f32,
    /// Fraction of channels entering InstanceNorm whose spatial variance is
    /// below `1e-5`.
    pub in_dead: f32,
}

/// What the training framework needs from a student model.
pub trait StudentModel<B: Backend>: Module<B> {
    /// Map a content batch [N, 3, H, W] in [0, 1] to a stylised batch of the same shape.
    fn stylize(&self, content: Tensor<B, 4>) -> Tensor<B, 4>;

    /// One-line description, printed once at startup.
    fn describe(&self) -> String {
        let name = std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("model");
        format!("{name} ({} parameters)", self.num_params())
    }

    /// Forward pass that also returns intermediate feature maps, in the same order as
    /// [crate::teacher::Stylizer::features] so feature-level distillation can pair them up.
    ///
    /// The default implementation returns no features, which disables feature distillation
    /// for that model.
    fn intermediate(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        (self.stylize(content), Vec::new())
    }

    /// Hook after a checkpoint is loaded (and also at the start of a fresh run).
    fn after_load(&mut self) {}

    /// Trunk health on one content image. The default is no measurement.
    fn sample_signal(&self, _content: Tensor<B, 4>) -> Option<SampleSignal> {
        None
    }
}
