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

pub mod tiny;
pub mod tiny_dw;

use burn::module::Module;
use burn::prelude::*;

pub use tiny::TinyStudent;
pub use tiny_dw::TinyDwStudent;

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
}
