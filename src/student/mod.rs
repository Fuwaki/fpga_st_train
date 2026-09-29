//! The student model contract.
//!
//! # Integrating your own model
//!
//! There is exactly one integration point. Implement [StudentModel] for your network and
//! point build_student in main.rs at it:
//!
//! 1. add your model to this module,
//! 2. implement [StudentModel] (one method: content in, stylised out),
//! 3. change the one line in main.rs marked REPLACE ME.
//!
//! Nothing else in the framework needs to know about your architecture: the loss, the
//! optimiser, the scheduler, checkpointing, sampling and evaluation all go through this trait.

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

    /// Forward pass with low-bit fake quantisation of activations and weights.
    ///
    /// `scheme` carries the bit widths and one symmetric activation scale per point of
    /// [Self::quant_points]. The default ignores it and runs in fp32, which is what a model
    /// without a quantisation-aware path should do.
    fn forward_quantised(
        &self,
        content: Tensor<B, 4>,
        _scheme: &crate::quant::QuantScheme,
    ) -> Tensor<B, 4> {
        self.stylize(content)
    }

    /// Names of this model's quantised activation points, in consumption order.
    ///
    /// An empty vector means the model does not support int8 quantisation.
    fn quant_points(&self) -> Vec<String> {
        Vec::new()
    }

    /// Report the largest absolute activation at every quantised point of one batch.
    ///
    /// This is the calibration pass: run it on a few real content crops and the reported
    /// maxima become the int8 activation scales. The callback form (rather than a returned
    /// vector) keeps the device readback inside the model, one point at a time.
    fn observe_activations(&self, _content: Tensor<B, 4>, _observe: &mut dyn FnMut(usize, f32)) {}

    /// Hook after a checkpoint is loaded (and also at the start of a fresh run).
    fn after_load(&mut self) {}
}
