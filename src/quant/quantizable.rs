//! Contract for models that support calibration, QAT, and integer export.

use burn::prelude::*;

use crate::quant::QuantScheme;

/// Low-bit fake-quant forward, activation profiling, and export layout naming.
pub trait Quantizable<B: Backend> {
    /// Quantised activation points in consumption order (`points[0]` is the image).
    fn quant_points(&self) -> Vec<String>;

    /// Fake-quantised forward used during QAT and `sim` inference.
    fn forward_quantised(&self, content: Tensor<B, 4>, scheme: &QuantScheme) -> Tensor<B, 4>;

    /// Measure per-point activation magnitudes on one batch (calibration pass).
    fn observe_activations(&self, content: Tensor<B, 4>, observe: &mut dyn FnMut(usize, f32));

    fn supports_quant(&self) -> bool {
        !self.quant_points().is_empty()
    }
}
