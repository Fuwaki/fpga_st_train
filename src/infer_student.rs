//! Type-erased fp32 student for inference and quant simulation.
//!
//! New architectures implement [StudentModel] and are wrapped at load time — deploy code
//! does not need a new enum variant per arch.

use burn::prelude::*;

use crate::InferBackend;
use crate::quant::{QuantScheme, Quantizable};
use crate::student::StudentModel;

/// Runtime student used by [crate::deploy::Engine] (fp32 and sim modes).
pub trait InferStudent {
    fn describe(&self) -> String;
    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Tensor<InferBackend, 4>;
    fn forward_quantised(
        &self,
        content: Tensor<InferBackend, 4>,
        scheme: &QuantScheme,
    ) -> Tensor<InferBackend, 4>;
}

impl<M: StudentModel<InferBackend> + Quantizable<InferBackend> + 'static> InferStudent for M {
    fn describe(&self) -> String {
        format!("fp32 {}", StudentModel::describe(self))
    }

    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Tensor<InferBackend, 4> {
        StudentModel::stylize(self, content)
    }

    fn forward_quantised(
        &self,
        content: Tensor<InferBackend, 4>,
        scheme: &QuantScheme,
    ) -> Tensor<InferBackend, 4> {
        Quantizable::forward_quantised(self, content, scheme)
    }
}

/// Loaded `model.mpk` on the inference backend.
pub struct StudentHandle(Box<dyn InferStudent + 'static>);

impl StudentHandle {
    pub fn new<M: StudentModel<InferBackend> + Quantizable<InferBackend> + 'static>(
        model: M,
    ) -> Self {
        Self(Box::new(model))
    }
}

impl InferStudent for StudentHandle {
    fn describe(&self) -> String {
        self.0.describe()
    }

    fn stylize(&self, content: Tensor<InferBackend, 4>) -> Tensor<InferBackend, 4> {
        self.0.stylize(content)
    }

    fn forward_quantised(
        &self,
        content: Tensor<InferBackend, 4>,
        scheme: &QuantScheme,
    ) -> Tensor<InferBackend, 4> {
        self.0.forward_quantised(content, scheme)
    }
}
