//! A teacher with the style branch precomputed.
//!
//! MicroAST is a *two-input* network: style image -> style encoder -> modulator ->
//! (weights, biases), and the decoder consumes those modulation vectors. When the style is
//! fixed -- which is exactly the case for the fixed-style student trained here -- all of
//! that is a constant.
//!
//! Running the style encoder on every training step is not free: the style is a 256x513
//! hanging scroll, i.e. 2x the pixels of a 256x256 training crop, so recomputing it would
//! roughly double the per-step teacher cost. This wrapper computes the style features and
//! modulation once and reuses them, leaving only content-encoder + decoder per call
//! (the ~1.82 GMACs @256^2 the evaluation report measured).

use burn::tensor::Tensor;

use crate::backend::InferBackend;
use crate::teacher::microast::MicroAst;

/// MicroAST with a fixed style baked into cached modulation vectors.
pub struct Stylizer {
    model: MicroAst<InferBackend>,
    style_feats: Vec<Tensor<InferBackend, 4>>,
    weights: Vec<Tensor<InferBackend, 4>>,
    biases: Vec<Tensor<InferBackend, 4>>,
}

impl Stylizer {
    /// Precompute the style branch.
    pub fn new(model: MicroAst<InferBackend>, style: Tensor<InferBackend, 4>) -> Self {
        let style_feats = model.style_encoder.forward(style);
        let (weights, biases) = model.modulator.forward(&style_feats);
        Self {
            model,
            style_feats,
            weights,
            biases,
        }
    }

    /// Stylise a content batch. alpha is fixed at 1.0: the released checkpoints only
    /// behave at 1.0 (0.5 collapses to a flat colour -- see teacher-eval/REPORT.md 8.4).
    pub fn stylize(&self, content: Tensor<InferBackend, 4>) -> Tensor<InferBackend, 4> {
        let content_feats = self.model.content_encoder.forward(content);
        self.model.decoder.forward(
            &content_feats,
            &self.style_feats,
            &self.weights,
            &self.biases,
            1.0,
        )
    }

    /// Access to the underlying network (e.g. for an unmodulated reference pass).
    pub fn model(&self) -> &MicroAst<InferBackend> {
        &self.model
    }

    /// Forward pass that also returns the decoder's intermediate feature maps.
    ///
    /// These are the "hints" used by feature-level distillation: the student keeps the
    /// teacher's channel plan on purpose (64 @1/4, 32 @1/2, 16 @1), so every returned map
    /// has a student counterpart of exactly the same shape. The full-resolution 16-channel
    /// map is the one that already carries the brush texture, one convolution away from RGB.
    ///
    /// Order: [trunk (64 @1/4), half (32 @1/2), full (16 @1)].
    pub fn features(
        &self,
        content: Tensor<InferBackend, 4>,
    ) -> (Tensor<InferBackend, 4>, Vec<Tensor<InferBackend, 4>>) {
        use crate::teacher::microast::{adaptive_instance_norm, upsample2x};
        use burn::tensor::activation::relu;

        let decoder = &self.model.decoder;
        let content_feats = self.model.content_encoder.forward(content);

        // Exactly Decoder::forward with alpha = 1.0 (the released checkpoints only behave
        // at 1.0), stopping early to capture the intermediate maps.
        let x1 = adaptive_instance_norm(content_feats[1].clone(), self.style_feats[1].clone());
        let x2 =
            decoder
                .dec1
                .forward_modulated(x1, self.weights[1].clone(), self.biases[1].clone());
        let x3 = adaptive_instance_norm(x2, self.style_feats[0].clone());
        let x4 =
            decoder
                .dec2
                .forward_modulated(x3, self.weights[0].clone(), self.biases[0].clone());
        let trunk = x4.clone();

        let [_, _, h, w] = x4.dims();
        let mut y = upsample2x(x4, h, w);
        y = relu(decoder.up[0].forward(y));
        y = relu(decoder.up[1].forward(y));
        let half = y.clone();

        let [_, _, h, w] = y.dims();
        y = upsample2x(y, h, w);
        y = relu(decoder.up[2].forward(y));
        y = relu(decoder.up[3].forward(y));
        let full = y.clone();

        (decoder.up[4].forward(y), vec![trunk, half, full])
    }
}

impl crate::teacher::DistillTeacher for Stylizer {
    fn stylize(&self, content: Tensor<InferBackend, 4>) -> anyhow::Result<Tensor<InferBackend, 4>> {
        Ok(Self::stylize(self, content))
    }

    fn features(
        &self,
        content: Tensor<InferBackend, 4>,
    ) -> anyhow::Result<(Tensor<InferBackend, 4>, Vec<Tensor<InferBackend, 4>>)> {
        Ok(Self::features(self, content))
    }

    fn has_features(&self) -> bool {
        true
    }
}
