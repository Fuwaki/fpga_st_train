//! Loading the released MicroAST checkpoints into the Burn model.
//!
//! Upstream ships four torch.save files:
//!
//!   content_encoder_iter_160000.pth.tar
//!   style_encoder_iter_160000.pth.tar
//!   modulator_iter_160000.pth.tar
//!   decoder_iter_160000.pth.tar
//!
//! They are read straight out of the PyTorch pickle/tar container by burn-store --
//! no conversion step, no intermediate format.
//!
//! # Why a hand-written loader
//!
//! Burn's module paths and PyTorch's state-dict paths do not line up: PyTorch keeps
//! nn.Sequential slots for the ReLUs (enc1.0, enc1.2, enc1.4, ...) while the Burn model
//! stores only the convs in a Vec. Rather than depend on fragile regex remapping, the
//! mapping is spelled out here, and [Loader::finish] asserts that *every* tensor in the
//! checkpoint was consumed. A silently-dropped tensor is the classic "it loads fine but
//! the output is garbage" bug.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::prelude::*;
use burn_store::pytorch::PytorchReader;

use crate::teacher::microast::{ConvLayer, Decoder, Encoder, MicroAst, Modulator, ResidualLayer};

/// PyTorch nn.Sequential indices of the encoder's five convolutions.
const ENC_CONV_SLOTS: [usize; 5] = [0, 2, 4, 6, 8];

/// PyTorch nn.Sequential indices of the decoder's five convolutions (dec3).
const DEC_CONV_SLOTS: [usize; 5] = [1, 3, 6, 8, 10];

struct Loader<'src, B: Backend> {
    reader: &'src PytorchReader,
    device: &'src B::Device,
    used: Vec<String>,
}

impl<'src, B: Backend> Loader<'src, B> {
    fn new(reader: &'src PytorchReader, device: &'src B::Device) -> Self {
        Self {
            reader,
            device,
            used: Vec::new(),
        }
    }

    fn tensor<const D: usize>(&mut self, name: &str) -> Result<Tensor<B, D>> {
        let snapshot = self
            .reader
            .get(name)
            .with_context(|| format!("checkpoint has no tensor named {name}"))?;
        let data = snapshot
            .to_data()
            .with_context(|| format!("could not decode tensor {name}"))?;
        self.used.push(name.to_owned());
        Ok(Tensor::from_data(data, self.device))
    }

    fn conv(&mut self, layer: &mut ConvLayer<B>, prefix: &str) -> Result<()> {
        let weight: Tensor<B, 4> = self.tensor(&format!("{prefix}.weight"))?;
        let bias: Tensor<B, 1> = self.tensor(&format!("{prefix}.bias"))?;
        layer.conv.weight = Param::from_tensor(weight);
        layer.conv.bias = Some(Param::from_tensor(bias));
        Ok(())
    }

    fn residual(&mut self, layer: &mut ResidualLayer<B>, prefix: &str) -> Result<()> {
        self.conv(&mut layer.conv1, &format!("{prefix}.conv1.conv_layer"))?;
        self.conv(&mut layer.conv2, &format!("{prefix}.conv2.conv_layer"))?;
        Ok(())
    }

    /// Fail loudly if the checkpoint and the model disagree in either direction.
    fn finish(self, what: &str) -> Result<()> {
        let mut available = self.reader.keys();
        available.sort();
        let mut consumed = self.used;
        consumed.sort();
        consumed.dedup();

        let never_loaded: Vec<&String> =
            available.iter().filter(|k| !consumed.contains(k)).collect();
        let asked_but_absent: Vec<&String> =
            consumed.iter().filter(|k| !available.contains(k)).collect();

        if !never_loaded.is_empty() || !asked_but_absent.is_empty() {
            bail!(
                "{what}: checkpoint/model mismatch\n  \
                 in checkpoint but never loaded: {never_loaded:?}\n  \
                 requested but not in checkpoint: {asked_but_absent:?}"
            );
        }
        Ok(())
    }
}

fn open(dir: &Path, name: &str) -> Result<PytorchReader> {
    let path = dir.join(format!("{name}_iter_160000.pth.tar"));
    PytorchReader::new(&path)
        .with_context(|| format!("failed to read MicroAST checkpoint {}", path.display()))
}

fn load_encoder<B: Backend>(
    enc: &mut Encoder<B>,
    reader: &PytorchReader,
    device: &B::Device,
) -> Result<()> {
    let mut loader = Loader::<B>::new(reader, device);
    for (slot, py_index) in ENC_CONV_SLOTS.into_iter().enumerate() {
        loader.conv(&mut enc.convs[slot], &format!("enc1.{py_index}.conv_layer"))?;
    }
    loader.residual(&mut enc.res, "enc1.10")?;
    loader.residual(&mut enc.res2, "enc2.0")?;
    loader.finish("encoder")
}

fn load_modulator<B: Backend>(
    modu: &mut Modulator<B>,
    reader: &PytorchReader,
    device: &B::Device,
) -> Result<()> {
    let mut loader = Loader::<B>::new(reader, device);
    loader.conv(&mut modu.weight1, "weight1.0.conv_layer")?;
    loader.conv(&mut modu.bias1, "bias1.0.conv_layer")?;
    loader.conv(&mut modu.weight2, "weight2.0.conv_layer")?;
    loader.conv(&mut modu.bias2, "bias2.0.conv_layer")?;
    loader.finish("modulator")
}

fn load_decoder<B: Backend>(
    dec: &mut Decoder<B>,
    reader: &PytorchReader,
    device: &B::Device,
) -> Result<()> {
    let mut loader = Loader::<B>::new(reader, device);
    loader.residual(&mut dec.dec1, "dec1")?;
    loader.residual(&mut dec.dec2, "dec2")?;
    for (slot, py_index) in DEC_CONV_SLOTS.into_iter().enumerate() {
        loader.conv(&mut dec.up[slot], &format!("dec3.{py_index}.conv_layer"))?;
    }
    loader.finish("decoder")
}

/// Load the four released checkpoints from a directory into an existing model.
pub fn load_into<B: Backend>(
    model: &mut MicroAst<B>,
    dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<()> {
    let dir = dir.as_ref();
    let reader = open(dir, "content_encoder")?;
    load_encoder(&mut model.content_encoder, &reader, device)?;
    let reader = open(dir, "style_encoder")?;
    load_encoder(&mut model.style_encoder, &reader, device)?;
    let reader = open(dir, "modulator")?;
    load_modulator(&mut model.modulator, &reader, device)?;
    let reader = open(dir, "decoder")?;
    load_decoder(&mut model.decoder, &reader, device)?;
    Ok(())
}

/// Build a MicroAST model with the released weights loaded.
pub fn load<B: Backend>(dir: impl AsRef<Path>, device: &B::Device) -> Result<MicroAst<B>> {
    let mut model = MicroAst::<B>::new(device);
    load_into(&mut model, dir, device)?;
    Ok(model)
}
