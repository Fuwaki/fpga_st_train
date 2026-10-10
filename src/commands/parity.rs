use std::path::Path;

use anyhow::{Result, bail};
use burn::prelude::*;

use fpga_st_train::backend::select_device;
use fpga_st_train::data;
use fpga_st_train::teacher::Teacher;
use fpga_st_train::teacher::microast::MicroAst;
use fpga_st_train::teacher::weights;
use fpga_st_train::InferBackend;

use crate::cli::ParityArgs;

pub fn run(args: ParityArgs, device_spec: &str) -> Result<()> {
    let device = select_device(device_spec);
    let model: MicroAst<InferBackend> = weights::load(&args.teacher, &device)?;
    let content = tensor_from_raw::<InferBackend>(&args.content_raw, &device)?;
    let style = tensor_from_raw::<InferBackend>(&args.style_raw, &device)?;
    let output = model.stylize(content, style, args.alpha);
    let (shape, values) = data::tensor_to_f32(output)?;
    data::save_f32(&args.out_raw, &shape, &values)?;
    println!("wrote {} with shape {shape:?}", args.out_raw.display());
    Ok(())
}

fn tensor_from_raw<B: Backend>(path: &Path, device: &B::Device) -> Result<Tensor<B, 4>> {
    let (shape, values) = data::load_f32(path)?;
    if shape.len() != 4 {
        bail!("{}: expected a 4D shape, got {shape:?}", path.display());
    }
    let data = TensorData::new(values, [shape[0], shape[1], shape[2], shape[3]]);
    Ok(Tensor::from_data(data, device))
}
