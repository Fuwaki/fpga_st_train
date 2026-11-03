use anyhow::Result;

use fpga_st_train::backend::select_device;
use fpga_st_train::deploy;

use crate::cli::InferArgs;

pub fn run(args: InferArgs, device_spec: &str) -> Result<()> {
    let device = select_device(device_spec);
    let (mode, engine) =
        deploy::resolve_engine(&args.model, &args.mode, args.calibration, device_spec)?;
    deploy::stylise_contents(
        &engine,
        &mode,
        &device,
        &args.content,
        &args.out,
        args.size,
        &args.ext,
        args.style,
    )
}
