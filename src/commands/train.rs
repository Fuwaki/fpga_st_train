use std::path::Path;

use anyhow::Result;

use fpga_st_train::config::Config;
use fpga_st_train::registry;
use fpga_st_train::run_log::RunProvenance;

use crate::cli::TrainArgs;

pub fn run(args: TrainArgs, device_spec: &str) -> Result<()> {
    let mut config = Config::load(&args.config)?;
    let base = args
        .config
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    config.rebase(&base);
    if let Some(iterations) = args.iterations {
        config.optim.iterations = iterations;
    }
    if let Some(dir) = args.output_dir {
        config.run.output_dir = dir;
    }
    if device_spec != "default" {
        config.run.device = device_spec.to_owned();
    }
    let provenance = RunProvenance {
        config_path: Some(args.config),
    };
    registry::train(config, provenance)
}
