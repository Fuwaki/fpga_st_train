use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use fpga_st_train::config::Config;
use fpga_st_train::registry;
use fpga_st_train::run_log::RunProvenance;

use crate::cli::TrainArgs;

pub fn run(args: TrainArgs, device_spec: &str) -> Result<()> {
    if args.output_dir.is_some() && !args.with.is_empty() {
        bail!("--output-dir is for a single run; each --with config keeps its own output_dir");
    }
    let mut paths = vec![args.config.clone()];
    paths.extend(args.with.iter().cloned());
    let mut configs = Vec::with_capacity(paths.len());
    let mut provenances = Vec::with_capacity(paths.len());
    for path in paths {
        configs.push(load_config(&path, &args, device_spec)?);
        provenances.push(RunProvenance {
            config_path: Some(path),
        });
    }
    if configs.len() == 1 {
        registry::train(configs.remove(0), provenances.remove(0))
    } else {
        registry::train_cohort(configs, provenances)
    }
}

fn load_config(path: &Path, args: &TrainArgs, device_spec: &str) -> Result<Config> {
    let mut config = Config::load(path)?;
    let base = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    config.rebase(&base);
    let teacher_weights = args
        .teacher_weights
        .clone()
        .or_else(|| std::env::var_os("TEACHER_WEIGHTS_DIR").map(PathBuf::from));
    if let Some(dir) = teacher_weights {
        config.teacher.weights_dir = if dir.is_relative() {
            base.join(dir)
        } else {
            dir
        };
    }
    if let Some(iterations) = args.iterations {
        config.optim.iterations = iterations;
    }
    if let Some(dir) = &args.output_dir {
        config.run.output_dir = dir.clone();
    }
    if device_spec != "default" {
        config.run.device = device_spec.to_owned();
    }
    Ok(config)
}
