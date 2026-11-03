//! Subcommand implementations for the binary.

mod calibrate;
mod export_int8;
mod fold_yc;
mod infer;
mod metrics;
mod pack_styles;
mod parity;
mod probe;
mod prune;
mod stylize;
mod train;

use anyhow::Result;

use crate::cli::Command;

pub fn dispatch(command: Command, device: &str) -> Result<()> {
    match command {
        Command::Stylize(args) => stylize::run(args, device),
        Command::InitConfig(args) => {
            fpga_st_train::config::Config::write_default(&args.out)?;
            println!("wrote {}", args.out.display());
            Ok(())
        }
        Command::Train(args) => train::run(args, device),
        Command::Prune(args) => prune::run(args, device),
        Command::FoldYc(args) => fold_yc::run(args, device),
        Command::PackStyles(args) => pack_styles::run(args, device),
        Command::SplitStyle(args) => pack_styles::split(args, device),
        Command::Infer(args) => infer::run(args, device),
        Command::Calibrate(args) => calibrate::run(args, device),
        Command::ExportInt8(args) => export_int8::run(args, device),
        Command::Metrics(args) => metrics::run(args),
        Command::Parity(args) => parity::run(args, device),
        Command::Probe(args) => probe::run(args, device),
    }
}
