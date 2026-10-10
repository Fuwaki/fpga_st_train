//! `fpga_st_train` binary: CLI parsing and dispatch only.
//!
//! Command bodies live under [commands]; shared model loading and training dispatch live in
//! the `fpga_st_train` library ([registry], [deploy], [train]).

mod cli;
mod commands;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = cli::Cli::parse();
    commands::dispatch(cli.command, &cli.device)
}
