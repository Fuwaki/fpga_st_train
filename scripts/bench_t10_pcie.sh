#!/usr/bin/env bash
# Bench Tesla T10 (wgpu DiscreteGpu d1) with vulkan feature.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:${PATH}"
BIN=./target/release/fpga_st_train
cargo build --release --no-default-features --features vulkan
"$BIN" --device d1 train --config configs/bench_t10_pcie.json
