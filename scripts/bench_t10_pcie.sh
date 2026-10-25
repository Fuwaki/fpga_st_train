#!/usr/bin/env bash
# Historical vulkan PCIe bench. Default training is cuda on d0; this script
# stays on vulkan and wgpu's discrete-adapter index (d1 = T10 when Vega is d0).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin:${PATH}"
BIN=./target/release/fpga_st_train
cargo build --release --no-default-features --features vulkan
"$BIN" --device d1 train --config configs/bench_t10_pcie.json
