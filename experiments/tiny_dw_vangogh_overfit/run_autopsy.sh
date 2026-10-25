#!/usr/bin/env bash
# 1000-step autopsy of the V1 cliff. One teacher forward, three guided students.
# h2_24: V2 head, H/2 width 24. full16: V2 width, 16ch full-res layer, FiLM off.
# full16_film: that same layer with FiLM on.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh

device="${1:-d0}"
logdir="experiments/tiny_dw_vangogh_overfit/log"
cfg="experiments/tiny_dw_vangogh_overfit/configs"
mkdir -p "$logdir"

cargo run --release --offline --no-default-features --features cuda -- \
  --device "${device}" train \
  --config "${cfg}/h2_24.json" \
  --with "${cfg}/full16.json" \
  --with "${cfg}/full16_film.json" \
  2>&1 | tee "${logdir}/autopsy.log"
