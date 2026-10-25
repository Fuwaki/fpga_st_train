#!/usr/bin/env bash
# 1000-step injection round on the V2 skeleton (up_h2=48, full=0).
# shuffle_guide: pixel-shuffle only on H/2 → full.
# guide16: 16-channel paint, nearest upsample.
# guide_linear: signed guide, still added.
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
  --config "${cfg}/shuffle_guide.json" \
  --with "${cfg}/guide16.json" \
  --with "${cfg}/guide_linear.json" \
  2>&1 | tee "${logdir}/inject.log"
