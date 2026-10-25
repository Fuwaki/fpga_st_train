#!/usr/bin/env bash
# One city frame, long edge 640, no random crop. film_both is the control for
# how far V1 and V2 can still move once the dataset average is gone.
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
  --config "${cfg}/film.json" \
  --with "${cfg}/v1.json" \
  --with "${cfg}/v2.json" \
  2>&1 | tee "${logdir}/overfit.log"
