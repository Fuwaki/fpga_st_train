#!/usr/bin/env bash
#
# Train one style. This is the only step that needs the config file: everything downstream is
# driven by the self-describing run directory it produces.
#
#   scripts/train.sh <tag> [device]
#
# <tag> names configs/<tag>.json. The checkpoint, the progress samples and the model card go
# to runs/<tag>/, and the complete stdout is kept in runs/log/<tag>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

tag="${1:?usage: scripts/train.sh <tag> [device]}"
device="${2:-d0}"
config="configs/${tag}.json"
if [ ! -f "$config" ]; then
    echo "no such config: $config" >&2
    exit 1
fi

mkdir -p runs/log "runs/${tag}"
echo "training ${tag} from ${config} on ${device} (vulkan, log: runs/log/${tag}.log)"
cargo run --release --no-default-features --features vulkan -- --device "${device}" train --config "${config}" 2>&1 | tee "runs/log/${tag}.log"
