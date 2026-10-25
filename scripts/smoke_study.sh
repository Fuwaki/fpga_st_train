#!/usr/bin/env bash
# Quick check: generate configs, resolve paths, optional short train.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh

study="${1:-tiny_dw_vangogh_capacity}"
trial="${2:-b32_rb3}"

python scripts/study_gen.py "experiments/${study}"

cfg="experiments/${study}/configs/${trial}.json"
python3 - <<PY
import json
from pathlib import Path
ROOT = Path(".").resolve()
cfg = ROOT / "${cfg}"
data = json.loads(cfg.read_text())
base = cfg.parent
content = (base / data["data"]["content_dir"]).resolve()
print("config:", cfg)
print("content_dir resolves:", content)
if not content.is_dir():
    raise SystemExit("content_dir missing (expected on your training machine)")
PY

iter="${SMOKE_ITERATIONS:-50}"
echo "smoke train ${trial} iterations=${iter}"
cargo run --release --no-default-features --features cuda -- \
  --device "${DEVICE:-d0}" train --config "${cfg}" --iterations "${iter}"

test -f "experiments/${study}/trials/${trial}/metrics.jsonl" && echo "ok: metrics.jsonl"
