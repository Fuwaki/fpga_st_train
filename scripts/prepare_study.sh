#!/usr/bin/env bash
# One-shot setup before scripts/study_run.sh (data, configs, release build, optional HF warmup).
#
#   scripts/prepare_study.sh [study_id] [device]
#
# Env:
#   SKIP_BUILD=1       — skip `cargo build --release`
#   WARMUP_ITER=0      — skip short train smoke (default 2)
#   WARMUP_TRIAL=b32_rb4
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh

study_id="${1:-tiny_dw_vangogh_capacity}"
device="${2:-d0}"
warmup_iter="${WARMUP_ITER:-2}"
warmup_trial="${WARMUP_TRIAL:-b32_rb4}"

echo "== style + demo assets"
bash scripts/fetch_style_transfer_data.sh

echo "== study configs: ${study_id}"
python scripts/study_gen.py "experiments/${study_id}"

teacher_dir="$(cd .. && pwd)/teacher-eval/MicroAST/models"
if [ ! -f "${teacher_dir}/decoder_iter_160000.pth.tar" ]; then
  echo "missing teacher weights under ${teacher_dir}" >&2
  echo "  set TEACHER_WEIGHTS_DIR=... or restore MicroAST models (see experiments/${study_id}/README.md)" >&2
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path
ROOT = Path(".").resolve()
cfg = ROOT / "experiments/${study_id}/configs/${warmup_trial}.json"
data = json.loads(cfg.read_text())
base = cfg.parent
def resolve(p):
    return (base / p).resolve()
for key, rel in [
    ("teacher.weights_dir", data["teacher"]["weights_dir"]),
    ("teacher.style", data["teacher"]["style"]),
    ("run.sample_content", data["run"]["sample_content"]),
]:
    path = resolve(rel)
    if not path.exists():
        raise SystemExit(f"missing {key}: {path}")
print("paths ok for trial ${warmup_trial}")
print("  teacher:", resolve(data["teacher"]["weights_dir"]))
print("  hf repo:", data["data"].get("hf"))
PY

if [ "${SKIP_BUILD:-}" != "1" ]; then
  echo "== release build (cuda)"
  cargo build --release --no-default-features --features cuda
fi

if [ "${warmup_iter}" != "0" ]; then
  cfg="experiments/${study_id}/configs/${warmup_trial}.json"
  echo "== HF + train smoke: ${warmup_trial} iterations=${warmup_iter} device=${device}"
  echo "   (first sample builds full shard index — downloads all COCO parquet into ~/.cache/huggingface)"
  cargo run --release --no-default-features --features cuda -- \
    --device "${device}" train --config "${cfg}" --iterations "${warmup_iter}"
  trial_dir="experiments/${study_id}/trials/${warmup_trial}"
  if [ -f "${trial_dir}/metrics.jsonl" ]; then
    echo "smoke ok: ${trial_dir}/metrics.jsonl"
    echo "removing partial trial dir (full study_run will recreate)"
    rm -rf "${trial_dir}"
  fi
fi

echo ""
echo "Ready. Full study:"
echo "  cd $(pwd)"
echo "  scripts/study_run.sh ${study_id} ${device}"
echo "Optional: TEACHER_WEIGHTS_DIR=...  ITERATIONS=... (override optim.iterations)"
