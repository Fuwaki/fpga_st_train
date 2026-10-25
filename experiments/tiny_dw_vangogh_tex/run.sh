#!/usr/bin/env bash
# 500-step matrix. A cohort shares one MicroAST forward, then backprops each student.
#
#   film_l1 / film_stroke / film_tex     tiny_dw film_both
#   v1_l1                                guided, nearest, one full-res layer
#   v1_stroke + v1s_stroke               + stroke
#   v1_tex + v1s_tex                     + stroke + texture energy
#
# v1s is the same trunk as v1, with pixel-shuffle upsamples and two 16-ch full-res layers.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh

device="${1:-d0}"
logdir="experiments/tiny_dw_vangogh_tex/log"
cfg="experiments/tiny_dw_vangogh_tex/configs"
mkdir -p "$logdir"

run() {
  local name="$1"
  shift
  local attempt=1
  while [ "${attempt}" -le 3 ]; do
    echo "=== ${name} (attempt ${attempt})"
    if cargo run --release --offline --no-default-features --features cuda -- \
      --device "${device}" train "$@" \
      2>&1 | tee "${logdir}/${name}.log"; then
      return 0
    fi
    echo "=== ${name} failed, retrying" >&2
    attempt=$((attempt + 1))
    sleep 15
  done
  return 1
}

skip_if_done() {
  local name="$1"
  shift
  local pending=0
  local path trial
  for path in "$@"; do
    trial="$(basename "${path}" .json)"
    if [ ! -f "experiments/tiny_dw_vangogh_tex/trials/${trial}/model.mpk" ]; then
      pending=1
    fi
  done
  if [ "${pending}" -eq 0 ]; then
    echo "=== skip ${name} (model.mpk already present)"
    return 0
  fi
  local primary="$1"
  shift
  local args=(--config "${primary}")
  local extra
  for extra in "$@"; do
    args+=(--with "${extra}")
  done
  run "${name}" "${args[@]}"
}

# One teacher forward per loss column. film_l1 already finished, and that column
# has no third student, so v1_l1 stays a single run.
skip_if_done film_l1 "${cfg}/film_l1.json"
skip_if_done stroke "${cfg}/film_stroke.json" "${cfg}/v1_stroke.json" "${cfg}/v1s_stroke.json"
skip_if_done tex "${cfg}/film_tex.json" "${cfg}/v1_tex.json" "${cfg}/v1s_tex.json"
skip_if_done v1_l1 "${cfg}/v1_l1.json"
