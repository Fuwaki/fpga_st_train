#!/usr/bin/env bash
#
# Run every config generated for a study (sequential).
#
#   scripts/study_run.sh <study_id> [device]
#   GROUP=3 scripts/study_run.sh <study_id> [device]
#
# GROUP shares one teacher forward across that many students, then backprops each.
#
# Requires: python scripts/study_gen.py experiments/<study_id>  (or existing configs/)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh

study_id="${1:?usage: scripts/study_run.sh <study_id> [device]}"
device="${2:-d0}"
# Optional: ITERATIONS=40 for smoke; SMOKE=1 also sets data.limit via patched configs (use --iterations only).
iterations="${ITERATIONS:-}"
study_dir="experiments/${study_id}"
configs="${study_dir}/configs"

if [ ! -d "$configs" ]; then
  echo "missing ${configs}; run: python scripts/study_gen.py ${study_dir}" >&2
  exit 1
fi

mkdir -p "${study_dir}/log"
teacher_extra=()
if [ -n "${TEACHER_WEIGHTS_DIR:-}" ]; then
  teacher_extra=(--teacher-weights "${TEACHER_WEIGHTS_DIR}")
fi
# GROUP=N shares one teacher forward across N students, then backprops each.
group="${GROUP:-1}"
if ! [[ "${group}" =~ ^[0-9]+$ ]] || [ "${group}" -lt 1 ]; then
  echo "GROUP must be an integer >= 1" >&2
  exit 1
fi
shopt -s nullglob
mapfile -t all_cfgs < <(printf '%s\n' "${configs}"/*.json | sort)
cfgs=()
for cfg in "${all_cfgs[@]}"; do
  trial="$(basename "$cfg" .json)"
  if [ "$trial" != "study" ] && [ -f "$cfg" ]; then
    cfgs+=("$cfg")
  fi
done

idx=0
while [ "${idx}" -lt "${#cfgs[@]}" ]; do
  primary="${cfgs[$idx]}"
  names="$(basename "$primary" .json)"
  with_args=()
  j=1
  while [ "$j" -lt "$group" ] && [ $((idx + j)) -lt "${#cfgs[@]}" ]; do
    extra_cfg="${cfgs[$((idx + j))]}"
    with_args+=(--with "$extra_cfg")
    names="${names}+$(basename "$extra_cfg" .json)"
    j=$((j + 1))
  done
  if [ "$group" -eq 1 ]; then
    log="${study_dir}/log/$(basename "$primary" .json).log"
  else
    log="${study_dir}/log/${names}.log"
  fi
  echo "=== ${names} -> ${log}"
  extra=()
  if [ -n "${iterations}" ]; then
    extra+=(--iterations "${iterations}")
  fi
  cargo run --release --no-default-features --features cuda -- \
    --device "${device}" train --config "${primary}" "${with_args[@]}" "${extra[@]}" "${teacher_extra[@]}" \
    2>&1 | tee "${log}"
  idx=$((idx + group))
done
