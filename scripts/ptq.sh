#!/usr/bin/env bash
#
# Post-training quantisation of one model directory: calibrate the activation ranges, export
# the integer graph, then measure what the quantisation actually cost.
#
#   scripts/ptq.sh <model-dir> <w8a8|w4a8> [safety] [device] [content-dir]
#
# Writes <model-dir>/quant/<bits>/ and leaves it ready for scripts/package.sh. The export is
# run twice afterwards (simulation and integer reference) so an export bug shows up here and
# not in the deliverable.
set -euo pipefail
cd "$(dirname "$0")/.."
bin="${FPGA_ST_BIN:-./target/release/fpga_st_train}"

model="${1:?usage: scripts/ptq.sh <model-dir> <w8a8|w4a8> [safety] [device] [content-dir]}"
bits="${2:?}"
safety="${3:-1.0}"
device="${4:-d0}"
content="${5:-../style-transfer/data/content}"

out="${model}/quant/${bits}"
mkdir -p "${out}"

echo "== calibrate ${bits} on ${model} (safety ${safety}, content ${content})"
"${bin}" --device "${device}" calibrate \
    --model "${model}" \
    --content "${content}" \
    --batches 20 \
    --safety "${safety}" \
    --bits "${bits}" \
    --out "${out}/calibration.json"

echo "== export ${bits} to ${out}"
"${bin}" --device "${device}" export-int8 \
    --model "${model}" \
    --calibration "${out}/calibration.json" \
    --out "${out}"

echo "== evaluate ${bits}"
python3 scripts/evaluate.py --model "${model}" --quant "${out}" --device "${device}"
