#!/usr/bin/env bash
#
# Assemble deliverables/ from finished trial directories (tiny_dw v6+).
#
#   deliverables/README.md
#   deliverables/fp32/<name>/model.{json,mpk}
#   deliverables/w8a8|w4a8/<name>/manifest.json + weights.bin + ...
#   deliverables/gallery/
#
# Default packages (override with PACKAGES="name:trial_dir ..."):
#   vangogh          film_both (sep int8 path)
#   vangogh_lut_dw   lut_dw_b4 (LUT depthwise + same decoder)
#
# Run scripts/ptq.sh on each trial for w8a8 and w4a8 first.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/cuda_env.sh
bin="${FPGA_ST_BIN:-./target/release/fpga_st_train}"
device="${DEVICE:-d0}"
size="${SIZE:-384}"
content="${CONTENT:-../style-transfer/demo_inputs}"

if [ -z "${PACKAGES:-}" ]; then
    PACKAGES="vangogh:experiments/tiny_dw_vangogh_film/trials/film_both"
    PACKAGES="$PACKAGES vangogh_lut_dw:experiments/tiny_dw_vangogh_lut_dw/trials/lut_dw_b4"
fi

rm -rf deliverables
mkdir -p deliverables

teacher_style="${TEACHER_STYLE:-../style-transfer/data/styles/vangogh/starry_night.jpg}"

for entry in ${PACKAGES}; do
    name="${entry%%:*}"
    trial="${entry#*:}"
    if [ ! -f "${trial}/model.mpk" ]; then
        echo "missing ${trial}/model.mpk" >&2
        exit 1
    fi

    echo "== package ${name} from ${trial}"
    mkdir -p "deliverables/fp32/${name}"
    cp "${trial}/model.json" "${trial}/model.mpk" "deliverables/fp32/${name}/"

    for bits in w8a8 w4a8; do
        from="${trial}/quant/${bits}"
        if [ ! -f "${from}/manifest.json" ]; then
            echo "skip ${name}/${bits}: no ${from}/manifest.json (run scripts/ptq.sh ${trial} ${bits})" >&2
            continue
        fi
        mkdir -p "deliverables/${bits}/${name}"
        cp "${from}/model.json" "${from}/manifest.json" "${from}/weights.bin" \
           "${from}/calibration.json" "${from}/metrics.json" "deliverables/${bits}/${name}/"
    done

    cache="deliverables/gallery/_cache/${name}"
    mkdir -p "${cache}/fp32" "${cache}/teacher"
    if [ -z "$(ls -A "${cache}/teacher" 2>/dev/null || true)" ]; then
        echo "== teacher renders for ${name}"
        "${bin}" --device "${device}" stylize \
            --teacher ../teacher-eval/MicroAST/models \
            --content "${content}" \
            --style "${teacher_style}" \
            --out "${cache}/teacher" \
            --content-size "${size}" \
            --style-size 256
    fi
    if [ -z "$(ls -A "${cache}/fp32" 2>/dev/null || true)" ]; then
        echo "== fp32 student renders for ${name}"
        "${bin}" --device "${device}" infer \
            --model "deliverables/fp32/${name}" \
            --content "${content}" \
            --out "${cache}/fp32" \
            --mode fp32 \
            --size "${size}"
    fi
    for bits in w8a8 w4a8; do
        quant="deliverables/${bits}/${name}"
        if [ ! -f "${quant}/manifest.json" ]; then
            continue
        fi
        mkdir -p "${cache}/${bits}"
        trial_int="${trial}/quant/${bits}/eval/int"
        if [ -d "${trial_int}" ] && [ -n "$(ls -A "${trial_int}"/*.png 2>/dev/null || true)" ]; then
            mkdir -p "${cache}/${bits}"
            cp "${trial_int}"/*.png "${cache}/${bits}/"
        elif [ -z "$(ls -A "${cache}/${bits}" 2>/dev/null || true)" ]; then
            echo "== integer reference ${bits} for ${name}"
            "${bin}" --device "${device}" infer \
                --model "${quant}" \
                --content "${content}" \
                --out "${cache}/${bits}" \
                --mode int \
                --size "${size}"
        fi
    done
done

gallery_styles=""
for entry in ${PACKAGES}; do
    gallery_styles="${gallery_styles} ${entry%%:*}"
done
GALLERY_STYLES="${gallery_styles}" python3 scripts/gallery.py

if [ -f docs/deliverables-vangogh-dw.md ]; then
    cp docs/deliverables-vangogh-dw.md deliverables/README.md
else
    cp docs/deliverables.md deliverables/README.md
fi

echo
echo "deliverables:"
find deliverables -maxdepth 3 -type f | sort | sed "s|^|  |"
du -sh deliverables
