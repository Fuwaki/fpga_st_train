#!/usr/bin/env bash
#
# Assemble the deliverable tree from the finished runs, and rebuild the gallery.
#
#   deliverables/README.md        handover document: what each file is, how to run it
#   deliverables/fp32/<style>/    the trained fp32 student
#   deliverables/w8a8/<style>/    8-bit weights, 8-bit activations
#   deliverables/w4a8/<style>/    4-bit weights, 8-bit activations
#   deliverables/gallery/         rendered comparison of every variant
#
# Run scripts/train.sh and scripts/ptq.sh for both styles first.
set -euo pipefail
cd "$(dirname "$0")/.."
bin="${FPGA_ST_BIN:-./target/release/fpga_st_train}"
styles="${STYLES:-ink vangogh}"
device="${DEVICE:-d0}"
size="${SIZE:-384}"
content="${CONTENT:-../style-transfer/demo_inputs}"

rm -rf deliverables
mkdir -p deliverables

for style in ${styles}; do
    src="runs/${style}"
    if [ ! -f "${src}/model.mpk" ]; then
        echo "missing ${src}/model.mpk -- run scripts/train.sh ${style}" >&2
        exit 1
    fi

    mkdir -p "deliverables/fp32/${style}"
    cp "${src}/model.json" "${src}/model.mpk" "deliverables/fp32/${style}/"

    for bits in w8a8 w4a8; do
        from="${src}/quant/${bits}"
        if [ ! -f "${from}/manifest.json" ]; then
            echo "missing ${from}/manifest.json -- run scripts/ptq.sh ${src} ${bits}" >&2
            exit 1
        fi
        mkdir -p "deliverables/${bits}/${style}"
        cp "${from}/model.json" "${from}/manifest.json" "${from}/weights.bin" \
           "${from}/calibration.json" "${from}/metrics.json" "deliverables/${bits}/${style}/"
    done

    # Teacher renders: gallery material only, never shipped as a deliverable.
    teacher="${src}/teacher"
    if [ -z "$(ls -A "${teacher}" 2>/dev/null || true)" ]; then
        style_image="$(python3 -c "import json,sys; print(json.load(open('configs/${style}.json'))['teacher']['style'])")"
        echo "== teacher renders for ${style}"
        "${bin}" --device "${device}" stylize \
            --teacher ../teacher-eval/MicroAST/models \
            --content "${content}" \
            --style "configs/${style_image}" \
            --out "${teacher}" \
            --content-size "${size}" \
            --style-size 256
    fi
done

python3 scripts/gallery.py

cp docs/deliverables.md deliverables/README.md
echo
echo "deliverables:"
find deliverables -maxdepth 3 -type f | sort | sed "s|^|  |"
du -sh deliverables
