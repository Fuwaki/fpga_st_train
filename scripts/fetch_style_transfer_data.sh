#!/usr/bin/env bash
# Style + demo progress images only (formal training content = HF `data.hf`, see style-transfer/README.md).
#
#   scripts/fetch_style_transfer_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(cd .. && pwd)"
ST="$ROOT/style-transfer"
UA="Mozilla/5.0 (compatible; fpga_st_train/1.0)"

mkdir -p "$ST/data/styles/vangogh" "$ST/demo_inputs"

echo "== style: Van Gogh Starry Night"
curl -fsSL -A "$UA" -o "$ST/data/styles/vangogh/starry_night.jpg" \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/960px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

echo "== demo_inputs (6 gallery images, 3:2 @ 768x512)"
demos=(animal cat city landscape portrait water)
for name in "${demos[@]}"; do
  curl -fsSL -o "$ST/demo_inputs/${name}.jpg" \
    "https://picsum.photos/seed/fpga_demo_${name}/768/512"
done
rm -f "$ST/demo_inputs"/demo_*.png 2>/dev/null || true

python3 - <<PY
from pathlib import Path
from PIL import Image
import sys

root = Path("${ST}")
for p in [root / "data/styles/vangogh/starry_night.jpg"] + sorted((root / "demo_inputs").glob("*.jpg")):
    im = Image.open(p).convert("RGB")
    if all(a == b for a, b in im.getextrema()):
        print(f"flat image: {p}", file=sys.stderr)
        sys.exit(1)
print("style +", len(list((root / "demo_inputs").glob("*.jpg"))), "demos ok")
PY

echo "done -> $ST"
