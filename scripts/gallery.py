#!/usr/bin/env python3
"""Build the visual comparison that ships with the deliverables.

Rows are content images, one sheet per package name, columns are: the content, the MicroAST
teacher, the fp32 student, and each exported variant run by the integer reference interpreter.
"""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DELIVERABLES = ROOT / "deliverables"
CACHE = DELIVERABLES / "gallery" / "_cache"
OUT = DELIVERABLES / "gallery"
STYLES = os.environ.get("GALLERY_STYLES", "vangogh").split()

TILE = 240
GAP = 6
HEADER = 30

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
]
CONTENT = ROOT / "style-transfer" / "demo_inputs"


def font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def find(directory, stem):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.iterdir()):
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if candidate.stem == stem or candidate.stem.split("__")[0] == stem:
            return candidate
    return None


def tile(path, size):
    if path is None:
        return Image.new("RGB", (size, size), (32, 32, 36))
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (18, 18, 20))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def load_metrics(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def columns_for(style):
    base = CACHE / style
    cols = [
        ("content", CONTENT),
        ("MicroAST teacher", base / "teacher"),
        ("student fp32", base / "fp32"),
    ]
    quant_metrics = []
    for bits in ("w8a8", "w4a8"):
        quant_dir = DELIVERABLES / bits / style
        if not (quant_dir / "manifest.json").is_file():
            continue
        cols.append(("student " + bits.upper(), base / bits))
        quant_metrics.append(load_metrics(quant_dir / "metrics.json"))
    return cols, quant_metrics


def sheet(style):
    cols, quant_metrics = columns_for(style)
    render_dir = CACHE / style / "fp32"
    if not render_dir.is_dir():
        raise SystemExit("no renders under {}; run scripts/package.sh first".format(render_dir))
    stems = sorted(p.stem for p in render_dir.glob("*.png"))

    width = len(cols) * TILE + (len(cols) + 1) * GAP
    height = HEADER + len(stems) * (TILE + GAP) + GAP
    out = Image.new("RGB", (width, height), (12, 12, 14))
    draw = ImageDraw.Draw(out)
    label_font = font(15)
    header_font = font(17)

    for c, (name, _directory) in enumerate(cols):
        x = GAP + c * (TILE + GAP)
        draw.text((x + 4, 7), name, fill=(235, 235, 240), font=header_font)

    metrics_by_column = {}
    for offset, metrics in enumerate(quant_metrics):
        metrics_by_column[len(cols) - len(quant_metrics) + offset] = metrics

    for r, stem in enumerate(stems):
        y = HEADER + GAP // 2 + r * (TILE + GAP)
        for c, (_name, directory) in enumerate(cols):
            x = GAP + c * (TILE + GAP)
            path = find(directory, stem)
            out.paste(tile(path, TILE), (x, y))
            metrics = metrics_by_column.get(c)
            if metrics is None or path is None:
                continue
            value = metrics.get("per_image", {}).get(path.name, {}).get("fp32_vs_int_db")
            if value is None:
                continue
            draw.rectangle([x + 3, y + 3, x + 76, y + 23], fill=(0, 0, 0))
            draw.text(
                (x + 7, y + 5),
                "{:.1f} dB".format(value),
                fill=(255, 235, 120),
                font=label_font,
            )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for style in STYLES:
        if not (CACHE / style / "fp32").is_dir():
            print("skip gallery sheet {}: no cache".format(style))
            continue
        path = OUT / (style + ".png")
        image = sheet(style)
        image.save(path)
        made.append(path)
        print("wrote {} ({}x{})".format(path, image.width, image.height))

    if not made:
        raise SystemExit("no gallery sheets produced")

    images = [Image.open(p) for p in made]
    width = max(i.width for i in images)
    height = sum(i.height for i in images) + GAP * (len(images) - 1)
    combined = Image.new("RGB", (width, height), (12, 12, 14))
    y = 0
    for image in images:
        combined.paste(image, (0, y))
        y += image.height + GAP
    overview = OUT / "overview.png"
    combined.save(overview)
    print("wrote {} ({}x{})".format(overview, combined.width, combined.height))

    rows = "".join(
        '<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>'.format(
            name=p.name
        )
        for p in made + [overview]
    )
    (OUT / "index.html").write_text(
        "<!doctype html>\n<meta charset=utf-8>\n<title>fpga_st_train gallery</title>\n"
        "<style>body{background:#0c0c0e;color:#e8e8ee;font:14px/1.5 system-ui,sans-serif;"
        "margin:24px}figure{margin:0 0 28px}img{max-width:100%;border:1px solid #2a2a30;"
        "display:block}figcaption{color:#9aa;padding-top:6px}</style>\n"
        "<h1>fpga_st_train &mdash; quantisation gallery</h1>\n"
        "<p>Rows are demo content images. Columns: content, MicroAST teacher, fp32 student,"
        " W8A8 / W4A8 integer reference. Tile labels are PSNR vs fp32 student.</p>\n" + rows
    )
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
