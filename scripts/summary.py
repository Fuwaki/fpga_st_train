#!/usr/bin/env python3
"""Print the deliverable tables as markdown, straight from the metrics.json files.

Every number here is read back from the exports themselves (manifest.json) and from the
measurements scripts/evaluate.py wrote (metrics.json), so this table cannot disagree with the
files that ship.

PSNR is not comparable across styles -- a starry-night render has about twice the contrast of
an ink render, so the same relative error costs it several dB. The rmse column (in 1/255
steps) and rmse/sigma (that error over the signal standard deviation) are.

Usage:
  scripts/summary.py [runs]         # defaults to the runs/ directory
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parameters(manifest):
    """Conv weights plus LUT table entry count (informative, not trainable param count)."""
    total = 0
    for layer in manifest.get("layers", []):
        kh, kw = layer["kernel"]
        total += layer["out_channels"] * layer["in_channels"] * kh * kw
        total += layer["out_channels"]
    for lut in manifest.get("lut_layers", []):
        total += len(lut.get("tables", []))
    return total


def collect(runs):
    rows = []
    runs = Path(runs)
    if (runs / "w8a8").is_dir() or (runs / "w4a8").is_dir():
        for bits_dir in sorted(p for p in runs.iterdir() if p.is_dir() and p.name in ("w8a8", "w4a8")):
            for style_dir in sorted(bits_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                manifest_path = style_dir / "manifest.json"
                metrics_path = style_dir / "metrics.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text())
                metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
                rows.append((style_dir.name, bits_dir.name, manifest, metrics))
        return rows
    for style_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        quant_root = style_dir / "quant"
        if not quant_root.is_dir():
            continue
        for quant_dir in sorted(quant_root.iterdir()):
            manifest_path = quant_dir / "manifest.json"
            metrics_path = quant_dir / "metrics.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
            rows.append((style_dir.name, quant_dir.name, manifest, metrics))
    return rows


def fmt(value, spec="{:.2f}"):
    return "n/a" if value is None else spec.format(value)


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs"
    rows = collect(runs)
    if not rows:
        raise SystemExit("no <runs>/<style>/quant/<bits>/manifest.json found")

    print("| style | variant | bits | packing | bytes | parameters | int vs fp32 | rmse | rmse/sigma | int vs sim |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for style, variant, manifest, metrics in rows:
        bits = manifest["bits"]
        mean = (metrics or {}).get("mean", {})
        print("| {} | {} | W{}A{} | {} | {:,} | {:,} | {} dB | {} LSB | {} | {} dB |".format(
            style, variant, bits["weights"], bits["acts"], manifest["weight_packing"],
            manifest["weight_bytes"], parameters(manifest),
            fmt(mean.get("fp32_vs_int_db")), fmt(mean.get("rmse_lsb")),
            fmt(mean.get("rmse_over_signal_std"), "{:.1%}"),
            fmt(mean.get("sim_vs_int_db"))))


if __name__ == "__main__":
    main()
