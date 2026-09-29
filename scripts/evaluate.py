#!/usr/bin/env python3
"""Measure what quantisation cost, for one model directory and one export.

Every export is rendered three ways over the demo set:

  fp32  the trained model, unquantised              (the reference)
  sim   the same model with fake quantisation       (what calibration optimised)
  int   the exported integer graph, run by the      (what the FPGA will execute)
        reference interpreter in src/intref.rs

int-vs-fp32 is the honest end-to-end number for the accelerator. sim-vs-int is a consistency
check: it has to be far above int-vs-fp32, otherwise the manifest and the training code
disagree about the same graph.

Usage:
  scripts/evaluate.py --model runs/ink --quant runs/ink/quant/w4a8 [--content DIR] [--size N]
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/fpga_st_train"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def render(model, mode, out, content, size, device, calibration=None):
    """Run the inference tool once; stderr is left alone so progress shows up live."""
    cmd = [BIN, "infer", "--model", model, "--content", content, "--out", out,
           "--mode", mode, "--size", str(size)]
    if mode != "int":
        cmd += ["--device", device]
    if calibration is not None:
        cmd += ["--calibration", calibration]
    print("  $", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def load_dir(directory):
    return {p.name: np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
            for p in sorted(Path(directory).glob("*.png"))}


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--model", required=True, help="run directory (model.json + model.mpk)")
    parser.add_argument("--quant", required=True, help="export directory to evaluate")
    parser.add_argument("--content", default=str(ROOT.parent / "style-transfer" / "demo_inputs"))
    parser.add_argument("--device", default="d0")
    parser.add_argument("--size", type=int, default=384, help="long edge of the demo renders")
    args = parser.parse_args()

    model = Path(args.model)
    quant = Path(args.quant)
    fp32_dir = model / "eval" / "fp32"
    sim_dir = quant / "eval" / "sim"
    int_dir = quant / "eval" / "int"
    for directory in (fp32_dir, sim_dir, int_dir):
        directory.mkdir(parents=True, exist_ok=True)

    expected = sum(1 for p in Path(args.content).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(list(fp32_dir.glob("*.png"))) != expected:
        print("fp32 reference")
        render(model, "fp32", fp32_dir, args.content, args.size, args.device)
    else:
        print("fp32 reference: reusing", fp32_dir)
    print("fake-quantised simulation")
    render(model, "sim", sim_dir, args.content, args.size, args.device, quant / "calibration.json")
    print("integer reference")
    # The integer engine reads the *export* directory (manifest.json + weights.bin), not the
    # run directory: that is the whole point of the export being self-contained.
    render(quant, "int", int_dir, args.content, args.size, args.device)

    fp32 = load_dir(fp32_dir)
    sim = load_dir(sim_dir)
    fixed = load_dir(int_dir)
    names = sorted(set(fp32) & set(sim) & set(fixed))
    if not names:
        raise SystemExit("no matching renders; did any of the three passes fail?")

    per_image = {}
    for name in names:
        # PSNR alone is not comparable across styles: a starry-night render has roughly twice
        # the contrast of an ink render, so the same relative error costs it several dB. The
        # error in 1/255 steps, and that error divided by the signal's own standard deviation,
        # are the two numbers that do compare.
        diff = fixed[name] - fp32[name]
        rmse = float(np.sqrt((diff ** 2).mean()))
        signal = float(fp32[name].std())
        per_image[name] = {
            "fp32_vs_sim_db": psnr(sim[name], fp32[name]),
            "fp32_vs_int_db": psnr(fixed[name], fp32[name]),
            "sim_vs_int_db": psnr(fixed[name], sim[name]),
            "rmse_lsb": rmse,
            "rmse_over_signal_std": rmse / signal if signal > 0 else 0.0,
            "max_abs_error_lsb": float(np.abs(diff).max()),
        }

    def mean(key):
        return sum(v[key] for v in per_image.values()) / len(per_image)

    manifest = json.loads((quant / "manifest.json").read_text())
    metrics = {
        "model": str(model),
        "quant": str(quant),
        "arch": manifest["arch"],
        "bits": manifest["bits"],
        "weight_packing": manifest["weight_packing"],
        "weight_bytes": manifest["weight_bytes"],
        "calibration_batches": manifest.get("calibration_batches"),
        "content_images": len(names),
        "size": args.size,
        "mean": {key: mean(key) for key in
                 ("fp32_vs_sim_db", "fp32_vs_int_db", "sim_vs_int_db",
                  "rmse_lsb", "rmse_over_signal_std", "max_abs_error_lsb")},
        "per_image": per_image,
    }
    (quant / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    bits = metrics["bits"]
    label = "W{}A{}".format(bits["weights"], bits["acts"])
    print()
    print("{}  {}  ({} images at long edge {})".format(label, quant, len(names), args.size))
    print("  {:<18}{:>10}{:>10}{:>10}{:>12}{:>12}".format(
        "image", "sim/fp32", "int/fp32", "int/sim", "rmse/LSB", "rmse/sigma"))
    for name in names:
        row = per_image[name]
        print("  {:<18}{:>10.2f}{:>10.2f}{:>10.2f}{:>12.2f}{:>11.1%}".format(
            name, row["fp32_vs_sim_db"], row["fp32_vs_int_db"], row["sim_vs_int_db"],
            row["rmse_lsb"], row["rmse_over_signal_std"]))
    print("  {:<18}{:>10.2f}{:>10.2f}{:>10.2f}{:>12.2f}{:>11.1%}".format(
        "MEAN", metrics["mean"]["fp32_vs_sim_db"], metrics["mean"]["fp32_vs_int_db"],
        metrics["mean"]["sim_vs_int_db"], metrics["mean"]["rmse_lsb"],
        metrics["mean"]["rmse_over_signal_std"]))
    print("  weights {} bytes ({})".format(
        metrics["weight_bytes"], metrics["weight_packing"]))


if __name__ == "__main__":
    main()
