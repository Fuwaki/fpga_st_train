#!/usr/bin/env python3
"""Fold an exported model's 3-channel stem onto a 2-plane (Y + one chroma axis) input.

The deployed graph is unchanged: same 3-channel stem, same graph order, same `intref`.
Only `stem`'s weights, bias and per-channel scales are rewritten so that

    conv(W, RGB)  ==  conv(W2, [Y, C, 0]) + bias2

where `Y` is BT.601 luma and `C` is the input chroma (Cb,Cr) projected onto one axis,
stored as a byte with a +128 offset (`C_byte = round(C + 128)`). Dropping the third input
plane saves 1/3 of the stem's MACs and all of its third input line buffer; the dropped
chroma axis is the only information loss.

Because the +128 offset arrives with the input, it is subtracted back out of the bias:

    conv(W2, [Y, C+128, 0]) + bias2
  = conv(W2, [Y, C, 0]) + (128/255)*sum_k W2[:,1,k] + bias2

so `bias2 = b - (128/255)*sum_k W2[:,1,k]`.

Usage:
    scripts/fold_stem_2ch.py <export-dir> <out-dir> [--axis DEG]

The axis defaults to 135 deg, the pooled principal axis of the chroma covariance over
`style-transfer/data/content` (measured -43 deg) and the demo set (-42 deg).
Inputs must be prepared with the matching axis; see the note `runs/note/v8-*.md`.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# BT.601 full-range RGB -> YCbCr.
M = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ]
)
QMAX = 127  # weight codes are int8, one per byte


def fold(export_dir: Path, out_dir: Path, axis_deg: float) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(export_dir, out_dir)

    manifest_path = out_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    stem = next(l for l in m["layers"] if l["name"] == "stem")
    if stem["in_channels"] != 3:
        raise SystemExit(f"stem already has {stem['in_channels']} input channels")

    weight_file = out_dir / m["weight_file"]
    raw = np.frombuffer(weight_file.read_bytes(), dtype=np.int8).astype(np.float64)

    off, cnt = stem["weight_offset"], stem["weight_count"]
    out_ch = stem["out_channels"]
    kernel = stem["kernel"]
    # Burn stores the dense stem as (out, in, kh, kw).
    w = raw[off : off + cnt].reshape(out_ch, 3, *kernel)
    w = w * np.array(stem["weight_scale"], dtype=np.float64)[:, None, None, None]
    b = np.array(stem["bias"], dtype=np.float64)

    theta = np.deg2rad(axis_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    # rgb = M @ (Y, Cb, Cr); keeping only C = u @ (Cb, Cr) means
    # rgb ~= Msel @ (Y, C) with Msel = [M[:,0], M[:,1]*u0 + M[:,2]*u1].
    msel = np.stack([M[:, 0], M[:, 1] * u[0] + M[:, 2] * u[1]], axis=1)
    w2 = np.einsum("oikn,ij->ojkn", w, msel)
    bias2 = b - (128.0 / 255.0) * w2[:, 1].reshape(out_ch, -1).sum(1)

    wf = np.zeros((out_ch, 3, *kernel))
    wf[:, :2] = w2
    flat = wf.reshape(out_ch, -1)
    scale = np.maximum(np.abs(flat).max(1) / QMAX, 1e-8)
    codes = np.clip(np.round(flat / scale[:, None]), -QMAX, QMAX).astype(np.int8)

    raw[off : off + cnt] = codes.reshape(-1)
    weight_file.write_bytes(raw.astype(np.int8).tobytes())

    stem["weight_scale"] = scale.tolist()
    stem["bias"] = bias2.tolist()
    stem["bias_int32"] = np.round(bias2 / (stem["in_scale"] * scale)).astype(np.int64).tolist()
    stem["requant"] = (stem["in_scale"] * scale / stem["out_scale"]).tolist()
    manifest_path.write_text(json.dumps(m, indent=1))

    print(f"folded {export_dir} -> {out_dir} (axis {axis_deg} deg)")
    for i, name in enumerate(("in.Y", "in.C", "in.dropped")):
        print(f"  {name:<12} absmax {np.abs(wf[:, i]).max():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--axis", type=float, default=135.0, help="chroma axis in degrees")
    a = ap.parse_args()
    fold(a.export_dir, a.out_dir, a.axis)


if __name__ == "__main__":
    main()
