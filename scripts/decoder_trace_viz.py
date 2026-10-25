#!/usr/bin/env python3
"""Compare decoder stages dumped by examples/decoder_trace.rs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STAGES = (
    "up_trunk",
    "skip",
    "fused",
    "half_dw",
    "half_sep",
    "half_pw",
    "full_up",
    "full_dw",
    "full_sep",
    "full_pw",
    "rgb",
)


def load_map(directory: Path, sample: str, stage: str) -> np.ndarray:
    meta = json.loads((directory / f"{sample}_{stage}.json").read_text())
    raw = np.fromfile(directory / f"{sample}_{stage}.f32", dtype="<f4")
    return raw.reshape(meta["n"], meta["c"], meta["h"], meta["w"])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.reshape(-1), b.reshape(-1)
    return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))


def const_frac(x: np.ndarray) -> float:
    mu = x.mean(axis=(1, 2))
    const = float((mu * mu).sum() * x.shape[1] * x.shape[2])
    return const / (float((x * x).sum()) + 1e-12)


def channel_std(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).std(axis=1)


def top8(x: np.ndarray) -> float:
    energy = (x * x).reshape(x.shape[0], -1).sum(axis=1)
    order = np.argsort(energy)[::-1][:8]
    return float(energy[order].sum() / (energy.sum() + 1e-12))


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x * x).mean(axis=0))


def stage_row(x: np.ndarray) -> dict:
    std = channel_std(x)
    return {
        "shape": list(x.shape),
        "const_frac": const_frac(x),
        "median_std": float(np.median(std)),
        "max_std": float(std.max()),
        "dead": float((std < 1e-4).mean()),
        "top8": top8(x),
        "abs_mean": float(np.abs(x).mean()),
    }


def pair(a: np.ndarray, b: np.ndarray) -> dict:
    return {"cosine": cosine(a, b), "rel_l2": rel_l2(a, b)}


def pca_rgb(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    if c == 3:
        img = np.transpose(x, (1, 2, 0))
        lo, hi = np.percentile(img, [2, 98])
        return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    tokens = x.reshape(c, -1).T
    centered = tokens - tokens.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:3].T
    rgb = np.zeros_like(proj)
    for i in range(3):
        lo, hi = np.percentile(proj[:, i], [2, 98])
        rgb[:, i] = np.clip((proj[:, i] - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(h, w, 3)


def gray(image: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(image / (limit + 1e-8), 0, 1)


def render(directory: Path, sample: str, content: Path) -> dict:
    maps = {stage: load_map(directory, sample, stage) for stage in STAGES}
    numbers = {stage: stage_row(maps[stage]) for stage in STAGES}
    numbers["pairs"] = {
        "up_vs_skip": pair(maps["up_trunk"], maps["skip"]),
        "fused_vs_up": pair(maps["fused"], maps["up_trunk"]),
        "fused_vs_skip": pair(maps["fused"], maps["skip"]),
        "fused_vs_half_dw": pair(maps["fused"], maps["half_dw"]),
        "half_sep_vs_half_pw": pair(maps["half_sep"], maps["half_pw"]),
        "full_up_vs_full_dw": pair(maps["full_up"], maps["full_dw"]),
        "full_sep_vs_full_pw": pair(maps["full_sep"], maps["full_pw"]),
    }
    up_n = float(np.linalg.norm(maps["up_trunk"]))
    skip_n = float(np.linalg.norm(maps["skip"]))
    numbers["norm_up"] = up_n
    numbers["norm_skip"] = skip_n
    numbers["skip_over_up"] = skip_n / (up_n + 1e-8)
    numbers["rms_corr_up_skip"] = float(
        np.corrcoef(rms(maps["up_trunk"]).ravel(), rms(maps["skip"]).ravel())[0, 1]
    )

    content_img = np.asarray(Image.open(content).convert("RGB"))
    show = ["up_trunk", "skip", "fused", "half_pw", "full_pw", "rgb"]
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), constrained_layout=True)
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title("content")
    for ax, stage in zip(list(axes[0, 1:]) + [axes[1, 0]], show):
        ax.imshow(pca_rgb(maps[stage]))
        ax.set_title(stage)
    energy = [rms(maps[s]) for s in ("up_trunk", "skip", "fused", "half_pw", "full_pw")]
    limit = max(float(np.percentile(e, 99)) for e in energy)
    for ax, stage, image in zip(axes[1, 1:], ("up_trunk", "skip", "fused", "half_pw", "full_pw"), energy):
        ax.imshow(gray(image, limit), cmap="magma")
        ax.set_title(f"RMS {stage}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{sample}  skip/up norm {numbers['skip_over_up']:.2f}  "
        f"fused↔dw cos {numbers['pairs']['fused_vs_half_dw']['cosine']:.2f}  "
        f"sep↔pw cos {numbers['pairs']['half_sep_vs_half_pw']['cosine']:.2f}"
    )
    fig.savefig(directory / f"{sample}_overview.png", dpi=130)
    plt.close(fig)
    return numbers


def main() -> None:
    directory = Path("experiments/tiny_dw_vangogh_plugin/analysis/in_h4_decoder")
    trial = Path("experiments/tiny_dw_vangogh_plugin/trials/in_h4")
    report = {}
    for sample in ("animal", "cat", "city"):
        report[sample] = render(directory, sample, trial / f"content_{sample}.png")
        pairs = report[sample]["pairs"]
        print(sample, "skip/up", round(report[sample]["skip_over_up"], 3))
        for name, row in pairs.items():
            print(f"  {name:22} cos {row['cosine']:.3f}  relL2 {row['rel_l2']:.3f}")
        for stage in STAGES:
            row = report[sample][stage]
            print(
                f"  {stage:10} {row['shape']} const {row['const_frac']:.3f} "
                f"med_std {row['median_std']:.4f} dead {row['dead']:.2f} top8 {row['top8']:.2f}"
            )
    (directory / "stats.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
