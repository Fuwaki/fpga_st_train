#!/usr/bin/env python3
"""Visualise encoder / trunk / decoder-input maps dumped by examples/trunk_boundary.rs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STAGES = ("encoder", "trunk", "decoder_in")
TITLES = {
    "encoder": "encoder out  (trunk in)",
    "trunk": "after 16 residual layers",
    "decoder_in": "decoder in  (after IN)",
}


def load_map(directory: Path, sample: str, stage: str) -> np.ndarray:
    meta = json.loads((directory / f"{sample}_{stage}.json").read_text())
    raw = np.fromfile(directory / f"{sample}_{stage}.f32", dtype="<f4")
    return raw.reshape(meta["n"], meta["c"], meta["h"], meta["w"])[0]


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x * x).mean(axis=0))


def channel_std(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).std(axis=1)


def channel_mean(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).mean(axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8))


def spatial_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-8
    return num / den


def pca_rgb(x: np.ndarray, basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    c, h, w = x.shape
    tokens = x.reshape(c, -1).T
    center = tokens.mean(axis=0)
    centered = tokens - center
    if basis is None:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:3]
    proj = centered @ basis.T
    rgb = np.zeros_like(proj)
    for i in range(3):
        lo, hi = np.percentile(proj[:, i], [2, 98])
        rgb[:, i] = np.clip((proj[:, i] - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(h, w, 3), basis


def gray(image: np.ndarray, limit: float | None = None) -> np.ndarray:
    if limit is None:
        limit = float(np.percentile(image, 99))
    return np.clip(image / (limit + 1e-8), 0, 1)


def stats_for(maps: dict[str, np.ndarray]) -> dict:
    enc, trunk, dec = maps["encoder"], maps["trunk"], maps["decoder_in"]
    delta = trunk - enc
    return {
        "cosine_encoder_trunk": cosine(enc, trunk),
        "cosine_trunk_decoder": cosine(trunk, dec),
        "cosine_encoder_decoder": cosine(enc, dec),
        "rel_l2_residual": float(np.linalg.norm(delta) / (np.linalg.norm(enc) + 1e-8)),
        "rel_l2_in": float(np.linalg.norm(dec - trunk) / (np.linalg.norm(trunk) + 1e-8)),
        "encoder_abs_mean": float(np.abs(enc).mean()),
        "trunk_abs_mean": float(np.abs(trunk).mean()),
        "decoder_abs_mean": float(np.abs(dec).mean()),
        "encoder_ch_std_median": float(np.median(channel_std(enc))),
        "trunk_ch_std_median": float(np.median(channel_std(trunk))),
        "decoder_ch_std_median": float(np.median(channel_std(dec))),
        "decoder_ch_mean_abs": float(np.abs(channel_mean(dec)).mean()),
        "rms_corr_encoder_trunk": float(np.corrcoef(rms(enc).ravel(), rms(trunk).ravel())[0, 1]),
        "rms_corr_trunk_decoder": float(np.corrcoef(rms(trunk).ravel(), rms(dec).ravel())[0, 1]),
        "spatial_cosine_residual_mean": float(spatial_cosine(enc, trunk).mean()),
        "spatial_cosine_in_mean": float(spatial_cosine(trunk, dec).mean()),
    }


def render(directory: Path, sample: str, content: Path, output: Path) -> dict:
    maps = {stage: load_map(directory, sample, stage) for stage in STAGES}
    numbers = stats_for(maps)
    content_img = np.asarray(Image.open(content).convert("RGB"))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title("content")
    basis = None
    for ax, stage in zip(axes[0, 1:], STAGES):
        rgb, basis = pca_rgb(maps[stage], basis)
        ax.imshow(rgb)
        ax.set_title(TITLES[stage])
    energy_limit = max(float(np.percentile(rms(maps[stage]), 99)) for stage in STAGES)
    for ax, stage in zip(axes[1, :3], STAGES):
        ax.imshow(gray(rms(maps[stage]), energy_limit), cmap="magma")
        ax.set_title(f"channel RMS  {stage}")
    cos_map = spatial_cosine(maps["encoder"], maps["trunk"])
    im = axes[1, 3].imshow(cos_map, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 3].set_title("cosine  encoder vs trunk")
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{sample}   residual L2 {numbers['rel_l2_residual']:.2f}   "
        f"cosine enc→trunk {numbers['cosine_encoder_trunk']:.2f}   "
        f"IN L2 {numbers['rel_l2_in']:.2f}"
    )
    fig.savefig(output / f"{sample}_overview.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    channels = np.arange(maps["encoder"].shape[0])
    ax.plot(channels, channel_std(maps["encoder"]), label="encoder", lw=1.2)
    ax.plot(channels, channel_std(maps["trunk"]), label="after residual", lw=1.2)
    ax.plot(channels, channel_std(maps["decoder_in"]), label="after IN", lw=1.2)
    ax.set_xlabel("channel")
    ax.set_ylabel("spatial std")
    ax.legend()
    ax.set_title(f"{sample}  per-channel spatial std")
    fig.savefig(output / f"{sample}_channel_std.png", dpi=140)
    plt.close(fig)
    return numbers


def main() -> None:
    directory = Path("experiments/tiny_dw_vangogh_plugin/analysis/in_h4_trunk")
    trial = Path("experiments/tiny_dw_vangogh_plugin/trials/in_h4")
    report = {}
    for sample in ("animal", "cat", "city"):
        report[sample] = render(directory, sample, trial / f"content_{sample}.png", directory)
        print(sample, json.dumps(report[sample], indent=2))
    (directory / "stats.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
