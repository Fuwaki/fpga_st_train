#!/usr/bin/env python3
"""Merge study.json patches into per-trial training configs.

Usage:
  python scripts/study_gen.py experiments/dw_vangogh_lr
  python scripts/study_gen.py experiments/dw_vangogh_lr --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def path_depth_extra(from_base: Path, to_dir: Path) -> int:
    """How many extra `..` segments trial configs need vs `configs/`."""
    return len(to_dir.relative_to(ROOT).parts) - len(from_base.relative_to(ROOT).parts)


def deepen_relative(path: str, extra: int) -> str:
    if extra <= 0 or not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    return str(Path("../" * extra) / p)


def fix_paths_for_config_dir(cfg: dict, extra: int) -> None:
    """Keep the same absolute targets when configs live under experiments/.../configs/."""
    if extra <= 0:
        return
    run = cfg.setdefault("run", {})
    if run.get("sample_content"):
        run["sample_content"] = deepen_relative(run["sample_content"], extra)
    if run.get("init_from"):
        run["init_from"] = deepen_relative(run["init_from"], extra)

    data = cfg.setdefault("data", {})
    if data.get("content_dir"):
        data["content_dir"] = deepen_relative(data["content_dir"], extra)
    if data.get("pseudo_gt_dir"):
        data["pseudo_gt_dir"] = deepen_relative(data["pseudo_gt_dir"], extra)

    teacher = cfg.setdefault("teacher", {})
    if teacher.get("weights_dir"):
        teacher["weights_dir"] = deepen_relative(teacher["weights_dir"], extra)
    if teacher.get("style"):
        teacher["style"] = deepen_relative(teacher["style"], extra)
    if teacher.get("model"):
        teacher["model"] = deepen_relative(teacher["model"], extra)

    loss = cfg.setdefault("loss", {})
    if loss.get("vgg_weights"):
        loss["vgg_weights"] = deepen_relative(loss["vgg_weights"], extra)

    quant = cfg.get("quant")
    if quant and quant.get("calibration"):
        quant["calibration"] = deepen_relative(quant["calibration"], extra)


def deep_merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_dir",
        type=Path,
        help="e.g. experiments/dw_vangogh_lr",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print paths only, do not write",
    )
    args = parser.parse_args()

    study_dir = args.study_dir
    if not study_dir.is_absolute():
        study_dir = ROOT / study_dir
    study_path = study_dir / "study.json"
    if not study_path.is_file():
        print(f"missing {study_path}", file=sys.stderr)
        return 1

    spec = json.loads(study_path.read_text())
    study_id = spec.get("id") or study_dir.name
    base_rel = spec["base_config"]
    base_path = ROOT / base_rel
    if not base_path.is_file():
        print(f"missing base config {base_path}", file=sys.stderr)
        return 1

    base = json.loads(base_path.read_text())
    configs_dir = study_dir / "configs"
    base_config_dir = base_path.parent
    path_extra = path_depth_extra(base_config_dir, configs_dir)
    trials = spec.get("trials") or []
    if not trials:
        print("no trials in study.json", file=sys.stderr)
        return 1

    for trial in trials:
        trial_id = trial["id"]
        patch = trial.get("patch") or {}
        cfg = deep_merge(base, patch)
        fix_paths_for_config_dir(cfg, path_extra)
        # Paths in generated configs are rebased from experiments/<study>/configs/.
        out_dir = Path("..") / "trials" / trial_id
        cfg.setdefault("run", {})
        cfg["run"]["name"] = trial_id
        cfg["run"]["output_dir"] = str(out_dir).replace("\\", "/")

        out_path = configs_dir / f"{trial_id}.json"
        text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
        if args.dry_run:
            print(out_path.relative_to(ROOT))
            continue
        configs_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote {out_path.relative_to(ROOT)} -> {out_dir}")

    manifest = configs_dir / "MANIFEST"
    if not args.dry_run:
        lines = [t["id"] for t in trials]
        manifest.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
