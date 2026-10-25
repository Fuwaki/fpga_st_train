#!/usr/bin/env python3
"""Summarise metrics.jsonl for every trial under a study directory.

Usage:
  python scripts/study_collect.py experiments/dw_vangogh_lr
  python scripts/study_collect.py experiments/dw_vangogh_lr -o aggregate/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = "metrics.jsonl"


def load_events(path: Path) -> list[dict]:
    events = []
    if not path.is_file():
        return events
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def last_of_kind(events: list[dict], kind: str) -> dict | None:
    found = None
    for ev in events:
        if ev.get("kind") == kind:
            found = ev
    return found


def min_val(events: list[dict]) -> tuple[float | None, int | None]:
    best_loss = None
    best_step = None
    for ev in events:
        if ev.get("kind") != "val":
            continue
        loss = ev.get("val_loss")
        if loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_step = ev.get("step")
    return best_loss, best_step


def collect_trial(trial_dir: Path) -> dict:
    events = load_events(trial_dir / METRICS)
    run_end = last_of_kind(events, "run_end")
    sample = last_of_kind(events, "sample")
    val_loss, val_step = min_val(events)
    row = {
        "trial": trial_dir.name,
        "metrics": str(trial_dir / METRICS) if events else "",
        "final_step": run_end.get("step") if run_end else "",
        "wall_s": run_end.get("wall_s") if run_end else "",
        "train_loss": run_end.get("loss") if run_end else "",
        "min_val_loss": val_loss if val_loss is not None else "",
        "min_val_step": val_step if val_step is not None else "",
        "lapvar": sample.get("lapvar") if sample else "",
        "psnr_vs_teacher": sample.get("psnr_vs_teacher") if sample else "",
        "checkerboard": sample.get("checkerboard") if sample else "",
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="default: <study>/aggregate/summary.csv",
    )
    args = parser.parse_args()

    study_dir = args.study_dir
    if not study_dir.is_absolute():
        study_dir = ROOT / study_dir
    trials_root = study_dir / "trials"
    if not trials_root.is_dir():
        print(f"no trials/ under {study_dir}", file=sys.stderr)
        return 1

    rows = []
    for trial_dir in sorted(p for p in trials_root.iterdir() if p.is_dir()):
        rows.append(collect_trial(trial_dir))

    if not rows:
        print("no trial directories found", file=sys.stderr)
        return 1

    out = args.output or (study_dir / "aggregate" / "summary.csv")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
