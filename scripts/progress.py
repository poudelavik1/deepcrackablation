#!/usr/bin/env python3
"""Show how far the benchmark has got: one line per (model, seed) run.

    python scripts/progress.py                     # uses RUNS_ROOT from slurm/env.sh if sourced
    python scripts/progress.py --protocol original

Status: done (FINISHED written), running (history updated in the last 30 min),
stalled (no update for 30 min and not finished: failed, timed out or waiting to resume).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=os.environ.get("RUNS_ROOT"))
    parser.add_argument("--protocol", default="benchmark", help="benchmark | original | ablation_deepcrack")
    args = parser.parse_args()

    root = cfg.runs_root(args.runs_root) / args.protocol
    runs = sorted(root.glob("*/seed_*"))
    if not runs:
        print(f"No runs yet under {root}")
        return

    print(f"{'model':<18} {'seed':>4} {'status':<8} {'epoch':>7} {'best val IoU':>12} {'min/epoch':>9} {'last update':>12}")
    counts = {"done": 0, "running": 0, "stalled": 0}
    for run in runs:
        history = run / "history.csv"
        rows = list(csv.DictReader(history.open())) if history.exists() else []
        try:
            epochs = cfg.load_config(str(run / "config.yaml"))["train"]["epochs"]
        except Exception:
            epochs = "?"
        age = time.time() - history.stat().st_mtime if history.exists() else None
        if (run / "FINISHED").exists():
            status = "done"
        elif age is not None and age < 1800:
            status = "running"
        else:
            status = "stalled"
        counts[status] += 1
        best = max((float(r["val_iou"]) for r in rows), default=float("nan"))
        minutes = sum(float(r["seconds"]) for r in rows) / len(rows) / 60 if rows else float("nan")
        updated = f"{age / 60:.0f} min ago" if age is not None else "-"
        print(f"{run.parent.name:<18} {run.name[5:]:>4} {status:<8} {len(rows):>3}/{epochs:<3} "
              f"{best:>12.4f} {minutes:>9.1f} {updated:>12}")
    print(f"\n{len(runs)} runs started: {counts['done']} done, {counts['running']} running, {counts['stalled']} stalled")
    print("Done runs may stop before the last epoch: early stopping (patience) ends them.")


if __name__ == "__main__":
    main()
