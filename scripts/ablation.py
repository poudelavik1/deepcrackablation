#!/usr/bin/env python3
"""Ablation of one model's training recipe: data augmentation and optimizer.

The plan is configs/ablation/<model>.yaml. On the cluster use slurm/ablation.sh, which wraps:

    python scripts/ablation.py check                                 # build every variant (CPU, seconds)
    python scripts/ablation.py jobs --stage screen  --out jobs.txt   # stage 1: one change per variant
    python scripts/ablation.py jobs --stage combine --out jobs.txt   # stage 2: after stage 1
    python scripts/ablation.py jobs --stage final   --out jobs.txt   # stage 3: after stage 2
    python scripts/ablation.py jobs --stage test    --out jobs.txt   # final recipe + reference, test set
    python scripts/ablation.py report                                # tables, figures, decisions (no GPU)

Stages
  screen   every variant changes one thing of the benchmark protocol and is trained with seed 0;
           the reference (the benchmark protocol itself) with seeds 0 1 2, which measures seed noise.
  combine  the best optimizer (if it beats the reference mean by the margin) with the top-1, top-3 and
           all augmentation changes that do (best variant per factor), and those changes without it.
  final    the best candidates get seeds 1 2. The best over three seeds is the final recipe; it is
           tested against the reference with a Wilcoxon signed-rank test on per-tile val IoU.

Selection uses validation only (val IoU at the val-chosen threshold, as in the benchmark). Variant
configs are written fully resolved to $RUNS_ROOT/<protocol>/_configs/<variant>.yaml, results go to
$RUNS_ROOT/<protocol>/<variant>/seed_<n>/, tables and figures to $RUNS_ROOT/tables and figures.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg  # noqa: E402

DEFAULT_PLAN = REPO_ROOT / "configs" / "ablation" / "deepcrack.yaml"
STAGES = ("screen", "combine", "final", "test")
NOT_RECIPE = ("protocol", "run_name", "ablation", "seed", "model_info")   # keys that do not change training
REFERENCE = "ref"
ACTIVE_MINUTES = 30      # a run written to more recently than this is taken to be running (not queued again)


def lr_text(lr: float) -> str:
    """3e-4 style (no zero padding in the exponent)."""
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def recipe_key(config: dict[str, Any]) -> str:
    """Identity of a training recipe: the config without names and metadata."""
    return json.dumps({k: v for k, v in config.items() if k not in NOT_RECIPE}, sort_keys=True, default=str)


@dataclass
class Variant:
    name: str
    stage: str                     # screen | combine
    arm: str                       # reference | optimizer | augmentation | combination
    factor: str
    change: str                    # one line for tables
    override: dict[str, Any]       # merged over the model's benchmark config
    priority: int = 2              # job order (1 first)
    optimizer: str | None = None
    lr: float | None = None
    parts: list[str] = field(default_factory=list)   # combination: the variants it merges


@dataclass
class Decision:
    """What stage 1 says: the reference, its noise, the best optimizer, the positive augmentation changes."""
    ref_mean: float
    ref_std: float
    ref_n: int
    margin: float
    missing: list[str]
    best_optimizer: Variant | None
    optimizer_note: str
    edge_notes: list[str]
    positives: list[Variant]       # best variant per factor with delta >= margin, best first


class Plan:
    def __init__(self, path: Path, runs_root: Path):
        self.path = Path(path)
        self.data = cfg.load_yaml(self.path)
        self.protocol = self.data["protocol"]
        self.root = runs_root / self.protocol
        self.config_dir = self.root / "_configs"
        self.base = cfg.load_config(self.data["config"])    # the model's benchmark config
        self.model = self.base["model"]
        self.reference_seeds = [int(s) for s in self.data.get("reference_seeds", [0, 1, 2])]
        self.screen_seed = int(self.data.get("screen_seed", 0))
        self.final_seeds = [int(s) for s in self.data.get("final_seeds", [1, 2])]
        self.final_candidates = int(self.data.get("final_candidates", 2))
        self.final_controls = [str(n) for n in self.data.get("final_controls", None) or []]
        self.min_margin = float(self.data.get("margin", 0.003))
        self._screen: list[Variant] | None = None

    # ---- variants ---------------------------------------------------------------------------- #
    def reference(self) -> Variant:
        return Variant(REFERENCE, "screen", "reference", "reference",
                       self.data.get("reference", {}).get("change", "Benchmark protocol"), {}, priority=1)

    def optimizer_label(self, name: str) -> str:
        return str(self.data.get("optimizers", {}).get(name, {}).get("label", name))

    def screen_variants(self) -> list[Variant]:
        if self._screen is not None:
            return self._screen
        reference = self.reference()
        variants = [reference]
        reference_key = recipe_key(self.resolve(reference))
        for name, spec in (self.data.get("optimizers") or {}).items():
            lrs = [float(lr) for lr in spec["lrs"]]
            for index, lr in enumerate(lrs):
                train: dict[str, Any] = {"optimizer": name, "lr": lr, "optimizer_args": dict(spec.get("args") or {})}
                if "weight_decay" in spec:
                    train["weight_decay"] = float(spec["weight_decay"])
                variant = Variant(f"opt_{name}_lr{lr_text(lr)}", "screen", "optimizer", "optimizer",
                                  f"{spec.get('label', name)}, lr {lr_text(lr)}", {"train": train},
                                  priority=2 if index == len(lrs) // 2 else 3, optimizer=name, lr=lr)
                if recipe_key(self.resolve(variant)) == reference_key:
                    continue          # this grid point is the reference itself
                variants.append(variant)
        for name, spec in (self.data.get("augmentations") or {}).items():
            variants.append(Variant(name, "screen", "augmentation", str(spec["factor"]), str(spec["change"]),
                                    {"aug": dict(spec["set"])}))
        names = [v.name for v in variants]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise SystemExit(f"Duplicate variant names in {self.path}: {duplicates}")
        self._screen = variants
        return variants

    def screen_variant(self, name: str) -> Variant | None:
        return next((v for v in self.screen_variants() if v.name == name), None)

    def resolve(self, variant: Variant) -> dict[str, Any]:
        config = cfg.deep_merge(self.base, variant.override)
        config["protocol"] = self.protocol
        config["run_name"] = variant.name
        config["ablation"] = {"variant": variant.name, "stage": variant.stage, "arm": variant.arm,
                              "factor": variant.factor, "change": variant.change, "parts": variant.parts,
                              "plan": self.path.resolve().relative_to(REPO_ROOT).as_posix()
                              if self.path.resolve().is_relative_to(REPO_ROOT) else str(self.path)}
        return config

    def write_config(self, name: str, config: dict[str, Any]) -> Path:
        """Fully resolved config (no _base_), so the run does not depend on later edits of base.yaml."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        path = self.config_dir / f"{name}.yaml"
        cfg.save_config({"_base_": None, **config}, path)
        return path

    def run_dir(self, name: str, seed: int) -> Path:
        return self.root / name / f"seed_{seed}"

    def done(self, name: str, seed: int) -> bool:
        """Trained and evaluated on val."""
        run = self.run_dir(name, seed)
        return (run / "FINISHED").exists() and (run / "eval" / "val_metrics.json").exists()

    def active(self, name: str, seed: int) -> bool:
        """Not done, but a file of the run changed in the last ACTIVE_MINUTES: most likely a job is training
        or evaluating it right now, and a second job on the same folder would corrupt it."""
        run = self.run_dir(name, seed)
        times = [p.stat().st_mtime for p in run.rglob("*") if p.is_file()] if run.is_dir() else []
        return bool(times) and time.time() - max(times) < ACTIVE_MINUTES * 60


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def read_runs(plan: Plan):
    """One row per finished run: val metrics (tuned threshold) or, before evaluation, the training IoU."""
    import pandas as pd

    rows = []
    for run in sorted(plan.root.glob("*/seed_*")) if plan.root.is_dir() else []:
        if not (run / "FINISHED").exists():
            continue
        finished = json.loads((run / "FINISHED").read_text(encoding="utf-8"))
        row: dict[str, Any] = {"variant": run.parent.name, "seed": int(run.name.split("_")[1]),
                               "best_epoch": finished.get("best_epoch"), "epochs_run": finished.get("epochs_run")}
        metrics_path = run / "eval" / "val_metrics.json"
        if metrics_path.exists():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            values = result["metrics"]
            row.update(iou=values["iou"], f1=values["f1"], precision=values["precision"], recall=values["recall"],
                       tol_f1_2=values.get("tol_f1_2"), cldice=values.get("cldice"), ap=result.get("ap"),
                       threshold=result.get("threshold"), source="val")
        else:
            row.update(iou=finished.get("best_val_iou"), source="train@0.5")
        history = run / "history.csv"
        if history.exists():
            row["minutes"] = round(float(pd.read_csv(history)["seconds"].sum()) / 60, 1)
        rows.append(row)
    columns = ["variant", "seed", "iou", "f1", "precision", "recall", "tol_f1_2", "cldice", "ap", "threshold",
               "best_epoch", "epochs_run", "minutes", "source"]
    return pd.DataFrame(rows, columns=columns)


def value(runs, name: str, seed: int) -> float:
    match = runs[(runs["variant"] == name) & (runs["seed"] == seed)]
    return float(match["iou"].iloc[0]) if len(match) else float("nan")


def decide(plan: Plan, runs) -> Decision:
    """Stage-1 decisions. Deltas are the seed-0 val IoU minus the reference mean over its seeds."""
    reference = runs[runs["variant"] == REFERENCE]["iou"]
    ref_mean = float(reference.mean()) if len(reference) else float("nan")
    ref_std = float(reference.std(ddof=1)) if len(reference) > 1 else float("nan")
    margin = max(plan.min_margin, ref_std if math.isfinite(ref_std) else 0.0)
    missing = []
    for variant in plan.screen_variants():
        seeds = plan.reference_seeds if variant.name == REFERENCE else [plan.screen_seed]
        missing += [f"{variant.name}/seed_{s}" for s in seeds if not plan.done(variant.name, s)]

    def delta(variant: Variant) -> float:
        return value(runs, variant.name, plan.screen_seed) - ref_mean

    optimizers = [v for v in plan.screen_variants() if v.arm == "optimizer"]
    scored = [(delta(v), v) for v in optimizers if math.isfinite(delta(v))]
    best_optimizer, note = None, "no optimizer run finished yet"
    if scored:
        best_delta, best = max(scored, key=lambda item: item[0])
        if best_delta >= margin:
            best_optimizer, note = best, f"{best.change} ({best_delta:+.4f} over the reference)"
        else:
            note = (f"keep the reference optimizer: the best, {best.change}, is {best_delta:+.4f} "
                    f"(below the margin {margin:.4f})")

    edge_notes = []
    for name, spec in (plan.data.get("optimizers") or {}).items():
        lrs = [float(lr) for lr in spec["lrs"]]
        points = []
        for lr in lrs:
            variant = next((v for v in optimizers if v.optimizer == name and v.lr == lr), None)
            points.append(ref_mean if variant is None else value(runs, variant.name, plan.screen_seed))
        if all(math.isfinite(p) for p in points) and len(points) >= 2:
            best_index = max(range(len(points)), key=points.__getitem__)
            # Only a clear slope towards the edge counts (a flat curve picks an edge by noise): the edge
            # point must beat its neighbour in the grid by the margin.
            neighbour = points[1] if best_index == 0 else points[-2]
            if best_index in (0, len(points) - 1) and points[best_index] - neighbour >= margin:
                side = "lowest" if best_index == 0 else "highest"
                edge_notes.append(f"{plan.optimizer_label(name)}: best at the {side} lr of the grid "
                                  f"({lr_text(lrs[best_index])}); its optimum may lie outside the grid")

    best_per_factor: dict[str, tuple[float, Variant]] = {}
    for variant in plan.screen_variants():
        if variant.arm != "augmentation" or variant.factor == "none":
            continue
        change = delta(variant)
        if math.isfinite(change) and (variant.factor not in best_per_factor or change > best_per_factor[variant.factor][0]):
            best_per_factor[variant.factor] = (change, variant)
    positives = [v for d, v in sorted(best_per_factor.values(), key=lambda item: -item[0]) if d >= margin]
    return Decision(ref_mean, ref_std, len(reference), margin, missing, best_optimizer, note, edge_notes, positives)


def combine_variants(plan: Plan, decision: Decision) -> list[Variant]:
    """Stage-2 candidates, without recipes that stage 1 already trained."""
    optimizer = decision.best_optimizer
    optimizer_override = optimizer.override if optimizer else {}
    optimizer_text = optimizer.change if optimizer else "AdamW, lr 3e-4"
    positives = decision.positives

    def merged(subset: list[Variant]) -> dict[str, Any]:
        aug: dict[str, Any] = {}
        for variant in subset:
            aug.update(copy.deepcopy(variant.override["aug"]))
        return aug

    candidates = []
    for count in sorted({min(k, len(positives)) for k in (1, 3, len(positives))} - {0}):
        subset = positives[:count]
        candidates.append(Variant(
            f"combo_top{count}", "combine", "combination", "combination",
            f"{optimizer_text}; " + "; ".join(v.change for v in subset),
            cfg.deep_merge(optimizer_override, {"aug": merged(subset)}),
            parts=([optimizer.name] if optimizer else []) + [v.name for v in subset]))
    if optimizer and positives:     # the augmentation changes alone, to separate the two effects
        candidates.append(Variant(
            f"combo_top{len(positives)}_refopt", "combine", "combination", "combination",
            "AdamW, lr 3e-4; " + "; ".join(v.change for v in positives),
            {"aug": merged(positives)}, parts=[v.name for v in positives]))
    seen = {recipe_key(plan.resolve(v)) for v in plan.screen_variants()}
    unique = []
    for candidate in candidates:
        key = recipe_key(plan.resolve(candidate))
        if key not in seen:          # e.g. top-1 with the reference optimizer is that stage-1 variant
            seen.add(key)
            unique.append(candidate)
    return unique


def final_pool(plan: Plan, runs, decision: Decision) -> tuple[list[dict[str, Any]], list[str]]:
    """(candidates ranked by seed-0 val IoU, combinations without a finished seed-0 run)."""
    pool = ([decision.best_optimizer] if decision.best_optimizer else []) + decision.positives[:1]
    combos = combine_variants(plan, decision)
    pool += combos
    missing = [f"{v.name}/seed_{plan.screen_seed}" for v in combos if not plan.done(v.name, plan.screen_seed)]
    ranked = []
    for variant in pool:
        iou = value(runs, variant.name, plan.screen_seed)
        if math.isfinite(iou):
            ranked.append({"variant": variant, "iou": iou, "delta": iou - decision.ref_mean})
    ranked.sort(key=lambda row: -row["iou"])
    return ranked, missing


def final_controls(plan: Plan, runs, chosen: list[dict[str, Any]]) -> list[str]:
    """final_controls of the plan that stage 3 trains too: finished with seed 0, not already a candidate."""
    taken = {row["variant"].name for row in chosen}
    names = []
    for name in plan.final_controls:
        if name in taken or name == REFERENCE:
            continue
        if not math.isfinite(value(runs, name, plan.screen_seed)):
            raise SystemExit(f"final_controls: {name} has no evaluated seed-{plan.screen_seed} run under {plan.root}")
        names.append(name)
    return names


def confirmed(plan: Plan, runs) -> list[str]:
    """Variants trained (and evaluated) with every seed of the reference."""
    needed = set(plan.reference_seeds)
    names = []
    for name, group in runs[runs["source"] == "val"].groupby("variant"):
        if needed <= set(group["seed"]):
            names.append(name)
    return names


def final_result(plan: Plan, runs) -> dict[str, Any] | None:
    """Best variant over the reference seeds, and its test against the reference."""
    import numpy as np

    names = confirmed(plan, runs)
    if REFERENCE not in names:
        return None
    rows = []
    for name in names:
        subset = runs[(runs["variant"] == name) & runs["seed"].isin(plan.reference_seeds)]
        rows.append({"variant": name, "mean": float(subset["iou"].mean()), "std": float(subset["iou"].std(ddof=1)),
                     "n": len(subset)})
    rows.sort(key=lambda r: -r["mean"])
    winner = rows[0]["variant"]
    reference = next(r for r in rows if r["variant"] == REFERENCE)
    result = {"rows": rows, "winner": winner, "reference": reference, "p_value": float("nan"),
              "median_tile_diff": float("nan"), "seed_diffs": []}
    if winner != REFERENCE:
        result["seed_diffs"] = [value(runs, winner, s) - value(runs, REFERENCE, s) for s in plan.reference_seeds]
        tiles = per_tile_iou(plan, [winner, REFERENCE])
        if len(tiles) >= 6 and not np.allclose(tiles[winner], tiles[REFERENCE]):
            from scipy.stats import wilcoxon

            result["p_value"] = float(wilcoxon(tiles[winner], tiles[REFERENCE], zero_method="wilcox").pvalue)
            result["median_tile_diff"] = float((tiles[winner] - tiles[REFERENCE]).median())
    return result


def test_result(plan: Plan, names: list[str]) -> dict[str, Any] | None:
    """Test-set metrics (threshold chosen on val) of these variants over the reference seeds, once all exist."""
    import numpy as np
    import pandas as pd

    rows = []
    for name in names:
        for seed in plan.reference_seeds:
            path = plan.run_dir(name, seed) / "eval" / "test_metrics.json"
            if not path.exists():
                return None
            result = json.loads(path.read_text(encoding="utf-8"))
            values = result["metrics"]
            rows.append({"variant": name, "seed": seed, "iou": values["iou"], "f1": values["f1"],
                         "precision": values["precision"], "recall": values["recall"],
                         "tol_f1_2": values.get("tol_f1_2"), "ap": result.get("ap"), "threshold": result.get("threshold")})
    runs = pd.DataFrame(rows)
    out = {"runs": runs, "p_value": float("nan"), "median_tile_diff": float("nan"), "seed_diffs": []}
    if len(names) == 2:
        first, second = names
        out["seed_diffs"] = [float(runs[(runs.variant == first) & (runs.seed == s)].iou.iloc[0]
                                   - runs[(runs.variant == second) & (runs.seed == s)].iou.iloc[0])
                             for s in plan.reference_seeds]
        tiles = per_tile_iou(plan, names, split="test")
        if len(tiles) >= 6 and not np.allclose(tiles[first], tiles[second]):
            from scipy.stats import wilcoxon

            out["p_value"] = float(wilcoxon(tiles[first], tiles[second], zero_method="wilcox").pvalue)
            out["median_tile_diff"] = float((tiles[first] - tiles[second]).median())
    return out


def per_tile_iou(plan: Plan, names: list[str], split: str = "val"):
    """Per-tile IoU of a split averaged over the reference seeds: rows = tiles, columns = variants."""
    import pandas as pd

    columns = {}
    for name in names:
        frames = []
        for seed in plan.reference_seeds:
            path = plan.run_dir(name, seed) / "eval" / f"{split}_per_tile.csv"
            if path.exists():
                frames.append(pd.read_csv(path).set_index("name")["iou"])
        if frames:
            columns[name] = pd.concat(frames, axis=1).mean(axis=1)
    return pd.DataFrame(columns).dropna()


def run_config(plan: Plan, name: str) -> dict[str, Any]:
    """The config a variant was trained with (from its first finished run)."""
    for run in sorted((plan.root / name).glob("seed_*")):
        if (run / "config.yaml").exists():
            config = cfg.load_yaml(run / "config.yaml")
            return {k: v for k, v in config.items() if k not in ("seed", "model_info")}
    raise SystemExit(f"No trained run of {name} under {plan.root}")


def recipe_diff(config: dict[str, Any], base: dict[str, Any], prefix: str = "") -> list[str]:
    """`section.key: benchmark -> variant` for every setting that differs from the benchmark config."""
    lines = []
    for key in sorted(set(config) | set(base)):
        if not prefix and key in NOT_RECIPE:
            continue
        new, old = config.get(key), base.get(key)
        if isinstance(new, dict) and isinstance(old, dict):
            lines += recipe_diff(new, old, f"{prefix}{key}.")
        elif new != old:
            lines.append(f"{prefix}{key}: {old} -> {new}")
    return lines


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def jobs(plan: Plan, stage: str, only: list[str], allow_incomplete: bool,
         include_active: bool = False) -> list[tuple[str, int, Path]]:
    """(variant, seed, config path) to run, without finished + evaluated runs and runs in progress."""
    runs = read_runs(plan) if stage != "screen" else None
    entries: list[tuple[int, str, int, dict[str, Any] | Path]] = []   # (priority, name, seed, config)

    if stage == "screen":
        for variant in plan.screen_variants():
            seeds = plan.reference_seeds if variant.name == REFERENCE else [plan.screen_seed]
            for seed in seeds:
                entries.append((variant.priority if seed == plan.screen_seed else 1, variant.name, seed,
                                plan.resolve(variant)))
    else:
        decision = decide(plan, runs)
        if decision.missing and not allow_incomplete:
            raise SystemExit(f"Stage 1 is not complete ({len(decision.missing)} runs missing, e.g. "
                             f"{', '.join(decision.missing[:4])}). Wait for it, resubmit with "
                             f"`bash slurm/ablation.sh screen`, or pass --allow-incomplete.")
        if stage == "combine":
            for variant in combine_variants(plan, decision):
                entries.append((2, variant.name, plan.screen_seed, plan.resolve(variant)))
            if not entries:
                print("Nothing to combine: no optimizer and no augmentation change beats the reference by the "
                      f"margin ({decision.margin:.4f}). The benchmark protocol stays the recipe.", file=sys.stderr)
        elif stage == "final":
            ranked, missing = final_pool(plan, runs, decision)
            if missing and not allow_incomplete:
                raise SystemExit(f"Stage 2 is not complete: {', '.join(missing)}. Run `bash slurm/ablation.sh "
                                 "combine` and wait for it, or pass --allow-incomplete.")
            chosen = [row for row in ranked if row["delta"] > 0][:plan.final_candidates]
            for row in chosen:
                name = row["variant"].name
                print(f"candidate {name}: seed {plan.screen_seed} val IoU {row['iou']:.4f} ({row['delta']:+.4f})",
                      file=sys.stderr)
                for seed in plan.final_seeds:
                    entries.append((2, name, seed, run_config(plan, name)))
            for name in final_controls(plan, runs, chosen):
                iou = value(runs, name, plan.screen_seed)
                print(f"control   {name}: seed {plan.screen_seed} val IoU {iou:.4f} "
                      f"({iou - decision.ref_mean:+.4f}), from final_controls in the plan", file=sys.stderr)
                for seed in plan.final_seeds:
                    entries.append((3, name, seed, run_config(plan, name)))
            if not chosen:
                print("No candidate beats the reference mean with seed 0: the benchmark protocol is the final "
                      "recipe (nothing to confirm).", file=sys.stderr)
            reference = plan.resolve(plan.reference())
            entries += [(1, REFERENCE, s, reference) for s in plan.reference_seeds]
        elif stage == "test":
            result = final_result(plan, runs)
            if result is None:
                raise SystemExit("No final result yet: the reference and at least one candidate need "
                                 f"seeds {plan.reference_seeds} (stage 3).")
            for name in dict.fromkeys([result["winner"], REFERENCE]):
                config = run_config(plan, name)
                entries += [(1, name, s, config) for s in plan.reference_seeds]
            out = []
            for _, name, seed, config in entries:
                if (plan.run_dir(name, seed) / "eval" / "test_metrics.json").exists():
                    continue
                out.append((name, seed, plan.write_config(name, config)))
            return out

    if only:
        entries = [e for e in entries if any(fnmatch.fnmatch(e[1], pattern) for pattern in only)]
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    out, written = [], set()
    for _, name, seed, config in entries:
        if plan.done(name, seed):
            continue
        if not include_active and plan.active(name, seed):
            print(f"skip {name} seed {seed}: written to in the last {ACTIVE_MINUTES} min, so probably running now "
                  "(it will be queued by a later resubmission if it fails; --include-active queues it anyway)",
                  file=sys.stderr)
            continue
        path = plan.config_dir / f"{name}.yaml"
        if name not in written:
            path = plan.write_config(name, config)
            written.add(name)
        out.append((name, seed, path))
    return out


# --------------------------------------------------------------------------- #
# Check: build every variant once (catches albumentations / torch API differences before 50 runs)
# --------------------------------------------------------------------------- #
def check(plan: Plan) -> int:
    import numpy as np
    import torch

    import data
    from optimizers import build_optimizer
    from train import cutmix

    crop = int(plan.base["data"]["crop_size"])
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (crop + 44, crop + 20, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[40:44, :] = 1                        # a horizontal "crack"
    failures = 0
    for variant in plan.screen_variants():
        config = plan.resolve(variant)
        try:
            # Every step that is on is forced to p = 1, so each one really runs once.
            config["aug"] = {k: (1.0 if k.endswith("_p") and float(v or 0) > 0 else v) for k, v in config["aug"].items()}
            transform = data.build_train_transform(config, seed=0)
            for _ in range(3):
                out = transform(image=image, mask=mask)
                assert out["image"].shape == (crop, crop, 3), out["image"].shape
                assert out["mask"].shape == (crop, crop) and set(np.unique(out["mask"])) <= {0, 1}
            images, masks = data.to_tensors(out["image"], out["mask"])
            mixed = cutmix(images.unsqueeze(0).repeat(4, 1, 1, 1), masks.unsqueeze(0).repeat(4, 1, 1, 1),
                           float(config["aug"].get("cutmix_p", 0) or 0))
            assert mixed[0].shape == (4, 3, crop, crop)
            layer = torch.nn.Conv2d(3, 1, 3)
            optimizer = build_optimizer([{"params": list(layer.parameters()), "lr": config["train"]["lr"]}],
                                        config["train"])
            layer(torch.rand(2, 3, 8, 8)).mean().backward()
            optimizer.step()
            steps = len(transform.transforms) if hasattr(transform, "transforms") else "?"
            print(f"ok    {variant.name:<26} {type(optimizer).__name__:<8} {steps} augmentation steps")
        except Exception as error:   # noqa: BLE001 - report every variant, then fail
            failures += 1
            print(f"FAIL  {variant.name:<26} {type(error).__name__}: {error}")
    import albumentations

    print(f"\n{len(plan.screen_variants()) - failures}/{len(plan.screen_variants())} variants ok "
          f"(albumentations {albumentations.__version__}, torch {torch.__version__})")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
LINE, BEST = "#4A90C2", "#08306B"                           # lr sweep line, its best point (one hue)
BETTER, WORSE, NEUTRAL = "#0072B2", "#D55E00", "#9E9E9E"   # Okabe-Ito blue / vermillion, grey = within noise
BAND = "#E8E8E8"                                            # +- margin band behind the marks


def optimizer_table(plan: Plan, runs, decision: Decision):
    import pandas as pd

    rows = []
    for name, spec in (plan.data.get("optimizers") or {}).items():
        for lr in [float(x) for x in spec["lrs"]]:
            variant = next((v for v in plan.screen_variants() if v.optimizer == name and v.lr == lr), None)
            if variant is None:       # the reference grid point: the reference mean over its seeds
                subset = runs[runs["variant"] == REFERENCE]
                iou, row_name = decision.ref_mean, f"{REFERENCE} (mean)"
                f1, epoch = subset["f1"].mean(), None
            else:
                subset = runs[(runs["variant"] == variant.name) & (runs["seed"] == plan.screen_seed)]
                iou, row_name = value(runs, variant.name, plan.screen_seed), variant.name
                f1 = float(subset["f1"].iloc[0]) if len(subset) else float("nan")
                epoch = subset["best_epoch"].iloc[0] if len(subset) else None
            rows.append({"optimizer": plan.optimizer_label(name), "lr": lr_text(lr), "variant": row_name,
                         "val_iou": iou, "delta_iou": iou - decision.ref_mean, "val_f1": f1, "best_epoch": epoch,
                         "lr_value": lr})
    return pd.DataFrame(rows)


def augmentation_table(plan: Plan, runs, decision: Decision):
    import pandas as pd

    positive = {v.name for v in decision.positives}
    rows = []
    for variant in plan.screen_variants():
        if variant.arm != "augmentation":
            continue
        subset = runs[(runs["variant"] == variant.name) & (runs["seed"] == plan.screen_seed)]
        iou = float(subset["iou"].iloc[0]) if len(subset) else float("nan")
        change = iou - decision.ref_mean
        if not math.isfinite(change):
            verdict = "missing"
        elif variant.name in positive:
            verdict = "selected"
        elif change >= decision.margin:
            verdict = "better (not best of its factor)"
        elif change <= -decision.margin:
            verdict = "worse"
        else:
            verdict = "within noise"
        rows.append({"variant": variant.name, "factor": variant.factor, "change": variant.change, "val_iou": iou,
                     "delta_iou": change, "val_f1": float(subset["f1"].iloc[0]) if len(subset) else float("nan"),
                     "val_precision": float(subset["precision"].iloc[0]) if len(subset) else float("nan"),
                     "val_recall": float(subset["recall"].iloc[0]) if len(subset) else float("nan"),
                     "verdict": verdict})
    return pd.DataFrame(rows).sort_values("delta_iou", ascending=False, na_position="last")


def figure_optimizers(plan: Plan, table, decision: Decision, out: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import plots

    table = table.dropna(subset=["val_iou"])
    if table.empty:
        return None
    # One panel per optimizer, val IoU against lr; best optimizer first (top left), all on one y scale.
    order = table.groupby("optimizer", sort=False)["val_iou"].max().sort_values(ascending=False).index.tolist()
    columns = 3
    rows = math.ceil(len(order) / columns)
    plots.setup()
    figure, axes = plt.subplots(rows, columns, sharey=True, figsize=(7.0, 2.0 * rows + 0.6), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(order):
            axis.set_visible(False)
            continue
        optimizer = order[index]
        points = table[table["optimizer"] == optimizer].sort_values("lr_value")
        if math.isfinite(decision.ref_mean):
            axis.axhspan(decision.ref_mean - decision.margin, decision.ref_mean + decision.margin, color=BAND,
                         linewidth=0, zorder=0)
            axis.axhline(decision.ref_mean, color="black", linewidth=0.8, linestyle="--", zorder=1)
        axis.plot(points["lr_value"], points["val_iou"], "-o", color=LINE, linewidth=1.5, markersize=5,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=2)
        best = points.loc[points["val_iou"].idxmax()]
        axis.plot(best["lr_value"], best["val_iou"], "o", color=BEST, markersize=7.5, markeredgecolor="white",
                  markeredgewidth=0.8, zorder=3)
        axis.set_xscale("log")
        axis.set_xticks(points["lr_value"].tolist(), points["lr"].tolist(), fontsize=7.5)
        axis.minorticks_off()
        low, high = points["lr_value"].min(), points["lr_value"].max()
        axis.set_xlim(low / 1.8, high * 1.8)
        axis.set_title(f"{optimizer}  (best {best['val_iou']:.4f})", fontsize=8.5)
        axis.grid(axis="x", visible=False)
    top, bottom = table["val_iou"].max(), table["val_iou"].min()
    axes.flat[0].set_ylim(bottom - 0.06 * (top - bottom), top + 0.10 * (top - bottom))   # shared: room above the best
    figure.supxlabel("Learning rate (log scale)", fontsize=9, y=0.055)
    figure.supylabel(f"Val IoU (seed {plan.screen_seed}, threshold chosen on val)", fontsize=9)
    figure.subplots_adjust(hspace=0.55, wspace=0.12, bottom=0.115)
    handles = [Line2D([], [], color=LINE, marker="o", markersize=5, linewidth=1.5, markeredgecolor="white"),
               Line2D([], [], color=BEST, marker="o", linestyle="none", markersize=7.5, markeredgecolor="white"),
               Line2D([], [], color="black", linestyle="--", linewidth=0.8), Patch(color=BAND)]
    labels = ["one run per lr", "best lr of the optimizer", f"reference mean ({decision.ref_n} seeds)",
              f"± margin {decision.margin:.3f}"]
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False)
    name = f"{plan.protocol}_optimizers"
    plots.save(figure, out, name)
    return out / f"{name}.png"


def figure_augmentation(plan: Plan, table, decision: Decision, out: Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import plots

    table = table.dropna(subset=["delta_iou"]).sort_values("delta_iou")   # best on top
    if table.empty:
        return None
    plots.setup()
    figure, axis = plt.subplots(figsize=(6.4, 0.27 * len(table) + 1.0))
    axis.axvspan(-decision.margin, decision.margin, color=BAND, linewidth=0, zorder=0)
    axis.axvline(0, color="black", linewidth=0.8, zorder=1)
    colours = [BETTER if d >= decision.margin else WORSE if d <= -decision.margin else NEUTRAL
               for d in table["delta_iou"]]
    axis.barh(range(len(table)), table["delta_iou"], height=0.62, color=colours, edgecolor="white", linewidth=1.0,
              zorder=2)
    axis.set_yticks(range(len(table)), table["change"], fontsize=8)
    axis.set_ylim(-0.6, len(table) - 0.4)
    axis.set_xlabel(f"Δ val IoU vs. reference mean (seed {plan.screen_seed} vs. {decision.ref_n} seeds)")
    axis.grid(axis="y", visible=False)
    handles = [Patch(color=BETTER), Patch(color=NEUTRAL), Patch(color=WORSE)]
    labels = [f"better by ≥ {decision.margin:.3f}", "within the margin (seed noise)", f"worse by ≥ {decision.margin:.3f}"]
    axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
    name = f"{plan.protocol}_augmentation"
    plots.save(figure, out, name)
    return out / f"{name}.png"


def report(plan: Plan, runs_root: Path) -> int:
    import pandas as pd

    from make_report import write_table

    runs = read_runs(plan)
    if runs.empty:
        print(f"No finished runs under {plan.root}")
        return 0
    tables, figures = runs_root / "tables", runs_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    decision = decide(plan, runs)
    lines = [f"Ablation of {plan.model} ({plan.path.name}) - validation only, protocol '{plan.protocol}'",
             "Metric: pooled val IoU at the threshold with the best pooled F1 on val.", ""]
    unevaluated = runs[runs["source"] != "val"]
    if len(unevaluated):
        lines += [f"WARNING: {len(unevaluated)} finished runs have no val evaluation (shown with their training IoU "
                  "at threshold 0.5). Evaluate them: python evaluate.py --split val --protocol "
                  f"{plan.protocol} --no-efficiency", ""]
    lines.append(f"Reference (benchmark protocol): {decision.ref_mean:.4f} ± {decision.ref_std:.4f} "
                 f"({decision.ref_n} seeds). Margin for a positive change: {decision.margin:.4f}.")
    total = sum(len(plan.reference_seeds) if v.name == REFERENCE else 1 for v in plan.screen_variants())
    lines.append(f"Stage 1: {total - len(decision.missing)}/{total} runs done"
                 + (f"; missing: {', '.join(decision.missing[:8])}{' ...' if len(decision.missing) > 8 else ''}"
                    if decision.missing else "."))

    runs.to_csv(tables / f"{plan.protocol}_runs.csv", index=False, encoding="utf-8-sig")
    optimizers = optimizer_table(plan, runs, decision)
    augmentation = augmentation_table(plan, runs, decision)
    write_table(optimizers.drop(columns="lr_value").round(4), tables, f"{plan.protocol}_optimizer",
                f"{plan.model}: optimizers and learning rates (val IoU, seed {plan.screen_seed}; reference "
                f"= mean of {decision.ref_n} seeds)")
    write_table(augmentation.round(4), tables, f"{plan.protocol}_augmentation",
                f"{plan.model}: augmentation changes against the benchmark augmentation (val IoU, seed "
                f"{plan.screen_seed})")

    lines += ["", f"Optimizers (seed {plan.screen_seed}, delta vs reference mean):"]
    for optimizer, group in optimizers.groupby("optimizer", sort=False):
        cells = [f"lr {r.lr}: {r.val_iou:.4f} ({r.delta_iou:+.4f})" if math.isfinite(r.val_iou) else f"lr {r.lr}: -"
                 for r in group.itertuples()]
        lines.append(f"  {optimizer:<16} " + " | ".join(cells))
    lines.append(f"  -> {decision.optimizer_note}")
    lines += [f"  note: {note}" for note in decision.edge_notes]

    lines += ["", f"Augmentation (seed {plan.screen_seed}, delta vs reference mean), best first:"]
    for row in augmentation.itertuples():
        shown = f"{row.delta_iou:+.4f}" if math.isfinite(row.delta_iou) else "   -   "
        lines.append(f"  {shown}  {row.change:<58} {row.verdict}")
    lines.append("  -> positive changes (best per factor): "
                 + (", ".join(v.name for v in decision.positives) if decision.positives else "none"))

    combos = combine_variants(plan, decision)
    lines += ["", "Stage 2 candidates:" if combos else "Stage 2: nothing to combine."]
    for variant in combos:
        done = "done" if plan.done(variant.name, plan.screen_seed) else "not run yet"
        iou = value(runs, variant.name, plan.screen_seed)
        shown = f"{iou:.4f} ({iou - decision.ref_mean:+.4f})" if math.isfinite(iou) else ""
        lines.append(f"  {variant.name:<22} {done:<12} {shown}  {variant.change}")

    ranked, _ = final_pool(plan, runs, decision)
    if ranked:
        lines += ["", f"Stage 3 pool (seed {plan.screen_seed}, best first; the top {plan.final_candidates} with a "
                  "positive delta get the extra seeds):"]
        lines += [f"  {row['variant'].name:<22} {row['iou']:.4f} ({row['delta']:+.4f})" for row in ranked]
        chosen = [row for row in ranked if row["delta"] > 0][:plan.final_candidates]
        for name in final_controls(plan, runs, chosen):
            iou = value(runs, name, plan.screen_seed)
            lines.append(f"  {name:<22} {iou:.4f} ({iou - decision.ref_mean:+.4f})  control (final_controls)")

    result = final_result(plan, runs)
    next_step = None
    if result and len(result["rows"]) > 1:
        rows = []
        for row in result["rows"]:
            subset = runs[(runs["variant"] == row["variant"]) & runs["seed"].isin(plan.reference_seeds)]
            rows.append({"variant": row["variant"], "n_seeds": row["n"],
                         "val_iou": f"{row['mean']:.4f} ± {row['std']:.4f}",
                         "delta_iou": row["mean"] - result["reference"]["mean"],
                         "val_f1": subset["f1"].mean(), "tol_f1_2": subset["tol_f1_2"].mean(),
                         "val_precision": subset["precision"].mean(), "val_recall": subset["recall"].mean()})
        write_table(pd.DataFrame(rows).round(4), tables, f"{plan.protocol}_final",
                    f"{plan.model}: final candidates over seeds {plan.reference_seeds} (val)")
        winner = result["winner"]
        lines += ["", f"Final (seeds {plan.reference_seeds}):"]
        lines += [f"  {r['variant']:<22} {r['mean']:.4f} ± {r['std']:.4f}" for r in result["rows"]]
        if winner == REFERENCE:
            lines.append("  -> the benchmark protocol stays the best recipe; no tuned recipe beats it over three seeds.")
        else:
            gain = result["rows"][0]["mean"] - result["reference"]["mean"]
            significant = math.isfinite(result["p_value"]) and result["p_value"] < 0.05
            lines.append(f"  -> final recipe: {winner}, {gain:+.4f} val IoU over the reference; per seed "
                         + ", ".join(f"{d:+.4f}" for d in result["seed_diffs"])
                         + f"; Wilcoxon on per-tile IoU p = {result['p_value']:.3g}"
                         + ("" if significant else " (NOT significant: the reference recipe is as good)"))
        config = run_config(plan, winner)
        final_path = plan.root / "FINAL.yaml"
        cfg.save_config({"_base_": None, **config}, final_path)
        lines.append(f"  config: {final_path}")
        lines += ["  changes from the benchmark protocol:"]
        lines += [f"    {line}" for line in recipe_diff(config, plan.base)] or ["    none"]
        next_step = "bash slurm/ablation.sh test   (once: final recipe and reference on the test set)"
        test = test_result(plan, list(dict.fromkeys([winner, REFERENCE])))
        if test is not None:
            test_runs = test["runs"]
            summary = test_runs.groupby("variant", sort=False).agg(
                n_seeds=("iou", "size"), test_iou=("iou", "mean"), test_iou_std=("iou", "std"), test_f1=("f1", "mean"),
                tol_f1_2=("tol_f1_2", "mean"), test_precision=("precision", "mean"),
                test_recall=("recall", "mean"), test_ap=("ap", "mean")).reset_index()
            write_table(summary.round(4), tables, f"{plan.protocol}_test",
                        f"{plan.model}: final recipe and reference on the test set over seeds {plan.reference_seeds} "
                        "(threshold chosen on val)")
            lines += ["", f"Test set (once, threshold chosen on val, seeds {plan.reference_seeds}):"]
            for row in summary.itertuples():
                per_seed = ", ".join(f"{v:.4f}" for v in test_runs[test_runs.variant == row.variant].iou)
                lines.append(f"  {row.variant:<22} IoU {row.test_iou:.4f} ± {row.test_iou_std:.4f}  F1 {row.test_f1:.4f}  "
                             f"P {row.test_precision:.4f}  R {row.test_recall:.4f}  (seeds: {per_seed})")
            if test["seed_diffs"]:
                gain = float(summary.test_iou.iloc[0] - summary.test_iou.iloc[1])
                lines.append(f"  -> {winner} vs reference on test: {gain:+.4f} IoU; per seed "
                             + ", ".join(f"{d:+.4f}" for d in test["seed_diffs"])
                             + f"; Wilcoxon on per-tile IoU p = {test['p_value']:.3g}")
            next_step = "done (the test set has been used; do not tune further on it)"
    elif decision.missing:
        next_step = "wait for stage 1, then: bash slurm/ablation.sh combine"
    elif any(not plan.done(v.name, plan.screen_seed) for v in combos):
        next_step = "bash slurm/ablation.sh combine"
    else:
        next_step = "bash slurm/ablation.sh final"

    for path in (figure_optimizers(plan, optimizers, decision, figures),
                 figure_augmentation(plan, augmentation, decision, figures)):
        if path:
            lines.append(f"figure: {path}")
    lines += ["", f"Next step: {next_step}"]
    text = "\n".join(lines)
    (tables / f"{plan.protocol}_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "jobs", "report"))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="configs/ablation/<model>.yaml")
    parser.add_argument("--runs-root", default=None, help="Default: $RUNS_ROOT, else ./runs")
    parser.add_argument("--stage", choices=STAGES, help="jobs: which stage")
    parser.add_argument("--out", help="jobs: job list to write (one 'variant seed config' line per task)")
    parser.add_argument("--only", nargs="*", default=[], help="jobs: keep variants matching these globs, e.g. 'opt_*'")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="jobs: decide from the runs finished so far (not for the thesis numbers)")
    parser.add_argument("--include-active", action="store_true",
                        help=f"jobs: also queue runs written to in the last {ACTIVE_MINUTES} min (normally running)")
    args = parser.parse_args()

    runs_root = cfg.runs_root(args.runs_root)
    plan_path = Path(args.plan)
    if not plan_path.exists() and (REPO_ROOT / plan_path).exists():
        plan_path = REPO_ROOT / plan_path
    plan = Plan(plan_path, runs_root)
    if args.command == "check":
        return check(plan)
    if args.command == "report":
        return report(plan, runs_root)

    if not args.stage or not args.out:
        parser.error("jobs needs --stage and --out")
    selected = jobs(plan, args.stage, args.only, args.allow_incomplete, args.include_active)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(f"{n} {s} {p.as_posix()}\n" for n, s, p in selected), encoding="utf-8",
                              newline="\n")
    for index, (name, seed, _) in enumerate(selected):
        print(f"{index:3d}  {name} seed {seed}", file=sys.stderr)
    hours = float(plan.data.get("hours_per_run", 0) or 0)
    if selected and args.stage != "test" and hours:
        print(f"{len(selected)} training runs, about {len(selected) * hours:.0f} GPU-hours at ~{hours} h per run",
              file=sys.stderr)
    print(len(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
