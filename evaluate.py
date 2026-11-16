"""Evaluate trained runs. Validation chooses the threshold; the test set only applies it.

    python evaluate.py --split val                 # all finished benchmark runs
    python evaluate.py --split test                # once, at the end (needs the val results)
    python evaluate.py --split val --models unet_r34 --seeds 0
    python evaluate.py --split val --model nnunet --seed 0 --predictions-dir <dir>   # external model
    python evaluate.py --split val --config <run config> --seed 0   # exactly the run train.py writes for it
    python evaluate.py --split val --protocol ablation_deepcrack    # ablation variants (folder = variant)

Per run it writes, under <run>/eval/:
    <split>_metrics.json    pooled metrics, AP, chosen threshold, tolerance F1, clDice
    <split>_per_tile.csv    one row per tile
    <split>_curves.csv      precision / recall / F1 / IoU at 255 thresholds
and per model <model>/efficiency.json (params, FLOPs, time per 512x512 tile, GPU name).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

import config as cfg
import metrics
from models import REGISTRY, get_spec

Tile = tuple[str, np.ndarray, np.ndarray, np.ndarray]   # name, prob_u8, gt {0,1}, RGB image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--protocol", default="benchmark", help="benchmark | original")
    parser.add_argument("--models", nargs="*", default=None, help="Default: every model with runs")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing results")
    parser.add_argument("--include-unfinished", action="store_true", help="Also evaluate runs without FINISHED")
    parser.add_argument("--no-efficiency", action="store_true", help="Skip params / FLOPs / timing")
    parser.add_argument("--model", default=None, help="With --predictions-dir: registry name of the external model")
    parser.add_argument("--seed", type=int, default=0, help="With --predictions-dir or --config: seed of the run")
    parser.add_argument("--config", default=None,
                        help="Evaluate only the run that `train.py --config <this> --seed <n>` writes (no efficiency)")
    parser.add_argument("--predictions-dir", default=None,
                        help="Folder of 8-bit probability PNGs named <tile>.png (in <dir>/<split>/ or <dir>/)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def predict_run(run: Path, split: str, root: Path, workers: int) -> tuple[dict[str, Any], Iterator[Tile]]:
    """(run config, iterator of tiles) for a checkpoint trained by train.py."""
    import torch

    import data
    import engine
    from models import build_model

    engine.use_offline_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    # Built as trained (auxiliary heads included, so the weights load); eval mode returns only the logits.
    model = build_model(config["model"], pretrained=False, config=config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = data.eval_loader(root, split, config, workers)

    def tiles() -> Iterator[Tile]:
        for batch in loader:
            probability = engine.predict(model, batch["image"].to(device), config, bool(config["train"]["amp"]))
            prob_u8 = (probability * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = (batch["image"] * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
            masks = batch["mask"].to(torch.uint8).numpy()
            for index, name in enumerate(batch["name"]):
                yield name, prob_u8[index, 0], masks[index, 0], images[index]

    return config, tiles()


def predict_files(predictions: Path, split: str, root: Path, config: dict[str, Any]) -> Iterator[Tile]:
    """Tiles of an external model: probability PNGs (0..255) named after the tiles."""
    import data

    folder = predictions / split if (predictions / split).is_dir() else predictions
    for name, image_path, mask_path in data.collect_pairs(root, split, config["data"].get("exclude_regex")):
        image, mask = data.read_pair(image_path, mask_path)
        path = folder / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"No prediction for tile {name}: {path}")
        prob_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if prob_u8.shape != mask.shape:
            raise ValueError(f"{path}: prediction {prob_u8.shape} does not match the mask {mask.shape}")
        yield name, prob_u8, mask, image


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def tile_properties(image: np.ndarray, gt: np.ndarray, gt_skeleton: np.ndarray) -> dict[str, float]:
    """Model-independent descriptors used to pick the qualitative examples."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    scale = 256.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    illumination = cv2.GaussianBlur(small, (0, 0), 16)
    skeleton_pixels = int(gt_skeleton.sum())
    return {
        "gt_pixels": int(gt.sum()),
        "gt_width": float(gt.sum() / skeleton_pixels) if skeleton_pixels else 0.0,   # mean crack width, px
        "brightness": float(gray.mean()),
        "illumination_std": float(illumination.std()),   # shadows / stains: uneven low-frequency brightness
    }


def score(tiles: Iterator[Tile], split: str, tolerances: list[int], threshold_index: int | None) -> dict[str, Any]:
    """All metrics of one run on one split. `threshold_index` None: choose it here (validation)."""
    import data

    stored: list[Tile] = list(tiles)
    positive = np.zeros(metrics.LEVELS, np.int64)
    negative = np.zeros(metrics.LEVELS, np.int64)
    for _, prob_u8, gt, _ in stored:
        tile_positive, tile_negative = metrics.histograms(prob_u8, gt)
        positive += tile_positive
        negative += tile_negative
    curve = metrics.curves(positive, negative)
    chosen_here = threshold_index is None
    if chosen_here:
        threshold_index = metrics.best_threshold_index(curve)

    def records_at(index: int, with_properties: bool) -> list[dict[str, Any]]:
        rows = []
        for name, prob_u8, gt, image in stored:
            gt_skeleton = metrics.skeleton(gt)
            row = {"name": name, "group": data.source_group(name)}
            row.update(metrics.tile_record(prob_u8, gt, index, tolerances, gt_skeleton))
            if with_properties:
                row.update(tile_properties(image, gt, gt_skeleton))
            rows.append(row)
        return rows

    records = records_at(threshold_index, True)
    pooled = metrics.pooled_metrics(records, tolerances)
    half_index = 128
    half = pooled if threshold_index == half_index else metrics.pooled_metrics(records_at(half_index, False), tolerances)
    groups: dict[str, dict[str, float]] = {}
    for group in sorted({r["group"] for r in records}):
        members = [r for r in records if r["group"] == group]
        counts = [sum(r[key] for r in members) for key in ("tp", "fp", "fn")]
        groups[group] = {"n_tiles": len(members), **metrics.scores(*counts)}
    return {
        "split": split, "n_tiles": len(records),
        "threshold": threshold_index / 255.0, "threshold_index": threshold_index,
        "threshold_source": f"best pooled F1 on {split}" if chosen_here else "chosen on val",
        "ap": metrics.average_precision(curve),
        "metrics": pooled, "metrics_at_0.5": {k: half[k] for k in ("iou", "f1", "precision", "recall")},
        "per_group": groups, "records": records, "curve": curve,
    }


def write_results(result: dict[str, Any], out: Path, header: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split = result["split"]
    records, curve = result.pop("records"), result.pop("curve")
    with (out / f"{split}_per_tile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (out / f"{split}_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(curve))
        writer.writerows(zip(*[np.round(curve[key], 6) for key in curve]))
    (out / f"{split}_metrics.json").write_text(json.dumps({**header, **result}, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #
def count_flops(model, example) -> tuple[float, str]:
    """(FLOPs in G for one 512x512 tile, tool). fvcore if installed, else thop, else NaN."""
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, example)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return analysis.total() / 1e9, "fvcore"
    except ImportError:
        pass
    try:
        from thop import profile

        operations, _ = profile(copy.deepcopy(model), inputs=(example,), verbose=False)
        return operations / 1e9, "thop"
    except ImportError:
        return float("nan"), "not installed (pip install thop)"


def measure_efficiency(name: str, config: dict[str, Any], warmup: int = 10, repeats: int = 50) -> dict[str, Any]:
    """Parameters, FLOPs and mean inference time of one 512x512 tile (batch 1, FP32)."""
    import torch

    import engine
    from models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, pretrained=False, config={"recipe": {"deep_supervision": False}}).to(device).eval()
    example = torch.rand(1, 3, 512, 512, device=device)
    result: dict[str, Any] = {"model": name, "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                              "input": "1x3x512x512", "device": engine.device_name(device)}
    try:
        with torch.no_grad():
            result["flops_g"], result["flops_tool"] = count_flops(model, example)
    except Exception as exc:   # a custom operator the counter cannot trace
        result["flops_g"], result["flops_tool"] = float("nan"), f"failed: {type(exc).__name__}"
    if device.type != "cuda":
        warmup, repeats = 1, 3
    times = []
    with torch.no_grad():
        for index in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if index >= warmup:
                times.append((time.perf_counter() - started) * 1000.0)
    result.update(time_ms=float(np.mean(times)), time_ms_std=float(np.std(times)), repeats=repeats)
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_threshold(out: Path) -> int:
    path = out / "val_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: evaluate --split val first (the threshold is chosen on val).")
    return int(json.loads(path.read_text(encoding="utf-8"))["threshold_index"])


def header_for(name: str, seed: int, protocol: str, run_name: str | None = None) -> dict[str, Any]:
    spec = get_spec(name)
    header = {"model": name, "seed": seed, "protocol": protocol, "family": spec.family, "tier": spec.tier,
              "label": spec.label}
    if run_name and run_name != name:
        header["run_name"] = run_name
    return header


def registry_name(run: Path) -> str | None:
    """Model of a run: its folder name, or for an ablation variant the model in its config.yaml."""
    if run.parent.name in REGISTRY:
        return run.parent.name
    path = run / "config.yaml"
    if path.exists():
        model = (cfg.load_yaml(path) or {}).get("model")
        if model in REGISTRY:
            return model
    return None


def evaluate_run(run: Path, name: str, seed: int, args: argparse.Namespace, root: Path, workers: int) -> None:
    """Score one run on args.split (skipped when already done, unless --overwrite)."""
    out = run / "eval"
    if (out / f"{args.split}_metrics.json").exists() and not args.overwrite:
        print(f"[skip] {run.parent.name} seed {seed} {args.split}: already evaluated (use --overwrite)")
        return
    threshold = None if args.split == "val" else load_threshold(out)
    started = time.time()
    config, tiles = predict_run(run, args.split, root, workers)
    result = score(tiles, args.split, list(config["eval"]["tolerances"]), threshold)
    write_results(result, out, header_for(name, seed, config.get("protocol", args.protocol), run.parent.name))
    print(f"[ok] {run.parent.name} seed {seed} {args.split}: IoU {result['metrics']['iou']:.4f} "
          f"F1 {result['metrics']['f1']:.4f} AP {result['ap']:.4f} @ threshold {result['threshold']:.3f} "
          f"({time.time() - started:.0f}s)", flush=True)


def main() -> int:
    args = parse_args()
    runs = cfg.runs_root(args.runs_root)
    root = cfg.data_root(args.data_root)
    workers = args.workers if args.workers is not None else cfg.num_workers()

    if args.predictions_dir:
        if not args.model:
            raise SystemExit("--predictions-dir needs --model")
        run = cfg.run_dir(runs, args.protocol, args.model, args.seed)
        out = run / "eval"
        if (out / f"{args.split}_metrics.json").exists() and not args.overwrite:
            print(f"[skip] {args.model} seed {args.seed} {args.split}: already evaluated (use --overwrite)")
            return 0
        config = cfg.load_config(cfg.CONFIG_DIR / "base.yaml")
        threshold = None if args.split == "val" else load_threshold(out)
        result = score(predict_files(Path(args.predictions_dir), args.split, root, config), args.split,
                       list(config["eval"]["tolerances"]), threshold)
        write_results(result, out, header_for(args.model, args.seed, args.protocol))
        (run / "FINISHED").write_text(json.dumps({"model": args.model, "seed": args.seed, "external": True}))
        print(f"[ok] {args.model} seed {args.seed} {args.split}: IoU {result['metrics']['iou']:.4f} "
              f"F1 {result['metrics']['f1']:.4f} @ threshold {result['threshold']:.3f}")
        return 0

    if args.config:
        config = cfg.load_config(args.config)
        run = cfg.run_dir(runs, config["protocol"], config.get("run_name") or config["model"], args.seed)
        if not (run / "best.pt").exists():
            raise SystemExit(f"No best.pt in {run}: train it first")
        evaluate_run(run, config["model"], args.seed, args, root, workers)
        return 0

    base = runs / args.protocol
    candidates = sorted(p for p in base.glob("*/seed_*") if (p / "best.pt").exists()) if base.is_dir() else []
    failures = 0
    done_models: set[str] = set()
    for run in candidates:
        folder, seed = run.parent.name, int(run.name.split("_")[1])
        if (args.models and folder not in args.models) or (args.seeds is not None and seed not in args.seeds):
            continue
        name = registry_name(run)
        if name is None:
            print(f"[skip] {folder}: not in the registry")
            continue
        if not (run / "FINISHED").exists() and not args.include_unfinished:
            print(f"[skip] {folder} seed {seed}: training not finished")
            continue
        try:
            evaluate_run(run, name, seed, args, root, workers)
            efficiency_path = run.parent / "efficiency.json"
            if not args.no_efficiency and name not in done_models and (args.overwrite or not efficiency_path.exists()):
                efficiency = measure_efficiency(name, {})
                efficiency_path.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
                print(f"     {name}: {efficiency['params_m']:.1f} M params, {efficiency['flops_g']:.1f} GFLOPs, "
                      f"{efficiency['time_ms']:.1f} ms / 512x512 tile on {efficiency['device']}", flush=True)
            done_models.add(name)
        except Exception:
            failures += 1
            print(f"[FAIL] {name} seed {seed} {args.split}:\n{traceback.format_exc()}", flush=True)
    if not candidates:
        print(f"No runs with best.pt under {base}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
