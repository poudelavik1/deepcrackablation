"""Train one model with one seed under the benchmark protocol.

    python train.py --config configs/unet_r34.yaml --seed 0 --data-root $DATA_ROOT

Writes to $RUNS_ROOT/<protocol>/<model>/seed_<n>/: best.pt (best val IoU), last.pt
(model + optimizer + scheduler + scaler + epoch, for resuming), history.csv, config.yaml
and FINISHED. A finished run is skipped; an unfinished one resumes from last.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import torch

import config as cfg
import data
import engine
import metrics
from losses import build_loss, split_outputs
from models import build_model, get_spec
from optimizers import build_optimizer

HISTORY_FIELDS = ["epoch", "lr", "train_loss", "train_iou", "train_f1", "val_loss", "val_iou", "val_f1",
                  "val_precision", "val_recall", "seconds", "batch_size", "accum_steps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="configs/<model>.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None, help="Default: $DATA_ROOT")
    parser.add_argument("--runs-root", default=None, help="Default: $RUNS_ROOT, else ./runs")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. train.epochs=10 train.batch_size=8")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on ~20 tiles")
    parser.add_argument("--no-pretrained", action="store_true", help="Random initialization (no downloads)")
    parser.add_argument("--workers", type=int, default=None, help="Default: SLURM_CPUS_PER_TASK")
    parser.add_argument("--force", action="store_true", help="Train even if the run is finished")
    return parser.parse_args()


def make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train = config["train"]
    factor = float(config["recipe"].get("encoder_lr_factor", 1.0))
    encoder = getattr(model, "encoder", None)
    if factor != 1.0 and encoder is not None:
        encoder_ids = {id(p) for p in encoder.parameters()}
        groups = [
            {"params": [p for p in model.parameters() if id(p) in encoder_ids], "lr": train["lr"] * factor},
            {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": train["lr"]},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": train["lr"]}]
    return build_optimizer(groups, train)   # train.optimizer, AdamW in the benchmark


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to lr * min_lr_ratio. Stepped once per epoch."""
    import math

    train = config["train"]
    epochs, warmup, floor = int(train["epochs"]), float(train["warmup_epochs"]), float(train["min_lr_ratio"])

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1.0, epochs - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def cutmix(images: torch.Tensor, masks: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix for segmentation: with probability p per sample, a box of another tile in the
    batch (image and mask together) replaces the same box. Box area ~ U(0, 1) of the crop.
    Random numbers come from the seeded CPU generator, so runs stay reproducible."""
    count, _, height, width = images.shape
    if count < 2 or p <= 0:
        return images, masks
    chosen = (torch.rand(count) < p).nonzero().flatten().tolist()
    if not chosen:
        return images, masks
    images, masks = images.clone(), masks.clone()
    sources = torch.randperm(count).tolist()
    for index in chosen:
        source = sources[index] if sources[index] != index else (index + 1) % count
        side = float(torch.rand(1)) ** 0.5            # box side ratio; area ratio ~ U(0, 1)
        box_h, box_w = max(1, round(height * side)), max(1, round(width * side))
        top = int(torch.randint(0, height - box_h + 1, (1,)))
        left = int(torch.randint(0, width - box_w + 1, (1,)))
        images[index, :, top:top + box_h, left:left + box_w] = images[source, :, top:top + box_h, left:left + box_w]
        masks[index, :, top:top + box_h, left:left + box_w] = masks[source, :, top:top + box_h, left:left + box_w]
    return images, masks


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, config, epoch, ema, accum_steps) -> dict[str, float]:
    model.train()
    amp = bool(config["train"]["amp"])
    clip = float(config["train"].get("grad_clip", 0) or 0)
    epochs = int(config["train"]["epochs"])
    threshold = float(config["eval"]["threshold"])
    cutmix_p = float(config["aug"].get("cutmix_p", 0) or 0)
    total_loss, batches, skipped = 0.0, 0, 0
    tp = fp = fn = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if cutmix_p > 0:
            images, masks = cutmix(images, masks, cutmix_p)
        with engine.autocast(device, amp):
            outputs = model(images)
        progress = (epoch + step / max(1, len(loader))) / epochs
        loss = loss_fn(outputs, masks, progress)   # the losses cast logits to float32 themselves
        if not torch.isfinite(loss):
            # An fp16 overflow in one batch should not kill the run (MA-Net in v1): skip the step.
            # Many of them means the run has diverged: stop with an error.
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if skipped > max(20, len(loader) // 50):
                raise FloatingPointError(f"Non-finite loss in {skipped} steps of epoch {epoch + 1}: the run diverged")
            print(f"[skip] non-finite loss at epoch {epoch + 1}, step {step} ({skipped} this epoch)", flush=True)
            continue
        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        total_loss += float(loss.detach())
        batches += 1
        with torch.no_grad():
            pred = torch.sigmoid(split_outputs(outputs)[0].detach().float()) >= threshold
            truth = masks >= 0.5
            tp += float((pred & truth).sum())
            fp += float((pred & ~truth).sum())
            fn += float((~pred & truth).sum())
    score = metrics.scores(tp, fp, fn)
    return {"train_loss": total_loss / max(1, batches), "train_iou": score["iou"], "train_f1": score["f1"]}


@torch.no_grad()
def validate(model, loader, loss_fn, device, config) -> dict[str, float]:
    """Pooled val IoU/F1 at the fixed threshold, and the loss on whole tiles."""
    model.eval()
    amp = bool(config["train"]["amp"])
    threshold = float(config["eval"]["threshold"])
    tp = fp = fn = 0.0
    loss_sum, pixels = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        probability = engine.predict(model, images, config, amp).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(probability) - torch.log1p(-probability)
        weight = masks.numel()
        loss_sum += float(loss_fn(logits, masks, 1.0)) * weight
        pixels += weight
        pred, truth = probability >= threshold, masks >= 0.5
        tp += float((pred & truth).sum())
        fp += float((pred & ~truth).sum())
        fn += float((~pred & truth).sum())
    score = metrics.scores(tp, fp, fn)
    return {"val_loss": loss_sum / max(1, pixels), "val_iou": score["iou"], "val_f1": score["f1"],
            "val_precision": score["precision"], "val_recall": score["recall"]}


def append_history(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def truncate_history(path: Path, last_epoch: int) -> None:
    """Drop rows after `last_epoch` (a job may die between writing the CSV and the checkpoint)."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) <= last_epoch]
    path.unlink()
    for row in rows:
        append_history(path, row)


def main() -> int:
    args = parse_args()
    engine.use_offline_weights()
    overrides = list(args.set)
    if args.smoke_test:
        overrides += ["train.epochs=1", "train.warmup_epochs=0", "train.batch_size=4", "train.eval_batch_size=4",
                      "data.max_train_samples=20", "data.max_eval_samples=8"]
    if args.no_pretrained:
        overrides.append("train.pretrained=false")
    config = cfg.load_config(args.config, overrides)
    name = config["model"]
    spec = get_spec(name)
    config["seed"] = args.seed
    config["model_info"] = {"family": spec.family, "tier": spec.tier, "label": spec.label}

    out = cfg.run_dir(cfg.runs_root(args.runs_root), config["protocol"], config.get("run_name") or name, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    finished, last_path, best_path, history_path = out / "FINISHED", out / "last.pt", out / "best.pt", out / "history.csv"
    if finished.exists() and not args.force:
        print(f"[skip] {name} seed {args.seed}: already finished ({out})")
        return 0

    root = cfg.data_root(args.data_root)
    engine.seed_everything(args.seed, bool(config["train"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = args.workers if args.workers is not None else cfg.num_workers()
    print(f"[run] {name} | seed {args.seed} | protocol {config['protocol']} | device {engine.device_name(device)} "
          f"| workers {workers}\n      data {root}\n      out  {out}", flush=True)

    model = build_model(name, bool(config["train"]["pretrained"]), config).to(device)
    loss_fn = build_loss(config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = engine.grad_scaler(device, bool(config["train"]["amp"]))
    ema_decay = float(config["recipe"].get("ema_decay", 0) or 0)
    ema = engine.ModelEMA(model, ema_decay) if ema_decay > 0 else None

    batch_size, accum_steps = int(config["train"]["batch_size"]), int(config["train"]["accum_steps"])
    state = {"epoch": 0, "best_iou": -1.0, "best_epoch": 0, "bad_epochs": 0}
    if last_path.exists() and not args.force:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.module.load_state_dict(checkpoint["ema"])
            ema.updates = checkpoint["ema_updates"]
        state = checkpoint["state"]
        batch_size, accum_steps = checkpoint["batch_size"], checkpoint["accum_steps"]
        torch.set_rng_state(checkpoint["torch_rng"])
        truncate_history(history_path, state["epoch"])
        print(f"[resume] from epoch {state['epoch']} (best val IoU {state['best_iou']:.4f} at epoch {state['best_epoch']})")
    else:
        for stale in (history_path, best_path, last_path, finished):
            stale.unlink(missing_ok=True)
    cfg.save_config(config, out / "config.yaml")

    loader = data.train_loader(root, config, args.seed, workers, batch_size)
    val = data.eval_loader(root, "val", config, workers)
    print(f"      train tiles {len(loader.dataset)} | val tiles {len(val.dataset)} | "
          f"batch {batch_size} x accum {accum_steps} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M",
          flush=True)

    epochs, patience = int(config["train"]["epochs"]), int(config["train"]["patience"])
    min_epochs = int(config["train"].get("min_epochs", 0) or 0)
    stopped_early = False
    while state["epoch"] < epochs:
        epoch = state["epoch"]
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            train_stats = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, config, epoch, ema, accum_steps)
        except engine.OutOfMemoryError:
            if batch_size <= 1:
                raise
            # Same effective batch with less memory: halve the batch, double the accumulation.
            batch_size, accum_steps = batch_size // 2, accum_steps * 2
            print(f"[oom] retrying epoch {epoch + 1} with batch {batch_size} x accum {accum_steps}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            del loader
            torch.cuda.empty_cache()
            loader = data.train_loader(root, config, args.seed, workers, batch_size)
            continue
        eval_model = ema.module if ema is not None else model
        val_stats = validate(eval_model, val, loss_fn, device, config)
        lr = optimizer.param_groups[-1]["lr"]
        scheduler.step()
        state["epoch"] = epoch + 1

        improved = val_stats["val_iou"] > state["best_iou"]
        if improved:
            state.update(best_iou=val_stats["val_iou"], best_epoch=epoch + 1, bad_epochs=0)
            engine.atomic_save({"model": eval_model.state_dict(), "config": config, "epoch": epoch + 1,
                                "val_iou": val_stats["val_iou"]}, best_path)
        else:
            state["bad_epochs"] += 1
        seconds = time.time() - started
        append_history(history_path, {"epoch": epoch + 1, "lr": lr, **train_stats, **val_stats, "seconds": round(seconds, 1),
                                      "batch_size": batch_size, "accum_steps": accum_steps})
        engine.atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "ema": ema.module.state_dict() if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else 0, "state": state, "config": config,
            "batch_size": batch_size, "accum_steps": accum_steps, "torch_rng": torch.get_rng_state(),
        }, last_path)
        memory = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device.type == "cuda" else ""
        print(f"epoch {epoch + 1:3d}/{epochs} | lr {lr:.2e} | train loss {train_stats['train_loss']:.4f} "
              f"IoU {train_stats['train_iou']:.4f} | val loss {val_stats['val_loss']:.4f} IoU {val_stats['val_iou']:.4f} "
              f"F1 {val_stats['val_f1']:.4f}{' *' if improved else ''} | {seconds:.0f}s{memory}", flush=True)
        if state["bad_epochs"] >= patience and state["epoch"] >= min_epochs:
            stopped_early = True
            print(f"[early stop] no val IoU improvement for {patience} epochs")
            break

    summary = {"model": name, "seed": args.seed, "protocol": config["protocol"], "best_val_iou": state["best_iou"],
               "best_epoch": state["best_epoch"], "epochs_run": state["epoch"], "stopped_early": stopped_early,
               "batch_size": batch_size, "accum_steps": accum_steps, "device": engine.device_name(device),
               "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    finished.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {name} seed {args.seed}: best val IoU {state['best_iou']:.4f} at epoch {state['best_epoch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
