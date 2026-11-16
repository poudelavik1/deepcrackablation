# Crack segmentation baseline benchmark

Baseline benchmark for binary concrete-crack segmentation (crack vs background, many
cracks 1-4 px wide) on tiles of drone orthomosaics. Its purpose is to **select one
model**; ablation studies are added as configs, not code (the DeepCrack augmentation x optimizer
ablation: `configs/ablation/deepcrack.yaml`, run with `slurm/ablation.sh`, see SERVER_GUIDE.md section 15).

Code only: the dataset, checkpoints and results are never committed.
Running on the cluster: see [SERVER_GUIDE.md](SERVER_GUIDE.md).

## What is compared

All models are in one registry, [models/registry.py](models/registry.py). Adding one is
one line there, then `python scripts/make_configs.py`.

| Family | Models | Tier |
|---|---|---|
| my-existing | `hairline_unet`, `swin_unet` (architectures ported unchanged) | core |
| CNN | `unet_r34`, `unet_r50`, `unet_effb4`, `unetpp_r34`, `deeplabv3p_r34`, `fpn_r34`, `unet_r34_scse` | core |
| CNN | `unet_convnext_t`, `linknet_r34`, `manet_r34` | tier1 |
| Transformer | `segformer_b2` | tier1 |
| crack-specific | `deepcrack` (Liu et al., 2019) | tier1 |
| crack-specific | `crackformer` (official code, cloned) | tier2 |
| Mamba | `scsegamba` (official code, own env) | tier2 |
| auto-configured | `nnunet` (nnU-Net v2, 2D, own pipeline, same split) | tier2 |

## The protocol (identical for every model)

Defined once in [configs/base.yaml](configs/base.yaml); a per-model config only names
the model.

- Existing `split_80_10_10` split. Random 256x256 training crops; one fixed augmentation
  (flips, 90-degree rotations, mild affine, mild brightness/contrast/HSV, light blur/noise).
- AdamW, 2 warm-up epochs then cosine, 80 epochs, batch 16 (or the same effective batch by
  gradient accumulation), BCE + Dice, AMP, gradient clipping, early stopping on val IoU.
- Seeds 0, 1, 2. ImageNet-pretrained encoders where the model has one.
- Auxiliary outputs (deep supervision) are not supervised, no EMA, no TTA.
- Evaluation on whole tiles at native size; images larger than 1024 px by sliding window.

The **original recipes** of `hairline_unet` and `swin_unet` (their loss, EMA, deep
supervision, TTA) are in `configs/original/` and are reported in a separate table.

## Evaluation

- Model selection uses the **validation** set only. The threshold is the one with the
  best pooled F1 on validation; the test set is evaluated once with that threshold.
- IoU, F1/Dice, precision, recall (pooled over all pixels); average precision and the
  PR curve (255 thresholds); F1 against threshold; tolerance F1 at 0/1/2/3/5 px
  (distance transform); clDice; per-tile metrics in CSV; IoU per source group.
- Parameters, FLOPs, and GPU time per 512x512 tile (with the GPU model).
- Ranking by mean val IoU over seeds (rank by tolerance F1 @ 2 px alongside), Wilcoxon
  signed-rank test on per-tile val IoU, and a flag when the top two are not clearly
  different, in which case the faster/smaller one is recommended.

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                 # synthetic data, every model, 1 epoch, figures + tables
```

```bash
export DATA_ROOT=/path/to/split_80_10_10 RUNS_ROOT=/path/to/runs
python train.py --config configs/unet_r34.yaml --seed 0
python evaluate.py --split val
python make_report.py --qualitative          # tables + figures 1-8
python evaluate.py --split test              # once, at the end
```

## Layout

```
config.py        YAML loading (base + per-model override + --set key=value)
data.py          tile pairs, augmentation, loaders
models/          registry.py + one file per non-smp model
losses.py        BCE+Dice (benchmark), HairlineCrackLoss (original recipe)
optimizers.py    optimizer by name (train.optimizer; AdamW in the benchmark, Lion implemented here)
metrics.py       all metrics (numpy)
engine.py        seeding, EMA, whole-tile and sliding-window prediction
train.py         one (model, seed) run; resumable; skips finished runs
evaluate.py      val / test metrics, per-tile CSV, curves, efficiency
results.py       loads result files (pandas only)
plots.py         figures 1-8
make_report.py   tables (CSV + LaTeX), ranking, significance, figures
configs/         base.yaml, <model>.yaml, original/<model>.yaml, ablation/<model>.yaml (ablation plans)
scripts/         dataset inspection, env check, weights, job list, smoke test, Tier 2 installers
slurm/           array job, evaluation, smoke test, nnU-Net, SCSegamba
```

Outputs: `$RUNS_ROOT/benchmark/<model>/seed_<n>/` (`best.pt`, `last.pt`, `history.csv`,
`config.yaml`, `FINISHED`, `eval/`), `$RUNS_ROOT/figures/`, `$RUNS_ROOT/tables/`.

