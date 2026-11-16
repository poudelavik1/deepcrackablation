# Server guide (ASL Slurm cluster)

Login: `ssh asl.ait.ac.th` (host `ait-slurm`, port 44065, user `se-st126157`). Every command below runs **on the server, in
the repo folder** (`cd ~/baselinecheck`), unless it says "laptop". In a new login shell, load conda first:
`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate swinunet`. The dataset stays at
`~/dataset_20261113/split_80_10_10` and is never copied into the repo.

## 1. Get the code

```bash
ssh se-st126157@ait-slurm
git clone https://github.com/poudelavik1/baselinecheck.git ~/baselinecheck    # first time
cd ~/baselinecheck && git pull                                                # afterwards
```

## 2. Settings (once)

```bash
cp slurm/env.example.sh slurm/env.sh
nano slurm/env.sh        # ENV_NAME, ENV_DIR, DATA_ROOT, RUNS_ROOT, WEIGHTS_CACHE, PARTITION
```

`slurm/env.sh` is ignored by git. Every sbatch script reads it.

## 3. Environment

Use your existing env (`<ENV_NAME>`) if it has PyTorch, or create a new one:

```bash
conda env create -f environment.yml -n crackbench        # new
conda env update -f environment.yml -n <ENV_NAME>        # or: update an existing env
conda activate <ENV_NAME>
pip install -r requirements.txt                          # same thing, pip only
```

Check it (paste both outputs back when asked):

```bash
bash scripts/check_env.sh 2>&1 | tee check_env_login.txt
srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
     env ENV_DIR=$HOME/.conda/envs/<ENV_NAME> bash scripts/check_env.sh 2>&1 | tee check_env_gpu.txt
python scripts/inspect_dataset.py --data-root ~/dataset_20261113/split_80_10_10 | tee inspect_dataset.txt
```

The jobs do not `conda activate` (conda is not on the compute nodes): they put
`$ENV_DIR/bin` on `PATH`, like the earlier swinunet jobs.

## 4. Pre-download the pretrained weights (login node, needs internet)

```bash
source slurm/env.sh && conda activate $ENV_NAME
python scripts/download_weights.py          # fills $WEIGHTS_CACHE, then re-checks with the network off
```

Jobs run with `OFFLINE=1`, so a missing weight fails immediately instead of hanging.

## 5. Smoke test on the cluster

```bash
sbatch slurm/smoke_test.sbatch              # 1 epoch on ~20 real tiles, all models, figures + tables
tail -f logs/crackbench_smoke_<jobid>.out   # ends with "SMOKE TEST: n/n steps ok"
```

## 6. Choose the batch size and estimate the time

```bash
srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:40:00 \
     env PATH=$HOME/.conda/envs/<ENV_NAME>/bin:$PATH python scripts/estimate_compute.py | tee estimate.txt
```

It prints memory, steps/s and hours per run for every model, and GPU-hours per tier for
3 seeds. If a model does not fit at batch 16, put this in its `configs/<model>.yaml`
(same effective batch): `train: {batch_size: 8, accum_steps: 2}`. `train.py` also does
this by itself when it hits an out-of-memory error.

## 7. Submit the benchmark

```bash
bash slurm/submit.sh --tiers core tier1          # Tier 0 + Tier 1, seeds 0 1 2 (one array job)
bash slurm/submit.sh                              # everything in the main env (adds CrackFormer if installed)
bash slurm/submit.sh --families CNN               # one family
bash slurm/submit.sh --models unet_r34 --seeds 0  # one run
DRY_RUN=1 bash slurm/submit.sh --tiers tier1      # only print the job list
EVAL=1 bash slurm/submit.sh --tiers core tier1    # also queue evaluation + report after the array
```

The array size is the length of the generated job list, so it follows the registry.
`MAX_PARALLEL` in `slurm/env.sh` is the throttle (`--array=0-N%K`). Change the time limit
with `SBATCH_ARGS="--time=24:00:00" bash slurm/submit.sh ...`.

## 8. Monitor

```bash
squeue -u $USER
sacct -j <array_job_id> --format=JobID%20,JobName,State,Elapsed,ExitCode
tail -f logs/crackbench_<array_job_id>_<task>.out
grep -l "exit 1\|Traceback\|CANCELLED\|TIME LIMIT" logs/crackbench_<array_job_id>_*.out   # failed tasks
tail -n 3 $RUNS_ROOT/benchmark/*/seed_*/history.csv                                       # progress
ls $RUNS_ROOT/benchmark/*/seed_*/FINISHED | wc -l                                         # finished runs
```

## 9. Resubmit failed or timed-out tasks

```bash
bash slurm/submit.sh --missing --tiers core tier1
```

`--missing` lists only runs without a `FINISHED` file. A run killed by the time limit
resumes from `last.pt` (model, optimizer, scheduler, scaler, epoch); a finished run is
skipped. Submitting the same list twice is therefore always safe.

## 10. Evaluate, tables, figures

```bash
sbatch slurm/evaluate.sbatch                                        # validation: metrics, ranking, figures 1-8
sbatch --dependency=afterok:<array_job_id> slurm/evaluate.sbatch    # chained: only if every task succeeded
sbatch --dependency=afterany:<array_job_id> slurm/evaluate.sbatch   # chained: whatever finished
cat $RUNS_ROOT/tables/report.txt                                    # ranking, winner, significance
```

**Test set, once, at the very end** (after the model is chosen on validation):

```bash
sbatch --export=ALL,RUN_TEST=1 slurm/evaluate.sbatch
```

`evaluate.py` will not recompute existing test results unless you pass `--overwrite`.

## 11. Original-recipe runs (separate table)

```bash
bash slurm/submit.sh --protocol original          # hairline_unet and swin_unet with their own recipe
```

They are written to `$RUNS_ROOT/original/` and reported in `tables/original_recipe.*`,
never in the benchmark table. Results of your earlier scripts (`test_metrics.json`,
`training_history.csv`, `val_threshold_sweep.csv`) found below `PREVIOUS_RUNS`
(`slurm/env.sh`) appear in `tables/previous_runs.*` as "previous run, not same protocol".

## 12. Tier 2

Each one is isolated: a failure does not affect anything else.

**CrackFormer** (main env; official code cloned, not copied):

```bash
bash scripts/install_third_party.sh crackformer
bash slurm/submit.sh --models crackformer
```

**SCSegamba** (own env; Mamba needs compiled CUDA kernels):

```bash
bash scripts/install_scsegamba.sh
srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 \
     $HOME/.conda/envs/scsegamba/bin/python scripts/test_scsegamba.py      # must end with PASS
sbatch slurm/scsegamba.sbatch                                              # seeds 0 1 2, trains + evaluates val
```

If the install or the test fails, skip it and send me the output.

**nnU-Net v2** (own env; same split, scored by `evaluate.py`):

```bash
bash scripts/install_nnunet.sh
sbatch slurm/nnunet.sbatch                 # convert -> plan -> train fold 0 (= my split) -> predict val -> evaluate
sbatch slurm/nnunet.sbatch                 # resubmit to continue after the time limit
sbatch --export=ALL,RUN_TEST=1 slurm/nnunet.sbatch      # at the very end
```

After SCSegamba / nnU-Net finish, rerun `sbatch slurm/evaluate.sbatch` to add them to the
tables and figures.

## 13. Copy the results to the laptop (Windows / PowerShell)

Only figures, tables, CSV and JSON: no checkpoints. Run **on the laptop**.

With `rsync` (Git Bash or WSL):

```bash
rsync -avm --exclude='nnunet/' --exclude='_smoke/' --exclude='joblists/' \
  --include='*/' --include='*.csv' --include='*.json' --include='*.yaml' --include='*.txt' \
  --include='*.tex' --include='*.png' --include='*.pdf' --include='FINISHED' --exclude='*' \
  asl.ait.ac.th:crack_benchmark_runs/ ./runs/
```

Without rsync (PowerShell; packs the small files on the server, then copies one archive):

```powershell
ssh asl.ait.ac.th "cd ~/crack_benchmark_runs && find . \( -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.txt' -o -name '*.tex' -o -name '*.png' -o -name '*.pdf' -o -name FINISHED \) -not -path './nnunet/*' -not -path './_smoke/*' | tar czf ~/results.tgz -T -"
scp asl.ait.ac.th:results.tgz .
New-Item -ItemType Directory -Force runs | Out-Null
tar -xzf results.tgz -C runs
```

Then, on the laptop, figures 1-7 and all tables are rebuilt from those files (no GPU, no
dataset):

```powershell
python make_report.py --runs-root runs
```

Figure 8 (qualitative grid) needs the dataset and checkpoints, so it is rendered on the
server by `slurm/evaluate.sbatch` and arrives with the rsync.

## 14. Protocol v2 (current) and the v1 results

v1 (16 Nov 2026, `~/crack_benchmark_runs`) is kept for reference but not used for the
selection: without weight averaging, val IoU of the pretrained models swung by up to 0.3
between epochs, early stopping cut them at 16-40 epochs on that noise, and MA-Net died on
an fp16 NaN. v2 changes, identical for every model (`configs/base.yaml`, `train.py`):

- EMA of the weights (`recipe.ema_decay: 0.999`) for validation and `best.pt`;
- early stopping: patience 20, and never before `train.min_epochs: 30`;
- a non-finite loss skips that step; more than max(20, 2 % of the steps) in one epoch stops the run.

v2 writes to its own folder, so v1 is never mixed in:

```bash
git pull
sed -i 's|RUNS_ROOT="$HOME/crack_benchmark_runs"|RUNS_ROOT="$HOME/crack_benchmark_runs_v2"|' slurm/env.sh && grep RUNS_ROOT slurm/env.sh
bash slurm/submit.sh --models deeplabv3p_r34 unet_convnext_t --seeds 0       # pilot: the two noisiest in v1
```

When the pilot has finished, check that val IoU no longer swings
(`cut -d, -f1,7 $RUNS_ROOT/benchmark/deeplabv3p_r34/seed_0/history.csv`), then run the rest
(`--missing` skips the pilot runs; `tier2` adds CrackFormer-II, the only Tier 2 model of the main env):

```bash
EVAL=1 bash slurm/submit.sh --missing --tiers core tier1 tier2
```

## 15. Ablation of DeepCrack: augmentation and optimizer

Plan: [configs/ablation/deepcrack.yaml](configs/ablation/deepcrack.yaml). Results go to
`$RUNS_ROOT/ablation_deepcrack/<variant>/seed_<n>/`; every training task also evaluates its run on
val, so no separate evaluation job is needed. One run is ~1.1 h, and the cluster runs one job at a time.

| Stage | What runs | Runs | Time |
|---|---|---|---|
| 1 `screen` | reference (benchmark protocol) x seeds 0 1 2; 9 optimizers x 3 learning rates; 24 augmentation changes (none, leave-one-out, stronger, 14 added steps, CutMix), each seed 0 | 53 | ~2.5 days |
| 2 `combine` | best optimizer + top-1 / top-3 / all augmentation changes that beat the reference by the margin (0.003, or the reference seed std if larger) | <= 4 | ~4 h |
| 3 `final` | seeds 1 2 for the 2 best candidates; the best over 3 seeds vs. the reference (Wilcoxon on per-tile val IoU) is the final recipe | <= 4 | ~4 h |

```bash
git pull
bash slurm/ablation.sh check          # every variant builds (CPU, ~1 min); `screen` runs it too
bash slurm/ablation.sh screen         # stage 1
bash slurm/ablation.sh report         # any time: tables, figures, decisions, and the next command to run
bash slurm/ablation.sh combine        # when stage 1 is done (it refuses earlier)
bash slurm/ablation.sh final          # when stage 2 is done
bash slurm/ablation.sh test           # ONCE at the very end: final recipe + reference on the test set
python scripts/progress.py --protocol ablation_deepcrack     # per-run progress
```

Resubmitting a stage queues only the runs that are not finished and evaluated, and leaves out runs
written to in the last 30 min (running now), so after a time-out just run the same command again.
To trim stage 1: `bash slurm/ablation.sh screen --only ref 'aug_*'`.

**Two runs per GPU job.** The account may run one job at a time (`AssocMaxJobsLimit` in `squeue`), and
one DeepCrack run leaves the A6000 partly idle. `PACK=2` trains two runs side by side in every array
task (16 CPUs, 48 GB; set `CPUS_PER_RUN` / `MEM_PER_RUN_GB` if the job waits on `Resources`). Each run
logs to `logs/ablation_<job>_<task>.<run>.out`; the task log records GPU load every 10 min. To switch
an array that is already queued, cancel only its pending tasks and resubmit (the running run is kept):

```bash
scancel --state=PENDING <array job id>
PACK=2 bash slurm/ablation.sh screen
```
`report` writes `tables/ablation_deepcrack_{report.txt,optimizer,augmentation,final}.*`,
`figures/ablation_deepcrack_{optimizers,augmentation}.{png,pdf}` and the winning config to
`$RUNS_ROOT/ablation_deepcrack/FINAL.yaml` (its differences from the benchmark are listed in the report).
