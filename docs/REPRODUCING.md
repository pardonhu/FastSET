# Reproducing FastSET

Run commands from the repository root. Use a fresh output directory for each run.
The final submitted plots can be rebuilt with `scripts/reproduce_paper.py`; GPU
experiments use the shared `fastset` package and the numerical configurations in
`configs/experiments/`.

## 1. Exact archived A2B inputs and pretrained models

```bash
python scripts/download_assets.py --asset weights --output data/weights
python scripts/download_assets.py --asset a2b-data --output data/a2b_archived
python scripts/run_experiment.py --experiment A2B_d8_w64 \
  --manifest data/a2b_archived/A2B_paper.json --data-root data/a2b_archived \
  --gpu 0 --output results/a2b_001
```

The A2B bundle preserves the original 170 adaptation batches, 100 validation
batches, and 100 PEBS evaluation batches. Every original file is included with its SHA256 unchanged. Batch order is retained.
Source pretraining frames are generated separately; the archived A2B bundle is an
online-adaptation/evaluation dataset. The 8–64 and E2E source states are also
included in Git. The larger Release model bundle includes 40 splits at two source
budgets, Joint JP, alternative backbones, TDL-B and four RT source environments,
and the four immutable main-experiment final models for independent testing.

## 2. Generate another dataset

```bash
python scripts/prepare_data.py --case A2C --scale paper --gpu 0 \
  --output data/tdl --plan-only
python scripts/prepare_data.py --case A2C --scale paper --gpu 0 --output data/tdl
python scripts/run_experiment.py --experiment A2C_CNN \
  --manifest data/tdl/A2C_paper.json --data-root data/tdl \
  --gpu 0 --output results/a2c_e2e_001
```

Generation caches each complete split under a hash of its parameters. Identical
source datasets are reused across cases. Failed or partial splits remain visible
and are not silently treated as complete. TDL generation runs the channel's
linear algebra on CPU, matching the supported MatrixSquareRoot placement, while
neural updates and decoding execute on GPU. New random samples and different
TensorFlow versions can change error counts; the archived input bundle identifies
the exact A2B samples used in the submitted measurements.

For an execution smoke test, use `--scale smoke` during preparation and `--smoke`
on `run_experiment.py`. The latter reduces each active source stage to one epoch
and each online stage to one iteration. The saved configuration records those
smaller budgets.

## 3. Pretrain from scratch

Add `--pretrain` to the experiment command. Source training always uses the
source protocol, including seed42, batch32, and learning rate 1e-3. The online
learning-rate and batch-order ablations do not change source pretraining.
Each epoch saves a model. The final source weights are copied into a fresh online
state with zero Adam iterations and the configured RNG seed.

For an existing TensorFlow source checkpoint with nonzero Adam iterations:

```bash
fastset reset-source --config configs/experiments/A2B_d8_w64.json \
  --manifest data/a2b_archived/A2B_paper.json --data-root data/a2b_archived \
  --source PATH/TO/SOURCE_PREFIX --gpu 0 --output results/reset_source_001
```

The reset checks that every model weight remains unchanged. No training occurs.
Pass the exported `results/reset_source_001/online_initial` to `--source`.

## 4. Offline PEBS selection

```bash
python scripts/run_experiment.py --experiment pebs_d8_w64 \
  --manifest data/a2b_archived/A2B_paper.json --data-root data/a2b_archived \
  --gpu 0 --output results/pebs_d8_w64_001 --phases online timing pebs
fastset pebs --alpha 0.8 --output results/archived_pebs_alpha08.csv
```

Repeat the first command for each `pebs_d*_w*` catalog entry. The online phase
always completes 5 estimator + 20 demapper iterations. `pebs` evaluates every
model with validation BER below 2e-4 on the separate evaluation samples at 12 dB.
The timing phase restores the source model and repeats the recorded update
sequence three times without BER checks or model writes. The second command
recomputes the published ranking from the archived trajectory CSV. Its fixed E2E
reference is iteration 50, evaluation BER 1.7563252272169e-4 and 10.196998199982772 s.

## 5. Ablation, batch size, and training-pool size

Use `--list --group ablation` for the seven methods and three batch-order seeds.
Joint SP starts from the same sequentially pretrained weights as FastSET; Joint
JP starts from separately jointly pretrained weights. Joint updates use MSE+BCE
with unit coefficients and both module gradients. Fixed ablation budgets are 50
A2B and 84 A2C total iterations.

The batch-size experiment reads the same 32-frame storage files. Batch16 uses two
successive halves; batch64 joins two successive files. Estimator and demapper
streams restart as explicitly specified in the config. The data-volume study
uses nested per-SNR subsets and cycles only when a declared small pool is
exhausted. It records first-hit performance and retains the complete 25-iteration
split / 100-iteration E2E trajectory.

## 6. Ray-traced experiments

```bash
python scripts/prepare_data.py --case Paris_P1 --scale paper --gpu 0 \
  --output data/rt
python scripts/run_experiment.py --experiment Paris_P1_d8_w64 \
  --manifest data/rt/Paris_P1_paper.json --data-root data/rt \
  --gpu 0 --output results/paris_001
```

The repository includes the exact 512 source and 9 distinct target channel
responses. `prepare_data.py` uses the original position/role/frame seed derivation
and interference generator. Source frames are synthesized on demand. Adaptation,
validation and test frames are materialized before timing, so random-sample
synthesis is not charged as receiver input processing. The frozen-channel package
requires no city mesh download or new position selection. RT validation uses 256
frames and evaluation uses 3,200 frames per SNR.

## 7. Conventional and source-domain evaluation

```bash
fastset baselines --config configs/experiments/A2B_d8_w64.json \
  --manifest data/a2b_archived/A2B_paper.json --data-root data/a2b_archived \
  --source results/a2b_001/online/iteration_001/checkpoint \
  --split validation --gpu 0 --output results/a2b_baselines_001
```

This evaluates LS, Perfect CSI, and the adapted CNN estimator followed by
LMMSE/APP/LDPC. `fastset evaluate --help` exposes `--split` and `--snrs`. Single-SNR
TDL evaluation inputs are noise-scaled for paired SNR curves; H and bits remain
fixed. A source-domain BER comparison uses a separately generated source-domain
evaluation set and the pretrained models, with no target adaptation.

For the supplementary independent check, prepare A2B/A2C with `--independent` and
evaluate `data/weights/final/CASE/METHOD/online_initial` with `--snrs 12`. Those
four files contain target-adapted final weights and are for evaluation only;
training commands reject their nonzero Adam iteration counters.

## Outputs and resource control

Each phase contains `config.json`, `lineage.json`, `environment.json`, a full log,
CSV results, and `result.json` or `failure.txt`. Online runs also save
`batch_sequence.json` and every model. Integer bit/block errors and denominators
are preserved. The experiment runner's `--resume` skips only complete phases
whose top-level plan is unchanged; an interrupted phase is retained for diagnosis.
Run another phase into a fresh directory when restarting an incomplete phase.

Use separate experiment IDs/output directories on separate GPUs. A cooperative
per-device lock and a current NVIDIA process check prevent this runner from
starting on an occupied device. The runner never terminates another GPU process.
Default CPU thread counts are limited to two. GPU memory grows on demand.

## 8. Collect new measurements

```bash
python scripts/summarize_runs.py --runs results/pebs_d8_w64_001 results/pebs_CNN_001 \
  --output results/new_measurements_001 --alpha 0.8
```

The collector joins integer error counts, saved-model iterations and local timing
into `trajectories.csv`, `first_hits.csv`, `ber.csv`, and PEBS tables when the
required phase outputs are present. PEBS requires the same dataset manifest for
every input and one fixed E2E first-hit reference. Use `fastset plot` to plot a
new CSV. Archived paper plots remain tied to the released measurements.
