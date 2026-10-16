# Reproduction guide

## Three distinct levels

1. **Replot measured results:** install CPU dependencies and run `fastset
   paper-figures`. The bundled numbers are real measured data; no model is run.
2. **Re-evaluate or adapt released checkpoints:** obtain the archived dataset
   payload matching the manifests, then run the CLI. This requires a GPU and
   the pinned environment. New portable runner acceptance is still pending.
3. **Retrain source models and rebuild datasets:** source pretraining is exposed
   by the CLI; TDL generation creates a new dataset version. Exact paper input
   reproduction requires the archived dataset, not merely the same seed.

The release is not yet a complete level-2/3 artifact. A unified TDL suite now
generates the inputs and assembled manifests locally; see `DATA_GENERATION.md`.
The generation-to-results GPU validation, RT scene preparation and specialized
experiment adapters remain listed in `RELEASE_STATUS.md` and `COVERAGE.md`.
The archive-based examples below apply only if the original data is already
available. With newly generated data, substitute `data/tdl_v1/A2B.json` and
`data/tdl_v1` for the manifest and data-root arguments.

## Source pretraining

```bash
fastset pretrain --config configs/a2b_fastset.json \
  --manifest artifacts/manifests/A2B.json --data-root "$FASTSET_DATA_ROOT" \
  --gpu 0 --output results/source_A/run_001
```

The split is randomly initialized with seed 42. An epoch traverses all 3,000
source batches. The source SNR range is 4–18 dB inclusive. The model weights
from the fixed final epoch are retained while Adam state and Sionna RNG are
reset to their online initial states. The resulting `online_initial` checkpoint
is the online source. Estimator pretraining uses channel MSE; demapper
pretraining uses coded-bit BCE while the estimator is frozen.

For CNN, use `configs/a2b_cnn.json`. For the separate Joint-pretraining control,
use `configs/ablation/a2c_joint_pretraining_control.json`. That control updates
both modules using MSE+BCE and a separate Adam per module, as in the measured
campaign. It changes the pretraining treatment and must be labeled separately.

This adapter saves each completed epoch but currently starts new runs only;
automatic resume from an incomplete run has not been ported. Failed runs and
their checkpoints are preserved. Do not claim unattended resumption support.

## Online adaptation

Use `online` with a source-pretrained **online_initial** checkpoint. Loaded Adam
iteration counters must be zero. Every run evaluates its initial state, then
evaluates after each update, including estimator updates. With `early_stop=true`,
BER strictly below the target ends training before another update. A hit at
iteration zero is represented honestly as zero adaptation steps.

Inputs are shuffled with a seeded NumPy `RandomState`. `batch_order` is explicit:

- `restart_each_stage`: demapper starts at the first shuffled batch again.
  This preserves the main/selection experiment convention.
- `global`: all methods consume the same permutation prefix; changing from
  estimator to demapper does not rewind. This is the ablation convention.

The portable adapter refuses budgets longer than the declared single
permutation. It will not silently create a different repeated-data experiment.

## Ablation

Use `configs/ablation/a2c_{fastset,est_only,dem_only,joint,cnn}.json` in separate
runs. All split-method runs use the same 8-64 source checkpoint. CNN uses its
own source-pretrained checkpoint. A2C runs 84 iterations; A2B runs 50. FastSET
uses estimator 5, then demapper for the remaining budget. Other methods update
their declared module(s) throughout. Neither threshold stops these runs.

Do not initialize the inactive module randomly in an `est_only` or `dem_only`
comparison: that changes the question and removes the common pretrained receiver.

## Timing replay

After online adaptation, replay its saved trajectory in a new run directory:

```bash
fastset timing --config configs/a2b_fastset.json \
  --manifest artifacts/manifests/A2B.json --data-root "$FASTSET_DATA_ROOT" \
  --source artifacts/checkpoints/d8_w64/online_initial \
  --trajectory results/a2b/run_001/iterations.csv \
  --gpu 0 --output results/a2b_timing/run_001
```

Configuration, manifest, source checkpoint and source-code hashes must match the
online trajectory. The timer performs no per-iteration BER evaluation. See
`TIMING.md` before comparing the output to a paper table.

## Independent evaluation

Use `evaluate` with the **selected online checkpoint**, a manifest containing an
`evaluation` split, and the same receiver config. It computes BER separately for
each declared SNR and preserves integer error/bit counts. It does not update the
model or choose a new checkpoint on the evaluation set. The bundled A2B/A2C
manifests currently describe source/online/validation inputs, not the full
independent evaluation data; the latter must be exported before full release.

## GPU numerical acceptance before publication

A small runnable software smoke can be prepared without loading the full data:

```bash
python scripts/prepare_smoke.py --config configs/a2b_fastset.json \
  --manifest artifacts/manifests/A2B.json --output results/smoke_inputs
fastset online --config results/smoke_inputs/config.json \
  --manifest results/smoke_inputs/manifest.json --data-root "$FASTSET_DATA_ROOT" \
  --source artifacts/checkpoints/d8_w64/online_initial \
  --gpu 0 --output results/smoke_run
```

The referenced three data files must exist. The smoke runs one estimator and
one demapper iteration and evaluates one 32-frame batch; it is not a paper
result and is not by itself an equivalence test. Use the CNN/Joint configs with
their matching source states to prepare the corresponding two-step checks.

Use an idle GPU, the pinned environment and independent fresh processes for the
existing campaign worker and the new portable worker. Freeze the same source
checkpoint, one training batch and one evaluation batch before starting. Compare:

1. all weight names/order/shapes and exact checkpoint readback;
2. initial channel estimates, LLRs, and decoded bits;
3. one estimator, one demapper, one CNN, and one Joint update: loss, gradients or
   updated weights, with preset `rtol=1e-5, atol=1e-6`;
4. decoded integer bit/block errors;
5. a short complete trajectory with identical batch order and stop decisions.

Do not adjust tolerance after seeing a mismatch. CPU source/AST checks and a
successful package build cannot substitute for this test. Existing campaign
validation remains valid for its original code; it does not automatically
validate newly organized CLI orchestration.

