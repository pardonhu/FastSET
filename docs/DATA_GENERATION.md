# Generate datasets locally

The distribution policy is **generation scripts, not hosted raw datasets**.
The TDL suite creates source A, online B/C, validation and evaluation splits,
then assembles `A2B.json` / `A2C.json` with relative paths and file SHA256 values.
RT scene generation is a separate unfinished adapter; this command does not
pretend that a TDL channel is ray-traced urban data.

Install the GPU dependencies as described in the README, then inspect the plan
without allocating a GPU or writing data:

```bash
fastset generate-suite --targets B C --scale paper --gpu 0 \
  --output data/tdl_v1 --plan-only
```

Generate the same *dataset sizes and domain parameters* as the declared protocol:

```bash
fastset generate-suite --targets B C --scale paper --gpu 0 --output data/tdl_v1
```

The full A/B/C suite contains 6,540 batches, approximately **48 GB uncompressed**:
source A 3,000; each target online 170, validation 100, evaluation 1,500. Source A
is stored once and shared by both assembled manifests. Every batch has 32 frames.
Channel generation retains the reference generator's CPU placement, including
MatrixSquareRoot, while startup verifies the selected GPU environment. This is
not a neural CPU fallback. Each split has a separate fresh seeded process.

For a small software check, explicitly select `--scale smoke`: 14 total batches
for A/B/C, about 104 MB, at 12 dB. This reduced dataset is not a paper experiment
and cannot run the full 50/84-iteration protocol. Use `scripts/prepare_smoke.py`
to prepare a matching two-iteration model check.

```bash
fastset generate-suite --targets B --scale smoke --gpu 0 --output data/smoke
python scripts/prepare_smoke.py --config configs/a2b_fastset.json \
  --manifest data/smoke/A2B.json --output results/smoke_inputs
fastset online --config results/smoke_inputs/config.json \
  --manifest results/smoke_inputs/manifest.json --data-root data/smoke \
  --source artifacts/checkpoints/d8_w64/online_initial \
  --gpu 0 --output results/smoke_run
```

Each child split retains its generation config, seed, hashes, environment and log.
`--resume` can reuse completed splits only when configs match; it does not silently
overwrite an interrupted split. Incomplete output stays available for diagnosis.

## What reproduction means here

The new source/target draws follow the configured distributions and sizes. They
are not asserted to be the paper's original sample sequence: seed alone does not
fix the entire RNG call history or cross-version GPU arithmetic. The suite's
manifest explicitly records `bit_identical_to_paper_data=false`.

Use bundled measured CSVs to reproduce the recorded plots exactly in data values.
Use generated inputs to rerun the declared method/protocol, then report the new
measured values. Matching all published BER points or selected architectures
requires an actual end-to-end validation, which is still pending for this public
development snapshot. Never label newly generated values as the archived results.
