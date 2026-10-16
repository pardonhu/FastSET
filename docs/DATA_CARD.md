# Data card

## Link and supervision

The shared receiver uses a single-stream OFDM link, 264 subcarriers, 14 OFDM
symbols, pilot symbols 2 and 11, 16-QAM, and LDPC decoding. One stored batch
contains 32 frames. Each frame has 8,142 information bits and 12,672 coded bits.
100 stored batches therefore contain 3,200 frames and **26,054,400 information
bits**. BER reports decoded information-bit errors, not coded-bit BCE.

Estimator adaptation uses simulator channel labels and demapper adaptation uses
transmitted coded-bit labels. This is a supervised adaptation experiment; it is
not a demonstration that these labels are available without overhead in a deployed
receiver. The true channel argument is not used as an estimator forward input.

## TDL domains

| Domain | Model | Delay spread | Speed range |
|---|---|---|---|
| A | TDL-A | 50 ns | 0–1.5 m/s |
| B | TDL-B | 500 ns | 0–5 m/s |
| C | TDL-C | 1,000 ns | 0–10 m/s |

Source training: 200 batches/SNR at 4–18 dB, 3,000 batches total.
Online training: 10 batches/SNR at 4–20 dB, 170 batches total.
Validation: 100 batches at 12 dB, historical seed 56 stream.
Independent evaluation uses separate data (including seed 57 in PEBS). Seed 32
diagnostics are not substituted for the frozen paper protocol.

The `snr` field retains the experiment's axis label. The generator supplies it
to Sionna's `ebnodb2no`; it is the generator's Eb/N0 control, not an independently
measured post-channel receive SNR. Do not silently reinterpret the axis.

## Files and manifests

Legacy arrays are read with a restricted NumPy pickle reader. Arbitrary Python
objects, nonfinite arrays and trailing content are rejected. Pickles remain
large untrusted inputs; consume only distributed, verified artifacts.

Required keys are `ebno_db`, `batch_b`, `batch_tx_codeword_bits`, `batch_x_rg`,
`batch_pilots_rg`, `batch_h_freq`, `batch_y`, `batch_N0`, and `batch_h_ls_est`.
Each manifest entry declares a relative path, SNR, and a SHA256 when available.
Absolute paths and paths escaping the data root are rejected.

`artifacts/manifests/A2B.json` and `A2C.json` preserve existing batch order and
online/validation file hashes. The historical source-training inventory did
not hash every source file; these hashes are **missing, not fabricated**.
`--verify-hashes` rejects such an incomplete archival inventory. The unified
generation suite hashes every newly written file and creates complete manifests;
use those manifests when rerunning with generated data.

No dataset payload is included in this code directory. Source training alone
is approximately 22.2 GB in the current uncompressed pickle format. The distribution uses generation scripts instead of hosted raw datasets;
see `DATA_GENERATION.md`. Do not commit the full data tree into Git history.

## Generation versus exact reproduction

`fastset generate --config configs/data/B_validation.json --gpu 0 --output
results/new_B_validation` creates a **new dataset version**. Equal seeds are
insufficient to promise equality to archived data: earlier generator calls,
SNR traversal and library versions affect RNG streams. Never mix fresh inputs
and archived results under the same dataset identifier.

Generation outputs one role per run. To combine roles, preserve each run's
relative directory under one data root and prefix the manifest paths accordingly.
Use disjoint validation/evaluation seeds and never use evaluation data to stop.

## Ray tracing

RT CSVs include Paris P2/P3, other-city target conditions, cross-city adaptation,
and delay experiments. These are controlled constructed conditions, including
channel transformations and interference, not untouched measured urban traces.
Selected receiver positions and difficulty settings were explored during study
design; do not claim blind environment selection.

Exact scene meshes, positions, ray-tracing parameters, channel transformations,
interference parameters and per-role RNG streams must be released together.
Their packaging and asset licensing are not yet complete in this candidate.
The generic receiver can consume RT batches in the same schema, but this does
not make the full RT scene-generation pipeline publicly reproducible yet.

