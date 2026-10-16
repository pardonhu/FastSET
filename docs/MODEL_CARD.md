# Model card

## Released initial states

| Model | Estimator parameters | Demapper parameters | Total |
|---|---:|---:|---:|
| FastSET 8-64 | 260,866 | 370,004 | 630,870 |
| CNN | — | — | 661,188 |

The split identifier describes estimator depth and initial width. Demapper
depth/width follow the original parameter-budget rule in `common.architecture`.
The inherited channel-growth rule is preserved, including its depth-dependent
behavior; it is not replaced by a new smoother architecture family.

The bundled checkpoints contain source-trained weights and zero online Adam
iteration counters. Model identifiers, hashes and original archive identities
are in `artifacts/INDEX.json`. These are the source states used by the existing
8-64/CNN experiment family, not newly pretrained or target-selected replacements.

Pretraining settings: estimator 5 epochs / demapper 10 epochs, CNN 10 epochs,
batch32, seed42, Adam lr `1e-3`, epsilon `1e-7`. The new Joint-pretraining control
is a separate run and is not silently merged into these checkpoint files.

## Intended use and limits

Research reproduction under the documented simulated/constructed channel
conditions. A validation threshold at 12 dB is not a guarantee at other SNRs,
for unseen channel distributions, or on physical radio hardware. Timing excludes
specific costs as documented in `TIMING.md`; it is not a deployment deadline.

The 40-split configuration grid is included. Only 8-64 and CNN initial weights
are bundled in this release candidate; the remaining pretrained candidates
require an external checkpoint artifact release or retraining. Do not call a
40-config directory a 40-checkpoint release.

