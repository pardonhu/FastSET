# Experiment coverage

This table separates completed author experiments from public-runner readiness.
“CSV” means measured results are bundled; it does not mean end-to-end reproduction
from a fresh clone has been validated.

| Experiment | CSV/plot support | Portable training support | Remaining release work |
|---|---|---|---|
| A2B/A2C FastSET and CNN | Yes | Shared pretrain/online/evaluate | Exact evaluation data; GPU acceptance |
| 40-split PEBS / depth / width | Yes | 40 configs, shared models | Candidate checkpoints, seed57 evaluation manifests, selection integration |
| A2C ablation (+ A2B supplement) | Yes, five original treatments | Full-budget configs and Joint objective | GPU acceptance; new Joint10 results remain separate |
| Learning-rate / batch diagnostics | Frozen CSVs | No dedicated validated runner | Port final specialized schedules; batch64 unsupported by this fixed-batch runtime |
| Transformer / MLP backbone | Frozen CSVs; model/spec library preserved | Not exposed as a validated CLI | Port final backbone adapter and fixed specs |
| LS / Perfect / other conventional curves | Frozen CSVs | Not exposed by the new evaluation CLI | Port conventional receiver evaluation separately |
| TDL B2C / delay / speed | Frozen CSVs | Shared receiver, new-data TDL generator | Freeze exact per-case manifests/checkpoints |
| RT cities / cross-city / Paris P3 / delay | Frozen CSVs; interference helper preserved | Shared receiver can read same batch schema | Scene assets/license, construction configs and portable RT generator |
| Continuous resident timing decomposition | CSV diagnostics | New CLI implements separated replay only | Port continuous measurement entry point |

The release directory contains no SSH scheduler or server credentials. Parallel
experiments can run as independent CLI processes on explicitly selected GPUs;
they share the same implementation and write separate result directories.
Device leases are cooperative and user-scoped; they do not reserve other users'
jobs or replace a cluster scheduler.

