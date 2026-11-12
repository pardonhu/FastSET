# Validation record — v1.0.0

The machine-readable record is [VALIDATION_v1.json](../artifacts/VALIDATION_v1.json).
The release contains 152 experiment configurations, the submitted plotting
sources and 48 CSV tables, 521 frozen RT responses, and versioned model/data
bundles. The model bundle contains 101 source/final model states. MIT licensing
was selected by the authors.

## Executed checks

| Check | Result and scope |
|---|---|
| CPU protocol/data/artifact regressions | 30 tests passed in a fresh virtual environment |
| Fresh installation | Editable installation and command-line entry point passed |
| Submission plotting | All 16 plotting scripts completed; 53 PDF outputs, including diagnostic layouts |
| Figure data | Integer BER counts and denominators checked; source/CSV hashes recorded |
| FastSET GPU updates | Estimator and demapper steps, frozen-module hashes, LDPC counts passed |
| E2E and Joint | Actual GPU updates and post-LDPC evaluation passed |
| Batch16 and batch64 | Actual GPU updates using split/combined storage batches passed |
| Alternative backbones | Transformer split and MLP E2E small update/evaluation checks passed |
| Source model conversion | All 14 imports restored; weight-preserving Adam/RNG reset checked |
| RT pipeline | Target materialization, source pretraining, online adaptation, three timing replays passed on a small dataset |
| TDL pipeline | Generation, source pretraining, online adaptation and timing passed on a small dataset |
| LS / Perfect CSI / CNNEst | All three baseline paths executed |
| Full A2B validation | 3,200 frames, 26,054,400 bits; 8–64 reached the target at estimator iteration1 |
| Timing replay | Three repetitions; final integer counts matched the online run |
| Timing components | Read, deserialize/check, tensor preparation, compute components sum to measured time |
| PEBS | Maximum qualifying-model selection tested against the archived ranking |
| Package hygiene | No private absolute paths, credentials, symlinks or untracked experiment logs in release files |

The full A2B check observed 3,927 bit errors versus 3,928 in the archived main-paper
measurement. The first-hit iteration and BER conclusion were unchanged. The
validation record retains both counts. A cross-version training replay is not
labeled bit-identical to the historical campaign.

Full source-pretraining budgets for all 152 configurations were not executed
again during repository packaging. The branch checks above validate the portable
execution paths at their stated scales; the final plotting workflow uses the
archived full experimental measurements. GPU training and wall time on a new
machine produce a new versioned experimental result.

## Released assets

- `fastset-models-v1.0.0.tar.gz`: source-pretrained grid models at demapper epochs5
  and10, RT/backbone/Joint/source-B models, and four fixed final models.
- `fastset-a2b-inputs-v1.0.0.tar.gz`: the exact original A2B online, validation and
  PEBS-evaluation files. No regenerated sample is substituted in this bundle.
- The remaining cases have explicit generation recipes; RT propagation
  coefficients are included directly in Git.

The archive names, sizes, SHA256 values and model inventories are recorded under
`artifacts/`. `scripts/download_assets.py` validates downloads before extracting.
