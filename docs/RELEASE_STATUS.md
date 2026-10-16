# Release status: candidate 0.1.0rc1

## Included

- Installable `src/` Python package and CPU/GPU dependency separation.
- Shared receiver definitions, deterministic bootstrap, TDL generator,
  pretraining, online, checkpoint evaluation and separated timing replay.
- Explicit 40-split and full-budget ablation configs; no permanent path injection.
- Measured CSVs, two real source checkpoints, relative dataset inventories,
  provenance hashes, and CPU figure reproduction.
- CPU tests, GitHub Actions workflow, citation draft and data/model/timing docs.

## Must be completed before describing this as a full paper release

1. Author-selected repository/data/checkpoint licenses and upstream attribution.
2. Validate the unified TDL generation-to-training workflow in a clean GPU
   environment. Raw datasets will not be uploaded; generation scripts are the
   distribution path. Archived manifests remain provenance, not download promises.
3. Release remaining 40-split checkpoint artifacts with architecture/lineage.
4. Numerical GPU acceptance of the portable orchestration in the pinned
   environment. Original experiment validation is not renamed as this test.
5. Complete specialized adapters in `COVERAGE.md`, especially RT construction
   assets and continuous timing, before claiming all experiments run from clone.
6. Freeze final manuscript-to-artifact map after the concurrent manuscript and
   Joint-pretraining work finishes. Current CSVs are a documented snapshot.
7. Add a validated release tag, archival DOI and final paper citation when
   actually available. Do not insert invented links, venues or badges.

Repository: https://github.com/pardonhu/FastSET. This is an incremental public
development snapshot, not a claim that reviewers can already reproduce all
paper experiments from a fresh clone. See `artifacts/VALIDATION.json` for the
checks actually performed. The running experiment environment and schedules are
independent of this repository.
