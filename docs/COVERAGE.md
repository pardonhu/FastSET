# Experiment coverage

The current submission is indexed in [EXPERIMENTS.md](EXPERIMENTS.md). It covers
the main manuscript, supplementary experiments, and the experimental comparisons
in the Summary of Differences. The source/evaluation/data-generation workflows are
in [REPRODUCING.md](REPRODUCING.md).

`paper/` is the current publication snapshot. `artifacts/tables/` retains the
older development snapshot for provenance and regression checks; use `paper/`
for the final submitted curves and PEBS ranking. The `fastset paper-figures`
command now invokes the current submission figure runner.

Actual execution checks and their scope are listed in
[RELEASE_STATUS.md](RELEASE_STATUS.md).
