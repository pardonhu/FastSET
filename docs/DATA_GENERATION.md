# Dataset preparation

Use `python scripts/prepare_data.py --case CASE --scale paper --gpu 0 --output DIR`.
The available cases are in `configs/datasets.json`. Add `--plan-only` to inspect
the parameters and estimated TDL storage, or use `--scale smoke` for a short
execution check. The source and standard target sizes are documented in
[DATA_CARD.md](DATA_CARD.md).

Completed TDL splits are cached by their configuration hash. Online data-volume
subsets share the first ten batches per SNR and a single 90-batch extension; the
extension advances the RNG through the original 170 batches first. Per-SNR
prefixes give nested 1/5/10/20/100-batch pools. Source pretraining inputs are shared
across compatible cases. Raw frame data consume disk space; the 96,000-frame
source set alone is approximately 22.2 GB in the original pickle layout.

RT coefficients are bundled in `artifacts/rt`. Target frames are materialized
before adaptation; source frames are synthesized on demand. This preserves the
original role/frame streams while avoiding a separate 22 GB source-frame copy
for each city. RT dataset construction and full commands are in
[REPRODUCING.md](REPRODUCING.md).

The legacy `fastset generate-suite` command remains available for the original
A2B/A2C development workflow. New submission experiments use `prepare_data.py`,
which includes the paired SNR evaluation convention and expanded case catalog.
