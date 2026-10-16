# Timing definitions

Do not use elapsed time between logged BER points as training time. LDPC/BER
evaluation may dominate that interval, and Joint performs more work per iteration
than an estimator-only update.

## Existing artifact families

- Ablation `total_seconds`: median cumulative training replay plus separately
  measured median cumulative input time, three repetitions. Input includes
  file read, deserialization and validation. Training includes tensor conversion,
  device transfer, forward/backward pass, Adam update and synchronization.
- PEBS `total_seconds`: the corresponding frozen input-inclusive timing table.
  Its original compute-only field remains visible for provenance.
- `timing_summary.csv` and `timing_all_repeats.csv`: later continuous resident
  timing diagnostics with explicit read/deserialize/tensor/update components.
  These may differ from the historical main-table timing. They are separate
  measurements, not values to mix or average with the earlier table.

The source CSV field names and provenance must remain attached to a plotted
comparison. Use the same timing definition and hardware for all methods in
that comparison.

## Portable `fastset timing`

The new adapter implements the ablation-style separated input/training replay.
It measures actual input work for the selected batches, warms graph execution,
restores the same source checkpoint before each of three repetitions, and
synchronizes the GPU around updates. It reports per-iteration samples and the
sum of the two cumulative medians. This sum is **not** a continuous end-to-end
latency measurement. The ordinary operating-system page cache is left intact.

It excludes BER/LDPC evaluation, compilation warmup, checkpoint and log writing,
data generation, network transfers and initial model construction. Final BER is
checked outside the timer; any integer-count difference from the online run is
reported, not hidden. Cross-version/device timing replays may need numerical
investigation before they support a comparison.

The portable adapter does not implement the newer continuous component-timing
diagnostic as an identical replacement. That specialized adapter remains a
release gap. Original measured CSVs are included unchanged numerically.

