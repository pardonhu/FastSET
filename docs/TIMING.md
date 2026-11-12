# Adaptation computation time

The receiver is resident and warmed before measurement. Timing begins immediately
before adaptation inputs are read. File reading, deserialization and checks,
tensor preparation/transfer, optimizer updates, and device synchronization are
included. BER/LDPC evaluation, model loading, compilation, model saving, and log
writing are outside the timed interval. Normal OS file caching is retained.

`fastset timing` restores the same source weights and zero-state online Adam,
then executes the exact recorded batch plan three times. Both stages and input
indices are checked against the online trajectory. The code, config, input
manifest and source model hashes must match. The final integer BER counts are
compared with the online run and reported explicitly. Repetitions also compare
final weights. Each cumulative result uses the components of the trial with
median cumulative total, so component values sum to the reported time.

`fastset timing-breakdown` accepts the same arguments. For the CNN/split models,
`timing_repeats.csv` separates `read_seconds`, `deserialize_check_seconds`,
`tensor_prepare_seconds`, and `compute_seconds`. These add to the per-iteration
elapsed time. A separate instrumented replay generates a timing decomposition;
its total can differ from the minimally instrumented measurement. The archived
Table II measurements remain available in `paper/tables/` and `plot_timing.py`.

The paper's archived trajectories and Table II were measured on one RTX 3090 Ti.
Source training performed on other GPUs does not supply published time values.
A reproduction on another machine should report that machine and its own times.
Use the original A2B pickle bundle for comparable input handling; compressed NumPy
input files add decompression to the measured input processing.

The supplementary checking-frequency figure is an estimate. It adds the declared
mean BER-evaluation cost (16.39 s/check) to recorded training/input time according
to each checking interval. `plot_check_frequency.py` retains the arithmetic and
labels; the estimate is separate from measured receiver computation time. The
fixed-time figure associates each saved model's BER with cumulative measured
adaptation time and does not invent intermediate trained models.
