"""PEBS statistics from frozen measured checkpoints; never retrains a model."""

import math


def score(ber, seconds, reference_ber, reference_seconds, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if not all(
        math.isfinite(v) and v > 0 for v in (ber, seconds, reference_ber, reference_seconds)
    ):
        raise ValueError(
            "PEBS ratios require positive measured BER and time; no pseudocount is inserted"
        )
    return alpha * reference_ber / ber + (1 - alpha) * reference_seconds / seconds


def summarize(path, output, alpha):
    from .plotting import read_csv
    from .experiment import csvsave

    rows = read_csv(path)
    references = [r for r in rows if r["candidate"] == "CNN"]
    if len(references) != 1:
        raise ValueError("Expected exactly one fixed CNN reference row")
    ref = references[0]
    result = []
    for row in rows:
        if row["validation_reached"] != "True":
            result.append(dict(candidate=row["candidate"], alpha=alpha, eligible=False, pebs=""))
            continue
        value = score(
            float(row["test57_ber"]),
            float(row["total_seconds"]),
            float(ref["test57_ber"]),
            float(ref["total_seconds"]),
            alpha,
        )
        result.append(dict(candidate=row["candidate"], alpha=alpha, eligible=True, pebs=value))
    csvsave(output, result)
