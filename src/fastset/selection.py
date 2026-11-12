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
    if rows and "val2_ber" in rows[0]:
        return summarize_trajectories(rows, output, alpha)
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


def summarize_trajectories(rows, output, alpha):
    """Select the highest score over eligible est/dem models for each split.

    The reference is one fixed E2E first-hit model. Candidate validation is
    performed on val1; BER in the score is measured on separate val2 samples.
    Ties prefer less adaptation time, then fewer iterations, as in the paper.
    """
    from .experiment import csvsave

    refs = [r for r in rows if r["candidate"] == "CNN"]
    if len(refs) != 1:
        raise ValueError("Expected one fixed E2E reference model")
    ref = refs[0]
    groups = {}
    for row in rows:
        if float(row["val1_ber"]) >= 2e-4:
            continue
        value = score(
            float(row["val2_ber"]),
            float(row["total_seconds"]),
            float(ref["val2_ber"]),
            float(ref["total_seconds"]),
            alpha,
        )
        result = dict(
            candidate=row["candidate"],
            alpha=alpha,
            eligible=True,
            pebs=value,
            total_iteration=int(row["total_iteration"]),
            stage=row["stage"],
            val1_ber=float(row["val1_ber"]),
            val2_ber=float(row["val2_ber"]),
            total_seconds=float(row["total_seconds"]),
        )
        key = (value, -result["total_seconds"], -result["total_iteration"])
        if row["candidate"] not in groups or key > groups[row["candidate"]][0]:
            groups[row["candidate"]] = (key, result)
    csvsave(output, [groups[c][1] for c in sorted(groups)])
