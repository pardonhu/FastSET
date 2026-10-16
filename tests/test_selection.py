import csv
from pathlib import Path
import pytest
from fastset.selection import score


def test_pebs_reproduces_all_frozen_scores():
    p = Path(__file__).resolve().parents[1] / "artifacts/tables/pebs_first_hit.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    ref = next(r for r in rows if r["candidate"] == "CNN")
    for r in rows:
        value = score(
            float(r["test57_ber"]),
            float(r["total_seconds"]),
            float(ref["test57_ber"]),
            float(ref["total_seconds"]),
            0.8,
        )
        assert value == pytest.approx(float(r["pebs_alpha_0_8"]), rel=1e-12)


def test_no_invented_ber_for_zero_errors():
    with pytest.raises(ValueError, match="pseudocount"):
        score(0, 0.1, 2e-4, 1, 0.8)
