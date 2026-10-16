import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_artifact_hashes():
    for item in json.loads((ROOT / "artifacts/INDEX.json").read_text()):
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_measurements_keep_integer_counts_and_no_fake_zero_floor():
    checked = 0
    for path in (ROOT / "artifacts/tables").glob("*.csv"):
        with path.open() as f:
            for row in csv.DictReader(f):
                if not all(row.get(k) for k in ("bit_errors", "bits", "ber")):
                    continue
                errors, bits, ber = int(row["bit_errors"]), int(row["bits"]), float(row["ber"])
                assert 0 <= errors <= bits and bits > 0
                assert abs(errors / bits - ber) < 1e-12
                if errors == 0:
                    assert ber == 0
                checked += 1
    assert checked > 1000


def test_initial_states_and_budget_are_present_in_ablation():
    rows = list(csv.DictReader((ROOT / "artifacts/tables/ablation_iterations.csv").open()))
    for case, budget in (("A2B", 50), ("A2C", 84)):
        for method in ("fastset", "est_only", "dem_only", "joint", "cnn"):
            subset = [r for r in rows if r["scenario"] == case and r["method"] == method]
            assert [int(r["total_iteration"]) for r in subset] == list(range(budget + 1))
            assert all(int(r["bits"]) == 26054400 for r in subset)


def test_import_and_help_do_not_initialize_tensorflow():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fastset.cli,sys; assert 'tensorflow' not in sys.modules; fastset.cli.parser().parse_args(['doctor'])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_recorded_model_semantics_unchanged():
    from fastset.semantic import fingerprint

    hashes = json.loads((ROOT / "artifacts/semantic_fingerprints.json").read_text())
    for relative, expected in hashes.items():
        assert fingerprint((ROOT / "src/fastset" / relative).read_text()) == expected
