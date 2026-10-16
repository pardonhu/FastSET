"""Orchestration regression tests with a deterministic non-neural test double."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from fastset import experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "early_stop,initial_errors,expected_steps", [(True, 0, 0), (True, 50, 1), (False, 50, 7)]
)
def test_actual_online_loop_checks_initial_and_estimator(
    monkeypatch, tmp_path, early_stop, initial_errors, expected_steps
):
    config = json.loads((ROOT / "configs/a2b_fastset.json").read_text())
    config["online"].update(est_iterations=5, dem_iterations=2, early_stop=early_stop)

    class Part:
        def __init__(self):
            self.value = np.zeros(1)

        def get_weights(self):
            return [self.value.copy()]

    class Checkpoint:
        def write(self, prefix):
            Path(prefix + ".index").write_bytes(b"test")
            return prefix

    parts = {"est": Part(), "dem": Part()}
    opts = {k: SimpleNamespace(iterations=0) for k in parts}
    calls = []

    def step(stage, batch):
        calls.append(stage)
        parts[stage].value += 1
        opts[stage].iterations += 1
        return {"loss": 1.0}

    def evaluate(entries):
        errors = 1 if calls else initial_errors
        return dict(bit_errors=errors, bits=100000, ber=errors / 100000)

    manifest = {"online_train": [{"path": f"batch{i}"} for i in range(10)], "test56": [{}]}
    monkeypatch.setattr(
        experiment,
        "setup",
        lambda args: (
            config,
            tmp_path,
            None,
            None,
            parts,
            opts,
            Checkpoint(),
            manifest,
            step,
            evaluate,
        ),
    )
    from fastset import model_runtime

    monkeypatch.setattr(model_runtime, "checked", lambda value: value)
    experiment.online(SimpleNamespace())
    assert len(calls) == expected_steps
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["first_hit"]["total_iteration"] == (0 if initial_errors == 0 else 1)
    if early_stop:
        assert opts["dem"].iterations == 0
    else:
        assert calls == ["est"] * 5 + ["dem"] * 2
