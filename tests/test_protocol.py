import copy
import json
from pathlib import Path
import pytest
from fastset.common import architecture
from fastset.protocol import validate, schedule, batch_indices, first_hit, load_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/a2b_fastset.json").read_text())


def test_all_declared_configs_and_parameter_budget():
    files = [p for p in (ROOT / "configs").rglob("*.json") if p.parent.name != "data"]
    for p in files:
        c = validate(json.loads(p.read_text()))
        batch_indices(c, 170)
    models = [architecture(d, w) for d in range(1, 11) for w in (8, 16, 32, 64)]
    assert len({m["id"] for m in models}) == 40
    assert all(m["parameters"] <= 661188 for m in models)
    selected = architecture(8, 64)
    assert (selected["est_parameters"], selected["dem_parameters"], selected["parameters"]) == (
        260866,
        370004,
        630870,
    )


def test_ablation_does_not_rewind_but_main_does(config):
    main = batch_indices(config, 170)
    assert main[0] == main[5]
    config["batch_order"] = "global"
    full = batch_indices(config, 170)
    assert full[:5] == main[:5]
    assert len(set(full)) == len(full)
    assert full[5] != main[5]


def test_threshold_is_strict_and_checks_initial_and_estimator():
    rows = [
        dict(total_iteration=0, stage="initial", bit_errors=20, bits=100000),
        dict(total_iteration=1, stage="est", bit_errors=19, bits=100000),
    ]
    assert first_hit(rows, 2e-4)["stage"] == "est"
    rows[0]["bit_errors"] = 0
    assert first_hit(rows, 2e-4)["total_iteration"] == 0


def test_budgets_are_iterations_not_epochs(config):
    config["online"]["dem_iterations"] = 20
    assert len(schedule(config)) == 25
    config["online"]["dem_iterations"] = 200
    with pytest.raises(ValueError, match="Budget exceeds"):
        batch_indices(config, 170)


def test_refuses_unmeasured_batchsize_and_unknown_keys(config):
    config["batch_size"] = 8
    with pytest.raises(ValueError, match="batch_size"):
        validate(config)
    config["batch_size"] = 32
    config["typo_stop"] = True
    with pytest.raises(ValueError, match="keys"):
        validate(config)


def test_split_overlap_path_escape_and_hash_mismatch(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "train").write_bytes(b"train")
    (root / "validation").write_bytes(b"validation")
    m = {
        "online_train": [{"path": "train", "snr": 12}],
        "validation": [{"path": "validation", "snr": 12}],
    }
    assert load_manifest(m, root)["test56"][0]["path"] == str(root / "validation")
    overlap = copy.deepcopy(m)
    overlap["validation"][0]["path"] = "train"
    with pytest.raises(ValueError, match="overlap"):
        load_manifest(overlap, root)
    (tmp_path / "outside").write_bytes(b"outside")
    escape = copy.deepcopy(m)
    escape["validation"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(escape, root)
    m["online_train"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        load_manifest(m, root, verify_hashes=True)
