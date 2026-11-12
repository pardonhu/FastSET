"""Submission-protocol regressions; CPU only."""

import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from fastset.common import order
from fastset.protocol import validate, batch_plan
from fastset.selection import summarize

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return json.loads((ROOT / f"configs/experiments/{name}.json").read_text())


def test_catalog_resolves():
    catalog = json.loads((ROOT / "configs/experiments.json").read_text())
    cases = json.loads((ROOT / "configs/datasets.json").read_text())
    assert len([x for x in catalog if x.startswith("pebs_d")]) == 40
    for item in catalog.values():
        validate(json.loads((ROOT / item["config"]).read_text()))
        assert item["case"] in cases
    for case in ("A2B", "A2C"):
        assert len([x for x in catalog if x.startswith("ablation_" + case)]) == 21


def test_batch_frame_grouping_and_stage_restart():
    ids = order(170, 1, 42)[0]
    p = batch_plan(config("batch_size_16"), 170)
    assert p[0] == dict(indices=[ids[0]], start=0, stop=16)
    assert p[1] == dict(indices=[ids[0]], start=16, stop=32)
    assert p[5] == p[0]
    p = batch_plan(config("batch_size_64"), 170)
    assert p[0]["indices"] == ids[:2] and p[1]["indices"] == ids[2:4]
    assert p[5] == p[0]


def test_rt_stream_and_explicit_small_pool_cycles():
    p = batch_plan(config("Paris_P1_d8_w64"), 50)
    assert [r["indices"] for r in p[:6]] == [[0], [1], [2], [3], [4], [0]]
    c = config("volume_1_CNN")
    p = batch_plan(c, 17)
    assert len(p) == 100
    assert [r["indices"][0] for r in p] == sum(order(17, 6, 42), [])[:100]
    c["batch_order"] = "global"
    with pytest.raises(ValueError):
        batch_plan(c, 17)


def test_maximum_pebs_is_not_first_hit(tmp_path):
    out = tmp_path / "pebs.csv"
    summarize(ROOT / "paper/tables/plot_pebs_DATA_CSV.csv", out, 0.8)
    rows = list(csv.DictReader(out.open()))
    best = max(rows, key=lambda r: float(r["pebs"]))
    assert best["candidate"] == "d8_w64"
    assert float(best["pebs"]) == pytest.approx(19.06, abs=0.01)
    r = next(r for r in rows if r["candidate"] == "d4_w64")
    assert int(r["total_iteration"]) > 3


def test_archived_integer_counts():
    checked = 0
    for p in (ROOT / "paper/tables").glob("*.csv"):
        for row in csv.DictReader(p.open()):
            for prefix in ("", "val1_", "val2_"):
                keys = [prefix + k for k in ("bit_errors", "bits", "ber")]
                if not all(row.get(k) not in (None, "") for k in keys):
                    continue
                e, b, v = map(float, [row[k] for k in keys])
                assert e.is_integer() and b.is_integer() and 0 <= e <= b and b > 0
                assert v == pytest.approx(e / b, abs=1e-14)
                checked += 1
    assert checked > 5000


def test_all_rt_channels_are_valid():
    index = json.loads((ROOT / "artifacts/rt/channels.json").read_text())
    seen = set()
    for city in ("paris", "munich", "florence", "san_francisco"):
        assert len(index["source/" + city]) == 128
    for entries in index.values():
        for e in entries.values():
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            with np.load(ROOT / e["path"], allow_pickle=False) as f:
                h = f[f"h{e['row']}"]
            assert h.shape == (14, 264) and np.isfinite(h).all() and np.any(h != 0)
    assert len(seen) == 521


def test_generation_stream_roles():
    spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = json.loads((ROOT / "configs/datasets.json").read_text())
    roles = {j["config"]["role"]: j["config"] for j in mod.tdl_jobs(cases["A2B"], "paper")}
    assert roles["source_train"]["batches_per_snr"] == 200
    assert roles["validation"]["seed"] == 56 and roles["evaluation"]["seed"] == 57
    assert roles["validation"]["snrs_db"] == [12]
    assert roles["online_train"]["snrs_db"] == list(range(4, 21))


def test_explicit_source_is_never_replaced(tmp_path):
    import subprocess
    import sys

    manifest = tmp_path / "data.json"
    manifest.write_text("{}")
    source = tmp_path / "user_selected_model"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_experiment.py"),
            "--experiment",
            "A2B_d8_w64",
            "--manifest",
            str(manifest),
            "--data-root",
            str(tmp_path),
            "--gpu",
            "0",
            "--output",
            str(tmp_path / "run"),
            "--source",
            str(source),
            "--plan-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["source"] == str(source.resolve())
