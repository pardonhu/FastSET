import json
from pathlib import Path
from types import SimpleNamespace
from fastset.dataset_suite import plan, assemble, run
from fastset.common import save

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_scale_and_disjoint_role_seeds():
    full = plan(ROOT / "configs/data", ["B", "C"], "paper")
    assert sum(j["batches"] for j in full) == 6540
    assert full[0]["batches"] == 3000
    for job in full:
        assert (
            job["config"]["seed"]
            == {"source_train": 42, "online_train": 42, "validation": 56, "evaluation": 57}[
                job["role"]
            ]
        )
    assert sum(j["batches"] for j in plan(ROOT / "configs/data", ["B", "C"], "smoke")) == 14


def test_assembled_manifests_do_not_mix_target_domains(tmp_path):
    jobs = plan(ROOT / "configs/data", ["B", "C"], "smoke")
    for job in jobs:
        save(
            tmp_path / job["id"] / "manifest.json",
            {
                job["role"]: [
                    {
                        "path": f"{job['role']}/12.00/batch_{i:04d}.pkl",
                        "snr": 12,
                        "sha256": str(i) * 64,
                    }
                    for i in range(2)
                ]
            },
        )
        save(tmp_path / job["id"] / "lineage.json", {"config": job["config"]})
    assemble(tmp_path, jobs, ["B", "C"])
    for target in ("B", "C"):
        m = json.loads((tmp_path / f"A2{target}.json").read_text())
        assert set(m) == {"source_train", "online_train", "validation", "evaluation"}
        assert all(e["path"].startswith("A_source_train/") for e in m["source_train"])
        assert all(
            e["path"].startswith(target + "_")
            for role in m
            if role != "source_train"
            for e in m[role]
        )


def test_plan_only_does_not_create_data_or_import_tensorflow(tmp_path, capsys):
    import sys

    assert "tensorflow" not in sys.modules
    out = tmp_path / "not_created"
    run(
        SimpleNamespace(
            config_dir=ROOT / "configs/data",
            targets=["B"],
            scale="smoke",
            plan_only=True,
            output=out,
            resume=False,
            gpu="0",
        )
    )
    assert json.loads(capsys.readouterr().out)["total_batches"] == 8
    assert not out.exists()
    assert "tensorflow" not in sys.modules
