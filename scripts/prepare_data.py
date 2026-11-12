#!/usr/bin/env python3
"""Prepare one declared dataset; reuse only complete, configuration-matched splits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2) + "\n")


def identity(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def tdl_jobs(case, scale):
    jobs = []
    for role, seed, n, snrs in [
        ("source_train", 42, 200, range(4, 19)),
        ("online_train", 42, 10, range(4, 21)),
        ("validation", 56, 100, [12]),
        ("evaluation", 57, 100, [12]),
    ]:
        settings = case["source"] if role == "source_train" else case["target"]
        if scale == "smoke":
            n = 2
            snrs = [12]
        c = dict(
            settings, seed=seed, role=role, batches_per_snr=n, snrs_db=list(snrs), batch_size=32
        )
        jobs.append(dict(id=identity(c), config=c))
    if case["online_batches_per_snr"] > 10 and scale == "paper":
        extension = dict(jobs[1]["config"], batches_per_snr=90, advance_batches_per_snr=10)
        jobs.append(dict(id=identity(extension), config=extension))
    return jobs


def rt_manifest(case, root, scale):
    # Copy compact propagation coefficients once, not generated frames per job.
    destination = root / "rt_assets"
    if not destination.exists():
        shutil.copytree(ROOT / "artifacts/rt", destination)
    original = read(destination / "channels.json")
    mapping = {}
    for group, entries in original.items():
        mapping[group] = {
            k: {**v, "path": str(Path("rt_assets/channels") / Path(v["path"]).name)}
            for k, v in entries.items()
        }
    path = root / "rt_channels.json"
    save(path, mapping)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    source_group = "source/" + case["source_city"]
    target_group = "target/" + case["target_case"]
    pos = next(iter(mapping[target_group]))
    source_positions = list(mapping[source_group])

    def entry(group, role, snr, samples):
        return dict(
            kind="rt",
            path="rt_channels.json",
            sha256=h,
            channel_group=group,
            role=role,
            snr=snr,
            samples=samples,
            sir_db=case["sir_db"] if role != "source" else None,
        )

    source = []
    for snr in [12] if scale == "smoke" else range(4, 19):
        samples = [(p, f) for p in source_positions for f in range(50)]
        for i in range(0, 64 if scale == "smoke" else len(samples), 32):
            source.append(entry(source_group, "source", snr, samples[i : i + 32]))
    online = [
        entry(target_group, "adapt", 4 + i % 15, [(pos, i * 32 + j) for j in range(32)])
        for i in range(2 if scale == "smoke" else 50)
    ]
    validation = [
        entry(target_group, "validation", 12, [(pos, i + j) for j in range(32)])
        for i in range(0, 32 if scale == "smoke" else 256, 32)
    ]
    evaluation = [
        entry(target_group, "test", snr, [(pos, i + j) for j in range(32)])
        for snr in ([12] if scale == "smoke" else range(4, 19))
        for i in range(0, 32 if scale == "smoke" else 3200, 32)
    ]
    return dict(
        source_train=source, online_train=online, validation=validation, evaluation=evaluation
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--scale", choices=["smoke", "paper"], default="paper")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--independent", action="store_true", help="Use the final independent-test seed at 12 dB"
    )
    a = p.parse_args()
    case = read(ROOT / "configs/datasets.json")[a.case]
    root = a.output.resolve()
    jobs = tdl_jobs(case, a.scale) if case["kind"] == "tdl" else []
    if a.independent:
        if case["kind"] != "tdl":
            raise ValueError("RT already has independent test streams")
        c = next(j["config"] for j in jobs if j["config"]["role"] == "evaluation")
        c.update(seed=20261110, snrs_db=[12])
        next(j for j in jobs if j["config"] is c)["id"] = identity(c)
    if a.plan_only:
        print(
            json.dumps(
                dict(
                    case=a.case,
                    definition=case,
                    scale=a.scale,
                    jobs=jobs,
                    estimated_raw_GB=sum(
                        len(j["config"]["snrs_db"]) * j["config"]["batches_per_snr"] for j in jobs
                    )
                    * 0.007396182,
                ),
                indent=2,
            )
        )
        return
    root.mkdir(parents=True, exist_ok=True)
    name = a.case + ("_independent" if a.independent else "") + "_" + a.scale
    manifest_path = root / (name + ".json")
    if manifest_path.exists():
        raise ValueError(f"Manifest already exists: {manifest_path}; use it directly")
    if case["kind"] == "rt":
        manifest = rt_manifest(case, root, a.scale)
        save(root / (name + "_virtual.json"), manifest)
        # Materialize adaptation and evaluation batches before timing. Source
        # pretraining streams the same deterministic frames to save disk space.
        config = root / (name + "_materialize.json")
        save(config, dict(seed=42))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fastset",
                "materialize-rt",
                "--config",
                str(config),
                "--manifest",
                str(root / (name + "_virtual.json")),
                "--data-root",
                str(root),
                "--gpu",
                a.gpu,
                "--output",
                str(root / (name + "_frames")),
            ],
            check=True,
        )
        generated = read(root / (name + "_frames") / "manifest.json")
        for role in ("online_train", "validation", "evaluation"):
            manifest.setdefault(role, [])
            manifest[role] += [
                {**e, "path": str(Path(name + "_frames") / e["path"])} for e in generated[role]
            ]
    else:
        manifest = {}
        for job in jobs:
            cfg = job["config"]
            folder = root / "splits" / job["id"]
            path = root / "configs" / (job["id"] + ".json")
            save(path, cfg)
            if (folder / "result.json").exists():
                if (
                    read(folder / "lineage.json")["config"] != cfg
                    or read(folder / "result.json")["status"] != "complete"
                ):
                    raise ValueError("Dataset cache mismatch")
                print("REUSE", job["id"], flush=True)
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fastset",
                        "generate",
                        "--config",
                        str(path),
                        "--output",
                        str(folder),
                        "--gpu",
                        a.gpu,
                    ],
                    check=True,
                )
            role = cfg["role"]
            manifest.setdefault(role, [])
            manifest[role] += [
                {**e, "path": str(Path("splits") / job["id"] / e["path"])}
                for e in read(folder / "manifest.json")[role]
            ]
    if case["kind"] == "tdl" and a.scale == "paper":
        entries = manifest["online_train"]
        n = case["online_batches_per_snr"]
        manifest["online_train"] = [
            e for snr in range(4, 21) for e in [x for x in entries if x["snr"] == snr][:n]
        ]
    save(manifest_path, manifest)
    print("MANIFEST", manifest_path, flush=True)


if __name__ == "__main__":
    main()
