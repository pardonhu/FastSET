"""Generate TDL splits and assemble portable experiment manifests."""

import json
from pathlib import Path
import subprocess
import sys
from .common import read, save


def plan(config_dir, targets, scale):
    if scale not in {"smoke", "paper"} or not targets or not set(targets) <= {"B", "C"}:
        raise ValueError("Choose smoke/paper and target B and/or C")
    roles = [("A", "source_train")]
    roles += [
        (domain, role)
        for domain in dict.fromkeys(targets)
        for role in ("online_train", "validation", "evaluation")
    ]
    jobs = []
    for domain, role in roles:
        config = read(Path(config_dir) / f"{domain}_{role}.json")
        if config["tdl_model"] != domain or config["role"] != role:
            raise ValueError("Generation config does not match its declared role/domain")
        if scale == "smoke":
            config.update(snrs_db=[12], batches_per_snr=2)
        jobs.append(
            {
                "id": f"{domain}_{role}",
                "domain": domain,
                "role": role,
                "config": config,
                "batches": len(config["snrs_db"]) * config["batches_per_snr"],
            }
        )
    return jobs


def assemble(root, jobs, targets):
    manifests = {}
    for target in dict.fromkeys(targets):
        combined = {}
        for job in jobs:
            if job["domain"] not in {"A", target}:
                continue
            child = read(root / job["id"] / "manifest.json")
            entries = child[job["role"]]
            if len(entries) != job["batches"]:
                raise ValueError("Incomplete generated split: " + job["id"])
            if read(root / job["id"] / "lineage.json")["config"] != job["config"]:
                raise ValueError("Generated split configuration changed")
            combined[job["role"]] = []
            for entry in entries:
                path = Path(entry["path"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Generated manifest escapes the dataset root")
                combined[job["role"]].append({**entry, "path": (Path(job["id"]) / path).as_posix()})
        save(root / f"A2{target}.json", combined)
        manifests[f"A2{target}"] = f"A2{target}.json"
    return manifests


def run(args):
    jobs = plan(args.config_dir, args.targets, args.scale)
    summary = {
        "scale": args.scale,
        "targets": args.targets,
        "total_batches": sum(j["batches"] for j in jobs),
        "estimated_uncompressed_gb": sum(j["batches"] for j in jobs) * 7396182 / 1e9,
        "bit_identical_to_paper_data": False,
        "jobs": jobs,
    }
    if args.plan_only:
        print(json.dumps(summary, indent=2))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and (root / "generation_plan.json").exists():
        if read(root / "generation_plan.json") != summary:
            raise ValueError("Cannot resume with a changed generation plan")
    save(root / "generation_plan.json", summary)
    for job in jobs:
        child = root / job["id"]
        config_path = root / "configs" / f"{job['id']}.json"
        if args.resume and (child / "result.json").exists():
            if (
                read(child / "result.json")["status"] != "complete"
                or read(child / "lineage.json")["config"] != job["config"]
            ):
                raise ValueError("Completed split does not match requested config")
            print("REUSE_COMPLETE_SPLIT", job["id"], flush=True)
            continue
        if child.exists():
            raise ValueError(f"Incomplete split retained at {child}; use a new dataset directory")
        save(config_path, job["config"])
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "fastset",
                "generate",
                "--config",
                str(config_path),
                "--output",
                str(child),
                "--gpu",
                args.gpu,
            ],
            check=True,
        )
    manifests = assemble(root, jobs, args.targets)
    save(
        root / "result.json",
        {
            "status": "complete",
            "manifests": manifests,
            "scale": args.scale,
            "bit_identical_to_paper_data": False,
        },
    )
    print("DATASET_COMPLETE", json.dumps(manifests), flush=True)
