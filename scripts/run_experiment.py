#!/usr/bin/env python3
"""Run a catalog experiment with isolated outputs and explicit data/model inputs.

The phase logs are streamed to the terminal and retained in each phase folder.
Completed phases can be resumed only with identical configuration and manifest.
Pretraining always uses seed42, batch32 and lr1e-3; an online ablation never
silently changes the source pretraining treatment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from fastset.common import read, save
from fastset.data import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment")
    p.add_argument("--list", action="store_true")
    p.add_argument("--group")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--gpu")
    p.add_argument("--output", type=Path)
    p.add_argument("--weights", type=Path, default=ROOT / "data/weights")
    p.add_argument("--source", type=Path)
    p.add_argument("--pretrain", action="store_true")
    p.add_argument(
        "--phases",
        nargs="+",
        choices=["pretrain", "online", "evaluate", "timing", "pebs"],
        default=["online", "evaluate", "timing"],
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One iteration per active stage and one epoch per active source stage",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    catalog = read(ROOT / "configs/experiments.json")
    if a.list:
        for name, item in catalog.items():
            if a.group is None or item["group"] == a.group:
                print(name, item["case"], item["group"])
        return
    if not a.experiment or not all((a.manifest, a.data_root, a.gpu, a.output)):
        p.error("experiment, manifest, data-root, gpu and output are required")
    item = catalog[a.experiment]
    config = read(ROOT / item["config"])
    if a.smoke:
        for key in config["online"]:
            if key.endswith("_iterations"):
                config["online"][key] = int(config["online"][key] > 0)
        config["online"]["early_stop"] = False
        for key in config["pretrain"]:
            if key.endswith("_epochs"):
                config["pretrain"][key] = int(config["pretrain"][key] > 0)

    out = a.output.resolve()
    source = a.source or a.weights / item["initial"] / "online_initial"
    if (
        a.source is None
        and not source.with_suffix(".index").exists()
        and item["initial"] in ("A/d8_w64", "A/CNN")
    ):
        source = ROOT / "artifacts/checkpoints" / item["initial"].split("/")[-1] / "online_initial"
    phases = list(a.phases)
    if a.pretrain and "pretrain" not in phases:
        phases.insert(0, "pretrain")
    if "pretrain" in phases:
        source = out / "pretrain/online_initial"
    plan = dict(
        experiment=a.experiment,
        config=config,
        manifest_sha256=sha(a.manifest),
        data_root=str(a.data_root.resolve()),
        source=str(source.resolve()),
        phases=phases,
    )
    if a.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if out.exists():
        if not a.resume or read(out / "plan.json") != plan:
            raise ValueError("Output exists or resume inputs changed")
    else:
        out.mkdir(parents=True)
        save(out / "plan.json", plan)
    cfgpath = out / "config.json"
    save(cfgpath, config)
    for phase in phases:
        dest = out / phase
        if (
            a.resume
            and (dest / "result.json").exists()
            and read(dest / "result.json")["status"] == "complete"
        ):
            print("REUSE_COMPLETE_PHASE", phase, flush=True)
            continue
        command = "evaluate-trajectory" if phase == "pebs" else phase
        used_cfg = cfgpath
        if phase == "pretrain":
            pre = dict(config, seed=42, batch_size=32, learning_rate=0.001)
            used_cfg = out / "pretrain_config.json"
            save(used_cfg, pre)
        cmd = [
            sys.executable,
            "-m",
            "fastset",
            command,
            "--config",
            str(used_cfg),
            "--manifest",
            str(a.manifest.resolve()),
            "--data-root",
            str(a.data_root.resolve()),
            "--gpu",
            a.gpu,
            "--output",
            str(dest),
        ]
        if phase != "pretrain":
            selected = source
            if phase == "evaluate":
                report = read(out / "online/result.json")
                selected = out / "online" / report["checkpoint"]
                if item.get("evaluation_model") == "first_hit" and report.get("first_hit"):
                    it = report["first_hit"]["total_iteration"]
                    selected = (
                        out
                        / "online"
                        / ("initial" if it == 0 else f"iteration_{it:03d}/checkpoint")
                    )

                cmd += ["--split", item["evaluation_role"]]
            cmd += ["--source", str(selected)]
        if phase in ("timing", "pebs"):
            cmd += ["--trajectory", str(out / "online/iterations.csv")]
        print("PHASE", phase, flush=True)
        subprocess.run(cmd, check=True)
    save(out / "result.json", dict(status="complete", experiment=a.experiment, phases=phases))


if __name__ == "__main__":
    main()
