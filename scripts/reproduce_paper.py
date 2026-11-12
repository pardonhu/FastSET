#!/usr/bin/env python3
"""Rebuild submission figures from archived measurements, without TensorFlow.

Every run gets a new directory containing the exact plotting sources, logs,
figure outputs, and input hashes. Numerical data are embedded in the scripts and
also exported as CSV under paper/tables for independent analysis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "main": [
        "plot_width_depth_combined",
        "plot_pebs",
        "plot_hyperparameters",
        "plot_main",
        "plot_ablation_a2b_log10",
        "plot_ablation_a2c_log10",
        "plot_speed",
        "plot_rt_city_geometry",
        "plot_rt",
        "plot_timing",
    ],
    "supplement": [
        "plot_supplement",
        "plot_joint_restored",
        "plot_check_frequency",
        "plot_fixed_time",
        "plot_data_volume",
    ],
    "differences": ["plot_summary_differences"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", *GROUPS], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "paper/plotting", out / "plotting")
    (out / "logs").mkdir()
    for folder in ("current", "supplement_review", "data_volume", "differences"):
        (out / "figs" / folder).mkdir(parents=True)
    names = [n for k, values in GROUPS.items() if args.group in ("all", k) for n in values]
    report = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        status="running",
        group=args.group,
        python=sys.version,
        scripts=[],
        source_hashes={},
    )
    for p in sorted((out / "plotting").glob("*.py")):
        report["source_hashes"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(out / "matplotlib-cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        for i, name in enumerate(names, 1):
            print(f"[{i}/{len(names)}] {name}", flush=True)
            with (out / "logs" / (name + ".log")).open("w") as log:
                result = subprocess.run(
                    [sys.executable, str(out / "plotting" / (name + ".py"))],
                    cwd=out,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            report["scripts"].append(dict(name=name, exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {out / "logs" / (name + ".log")}')
        report["status"] = "complete"
        report["pdf_outputs"] = [str(p.relative_to(out)) for p in sorted(out.rglob("*.pdf"))]
        print(f'Complete: {len(report["pdf_outputs"])} PDF outputs in {out}', flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (out / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
