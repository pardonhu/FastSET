"""CPU-safe CLI; neural commands run in a fresh, leased GPU subprocess."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def parser():
    p = argparse.ArgumentParser(prog="fastset")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Inspect installation and visible devices without importing TensorFlow"
    )
    suite = sub.add_parser("generate-suite", help="Generate and assemble A2B/A2C TDL datasets")
    suite.add_argument("--config-dir", type=Path, default=Path("configs/data"))
    suite.add_argument("--targets", nargs="+", choices=["B", "C"], default=["B", "C"])
    suite.add_argument("--scale", choices=["smoke", "paper"], required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--gpu", required=True)
    suite.add_argument("--plan-only", action="store_true")
    suite.add_argument("--resume", action="store_true")
    check = sub.add_parser("check", help="Validate config and optionally dataset layout/hashes")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--manifest", type=Path)
    check.add_argument("--data-root", type=Path)
    check.add_argument("--verify-hashes", action="store_true")
    plot = sub.add_parser("plot", help="Plot a numeric CSV without importing TensorFlow")
    plot.add_argument("--csv", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--x", default="total_seconds")
    plot.add_argument("--y", default="ber")
    plot.add_argument("--group", default="method")
    plot.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE")
    plot.add_argument("--xlabel", default="Adaptation time (s)")
    paper = sub.add_parser("paper-figures", help="Regenerate figures from bundled measured CSVs")
    paper.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    paper.add_argument("--output", type=Path, default=Path("results/paper_figures"))
    pebs = sub.add_parser("pebs", help="Recompute PEBS from frozen first-hit measurements")
    pebs.add_argument("--csv", type=Path, default=Path("artifacts/tables/pebs_first_hit.csv"))
    pebs.add_argument("--output", type=Path, required=True)
    pebs.add_argument("--alpha", type=float, default=0.8)
    for name in ("pretrain", "online", "evaluate", "timing"):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--data-root", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--gpu", required=True, help="Explicit device index or GPU UUID")
        item.add_argument("--source", type=Path, required=name != "pretrain")
        item.add_argument("--verify-hashes", action="store_true")
        item.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
        if name == "timing":
            item.add_argument("--trajectory", type=Path, required=True)
    gen = sub.add_parser(
        "generate", help="Generate a separately versioned TDL dataset; not paper-identical"
    )
    gen.add_argument("--config", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--gpu", required=True)
    gen.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def launch(args):
    from .gpu import lease
    from .common import read

    config = read(args.config)
    out = args.output.resolve()
    # Reserve a fresh result directory before launching; failed runs remain visible.
    out.mkdir(parents=True, exist_ok=False)
    with lease(args.gpu, config["seed"]) as env:
        with (out / "run.log").open("w", buffering=1) as log:
            child = subprocess.Popen(
                [sys.executable, "-B", "-u", "-m", "fastset", *sys.argv[1:], "--_worker"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for line in child.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
    if code:
        raise SystemExit(code)


def main():
    args = parser().parse_args()
    if args.command == "doctor":
        import importlib.metadata as md

        versions = {}
        for name in ("fastset-research", "numpy", "tensorflow", "sionna", "matplotlib"):
            try:
                versions[name] = md.version(name)
            except md.PackageNotFoundError:
                versions[name] = "not installed"
        print(json.dumps({"python": sys.version, "versions": versions}, indent=2))
        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv",
                ],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("NVIDIA device query unavailable. CPU checks and plotting remain usable.")
    elif args.command == "check":
        from .common import read
        from .protocol import validate, candidate, load_manifest, batch_indices

        config = validate(read(args.config))
        report = {"configuration": "valid", "candidate": candidate(config)}
        if args.manifest:
            if not args.data_root:
                raise SystemExit("--manifest requires --data-root")
            manifest = load_manifest(read(args.manifest), args.data_root, args.verify_hashes)
            report["split_batches"] = {k: len(v) for k, v in manifest.items()}
            batch_indices(config, len(manifest["online_train"]))
        print(json.dumps(report, indent=2))
    elif args.command == "plot":
        from .plotting import plot_csv

        plot_csv(args)
    elif args.command == "paper-figures":
        from .plotting import paper_figures

        paper_figures(args.artifacts, args.output)
    elif args.command == "pebs":
        from .selection import summarize

        summarize(args.csv, args.output, args.alpha)
    elif args.command == "generate-suite":
        from .dataset_suite import run

        run(args)
    elif not args._worker:
        launch(args)
    else:
        try:
            if args.command == "generate":
                from .generation import generate

                generate(args)
            else:
                from .experiment import run

                run(args)
        except BaseException:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "failure.txt").write_text(traceback.format_exc())
            raise
