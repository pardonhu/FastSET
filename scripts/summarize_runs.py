#!/usr/bin/env python3
"""Join measured online, evaluation and local timing outputs as reusable CSVs."""
import argparse
import csv
from pathlib import Path
from fastset.common import read, save
from fastset.experiment import csvsave
from fastset.protocol import candidate
from fastset.selection import summarize_trajectories


def rows(p):
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    curves = []
    summary = []
    ber = []
    pebs = []
    cases = set()
    for root in a.runs:
        plan = read(root / "plan.json")
        cfg = read(root / "config.json")
        c = candidate(cfg)["id"]
        online = read(root / "online/result.json")
        timing = {int(r["total_iteration"]): r for r in rows(root / "timing/timing.csv")}
        trajectory = rows(root / "online/iterations.csv")
        for row in trajectory:
            it = int(row["total_iteration"])
            curves.append(
                dict(
                    experiment=plan["experiment"],
                    candidate=c,
                    **row,
                    **{k: v for k, v in timing.get(it, {}).items() if k != "total_iteration"},
                )
            )
        hit = online["first_hit"]
        it = hit["total_iteration"] if hit else None
        summary.append(
            dict(
                experiment=plan["experiment"],
                candidate=c,
                target_reached=online["target_reached"],
                first_hit_iteration=it if hit else "",
                first_hit_stage=hit["stage"] if hit else "",
                first_hit_ber=hit["ber"] if hit else "",
                first_hit_seconds=timing.get(it, {}).get("total_seconds", ""),
                final_iteration=online["final"]["total_iteration"],
                final_ber=online["final"]["ber"],
            )
        )
        for row in rows(root / "evaluate/ber.csv"):
            ber.append(dict(experiment=plan["experiment"], candidate=c, **row))
        for row in rows(root / "pebs/qualifying_models.csv"):
            it = int(row["total_iteration"])
            if c == "CNN" and (not hit or it != hit["total_iteration"]):
                continue
            if it not in timing:
                raise ValueError("PEBS requires local timing for each evaluated model")
            pebs.append(dict(**row, total_seconds=float(timing[it]["total_seconds"])))
            cases.add(plan["manifest_sha256"])
    csvsave(a.output / "trajectories.csv", curves)
    csvsave(a.output / "first_hits.csv", summary)
    csvsave(a.output / "ber.csv", ber)
    if pebs:
        if len(cases) != 1:
            raise ValueError("PEBS inputs must use one identical dataset manifest")
        csvsave(a.output / "pebs_models.csv", pebs)
        summarize_trajectories(pebs, a.output / "pebs_ranking.csv", a.alpha)
    save(
        a.output / "result.json",
        dict(status="complete", runs=len(a.runs), curves=len(curves), alpha=a.alpha),
    )


if __name__ == "__main__":
    main()
