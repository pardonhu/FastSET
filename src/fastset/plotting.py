"""Regenerate figures from measured CSVs; never synthesize BER samples."""

import csv
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np

LABELS = {
    "fastset": "FastSET",
    "est_only": "est. only",
    "dem_only": "dem. only",
    "joint": "Joint (staged pretraining)",
    "cnn": "CNN",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    return plt


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=240, bbox_inches="tight")


def plot_csv(args):
    rows = read_csv(args.csv)
    for condition in args.filter:
        field, value = condition.split("=", 1)
        rows = [r for r in rows if r[field] == value]
    if not rows:
        raise ValueError("No measured rows match the requested selection")
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    groups = list(dict.fromkeys(r[args.group] for r in rows))
    zero_present = False
    for index, group in enumerate(groups):
        selected = sorted(
            [r for r in rows if r[args.group] == group], key=lambda r: float(r[args.x])
        )
        x = [float(r[args.x]) for r in selected]
        y = [float(r[args.y]) for r in selected]
        if any(v < 0 for v in y) or not np.isfinite(x + y).all():
            raise ValueError("Plot values must be finite and BER nonnegative")
        if len(set(x)) != len(x):
            raise ValueError("Repeated x in a series: select a single scenario/panel first")
        # Zero errors do not imply BER=0. Display the empirical one-error scale
        # as a downward marker, not as an invented measured positive BER.
        positive = [v if v > 0 else np.nan for v in y]
        (line,) = ax.semilogy(
            x,
            positive,
            label=LABELS.get(group, group),
            marker=MARKERS[index % len(MARKERS)],
            ms=4,
            markevery=max(1, len(selected) // 12),
            lw=1.5,
        )
        for xx, yy, row in zip(x, y, selected):
            if yy == 0:
                bits = int(row["bits"])
                ax.scatter([xx], [1 / bits], marker="v", color=line.get_color(), s=24)
                zero_present = True
    if args.y == "ber":
        for target in (2e-4, 2e-5):
            ax.axhline(target, color="0.5", ls="--" if target == 2e-4 else ":", lw=0.8)
    ax.set(xlabel=args.xlabel, ylabel=args.y.upper())
    ax.grid(which="both", alpha=0.18)
    ax.legend(
        ncol=min(3, len(groups)), loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False
    )
    if zero_present:
        fig.text(
            0.5,
            -0.03,
            "Downward markers: zero observed errors, shown at 1 / tested bits.",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)


def paper_figures(artifacts, output):
    artifacts, output = Path(artifacts), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for case in ("A2B", "A2C"):
        plot_csv(
            SimpleNamespace(
                csv=artifacts / "tables/ablation_iterations.csv",
                output=output / f"ablation_{case}",
                x="total_seconds",
                y="ber",
                group="method",
                filter=[f"scenario={case}"],
                xlabel="Input + training time (s)",
            )
        )
    for collection in ("main", "misc"):
        path = artifacts / f"tables/{collection}_plotted_points.csv"
        rows = read_csv(path)
        for asset, panel in dict.fromkeys((r["asset"], r["panel"]) for r in rows):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{collection}_{asset}_{panel or 'all'}")
            xlabel = (
                "Input + training time (s)"
                if asset in {"adaptation_time_comparison", "backbone_time"}
                else (
                    "Iteration"
                    if asset in {"learning_rate_iterations", "batch32_vs64"}
                    else "SNR (dB)"
                )
            )
            plot_csv(
                SimpleNamespace(
                    csv=path,
                    output=output / stem,
                    x="x",
                    y="ber",
                    group="series",
                    filter=[f"asset={asset}", f"panel={panel}"],
                    xlabel=xlabel,
                )
            )
    # RT needs both method and adaptation phase in the series key.
    for table, casekey in (("rt_ber_curves", "case"), ("rt_delay_ber", "delay_ns")):
        rows = read_csv(artifacts / f"tables/{table}.csv")
        for row in rows:
            row["series"] = row["method"] + " / " + row["phase"]
        from .experiment import csvsave

        plotted = output / f"{table}_plotted.csv"
        csvsave(plotted, rows)
        for case in dict.fromkeys(r[casekey] for r in rows):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{table}_{case}")
            plot_csv(
                SimpleNamespace(
                    csv=plotted,
                    output=output / stem,
                    x="snr_db",
                    y="ber",
                    group="series",
                    filter=[f"{casekey}={case}"],
                    xlabel="SNR (dB)",
                )
            )
    plot_pebs(artifacts / "tables/pebs_first_hit.csv", output)
    from .common import save

    save(
        output / "PROTOCOL.json",
        {
            "source": "bundled measured CSV snapshot",
            "smoothing": False,
            "ablation_main": "A2C",
            "ablation_A2B": "supplemental",
            "joint_pretraining": "staged, not new Joint10 control",
            "timing": "use each CSV's frozen measurement scope; see docs/TIMING.md",
        },
    )


def plot_pebs(path, output):
    rows = read_csv(path)
    candidates = [r for r in rows if r["candidate"].startswith("d")]
    if len(candidates) != 40:
        raise ValueError("PEBS requires all 40 measured candidates")
    matrix = np.full((10, 4), np.nan)
    for r in candidates:
        depth, width = (int(v) for v in r["candidate"][1:].split("_w"))
        matrix[depth - 1, (8, 16, 32, 64).index(width)] = float(r["pebs_alpha_0_8"])
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(5.3, 6.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    for i in range(10):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] > np.nanmax(matrix) * 0.65 else "white",
                fontsize=9,
            )
    ax.set(
        xticks=range(4),
        xticklabels=(8, 16, 32, 64),
        yticks=range(10),
        yticklabels=range(1, 11),
        xlabel="Estimator initial width",
        ylabel="Estimator depth",
    )
    fig.colorbar(im, ax=ax, label="PEBS (alpha = 0.8)")
    fig.tight_layout()
    save_figure(fig, output / "pebs_heatmap")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    x = np.array([float(r["total_seconds"]) for r in candidates])
    y = np.array([float(r["test57_ber"]) for r in candidates])
    nondominated = np.array(
        [not np.any((x <= xx) & (y <= yy) & ((x < xx) | (y < yy))) for xx, yy in zip(x, y)]
    )
    ax.scatter(x, y, color="0.6", s=25, label="Measured splits")
    ids = np.flatnonzero(nondominated)
    ids = ids[np.argsort(x[ids])]
    ax.plot(x[ids], y[ids], "o-", color="#0072b2", label="Finite-candidate frontier")
    for i in ids:
        ax.annotate(
            candidates[i]["candidate"].replace("d", "").replace("_w", "–"),
            (x[i], y[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set(xlabel="Input + training time (s)", ylabel="Evaluation BER (seed 57)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "pebs_pareto")
    plt.close(fig)
