"""Portable orchestration around the measured model and update functions.

The original campaign runners remain untouched. This adapter is a release
candidate: GPU numerical acceptance is tracked separately from CPU tests.
"""

import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .common import read, save, order
from .data import sha
from .protocol import validate, candidate, schedule, batch_indices, load_manifest, first_hit


def csvsave(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def digest(model):
    h = hashlib.sha256()
    for value in model.get_weights():
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def emit(event, **values):
    print(event, json.dumps(values, allow_nan=False), flush=True)


def checkpoint_files(prefix):
    prefix = Path(prefix)
    files = sorted(prefix.parent.glob(prefix.name + ".*"))
    if not files or not Path(str(prefix) + ".index").is_file():
        raise ValueError("Expected a TensorFlow checkpoint prefix with index and data files")
    return {p.name: sha(p) for p in files}


def setup(args):
    config = validate(read(args.config))
    manifest = load_manifest(read(args.manifest), args.data_root, args.verify_hashes)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise ValueError("Output already contains a run; use a new run directory")
    save(out / "config.json", config)
    save(out / "data/manifest.json", manifest)
    save(
        out / "lineage.json",
        {
            "phase": args.command,
            "config_sha256": sha(args.config),
            "manifest_sha256": sha(args.manifest),
            "full_data_hash_check": args.verify_hashes,
            "model": candidate(config),
            "seed": config["seed"],
            "code_hashes": {
                str(p.relative_to(Path(__file__).parent)): sha(p)
                for p in Path(__file__).parent.rglob("*.py")
            },
            "source_checkpoint_hashes": checkpoint_files(args.source) if args.source else None,
            "timing_scope": "reported separately; evaluation wall time is never adaptation time",
        },
    )
    from .model_runtime import runtime

    tf, model, parts, opts, ck, manifest, _, update, evaluate = runtime(
        out, config, candidate(config), out
    )
    if args.source:
        ck.read(str(Path(args.source).resolve())).assert_consumed()
    if (
        args.command != "evaluate"
        and args.source
        and any(int(o.iterations) != 0 for o in opts.values())
    ):
        raise ValueError(
            "Online/timing source must be a pretrained online_initial with zero Adam iterations"
        )
    # Learning rate is checkpointed by Keras. The declared run config must not
    # be silently overridden by the source checkpoint's optimizer scalar.
    for optimizer in opts.values():
        optimizer.learning_rate.assign(config["learning_rate"])
    save(
        out / "weight_schema.json",
        [
            {"order": i, "name": w.name, "shape": w.shape.as_list(), "dtype": w.dtype.name}
            for i, w in enumerate(model.weights)
        ],
    )
    if "est" in parts:
        from .joint import make_joint

        joint = make_joint(tf, parts, opts)
    else:
        joint = None

    def step(stage, batch):
        if stage != "joint":
            return {"loss": update(stage, batch)}
        values = joint(
            *[
                tf.convert_to_tensor(batch[k])
                for k in (
                    "batch_pilots_rg",
                    "batch_y",
                    "batch_N0",
                    "batch_h_freq",
                    "batch_tx_codeword_bits",
                )
            ]
        )
        result = dict(zip(("loss", "mse", "bce"), (float(v.numpy()) for v in values)))
        tf.test.experimental.sync_devices()
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError("Nonfinite Joint loss")
        return result

    return config, out, tf, model, parts, opts, ck, manifest, step, evaluate


def pretrain(args):
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    from .model_runtime import checked

    pre = config["pretrain"]
    stages = ["est", "dem"] if pre["mode"] == "sequential" else [pre["mode"]]
    if not manifest["source_train"]:
        raise ValueError("source_train is required for pretraining")
    n = len(manifest["source_train"])
    max_epochs = max(pre[s + "_epochs"] for s in stages)
    if max_epochs == 0:
        raise ValueError("Pretraining budget is empty")
    orders = order(n, max_epochs, config["seed"])
    blank = ck.write(str(out / "random_initial"))
    rows = []
    for stage in stages:
        for epoch in range(pre[stage + "_epochs"]):
            begin, losses = time.perf_counter(), []
            for iteration, index in enumerate(orders[epoch], 1):
                metrics = step(stage, checked(manifest["source_train"][index]))
                losses.append(metrics["loss"])
                if iteration % 100 == 0 or iteration == n:
                    progress = dict(
                        stage=stage,
                        epoch=epoch + 1,
                        batch_in_epoch=iteration,
                        batches_per_epoch=n,
                        **metrics,
                    )
                    save(out / "progress.json", progress)
                    emit("PRETRAIN", **progress)
            folder = out / f"{stage}_epoch_{epoch + 1:03d}"
            folder.mkdir()
            prefix = ck.write(str(folder / "checkpoint"))
            rows.append(
                dict(
                    stage=stage,
                    epoch=epoch + 1,
                    iterations=n,
                    mean_loss=float(np.mean(losses)),
                    wall_seconds=time.perf_counter() - begin,
                    checkpoint=str(Path(prefix).relative_to(out)),
                )
            )
            csvsave(out / "epochs.csv", rows)
    weights = model.get_weights()
    ck.read(blank).assert_consumed()
    model.set_weights(weights)
    if any(int(o.iterations) != 0 for o in opts.values()):
        raise ValueError("Online Adam reset failed")
    initial = ck.write(str(out / "online_initial"))
    save(
        out / "result.json",
        dict(
            status="complete",
            phase="pretrain",
            checkpoint="online_initial",
            checkpoint_hashes=checkpoint_files(initial),
            selection="fixed final source epoch; no target-domain selection",
            optimizer_iterations={k: int(o.iterations) for k, o in opts.items()},
        ),
    )


def online(args):
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    from .model_runtime import checked

    seq = schedule(config)
    ids = batch_indices(config, len(manifest["online_train"]))
    save(out / "batch_sequence.json", [manifest["online_train"][i] for i in ids])
    rows = [
        dict(total_iteration=0, stage="initial", stage_iteration=0, **evaluate(manifest["test56"]))
    ]
    initial = ck.write(str(out / "initial"))
    selected = initial
    emit("INITIAL", **rows[0])
    csvsave(out / "iterations.csv", rows)
    for total, (stage, index) in enumerate(zip(seq, ids), 1):
        if config["online"]["early_stop"] and first_hit(rows, config["target_ber"]) is not None:
            break
        inactive = "dem" if stage == "est" else "est" if stage == "dem" else None
        frozen = digest(parts[inactive]) if inactive else None
        metrics = step(stage, checked(manifest["online_train"][index]))
        if inactive and digest(parts[inactive]) != frozen:
            raise ValueError("Frozen module changed")
        row = dict(
            total_iteration=total,
            stage=stage,
            stage_iteration=seq[:total].count(stage),
            **metrics,
            **evaluate(manifest["test56"]),
            **{k + "_optimizer_iterations": int(o.iterations) for k, o in opts.items()},
        )
        folder = out / f"iteration_{total:03d}"
        folder.mkdir()
        selected = ck.write(str(folder / "checkpoint"))
        rows.append(row)
        csvsave(out / "iterations.csv", rows)
        save(out / "progress.json", row)
        emit("ITERATION", **row)
    hit = first_hit(rows, config["target_ber"])
    save(
        out / "result.json",
        dict(
            status="complete",
            phase="online",
            first_hit=hit,
            target_reached=hit is not None,
            final=rows[-1],
            early_stop=config["online"]["early_stop"],
            checkpoint=str(Path(selected).relative_to(out)),
            checkpoint_hashes=checkpoint_files(selected),
        ),
    )


def evaluate_checkpoint(args):
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    entries = manifest.get("evaluation")
    if not entries:
        raise ValueError("evaluation split is required; validation is not silently substituted")
    rows = []
    for snr in sorted({e["snr"] for e in entries}):
        rows.append(dict(snr_db=snr, **evaluate([e for e in entries if e["snr"] == snr])))
        csvsave(out / "ber.csv", rows)
        emit("EVALUATE", **rows[-1])
    save(out / "result.json", dict(status="complete", phase="evaluate", rows=rows))


def timing(args):
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    from .model_runtime import checked

    trajectory = Path(args.trajectory)
    with trajectory.open() as stream:
        measured = list(csv.DictReader(stream))
    seq = schedule(config)
    seq = seq[: len(measured) - 1]
    if (
        not measured
        or len(measured) - 1 > len(schedule(config))
        or [r["stage"] for r in measured] != ["initial"] + seq
    ):
        raise ValueError("Timing trajectory does not match the configured schedule")
    if [int(r["total_iteration"]) for r in measured] != list(range(len(measured))):
        raise ValueError("Non-contiguous timing trajectory")
    # A timing replay must be tied to the actual online run, not a hand-edited CSV.
    online_lineage = read(trajectory.parent / "lineage.json")
    line = read(out / "lineage.json")
    for field in ("config_sha256", "manifest_sha256", "source_checkpoint_hashes", "code_hashes"):
        if online_lineage[field] != line[field]:
            raise ValueError("Timing/online lineage mismatch: " + field)
    ids = batch_indices(config, len(manifest["online_train"]))[: len(seq)]
    input_rows, cache = [], {}
    for repeat in range(1, 4):
        elapsed = 0.0
        for total, index in enumerate(ids, 1):
            start = time.perf_counter()
            batch = checked(manifest["online_train"][index])
            dt = time.perf_counter() - start
            elapsed += dt
            cache[index] = batch
            input_rows.append(
                dict(repeat=repeat, total_iteration=total, seconds=dt, cumulative_seconds=elapsed)
            )
    for stage in dict.fromkeys(seq):
        step(stage, cache[ids[0]])
    rows, reference = [], None
    for repeat in range(1, 4):
        ck.read(str(Path(args.source).resolve())).assert_consumed()
        for optimizer in opts.values():
            optimizer.learning_rate.assign(config["learning_rate"])
        elapsed = 0.0
        for total, (stage, index) in enumerate(zip(seq, ids), 1):
            tf.test.experimental.sync_devices()
            start = time.perf_counter()
            metrics = step(stage, cache[index])
            tf.test.experimental.sync_devices()
            dt = time.perf_counter() - start
            elapsed += dt
            rows.append(
                dict(
                    repeat=repeat,
                    total_iteration=total,
                    stage=stage,
                    seconds=dt,
                    cumulative_seconds=elapsed,
                    **metrics,
                )
            )
        weights = model.get_weights()
        if reference is None:
            reference = weights
        elif not all(np.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(reference, weights)):
            raise ValueError("Timing repetitions diverged")
        emit("TIMING_REPEAT", repeat=repeat, seconds=elapsed)
    summary = [dict(total_iteration=0, training_seconds=0.0, input_seconds=0.0, total_seconds=0.0)]
    for i in range(1, len(seq) + 1):
        train = float(
            np.median([r["cumulative_seconds"] for r in rows if r["total_iteration"] == i])
        )
        data = float(
            np.median([r["cumulative_seconds"] for r in input_rows if r["total_iteration"] == i])
        )
        summary.append(
            dict(
                total_iteration=i,
                training_seconds=train,
                input_seconds=data,
                total_seconds=train + data,
            )
        )
    csvsave(out / "training_repeats.csv", rows)
    csvsave(out / "input_repeats.csv", input_rows)
    csvsave(out / "timing.csv", summary)
    final = evaluate(manifest["test56"])
    original = {k: int(measured[-1][k]) for k in ("bit_errors", "bits")}
    save(
        out / "result.json",
        dict(
            status="complete",
            phase="timing",
            repeats=3,
            local_final=final,
            online_final=original,
            integer_counts_match=all(final[k] == original[k] for k in original),
            trajectory_sha256=sha(trajectory),
            definition="sum of separately measured median cumulative input and training times; not end-to-end latency",
            includes="input: read/deserialize/validate; training: tensor conversion, transfer, update, sync",
            excludes="BER/decoder evaluation, compilation warmup, checkpoint/log writing, generation, network",
        ),
    )


def run(args):
    {"pretrain": pretrain, "online": online, "evaluate": evaluate_checkpoint, "timing": timing}[
        args.command
    ](args)
