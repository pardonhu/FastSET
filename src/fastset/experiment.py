"""Portable orchestration around the measured model and update functions.

All experiment families share these phase functions. CPU regressions and GPU
execution checks are recorded separately in the release validation record.
"""

import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .common import read, save, order
from .data import sha
from .protocol import (
    validate,
    candidate,
    schedule,
    batch_indices,
    load_manifest,
    first_hit,
    batch_plan,
)


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
    if isinstance(config["architecture"], dict) and "family" in config["architecture"]:
        from .backbones.runtime import runtime
    else:
        from .model_runtime import runtime

    tf, model, parts, opts, ck, manifest, _, update, evaluate = runtime(
        out, config, candidate(config), out
    )
    if args.source:
        ck.read(str(Path(args.source).resolve())).assert_consumed()
    if (
        args.command not in ("evaluate", "evaluate-trajectory", "baselines")
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
    if "est" in parts and not (
        isinstance(config["architecture"], dict) and "family" in config["architecture"]
    ):
        from .joint import make_joint

        joint = make_joint(tf, parts, opts)
    else:
        joint = None

    def step(stage, batch):
        if stage != "joint":
            if args.command == "timing-breakdown":
                if not hasattr(update, "prepare"):
                    raise ValueError("Component timing supports the original CNN/split runtime")
                begin = time.perf_counter()
                prepared = update.prepare(batch)
                tf.test.experimental.sync_devices()
                prepared_seconds = time.perf_counter() - begin
                return dict(loss=update.prepared(stage, prepared), tensor_seconds=prepared_seconds)
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


def online_batch(entries, plan, timings=None):
    """Read the exact files/slices named by an online batch plan."""
    from .model_runtime import checked

    pieces = [
        checked(entries[i], timings=timings) if timings is not None else checked(entries[i])
        for i in plan["indices"]
    ]
    batch = (
        pieces[0]
        if len(pieces) == 1
        else {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in pieces[0]}
    )
    return {key: value[plan["start"] : plan["stop"]] for key, value in batch.items()}


def online(args):
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    from .model_runtime import checked

    seq = schedule(config)
    plans = batch_plan(config, len(manifest["online_train"]))
    save(out / "batch_sequence.json", plans)
    rows = [
        dict(
            total_iteration=0,
            stage="initial",
            stage_iteration=0,
            **evaluate([e for e in manifest["test56"] if e["snr"] == 12]),
        )
    ]
    initial = ck.write(str(out / "initial"))
    selected = initial
    emit("INITIAL", **rows[0])
    csvsave(out / "iterations.csv", rows)
    for total, (stage, index) in enumerate(zip(seq, plans), 1):
        if config["online"]["early_stop"] and first_hit(rows, config["target_ber"]) is not None:
            break
        inactive = "dem" if stage == "est" else "est" if stage == "dem" else None
        frozen = digest(parts[inactive]) if inactive else None
        metrics = step(stage, online_batch(manifest["online_train"], index))
        if inactive and digest(parts[inactive]) != frozen:
            raise ValueError("Frozen module changed")
        row = dict(
            total_iteration=total,
            stage=stage,
            stage_iteration=seq[:total].count(stage),
            **metrics,
            **evaluate([e for e in manifest["test56"] if e["snr"] == 12]),
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
    role = getattr(args, "split", "evaluation")
    entries = manifest.get("test56" if role == "validation" else role)
    if not entries:
        raise ValueError("evaluation split is required; validation is not silently substituted")
    rows = []
    for snr in getattr(args, "snrs", sorted({e["snr"] for e in entries})):
        selected = [e for e in entries if e["snr"] == snr]
        if (
            not selected
            and {e["snr"] for e in entries} == {12}
            and all(e.get("kind") != "rt" for e in entries)
        ):
            selected = [dict(e, evaluation_snr=snr) for e in entries]
        if not selected:
            raise ValueError(f"No evaluation inputs at {snr} dB")
        rows.append(dict(snr_db=snr, **evaluate(selected)))
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
    plans = batch_plan(config, len(manifest["online_train"]))[: len(seq)]
    if not seq:
        csvsave(
            out / "timing.csv",
            [dict(total_iteration=0, input_seconds=0.0, training_seconds=0.0, total_seconds=0.0)],
        )
        save(out / "result.json", dict(status="complete", repeats=3, zero_update=True))
        return
    for stage in dict.fromkeys(seq):
        step(stage, online_batch(manifest["online_train"], plans[0]))
    records, reference = [], None
    for repeat in range(1, 4):
        ck.read(str(Path(args.source).resolve())).assert_consumed()
        for optimizer in opts.values():
            optimizer.learning_rate.assign(config["learning_rate"])
        elapsed = inputs = training = 0.0
        for total, (stage, plan) in enumerate(zip(seq, plans), 1):
            tf.test.experimental.sync_devices()
            begin = time.perf_counter()
            input_metrics = {}
            batch = online_batch(manifest["online_train"], plan, timings=input_metrics)
            loaded = time.perf_counter()
            metrics = step(stage, batch)
            tf.test.experimental.sync_devices()
            end = time.perf_counter()
            inputs += loaded - begin
            training += end - loaded
            elapsed += end - begin
            records.append(
                dict(
                    repeat=repeat,
                    total_iteration=total,
                    stage=stage,
                    input_seconds=loaded - begin,
                    training_seconds=end - loaded,
                    cumulative_input_seconds=inputs,
                    cumulative_training_seconds=training,
                    cumulative_seconds=elapsed,
                    read_seconds=input_metrics.get("read_seconds", 0.0),
                    deserialize_check_seconds=loaded
                    - begin
                    - input_metrics.get("read_seconds", 0.0),
                    tensor_prepare_seconds=metrics.get("tensor_seconds", 0.0),
                    compute_seconds=end - loaded - metrics.get("tensor_seconds", 0.0),
                    **metrics,
                )
            )
        weights = model.get_weights()
        if reference is None:
            reference = weights
        elif not all(np.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(reference, weights)):
            raise ValueError("Timing repetitions diverged")
        emit("TIMING_REPEAT", repeat=repeat, seconds=elapsed)
    summary = [dict(total_iteration=0, input_seconds=0.0, training_seconds=0.0, total_seconds=0.0)]
    for total in range(1, len(seq) + 1):
        trials = [r for r in records if r["total_iteration"] == total]
        # Use the components of the median-total trial so they add to the total.
        r = sorted(trials, key=lambda r: r["cumulative_seconds"])[1]
        summary.append(
            dict(
                total_iteration=total,
                input_seconds=r["cumulative_input_seconds"],
                training_seconds=r["cumulative_training_seconds"],
                total_seconds=r["cumulative_seconds"],
            )
        )
    csvsave(out / "timing_repeats.csv", records)
    csvsave(out / "timing.csv", summary)
    final = evaluate([e for e in manifest["test56"] if e["snr"] == 12])
    original = {k: int(measured[-1][k]) for k in ("bit_errors", "bits")}
    save(
        out / "result.json",
        dict(
            status="complete",
            repeats=3,
            local_final=final,
            online_final=original,
            integer_counts_match=all(final[k] == original[k] for k in original),
            trajectory_sha256=sha(trajectory),
            definition="Median of three resident replays; continuous input + training per iteration",
            includes=[
                "read",
                "deserialize",
                "input validation",
                "tensor conversion",
                "transfer",
                "update",
                "synchronization",
            ],
            excludes=["BER evaluation", "model loading", "warmup", "checkpoint writes", "logging"],
        ),
    )


def run(args):
    {
        "pretrain": pretrain,
        "online": online,
        "evaluate": evaluate_checkpoint,
        "timing": timing,
        "timing-breakdown": timing,
        "evaluate-trajectory": evaluate_trajectory,
        "reset-source": reset_source,
        "baselines": baselines,
    }[args.command](args)


def evaluate_trajectory(args):
    """Evaluate every qualifying saved model on the independent PEBS dataset."""
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    trajectory = Path(args.trajectory)
    rows = list(csv.DictReader(trajectory.open()))
    results = []
    entries = [e for e in manifest.get("evaluation", []) if e["snr"] == 12]
    if not entries:
        raise ValueError("PEBS requires independent evaluation at 12 dB")
    for row in rows:
        it = int(row["total_iteration"])
        if it == 0 or float(row["ber"]) >= config["target_ber"]:
            continue
        prefix = trajectory.parent / f"iteration_{it:03d}" / "checkpoint"
        ck.read(str(prefix)).assert_consumed()
        metric = evaluate(entries)
        results.append(
            dict(
                candidate=candidate(config)["id"],
                total_iteration=it,
                stage=row["stage"],
                val1_ber=float(row["ber"]),
                val2_ber=metric["ber"],
                **metric,
            )
        )
        csvsave(out / "qualifying_models.csv", results)
        emit("PEBS_EVALUATION", **results[-1])
    save(out / "result.json", dict(status="complete", qualifying_models=len(results)))


def reset_source(args):
    """Retain all source-trained weights while resetting Adam and the RNG.

    Used when importing an existing source checkpoint whose optimizer slots
    still contain source-training history. No optimizer step is performed.
    """
    prefix = args.source
    args.source = None
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    blank = ck.write(str(out / "random_initial"))
    ck.read(str(prefix.resolve())).assert_consumed()
    trained = digest(model)
    weights = model.get_weights()
    ck.read(blank).assert_consumed()
    model.set_weights(weights)
    if trained != digest(model) or any(int(o.iterations) != 0 for o in opts.values()):
        raise ValueError("Source reset changed weights or retained optimizer history")
    saved = ck.write(str(out / "online_initial"))
    save(
        out / "result.json",
        dict(
            status="complete",
            source_hashes=checkpoint_files(prefix),
            checkpoint_hashes=checkpoint_files(saved),
            weights_sha256=trained,
            weights_unchanged=True,
            optimizer_iterations={k: int(v.iterations) for k, v in opts.items()},
        ),
    )


def baselines(args):
    """Original LS/true-H/neural-estimate LMMSE + APP + LDPC receivers."""
    config, out, tf, model, parts, opts, ck, manifest, step, evaluate = setup(args)
    from .link import BasicConfig
    from .model_runtime import checked
    from .data import counts
    import sionna

    c = BasicConfig()
    rows = []
    role = getattr(args, "split", "validation")
    entries = manifest["test56" if role == "validation" else role]

    @tf.function
    def linear(y, n, h, var):
        # Sionna's LMMSE MatrixSquareRoot uses the original CPU workaround;
        # neural estimators and the LDPC decoder retain GPU execution.
        with tf.device("/CPU:0"):
            yy = sionna.utils.insert_dims(y, 2, axis=1)
            xx, nn = c._lmmse_equ([yy, h, var, n])
            return c._demapper([xx, nn])

    @tf.function
    def decode(z):
        with tf.device("/GPU:0"):
            return c._decoder(z)

    for snr in args.snrs:
        selected = [e for e in entries if e["snr"] == snr]
        if not selected and {e["snr"] for e in entries} == {12}:
            selected = [dict(e, evaluation_snr=snr) for e in entries]
        if not selected:
            raise ValueError("Missing baseline input SNR")
        for method in args.methods:
            total = dict(bit_errors=0, bits=0, block_errors=0, blocks=0)
            for entry in selected:
                d = checked(entry)
                p, y, n, h = [
                    tf.convert_to_tensor(d[k])
                    for k in ("batch_pilots_rg", "batch_y", "batch_N0", "batch_h_freq")
                ]
                if method == "CNNEst":
                    if "est" not in parts:
                        raise ValueError("CNNEst requires a split-model checkpoint")
                    with tf.device("/GPU:0"):
                        hh, var = parts["est"]([p, y, n, h], training=False)
                        hh = tf.complex(hh[..., 0], hh[..., 1])
                elif method == "Perfect":
                    hh = sionna.utils.insert_dims(h, 4, axis=1)
                    var = 0.0
                else:
                    with tf.device("/CPU:0"):
                        hh, var = c._ls_est([y[:, None, None], n])
                pred = decode(linear(y, n, hh, var))
                metric = counts(d["batch_b"], pred.numpy())
                for key in total:
                    total[key] += metric[key]
            row = dict(
                method=method,
                snr_db=snr,
                **total,
                ber=total["bit_errors"] / total["bits"],
                bler=total["block_errors"] / total["blocks"],
            )
            rows.append(row)
            csvsave(out / "ber.csv", rows)
            emit("BASELINE", **row)
    save(out / "result.json", dict(status="complete", rows=rows))
