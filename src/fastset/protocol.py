"""Explicit experiment rules; no TensorFlow dependency."""

from pathlib import Path
import math
from .common import architecture, order
from .data import sha

METHODS = {"fastset", "est_only", "dem_only", "joint", "cnn"}


def validate(config):
    required = {
        "seed",
        "batch_size",
        "learning_rate",
        "adam_epsilon",
        "method",
        "architecture",
        "pretrain",
        "online",
        "batch_order",
        "target_ber",
    }
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an explicit uint32 integer")
    if config["batch_size"] != 32:
        raise ValueError("This release preserves the measured batch_size=32 protocol")
    if config["method"] not in METHODS:
        raise ValueError("Unsupported method")
    for field in ("learning_rate", "adam_epsilon", "target_ber"):
        if not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be finite and positive")
    if config["target_ber"] >= 1:
        raise ValueError("target_ber must be less than one")
    if config["batch_order"] not in {"global", "restart_each_stage"}:
        raise ValueError("Declare the measured batch-order convention")
    spec = config["architecture"]
    if spec == "CNN":
        if config["method"] != "cnn":
            raise ValueError("CNN architecture requires method=cnn")
    else:
        if config["method"] == "cnn" or set(spec) != {"depth", "width"}:
            raise ValueError("A split needs depth and width")
        if (
            type(spec["depth"]) is not int
            or not 1 <= spec["depth"] <= 10
            or spec["width"] not in (8, 16, 32, 64)
        ):
            raise ValueError("Only the measured depth1..10 x width8/16/32/64 grid is supported")
        architecture(spec["depth"], spec["width"])
    pre = config["pretrain"]
    if set(pre) != {"mode", "est_epochs", "dem_epochs", "joint_epochs", "cnn_epochs"}:
        raise ValueError("Incomplete pretraining protocol")
    if pre["mode"] not in {"sequential", "joint", "cnn"}:
        raise ValueError("Unknown pretraining mode")
    for key, value in pre.items():
        if key != "mode" and (type(value) is not int or value < 0):
            raise ValueError("Pretraining budgets must be nonnegative integer epochs")
    if (pre["mode"] == "cnn") != (spec == "CNN"):
        raise ValueError("Pretraining mode and model disagree")
    pre_stages = {"sequential": {"est", "dem"}, "joint": {"joint"}, "cnn": {"cnn"}}[pre["mode"]]
    if any(pre[s + "_epochs"] != 0 for s in {"est", "dem", "joint", "cnn"} - pre_stages):
        raise ValueError("Inactive pretraining budgets must be zero")
    online = config["online"]
    if set(online) != {
        "est_iterations",
        "dem_iterations",
        "joint_iterations",
        "cnn_iterations",
        "early_stop",
    }:
        raise ValueError("Incomplete online protocol")
    if type(online["early_stop"]) is not bool:
        raise ValueError("early_stop must be a boolean")
    for key, value in online.items():
        if key != "early_stop" and (type(value) is not int or value < 0):
            raise ValueError("Online budgets must be nonnegative integer iterations")
    active = {
        "fastset": {"est", "dem"},
        "est_only": {"est"},
        "dem_only": {"dem"},
        "joint": {"joint"},
        "cnn": {"cnn"},
    }[config["method"]]
    if any(online[s + "_iterations"] != 0 for s in {"est", "dem", "joint", "cnn"} - active):
        raise ValueError("Inactive online budgets must be zero")
    if not schedule(config):
        raise ValueError("An online experiment needs a positive iteration budget")
    return config


def candidate(config):
    spec = config["architecture"]
    return (
        {"id": "CNN", "parameters": 661188}
        if spec == "CNN"
        else architecture(spec["depth"], spec["width"])
    )


def schedule(config):
    method, online = config["method"], config["online"]
    stages = {
        "fastset": ["est", "dem"],
        "est_only": ["est"],
        "dem_only": ["dem"],
        "joint": ["joint"],
        "cnn": ["cnn"],
    }[method]
    return [stage for stage in stages for _ in range(online[stage + "_iterations"])]


def batch_indices(config, n):
    seq = schedule(config)
    if n <= 0:
        raise ValueError("Empty online training split")
    # This is deliberately bounded to the single-permutation historical protocol.
    # Extending across epochs requires a separately declared experiment.
    if (len(seq) if config["batch_order"] == "global" else max(seq.count(s) for s in set(seq))) > n:
        raise ValueError("Budget exceeds one data permutation; do not silently cycle batches")
    ids = order(n, 1, config["seed"])[0]
    if config["batch_order"] == "global":
        return ids[: len(seq)]
    seen, result = {}, []
    for stage in seq:
        i = seen.get(stage, 0)
        result.append(ids[i])
        seen[stage] = i + 1
    return result


def first_hit(rows, target):
    return next((r for r in rows if r["bit_errors"] / r["bits"] < target), None)


def load_manifest(value, data_root, verify_hashes=False):
    """Resolve only paths beneath a user-selected data root; detect split leakage."""
    allowed = {"source_train", "online_train", "validation", "evaluation"}
    if not set(value) <= allowed or not {"online_train", "validation"} <= set(value):
        raise ValueError("Manifest must contain online_train and validation")
    root = Path(data_root).resolve(strict=True)
    result, owners = {}, {}
    for role, entries in value.items():
        result[role] = []
        for entry in entries:
            rel = Path(entry["path"])
            path = (root / rel).resolve(strict=True)
            if rel.is_absolute() or not path.is_relative_to(root):
                raise ValueError("Manifest path escapes data root")
            digest = entry.get("sha256")
            if digest is not None and (
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid SHA256")
            if verify_hashes and (digest is None or sha(path) != digest):
                raise ValueError(f"Missing/mismatched data hash: {rel}")
            # Split overlap is rejected both by physical path and supplied digest.
            for identity in (str(path), digest):
                if identity is None:
                    continue
                if identity in owners and owners[identity] != role:
                    raise ValueError("Data overlap across splits")
                owners[identity] = role
            result[role].append({**entry, "path": str(path)})
    if not result["online_train"] or not result["validation"]:
        raise ValueError("Training and validation must be nonempty")
    result.setdefault("source_train", [])
    # Alias only at the adapter boundary; historical runtime expects test56.
    result["test56"] = result.pop("validation")
    return result
