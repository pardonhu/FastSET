"""Freeze target RT frames before resident receiver timing."""

import pickle
from pathlib import Path
from ..common import read, save
from ..data import sha
from ..protocol import load_manifest


def run(args):
    from ..bootstrap import initialize_gpu

    initialize_gpu(read(args.config)["seed"], args.output)
    from .stream import checked

    m = load_manifest(read(args.manifest), args.data_root, True)
    m["validation"] = m.pop("test56")
    result = {}
    for role in ("online_train", "validation", "evaluation"):
        rows = []
        for i, e in enumerate(m[role]):
            p = args.output / role / f"batch_{i:05d}.pkl"
            p.parent.mkdir(exist_ok=True)
            with p.open("xb") as f:
                pickle.dump(checked(e), f, protocol=4)
            rows.append(dict(path=str(p.relative_to(args.output)), snr=e["snr"], sha256=sha(p)))
            if (i + 1) % 100 == 0:
                print("MATERIALIZE", role, i + 1, flush=True)
        result[role] = rows
    save(args.output / "manifest.json", result)
    save(
        args.output / "result.json",
        dict(status="complete", source_manifest_sha256=sha(args.manifest)),
    )
