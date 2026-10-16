"""Prepare an explicitly non-paper, two-iteration GPU smoke configuration.

This only writes small JSON files; it neither reads batch payloads nor runs a GPU.
Use the source-pretrained checkpoint appropriate to the selected architecture.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    original = json.loads(args.manifest.read_text())
    from fastset.protocol import validate

    validate(config)
    online = config["online"]
    for field in online:
        if field != "early_stop":
            online[field] = 0
    method = config["method"]
    if method == "fastset":
        online.update(est_iterations=1, dem_iterations=1)
    elif method in {"est_only", "dem_only"}:
        online[method.split("_")[0] + "_iterations"] = 2
    else:
        online[method + "_iterations"] = 2
    online["early_stop"] = False
    validate(config)
    manifest = {
        "source_train": [],
        "online_train": original["online_train"][:2],
        "validation": original["validation"][:1],
    }
    if len(manifest["online_train"]) != 2 or len(manifest["validation"]) != 1:
        raise ValueError("Smoke needs two train batches and one validation batch")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("config.json", config), ("manifest.json", manifest)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    (args.output / "NOT_PAPER_RESULTS.txt").write_text(
        "Two-iteration software smoke, one validation batch; not a full paper experiment.\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
