"""New TDL datasets with explicit seed/SNR/role lineage.

Re-generation is not a promise of bit-identical reproduction of archived inputs.
The published input manifests, rather than this seed alone, identify paper data.
"""

from pathlib import Path
import pickle
import numpy as np
from .common import read, save
from .data import sha

KEYS = (
    "ebno_db",
    "batch_b",
    "batch_tx_codeword_bits",
    "batch_x_rg",
    "batch_pilots_rg",
    "batch_h_freq",
    "batch_y",
    "batch_N0",
    "batch_h_ls_est",
)


def generate(args):
    config = read(args.config)
    required = {
        "seed",
        "role",
        "tdl_model",
        "delay_spread_ns",
        "min_speed_m_s",
        "max_speed_m_s",
        "snrs_db",
        "batches_per_snr",
        "batch_size",
    }
    if set(config) != required or config["role"] not in {
        "source_train",
        "online_train",
        "validation",
        "evaluation",
    }:
        raise ValueError("Invalid generation schema")
    if config["batch_size"] != 32 or config["batches_per_snr"] < 1 or not config["snrs_db"]:
        raise ValueError("Expected batch32 and a nonempty data budget")
    if len(set(config["snrs_db"])) != len(config["snrs_db"]):
        raise ValueError("Repeated SNRs would overwrite data")
    from .bootstrap import initialize_gpu

    tf, _ = initialize_gpu(config["seed"], args.output)
    from .generator import BatchDataGenerator
    from .link import BasicConfig

    rows = []
    # Retain the source generator's CPU channel/MatrixSquareRoot placement.
    with tf.device("/CPU:0"):
        for snr in config["snrs_db"]:
            link = BasicConfig(
                tdl_model=config["tdl_model"],
                delay_spread=config["delay_spread_ns"] * 1e-9,
                min_speed=config["min_speed_m_s"],
                max_speed=config["max_speed_m_s"],
            )
            gen = BatchDataGenerator(link, np.array([float(snr)]))
            folder = args.output / config["role"] / f"{snr:.2f}"
            folder.mkdir(parents=True, exist_ok=False)
            for i in range(config["batches_per_snr"]):
                path = folder / f"batch_{i:04d}.pkl"
                batch = dict(zip(KEYS, gen.receive_data(32)))
                with path.open("xb") as stream:
                    pickle.dump(batch, stream, protocol=4)
                rows.append(
                    {"path": str(path.relative_to(args.output)), "snr": snr, "sha256": sha(path)}
                )
            save(args.output / "progress.json", {"snr": snr, "batches": len(rows)})
            print("GENERATED", config["role"], snr, len(rows), flush=True)
    save(args.output / "manifest.json", {config["role"]: rows})
    save(
        args.output / "lineage.json",
        {
            "config": config,
            "kind": "new_generated_dataset",
            "bit_identical_to_paper_data": False,
            "note": "RNG stream depends on full SNR order, generator calls and pinned environment",
        },
    )
    save(args.output / "result.json", {"status": "complete", "batches": len(rows)})
