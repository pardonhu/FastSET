"""Cooperative per-device lease. No process killing or system changes."""

import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile


@contextlib.contextmanager
def lease(device, seed):
    # Query only the selected device: an unhealthy unrelated GPU must not block it.
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(device), "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Select exactly one NVIDIA GPU by index or UUID")
    path = Path(tempfile.gettempdir()) / f"fastset-{os.getuid()}-{uuid}.lock"
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if processes:
            raise RuntimeError("Selected GPU is busy; no process was stopped")
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=uuid,
            PYTHONHASHSEED=str(seed),
            TF_DETERMINISTIC_OPS="1",
            TF_CUDNN_DETERMINISTIC="1",
            TF_ENABLE_ONEDNN_OPTS="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            TF_NUM_INTRAOP_THREADS="2",
            TF_NUM_INTEROP_THREADS="2",
            PYTHONDONTWRITEBYTECODE="1",
        )
        yield env
