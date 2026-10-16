"""Deterministic narrowband interference, independent of target bits and AWGN."""

import hashlib
import numpy as np


def add_interference(data, entry, sir_db, start=116, count=33):
    assert 0 <= start < start + count <= 264
    h = data["batch_h_freq"]
    y = data["batch_y"]
    assert h.shape == y.shape == (32, 14, 264)
    reference = np.mean(np.abs(h) ** 2, axis=(1, 2), dtype=np.float64)
    amplitude = np.sqrt(reference * 10.0 ** (-sir_db / 10.0) * 264 / count)
    z = np.zeros(y.shape, dtype=np.complex64)
    for j, (pos, frame) in enumerate(entry["samples"]):
        key = f'rt_narrowband_v1|42|{pos}|{entry["role"]}|{frame}'
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        signs = 2 * np.random.default_rng(seed).integers(0, 2, (2, 14, count)) - 1
        z[j, :, start : start + count] = (signs[0] + 1j * signs[1]) * amplitude[j] / np.sqrt(2)
    out = dict(data)
    out["batch_y"] = (y + z).astype(np.complex64)
    return out
