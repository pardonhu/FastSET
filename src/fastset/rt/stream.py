"""Recreate frozen RT links with the campaign's per-frame random streams.

Channel geometry is stored once. Payload, AWGN and interference are derived
from (position, role, frame), so validation and evaluation use different data.
The same payload/noise realization is retained across SNR for paired curves.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np


@lru_cache(1)
def link():
    from ..link import BasicConfig

    return BasicConfig()


@lru_cache(16)
def index(path):
    return json.loads(Path(path).read_text())


@lru_cache(256)
def channel(path, row):
    with np.load(path, allow_pickle=False) as z:
        h = z[f"h{row}"].copy()
    if h.shape != (14, 264) or not np.isfinite(h).all() or not np.any(h != 0):
        raise ValueError("Invalid frozen RT channel")
    return h


def checked(e):
    import tensorflow as tf
    import sionna
    from .interference import add_interference

    c = link()
    bits = []
    noise = []
    hs = []
    mapping = index(e["path"])[e["channel_group"]]
    if len(e["samples"]) != 32:
        raise ValueError("RT storage batch must contain 32 frames")
    for pos, frame in e["samples"]:
        seed = int.from_bytes(
            hashlib.sha256(f'rt_full_v2|42|{pos}|{e["role"]}|{frame}'.encode()).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        bits.append(rng.integers(0, 2, (1, 1, 8142)).astype("float32"))
        noise.append(
            ((rng.normal(size=(14, 264)) + 1j * rng.normal(size=(14, 264))) / np.sqrt(2)).astype(
                "complex64"
            )
        )
        item = mapping[pos]
        p = (Path(e["data_root"]) / item["path"]).resolve()
        if not p.is_relative_to(Path(e["data_root"]).resolve()):
            raise ValueError("RT channel path escapes data root")
        hs.append(channel(str(p), item["row"]))
    b = np.stack(bits)
    h = np.stack(hs)
    std = np.stack(noise)
    with tf.device("/CPU:0"):
        cw = c._encoder(b)
        x = c._rg_mapper(c._mapper(cw)).numpy()[:, 0, 0]
        no = float(sionna.utils.ebnodb2no(float(e["snr"]), 4, c._coderate, c._rg).numpy())
        y = h * x + std * np.sqrt(no)
        pilots = np.zeros((32, 14, 264, 1), np.complex64)
        for sym in (2, 11):
            pilots[:, sym, :, 0] = x[:, sym]
    d = dict(
        ebno_db=np.full(32, e["snr"]),
        batch_b=b,
        batch_tx_codeword_bits=cw.numpy(),
        batch_y=y,
        batch_h_freq=h,
        batch_N0=np.full(32, no, np.float32),
        batch_pilots_rg=pilots,
    )
    if e.get("sir_db") is not None:
        d = add_interference(d, e, e["sir_db"])
    return d
