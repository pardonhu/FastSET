"""Read-only, restricted NumPy pickle inputs and integer error accounting."""

import hashlib
import pickle
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_arrays(path):
    allowed = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
    }

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise ValueError("Rejected pickle global: " + module + "." + name)
            return allowed[module, name]

    with Path(path).open("rb") as stream:
        obj = Restricted(stream).load()
        if stream.read(1):
            raise ValueError("Trailing pickle content")
    if not isinstance(obj, (dict, list)):
        raise ValueError("Expected ndarray list or dictionary")
    values = obj.values() if isinstance(obj, dict) else obj
    for value in values:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype.kind not in "fciu"
            or not np.isfinite(value).all()
        ):
            raise ValueError("Expected finite numeric ndarray")
    return obj


def counts(bits, prediction):
    bits, prediction = np.asarray(bits), np.asarray(prediction)
    if (
        bits.shape != prediction.shape
        or not np.isin(bits, [0, 1]).all()
        or not np.isin(prediction, [0, 1]).all()
    ):
        raise ValueError("Binary arrays must have equal shapes")
    wrong = bits != prediction
    total, blocks = int(bits.size), int(np.prod(bits.shape[:-1]))
    be, ble = int(wrong.sum()), int(wrong.any(axis=-1).sum())
    return dict(
        bit_errors=be,
        bits=total,
        block_errors=ble,
        blocks=blocks,
        ber=be / total,
        bler=ble / blocks,
    )
