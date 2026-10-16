import pickle
import numpy as np
import pytest
from fastset.data import read_arrays, counts


def test_integer_bit_and_block_counts():
    bits = np.zeros((2, 1, 1, 4), dtype=np.float32)
    predicted = bits.copy()
    predicted[0, 0, 0, :2] = 1
    result = counts(bits, predicted)
    assert result == dict(bit_errors=2, bits=8, block_errors=1, blocks=2, ber=0.25, bler=0.5)
    assert all(type(result[k]) is int for k in ("bit_errors", "bits", "block_errors", "blocks"))


def test_restricted_loader_rejects_globals_and_trailing_data(tmp_path):
    p = tmp_path / "batch.pkl"
    p.write_bytes(pickle.dumps({"a": np.zeros(3, dtype=np.float32)}, protocol=4))
    assert read_arrays(p)["a"].shape == (3,)
    p.write_bytes(p.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="Trailing"):
        read_arrays(p)
    p.write_bytes(pickle.dumps(Exception("not an array")))
    with pytest.raises(ValueError, match="Rejected pickle global"):
        read_arrays(p)


def test_nonfinite_inputs_and_nonbinary_metrics_rejected(tmp_path):
    p = tmp_path / "batch.pkl"
    p.write_bytes(pickle.dumps({"a": np.array([np.nan])}, protocol=4))
    with pytest.raises(ValueError, match="finite"):
        read_arrays(p)
    with pytest.raises(ValueError, match="Binary"):
        counts(np.array([0]), np.array([2]))
