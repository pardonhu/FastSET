"""Public-API RNG bootstrap for pinned Sionna 0.19.2 / TensorFlow 2.15.1."""

import os
import sys


def import_tensorflow_for_bootstrap():
    # Sionna's public seed setter first lazily creates a Generator, then resets
    # it to the requested seed. Its lazy factory cannot run under determinism.
    # Defer the environment request BEFORE TensorFlow import, not by disabling
    # an initialized runtime. No model/data execution occurs in this window.
    if "tensorflow" in sys.modules:
        raise RuntimeError("RNG bootstrap must precede TensorFlow import; start a fresh worker")
    requested = os.environ.pop("TF_DETERMINISTIC_OPS", None)
    import tensorflow as tf

    return tf, requested


def seed_and_enable_determinism(tf, sionna, seed, requested):
    if seed is None:
        raise ValueError("An explicit seed is required")
    with tf.device("/CPU:0"):
        sionna.config.seed = int(seed)
        rng = sionna.config.tf_rng
        if "CPU:0" not in rng.state.device:
            raise RuntimeError("Sionna Generator must be created on CPU")
        initial_state = rng.state.numpy().tolist()
    tf.keras.utils.set_random_seed(int(seed))
    # Mandatory, unconditional: all model/link construction and evaluation
    # happen only AFTER this call. No private Sionna state or disable API.
    tf.config.experimental.enable_op_determinism()
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    return dict(
        seed=int(seed),
        requested_tf_deterministic_ops=requested,
        effective_tf_deterministic_ops="1",
        op_determinism_enabled=True,
        rng_device=rng.state.device,
        rng_algorithm=int(rng.algorithm),
        initial_rng_state=initial_state,
        initialization="public Sionna seed on CPU, then enable_op_determinism",
    )
