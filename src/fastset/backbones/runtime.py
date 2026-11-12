"""Batch32 A2B runtime for the frozen Transformer/Mixer architectures."""

from pathlib import Path
import numpy as np
from ..common import read
from ..model_runtime import checked


def runtime(root, cfg, item, out):
    from ..bootstrap import initialize_gpu

    tf, sionna = initialize_gpu(cfg["seed"], out)
    from ..link import BasicConfig
    from .models import Receiver
    from .spec import Spec

    c = BasicConfig()
    specs = {k: Spec(**v) for k, v in cfg["architecture"]["specs"].items()}
    m = read(root / "data/manifest.json")
    d = checked(m["online_train"][0])
    mode = cfg["architecture"]["mode"]

    def inputs(dd):
        return [
            tf.convert_to_tensor(dd[k])
            for k in ["batch_pilots_rg", "batch_y", "batch_N0", "batch_h_freq"]
        ]

    def ri(h):
        return tf.stack([tf.math.real(h), tf.math.imag(h)], -1)

    def ls(y, n):
        hh, _ = c._ls_est([y[:, None, None], n])
        return ri(tf.squeeze(hh, axis=[1, 2, 3, 4]))

    class Split(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.est = Receiver(specs["est"])
            self.dem = Receiver(specs["dem"])

        def call(self, x, training=False):
            p, y, n, h = x
            hh = self.est([p, y, n], training=training) + ls(y, n)
            return self.dem([p, hh, y, n], training=training)

    with tf.device("/GPU:0"):
        if mode == "e2e":
            model = Receiver(specs["e2e"])
            model(inputs(d)[:3])
            parts = {"cnn": model}
        else:
            model = Split()
            model(inputs(d))
            parts = {"est": model.est, "dem": model.dem}
        expected = (
            specs["e2e"].parameters()
            if mode == "e2e"
            else specs["est"].parameters() + specs["dem"].parameters()
        )
        assert model.count_params() == expected
        assert all("GPU" in w.device for w in model.weights)
        opts = {
            k: tf.keras.optimizers.Adam(cfg["learning_rate"], epsilon=cfg["adam_epsilon"])
            for k in parts
        }
        for k, o in opts.items():
            o.build(parts[k].trainable_variables)
    ck = tf.train.Checkpoint(
        model=model,
        **{k + "_optimizer": o for k, o in opts.items()},
        sionna_rng=sionna.config.tf_rng,
    )

    def logits(z):
        return tf.reshape(
            c._rg_demapper(sionna.utils.insert_dims(z, 2, axis=1)), [tf.shape(z)[0], 1, 1, c._n]
        )

    steps = {}
    for stage in parts:

        def make_step(stage):
            @tf.function
            def step(p, y, n, h, b):
                with tf.device("/GPU:0"):
                    hls = ls(y, n) if stage != "cnn" else None
                    hh = (
                        tf.stop_gradient(model.est([p, y, n], training=False)) + hls
                        if stage == "dem"
                        else None
                    )
                    with tf.GradientTape() as tape:
                        if stage == "est":
                            loss = tf.reduce_mean(
                                tf.square(model.est([p, y, n], training=True) + hls - ri(h))
                            )
                        else:
                            z = (
                                model([p, y, n], training=True)
                                if stage == "cnn"
                                else model.dem([p, hh, y, n], training=True)
                            )
                            loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=logits(z))
                            )
                    grads = tape.gradient(loss, parts[stage].trainable_variables)
                    for g in grads:
                        tf.debugging.assert_all_finite(g, "gradient")
                    opts[stage].apply_gradients(zip(grads, parts[stage].trainable_variables))
                    return loss

            return step

        steps[stage] = make_step(stage)

    @tf.function
    def predict(p, y, n, h):
        with tf.device("/GPU:0"):
            z = logits(model([p, y, n] if mode == "e2e" else [p, y, n, h], training=False))
            tf.debugging.assert_all_finite(z, "LLR")
            return c._decoder(z)

    def update(stage, dd):
        v = steps[stage](*inputs(dd), tf.convert_to_tensor(dd["batch_tx_codeword_bits"]))
        assert "GPU" in v.device
        value = float(v.numpy())
        tf.test.experimental.sync_devices()
        assert np.isfinite(value)
        return value

    def evaluate(entries):
        errors = bits = 0
        for e in entries:
            dd = checked(e)
            pred = predict(*inputs(dd))
            assert "GPU" in pred.device
            errors += int(np.count_nonzero(pred.numpy() != dd["batch_b"]))
            bits += dd["batch_b"].size
        return dict(bit_errors=errors, bits=bits, ber=errors / bits)

    return tf, model, parts, opts, ck, m, d, update, evaluate
