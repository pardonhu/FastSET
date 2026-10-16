"""Shared grid model/link/training runtime for the A-pretrain/B-online pilot.
Derived from the audited exp_grid_scratch runtime; the loader accepts all source
training SNRs. No import or mutation of the frozen scratch experiment runtime.
"""

from pathlib import Path
import numpy as np
from .common import read, sha
from fastset.data import read_arrays, counts


def checked(e):
    d = read_arrays(e["path"])
    assert d["batch_b"].shape == (32, 1, 1, 8142)
    assert d["batch_tx_codeword_bits"].shape == (32, 1, 1, 12672)
    assert d["batch_y"].shape == (32, 14, 264)
    assert np.all(d["ebno_db"] == e["snr"]), e["path"]
    d["batch_N0"] = d["batch_N0"].reshape(-1).astype("float32")
    assert all(np.isfinite(a).all() for a in d.values())
    return d


def runtime(root, cfg, item, out):
    from .bootstrap import initialize_gpu

    tf, sionna = initialize_gpu(cfg["seed"], out)
    from fastset.link import BasicConfig

    c = BasicConfig()
    assert c._n == 12672 and c._k == 8142
    from .models import grid

    ns = vars(grid)
    manifest = read(root / "data/manifest.json")
    d = checked(manifest["online_train"][0])

    def inp(d):
        return [
            tf.convert_to_tensor(d[k])
            for k in ["batch_pilots_rg", "batch_y", "batch_N0", "batch_h_freq"]
        ]

    with tf.device("/GPU:0"):
        if item["id"] == "CNN":
            model = ns["CNNMainNet"]()
            model(inp(d)[:3])
            parts = {"cnn": model}
        else:
            est = ns["CNNEstv1"](c, block_num=item["depth"], filter_num=item["width"])
            dem = ns["CNNDemapper"](c, block_num=item["dem_depth"], filter_num=item["dem_width"])
            model = ns["SequentialModel"](est, dem)
            model(inp(d))
            parts = {"est": est, "dem": dem}
        assert sum(np.prod(v.shape) for v in model.weights) == item["parameters"]
        assert all("GPU" in v.device for v in model.weights)
        opts = {
            k: tf.keras.optimizers.Adam(
                learning_rate=cfg["learning_rate"], epsilon=cfg["adam_epsilon"]
            )
            for k in parts
        }
        for k in parts:
            opts[k].build(parts[k].trainable_variables)
    ck = tf.train.Checkpoint(
        model=model,
        **{k + "_optimizer": v for k, v in opts.items()},
        sionna_rng=sionna.config.tf_rng,
    )

    def llr(z):
        return tf.reshape(
            c._rg_demapper(sionna.utils.insert_dims(z, 2, axis=1)), [tf.shape(z)[0], 1, 1, c._n]
        )

    steps = {}
    if "est" in parts:

        @tf.function
        def est_step(p, y, n, h, b):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    hh, _ = est([p, y, n, h], training=True)
                    target = tf.stack([tf.math.real(h), tf.math.imag(h)], axis=-1)
                    loss = tf.reduce_mean(tf.square(tf.squeeze(hh, axis=[1, 2, 3, 4]) - target))
                grads = tape.gradient(loss, est.trainable_variables)
                opts["est"].apply_gradients(zip(grads, est.trainable_variables))
                return loss

        @tf.function
        def dem_step(p, y, n, h, b):
            with tf.device("/GPU:0"):
                hh, var = est([p, y, n, h], training=False)
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=b, logits=llr(dem([p, hh, y, n, var], training=True))
                        )
                    )
                grads = tape.gradient(loss, dem.trainable_variables)
                opts["dem"].apply_gradients(zip(grads, dem.trainable_variables))
                return loss

        steps = {"est": est_step, "dem": dem_step}
    else:

        @tf.function
        def cnn_step(p, y, n, h, b):
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=b, logits=llr(model([p, y, n], training=True))
                        )
                    )
                grads = tape.gradient(loss, model.trainable_variables)
                opts["cnn"].apply_gradients(zip(grads, model.trainable_variables))
                return loss

        steps = {"cnn": cnn_step}

    @tf.function
    def predict(p, y, n, h):
        with tf.device("/GPU:0"):
            z = llr(model([p, y, n] if "cnn" in parts else [p, y, n, h], training=False))
            tf.debugging.assert_all_finite(z, "LLR")
            return c._decoder(z)

    @tf.function
    def mse(p, y, n, h):
        with tf.device("/GPU:0"):
            hh, _ = est([p, y, n, h], training=False)
            return tf.reduce_mean(
                tf.square(
                    tf.squeeze(hh, axis=[1, 2, 3, 4])
                    - tf.stack([tf.math.real(h), tf.math.imag(h)], axis=-1)
                )
            )

    def evaluate(entries, est_only=False):
        errors = bits = 0
        losses = []
        for e in entries:
            dd = checked(e)
            if est_only:
                losses.append(float(mse(*inp(dd)).numpy()))
            else:
                pred = predict(*inp(dd))
                assert "GPU" in pred.device
                m = counts(dd["batch_b"], pred.numpy())
                errors += m["bit_errors"]
                bits += m["bits"]
        return (
            dict(mse=float(np.mean(losses)))
            if est_only
            else dict(bit_errors=errors, bits=bits, ber=errors / bits)
        )

    def update(stage, dd):
        value = steps[stage](*inp(dd), tf.convert_to_tensor(dd["batch_tx_codeword_bits"]))
        assert "GPU" in value.device
        v = float(value.numpy())
        tf.test.experimental.sync_devices()
        assert np.isfinite(v)
        return v

    return tf, model, parts, opts, ck, manifest, d, update, evaluate
