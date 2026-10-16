"""Exact shared Joint objective snapshot; all campaign tasks import this one function."""


def make_joint(tf, parts, opts):
    from fastset.link import BasicConfig
    import sionna

    c = BasicConfig()
    ev = parts["est"].trainable_variables
    dv = parts["dem"].trainable_variables

    @tf.function
    def joint(p, y, n, h, b):
        with tf.device("/GPU:0"):
            with tf.GradientTape() as tape:
                hh, var = parts["est"]([p, y, n, h], training=True)
                z = parts["dem"]([p, hh, y, n, var], training=True)
                llr = tf.reshape(
                    c._rg_demapper(sionna.utils.insert_dims(z, 2, axis=1)), [32, 1, 1, c._n]
                )
                target = tf.stack([tf.math.real(h), tf.math.imag(h)], axis=-1)
                mse = tf.reduce_mean(tf.square(tf.squeeze(hh, axis=[1, 2, 3, 4]) - target))
                bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=llr))
                loss = mse + bce
            gs = tape.gradient(loss, ev + dv)
            assert all(g is not None for g in gs)
            for g in gs:
                tf.debugging.assert_all_finite(g, "joint gradient")
            with tf.control_dependencies([tf.identity(g) for g in gs]):
                opts["est"].apply_gradients(zip(gs[: len(ev)], ev))
                opts["dem"].apply_gradients(zip(gs[len(ev) :], dv))
            return loss, mse, bce

    return joint
