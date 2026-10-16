"""TF2.15模型。Est预测H-LS残差；Dem由调用者提供残差+LS。

仅在GPU引导成功后导入本模块。2symbol×6tone token保留所有RE；
不加载CNN权重、不擅改原pilot/LDPC/mapper链。
"""

import numpy as np
import tensorflow as tf
from .spec import Spec


def patchify(x):
    tf.debugging.assert_equal(tf.shape(x)[1:3], [14, 264])
    b = tf.shape(x)[0]
    c = tf.shape(x)[3]
    x = tf.reshape(x, [b, 7, 2, 44, 6, c])
    return tf.reshape(tf.transpose(x, [0, 1, 3, 2, 4, 5]), [b, 308, 12 * c])


def unpatchify(x, channels):
    tf.debugging.assert_equal(tf.shape(x)[1:], [308, 12 * channels])
    b = tf.shape(x)[0]
    x = tf.reshape(x, [b, 7, 44, 2, 6, channels])
    return tf.reshape(tf.transpose(x, [0, 1, 3, 2, 4, 5]), [b, 14, 264, channels])


def position_encoding(d):
    """固定二维sin/cos位置编码，行7、列44，顺序与patchify一致。"""
    row, col = np.meshgrid(np.arange(7), np.arange(44), indexing="ij")

    def enc(v):
        phase = v.reshape(-1, 1) / 10000 ** (np.arange(d // 4) / (d // 4))
        return np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)

    return tf.constant(np.concatenate([enc(row), enc(col)], axis=-1)[None], tf.float32)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d, heads, **kw):
        super().__init__(**kw)
        self.n1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.n2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=d // heads, dropout=0.0
        )
        self.f1 = tf.keras.layers.Dense(2 * d, activation=tf.nn.gelu)
        self.f2 = tf.keras.layers.Dense(d)

    def call(self, x, training=False):
        z = self.n1(x)
        x = x + self.attention(z, z, training=training)
        return x + self.f2(self.f1(self.n2(x)))


class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, d, hidden, **kw):
        super().__init__(**kw)
        self.n1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.n2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.t1 = tf.keras.layers.Dense(hidden, activation=tf.nn.gelu)
        self.t2 = tf.keras.layers.Dense(308)
        self.c1 = tf.keras.layers.Dense(2 * d, activation=tf.nn.gelu)
        self.c2 = tf.keras.layers.Dense(d)

    def call(self, x, training=False):
        z = tf.transpose(self.n1(x), [0, 2, 1])
        x = x + tf.transpose(self.t2(self.t1(z)), [0, 2, 1])
        return x + self.c2(self.c1(self.n2(x)))


class Receiver(tf.keras.Model):
    def __init__(self, spec):
        super().__init__(name=spec.family + "_" + spec.role)
        self.spec = spec
        self.projection = tf.keras.layers.Dense(spec.width, name="input_projection")
        self.position = position_encoding(spec.width)
        # 显式属性+列表保证每block同时被Keras和TF checkpoint跟踪。
        self.blocks = []
        for i in range(spec.depth):
            block = (
                TransformerBlock(spec.width, spec.heads, name=f"block_{i}")
                if spec.family == "transformer"
                else MixerBlock(spec.width, spec.token_hidden, name=f"block_{i}")
            )
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(12 * spec.outputs, name="output_projection")

    def features(self, inputs):
        if self.spec.role == "dem":
            p, h, y, no = inputs
        else:
            p, y, no = inputs
        p = tf.cast(p, tf.complex64)
        y = tf.cast(y, tf.complex64)
        no = tf.reshape(tf.cast(no, tf.float32), [-1, 1, 1, 1])
        tf.debugging.assert_positive(no)
        n = tf.broadcast_to(
            tf.math.log(no) / tf.math.log(tf.constant(10.0, tf.float32)),
            [tf.shape(y)[0], 14, 264, 1],
        )
        z = [
            tf.math.real(y)[..., None],
            tf.math.imag(y)[..., None],
            tf.math.real(p),
            tf.math.imag(p),
        ]
        if self.spec.role == "dem":
            z.append(tf.cast(h, tf.float32))
        return tf.concat(z + [n], axis=-1)

    def call(self, inputs, training=False):
        z = self.projection(patchify(self.features(inputs))) + self.position
        for block in self.blocks:
            z = block(z, training=training)
        return unpatchify(self.head(self.norm(z)), self.spec.outputs)


def make_models(specs):
    return {name: Receiver(spec) for name, spec in specs.items()}
