import tensorflow as tf
import sionna
import numpy as np


# Grid definitions extracted unchanged in AST from reviewed original source.
class CNNMainNetSRB(tf.keras.Model):

    def __init__(self, num_filters: int, dilation: list, skip_conv11: bool = False):
        super(CNNMainNetSRB, self).__init__(name="CNNMainNetSRB")
        self.trainable = True
        self._layer_norm_1 = tf.keras.layers.LayerNormalization(axis=-1)
        self._conv_1 = tf.keras.layers.SeparableConv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            padding="same",
            dilation_rate=dilation,
            activation=None,
        )
        self._layer_norm_2 = tf.keras.layers.LayerNormalization(axis=-1)
        self._conv_2 = tf.keras.layers.SeparableConv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            padding="same",
            dilation_rate=dilation,
            activation=None,
        )
        if skip_conv11 == True:
            self._skip_conv11 = tf.keras.layers.Conv2D(
                filters=num_filters, kernel_size=[1, 1], padding="same", activation=None
            )
        else:
            self._skip_conv11 = None

    def call(self, inputs, training=False, mode: str = "final"):
        z = self._layer_norm_1(inputs, training=training)
        z = tf.nn.relu(z)
        z = self._layer_norm_2(self._conv_1(z), training=training)
        z = tf.nn.relu(z)
        middle_output = z
        z = self._conv_2(z)
        if self._skip_conv11 != None:
            z = z + self._skip_conv11(inputs)
        else:
            z = z + inputs
        if mode == "final":
            return z
        else:
            return (middle_output, z)


class CNNMainNet(tf.keras.Model):

    def __init__(self):
        super(CNNMainNet, self).__init__(name="CNNMainNet")
        self.train_flops_per_sample = 661188 * 6
        self.trainable = True
        self._input_conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[3, 3], padding="same", activation=None
        )
        self._res_block_0 = CNNMainNetSRB(num_filters=64, dilation=[1, 1])
        self._res_block_1 = CNNMainNetSRB(num_filters=64, dilation=[1, 1])
        self._res_block_2 = CNNMainNetSRB(num_filters=128, dilation=[2, 3], skip_conv11=True)
        self._res_block_3 = CNNMainNetSRB(num_filters=128, dilation=[2, 3])
        self._res_block_4 = CNNMainNetSRB(num_filters=256, dilation=[2, 3], skip_conv11=True)
        self._res_block_5 = CNNMainNetSRB(num_filters=256, dilation=[3, 6])
        self._res_block_6 = CNNMainNetSRB(num_filters=256, dilation=[2, 3])
        self._res_block_7 = CNNMainNetSRB(num_filters=128, dilation=[2, 3], skip_conv11=True)
        self._res_block_8 = CNNMainNetSRB(num_filters=128, dilation=[2, 3])
        self._res_block_9 = CNNMainNetSRB(num_filters=64, dilation=[1, 1], skip_conv11=True)
        self._res_block_10 = CNNMainNetSRB(num_filters=64, dilation=[1, 1])
        self._output_conv = tf.keras.layers.Conv2D(
            filters=4, kernel_size=[3, 3], padding="same", activation=None
        )

    def call(self, inputs, training=False):
        batch_pilots_rg, batch_y, batch_N0 = inputs
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=1, axis=-1)
        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, num_dims=3, axis=1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1])
        z = tf.concat(
            [
                tf.math.real(batch_y),
                tf.math.imag(batch_y),
                tf.math.real(batch_pilots_rg),
                tf.math.imag(batch_pilots_rg),
                batch_N0,
            ],
            axis=-1,
        )
        input_conv_out = self._input_conv(z)
        res0_out = self._res_block_0(inputs=input_conv_out, training=training)
        res1_out = self._res_block_1(inputs=res0_out, training=training)
        res2_out = self._res_block_2(inputs=res1_out, training=training)
        res3_out = self._res_block_3(inputs=res2_out, training=training)
        res4_out = self._res_block_4(inputs=res3_out, training=training)
        res5_out = self._res_block_5(inputs=res4_out, training=training)
        res6_out = self._res_block_6(inputs=res5_out, training=training)
        res7_out = self._res_block_7(inputs=res6_out, training=training)
        res8_out = self._res_block_8(inputs=res7_out, training=training)
        res9_out = self._res_block_9(inputs=res8_out, training=training)
        res10_out = self._res_block_10(inputs=res9_out, training=training)
        output = self._output_conv(res10_out)
        return output


def get_filter_num(block_num, filter_num, i):
    coe = 1
    if block_num > 5:
        coe = 2
    if i == block_num - 1:
        return filter_num
    if i < block_num // 2:
        factor = 2 ** (i // coe)
    else:
        factor = 2 ** ((block_num - i - 1) // coe)
    return filter_num * factor


def get_dilation(i, block_num):
    if block_num > 5:
        dilation_list = [[1, 1], [2, 3], [3, 6], [2, 3], [1, 1]]
    else:
        dilation_list = [[1, 1], [2, 3], [1, 1]]
    index = min(i // 2, 4)
    return dilation_list[index]


class CNNEstv1(tf.keras.Model):

    def __init__(self, config, block_num=5, filter_num=32):
        super(CNNEstv1, self).__init__(name="CNNEstv1")
        self._input_conv = tf.keras.layers.Conv2D(
            filters=filter_num, kernel_size=[3, 3], padding="same", activation=None
        )
        self.blocks = [
            CNNMainNetSRB(
                num_filters=get_filter_num(block_num, filter_num, i),
                dilation=get_dilation(i, block_num),
                skip_conv11=True,
            )
            for i in range(block_num)
        ]
        self._output_conv = tf.keras.layers.Conv2D(
            filters=2, kernel_size=[3, 3], padding="same", activation=None
        )
        self.trainable = True
        self.mode = "neural"
        self.train_flops_per_sample = 0
        self._config = config

    def call(self, inputs, training=False):
        batch_pilots_rg, batch_y, batch_N0, batch_h_true = inputs
        batch_y_ls = sionna.utils.insert_dims(batch_y, num_dims=2, axis=1)
        batch_h_ls_est, var = self._config._ls_est([batch_y_ls, batch_N0])
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=1, axis=-1)
        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, num_dims=3, axis=1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1])
        z = tf.concat(
            [
                tf.math.real(batch_y),
                tf.math.imag(batch_y),
                tf.math.real(batch_pilots_rg),
                tf.math.imag(batch_pilots_rg),
                batch_N0,
            ],
            axis=-1,
        )
        x = self._input_conv(z)
        for block in self.blocks:
            x = block(x, training=training)
        output = self._output_conv(x)
        h_hat = sionna.utils.insert_dims(output, num_dims=4, axis=1)
        batch_h_ls_est = tf.expand_dims(batch_h_ls_est, axis=-1)
        batch_h_ls_est_re = tf.math.real(batch_h_ls_est)
        batch_h_ls_est_im = tf.math.imag(batch_h_ls_est)
        batch_h_ls_est = tf.concat([batch_h_ls_est_re, batch_h_ls_est_im], axis=-1)
        h_hat += batch_h_ls_est
        return (h_hat, var)


class CNNDemapper(tf.keras.Model):

    def __init__(self, config, block_num=10, filter_num=64):
        super(CNNDemapper, self).__init__(name="CNNDemapper")
        self.mode = "neural"
        self.input_conv = tf.keras.layers.Conv2D(
            filters=filter_num, kernel_size=[3, 3], padding="same", activation=None
        )
        self.blocks = [
            CNNMainNetSRB(
                num_filters=get_filter_num(block_num, filter_num, i),
                dilation=get_dilation(i, block_num),
                skip_conv11=True,
            )
            for i in range(block_num)
        ]
        self.output_conv = tf.keras.layers.Conv2D(
            filters=4, kernel_size=[3, 3], padding="same", activation=None
        )
        self.trainable = True
        self.train_flops_per_sample = 0
        self._config = config

    def call(self, inputs, training=False):
        batch_pilots_rg, batch_h_hat, batch_y, batch_N0, batch_var_est = inputs
        batch_h_hat = tf.squeeze(batch_h_hat, axis=[1, 2, 3, 4])
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=1, axis=-1)
        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, num_dims=3, axis=1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1])
        z = tf.concat(
            [
                tf.math.real(batch_y),
                tf.math.imag(batch_y),
                tf.math.real(batch_pilots_rg),
                tf.math.imag(batch_pilots_rg),
                batch_h_hat,
                batch_N0,
            ],
            axis=-1,
        )
        x = self.input_conv(z)
        for block in self.blocks:
            x = block(x, training=training)
        return self.output_conv(x)


class SequentialModel(tf.keras.Model):

    def __init__(self, est, dem):
        super().__init__()
        self.estimator = est
        self.demapper = dem

    def call(self, inputs, training=False):
        batch_pilots_rg, batch_y, batch_N0, batch_h_true = inputs
        batch_h_hat, batch_var_est = self.estimator(inputs, training=training)
        inputs = (batch_pilots_rg, batch_h_hat, batch_y, batch_N0, batch_var_est)
        llr_rg = self.demapper(inputs, training=training)
        return llr_rg
