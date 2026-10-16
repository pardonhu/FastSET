import sionna
import tensorflow as tf
import numpy as np
from .link import BasicConfig


class BatchDataGenerator:

    def __init__(self, init_config: BasicConfig, ebNo_dB_range: np.ndarray) -> None:
        """
        初始化 BatchDataGenerator 类。

        参数:
        init_config (BasicConfig): 包含通信系统基本配置的对象。
        ebNo_dB_range (np.ndarray): 信噪比（Eb/No）的取值范围数组。
        """
        self._config = init_config
        self._ebNo_dB_range = ebNo_dB_range

    def change_config(self, config: BasicConfig) -> None:
        """
        更改通信系统的配置。

        参数:
        config (BasicConfig): 新的通信系统配置对象。
        """
        self._config = config

    @tf.function
    def _receive_data(self, ebno_db, batch_size) -> tuple:
        """
        模拟通信系统的发射、信道传输过程，生成一批数据。

        参数:
        ebno_db (tf.Tensor): 每个样本的信噪比（Eb/No），形状为 [batch_size]。
        batch_size (int): 批次大小。

        返回:
        tuple: 包含发送比特、编码比特、调制符号、导频符号、信道频率响应、接收信号等数据的元组。
        """
        batch_N0 = sionna.utils.ebnodb2no(
            ebno_db, self._config._num_bits_per_symbol, self._config._coderate, self._config._rg
        )
        batch_N0 = tf.cast(batch_N0, dtype=tf.float32)
        batch_b = self._config._binary_source(
            [batch_size, 1, self._config._num_streams_per_tx, self._config._k]
        )
        batch_tx_codeword_bits = self._config._encoder(batch_b)
        batch_x = self._config._mapper(batch_tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x)
        cir = self._config._tdl(
            batch_size, self._config._rg.num_ofdm_symbols, 1 / self._config._rg.ofdm_symbol_duration
        )
        batch_h_freq = sionna.channel.cir_to_ofdm_channel(
            self._config._frequencies, *cir, normalize=True
        )
        batch_y = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0])
        batch_h_ls_est, batch_var_ls_est = self._config._ls_est([batch_y, batch_N0])
        mask = np.zeros((self._config._num_ofdm_symbols, self._config._fft_size, 1), dtype=np.bool8)
        for idx in self._config._pilot_ofdm_symbol_indices:
            mask[idx, :, :] = True
        mask = tf.constant(mask, dtype=tf.complex64)
        batch_pilots_rg = mask * tf.transpose(tf.squeeze(batch_x_rg, axis=1), [0, 2, 3, 1])
        batch_x_rg = tf.squeeze(batch_x_rg, axis=[1, 2])
        batch_y = tf.squeeze(batch_y, axis=[1, 2])
        batch_h_freq = tf.squeeze(batch_h_freq, axis=[1, 2, 3, 4])
        batch_h_ls_est = tf.squeeze(batch_h_ls_est, axis=[1, 2, 3, 4])
        return (
            batch_b,
            batch_tx_codeword_bits,
            batch_x_rg,
            batch_pilots_rg,
            batch_h_freq,
            batch_y,
            batch_N0,
            batch_h_ls_est,
        )

    def receive_data(self, batch_size) -> tuple:
        """
        调用 _receive_data 方法生成一批数据，并将 TensorFlow 张量转换为 NumPy 数组。

        参数:
        batch_size (int): 批次大小。

        返回:
        tuple: 包含信噪比、发送比特、编码比特、调制符号、导频符号、信道频率响应、接收信号等数据的元组，数据类型为 NumPy 数组。
        """
        ebno_db = np.array(
            [np.random.choice(self._ebNo_dB_range, 1) for _ in range(batch_size)]
        ).squeeze()
        (
            batch_b,
            batch_tx_codeword_bits,
            batch_x_rg,
            batch_pilots_rg,
            batch_h_freq,
            batch_y,
            batch_N0,
            batch_h_ls_est,
        ) = self._receive_data(ebno_db=ebno_db, batch_size=batch_size)
        return (
            ebno_db,
            batch_b.numpy(),
            batch_tx_codeword_bits.numpy(),
            batch_x_rg.numpy(),
            batch_pilots_rg.numpy(),
            batch_h_freq.numpy(),
            batch_y.numpy(),
            batch_N0.numpy(),
            batch_h_ls_est.numpy(),
        )
