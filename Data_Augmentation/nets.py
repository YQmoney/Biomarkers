#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/23 17:02
# @Author  : yuqian
# @email   : 760203432@qq.com
# @Site    :
# @File    : nets.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, concatenate, LeakyReLU, Dropout, BatchNormalization
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


def build_generator(x_dim, z_dim, condition_dim, alpha=0.2,  hidden_layer=[1024, 512, 512, 512, 256]):
    """
    auxiliary classifier gan
    :param x_dim:
    :param z_dim:
    :param condition_dim:
    :param hidden_layer:
    :return:
    """
    x_inputs = Input(shape=(z_dim,))
    condition_inputs = Input(shape=(condition_dim,))
    inputs = concatenate([x_inputs, condition_inputs])

    x = inputs
    for layer in hidden_layer:
        x = Dense(layer)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

    output = Dense(x_dim)(x)
    return keras.Model([x_inputs, condition_inputs], output, name="generator")


def build_discriminator_label(x_dim, y_dim, alpha=0.3, hidden_layer=[1024, 512, 512, 512, 256]):
    """

    :param x_dim:
    :param y_dim:
    :param alpha:
    :param hidden_layer:
    :return:
    """
    # 输入层
    x_inputs = Input(shape=(x_dim,))
    y_inputs = Input(shape=(y_dim,))
    inputs = concatenate([x_inputs, y_inputs])

    # 隐藏层
    x = inputs
    for layer in hidden_layer:
        x = Dense(layer)(x)
        x = tf.nn.relu(x)

    # 输出层
    discriminator_logit = Dense(1)(x)

    return keras.Model([x_inputs, y_inputs], discriminator_logit, name="discriminator")


# Conditional VAE 模型
class ConditionalVAE(keras.Model):
    def __init__(self, latent_dim, condition_dim, output_dim,
                 encoder_hidden_layer=[1024, 512, 512, 512, 256],
                 decoder_hidden_layer=[256, 512, 512, 512, 1024], alpha=0.3, dropout=0.3):
        """

        :param latent_dim:
        :param condition_dim:
        :param output_dim:
        :param encoder_hidden_layer:
        :param decoder_hidden_layer:
        :param alpha:
        :param dropout:
        """
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim

        self.encoder_hidden_layer = encoder_hidden_layer
        self.decoder_hidden_layer = decoder_hidden_layer
        self.alpha = alpha
        self.dropout = dropout
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    # 编码器
    def build_encoder(self, alpha=0.2):
        output_inputs = Input(shape=(self.output_dim,))
        condition_inputs = Input(shape=(self.condition_dim,))
        inputs = concatenate([output_inputs, condition_inputs])

        # 隐藏层
        x = inputs
        for layer in self.encoder_hidden_layer:
            x = Dense(layer)(x)
            x = LeakyReLU(alpha=alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        return keras.Model([output_inputs, condition_inputs], [z_mean, z_log_var], name="encoder")

    # 解码器
    def build_decoder(self, alpha=0.2):
        latent_inputs = Input(shape=(self.latent_dim,))
        condition_inputs = Input(shape=(self.condition_dim,))
        inputs = concatenate([latent_inputs, condition_inputs])

        # 隐藏层
        x = inputs
        for layer in self.decoder_hidden_layer:
            x = Dense(layer)(x)
            x = LeakyReLU(alpha=alpha)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)

        outputs = Dense(self.output_dim)(x)
        return keras.Model([latent_inputs, condition_inputs], outputs, name="decoder")

    # 采样函数
    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # VAE 损失函数
    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss

    # 前向传播
    def call(self, inputs):
        x, y = inputs
        z_mean, z_log_var = self.encoder([x, y])
        latent_samples = self.sampling(z_mean, z_log_var)
        reconstructed_output = self.decoder([latent_samples, y])
        return z_mean, z_log_var, latent_samples, reconstructed_output
