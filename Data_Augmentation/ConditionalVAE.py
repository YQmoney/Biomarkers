#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 14:58
# @Author  : yuqian
# @email   : 760203432@qq.com 
# @Site    : 
# @File    : ConditionalVAE.py
# @Software: PyCharm


import time
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
# from plot import tsne, loss, UMAP
from nets import ConditionalVAE, generate_data
from Get_data import read_dataset_standardscaler,gaussian_noise

np.random.seed(42)


class CVAE(keras.Model):
    def __init__(self, cvae, beta=1., **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.cvae = cvae

        self.cvae_loss_tracker = keras.metrics.Mean(name="cvae_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = beta

    @property
    def metrics(self):
        return [
            self.cvae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, inputs):
        train_x, train_y = inputs

        with tf.GradientTape() as vae_tape:
            z_mean, z_log_var, reconstructed_output = self.cvae([train_x, train_y])
            train_x = tf.cast(train_x, dtype="float32")
            reconstructed_output = tf.cast(reconstructed_output, dtype="float32")
            reconstruction_loss = tf.reduce_mean(tf.square(train_x - reconstructed_output))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
            cvae_loss = reconstruction_loss + self.beta * kl_loss
        grad_cvae = vae_tape.gradient(cvae_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad_cvae, self.trainable_variables))

        self.cvae_loss_tracker.update_state(cvae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.cvae_loss_tracker.result(),
            "r_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, inputs):
        test_x, test_y = inputs
        z_mean, z_log_var, reconstructed_output = self.cvae([test_x, test_y])
        test_x = tf.cast(test_x, dtype="float32")
        reconstructed_output = tf.cast(reconstructed_output, dtype="float32")
        reconstruction_loss = tf.reduce_mean(tf.square(test_x - reconstructed_output))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
        cvae_teat_loss = reconstruction_loss + self.beta * kl_loss

        self.cvae_loss_tracker.update_state(cvae_teat_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.cvae_loss_tracker.result(),
            "r_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == '__main__':
    path = '../../dataset/datasets'
    dataset = ["TCGA-BRCA"]

    z_dim = 128
    initial_lr = 1e-3
    epochs = 5000

    for i in range(len(dataset)):
        print("===================================", dataset[i], "==============================================")
        tcga_name = dataset[i].split("-")[1]
        X, y = read_dataset_standardscaler(path, tcga_name)
        # X, _ = drop_cols_minmax(X)
        normal = X[y['type'] == "normal"]
        cancer = X[y['type'] == "cancer"]

        one = OneHotEncoder()
        num_label = one.fit_transform(y.values)
        y = num_label.toarray()

        X = gaussian_noise(X)  # 添加高斯噪声

        x_dim = X.shape[1]
        y_dim = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True)

        cvae = ConditionalVAE(z_dim, y_dim, x_dim)
        vae_trainer = CVAE(cvae, name="vae")
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.5, beta_2=0.9)

        vae_trainer.compile(optimizer=optimizer)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                       restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        init_time = time.time()
        logs = vae_trainer.fit(X_train, y_train, epochs=epochs, batch_size=128,
                               validation_data=(X_test, y_test),
                               callbacks=[early_stopping, reduce_lr]
                               )
        print(f"Total time: {time.time() - init_time} (s)")

        tf.saved_model.save(cvae, "../../dataset/generate_and_save/ConditionalVAE/" + tcga_name + "_cvae")
