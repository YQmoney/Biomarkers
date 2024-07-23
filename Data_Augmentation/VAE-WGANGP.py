# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 14:44
# @Author  : Yu Qian
# @FileName: CVAEWGAN-GP.py
# @annotation:

import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

from CVAE import ConditionalVAE, CVAE
# from plot import loss
from nets import build_discriminator_label, build_generator
from Get_data import read_dataset_standardscaler, mkdir
import os

class CVAWGANGP(keras.Model):
    def __init__(self, generate, discriminate, data, y, z_dim, tcga_name, path='../../datasets',
                 gradient_penalty_weight=10.0, batch_size=32, lr=1e-5, epochs=10000, **kwargs):
        super(CVAWGANGP, self).__init__(**kwargs)
        """
         Wasserstein Generative Adversarial Network  + Gradient Penalty
        """
        self.gradient_penalty_weight = gradient_penalty_weight
        self.data = data
        self.y = y
        self.z_dim = z_dim
        self.y_dim = self.y.shape[1]

        self.epochs = epochs
        self.tr_max = np.max(self.data)
        self.generator = generate
        self.discriminator = discriminate
        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

        self.critic = 2
        self.num_to_generate = 1000
        self.num_classes = self.y.shape[1]

        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.data, self.y,
                                                                                              test_size=0.2,
                                                                                              random_state=42,
                                                                                              shuffle=True)

        self.train_dataset = (tf.data.Dataset.from_tensor_slices(
            (self.train_data.astype('float32'), self.train_label.astype('float32'))).batch(batch_size))
        self.test_dataset = (tf.data.Dataset.from_tensor_slices(
            (self.test_data.astype('float32'), self.test_label.astype('float32'))).batch(batch_size))

    def get_sample(self):
        numbers_tensor = tf.range([self.y_dim])
        repeated_tensor = tf.repeat(numbers_tensor, repeats=self.num_to_generate)
        condition = tf.one_hot(repeated_tensor, depth=self.y_dim, dtype=tf.int32)

        latent_samples = np.random.normal(0, 1, size=(self.num_to_generate * self.y_dim, self.z_dim))
        generated_data = self.generator.predict([latent_samples, condition])
        return generated_data, condition

    def gradient_penalty(self, data, generated_rlds, y):
        batch_size = data.shape[0]
        epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=self.tr_max)
        x_hat = (epsilon * data) + ((1 - epsilon) * generated_rlds)
        with tf.GradientTape() as gp:
            gp.watch(x_hat)
            d_hat = self.discriminator([x_hat, y])
        gradients = gp.gradient(d_hat, [x_hat])[0]
        d_gradient = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        d_regularizer = 10 * tf.reduce_mean(tf.square(d_gradient - 1.))
        return d_regularizer

    def generator_loss(self, train_x, fake_x):
        loss = tf.reduce_mean(tf.square(fake_x - train_x))

        # Huber loss
        # h = tf.keras.losses.Huber()
        # loss = h(fake_x - train_x).numpy()

        return loss

    def discriminator_loss(self, real_output, fake_output):
        """

        :param real_output:
        :param fake_output:
        :return:
        """
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def get_z(self, batch_size=1):
        """

        :param batch_size:
        :return:
        """
        gene_x = np.random.normal(0, 1, size=(batch_size, self.z_dim))
        return gene_x

    def discriminator_loss1(self, fake_output, real_output):
        """
        Calculate the loss for the discriminator
        :param fake_output:
        :param real_output:
        :return:
        """
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
        total_loss = real_loss + fake_loss
        return total_loss

    def main(self):
        start = time.time()
        for epoch in range(self.epochs):
            for step, train in enumerate(self.train_dataset):
                train_x, train_y = train[0], train[1]
                noise = self.get_z(train_x.shape[0])

                for _ in range(self.critic):
                    with tf.GradientTape() as disc_tape:
                        fake_x = self.generator([noise, train_y], training=False)
                        real_output = self.discriminator([train_x, train_y], training=True)
                        fake_output = self.discriminator([fake_x, train_y], training=True)

                        gp = self.gradient_penalty(train_x, fake_x, train_y)
                        disc_loss = self.discriminator_loss(real_output, fake_output) + gp

                        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                        self.discriminator.trainable_variables)

                    self.discriminator_optimizer.apply_gradients(
                        zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                with tf.GradientTape() as gen_tape:
                    fake_x = self.generator([noise, train_y], training=True)
                    gen_loss = self.generator_loss(train_x, fake_x)

                    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                self.generator_optimizer.apply_gradients(
                    zip(gradients_of_generator, self.generator.trainable_variables))

            for step, test in enumerate(self.test_dataset):
                test_x, test_y = test[0], test[1]
                noise = self.get_z(test_x.shape[0])

                fake_x = self.generator([noise, test_y])

                gen_loss = self.generator_loss(test_x, fake_x)

                real_logits = self.discriminator([test_x, test_y])
                fake_logits = self.discriminator([fake_x, test_y])

                gp = self.gradient_penalty(test_x, fake_x, test_y)
                disc_loss = self.discriminator_loss(real_logits, fake_logits) + gp
