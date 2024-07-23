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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
np.random.seed(42)


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
        self.sub_path = path + "/generate_and_save/WGANGP2/" + tcga_name + "/checkpoints"
        self.checkpoint_dir = self.sub_path + "/training_checkpoints"
        self.plot_dir = self.sub_path + "/plot"

        mkdir(self.sub_path)
        mkdir(self.checkpoint_dir)
        mkdir(self.plot_dir)

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

    def checkpoint(self):
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        return checkpoint

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
        gen_loss_results = []
        disc_loss_results = []
        gen_loss_test_avg_results = []
        disc_loss_test_avg_results = []
        start = time.time()

        summary_writer = tf.summary.create_file_writer(self.sub_path)

        print("******************************train**********************************************")
        for epoch in range(self.epochs):
            # train
            epoch_gen_loss_avg = tf.keras.metrics.Mean()
            epoch_disc_loss_avg = tf.keras.metrics.Mean()
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

                    epoch_disc_loss_avg(disc_loss)

                with tf.GradientTape() as gen_tape:
                    fake_x = self.generator([noise, train_y], training=True)
                    gen_loss = self.generator_loss(train_x, fake_x)

                    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                self.generator_optimizer.apply_gradients(
                    zip(gradients_of_generator, self.generator.trainable_variables))

                epoch_gen_loss_avg(gen_loss)
            disc_loss_results.append(epoch_disc_loss_avg.result())
            gen_loss_results.append(epoch_gen_loss_avg.result())

            # test
            epoch_disc_loss_test_avg = tf.keras.metrics.Mean()
            epoch_gen_loss_test_avg = tf.keras.metrics.Mean()
            for step, test in enumerate(self.test_dataset):
                test_x, test_y = test[0], test[1]
                noise = self.get_z(test_x.shape[0])

                fake_x = self.generator([noise, test_y])
                # cp = 10 * tf.reduce_mean(self.F1(fake_x, test_x) + self.F2(fake_x, test_x))
                # gen_loss = self.generator_loss(test_x, fake_x) + cp
                gen_loss = self.generator_loss(test_x, fake_x)

                real_logits = self.discriminator([test_x, test_y])
                fake_logits = self.discriminator([fake_x, test_y])

                gp = self.gradient_penalty(test_x, fake_x, test_y)
                disc_loss = self.discriminator_loss(real_logits, fake_logits) + gp

                # disc_loss = self.discriminator_loss1(fake_output, real_output)

                epoch_disc_loss_test_avg(disc_loss)
                epoch_gen_loss_test_avg(gen_loss)
            gen_loss_test_avg_results.append(epoch_gen_loss_test_avg.result())
            disc_loss_test_avg_results.append(epoch_disc_loss_test_avg.result())

            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', epoch_gen_loss_avg.result(), step=epoch)
                tf.summary.scalar('disc_loss', epoch_disc_loss_avg.result(), step=epoch)
            if (epoch + 1) % 100 == 0:
                print(
                    "Epoch {:03d}, Train_gen_Loss: {:.3f},Train_disc_Loss: {:.3f}, "
                    "Test_gen_Loss: {:.3f},Test_disc_Loss: {:.3f}".format(
                        epoch + 1,
                        epoch_gen_loss_avg.result(),
                        epoch_disc_loss_avg.result(),
                        epoch_gen_loss_test_avg.result(),
                        epoch_disc_loss_test_avg.result()
                    ))

            if (epoch + 1) % 5000 == 0:
                checkpoint_prefix = self.checkpoint_dir + "/cp-" + str(epoch + 1) + '.ckpt'
                self.checkpoint().save(file_prefix=checkpoint_prefix)
                print("Saving checkpoint for epoch{} at {}".format(epoch + 1, checkpoint_prefix))

        print("****************************Generates********************************************")

        generated, condition = self.get_sample()
        np.savez(self.sub_path + '/Loss_Gen.npz', gen_x=generated,
                 gen_y=condition,
                 disc_loss=disc_loss_results,
                 gen_loss=gen_loss_results,
                 gen_loss_test_avg_results=gen_loss_test_avg_results,
                 disc_loss_test_avg_results=disc_loss_test_avg_results)

        print("******************************time***********************************************")
        elapsed_time = time.time() - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    z_dim = 64
    path = '../../datasets/datasets'
    dataset = ["THCA"]
    for tcga_name in dataset:
        X, y = read_dataset_standardscaler(path, tcga_name)
        one = OneHotEncoder()
        num_label = one.fit_transform(y.values)
        y = num_label.toarray()

        x_dim = X.shape[1]
        y_dim = y.shape[1]

        cvae = ConditionalVAE(z_dim, y_dim, x_dim)
        CVAE(cvae, X, y, z_dim, tcga_name, batch_size=32, model="CVAE").train_step()

        cvae.encoder.trainable = False
        cvae.decoder.trainable = True
        generator = cvae.decoder
        # generator = build_generator(x_dim, z_dim, y_dim)

        discriminator = build_discriminator_label(x_dim, y_dim)
        CVAWGANGP(generator, discriminator, X, y, z_dim, tcga_name).main()
