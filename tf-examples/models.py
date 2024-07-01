import sys
from typing import Optional
from pathlib import Path
import csv

import keras
from keras import layers
from keras import ops
import tensorflow as tf

from utils import windows_to_wsl_path

def create_discriminator() -> keras.Sequential:
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    discriminator.summary()
    return discriminator


def create_generator(latent_dim: int = 128) -> keras.Sequential:
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    generator.summary()
    return generator


class GAN(keras.Model):
    def __init__(
            self,
            discriminator: keras.Sequential,
            generator: keras.Sequential,
            latent_dim: int
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)

        self.d_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.g_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.loss_fn: Optional[keras.losses.Loss] = None
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(
            self,
            d_optimizer: keras.optimizers.Optimizer,
            g_optimizer: keras.optimizers.Optimizer,
            loss_fn: keras.losses.Loss
    ):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = ops.concatenate([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = ops.concatenate([ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(42)
        self.output_dir = Path(r"C:\tmp\DCGAN\output")
        self.models_output = Path(r"C:\tmp\DCGAN\models-tf")
        self.log_filepath = Path(r"C:\tmp\DCGAN\models-tf\log.csv")

        if sys.platform == 'linux':
            self.output_dir = windows_to_wsl_path(self.output_dir)
            self.models_output = windows_to_wsl_path(self.models_output)
            self.log_filepath = windows_to_wsl_path(self.log_filepath)

        create_log_file_with_headers(log_file_path=self.log_filepath)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = keras.random.normal(
            shape=(self.num_img, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(
                str(self.output_dir / ("generated_img_%03d_%d.png" % (epoch, i)))
            )

        with open(self.log_filepath, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, logs['g_loss'], logs['d_loss']])

        # if epoch % 1 == 0:
        self.model.generator.save(
            str(self.models_output / f"generator_{epoch}.keras")
        )

        self.model.discriminator.save(
            str(self.models_output / f"discriminator_{epoch}.keras")
        )


def create_log_file_with_headers(log_file_path):
    headers = ['Epoch', 'g_loss', 'd_loss']
    with open(log_file_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(headers)


def create_gan(latent_dim: int) -> GAN:
    """"""
    discriminator = create_discriminator()
    generator = create_generator(latent_dim=latent_dim)
    gan = GAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=latent_dim
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    return gan

