from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *

DEBUG = False


def build_model(img_width, img_height, num_channels, latent_dim, num_classes):
    # the number of input channel for the generator and discriminator
    generator_in_channels = latent_dim + num_classes
    discriminator_in_channels = num_channels + num_classes
    print(generator_in_channels, discriminator_in_channels)

    # discriminator
    input_img = Input(shape=(img_width, img_height, discriminator_in_channels))
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2))(input_img)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    x_out = Dense(1, activation='sigmoid')(x)

    # build discriminator
    discriminator = keras.Model(input_img, [x_out], name="discriminator")
    discriminator.summary()

    # generator
    input_gen = Input(shape=(generator_in_channels,))
    y = Dense(512)(input_gen)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = Dense(512)(y)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = Dense(units=16 * 16 * 32)(y)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = keras.layers.Reshape((16, 16, 32))(y)
    y = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME")(y)
    y = ReLU()(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME")(y)
    y = ReLU()(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME")(y)
    y = ReLU()(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(y)

    # build generator
    generator = keras.Model([input_gen], y, name="generator")
    generator.summary()

    return generator, discriminator


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, image_size, num_classes):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.image_size = image_size
        self.num_classes = num_classes

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile(run_eagerly=DEBUG)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.image_size * self.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.image_size, self.image_size, self.num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }