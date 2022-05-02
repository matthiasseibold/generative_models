from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

DEBUG = False


def build_model(img_width, img_height, num_channels, latent_dim, num_classes):

    def conv_block(
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            use_bn=False,
            use_dropout=False,
            drop_value=0.5,
    ):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    IMG_SHAPE = (img_width, img_height, num_channels)

    discriminator_in_channels = IMG_SHAPE[2] + num_classes

    img_input = layers.Input(shape=(IMG_SHAPE[0], IMG_SHAPE[1], discriminator_in_channels))
    # Zero pad the input to make the input images size to (32, 32, 1).
    # x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        img_input,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=True,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    discriminator = keras.models.Model(img_input, x, name="discriminator")


    def upsample_block(
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            up_size=(2, 2),
            padding="same",
            use_bn=False,
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,):

        x = layers.UpSampling2D(up_size)(x)
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = layers.BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x


    generator_in_channels = latent_dim + num_classes

    noise = layers.Input(shape=(generator_in_channels,))
    x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(
        x,
        256,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 1, None, strides=(1, 1), use_bias=False, use_bn=True
    )
    # At this point, we have an output which has the same shape as the input, (32, 32, 1).
    # We will use a Cropping2D layer to make it (28, 28, 1).
    # x = layers.Cropping2D((2, 2))(x)

    generator = keras.models.Model(noise, x, name="generator")

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