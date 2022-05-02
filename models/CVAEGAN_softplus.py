import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

from models.resnet import ResNet18

DEBUG = False


def build_model(img_width, img_height, channels, n_latent, n_classes):

    # discriminator
    input_img = Input(shape=(img_width, img_width, channels))
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2))(input_img)
    x = LeakyReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(x)
    features = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2))(x)
    x = LeakyReLU()(features)
    x = GlobalMaxPooling2D()(x)
    x_out = Dense(1, activation='sigmoid')(x)

    # build discriminator
    discriminator = Model(input_img, [x_out, features], name="discriminator")
    discriminator.summary()

    # classifier
    original_model = ResNet18(input_shape=(img_width, img_width, channels), classes=n_classes)
    base_model = Model(original_model.input, original_model.layers[-3].output)
    input_img = base_model.input
    features = base_model.output
    x = GlobalAvgPool2D()(features)
    x_out = Dense(n_classes, activation='softmax')(x)

    # build classifier
    classifier = Model(input_img, [x_out, features], name="classifier")
    classifier.summary()

    def sampling(args):
        mu, log_var = args
        batch = K.shape(mu)[0]
        eps = K.random_normal(shape=(batch, n_latent))
        return mu + K.exp(log_var / 2) * eps

    # encoder
    input_img = Input(shape=(img_width, img_height, channels))
    input_cond = Input(shape=(n_classes,))
    x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    # x_encoded = Dense(n_latent, activation='relu')(x)
    mu = Dense(n_latent, activation='linear')(x)
    log_var = Dense(n_latent, activation='linear')(x)
    # encoder sampler
    z = Lambda(sampling, output_shape=(n_latent,))([mu, log_var])

    encoder = Model([input_img, input_cond], [mu, log_var, z], name='encoder')
    encoder.summary()

    # generator
    input_gen = Input(shape=(n_latent,))
    cond_in = Input(shape=(n_classes,))
    dec_input = Concatenate()([input_gen, cond_in])
    y = Dense(512, activation='relu')(dec_input)
    y = BatchNormalization()(y)
    y = Dense(512, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dense(units=16*16*32, activation='relu')(y)
    y = BatchNormalization()(y)
    y = keras.layers.Reshape((16, 16, 32))(y)
    y = keras.layers.Conv2DTranspose(
        filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation='softplus')(y)

    # build generator
    generator = Model([input_gen, cond_in], y, name="generator")
    generator.summary()

    return encoder, generator, discriminator, classifier


#######################################################################################################################
# custom model and train step

class CVAEGAN(keras.Model):

    def __init__(self, encoder, generator, discriminator, classifier, latent_dim, n_classes, batchsize):
        super(CVAEGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.encoder = encoder
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.BATCH_SIZE = batchsize

    def compile(self, e_optimizer, g_optimizer, d_optimizer, c_optimizer,
                classifier_loss,
                discriminator_loss,
                kl_loss,
                generator_loss,
                mean_fm_loss):
        if DEBUG:
            super(CVAEGAN, self).compile(run_eagerly=True)
        else:
            super(CVAEGAN, self).compile(run_eagerly=False)

        self.e_optimizer = e_optimizer
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.c_optimizer = c_optimizer

        self.classifier_loss = classifier_loss
        self.discriminator_loss = discriminator_loss
        self.kl_loss = kl_loss
        self.generator_loss = generator_loss
        self.mean_fm_loss = mean_fm_loss

        self.e_loss_metric = keras.metrics.Mean(name="enc_loss")
        self.g_loss_metric = keras.metrics.Mean(name="gen_loss")
        self.d_loss_metric = keras.metrics.Mean(name="disc_loss")
        self.c_loss_metric = keras.metrics.Mean(name="class_loss")
        # self.val_loss_metric = keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):

        # unpack data
        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, \
                tf.GradientTape() as class_tape:

            # feed image to encoder and get mu, log_var, z
            mu, log_var, z = self.encoder([real_images, one_hot_labels], training=True)

            # generate an image using the z from the encoder
            x_f = self.generator([z, one_hot_labels], training=True)

            # generate an image using random sampled z and one-hot vectors
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_one_hot_vectors = tf.one_hot(tf.random.categorical(tf.eye(10), self.BATCH_SIZE)[0, :], self.n_classes)
            x_p = self.generator([random_latent_vectors, random_one_hot_vectors], training=True)

            # get discriminator outputs
            disc_r, disc_r_features = self.discriminator(real_images, training=True)
            disc_f, disc_f_features = self.discriminator(x_f, training=True)
            disc_p, disc_p_features = self.discriminator(x_p, training=True)

            # get classifier outputs
            class_r, class_r_features = self.classifier(real_images, training=True)
            class_f, class_f_features = self.classifier(x_f, training=True)
            class_p, class_p_features = self.classifier(x_p, training=True)

            L_C = self.classifier_loss(one_hot_labels, class_r)
            L_KL = self.kl_loss(mu, log_var)
            L_D = self.discriminator_loss(disc_r, disc_f, disc_p)
            L_GD = self.mean_fm_loss(disc_r_features, disc_p_features)
            L_GC = self.mean_fm_loss(class_r_features, class_p_features)
            L_G = self.generator_loss(real_images, x_f,
                                      disc_r_features, disc_f_features,
                                      class_r_features, class_f_features)

            # loss weights from the paper
            lambda1 = 3
            lambda2 = 1
            lambda3 = 1e-3
            lambda4 = 1e-3

            encoder_loss = lambda1 * L_KL + lambda2 * L_G
            gen_loss = lambda2 * L_G + lambda3 * L_GD + lambda4 * L_GC
            disc_loss = L_D
            class_loss = L_C

        gradients_of_encoder = enc_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_classifier = class_tape.gradient(class_loss, self.classifier.trainable_variables)

        self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.c_optimizer.apply_gradients(zip(gradients_of_classifier, self.classifier.trainable_variables))

        # Update metrics
        self.e_loss_metric.update_state(encoder_loss)
        self.g_loss_metric.update_state(gen_loss)
        self.d_loss_metric.update_state(disc_loss)
        self.c_loss_metric.update_state(class_loss)
        return {
            "enc_loss": self.e_loss_metric.result(),
            "gen_loss": self.g_loss_metric.result(),
            "disc_loss": self.d_loss_metric.result(),
            "class_loss:": self.c_loss_metric.result()
        }

    # def test_step(self, data):
    #
    #     # unpack data
    #     real_images, one_hot_labels = data
    #     batch_size = tf.shape(real_images)[0]
    #
    #     # feed image to encoder and get mu, log_var, z
    #     mu, log_var, z = self.encoder([real_images, one_hot_labels], training=False)
    #
    #     # generate an image using the z from the encoder
    #     x_f = self.generator([z, one_hot_labels], training=False)
    #
    #     # generate an image using random sampled z and one-hot vectors
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_one_hot_vectors = tf.one_hot(tf.random.categorical(tf.eye(10), BATCH_SIZE)[0, :], n_classes)
    #     x_p = self.generator([random_latent_vectors, random_one_hot_vectors], training=True)
    #
    #     # get discriminator outputs
    #     _, disc_r_features = self.discriminator(real_images, training=True)
    #     _, disc_f_features = self.discriminator(x_f, training=True)
    #     _, disc_p_features = self.discriminator(x_p, training=True)
    #
    #     # get classifier outputs
    #     _, class_r_features = self.classifier(real_images, training=True)
    #     _, class_f_features = self.classifier(x_f, training=True)
    #     _, class_p_features = self.classifier(x_p, training=True)
    #
    #     L_GD = self.mean_fm_loss(disc_r_features, disc_p_features)
    #     L_GC = self.mean_fm_loss(class_r_features, class_p_features)
    #     L_G = self.generator_loss(real_images, x_f,
    #                               disc_r_features, disc_f_features,
    #                               class_r_features, class_f_features)
    #
    #     # loss weights from the paper
    #     lambda2 = 1
    #     lambda3 = 1e-3
    #     lambda4 = 1e-3
    #
    #     gen_loss = lambda2 * L_G + lambda3 * L_GD + lambda4 * L_GC
    #
    #     self.val_loss_metric.update_state(gen_loss)
    #
    #     return {
    #         "loss": self.val_loss_metric.result(),
    #     }
