import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow import keras as keras
from tensorflow.keras import backend as K


# reparametrization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def VAE(img_width, img_height, channels, latent_dim, optimizer):

    # encoder
    input_img = Input(shape=(img_width, img_height, channels))
    x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_img)
    x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2 * latent_dim)(x)
    x_split = tf.split(x, num_or_size_splits=2, axis=1, name='split')
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')(x_split)

    encoder = Model(input_img, [x_split[0], x_split[1], z], name='encoder')
    encoder.summary()

    # decoder
    input_gen = Input(shape=(latent_dim,))
    y = Dense(512, activation='relu')(input_gen)
    y = Dropout(0.2)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(units=16*16*32, activation='relu')(y)
    y = keras.layers.Reshape((16, 16, 32))(y)
    y = keras.layers.Conv2DTranspose(
        filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(y)
    y = keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(y)

    decoder = Model(input_gen, y, name='decoder')
    decoder.summary()

    # whole model
    output = decoder(encoder(input_img)[2])
    vae = Model(input_img, output, name='vae')
    vae.summary()

    # compile model
    reconstruction_loss = keras.losses.mse(keras.layers.Flatten()(input_img), keras.layers.Flatten()(output))
    # reconstruction_loss *= 28*28
    reconstruction_loss *= img_width * img_width

    kl_loss = 1 + x_split[1] - K.square(x_split[0]) - K.exp(x_split[1])
    kl_loss = K.sum(kl_loss, axis=1)
    kl_loss = -0.5 * kl_loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer)

    vae.summary()

    return vae, encoder, decoder