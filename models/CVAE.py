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


def CVAE(img_width, img_height, channels, n_classes, latent_dim, optimizer):
    input_img = Input(shape=(img_width, img_height, channels))
    input_cond = Input(shape=(n_classes,))
    x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    mu = Dense(latent_dim, activation='linear')(x)
    log_var = Dense(latent_dim, activation='linear')(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    encoder = Model([input_img, input_cond], [mu, log_var, z, input_cond], name='encoder')
    encoder.summary()

    # decoder
    input_gen = Input(shape=(latent_dim,))
    cond_in = Input(shape=(n_classes,))
    dec_input = Concatenate()([input_gen, cond_in])
    y = Dense(512, activation='relu')(dec_input)
    y = keras.layers.BatchNormalization()(y)
    y = Dense(512, activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = Dense(units=16*16*32, activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
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
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(y)

    decoder = Model([input_gen, cond_in], y, name='decoder')
    decoder.summary()

    # whole model
    output = decoder(encoder([[input_img, input_cond]])[2:])
    cvae = Model([input_img, input_cond], output, name='cvae')

    # loss
    reconstruction_loss = keras.losses.mse(keras.layers.Flatten()(input_img),
                                           keras.layers.Flatten()(output)) * 128 ** 2
    kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1)
    cvae_loss = reconstruction_loss + kl_loss

    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer=optimizer, run_eagerly=False)
    cvae.summary()

    return cvae, encoder, decoder
