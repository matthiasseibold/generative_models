""" Script to train a CVAE-GAN on spectrograms"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.CVAEGAN_softplus import *
from utils.SpectroDataLoader import SpectroDataLoader


#######################################################################################################################
# data
dataset_path = "./spectrograms/THA/"
np_loader = SpectroDataLoader(root_dir=dataset_path)
data_train, labels_train = np_loader.load_data(one_hot=True)

# dataset statistics
stats = np.load('./dataset_statistics/statistics_THA.npz')
mean = stats['arr_0']
std = stats['arr_1']
max = stats['arr_2']

# normalize data
data_train_normalized = np_loader.normalize_data(data_train, mean=mean, std=std)
# map the data to range [0 2]
data_train_normalized = (data_train_normalized / max) + 1
input_shape = data_train_normalized[0, :, :].shape

# dimensions
img_width = data_train_normalized.shape[1]
img_height = data_train_normalized.shape[2]
channels = data_train_normalized.shape[3]

# model params
n_latent = 128
n_classes = labels_train.shape[1]

BUFFER_SIZE = 3000
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((data_train_normalized, labels_train)).\
    shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

#######################################################################################################################
# losses + optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
mse = tf.keras.losses.MeanSquaredError()


def classifier_loss(y_true, y_pred):
    return categorical_cross_entropy(y_true, y_pred)


def discriminator_loss(real_output, fake_output, sampled_fake_output):
    ones_real = tf.ones_like(real_output)
    zeros_fake = tf.zeros_like(fake_output)
    zeros_sampled_fake = tf.zeros_like(sampled_fake_output)

    # add noise to the labels
    # ones_real += 0.05 * tf.random.uniform(tf.shape(real_output))
    # zeros_fake += 0.05 * tf.random.uniform(tf.shape(fake_output))
    # zeros_sampled_fake += 0.05 * tf.random.uniform(tf.shape(sampled_fake_output))

    real_loss = cross_entropy(ones_real, real_output)
    fake_loss = cross_entropy(zeros_fake, fake_output)
    sampled_fake_loss = cross_entropy(zeros_sampled_fake, sampled_fake_output)
    return real_loss + fake_loss + sampled_fake_loss


def kl_loss(mu, log_var):
    return 0.5 * K.mean(K.square(mu) + K.exp(log_var) - log_var - 1)


def generator_loss(x_original, x_generated, features_original_disc, features_generated_disc,
                   features_original_class, features_generated_class):

    # reconstruction_loss = K.mean(mse(Flatten()(x_original), Flatten()(x_generated)) * img_width * img_height)
    ssim_loss = K.mean((1 - tf.image.ssim(K.abs(x_original), K.abs(x_generated), 15.0)) *
                       img_width * img_height)
    # combined_loss = ssim_loss + reconstruction_loss
    # pairwise_fm_loss_disc = mse(features_original_disc, features_generated_disc)
    # pairwise_fm_loss_class = mse(features_original_class, features_generated_class)
    return ssim_loss


def mean_feature_matching_loss(features1, features2):
    return mse(features1, features2)


# callback that periodically saves generated images
class CVAEGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=n_classes, latent_dimension=n_latent):
        self.num_img = num_img
        self.latent_dim = latent_dimension

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        eye_classes = tf.eye(self.num_img, self.num_img)
        generated_images = self.model.generator([random_latent_vectors, eye_classes])
        generated_images -= 1
        generated_images *= 127.5
        generated_images += 127.5
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated/cvaegan/cvaegan_tha_ssim/generated_img_%03d_%d.png" % (epoch, i))


# build model
encoder, generator, discriminator, classifier = build_model(img_width, img_height, channels, n_latent, n_classes)

# train the model
epochs = 3000
gan = CVAEGAN(encoder=encoder, generator=generator, discriminator=discriminator,
              classifier=classifier, latent_dim=n_latent, n_classes=n_classes, batchsize=BATCH_SIZE)
gan.compile(
    e_optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    g_optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    d_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    c_optimizer=keras.optimizers.Adam(learning_rate=1e-4),

    classifier_loss=classifier_loss,
    discriminator_loss=discriminator_loss,
    kl_loss=kl_loss,
    generator_loss=generator_loss,
    mean_fm_loss=mean_feature_matching_loss,
)

EarlyStop = EarlyStopping(
    monitor='gen_loss',
    verbose=True,
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True)

LearnRate = ReduceLROnPlateau(
    monitor='gen_loss',
    verbose=True,
    patience=15,
    min_delta=0.0001)

gan.fit(train_dataset,
        epochs=epochs,
        # validation_data=test_dataset,
        callbacks=[CVAEGANMonitor(num_img=n_classes, latent_dimension=n_latent), EarlyStop, LearnRate])

# encoder.save('./checkpoints/cvaegan_encoder_keras_model_nlatent_{}_batchsize_{}.hdf5'.format(n_latent, BATCH_SIZE))
generator.save('./checkpoints/cvaegan_tha_ssim_generator_keras_model_nlatent_{}_batchsize_{}.hdf5'.format(n_latent,
                                                                                                    BATCH_SIZE))

# generate samples with weights from the best epoch for each class
random_latent_vectors = tf.random.normal(shape=(n_classes, n_latent))
eye_classes = tf.eye(n_classes, n_classes)
generated_images = generator([random_latent_vectors, eye_classes])
generated_images = (generated_images.numpy() - 1) * max
for i in range(n_classes):
    np.save("./generated/cvaegan/cvaegan_tha_ssim_npy/generated_spectrogram_class_" + str(i) + ".npy",
            generated_images[i, :, :, 0])
