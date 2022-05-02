""" Script to train a simple Convolutional Conditional GAN (CGAN) on MNIST"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.CVAE import *
from utils.SpectroDataLoader import *

#######################################################################################################################
# data
dataset_path = "./spectrograms/THA2/"
np_loader = SpectroDataLoader(root_dir=dataset_path)
data_train, labels_train = np_loader.load_data(one_hot=True)

# dataset statistics
stats = np.load('dataset_statistics/statistics_THA2.npz')
mean = stats['arr_0']
std = stats['arr_1']

# # normalize data
data_train_normalized = np_loader.normalize_data(data_train, mean=mean, std=std)
input_shape = data_train_normalized[0, :, :].shape

# dimensions
img_width = data_train_normalized.shape[1]
img_height = data_train_normalized.shape[2]
channels = data_train_normalized.shape[3]

BUFFER_SIZE = 3000
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((data_train_normalized, labels_train)).\
    shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
# model

# model params
latent_dim = 128
n_classes = labels_train.shape[1]

# model
cvae, encoder, decoder = CVAE(img_width, img_height, channels, n_classes, latent_dim, optimizer='adam')

# callbacks
es_cb = EarlyStopping(
    monitor='loss',
    verbose=True,
    patience=25,
    min_delta=0.0001,
    restore_best_weights=True)

lr_cb = ReduceLROnPlateau(
    monitor='loss',
    verbose=True,
    patience=8,
    min_delta=0.0001)


class CVAEMonitor(keras.callbacks.Callback):
    # def __init__(self, num_img=10, latent_dimension=latent_dim):
    #     self.num_img = num_img
    #     self.latent_dim = latent_dimension

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(n_classes, latent_dim))
        eye_classes = tf.eye(n_classes, n_classes)
        generated_images = decoder([random_latent_vectors, eye_classes])
        generated_images *= 127.5
        generated_images += 127.5
        generated_images.numpy()
        for i in range(n_classes):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated/cvae/cvae_tha2/generated_img_%03d_%d.png" % (epoch, i))


cb = [es_cb, lr_cb, CVAEMonitor()]

# train it
epochs = 500
cvae_train = cvae.fit(x=[data_train_normalized, labels_train],
                      batch_size=32,
                      epochs=epochs,
                      shuffle=True,
                      # validation_data=([data_test_normalized, labels_test]),
                      callbacks=cb)

# save model
decoder.save('./checkpoints/cvae_tha2_decoder_keras_model_nlatent_{}_batchsize_{}.hdf5'.format(latent_dim, 32))
# cvae.save('./checkpoints/cvae_keras_model_nlatent_{}_batchsize_{}.hdf5'.format(latent_dim, batch_size))

# generate samples with weights from the best epoch for each class
random_latent_vectors = tf.random.normal(shape=(n_classes, latent_dim))
eye_classes = tf.eye(n_classes, n_classes)
generated_images = decoder.predict([random_latent_vectors, eye_classes])
# generated_images = generated_images.numpy()
for i in range(n_classes):
    np.save("./generated/cvae/cvae_tha2_npy/generated_spectrogram_class_" + str(i) + ".npy",
            generated_images[i, :, :, 0])

# plot training statistics
# plt.figure()
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
# ax1 = plt.subplot(gs[0])
# plt.plot(cvae_train.history['loss'], label='loss')
# plt.plot(cvae_train.history['val_loss'], label='val_loss')
# plt.legend()
# ax2 = plt.subplot(gs[1])
# plt.plot(cvae_train.history['lr'], label='lr')
# plt.legend()
# ax2.set_xlabel('epoch')
# ax2.set_yscale('log')
# plt.tight_layout()
# plt.show()

