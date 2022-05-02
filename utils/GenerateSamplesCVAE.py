import numpy as np
import tensorflow as tf
from models.CVAE import CVAE

# hyperparams
img_width = 128
img_height = 128
channels = 1
n_latent = 128
n_classes = 10

# model
_, _, decoder = CVAE(img_width, img_height, channels, n_classes, n_latent, optimizer='adam')

# reload model
decoder.load_weights('../checkpoints/cvae_sd_decoder_keras_model_nlatent_128_batchsize_32.hdf5')

# generate samples
random_latent_vectors = tf.random.normal(shape=(n_classes, n_latent))
# eye_classes = tf.eye(n_classes, n_classes)
eye_classes = np.zeros((n_classes, n_classes))
for i in range(10):
    eye_classes[i, 9] = 1
generated_images = decoder.predict([random_latent_vectors, eye_classes])
# generated_images = generated_images.numpy()
for i in range(n_classes):
    np.save("../generated/cvae/cvae_sd_npy/generated_spectrogram_class_"
            + str(i) + ".npy", generated_images[i, :, :, 0])

