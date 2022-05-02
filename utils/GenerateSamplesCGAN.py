import numpy as np
from models.CGAN import *
import tensorflow as tf

# hyperparams
img_width = 128
img_height = 128
channels = 1
n_latent = 128
n_classes = 10

# build model
generator, discriminator = build_model(img_width, img_height, channels, n_latent, n_classes)

# reload model
generator.load_weights('../checkpoints/cgan_sd_generator_keras_model_nlatent_128_batchsize_32.hdf5')

# generate samples
random_latent_vectors = tf.random.normal(shape=(n_classes, n_latent))
# eye_classes = tf.eye(n_classes, n_classes)
eye_classes = np.zeros((n_classes, n_classes))
for i in range(10):
    eye_classes[i, 9] = 1
random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
generated_images = generator.predict(random_vector_labels)
for i in range(n_classes):
    np.save("../generated/cgan/cgan_sd_npy/generated_spectrogram_class_"
            + str(i) + ".npy", generated_images[i, :, :, 0])
