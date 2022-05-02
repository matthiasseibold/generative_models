import librosa.display
import os
import numpy as np
from models.CWGANGP import *
import tensorflow as tf
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree

root_dir = "/recostructed_wav/cwgan-gp"

# hyperparams
img_width = 64
img_height = 64
channels = 1
n_latent = 128
n_classes = 6

# load dataset statistics
stats = np.load('../dataset_statistics/statistics_THA4.npz')
mean = stats['arr_0']
std = stats['arr_1']

classes = ['Adjustment', 'Coagulation', 'Insertion', 'Reaming', 'Sawing', 'Suction']

# build model
# generator, discriminator = build_model(img_width, img_height, channels, n_latent, n_classes)
generator = get_generator_model(latent_dim=n_latent, num_classes=n_classes)

# reload model
generator.load_weights('../checkpoints/cwgan-gp-4-2/cwgan_gp_generator_epoch_580.hdf5')

random_latent_vectors = tf.random.normal(shape=(n_classes, n_latent))
eye_classes = np.eye(n_classes)

random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
generated_images = generator.predict(random_vector_labels)

for i in range(n_classes):
    spec = generated_images[i, :, :, 0]
    # de-normalize
    spec *= std
    spec += mean
    librosa.display.specshow(spec)
    plt.show()
    np.save(root_dir + "/" + classes[i], spec)
