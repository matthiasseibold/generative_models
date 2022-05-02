import numpy as np
from models.CVAEGAN import *
import tensorflow as tf

# hyperparams
img_width = 128
img_height = 128
channels = 1
n_latent = 128
n_classes = 5
n_samples = [861, 471, 1098, 1338, 636]

# load dataset statistics
stats = np.load('../dataset_statistics/statistics_THA.npz')
mean = stats['arr_0']
std = stats['arr_1']
max = stats['arr_2']

classes = ['Coagulation', 'Hammering', 'Reaming', 'Sawing', 'Suction']

# build model
_, generator, _, _ = build_model(img_width, img_height, channels, n_latent, n_classes)

# reload model
generator.load_weights('../checkpoints/cvaegan_tha_ssim_generator_keras_model_nlatent_128_batchsize_32.hdf5')

# generate samples
for j in range(n_classes):
    print("Processing class: " + str(j))
    random_latent_vectors = tf.random.normal(shape=(n_samples[j], n_latent))
    eye_classes = np.zeros((n_samples[j], n_classes))
    for i in range(n_samples[j]):
        eye_classes[i, j] = 1
    # random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
    generated_images = generator.predict([random_latent_vectors, eye_classes])
    for i in range(n_samples[j]):
        spec = generated_images[i, :, :, 0]
        # de-normalize
        spec -= 1
        spec *= max
        spec *= std
        spec += mean
        np.save("../spectrograms/THA-split/train-augmented-cvaegan-ssim/" + classes[j] + "/generated_spectrogram_class_"
                + str(i) + ".npy", spec)
