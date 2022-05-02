import numpy as np
from models.CVAEGAN import *
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt

# hyperparams
img_width = 128
img_height = 128
channels = 1
n_latent = 128
n_classes = 6
n_augment = np.repeat(2500, 6)
n_samples = np.asarray([446, 49, 795, 375, 101, 761])
n_samples = (n_augment - n_samples).tolist()

# load dataset statistics
stats = np.load('../dataset_statistics/statistics_THA2.npz')
mean = stats['arr_0']
std = stats['arr_1']

classes = ['Coagulation', 'Corkscrew', 'Insertion', 'Reaming', 'Sawing', 'Suction']

# build model
_, generator, _, _ = build_model(img_width, img_height, channels, n_latent, n_classes)

# reload model
generator.load_weights('../checkpoints/cvaegan_tha2_generator_keras_model_nlatent_128_batchsize_32.hdf5')

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
        spec *= std
        spec += mean
        librosa.display.specshow(spec)
        plt.show()
        # np.save("../spectrograms/THA2-split2/train-cvaegan/" + classes[j] + "/generated_spectrogram_class_"
        #         + str(i) + ".npy", spec)
