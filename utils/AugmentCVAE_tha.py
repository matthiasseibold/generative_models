import numpy as np
import tensorflow as tf
from models.CVAE import CVAE
import matplotlib.pyplot as plt
import librosa.display

# hyperparams
img_width = 128
img_height = 128
channels = 1
n_latent = 128
n_classes = 6
n_augment = np.repeat(300, n_classes)
n_samples = np.asarray([100, 10, 115, 50, 20, 170])
n_samples = (n_augment - n_samples).tolist()

# load dataset statistics
stats = np.load('../dataset_statistics/statistics_THA2.npz')
mean = stats['arr_0']
std = stats['arr_1']

classes = ['Coagulation', 'Corkscrew', 'Insertion', 'Reaming', 'Sawing', 'Suction']

# model
_, _, decoder = CVAE(img_width, img_height, channels, n_classes, n_latent, optimizer='adam')

# reload model
decoder.load_weights('../checkpoints/cvae_tha2_decoder_keras_model_nlatent_128_batchsize_32.hdf5')

# generate samples
folds = ["fold0", "fold1", "fold2", "fold3", "fold4"]
for fold in folds:
    print("--- " + fold + " ---")
    for j in range(n_classes):
        print("Processing class: " + str(j))
        random_latent_vectors = tf.random.normal(shape=(n_samples[j], n_latent))
        eye_classes = np.zeros((n_samples[j], n_classes))
        for i in range(n_samples[j]):
            eye_classes[i, j] = 1
        generated_images = decoder.predict([random_latent_vectors, eye_classes])
        for i in range(n_samples[j]):
            spec = generated_images[i, :, :, 0]
            # de-normalize
            spec *= std
            spec += mean
            librosa.display.specshow(spec)
            plt.show()
            # np.save("../spectrograms/THA2-kfold3/" + fold + "/train-cvae/" + classes[j] + "/generated_spectrogram_class_"
            #         + str(i) + ".npy", spec)