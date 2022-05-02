# imports
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models.CWGANGP import *
import tikzplotlib

files_count = 0
for c in range(10):

    fig, ax = plt.subplots(2, 6)
    root = "C:\Git\generative_models\spectrograms\THA4-kfold-64/fold0"

    classes = os.listdir(root)
    count = 0

    for cl in classes:
        files = os.listdir(root + "/" + cl)
        S1_log = np.load(root + "/" + cl + "/" + files[files_count])
        plt.sca(ax[0, count])
        librosa.display.specshow(S1_log)
        count += 1

    files_count += 1

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

    # build model
    generator = get_generator_model(latent_dim=n_latent, num_classes=n_classes)

    # reload model
    generator.load_weights('../checkpoints/cwgan-gp-4/cwgan_gp_generator_epoch_240.hdf5')


    random_latent_vectors = tf.random.normal(shape=(6, n_latent))
    eye_classes = np.eye(6)
    random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
    generated_images = generator.predict(random_vector_labels)

    for i in range(6):
        spec = generated_images[i, :, :, 0]
        # de-normalize
        spec *= std
        spec += mean
        plt.sca(ax[1, i])
        librosa.display.specshow(spec)

    fig.set_size_inches(18.5, 5.5, forward=True)

    # save to tikz
    tikzplotlib.save("../figures/test.tex")
    plt.show()


