"""Works only with tf 1.1x because of tensorflow.contrib module"""

import librosa.display
import os
import numpy as np
from specAugment import spec_augment_tensorflow
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree


def run_spec_augment(fold, cl, root_dir, configuration):

        print("Processing class: " + cl)
        files = os.listdir(root_dir + fold + "/" + cl)

        file_count = 0
        for file in files:

            print("Processing file: " + file)

            spec_original = np.load(root_dir + fold + "/" + cl + "/" + file)
            spec_original = librosa.db_to_power(spec_original)

            spec_augmented = spec_augment_tensorflow.spec_augment(mel_spectrogram=spec_original, time_masking_para=30)
            spec_augmented = librosa.power_to_db(spec_augmented)

            # fig, ax = plt.subplots(1, 2)
            # plt.sca(ax[0])
            # librosa.display.specshow(librosa.power_to_db(spec_original))
            # plt.sca(ax[1])
            # librosa.display.specshow(spec_augmented)
            # plt.show()

            np.save(root_dir + fold + configuration + "/" + cl + "/generated_spectrogram_class_"
                    + str(file_count) + ".npy", spec_augmented)

            file_count += 1


if __name__ == '__main__':

    root_dir = "../spectrograms/THA4-kfold-64/"
    configuration = "-spec-augment"

    folds = ["fold3"]

    for fold in folds:

        print("--- Processing " + fold + " ---")

        # copy folder
        copy_tree(root_dir + fold, root_dir + fold + configuration)

        classes = os.listdir(root_dir + fold + configuration)

        for cl in classes:
            run_spec_augment(fold, cl, root_dir, configuration)
