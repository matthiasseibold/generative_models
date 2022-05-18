import os
import math
import librosa
import numpy as np

number_mels = 64
hop_length = 256  # 86  # 173  # 512
window_length = 16380  # 11025  # 22050   # 65520

# the local path to the dataset
dataset_dir = "F:/datasets/generative_models/datasets/THADataset4/"

# the path to save the spectrograms
save_path = "../spectrograms/THA4-64/"

if not os.path.exists("../spectrograms/"):
    os.mkdir("../spectrograms/")

if not os.path.exists(save_path):
    os.mkdir(save_path)

classes = os.listdir(dataset_dir)
for cl in classes:
    files = os.listdir(dataset_dir + "/" + cl)

    if not os.path.exists(save_path + "/" + cl):
        os.mkdir(save_path + "/" + cl)

    for current_file in files:

        # print progress
        print("Processing... " + " class: " + cl + " - file: " + current_file)

        # read sample and convert to mono
        y, sr = librosa.load(dataset_dir + "/" + cl + "/" + current_file, sr=None, mono=True)

        i = 1
        split_count = 0

        while i <= (len(y) - window_length):

            # window
            window = y[i:i + window_length]

            # offset
            i = i + math.floor(window_length)  # no overlap

            # compute spectrogram
            S1 = librosa.feature.melspectrogram(window, sr=sr, n_mels=number_mels, hop_length=hop_length)
            # convert to power spectrogram
            S1_log = librosa.power_to_db(S1, ref=np.max)

            # readjust dimensions
            # S1_log = S1_log[:, :-1]

            # save array to file
            file_name = os.path.splitext(current_file)[0]
            # save_file = file_name + "_" + str(split_count)
            np.save(save_path + "/" + cl + "/" + file_name + "_" + str(split_count) + ".npy", S1_log)

            split_count += 1


