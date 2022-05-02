"""Script for reconstructing WAV files from generated log-mel spectrograms using the Griffin-Lim algorithm"""

import os
import librosa.display
import numpy as np
import soundfile as sf

count = 0
root_path = "/recostructed_wav/cwgan-gp"
classes = ['Adjustment', 'Coagulation', 'Insertion', 'Reaming', 'Sawing', 'Suction']

files = os.listdir(root_path)
for file in files:

    arr = np.load(root_path + "/" + file)

    # de-normalization is crucial for generated samples
    # stats = np.load('../dataset_statistics/statistics_THA4.npz')
    # mean = stats['arr_0']
    # std = stats['arr_1']
    # # de-normalize
    # arr *= std
    # arr += mean
    # print("AFTER de-normalization")
    # print("MAX: " + str(np.max(arr)))
    # print("MIN: " + str(np.min(arr)))

    arr = librosa.db_to_power(arr, ref=1000)
    print(arr.shape)
    print(np.max(arr))

    y = librosa.feature.inverse.mel_to_audio(arr, hop_length=256)
    sf.write(root_path + "/" + classes[count] + "_reconstructed.wav", y, 22050)

    count += 1
