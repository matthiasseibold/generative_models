import os
import numpy as np
import librosa
import soundfile

# init
count = 0
min_length = 100000
max_length = 0
length_sum = 0
min_file = 0
max_file = 0
root_dir = "C:/Users/seim/Downloads/UrbanSound8K/audio"
save_path = "../datasets/urbansound-8k-cut"

classes = os.listdir(root_dir)
for cl in classes:
    files = os.listdir(root_dir + "/" + cl)
    for file in files:

        print("Processing file: " + file)

        # read audio sample
        y, sr = librosa.load(root_dir + "/" + cl + "/" + file)

        # cut or pad with zeros
        y_new = np.zeros(65520,)
        if len(y) <= len(y_new):
            y_new[:len(y)] = y
        else:
            y_new = y[:len(y_new)]

        # write audio sample with uniform length (minimal sample length = 24782 samples)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + "/" + cl):
            os.makedirs(save_path + "/" + cl)
        soundfile.write(save_path + "/" + cl + "/" + file, y_new, samplerate=sr)

        if len(y) < min_length:
            min_length = len(y)
            min_file = file
        if len(y) > max_length:
            max_length = len(y)
            max_file = file

        length_sum += len(y)
        count += 1

    print("Processed " + str(count) + " files")
    print("Minimal sample length: " + str(min_length) + " - File Name: " + min_file)
    print("Maximal sample length: " + str(max_length) + " - File Name: " + max_file)
    print("Mean sample length: " + str(length_sum / count))

    # Urbansound-8k statistics
    # Processed 8732 files
    # Minimal sample length: 1103 - File Name: 87275-1-1-0.wav
    # Maximal sample length: 89009 - File Name: 36429-2-0-13.wav
    # Mean sample length: 79545.9054054054
