import os
import librosa
import soundfile

# init
count = 0
min_length = 100000
max_length = 0
length_sum = 0
min_file = 0
max_file = 0
root_dir = "C:\Git\generative_models\datasets\THADataset"
save_path = "C:\Git\generative_models\datasets\THADataset-cut"

classes = os.listdir(root_dir)
for cl in classes:
    files = os.listdir(root_dir + "/" + cl)
    for file in files:

        print("Processing file: " + file)

        # read audio sample
        y, sr = librosa.load(root_dir + "/" + cl + "/" + file)

        # cut or pad with zeros
        y_new = y[1:len(y)-44100]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + "/" + cl):
            os.makedirs(save_path + "/" + cl)
        soundfile.write(save_path + "/" + cl + "/" + file, y_new, samplerate=sr)

        count += 1

    print("Processed " + str(count) + " files")
