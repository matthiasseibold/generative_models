import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

class SpectroDataLoader:

    def __init__(self, root_dir=""):
        self.root_dir = root_dir

        self.classes = os.listdir(root_dir)
        self.num_classes = len(self.classes)

    def load_data(self, labels_file=""):

        labels = []
        data = []
        classes = []
        filenames = []

        if not labels_file == "":
            csv_df = pd.read_csv(labels_file)
            filenames = csv_df['filenames'].to_list()
            for i in range(10):
                classes.append(csv_df['classes_one_hot' + str(i+1)].to_numpy())
            classes = np.transpose(np.asarray(classes))

        files = os.listdir(str(self.root_dir))
        for filestr in files:

            print("Loading file: " + str(filestr))

            filestr_raw = os.path.splitext(filestr)[0]
            label = classes[filenames.index(filestr_raw), :]
            labels.append(np.float32(label))

            temp = np.load(self.root_dir + "/" + filestr)
            data.append(np.expand_dims(temp, axis=2))

        return np.asarray(data), np.asarray(labels)



    @staticmethod
    def normalize_data(input_data, mean, std):
        normalized = (input_data - mean) / std
        return normalized


# for debugging
if __name__ == '__main__':
    pass
