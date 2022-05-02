import os
import numpy as np
from tensorflow.keras.utils import to_categorical

class SpectroDataLoader:
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.num_classes = len(self.classes)

    def load_data(self, one_hot=True):

        labels = []
        data = []
        root_dir = self.root_dir

        classes = os.listdir(root_dir)
        for c in classes:
            files = os.listdir(str(root_dir) + "/" + c)
            for g in files:
                # print("Loading file: " + str(g))
                temp = np.load(str(root_dir) + "/" + c + "/" + g)
                labels.append(classes.index(c))
                data.append(np.expand_dims(temp, axis=2))

        if one_hot:
            return np.asarray(data), to_categorical(np.asarray(labels))
        else:
            return np.asarray(data), np.asarray(labels)

    @staticmethod
    def normalize_data(input_data, mean, std):
        normalized = (input_data - mean) / std
        return normalized


# for debugging
if __name__ == '__main__':
    pass