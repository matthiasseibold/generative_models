"""Tested with tensorflow 2.2.0"""

import numpy as np
import math
import tensorflow as tf


class SpectroDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, root_dir, n_classes, mean=0, std=1, batch_size=32, shuffle=True):
        """"Initialization"""
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.root_dir = root_dir

        # statistics
        self.mean = mean
        self.std = std

        # load first file and check the shape (assuming that all files in the dataset have the same shape)
        temp = np.load(self.root_dir + list_IDs[0] + '.npy')
        self.image_shape = np.expand_dims(temp, axis=2).shape

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return math.ceil(len(self.list_IDs) / self.batch_size)  # from tf doc

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        data, labels = self.__data_generation(list_IDs_temp)

        return data, labels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        # initialize as empty lists
        x = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            temp = np.load(self.root_dir + ID + '.npy')
            temp = (temp - self.mean) / self.std  # normalize the data using precomputed dataset statistics
            x.append(np.expand_dims(temp, axis=2))

            # Store class
            y.append(self.labels[ID])

        return np.asarray(x), tf.keras.utils.to_categorical(np.asarray(y), num_classes=self.n_classes)
