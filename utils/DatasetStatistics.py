import numpy as np
import os
from utils.SpectroDataLoader import SpectroDataLoader

# set up paths
dataset_path = "../spectrograms/THA4-64"


def compute_mean_std(path):
        mean = []
        std = []

        # dirs = [path + '/0',
        #         path + '/1',
        #         path + '/2',
        #         path + '/3',
        #         path + '/4',
        #         path + '/5',
        #         path + '/6',
        #         path + '/7',
        #         path + '/8',
        #         path + '/9']

        dirs = [path + '/Adjustment',
                path + '/Coagulation',
                path + '/Insertion',
                path + '/Reaming',
                path + '/Sawing',
                path + '/Suction']

        for root_dir in dirs:
                files = os.listdir(str(root_dir))
                for g in files:
                        print("Loading file: " + str(g))
                        temp = np.load(str(root_dir) + "/" + g)
                        # compute statistics
                        mean.append(np.mean(temp))
                        std.append(np.std(temp))

        return np.mean(mean), np.mean(std)


if __name__ == '__main__':
        mean, std = compute_mean_std(dataset_path)
        np_loader = SpectroDataLoader(root_dir=dataset_path)
        data, _ = np_loader.load_data(one_hot=True)
        data_train_normalized = np_loader.normalize_data(data, mean=mean, std=std)
        max = np.max(np.abs(data_train_normalized))
        print("Mean: " + str(mean))
        print("Std: " + str(std))
        print("Max: " + str(max))

        np.savez('../dataset_statistics/statistics_THA4-64', mean, std, max)
