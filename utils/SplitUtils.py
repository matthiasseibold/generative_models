"""Utilities for dataset splitting:

split_dataset():
    Splits the dataset into Train/Validation and Test data
    install tqdm to get a progress bar while copying files
        pip install split-folders tqdm

split_cross_val():
    Splits the dataset into K folds of train and test set to
    perform k-fold-cross validation

"""

import sys
import splitfolders as sf

from sklearn.model_selection import StratifiedKFold
from utils.splitall import *
import shutil
import os


def split_dataset():

    input_path = '../datasets/THADataset2-kfold2/fold4'
    output_path = '../datasets/THADataset2-kfold2/fold4'

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    sf.ratio(input_path, output=output_path, seed=7, ratio=(.8, .2))

    # Split val/test with a fixed number of items e.g. 100 for each set.
    # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
    # sf.fixed('input_folder', output="output", seed=1337, fixed=(100, 100), oversample=False)  # default values


def get_file_list(path):

    paths = []
    targets = []
    classes = os.listdir(path)
    class_num = 0
    for c in classes:
        files = os.listdir(str(path) + "/" + c)
        for f in files:
            paths.append(str(path) + "/" + c + "/" + f)
            targets.append(class_num)
        class_num += 1
    return paths, targets


def split_cross_val():

    input_path = 'F:\generative_models\datasets/THADataset4/'
    output_path = 'F:\generative_models\datasets/THADataset4-kfold/'
    num_folds = 5

    file_list, targets = get_file_list(input_path)
    kFold = StratifiedKFold(n_splits=num_folds, shuffle=True)

    fold_no = 0
    for _, test in kFold.split(file_list, targets):

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if not os.path.exists(output_path + "fold" + str(fold_no)):
            os.mkdir(output_path + "fold" + str(fold_no))

        print("Processing fold #" + str(fold_no))

        test_samples = [file_list[i] for i in test]
        for file in test_samples:
            parts = splitall(file)

            if not os.path.exists(output_path + "fold" + str(fold_no) + "/" + parts[-2] + "/"):
                os.mkdir(output_path + "fold" + str(fold_no) + "/" + parts[-2] + "/")

            shutil.copyfile(file, output_path + "fold" + str(fold_no) + "/" + parts[-2] + "/" + parts[-1])

        fold_no += 1


if __name__ == "__main__":
    # split_dataset()
    split_cross_val()
