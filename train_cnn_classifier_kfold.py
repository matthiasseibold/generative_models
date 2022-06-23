import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.applications.resnet import ResNet50

from models.resnet import ResNet18
from utils.SpectroDataLoader import *

import tensorflow.keras.backend as K

dataset_path = "./spectrograms/THA4-kfold-64"

# augmentations = ""
augmentations = "-cwgan-gp-580"
# augmentations = "-cwgan-gp-580-double"
# augmentations = "-cgan-2"
# augmentations = "-cvae"
# augmentations = "-pitch-shift"
# augmentations = "-time-stretch"
# augmentations = "-noise"
# augmentations = "-spec-augment"

for count_runs in range(3):

    training_scheme = [["/fold1", "/fold2", "/fold3", "/fold4"],
                       ["/fold2", "/fold3", "/fold4", "/fold0"],
                       ["/fold3", "/fold4", "/fold0", "/fold1"],
                       ["/fold4", "/fold0", "/fold1", "/fold2"],
                       ["/fold0", "/fold1", "/fold2", "/fold3"]]

    test_scheme = ["/fold0", "/fold1", "/fold2", "/fold3", "/fold4"]

    cm_list = []
    cr_list = []
    accuracies_list = []
    macro_f1 = []

    for i in range(5):

        # load test data
        np_loader_test = SpectroDataLoader(root_dir=dataset_path + test_scheme[i])
        data_test, labels_test = np_loader_test.load_data(one_hot=True)

        print("Train folds: ", end="")
        print(training_scheme[i])
        print("Test folds: " + test_scheme[i])

        # load training data
        data_train = np.empty([0, 64, 64, 1])
        labels_train = np.empty([0, 6])
        for j in range(4):
            np_loader_train = SpectroDataLoader(root_dir=dataset_path + training_scheme[i][j] + augmentations + "/")
            data_train_temp, labels_train_temp = np_loader_train.load_data(one_hot=True)
            data_train = np.append(data_train, data_train_temp, axis=0)
            labels_train = np.append(labels_train, labels_train_temp, axis=0)

        # dataset statistics
        stats = np.load('dataset_statistics/statistics_THA4.npz')
        mean = stats['arr_0']
        std = stats['arr_1']

        # # normalize data
        data_train_normalized = np_loader_train.normalize_data(data_train, mean=mean, std=std)
        # data_val_normalized = np_loader_val.normalize_data(data_val, mean=mean, std=std)
        data_test_normalized = np_loader_test.normalize_data(data_test, mean=mean, std=std)
        input_shape = data_train_normalized[0, :, :].shape

        # dimensions
        img_width = data_train_normalized.shape[1]
        img_height = data_train_normalized.shape[2]
        channels = data_train_normalized.shape[3]

        # model params
        n_classes = labels_train.shape[1]
        BUFFER_SIZE = 3000
        BATCH_SIZE = 32

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices((data_train_normalized, labels_train)).\
            shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(BATCH_SIZE)

        # val_dataset = tf.data.Dataset.from_tensor_slices((data_val_normalized, labels_val))
        # val_dataset = val_dataset.batch(BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices((data_test_normalized, labels_test))
        test_dataset = test_dataset.batch(BATCH_SIZE)

        # build model
        model = ResNet18(input_shape=input_shape, classes=np_loader_train.num_classes)
        # model.summary()

        # callbacks = EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True)
        callbacks = []

        # train with fit method
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        history = model.fit(train_dataset,
                            batch_size=32,
                            epochs=20,
                            # validation_data=val_dataset,
                            callbacks=callbacks)

        # evaluate the best performing model on the test set
        results = model.evaluate(test_dataset)
        print('Final test accuracy: ' + str(results[1] * 100) + '%')

        # print performance metrics
        y_pred = model.predict(data_test_normalized)
        print('Confusion Matrix')
        cm = confusion_matrix(labels_test.argmax(axis=1), (y_pred > 0.5).argmax(axis=1))
        print(cm)
        cm_list.append(cm)

        print('Classification Report')
        target_names = np_loader_train.classes
        cr = classification_report(labels_test.argmax(axis=1), (y_pred > 0.5).argmax(axis=1), target_names=target_names)
        print(cr)
        cr_list.append(cr)

        print('Macro F1-score')
        F1_score = f1_score(labels_test.argmax(axis=1), (y_pred > 0.5).argmax(axis=1), average='macro')
        print(F1_score)
        macro_f1.append(F1_score)

        K.clear_session()

    # print averaged statistics over k-fold cross val
    print('Results of K-Fold Cross Validation')
    print('Confusion Matrix (Mean)')
    print(np.mean(cm_list, axis=0))
    print('Confusion Matrix (Std)')
    print(np.std(cm_list, axis=0))

    print('Macro F1-score (Mean)')
    print(np.mean(macro_f1, axis=0))
    print('Macro F1-score (Std)')
    print(np.std(macro_f1, axis=0))
