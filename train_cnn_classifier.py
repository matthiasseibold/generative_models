import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.resnet import ResNet50

from models.resnet import ResNet18
from utils.SpectroDataLoader import *


dataset_path = "./spectrograms/THA2-split2"
np_loader_train = SpectroDataLoader(root_dir=dataset_path + "/train/")
np_loader_val = SpectroDataLoader(root_dir=dataset_path + "/val/")
np_loader_test = SpectroDataLoader(root_dir=dataset_path + "/test/")

# load data and labels as numpy array
data_train, labels_train = np_loader_train.load_data(one_hot=True)
data_val, labels_val = np_loader_val.load_data(one_hot=True)
data_test, labels_test = np_loader_test.load_data(one_hot=True)

# dataset statistics
stats = np.load('dataset_statistics/statistics_THA2.npz')
mean = stats['arr_0']
std = stats['arr_1']

# # normalize data
data_train_normalized = np_loader_train.normalize_data(data_train, mean=mean, std=std)
data_val_normalized = np_loader_val.normalize_data(data_val, mean=mean, std=std)
data_test_normalized = np_loader_test.normalize_data(data_test, mean=mean, std=std)
input_shape = data_test_normalized[0, :, :].shape

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

val_dataset = tf.data.Dataset.from_tensor_slices((data_val_normalized, labels_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((data_test_normalized, labels_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

# build model
model = ResNet18(input_shape=input_shape, classes=np_loader_train.num_classes)

# show the model summary
model.summary()

callbacks = [

    # early stopping
    EarlyStopping(monitor='val_acc',
                  patience=30,
                  restore_best_weights=True),

]

# train with fit method
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['acc'])

# fit with numpy array in RAM
history = model.fit(train_dataset,
                    batch_size=32,
                    epochs=500,
                    validation_data=val_dataset,
                    callbacks=callbacks)

# evaluate the best performing model on the test set
results = model.evaluate(test_dataset)
print('Final test accuracy: ' + str(results[1] * 100) + '%')

# print performance metrics
y_pred = model.predict(data_test_normalized)
print('Confusion Matrix')
cm = confusion_matrix(labels_test.argmax(axis=1), (y_pred > 0.5).argmax(axis=1))
print(cm)

print('Classification Report')
target_names = np_loader_train.classes
cr = classification_report(labels_test.argmax(axis=1), (y_pred > 0.5).argmax(axis=1), target_names=target_names)
print(cr)

print('Per class accuracies')
accuracies = cm.diagonal() / cm.sum(axis=1)
print(accuracies)

print('Mean per-class recall')
print(sum(accuracies) / n_classes)

model.save("./checkpoints/inception_score_model.hdf5")
