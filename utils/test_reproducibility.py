import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# GLOBAL SEED
tf.random.set_seed(3)
x_train = tf.random.normal((10,1), 1, 1, dtype=tf.float32)
y_train = tf.math.sin(x_train)
x_test = tf.random.normal((10,1), 2, 3, dtype=tf.float32)
y_test = tf.math.sin(x_test)

model = Sequential()
model.add(Dense(1200, input_shape=(1,), activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

loss="binary_crossentropy"
optimizer=tf.keras.optimizers.Adam(lr=0.01)
metrics=['mse']
epochs = 5
batch_size = 32
verbose = 1

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)
histpry = model.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, verbose = verbose)
predictions = model.predict(x_test)
print(predictions)