""" Script to train a simple Convolutional Conditional GAN (CGAN) on MNIST"""

from models.CGAN import *
from utils.SpectroDataLoader import *
import matplotlib.pyplot as plt

# load data
dataset_path = "./spectrograms/THA4-64/"
np_loader = SpectroDataLoader(root_dir=dataset_path)
data_train, labels_train = np_loader.load_data(one_hot=True)

# normalize data
stats = np.load('dataset_statistics/statistics_THA4-64.npz')
mean = stats['arr_0']
std = stats['arr_1']
data_train_normalized = np_loader.normalize_data(data_train, mean=mean, std=std)
input_shape = data_train_normalized[0, :, :].shape

# get dimensions
img_width = data_train_normalized.shape[1]
img_height = data_train_normalized.shape[2]
channels = data_train_normalized.shape[3]

# set model params
n_latent = 128
n_classes = labels_train.shape[1]
BUFFER_SIZE = len(data_train_normalized)
BATCH_SIZE = 64

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((data_train_normalized, labels_train)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

# build model
generator, discriminator = build_model(img_width, img_height, channels, n_latent, n_classes)


# callback that periodically saves generated images
class CGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=10, latent_dimension=n_latent):
        self.num_img = num_img
        self.latent_dim = latent_dimension

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        eye_classes = tf.eye(self.num_img, self.num_img)
        random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
        generated_images = self.model.generator(random_vector_labels)
        generated_images *= 127.5
        generated_images += 127.5
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated/cgan/cgan_tha2/generated_img_%03d_%d.png" % (epoch, i))


# callback that periodically saves model checkpoints
class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 10) == 0:
            self.model.generator.save("./checkpoints/cgan-2/cgan_generator_epoch_" + str(epoch) + ".hdf5")


epochs = 2001
cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=n_latent,
                          image_size=img_width, num_classes=n_classes)

cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

# train the model
history = cond_gan.fit(train_dataset, epochs=epochs, callbacks=[CGANMonitor(num_img=n_classes, latent_dimension=n_latent),
                                                      SaveModelCallback()])

# save weights of generator
generator.save('./checkpoints/cgan_tha2_generator_keras_model_nlatent_{}_batchsize_{}.hdf5'.format(n_latent, BATCH_SIZE))

# generate samples with the trained generator
random_latent_vectors = tf.random.normal(shape=(n_classes, n_latent))
eye_classes = tf.eye(n_classes, n_classes)
random_vector_labels = tf.concat([random_latent_vectors, eye_classes], axis=1)
generated_images = generator(random_vector_labels)
generated_images = generated_images.numpy()
for i in range(n_classes):
    np.save("./generated/cgan/cgan_tha2_npy/generated_spectrogram_class_" + str(i) + ".npy", generated_images[i, :, :, 0])

plt.plot(history.history['g_loss'])
plt.plot(history.history['d_loss'])
plt.title('CGAN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['generator_loss', 'discriminator_loss'])
plt.savefig('cgan.png', dpi=2000)
