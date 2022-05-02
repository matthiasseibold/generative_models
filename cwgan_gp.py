import keras.callbacks
from models.CWGANGP import *
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


# callback that periodically saves generated images
class CWGANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=n_classes, latent_dimension=n_latent):
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
            img.save("./generated/cwgan-gp/cwgan_gp_tha4_2/generated_img_%03d_%d.png" % (epoch, i))


# callback that periodically saves model checkpoints
class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 10) == 0:
            self.model.generator.save("./checkpoints/cwgan-gp-4-2/cwgan_gp_generator_epoch_" + str(epoch) + ".hdf5")


g_model = get_generator_model(latent_dim=n_latent, num_classes=n_classes)
g_model.summary()

d_model = get_discriminator_model(IMG_SHAPE=data_train_normalized.shape[1:4], num_classes=n_classes)
d_model.summary()

# Instantiate the optimizer for both networks (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)


# Define the loss functions for the discriminator, which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Set the number of epochs for trainining.
epochs = 751

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=n_latent,
    img_size=img_width,
    n_cl=n_classes,
    discriminator_extra_steps=5,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Start training the model.
history = wgan.fit(train_dataset, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[CWGANMonitor(), SaveModelCallback()])

# plot training history
plt.figure()
plt.plot(history.history['g_loss'])
plt.plot(history.history['d_loss'])
plt.title('CWGAN-GP model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['generator_loss', 'discriminator_loss'])
plt.savefig('cwgan_gp_2.png', dpi=2000)
plt.show()
