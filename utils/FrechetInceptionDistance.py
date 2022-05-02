import os
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from tensorflow.keras.models import Model
from scipy.linalg import sqrtm
from models.resnet import ResNet18
from models.CWGANGP import *
from utils.SpectroDataLoader import *
import tensorflow.keras.backend as K

modeltype = "cwgan-gp-4"

def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


if __name__ == '__main__':
	# data
	dataset_path = "../spectrograms/THA4-64/"
	np_loader = SpectroDataLoader(root_dir=dataset_path)
	data_train, labels_train = np_loader.load_data(one_hot=True)

	# dataset statistics
	stats = np.load('../dataset_statistics/statistics_THA4.npz')
	mean = stats['arr_0']
	std = stats['arr_1']

	# # normalize data
	data_train_normalized = np_loader.normalize_data(data_train, mean=mean, std=std)
	input_shape = data_train_normalized[0, :, :].shape

	# dimensions
	img_width = data_train_normalized.shape[1]
	img_height = data_train_normalized.shape[2]
	channels = data_train_normalized.shape[3]

	# model params
	n_latent = 128
	n_classes = labels_train.shape[1]

	files = sorted(os.listdir("../checkpoints/" + modeltype + "/"), key=len)
	# files = ["cwgan_gp_generator_epoch_1000.hdf5", "cwgan_gp_generator_epoch_1001.hdf5", "cwgan_gp_generator_epoch_1002.hdf5",
	# 		 "cwgan_gp_generator_epoch_1200.hdf5", "cwgan_gp_generator_epoch_1201.hdf5", "cwgan_gp_generator_epoch_1202.hdf5"]

	min_fid = 10000
	min_fid_file = ""

	for file in files:

		# model
		original_model = ResNet18(input_shape=input_shape, classes=np_loader.num_classes)
		original_model.load_weights("../checkpoints/inception_score_model.hdf5")
		# original_model.summary()
		model = Model(original_model.input, original_model.layers[-2].output)

		generator = get_generator_model(latent_dim=n_latent, num_classes=n_classes)
		# generator.load_weights("../checkpoints/cwgan_gp_tha2_generator_keras_model_nlatent_128_batchsize_32.hdf5")
		# generator.load_weights("../checkpoints/cgan_tha2_generator_keras_model_nlatent_128_batchsize_32.hdf5")

		generator.load_weights("../checkpoints/" + modeltype + "/" + file)
		print("File: " + file + "  ----------  ", end="")

		n_split = 10
		n_part = np.floor(data_train_normalized.shape[0] / n_split)
		generated_images = np.empty(shape=[0, 64, 64, 1])
		for i in range(n_split):
			ix_start, ix_end = i * n_part, i * n_part + n_part
			labels_temp = labels_train[int(ix_start):int(ix_end)]
			random_latent_vectors = np.random.normal(size=(labels_temp.shape[0], n_latent))
			random_vector_labels = np.append(random_latent_vectors, labels_temp, axis=1)
			generated = generator(random_vector_labels)
			generated_images = np.append(generated_images, generated, axis=0)

		# calculate FID
		fid = calculate_fid(model, data_train_normalized, generated_images)
		if fid < min_fid:
			min_fid = fid
			min_fid_file = file
		print("Frechet Inception Distance: " + str(fid))

		K.clear_session()

	print("Min FID: " + str(min_fid) + " --- File: " + min_fid_file)