from __future__ import print_function

import os
import glob
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
import tqdm
from PIL import Image
import matplotlib.gridspec as gridspec
from keras.layers import Dense
from keras.layers import Reshape, Lambda
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
#from GAN_Nets import get_disc_normal, get_gen_normal
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')
#from misc_layers import MinibatchDiscrimination, SubPixelUpscaling, CustomLRELU, bilinear2x
#from keras_contrib.layers.convolutional import SubPixelUpscaling
from keras.datasets import mnist
from keras.initializers import RandomNormal
from keras import metrics
from keras.utils import plot_model


from collections import deque
from scipy.ndimage import filters
from skimage import io
from skimage import transform

#### Starting Specs

batch_size = 64
input_shape = (64, 64, 3)
latent_dim = 128
intermediate_dim = 256
# epochs via minibatch
epochs = 1000
epsilon_std = 1.0
kernel_size = 5
optimizer = RMSprop(lr=0.0001)
load_images_at_start = False

def img_resize(img, rescale_size):
	h, w, c = img.shape
	img = img[20:h-20,:]
	# Smooth image before resize to avoid moire patterns
	scale = img.shape[0] / float(rescale_size)
	sigma = np.sqrt(scale) / 2.0
	img = filters.gaussian_filter(img, sigma=sigma)
	img = transform.resize(img, (rescale_size, rescale_size, 3), order=3, mode='reflect')
	img = (img*255).astype(np.uint8)
	return img

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

###############################
#### build encoder model
###############################

filter = 32
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
enc_input = inputs

encoder = Conv2D(filters = 64, kernel_size = (5,5), strides = (2,2), padding = "same", data_format = "channels_last")(enc_input)
encoder = BatchNormalization()(encoder)
encoder = Activation('relu')(encoder)
#encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

#encoder = Dropout(dropout_prob)(encoder)
encoder = Conv2D(filters = 128, kernel_size = (5,5), strides = (2,2), padding = "same", data_format = "channels_last")(encoder)
encoder = BatchNormalization()(encoder)
encoder = Activation('relu')(encoder)
#encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

#encoder = Dropout(dropout_prob)(encoder)
encoder = Conv2D(filters = 256, kernel_size = (5,5), strides = (2,2), padding = "same", data_format = "channels_last")(encoder)
encoder = BatchNormalization()(encoder)
encoder = Activation('relu')(encoder)
#encoder = MaxPooling2D(pool_size=(2, 2))(encoder)
sh = K.int_shape(encoder)
encoder = Flatten()(encoder)

#encoder = MinibatchEncoder(100,5)(encoder)
encoder = Dense(2048)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Activation('relu')(encoder)
z_mean = Dense(latent_dim, name='z_mean')(encoder)
z_log_var = Dense(latent_dim, name='z_log_var')(encoder)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(enc_input, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()

###############################
#### build decoder model
###############################

filter = 256
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# 8 x 8 x 256
x = Dense(sh[1] * sh[2] * sh[3])(latent_inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)	

# hm
x = Reshape((sh[1], sh[2], sh[3]))(x)

for i in range(3):
	x = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=2, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	filter //= 2
	if(i == 1): filter //= 2

outputs = Conv2D(filters=3, kernel_size=kernel_size, activation='tanh', padding='same', name='decoder_output', use_bias=False)(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()

###############################
#### build discriminator model
###############################

filter = 32
disc_input = Input(shape=input_shape, name='discriminator_input')
x = disc_input
x = Conv2D(filters=filter, kernel_size=kernel_size, activation='relu', padding='same', use_bias=False)(x)
filter *= 4
for i in range(3):
	x = Conv2D(filters=filter, kernel_size=kernel_size, strides=2, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	if(i==0): filter *= 2
x = Flatten()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)

# for VAEGAN
lth_layer = Activation('relu')(x)
disc_out = Dense(1, activation='sigmoid')(lth_layer)

# instantiate discriminator1 model
discriminator1 = Model(disc_input, disc_out, name='discriminator1')
# discriminator1.summary()


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


# instantiate generator model
discriminator1.trainable = False
outputs = discriminator1(decoder(latent_inputs))
generator_model = Model(latent_inputs, outputs, name='generator_model')
generator_model.compile(loss='binary_crossentropy', optimizer=optimizer)

# instantiate discriminator2 model
discriminator2 = Model(disc_input, disc_out, name='discriminator2')
discriminator2.compile(loss='binary_crossentropy',optimizer=optimizer)

discriminator_lth = Model(disc_input, lth_layer, name='discriminator_lth_layer')
discriminator_lth.trainable = False
outputs = discriminator_lth(decoder(encoder(inputs)[2]))
vae_model = Model(inputs, outputs)
def encoder_loss(y_true,y_pred):
	lth_loss = metrics.msle(K.flatten(y_true), K.flatten(y_pred))
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return (lth_loss + kl_loss)
vae_model.compile(loss=encoder_loss, optimizer=optimizer)


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
	
x_train = []
img_dir = "./img_align_celeba/"

if(load_images_at_start):
	x_train = []
	for name in sorted(os.listdir(img_dir)):
		img = io.imread(img_dir + name)
		img = img_resize(img, 64)
		img = norm_img(np.asarray(img_resize(img, 64)))
		print((int)(name.split('.')[0]))
	print("Loading done")
	x_train = np.asarray(x_train)

half_batch = int(batch_size / 2)
half_batch //= 2
half_batch //= 2

for epoch in range(epochs):
	img_array = np.random.randint(1, 202600, half_batch)
	if(load_images_at_start):
		img_array = img_array - 1
		imgs = x_train[img_array - 1]
	else:
		x_train = []
		for i in img_array:
			name = str(i)
			while (len(name) < 6): 
				name = '0' + name
			name = name + '.jpg'
			img = io.imread(img_dir + name)
			img = norm_img(np.asarray(img_resize(img,64)))
			x_train.append(img)
		imgs = np.asarray(x_train)
		img_array -= 1; 
	
	# vae losses / training
	if(load_images_at_start):
		imgs = x_train[img_array]
	else:
		imgs = np.asarray(x_train)
	imgs_lth = discriminator_lth.predict(imgs)
	enc_loss = vae_model.train_on_batch(imgs, imgs_lth)
	
	# discriminator losses / training
	disc_loss1 = discriminator2.train_on_batch(imgs, np.ones((half_batch,1)))
	lcode = encoder.predict(imgs)[2]
	lc_img = decoder.predict(lcode)
	disc_loss2 = discriminator2.train_on_batch(lc_img, np.zeros((half_batch,1)))
	noise = np.random.normal(0, 1, (half_batch, latent_dim))
	gen_imgs = decoder.predict(noise)
	disc_loss3 = discriminator2.train_on_batch(gen_imgs, np.zeros((half_batch,1)))
	
	# generator losses / training
	img_array = np.random.randint(1, 202600, batch_size)
	if(load_images_at_start): 
		img_array = img_array - 1;
		img_array2 = img_array[0:half_batch]
		imgs = x_train[img_array2]
	else:
		x_train = []
		for i in img_array:
			name = str(i)
			while (len(name) < 6): name = '0'+name
			name = name+'.jpg'
			img = io.imread(img_dir + name)
			img = norm_img(np.asarray(img_resize(img,64)))
			#img = (np.asarray(img_resize(img,64)) - 127.5) / 127.5 # tanh
			x_train.append(img)
		imgs = np.asarray(x_train[0:half_batch])
	latent_pred = encoder.predict(imgs)[2]
	gen_loss1 = generator_model.train_on_batch(latent_pred, np.ones((half_batch,1)))
	noise = np.random.normal(0, 1, (half_batch, latent_dim))
	gen_loss2 = generator_model.train_on_batch(noise, np.ones((half_batch,1)))
	
	if((epoch + 1) % 50 == 0 and epoch < epochs):
		print("Weights updated at epoch", epoch, "!")
		encoder.save_weights('VAEGAN_enc.h5')
		decoder.save_weights('VAEGAM_dec.h5')
		discriminator1.save_weights('VAEGAN_disc.h5')
	
	print("Steps", epoch + 1)
	print("Loss (D1 D2 D3):", disc_loss1, disc_loss2, disc_loss3)
	print("Loss (G1 G2):", gen_loss1, gen_loss2)
	print("Loss (VAE):", enc_loss)

print("Done. Weights updated.")
encoder.save_weights('VAEGAN_enc.h5')
decoder.save_weights('VAEGAN_dec.h5')
discriminator1.save_weights('VAEGAN_disc.h5')


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


decoder.load_weights('VAEGAN_dec.h5')
noise = np.random.normal(0, 1, (1, latent_dim))
res = denorm_img(decoder.predict(noise))
plt.imshow(res[0]), plt.show()
