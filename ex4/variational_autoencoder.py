'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

from __future__ import print_function

SAVE_PLOTS = False

import os
import random
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
if SAVE_PLOTS:
    mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

PATH = os.getcwd()
random.seed(11)

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
fix_std = 1.0


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


def fix_var_sampling(args):
    z_mean = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    z = z_mean + fix_std * epsilon
    return z


def encoder_model(vae_model, input_shape, fix_var_flag=False, conv_flag=False):
    """
    Build encoder model

    :parameter
    vae_model: pre trained vae model
    input_shape: encoder input shape
    fix_var_flag: True: fix variance, False: trained variance
    conv_flag: True: conv vae model, False: FC vae model

    :return
    encoder: encoder model with weights from the pre trained model
    """

    encoder_layers = int(np.ceil(len(vae_model.layers)/2))
    # build encoder model
    if conv_flag:
        encoder_input = Input(shape=input_shape)
        h_encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
        h_encoded = MaxPooling2D((2, 2), padding='same')(h_encoded)
        h_encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(h_encoded)
        h_encoded = MaxPooling2D((2, 2), padding='same')(h_encoded)
        h_encoded = Flatten()(h_encoded)
        h_encoded = Dense(16, activation='relu')(h_encoded)
    else:
        encoder_input = Input(shape=input_shape)
        h_encoded = Dense(intermediate_dim, activation='relu')(encoder_input)
    z_mean = Dense(latent_dim)(h_encoded)
    if not fix_var_flag:
        z_log_var = Dense(latent_dim)(h_encoded)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    else:
        z = Lambda(fix_var_sampling, output_shape=(latent_dim,))(z_mean)

     # get layers weights
    encoder = Model(encoder_input, z)
    for j in np.arange(1, encoder_layers):
        encoder.layers[j].set_weights(vae_model.layers[j].get_weights())
    return encoder


def decoder_model(vae_model, input_shape, intermediate_dim, conv_flag=False):
    """
    Build decoder model

    :parameter
    vae_model: pre trained vae model
    input_shape: encoder input shape
    intermediate_dim: first fc hidden layer size
    conv_flag: True: conv vae model, False: FC vae model

    :return
    decoder: decoder model with weights from the pre trained model
    """

    num_layers = len(vae_model.layers)
    # build decoder
    decoder_input = Input(shape=input_shape)
    h_decoded = Dense(intermediate_dim, activation='relu')(decoder_input)
    if conv_flag:
        # we instantiate these layers separately so as to reuse them later
        h_decoded = Dense(7*7*8, activation='relu')(h_decoded)
        h_decoded = Reshape((7, 7, 8))(h_decoded)
        h_decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        h_decoded = Conv2D(16, (3, 3), activation='relu')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h_decoded)
    else:
        x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    decoder = Model(inputs=decoder_input, outputs=x_decoded_mean)
    if conv_flag:
        for j in np.arange(1, 9):
            decoder.layers[j].set_weights(vae_model.layers[j+9].get_weights())
    else:
        for j in np.arange(1, 3):
            decoder.layers[j].set_weights(vae_model.layers[num_layers + j - 3].get_weights())
    return decoder


def vae_model(input_shape, intermediate_dim, latent_dim, fix_var_flag=False, conv_flag=False):
    """
    Build vae model (both fc anc conv)

    :parameter
    input_shape: vae input shape
    intermediate_dim: first fc hidden layer size
    latent_dim: latent layer dimension (z)
    fix_var_flag: True: fix variance, False: trained variance
    conv_flag: True: conv vae model, False: FC vae model

    :return
    model: initialize vae model (not trained)
    """

    shape = None
    # Encoder
    if conv_flag:
        x = Input(shape=input_shape)
        h_encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        h_encoded = MaxPooling2D((2, 2), padding='same')(h_encoded)
        h_encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(h_encoded)
        h_encoded = MaxPooling2D((2, 2), padding='same')(h_encoded)
        # shape info needed to build decoder model
        shape = K.int_shape(h_encoded)
        # generate latent vector Q(z|X)
        h_encoded = Flatten()(h_encoded)
        h_encoded = Dense(intermediate_dim, activation='relu')(h_encoded)
    else:
        x = Input(shape=input_shape)
        h_encoded = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h_encoded)
    if not fix_var_flag:
        z_log_var = Dense(latent_dim)(h_encoded)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    else:
        z_log_var = K.constant(np.log(fix_std**2), shape=(2,))
        z = Lambda(fix_var_sampling, output_shape=(latent_dim,))(z_mean)

    # Decoder
    h_decoded = Dense(intermediate_dim, activation='relu')(z)
    if conv_flag:
        # we instantiate these layers separately so as to reuse them later
        h_decoded = Dense(shape[1] * shape[2] * shape[3], activation='relu')(h_decoded)
        h_decoded = Reshape((shape[1], shape[2], shape[3]))(h_decoded)
        h_decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        h_decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h_decoded)
    else:
        x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    # Compute VAE loss
    xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # instantiate VAE model
    model = Model(x, x_decoded_mean)
    model.add_loss(vae_loss)
    model.compile(optimizer='rmsprop', loss='')
    model.summary()
    return model


# fc VAE
vae = vae_model((original_dim,), intermediate_dim, latent_dim)
# fc fix variance VAE
fix_var_vae = vae_model((original_dim,), intermediate_dim, latent_dim, fix_var_flag=True)
# conv VAE
conv_vae = vae_model((28, 28, 1), 16, latent_dim, conv_flag=True)

# load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_conv = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test_conv = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# train the VAE on MNIST digits
if os.path.isfile(os.path.join(PATH, 'vae_weights.h5')):
    vae.load_weights('vae_weights.h5')
else:
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            verbose=2)

    vae.save_weights('vae_weights.h5')

# load pre trained weights or train the models
if os.path.isfile(os.path.join(PATH, 'fix_var_vae_weights.h5')):
    fix_var_vae.load_weights('fix_var_vae_weights.h5')
else:
    fix_var_vae.fit(x_train,
                    shuffle=True,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, None),
                    verbose=2)

    fix_var_vae.save_weights('fix_var_vae_weights.h5')

if os.path.isfile(os.path.join(PATH, 'conv_vae_weights.h5')):
    conv_vae.load_weights('conv_vae_weights.h5')
else:
    conv_vae.fit(x_train_conv,
                 shuffle=True,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(x_test_conv, None),
                 verbose=2)

    conv_vae.save_weights('conv_vae_weights.h5')

# Create Encoders
encoder = encoder_model(vae, (original_dim,))
fix_var_encoder = encoder_model(fix_var_vae, (original_dim,), fix_var_flag=True)
conv_encoder = encoder_model(conv_vae, (28, 28, 1), conv_flag=True)

# Create Decoders
generator = decoder_model(vae, (latent_dim,), intermediate_dim)
fix_var_generator = decoder_model(fix_var_vae, (latent_dim,), intermediate_dim)
conv_generator = decoder_model(conv_vae, (latent_dim,), 16, conv_flag=True)

# section c
# ---------
# choose randomly 10 digits
ind = []
for digit in np.arange(y_test.max()+1):
    ind.append(random.sample(list(np.argwhere(y_test==digit)[0]), 1)[0])
# get latent space and generate digits for fc encoder
latent_space_data = encoder.predict(x_test)
latent_digits = latent_space_data[ind, :]
digits = generator.predict(latent_digits)
# get latent space and generate digits for fc encoder with fix variance
fix_latent_space_data = fix_var_encoder.predict(x_test)
fix_latent_digits = fix_latent_space_data[ind, :]
fix_digits = fix_var_generator.predict(fix_latent_digits)

# get latent space and generate digits for conv encoder
conv_latent_space_data = conv_encoder.predict(x_test_conv)
conv_latent_digits = conv_latent_space_data[ind, :]
conv_digits = conv_generator.predict(conv_latent_digits)

if SAVE_PLOTS:
    if not os.path.exists('1c'):
        os.makedirs('1c')
    if not os.path.exists('1d'):
        os.makedirs('1d')
    if not os.path.exists('1e'):
        os.makedirs('1e')

# plot digits mapped to latent space coordinates
digits_figures = []
for i in range(10):
    digit_figure = plt.figure()

    digit_figure.add_subplot(1, 3, 1) #VAE
    plt.imshow(np.reshape(digits[i], (28,28)))
    plt.gray()
    plt.axis('off')

    digit_figure.add_subplot(1, 3, 2) #Fix Variance VAE
    plt.imshow(np.reshape(fix_digits[i],(28,28)))
    plt.gray()
    plt.axis('off')

    digit_figure.add_subplot(1, 3, 3) # conv
    plt.imshow(np.squeeze(conv_digits[i, :, :, :]))
    plt.gray()
    plt.axis('off')

    digit_figure.suptitle('(1c) #%d generated with VAE, fix-var VAE, conv-VAE'%i)
    if SAVE_PLOTS:
        digit_figure.savefig('1c/digit_{}.png'.format(i))
    digits_figures.append(digit_figure)

latent_space_fig = plt.figure()
for i in np.arange(len(ind)):
    latent_space_fig.add_subplot(3, 4, (i+1))
    plt.scatter(latent_space_data[y_test == i, 0], latent_space_data[y_test == i, 1], s=1, c='b', alpha=1.0)
    plt.scatter(fix_latent_space_data[y_test == i, 0], fix_latent_space_data[y_test == i, 1], s=1, c='r', alpha=0.5)
    plt.scatter(conv_latent_space_data[y_test == i, 0], conv_latent_space_data[y_test == i, 1], s=1, c='g', alpha=0.5)
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    plt.grid()
    plt.title('digit #'+str(i))
latent_space_fig.suptitle('(1c) Latent Space Coordinates')
plt.legend(['VAE', 'fix var VAE', 'Conv VAE'], loc='upper center', bbox_to_anchor=(1.45, 0.8))
if SAVE_PLOTS:
    latent_space_fig.savefig('1c/entire_space_coordinates.png')

# print table of latent space # TODO
#print(latent_digits)

# generate 10 digits for each generator

# show all 10 digits
# digits_fig = plt.figure()
# for i in np.arange(len(ind)):
#     digits_fig.add_subplot(3, 4, i+1)
#     plt.imshow(np.reshape(digits[i],(28,28)))
#     plt.gray()
#     plt.axis('off')
#     plt.title('digit #'+str(i))
# digits_fig.suptitle('Digits Generator')
#
# fix_digits_fig = plt.figure()
# for i in np.arange(len(ind)):
#     fix_digits_fig.add_subplot(3, 4, i+1)
#     plt.imshow(np.reshape(fix_digits[i],(28,28)))
#     plt.gray()
#     plt.axis('off')
#     plt.title('digit #'+str(i))
# fix_digits_fig.suptitle('Digits Generator with fix variance')
#
# conv_digits_fig = plt.figure()
# for i in np.arange(len(ind)):
#     conv_digits_fig.add_subplot(3, 4, i+1)
#     plt.imshow(np.squeeze(conv_digits[i,:,:,:]))
#     plt.gray()
#     plt.axis('off')
#     plt.title('digit #'+str(i))
# conv_digits_fig.suptitle('Conv Architecture Digits Generator')

# section d
# ---------
z_sample = np.array([[0.5, 0.2]])
x_decoded = generator.predict(z_sample)
x_decoded_fix = fix_var_generator.predict(z_sample)
x_decoded_conv = conv_generator.predict(z_sample)

decoded_figure = plt.figure()

decoded_figure.add_subplot(1, 3, 1)  # VAE
plt.imshow(np.reshape(x_decoded, (28, 28)))
plt.gray()
plt.axis('off')

decoded_figure.add_subplot(1, 3, 2)  # Fix Variance VAE
plt.imshow(np.reshape(x_decoded_fix, (28, 28)))
plt.gray()
plt.axis('off')

decoded_figure.add_subplot(1, 3, 3)  # conv
plt.imshow(np.squeeze(conv_digits[i, :, :, :]))
plt.gray()
plt.axis('off')

decoded_figure.suptitle('(1d) (0.5, 0.2) decoded with VAE, fix-var VAE, conv-VAE')
if SAVE_PLOTS:
    decoded_figure.savefig('1d/1d.png')

# section e
# ---------
# interpolate coordination between two points for generation
# and then, generate digits from the interpolated coordination
points = latent_digits[random.sample(range(10), 2), :]
x_interp = np.linspace(points[0, 0], points[1, 0], 10)
y_interp = np.interp(x_interp, points[:, 0], points[:, 1])
z_interp = np.column_stack((x_interp, y_interp))
interp_digits = generator.predict(z_interp)

points = fix_latent_digits[random.sample(range(10), 2), :]
x_interp = np.linspace(points[0, 0], points[1, 0], 10)
y_interp = np.interp(x_interp, points[:, 0], points[:, 1])
z_interp = np.column_stack((x_interp, y_interp))
fix_interp_digits = generator.predict(z_interp)


points = conv_latent_digits[random.sample(range(10), 2), :]
x_interp = np.linspace(points[0, 0], points[1, 0], 10)
y_interp = np.interp(x_interp, points[:, 0], points[:, 1])
z_interp = np.column_stack((x_interp, y_interp))
conv_interp_digits = generator.predict(z_interp)


# show the digits
interp_digits_fig = plt.figure()
for i in np.arange(z_interp.shape[0]):
    interp_digits_fig.add_subplot(3, 4, i+1)
    plt.imshow(np.reshape(interp_digits[i], (28, 28)))
    plt.gray()
    plt.axis('off')
interp_digits_fig.suptitle('(1e) Interpolated Digits')
if SAVE_PLOTS:
    interp_digits_fig.savefig('1e/interpolated.png')

fix_interp_digits_fig = plt.figure()
for i in np.arange(z_interp.shape[0]):
    fix_interp_digits_fig.add_subplot(3, 4, i+1)
    plt.imshow(np.reshape(fix_interp_digits[i], (28, 28)))
    plt.gray()
    plt.axis('off')
fix_interp_digits_fig.suptitle('(1e) Interpolated Digits with fix variance')
if SAVE_PLOTS:
    fix_interp_digits_fig.savefig('1e/interpolated_fix_variance.png')

conv_interp_digits_fig = plt.figure()
for i in np.arange(z_interp.shape[0]):
    conv_interp_digits_fig.add_subplot(3, 4, i+1)
#    plt.imshow(np.squeeze(conv_interp_digits[i,:,:,:]))
    plt.imshow(np.reshape(conv_interp_digits[i], (28, 28)))
    plt.gray()
    plt.axis('off')
conv_interp_digits_fig.suptitle('(1e) Conv Architecture Interpolated Digits')
if SAVE_PLOTS:
    conv_interp_digits_fig.savefig('1e/interpolated_conv.png')

if not SAVE_PLOTS:
    plt.show()
