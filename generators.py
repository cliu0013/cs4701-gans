import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Reshape,
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)

"""
This creates the DCGAN generator model from the tutorial at
https://www.tensorflow.org/tutorials/generative/dcgan
"""
def make_default_dcgan_generator():
    model = tf.keras.Sequential()
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


"""
Very basic first attempt at a non-DC GAN generator. 
It has 3 FC layers: random noise -> 128 -> 784 -> (28*28) 
Doesn't work very well in my expirience
"""
def make_gan_generator_v1():
    model = tf.keras.Sequential()

    model.add(Dense(128, use_bias=False, input_shape=(10,)))
    model.add(Dense(28 * 28, input_shape=(128,)))
    model.add(Reshape((28, 28, 1)))

    return model


"""
Make First layer of generator bigger
"""
def make_gan_generator_v2():

    model = tf.keras.Sequential()

    model.add(Dense(128, use_bias=False, input_shape=(100,)))
    model.add(Dense(28 * 28, input_shape=(128,)))
    model.add(Reshape((28, 28, 1)))

    return model


"""
Even Bigger Generator
"""
def make_gan_generator_v3():

    model = tf.keras.Sequential()

    model.add(Dense(128, use_bias=False, input_shape=(100,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(28*28, activation='relu'))
    model.add(Reshape((28, 28, 1)))

    return model


"""
Biggest Generator
"""
"""
Make First layer of generator bigger
"""
def make_gan_generator_v4():

    model = tf.keras.Sequential()

    model.add(Dense(128, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(28*28, input_shape=(128,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(28*28, input_shape=(128,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((28, 28, 1)))

    return model
