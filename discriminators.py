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
This creates the DCGAN discriminator model from the tutorial at
https://www.tensorflow.org/tutorials/generative/dcgan
"""
def make_default_gcgan_discriminator():
    model = tf.keras.Sequential()
    
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


"""
Very basic first attempt at a non-DC GAN discriminator. 
It has 3 FC layers: image -> 128 -> 1  
Doesn't work very well in my expirience
"""
def make_gan_discriminator_v1():

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    return model


"""
Add an extra layer to non DC gan discriminator V1
"""
def make_gan_discriminator_v2():

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))

    return model

"""
Even Bigger Generator
"""
def make_gan_discriminator_v3():

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(728, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    return model

"""
Biggest Discriminator
"""
def make_gan_discriminator_v4():

    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(728, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    return model