import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import layer_utils
# from tensorflow.keras.utils.vis_utils import model_to_dot
# from tensorflow.keras.utils import plot_model
# from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K

# TBD
# K.set_image_data_format('channels_last')

if 0: # keras deprecated. but because Model has not been subclassed with a custom call method, the training arg is not needed?
    K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block, reg=None, batchnorm=False):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv3D(filters = F1, kernel_size = (1, 1, 1), strides = (1,1,1), padding = 'valid', name = conv_name_base + '2a', 
        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) # confined to batch 1 in 3d for now
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv3D(filters = F2, kernel_size = (f, f, f), strides = (1,1,1), padding = 'same', name = conv_name_base + '2b', 
        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv3D(filters = F3, kernel_size = (1, 1, 1), strides = (1,1,1), padding = 'valid', name = conv_name_base + '2c', 
        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2, reg=None, batchnorm=False):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv3D(F1, (1, 1, 1), strides = (s,s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) # skipping due to batch 1
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv3D(filters = F2, kernel_size = (f, f, f), strides = (1,1,1), padding = 'same', name = conv_name_base + '2b', 
        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv3D(filters = F3, kernel_size = (1, 1, 1), strides = (1,1,1), padding = 'valid', name = conv_name_base + '2c', 
        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X)
    if batchnorm:
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv3D(filters = F3, kernel_size = (1, 1, 1), strides = (s,s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=reg)(X_shortcut)
    if batchnorm:
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


def ResNet50(input_shape=(64, 64, 3), classes=6, reg=None, batchnorm=False):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding3D((3, 3, 3))(X_input)

    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    if batchnorm:
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1, reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', reg=reg, batchnorm=batchnorm)

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2, reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d', reg=reg, batchnorm=batchnorm)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2, reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f', reg=reg, batchnorm=batchnorm)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2, reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b', reg=reg, batchnorm=batchnorm)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c', reg=reg, batchnorm=batchnorm)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D((2,2,2), name="avg_pool", padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='linear', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


def ResNet39(input_shape=(64, 64, 3), classes=6, reg=None):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding3D((3, 3, 3))(X_input)

    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1, reg=reg)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', reg=reg)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', reg=reg)

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2, reg=reg)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b', reg=reg)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c', reg=reg)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d', reg=reg)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2, reg=reg)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b', reg=reg)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c', reg=reg)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d', reg=reg)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e', reg=reg)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f', reg=reg)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D((2,2,2), name="avg_pool", padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='linear', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet39')

    return model