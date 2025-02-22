import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
# from resnet import ResNet50,ResNet39

import torch

from nnunet.nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunet.nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunet.nnunetv2.utilities.get_network_from_plans import get_network_from_plans

def rmse(y_true, y_predict):
    return K.sqrt(K.mean(K.square(y_true-y_predict),axis=-1))

NCONV=2
def down_conv(x,F,k,strides=None,poolsize=(2,2,2),dropout=None,activation='relu',padding='same',kernel_initializer='he_normal',nconv=NCONV,kernel_regularizer=None):
    x2 = None
    for c in range(nconv):
        x = torch.nn.Conv3d(F, k, activation = activation, padding = padding, kernel_initializer = kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if dropout:
        x2 = x
        x = torch.nn.Dropout(dropout)(x)
    if poolsize:
        x2 = x
        x = torch.nn.MaxPool3d(pool_size=poolsize, strides=strides)(x)
    return x,x2

def up_conv(x,F,k,xcat=None,strides=(1,1,1),poolsize=(2,2,2),dropout=None,activation='relu',padding='same',kernel_initializer='he_normal',nconv=NCONV):
    x = torch.nn.Upsample(size=(2,2,2))(x)
    x = torch.nn.Conv3d(F, k-1, activation = activation, padding = padding, kernel_initializer = kernel_initializer)(x)
    if torch.is_tensor(xcat):
        x = torch.concatenate([xcat,x], axis = 4)
    for c in range(nconv):
        x = torch.nn.Conv3D(F, k, activation = activation, padding = padding, kernel_initializer = kernel_initializer)(x)
    return x


def unet5(pretrained_weights=None, input_size=(128,128,1), output_size=1):
    inputs = torch.Tensor(input_size)
    conv1,c1 = down_conv(inputs,64,3)
    conv2,c2 = down_conv(conv1,128,3)
    conv3,c3 = down_conv(conv2,256,3)
    conv4,c4 = down_conv(conv3,512,3,dropout=0.5)
    conv5,_ = down_conv(conv4,1024,3,dropout=0.5,poolsize=None)
    up6 = up_conv(conv5,512,3,c4)
    up7 = up_conv(up6,256,3,c3)
    up8 = up_conv(up7,128,3,c2)
    up9 = up_conv(up8,64,3,c1)

    mf = torch.flatten()(up9)
    mf = torch.nn.Linear(output_size, activation="linear", kernel_regularizer='l1')(mf)
    model = torch.nn.Module(inputs = inputs, outputs = mf)
    model.compile(optimizer = torch.optim.Adam(lr = 2e-4), loss = rmse, metrics = ['mean_squared_error'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet4(pretrained_weights=None, input_size=(128,128,1), output_size=1):
    inputs = torch.Tensor(input_size)
    conv1,c1 = down_conv(inputs,64,3)
    conv2,c2 = down_conv(conv1,128,3)
    conv3,c3 = down_conv(conv2,256,3,dropout=0.5)
    conv4,_ = down_conv(conv3,512,3,dropout=0.5,poolsize=None)
    up5 = up_conv(conv4,256,3,c3)
    up6 = up_conv(up5,128,3,c2)
    up7 = up_conv(up6,64,3,c1)

    mf = torch.flatten()(up7)
    mf = torch.nn.Linear(output_size, activation="linear", kernel_regularizer='l1')(mf)
    model = torch.nn.Module(inputs = inputs, outputs = mf)
    model.compile(optimizer = torch.optim.Adam(lr = 2e-4), loss = rmse, metrics = ['mean_squared_error'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def vgg4(pretrained_weights=None, input_size=(128,128,1), output_size=1, reg=None):
    inputs = torch.Tensor(input_size)
    conv1,c1 = down_conv(inputs,64,3,kernel_regularizer=reg, strides=(2,2,2))
    conv2,c2 = down_conv(conv1,128,3,kernel_regularizer=reg, strides=(2,2,2))
    conv3,c3 = down_conv(conv2,256,3,kernel_regularizer=reg, strides=(2,2,2))
    conv4,c4 = down_conv(conv3,512,3,kernel_regularizer=reg, strides=(2,2,2))

    mf = torch.flatten()(conv4)
    mf = torch.nn.Linear(units=4096, activation='relu')(mf)
    mf = torch.nn.Dropout(0.5)(mf)
    mf = torch.nn.Linear(units=4096, activation='relu')(mf)
    mf = torch.nn.Dropout(0.5)(mf)
    mf = torch.nn.Linear(output_size, activation="linear", kernel_regularizer=reg)(mf)
    model = torch.nn.Module(inputs = inputs, outputs = mf)
    model.compile(optimizer = Adam(lr = 2e-4), loss = rmse, metrics = ['mean_squared_error'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def resnet(pretrained_weights=None, input_size=(32,128,16), output_size=5, reg=None, batchnorm=False, summary=False):
    model = ResNet50(input_shape = input_size, classes = output_size, reg=reg, batchnorm=batchnorm)
    model.compile(optimizer=torch.optim.Adam(lr=2e-4), loss='mean_absolute_error', metrics=['mean_absolute_error','mean_squared_error'])
    if summary:
        model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

def nnunet_encoder(ckpt_dir):

    ckpt = torch.load(ckpt_dir, torch.device('cpu'),weights_only=False)

    plans=ckpt["init_args"]["plans"]
    configuration_name = ckpt['init_args']['configuration']
    dataset_json=ckpt["init_args"]["dataset_json"]

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager,
                                                    dataset_json)

    model=get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            allow_init=True,
            deep_supervision=False)
    model.load_state_dict(ckpt["network_weights"])

    encoder=model.encoder
    return encoder

def croporigin(pretrained_weights=None, input_size=(256,256,70), output_size=3, reg='l2', batchnorm=False, summary=False):
    model = ResNet50(input_shape = input_size, classes = output_size, reg=reg)
    model.compile(optimizer=torch.optim.Adam(lr=2e-4), loss=rmse, metrics=['mean_squared_error'])
    if summary:
        model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
    
   