# -*- coding: utf-8 -*-

import os
import platform
import glob
import argparse
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import pickle
import json
import re

from tensorflow.keras import callbacks
from keras_preprocessing.image.utils import (array_to_img,img_to_array)
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,Callback,CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import skimage.transform as trans
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# intellisense error on this decorator? somehow works
from tensorflow.random import set_seed

from datagen import RegressionGen
from checkpoint import ModelCheckpoint_json
import model as unet_model
from other import *
from display import *
from config import Config

# this kludge was for a cudnn failed to initialze error widely reported on internet, or else an out of memroy errors after it
# it did initialize even though the requested amount of memroy was less than 4Gb and nothing else was running on the gpu
# there are still memory warning about sub-optimal training, 
import tensorflow as tf
if platform.system() == 'Linux':    
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
else:
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# if True this switches to run on CPU
if False:
    tf.config.set_visible_devices([], 'GPU')


######
# main
######

# use agument=False for debugging
def main(P,thread=0):

    inputDirpng = '/'.join(P.imgdir.split('/')[:-1]) + '/img_png'

    # set the tensorflow seed
    if P.rseed > 0:
        set_tfdeterministic()
        if 0:
            set_tfenv(seed=P.rseed) # env vars don't seem to be needed, but if they are, they should probably be called from main thread.

    # load in regression labels
    (img_indx,lbl_data) = load_regression_labels(os.path.join(P.datadir,'lbl'),dof=P.dof,N=P.regressionN)

    # separate data into test/train
    if P.val_test: # separate validation, test
        (Xdata_train,Xdata_val,ydata_train,ydata_val) = train_test_split(img_indx,lbl_data,test_size=0.1,random_state=4,shuffle=P.val_test_shuffle)
        (Xdata_train,Xdata_test,ydata_train,ydata_test) = train_test_split(Xdata_train,ydata_train,test_size=0.1,random_state=4,shuffle=False) # no shuffle by definition
    else: # validation only
        (Xdata_train,Xdata_val,ydata_train,ydata_val) = train_test_split(img_indx,lbl_data,test_size=0.1,random_state=4,shuffle=P.val_test_shuffle)
        Xdata_test = Xdata_val
        ydata_test = ydata_val
        # resort the test data into ascending order for visualization purposes.
        test_index = np.argsort(Xdata_test)
        Xdata_test = np.array(Xdata_test)[test_index]
        ydata_test = ydata_test[test_index]

    output_resname = os.path.join(P.modelpath,'results.pkl')
    output_histname = os.path.join(P.modelpath,'hist.pkl')
    output_modelname = os.path.join(P.modelpath,'model')
    output_optimalname = os.path.join(P.modelpath,'optimal')
    output_pngdir = os.path.join(P.modelpath,'png')

    print('Thread {}: Model {:s}'.format(thread,P.modelname))
    if P.initial_model is not None:
        initial_modelpath = os.path.join(P.outputdir,P.initial_model)
    
    ###########
    #  training
    ###########

    # for regression there is nothing in keras to modify a scalar label so only transforms that can be applied to an angle
    # can't seem to give asymmetrical ranges with imagedatagenertor
    regression_aug = {}
    if P.augment:
        regression_aug = dict(fill_mode='nearest',
                        rotation_range=3,
                        zoom_range=0.05,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        vol_shift_range=0.1,
                        phi_rotation_range=2
                        # brightness_range=[1.0,1.0], # strange behaviour don't use
                        # samplewise_std_normalization=True
                        )

    # simple generator no way to capture augmentation
    # using do_shuffle??
    if P.retrain:
        regression_datagen = RegressionGen(Xdata_train,ydata_train,P.imgdir,output_size=P.dof,
                                            batch_size=P.batch,
                                            do_crop=1, # hard-coded here and below
                                            do_shuffle=True,
                                            do_plot=False,
                                            name='A',
                                            dims=P.mres,
                                            input_dims=P.input_dims,
                                            outputdir=P.modelpath,
                                            local=P.croporigin,
                                            lbldist=P.labeldist,
                                            do_labelnorm=P.labelnorm,
                                            labelvar=P.labelvar,
                                            **regression_aug
                                            )

        # doc says validation generator is not officially supported by keras?? 
        # but it works and doesn't break anything. it might not be running on the gpu.
        # provide seed for reproduceable crops
        validation_datagen = RegressionGen(Xdata_val,ydata_val,P.imgdir,output_size=P.dof,
                            batch_size=P.batch,do_crop=1,dims=P.mres,input_dims=P.input_dims,do_plot=False,seed=2,local=P.croporigin,
                            do_labelnorm=P.labelnorm,labelvar=P.labelvar,lbldist=regression_datagen.lbldist)

        prediction_lbldist = regression_datagen.lbldist # use label dist from regression
    else:
        prediction_lbldist = True     # re-read existing label dist

    # batch size 1 to get all data in prediction
    # provide seed for reproduceable crops
    prediction_datagen = RegressionGen(Xdata_test,ydata_test,P.imgdir,output_size=P.dof,outputdir=P.modelpath,
                        batch_size=1,dims=P.mres,input_dims=P.input_dims,do_crop=1,do_plot=False,seed=4,local=P.croporigin,
                        do_labelnorm=P.labelnorm,lbldist=prediction_lbldist)


    # callbacks
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=4,min_delta=0.01)
    mc1 = ModelCheckpoint_json(output_modelname+'-optimal',monitor='val_loss',mode='min',verbose=1,save_best_only=True)
    csv = CSVLogger(output_modelname+'-log.csv',append=True,separator=';')
    elapsedtime = TimeHistory()
    callbacks = [mc1]
    if P.csvlog:
        callbacks.append(csv)
        callbacks.append(elapsedtime)

    K.clear_session()
    if P.retrain:
        if P.checkdata:
            if True:
                pngdir = os.path.join(p.imgdir,'png_checkdata') # debugging output
                plist = glob.glob(pngdir+'/I*.png')
                for p in plist:
                    os.remove(p)
            else:
                pngdir = None
            # TODO: get rid of variant generators for display, just use main generator
            # display_img(regression_datagen.displayall_gen(nth=1,do_augment_y=True),vmax=0,pngdir=pngdir,delay=2,order=3,mslice=True)
            display_img(regression_datagen,vmax=0,pngdir=pngdir,delay=2,order=3,mslice=True)
        if P.checktest:
            if True:
                pngdir = os.path.join(p.imgdir,'png_checktest') # debugging output
                plist = glob.glob(pngdir+'/I*.png')
                for p in plist:
                    os.remove(p)
            else:
                pngdir = None
            display_img(validation_datagen,vmax=0,pngdir=pngdir,delay=2,order=3,mslice=True)

    if P.repredict:
        if P.checkdata:
            if True:
                pngdir = os.path.join(p.imgdir,'png_checkdata') # debugging output
                plist = glob.glob(pngdir+'/I*.png')
                for p in plist:
                    os.remove(p)
            else:
                pngdir = None
            # TODO: get rid of variant generators for display, just use main generator
            # display_img(regression_datagen.displayall_gen(nth=1,do_augment_y=True),vmax=0,pngdir=pngdir,delay=2,order=3,mslice=True)
            display_img(prediction_datagen,vmax=0,pngdir=pngdir,delay=2,order=3,mslice=True)
 
    if P.retrain:
        if P.initial_epoch > 1 and P.initial_model is not None:
            kerasmodel = unet_model.__dict__[P.model](input_size=P.mres+(1,),output_size=P.dof,pretrained_weights=P.initial_model+'.h5')
        else:
            # regularization, batchnorm hard-coded here
            kerasmodel = unet_model.__dict__[P.model](input_size=P.mres+(1,),output_size=P.dof,reg=l2(P.L2),batchnorm=P.batchnorm)
        if P.csvlog:
            verbose = 0
        else:
            verbose = 1
        history = kerasmodel.fit(    regression_datagen,
                                validation_data=validation_datagen,
                                callbacks=callbacks,
                                epochs=P.epochs,
                                shuffle=P.keras_shuffle,
                                verbose=verbose,
                                )

        if P.csvlog:
            print('Thread {}: time = {:.2f}, loess = {:.4f}'.format(thread,np.mean(elapsedtime.times),history.history['loss'][-1]))
        model_json = kerasmodel.to_json()
        with open(output_modelname+'.json','w') as fb:
            fb.write(model_json)
            fb.close()
        kerasmodel.save_weights(output_modelname+'.h5')
        with open(output_histname,'wb') as fb:
            trainhistory = {"history": history.history,"params": history.params,"rseed": P.rseed}
            pickle.dump(trainhistory,fb)
            fb.close()
        # resname = output_resname

    else:
        if os.path.exists(output_modelname+'-optimal.json'):
            with open(output_modelname+'-optimal.json','r') as fb:
                model_json = fb.read()
                fb.close()
            kerasmodel = model_from_json(model_json)
            kerasmodel.load_weights(output_modelname+'-optimal.h5')
        else:
            # loadmodel directly from h5 not working?
            # kerasmodel = load_model(output_modelname+'.h5')
            with open(output_modelname+'.json','r') as fb:
                model_json = fb.read()
                fb.close()
            kerasmodel = model_from_json(model_json)
            kerasmodel.load_weights(output_modelname+'.h5')

    ######
    # test
    ######

    if P.repredict or P.retrain:
        prediction_datagen.set_seed()
        res = kerasmodel.predict(prediction_datagen,verbose=0)
        if P.labelnorm:
            res = res * prediction_datagen.lbldist[1] + prediction_datagen.lbldist[0]
        # require an offset of 4 to match two keras preliminary __getitem__ calls.
        # prediction_datagen.set_seed(N=4)
        ydata_test_crop = prediction_datagen.lbl_crop
        res = np.squeeze(res)
        rmse = np.sqrt(np.mean(np.power(res - ydata_test_crop,2)))

        if P.repredict:
            try:
                with open(output_histname,'rb') as fb:
                    P.rseed = pickle.load(fb)['rseed'] # retain the original seed value
                    fb.close()
            except FileNotFoundError:
                P.rseed = 0
        with open(output_resname,'wb') as fb:
            pickle.dump((Xdata_test,ydata_test,ydata_test_crop,res,P.rseed,rmse),fb)
            fb.close()

        # try to release gpu
        tf.compat.v1.reset_default_graph()

    else:
        with open(output_resname,'rb') as fb:
            (Xdata_test,ydata_test,ydata_test_crop,res,P.rseed,rmse) = pickle.load(fb)
            fb.close()

    if not P.retrain:
        try: # full history available?
            with open(output_histname,'rb') as fb:
                trainhistory = pickle.load(fb)
                fb.close()
        except FileNotFoundError: # history up to last optimal checkpointnp.random.randint(-self.local,self.local+1)
            output_histname = re.sub('hist.pkl','model-optimal-hist.pickle',output_histname)
            with open(output_histname,'rb') as fb:
                trainhistory = pickle.load(fb)
                fb.close()
 
    ################
    # output display
    ################

    # error histogram and plots
    if not P.retrain:
        xvals=range(0,len(ydata_test))
        avals=ydata_test_crop
        errvals = res - ydata_test_crop
        errvals_mm=np.reshape(res[:,0:3]-ydata_test_crop[:,0:3],(3*len(ydata_test_crop),))
        if np.shape(ydata_test_crop)[1] > 3:
            errvals_deg=np.reshape(res[:,3:]-ydata_test_crop[:,3:],(2*len(ydata_test_crop),))
        else:
            errvals_deg = np.zeros(np.shape(ydata_test_crop)[0]*2,)
        deglim = (-5,5)
        pxlim = (-5,5)
        
        rmse_deg = np.sqrt(np.mean(np.power(np.abs(errvals_deg),2)))
        outliers = len(np.where(np.abs(errvals_deg) > deglim[1])[0])
        if outliers:
            rmse_deg = np.sqrt(np.mean(np.power(np.sort(np.abs(errvals_deg))[:-outliers],2)))
        rmse_mm = np.sqrt(np.mean(np.power(np.abs(errvals_mm),2)))
        outliers_mm = len(np.where(np.abs(errvals_mm) > pxlim[1])[0])
        if outliers_mm:
            rmse_mm = np.sqrt(np.mean(np.power(np.sort(np.abs(errvals_mm))[:-outliers_mm],2)))


        # results and loss plots
        if P.dof == 5:
            if 0:
                fit_res(P.modelname,res,xvals,avals,trainhistory,errvals,P.dof,output_pngdir,prediction_datagen)
            plot_res(P.modelname,res,xvals,avals,trainhistory,rmse_deg,rmse_mm,outliers,outliers_mm,errvals,P.dof,deglim,pxlim,output_pngdir,prediction_datagen)
        else:
            pxlim = (-10,10)
            errvals_mm=res[:,0:3]-ydata_val[:,0:3]
            outliers_mm = np.zeros(3,dtype=int)
            rmse_mm = np.zeros(3,)
            for i in range(3):
                outliers_mm[i] = len(np.where(np.abs(errvals_mm[:,i]) > pxlim[1])[0])
                if outliers_mm[i]:
                    rmse_mm[i] = np.sqrt(np.mean(np.power(np.sort(np.abs(errvals_mm[:,i]))[:-outliers_mm[i]],2),axis=0))
                else:
                    rmse_mm[i] = np.sqrt(np.mean(np.power(np.abs(errvals_mm[:,i]),2),axis=0))
            plot_res_3dof(P.modelname,res,xvals,avals,trainhistory,rmse_deg,rmse_mm,outliers,outliers_mm,errvals,P.dof,deglim,pxlim,output_pngdir,prediction_datagen)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=None) # input file, list of configfiles to run
    parser.add_argument('--modeldir',default=None) # output model directory for post-processing
    parser.add_argument('--predict',action='store_true',default=False)
    parser.add_argument('--resume',action='store_true',default=False)
    args = parser.parse_args()

    if args.config is None and args.modeldir is None:
        raise RuntimeError('Specify a config file')
    else:
        config = Config(args.config)

    if args.modeldir is not None:
        config = Config(modeldir=args.modeldir,predict=args.predict,resume=args.resume)
    main(config.p)