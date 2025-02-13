# quick script to parse 2d slices through the tumour from the 3D volumes
# and labels from .json params 

import argparse
import os
import glob
import sys
import pandas as pd
from struct import pack
import numpy as np
import json
import re
import nibabel as nb
import scipy
import scipy.misc
from scipy.interpolate import griddata
from scipy.ndimage import zoom,rotate
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from matplotlib.image import imsave
import operator

IMRES = 256
NSLICE = 5
zNSLICE = NSLICE
DOF = 3

# crop or pad to dimension cropres centering the image
def crop_pad(I,cropres=192,padres=384,origin=None):
    (nx,ny) = np.shape(I)
    I = np.pad(I,(((padres-nx)//2,(padres-nx)//2),((padres-ny)//2,(padres-ny)//2)))
    if origin is None:
        ros = np.where(I)
        croporigin = (int(np.median(np.unique(ros[0]))),int(np.median(np.unique(ros[1]))))
    else:
        croporigin = origin
    cropshape=(cropres,cropres)
    i1 = tuple(map(lambda x,delta : x-delta//2, croporigin, cropshape))
    i2 = tuple(map(operator.add, i1, cropshape  ))
    crop = tuple(map(slice,i1,i2))
    Icrop = I[crop]
    # I = I.copy(order='C')
    return Icrop,croporigin


######
# main
######

def main(inputdir,outputdir):
 
    # load labels and cases from master spreadsheet
    rootdir = os.path.split(inputdir)[0]
    df = pd.read_excel(os.path.join(rootdir,'03_04_2024 RAD NEC MET.xlsx'),sheet_name = 'Montage Jan 2021 - June 2023 ')
    dx = df[['ID','Final Dx']].dropna()
    dx = dx.sort_values('Final Dx',ignore_index=True,ascending=False)

    # create train test val
    # the multi-slices of each case cannot actually be mixed up between train test val, so
    # have to make this selection now and not at runtime
    nT=np.zeros(2,dtype=int)
    for i,d in enumerate(['T','RN']):
        nT[i] = len(dx.query("`Final Dx`=='{}'".format(d)))
        idx = np.argsort(np.random.random(nT[i]))
        nT_train = np.floor(nT[i] * 0.8).astype(int)
        nT_val = np.floor((nT[i] - nT_train)/2).astype(int)
        nT_test = nT[i] - nT_train - nT_val
        dx.loc[idx[:nT_train]+i*nT[0],'tvt'] = 'train'
        dx.loc[idx[nT_train:nT_train+nT_val]+i*nT[0],'tvt'] = 'val'
        dx.loc[idx[:nT[i]-nT_test-1:-1]+i*nT[0],'tvt'] = 'test'

    # clear existing files
    for root,dir,files in os.walk(outputdir):
        for f in files:
            os.remove(os.path.join(root,f))


    vfiles = ['t1ce_stripped.nii','t2flair_register_stripped.nii']
    vkeys = ['t1','t2']
    mfiles = ['objectmask_ET.nii']
    mkeys = ['m']

    # image count
    icount = 0
    lbl = {}

    cases = dx['ID'].tolist()
    for c in cases:
        print(c)
        cpath = os.path.join(inputdir,c)

        I = {}
        for i,v in enumerate(vfiles):
            img = nb.load(os.path.join(cpath,v))
            I[vkeys[i]] = np.array(img.dataobj)
        for i,m in enumerate(mfiles):
            img = nb.load(os.path.join(cpath,m))
            I[mkeys[i]] = np.array(img.dataobj)
            ET = np.where(I[mkeys[i]])
        # write out individual 2d slice through ET roi in each orthogonal dimension with random order
        for dim in range(3):
            if len(ET[dim]) > 8:
                dset = np.unique(ET[dim])[4:-4]
                dset_idx = np.argsort(np.random.random(len(dset)))
                dset = dset[dset_idx]
                for et_dim in dset:
                    # find crop origin from t1 image
                    _,crop_origin = crop_pad(I['t1'].take(indices=et_dim,axis=dim))
                    for ch in I.keys():
                        fname = os.path.join(outputdir,'img_{}_{:05d}.png'.format(ch,icount))
                        a = I[ch].take(indices=et_dim,axis=dim)
                        # crop and pad
                        a,_ = crop_pad(a,origin=crop_origin)
                        imsave(fname,a)
                        lbl[icount] = {'dx':dx.query("ID=='{}'".format(c))['Final Dx'].iloc[0],'tvt':dx.query("ID=='{}'".format(c))['tvt'].iloc[0]}
                    icount += 1
    with open(os.path.join(outputdir,'lbl.json'),'w') as fp:
        json.dump(lbl,fp)
    a=1
#   inputdir - a root dir containing case sub-dirs with the .nii volumes and a file of case labels
#   outputdir - flat dir for 2d training slices and matching label file

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir',default=None)
    parser.add_argument('--outputdir',default=None)

    args = parser.parse_args()


    main(args.inputdir,args.outputdir)