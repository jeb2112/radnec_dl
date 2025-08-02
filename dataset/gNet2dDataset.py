# Dataset for resnet and gnet. This will utilize a conventional resnet
# directory and filename structure.
# the former approach in nnunet2d_trainer_preprocess or nnUNet2dDataset of training
# a resnet via a nnunet file and dir convention, should be phased out. 
# the original intention for nnUNet2dDataset, of utilizing the encoder of the unet
# and adding a classifier output stage, could be retained and revisited.

import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
from PIL import ImageOps,Image
import nibabel as nb
import glob
import re
import copy
import os
import matplotlib.pyplot as plt
import random
import json
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class gNet2dDataset(Dataset):  
    def __init__(  
        self,   
        datadir, # ie a test,train,val dir
        transform = None,
        perturbation = 0,
        padding = 3,  
        image_size = (256, 256),  
        decimate = 0,
        in_memory = True,
        rgb = False,
        onehot=False,
        tag=None
    ):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datadir = datadir
        self.onehot = onehot

        self.classes = sorted(os.listdir(self.datadir))
        self.num_classes = len(self.classes)

        # create list of all classes and files
        self.imgs = []
        self.lbls = []
        self.cases = {}
        self.in_memory = True
        for c in self.classes:
            cpath = os.path.join(self.datadir,c)
            imgfiles = sorted(os.listdir(cpath))
            self.imgfiles.append([os.path.join(cpath,i) for i in imgfiles])
            self.lbls.append([c]*len(imgfiles))

        self.n = len(self.lbls)
        n0 = len(np.where(np.array(self.lbls)==0)[0])
        n1 = len(np.where(np.array(self.lbls)==1)[0])
        nmax = max((n0,n1))
        # for CrossEntropyLoss
        self.weights = torch.tensor([nmax/n0,nmax/n1]).to(self.device)

        # optionally decimate dataset for fast eval/debug
        if decimate:
            if decimate > self.n/decimate/10:
                raise ValueError('decimation {decimate} too high for n = {self.n}')
            random.seed(42)
            nsample = int(self.n/decimate)
            samples = sorted(random.sample(list(range(self.n)),nsample))
            self.lbls = [self.lbls[l] for l in samples]
            self.imgfiles = [imgfiles[l] for l in samples]
            self.n = nsample       

        # for common re-sizing
        self.image_size = image_size
        self.rgb = rgb
        # for general image processing
        self.transform = transform

        # for possible augmentation
        self.perturbation = perturbation
        self.padding = padding


        # load in to memory. for now this is the default.
        if in_memory:
            self.imgs = []
            self.cases = {}
            self.in_memory = True
            for imgfile in self.imgfiles:
                img = Image.open(imgfile)
                img = ImageOps.pad(img,self.image_size)
                img = np.array(img.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
                img /= 255.0
                self.imgs.append(img)

        else: # load from file dynamically. not coded yet
            imgs = sorted(os.listdir(self.imagedir))
            self.idx0 = int(re.search('[0-9]{6}',imgs[0]).group(0)) - 1
            self.in_memory = False

        return

    def __len__(self):  
        return self.n

    def __getitem__(self, idx):  

        inputs = {}

        if self.in_memory:
            inputs['img'] = torch.Tensor(self.imgs[idx])
            if self.transform and True:
                inputs['img'] = self.transform(inputs['img'])

            inputs['lbl'] = torch.tensor(self.lbls[idx][0],dtype=torch.long)  

        else: # read from file. not coded yet.

            idx += self.idx0
            t1file = glob.glob(os.path.join(self.imagedir,'img_' + str(idx+1).zfill(6) + '*_0001.png'))[0]
            # t1 = imread(t1file).astype(np.float32)
            t1 = Image.open(t1file)
            t1 = ImageOps.pad(t1,self.image_size)
            t1 = np.array(t1.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)

            inputs['img'] = torch.Tensor(t1)

            # create a dummy batch dimension for the labels as well
            inputs['lbl'] = torch.tensor(int(lbl), dtype=torch.float32).unsqueeze(0)
            # inputs['lbl'] = torch.tensor(lbl, dtype=torch.float32)
            # some general processing could done here explicitly 

        # debug plotting
        if False:
            plt.imshow(inputs['img'][0].numpy())

        return inputs

    # load a single nifti file
    def loadnifti(self,t1_file,dir,type=None):
        img_arr_t1 = None
        try:
            img_nb_t1 = nb.load(os.path.join(dir,t1_file))
        except FileNotFoundError:
            pass
        if type is not None:
            img_arr_t1 = img_arr_t1.astype(type)
        affine = img_nb_t1.affine
        return img_arr_t1,affine
    
    # write a single nifti file. use uint8 for masks 
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force "{}"'.format(filename))


    # calculate weights for data imbalance 
    # currently hard-coded for dataset, edit as necessary
    def balancedata(self):
        listlbls = np.array(self.lbls)
        # hard-coded here, has to match the actual dataset
        lbldict = {'GBM':[0],'MET':[1]}
        counts = {}
        for k in lbldict.keys():
            counts[k] = np.sum(np.all(listlbls == lbldict[k],axis=1))
        total_samples = sum(counts.values())  # 4428 + 1013 + 7846 + 3157

        # Compute total positive counts for each class (includes 'both')
        # hard-coded here. currently there are no tumor only cases.
        num_pos_GBM = counts['GBM']
        num_pos_MET = counts['MET']
        # Compute total negative counts per class
        num_neg_GBM = total_samples - num_pos_GBM
        num_neg_MET = total_samples - num_pos_MET

        # Compute pos_weight = (negative cases) / (positive cases)
        pos_weight_GBM = num_neg_GBM / num_pos_GBM
        pos_weight_MET = num_neg_MET / num_pos_MET

        pos_weight = [pos_weight_GBM,pos_weight_MET]
        if 0 in pos_weight:
            pos_weight = None
        else:
            # Convert to PyTorch tensor for BCEWithLogitsLoss
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)

        return pos_weight
