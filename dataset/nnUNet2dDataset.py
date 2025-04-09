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

class nnUNet2dDataset(Dataset):  
    def __init__(  
        self,   
        imgdir,
        lbldir,
        num_classes = 2,
        transform = None,
        perturbation = 0,
        padding = 3,  
        image_size = (256, 256),  
        decimate = 0,
        in_memory = False,
        rgb = False,
        onehot=False,
        tag=None
    ):  
        self.dataset = {}
        self.labeldir = lbldir
        self.imagedir = imgdir
        self.num_classes = num_classes
        self.onehot = onehot
        lblfiles = sorted(os.listdir(self.labeldir))
        self.n = len(lblfiles)
        self.lbls = []
        for l in lblfiles:
            with open(os.path.join(self.labeldir,l)) as fp:
                lbl_dx = json.load(fp)['dx']
                self.lbls.append(lbl_dx)

        imgfiles = sorted(os.listdir(self.imagedir))
        if False: # 2 ch
            imgfiles = [(imgfiles[a],imgfiles[a+1]) for a in range(0,2*self.n,2) ]
        else: # 3ch
            imgfiles = [(imgfiles[a],imgfiles[a+1],imgfiles[a+2]) for a in range(0,3*self.n,3) ]

        self.n = len(self.lbls)

        # optionally decimate dataset for fast eval/debug
        if decimate:
            if decimate > self.n/decimate/10:
                raise ValueError('decimation {decimate} too high for n = {self.n}')
            random.seed(42)
            nsample = int(self.n/decimate)
            samples = sorted(random.sample(list(range(self.n)),nsample))
            self.lbls = [self.lbls[l] for l in samples]
            imgfiles = [imgfiles[l] for l in samples]
            self.n = nsample       

        # for common re-sizing
        self.image_size = image_size
        self.rgb = rgb
        # for general image processing
        self.transform = transform

        # for possible augmentation
        self.perturbation = perturbation
        self.padding = padding

        self.imgfiles = imgfiles

        # inload to memory. for now this is the default.
        if in_memory:
            self.imgs = []
            self.cases = {}
            self.in_memory = True
            for lbl_dx,(t1pfile,flairfile,t1file) in zip(self.lbls,self.imgfiles): # 3ch
                t1p = Image.open(os.path.join(self.imagedir,t1pfile))
                t1p = ImageOps.pad(t1p,self.image_size)
                t1p = np.array(t1p.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
                t1p /= 255.0
                flair = Image.open(os.path.join(self.imagedir,flairfile))
                flair = ImageOps.pad(flair,self.image_size)
                flair = np.array(flair.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
                flair /= 255.0
                t1 = Image.open(os.path.join(self.imagedir,t1file))
                t1 = ImageOps.pad(t1,self.image_size)
                t1 = np.array(t1.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
                t1 /= 255.0

                img_stack = np.stack((t1p,flair,t1),axis=0)
                if self.rgb and False: # old 2ch
                    img_stack = np.concatenate((img_stack,np.expand_dims(np.mean(img_stack,axis=0),axis=0)))
                self.imgs.append(img_stack)

        else: # load from file dynamically. not updated lately
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

            if self.onehot:
                inputs['lbl'] = torch.tensor(self.lbls[idx][0],dtype=torch.long)  
            else:
                inputs['lbl'] = torch.tensor(self.lbls[idx],dtype=torch.float)

            inputs['fdata'] = self.imgfiles[idx]

        else: # read from file. generally too slow.

            idx += self.idx0
            t1file = glob.glob(os.path.join(self.imagedir,'img_' + str(idx+1).zfill(6) + '*_0001.png'))[0]
            # t1 = imread(t1file).astype(np.float32)
            t1 = Image.open(t1file)
            t1 = ImageOps.pad(t1,self.image_size)
            t1 = np.array(t1.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)

            flfile = glob.glob(os.path.join(self.imagedir,'img_' + str(idx+1).zfill(6) + '*_0003.png'))[0]
            # flair = imread(flfile).astype(np.float32)
            flair = Image.open(flfile)
            flair = ImageOps.pad(flair,self.image_size)
            flair = np.array(flair.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
            # explicit batch dimension needed here?
            # inputs['img'] = torch.Tensor(np.stack((t1,flair),axis=0)).unsqueeze(0)
            imgstack = np.stack((t1,flair),axis=0)
            if self.rgb:
                imgstack = np.concatenate((imgstack,np.expand_dims(np.mean(imgstack,axis=0),axis=0)))
            inputs['img'] = torch.Tensor(imgstack)

            lfile = glob.glob(os.path.join(self.labeldir,'img_' + str(idx+1).zfill(6) + '*.png'))[0] 
            l_arr = imread(lfile)
            lbl = 1 in np.unique(l_arr)
            # create a dummy batch dimension for the labels as well
            inputs['lbl'] = torch.tensor(int(lbl), dtype=torch.float32).unsqueeze(0)
            # inputs['lbl'] = torch.tensor(lbl, dtype=torch.float32)
            # some general processing could done here explicitly 

        if False:
            if self.transform is not None:
                input_image1 = self.transform(input_image)

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


    # calculate weights for data imbalance for multi-hot encoding
    # hard-coded for dataset
    def balancedata(self):
        listlbls = np.array(self.lbls)
        # hard-coded here, has to match the actual dataset
        lbldict = {'RN':[0,1],'both':[1,1]}
        counts = {}
        for k in lbldict.keys():
            counts[k] = np.sum(np.all(listlbls == lbldict[k],axis=1))
        total_samples = sum(counts.values())  # 4428 + 1013 + 7846 + 3157

        # Compute total positive counts for each class (includes 'both')
        # hard-coded here. currently there are no tumor only cases.
        num_pos_T = counts['both']
        num_pos_RN = counts['RN'] + counts['both']

        # Compute total negative counts per class
        num_neg_T = total_samples - num_pos_T
        num_neg_RN = total_samples - num_pos_RN

        # Compute pos_weight = (negative cases) / (positive cases)
        pos_weight_T = num_neg_T / num_pos_T
        pos_weight_RN = num_neg_RN / num_pos_RN

        pos_weight = [pos_weight_T,pos_weight_RN]
        if 0 in pos_weight:
            pos_weight = None
        else:
            # Convert to PyTorch tensor for BCEWithLogitsLoss
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pos_weight = pos_weight.to(device)

        return pos_weight
