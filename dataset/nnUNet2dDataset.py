import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
from PIL import ImageOps,Image
import nibabel as nb
import glob
import copy
import os
import matplotlib.pyplot as plt
from random import sample
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta
from torch.utils.data import Dataset
import torch

# from pc_sam.datasets.transforms import normalize_points


class nnUNet2dDataset(Dataset):  
    def __init__(  
        self,   
        datadir,
        transform = None,
        perturbation = 0,
        padding = 3,  
        image_size = (256, 256),  
        decimate = 0,
        in_memory = False
    ):  
        self.datadir = datadir
        self.dataset = {}
        self.labeldir = os.path.join(self.datadir,'labelsTr')
        self.imagedir = os.path.join(self.datadir,'imagesTr')
        self.dataset['lbl'] = sorted(os.listdir(self.labeldir))
        self.n = len(self.dataset['lbl'])
        self.dataset['img'] = sorted(os.listdir(self.imagedir))
        self.dataset['img'] = [(self.dataset['img'][a],self.dataset['img'][a+1]) for a in range(0,2*self.n,2) ]
        # self.dataset['img'] = zip(self.dataset['img'][::2],self.dataset['img'][1::2])
        # for common re-sizing
        self.image_size = image_size

        # for general image processing
        self.transform = transform

        # for possible augmentation
        self.perturbation = perturbation
        self.padding = padding

        # optionally decimate dataset for fast eval/debug
        if decimate:
            nsample = int(self.n/decimate)
            samples = sorted(sample(list(range(self.n)),nsample))
            self.dataset['lbl'] = [self.dataset['lbl'][l] for l in samples]
            self.dataset['img'] = [self.dataset['img'][l] for l in samples]
            self.n = nsample

        # optional inload to memory
        if in_memory:
            self.imgs = []
            self.lbls = []
            self.in_memory = True
            for t1,flair in self.dataset['img']:
                t1 = Image.open(os.path.join(self.imagedir,t1))
                t1 = ImageOps.pad(t1,self.image_size)
                t1 = np.array(t1.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
                flair = Image.open(os.path.join(self.imagedir,flair))
                flair = ImageOps.pad(flair,self.image_size)
                flair = np.array(flair.getdata()).reshape(self.image_size[0],self.image_size[1]).astype(np.float32)
           
                self.imgs.append(np.stack((t1,flair),axis=0))

            for lblfile in self.dataset['lbl']:
                l_arr = imread(os.path.join(self.labeldir,lblfile))
                lbl = np.max(l_arr) == 2
                self.lbls.append(lbl.astype(int))




    def __len__(self):  
        return self.n

    def __getitem__(self, idx):  

        inputs = {}

        if self.in_memory:
            inputs['img'] = torch.Tensor(self.imgs[idx])
            inputs['lbl'] = torch.tensor(self.lbls[idx],dtype=torch.float32).unsqueeze(0)

        else:

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
            inputs['img'] = torch.Tensor(np.stack((t1,flair),axis=0))
            lfile = glob.glob(os.path.join(self.imagedir,'img_' + str(idx+1).zfill(6) + '*.png'))[0] 
            l_arr = imread(lfile)
            lbl = np.max(l_arr) == 2
            # create a dummy batch dimension for the labels as well
            inputs['lbl'] = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
            # inputs['lbl'] = torch.tensor(lbl, dtype=torch.float32)
            # some general processing could done here explicitly 

        if False:
            if self.transform is not None:
                input_image1 = self.transform(input_image)

        # debug plotting
        if False:
            pass

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


