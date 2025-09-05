# script selects slices from each radnec 2d slice-classed volume
# and copies to a local gnet/rnet file and directory format for model training

# the 2d slice-classed labelling consists of a T/RN scalar label for each
# slice in the native AX plane. therefore there is no orthgonal or
# oblique slicing for augmentation. 

# furthermore, the images are to be masked to the tumor volume.

# the segs files for 2d slice-classing are all uncompressed, so will 
# not make a local copy of the dir. instead just work in the sync'd dbox dir. 
# this is slightly risky but all operations in this dir are just reads no
# writes so it should besafe.

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform
import skimage
import cv2
import os
import re
import nibabel as nb
from nibabel.processing import resample_from_to
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.io import imsave
import pandas as pd
import random
import glob
import copy
import platform
import json

# load a single nifti file
def loadnifti(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    if np.isnan(nb_header['scl_slope']) or np.isnan(nb_header['scl_inter']):
        img_nb_t1.header['scl_slope'] = 1
        img_nb_t1.header['scl_inter'] = 0
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(img_nb_t1.get_fdata(),axes=(2,1,0))
    # strange. getting some nan's and inf's in some of the sliceclass nifti's. 
    # setting the slope,inter as above wasn't the solution. the nan's and inf's
    # were in the source dataset.
    # just cast them all to zero.
    if True:
        img_arr_t1[np.isnan(img_arr_t1)] = 0
        img_arr_t1[np.isinf(img_arr_t1)] = 0
    if type is not None:
        if np.max(img_arr_t1) > 255:
            img_arr_t1 = (img_arr_t1 / np.max(img_arr_t1) * 255)
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine
    return img_arr_t1,affine

def load_dataset(cpath,type='t1c'):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if type in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

def load_xlsx(dboxdir):
    files = os.listdir(dboxdir)
    xlsfile = [f for f in files if f.endswith('xlsx')][0]
    df = pd.read_excel(os.path.join(dboxdir,xlsfile))
    return df


# main

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# define the top-level data dir. ie for either radnec, or brats
uname = platform.uname()
if 'dellxps' in uname.node:
    datadir = "/media/jbishop/WD4/brainmets/sunnybrook/radnec2/"
elif 'XPS-8950' in uname.node:
    datadir = "/home/jbishop/data/radnec_rnet/"
    # for radnec, a separate segmentation dir could be provided
    # segdir = os.path.join(datadir,'seg')
    segdir = "/home/jbishop/Dropbox/RADOGEN/R&D/FINAL SEGMENTATIONS"


# specify gunet, resnet processing by assigning one
# dir variable and leaving the other None
rnetdir = os.path.join(datadir,'resnet')
gnetdir = None

normalslice_flag = False # flag for outputting normal brain cross-sections

# obtain the case and finaldx information for radnec
xls = load_xlsx(segdir)
caselist = xls['ID'].tolist()
finaldx = xls['Final Dx'].tolist()
finaldx = [d.strip() for d in finaldx]

cases = {}
y = {}

casesTr,cases['test'],y_train,y['test'] = train_test_split(caselist,finaldx,stratify=finaldx,test_size=0.2,random_state=42)
cases['train'],cases['val'],y['train'],y['val'] = train_test_split(casesTr,y_train,stratify=y_train,test_size=0.2,random_state=43)

#  tally the list of study dirs per case
casestudies = {}
for c in caselist:
    casestudies[c] = []
    cdir = os.path.join(segdir,c)
    studies = sorted(os.listdir(cdir))
    if not all(s.startswith('20') for s in studies):
        casestudies[c] = ['.'] # ie images are in current dir, there is no study subdir
    else:
        casestudies[c] = studies

img_idx = 1
random.seed(42) # random numbers for normal slices
# assign numeric values for labels
lblT = 127
lblRN = 255

lblstr = {0:'RN',1:'T'}

if rnetdir: # resnet format dirs
    output_dir = {}
    for d in ['train','val','test','png']:
        output_dir[d] = os.path.join(rnetdir,d)
        try:
            shutil.rmtree(output_dir[d])
        except FileNotFoundError:
            pass
    for l in range(len(np.unique(finaldx))): 
        for d in ['train','val','test']:
            os.makedirs(os.path.join(output_dir[d],str(l)),exist_ok=True)
    os.makedirs(output_dir['png'],exist_ok=True)

elif gnetdir:
    pass # not implemented yet

# traain,val,test
for ck in cases.keys():
    print(ck)

    # arbitrary rotation for oblique slicing
    # not implemented for slice-class labelling.
    refvec = np.array([1,0,0],dtype=float)
    targetlist = [refvec]
    ntarget = len(targetlist)
    r_obl = {}
    for i,target in enumerate(targetlist): 
        r_obl[str(i)] = np.eye(3)

    for c,dx in zip(cases[ck],y[ck]):
        print('case = {}, {}'.format(c,dx))

        if False: #debugging
            if c != 'M0028':
                continue

        cdir = os.path.join(segdir,c)
        os.chdir(cdir)

        for s in casestudies[c]:

            imgs = {}
            Rimgs = {str(i):{} for i in range(ntarget)}
            Rlbls = {str(i):None for i in range(ntarget)}

            for ik in ['flair+','t1+','adc','dwi']:
                try:
                    filename = glob.glob(os.path.join(cdir,s,ik+'*zscore*'))[0]
                except IndexError:
                    raise IndexError
                imgs[ik],affine = loadnifti(os.path.split(filename)[1],os.path.join(cdir,os.path.split(filename)[0]))
                if np.any(np.isinf(imgs[ik])):
                    imgs[ik][np.where(np.isinf(imgs[ik]))] = 0
                for k in r_obl.keys():
                    center = np.array(np.shape(imgs[ik]))/2
                    offset = center - np.matmul(r_obl[k],center)
                    Rimgs[k][ik] = affine_transform(imgs[ik],r_obl[k],offset=offset,order=3)

            # load the mask
            masks = {}
            # use BLAST if indicated
            # currently spreadsheet has a blank/NaN
            if np.isnan(xls.loc[xls['ID']==c,'BLAST_ET'].values[0]):
                maskname = 'ET_mask.nii'
            else:
                maskname = 'ET_BLAST_mask.nii'
            masks['ET'],affine = loadnifti(maskname,os.path.join(cdir,s),type='uint8')
            # masks[tk][masks[tk] == 255] = 0
            # no easy way to display low contrast mask in a ping for spot verification. can't use anything
            # non-continguous like 127,255. so, for spot checking the pings can temporarily run this code using 127,255
            # but otherwise use 0,1,2 for nnunet. 

            # in 2d slice-classed labelling, the mask includes both 
            # T and/or RN and pixel by pixel information sin't given, 
            # so just using a nominal value of 255 here
            masks['lbl'] = lblRN*masks['ET']
 
            for k in r_obl.keys():
                center = np.array(np.shape(masks['lbl']))/2
                offset = center - np.matmul(r_obl[k],center)
                Rlbls[k] = affine_transform(masks['lbl'],r_obl[k],offset=offset,order=0)

            # oblique,orthogonal slices
            for k in r_obl.keys():
                if normalslice_flag: # use all brain volume slices. 
                    pset = np.where( (Rimgs[k]['t1+']) & (Rimgs[k]['flair+']) )
                else: # create only slices with a lesion cross-section.
                    pset = np.where(Rlbls[k])
                npixels = len(pset[0])
                # with slice-class label, can only do the axial slices
                dimset = [0] # 0,1,2
                for dim in dimset:
                    slices = np.unique(pset[dim])
                    for slice in slices:
                        # nnunet convention is RN=2,T=1,normal=0  
                        # for 2 classes, using RN=0, T=1

                        # for now will use non-tumor slices in tumor cases as RN slices
                        lbl = 0
                        if xls.loc[xls['ID'] == c,'Final Dx'].values[0] == 'T':                          
                            if slice in range(int(xls.loc[xls['ID']==c,'Tumor slices (start)'].values[0]),
                                                int(xls.loc[xls['ID']==c,'Tumor slices (end)'].values[0])+1):
                                lbl = 1

                        lblslice = np.moveaxis(Rlbls[k],dim,0)[slice]
                        imgslice = {}
                        rshape = tuple(np.roll(np.array(np.shape(Rimgs[k]['t1+'])),dim))[1:]
                        rslice = np.zeros((4,) + rshape,dtype='float')
                        mslice = np.zeros_like(rslice)
                        for i,ik in enumerate(['flair+','t1+','adc','dwi']):
                            imgslice[ik] = np.moveaxis(Rimgs[k][ik],dim,0)[slice]
                            rslice[i,:,:] = imgslice[ik] 
                            # mask the image with the lesion
                            mslice[i,:,:] = imgslice[ik] * (lblslice / lblRN)

                        skipslice = False
                        # exclude lesion slices with insufficient pixels. in the 2d slice-class
                        # labelling, all T slices are considered acceptable, only RN slices are 
                        # checked
                        if lbl == 0:
                            if len(np.where(lblslice)[0]) < 50:
                                skipslice = True

                        # process slice output
                        if skipslice:
                            continue
                        
                        # for a slice through the lesion
                        elif len(np.where(lblslice)[0]):
                            if rnetdir:
                                fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + s + '_' + str(slice) + '_' + 'r' + k
                                with open(os.path.join(output_dir[ck],str(lbl),fname),'wb') as fp:
                                    fp.write(mslice)

                            # create test output pngs
                            if True:
                                lbl_ros = np.where(lblslice)

                                lbl_img = np.ones(np.shape(rslice)[1:]+(4,))
                                lbl_img[:,:,:3] = np.moveaxis(mslice[:3,:,:],0,2)/np.max(mslice[:3,:,:])
                                lbl_ovly = np.ones_like(lbl_img)
                                nslice = ( rslice[0]-np.min(rslice[0])) / (np.max(rslice[0])-np.min(rslice[0])) 
                                lbl_ovly[:,:,:3] = np.stack([nslice]*3,axis=-1)
                                lbl_ovly[lbl_ros] = colors.to_rgb(defcolors[0]) +(0.5,)

                                ovly = np.concat((lbl_ovly,lbl_img),axis=1)
                                ovly  = (ovly*255).astype('uint8')
                                oname = 'ovly_' + str(img_idx).zfill(6) + '_' + c + '_' + s + '_' + str(slice) + '_' + 'r' + k + '.png'
                                cv2.putText(ovly, lblstr[lbl], (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                if rnetdir:
                                    cv2.imwrite(os.path.join(output_dir['png'],oname),cv2.cvtColor(ovly,cv2.COLOR_BGRA2RGBA))
                                # plt.imshow(lbl_ovly)
                                # plt.xticks([]),plt.yticks([])
                                # plt.text(10,10,lbl,c='w')
                                # plt.savefig(os.path.join(output_lbldir,fname),bbox_inches='tight',pad_inches=0)
                                # imsave(os.path.join(output_lbldir,fname),lbl_ovly,check_contrast=False)
                                a=1

                        # output a normal slice. not coded yet.
                        elif normalslice_flag:
                            if random.random() < 0.05:
                                # don't need all the normal slices just a few
                                for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                                    fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '_' + 'r' + k + '.png'
                                    imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                                with open(os.path.join(output_lbldir,lblfname),'w') as fp:
                                    json.dump({'dx':lbl},fp) 
                            else:
                                continue
                        else:
                            continue

                        img_idx += 1

a=1
