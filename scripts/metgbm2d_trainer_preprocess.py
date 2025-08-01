# script selects slices from each brats 3d segmented volume
# and copies to a gnet file and directory format for 2d classification model training

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform
from scipy import stats
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
import cc3d
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
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(img_nb_t1.get_fdata(),axes=(2,1,0))
    if type is not None:
        if np.max(img_arr_t1) > 255:
            img_arr_t1 = (img_arr_t1 / np.max(img_arr_t1) * 255)
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine
    return img_arr_t1,affine

# write a single nifti file. use uint8 for masks 
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    if norm:
        img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
    # using nibabel nifti coordinates
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    if True:
        os.system('gzip --force "{}"'.format(filename))


def load_dataset(cpath,type='t1c'):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if type in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

# main

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
onehot=True

# define the top-level data dir. ie for either radnec, or brats
uname = platform.uname()
if 'dellxps' in uname.node:
    datadir = "/media/jbishop/WD4/brainmets/sunnybrook/radnec2/"
elif 'XPS-8950' in uname.node:
    datadir = "/home/jbishop/data/met-gbm/"

gnetdir = os.path.join(datadir,'gnet')
metdir = os.path.join(datadir,'brats-met')
gbmdir = os.path.join(datadir,'brats-2021')
ftype = {'met':['t1c','t2f','t1n'],'gbm':['t1ce','flair','t1']}
iset = ['t1+','flair+','t1']

normalslice_flag = False # flag for outputting normal brain cross-sections

# assemble the case and finaldx information for met-gbm
# the original 3d datasets from brats are already separated into met and gbm classes
# since they were different brats competitions. 

gbmcaselist = os.listdir(os.path.join(gbmdir,'raw'))
metcaselist = os.listdir(os.path.join(metdir,'raw'))
caselist = gbmcaselist + metcaselist
finaldx = [0]*len(gbmcaselist) + [1]*len(metcaselist)
dirlist = [gbmdir]*len(gbmcaselist) + [metdir]*len(metcaselist)
cases = {}
y = {}
decimate = True

random.seed(42) # random for reproduceability

if decimate:
    dset = np.array(sorted(random.sample(range(0,len(caselist)),int(len(caselist)/20))))
    caselist = [caselist[d] for d in dset]
    finaldx = [finaldx[d] for d in dset]

casesTr,cases['test'],y_train,y['test'] = train_test_split(caselist,finaldx,stratify=finaldx,test_size=0.2,random_state=42)
cases['train'],cases['val'],y['train'],y['val'] = train_test_split(casesTr,y_train,stratify=y_train,test_size=0.2,random_state=43)

img_idx = 1
# assign numeric values for labels
# conventions for brats
lbl_ET = 3
lbl_WT = 1
# convetion for classification
lblGBM = 127
lblMET = 255

outputdir = {'train':os.path.join(gnetdir,'train'),
             'test':os.path.join(gnetdir,'test'),
             'val':os.path.join(gnetdir,'val'),
             'png':os.path.join(gnetdir,'png')}

try:
    for k in outputdir.keys():
        shutil.rmtree(outputdir[k])
except FileNotFoundError:
    pass
for l in range(len(np.unique(finaldx))): 
    for k  in ['train','test','val']:
        os.makedirs(os.path.join(outputdir[k],str(l)),exist_ok=True)
os.makedirs(outputdir['png'],exist_ok=True)

for ck in ['train','val','test']:
    print(ck)

    # arbitrary rotation for oblique slicing
    refvec = np.array([1,0,0],dtype=float)
    targetlist = [refvec,np.ones((3))/np.sqrt(3),np.array([1,-1,1])/np.sqrt(3)]
    ntarget = len(targetlist)
    r_obl = {}
    for i,target in enumerate(targetlist): 
        Rangle = np.arccos(np.dot(refvec,target))
        if np.abs(Rangle) != 0.0:
            Raxis = np.cross(refvec,target)
            Raxis /= np.linalg.norm(Raxis)
            skewsim_matrix = np.array([[0,-Raxis[2],Raxis[1]],[Raxis[2],0,-Raxis[0]],[-Raxis[1],Raxis[0],0]])
            # R = np.eye(3) + np.sin(Rangle) * skewsim_matrix + (1-np.cos(Rangle)) * np.matmul(skewsim_matrix,skewsim_matrix)
            r_obl[str(i)] = Rotation.from_rotvec(Rangle*Raxis,degrees=False).as_matrix()
        else:
            r_obl[str(i)] = np.eye(3)

    for c,dx in zip(cases[ck],y[ck]):
        print('case = {}, {}'.format(c,dx))

        if False: #debugging
            if c != 'M0012':
                continue

        if 'MET' in c:
            kk = 'met'
            cdir = os.path.join(metdir,'raw',c)
        else:
            kk = 'gbm'
            cdir = os.path.join(gbmdir,'raw',c)
        os.chdir(cdir)

        imgs = {}
        segs = {}
        Rimgs = {str(i):{} for i in range(ntarget)}
        Rlbls = {str(i):None for i in range(ntarget)}

        for ik,fkey in zip(iset,ftype[kk]):
            try:
                filename = glob.glob('*' + fkey + '*')[0]
                segname = glob.glob('*seg*')[0]
            except IndexError:
                continue

            imgs[ik],affine = loadnifti(filename,cdir,type='float32')
            assert np.max(imgs[ik] <= 255.0)
            if ik == 't1+':
                imgs['seg'] = np.zeros_like(imgs[ik],dtype='uint8')
                seg,_ = loadnifti(segname,cdir,type='uint8')
                if kk == 'gbm':
                    seg[seg==4] = 3 # match brats 2021 to brats 2023 convention
                seg_mask = np.zeros_like(seg)
                seg_mask[seg>0] = 1
                clabels = cc3d.connected_components(seg_mask,connectivity=6)
                lesions = cc3d.dust(clabels,connectivity=6,threshold=50)
                # select median lesion to work with 
                lvals = sorted([(ll,len(lesions[lesions==ll])) for ll in range(1,np.max(lesions)+1)],key = lambda x:x[1])
                if len(lvals) == 1:
                    mll = lvals[0][0]
                else:
                    if len(lvals) % 2 == 0:
                        lvals = lvals[1:] # discard smallest lesion
                    mlength = np.median([l for _,l in lvals]) 
                    mll = [ll for ll,l in lvals if l == mlength][0]
                lesions[lesions != mll] = 0
                seg[lesions==0] = 0
                if False:
                    writenifti(seg,os.path.join(output_pngdir,c+'_seg.nii'),type='uint8',affine=affine)
                    writenifti(lesions,os.path.join(output_pngdir,c+'_lesions.nii'),type='uint8',affine=affine)

                if kk == 'met':
                    imgs['seg'][seg==lbl_ET] = lblMET
                elif kk == 'gbm':
                    imgs['seg'][seg==lbl_ET] = lblGBM
            for k in r_obl.keys():
                center = np.array(np.shape(imgs[ik]))/2
                offset = center - np.matmul(r_obl[k],center)
                Rimgs[k][ik] = affine_transform(imgs[ik],r_obl[k],offset=offset,order=3)
                Rlbls[k] = affine_transform(imgs['seg'],r_obl[k],offset=offset,order=0)

        # oblique,orthogonal slices
        for k in r_obl.keys():
            if normalslice_flag: # use all brain volume slices. since brains are not extracted
                # use estimate to exclude the air background
                noise = np.max(imgs['t1+'][:,0,0])
                pset = np.where( (Rimgs[k]['t1+'] > 20*noise) & (Rimgs[k]['flair+'] > 20*noise) )
            else: # create only slices with a lesion cross-section.
                pset = np.where(Rlbls[k])
            npixels = len(pset[0])
            for dim in range(3):
                slices = np.unique(pset[dim])
                for slice in slices:
                    imgslice = {}
                    lblslice = np.moveaxis(Rlbls[k],dim,0)[slice]
                    if np.max(lblslice) > lblMET:
                        raise ValueError
                    # nnunet convention is RN=2,T=1,normal=0  
                    # for 2 classes, using RN=0, both=1                          
                    lblset = set(np.unique(lblslice)) 
                    if onehot:
                        if lblset == {0}:
                            continue
                            lbl = [0] # normal slice
                        elif lblset == {0, lblMET}:  
                            lbl = [1] # met
                        elif lblset == {0, lblGBM}:
                            lbl = [0] # gbm            

                    for ik in iset:
                        imgslice[ik] = np.moveaxis(Rimgs[k][ik],dim,0)[slice]

                    # output a tumor slice. check minimum number of pixels in each label compartment, and total.
                    lblfname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + str(slice) + '.json'
                    skipslice = False
                    for l in lblset:
                        if len(np.where(lblslice==l)[0]) < 20:
                            skipslice = True
                    if skipslice:
                        continue
                    elif len(np.where(lblslice)[0]) > 49:
                        rslice = np.zeros(imgslice[ik].shape+(3,),dtype='uint8')
                        for i,ik in enumerate(iset):
                            rslice[:,:,i] = imgslice[ik]
                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + str(slice) + '_' + 'r' + k + '.png'
                        imsave(os.path.join(outputdir[ck],str(lbl[0]),fname),rslice,check_contrast=False)

                        # create test output pngs
                        if True:
                            if random.random() <= 0.05:
                                lbl_ros = np.where(lblslice)
                                lbl_rosgbm = np.where(lblslice == lblGBM)
                                lbl_rosmet = np.where(lblslice == lblMET)

                                lbl_img = skimage.color.gray2rgba(np.copy(imgslice['t1+'])/255)
                                lbl_ovly = np.copy(lbl_img)
                                lbl_ovly[lbl_rosgbm] = colors.to_rgb(defcolors[0]) +(0.5,)
                                lbl_ovly[lbl_rosmet] = colors.to_rgb(defcolors[1]) +(0.5,)

                                lbl_ovly = np.concat((lbl_ovly,lbl_img),axis=1)
                                lbl_ovly  = (lbl_ovly*255).astype('uint8')
                                fname = 'ovly_' + str(img_idx).zfill(6) + '_' + c + '_' + str(slice) + '_' + 'r' + k + '.png'
                                cv2.putText(lbl_ovly, str(lbl), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                                cv2.imwrite(os.path.join(outputdir['png'],fname),cv2.cvtColor(lbl_ovly,cv2.COLOR_BGRA2RGBA))
                                # plt.imshow(lbl_ovly)
                                # plt.xticks([]),plt.yticks([])
                                # plt.text(10,10,lbl,c='w')
                                # plt.savefig(os.path.join(output_lbldir,fname),bbox_inches='tight',pad_inches=0)
                                # imsave(os.path.join(output_lbldir,fname),lbl_ovly,check_contrast=False)
                                a=1

                    elif len(np.where(lblslice)[0]) > 0: # don't want to use these cross-sections at all.
                        continue
                    # output a normal slice. not coded yet.
                    # elif normalslice_flag:
                    #     if random.random() < 0.05:
                    #         # don't need all the normal slices just a few
                    #         for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                    #             fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '_' + 'r' + k + '.png'
                    #             imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                    #         with open(os.path.join(output_lbldir,lblfname),'w') as fp:
                    #             json.dump({'dx':lbl},fp) 
                    #     else:
                    #         continue
                    else:
                        continue


                    # else: # no normal slices. not updated.
                    #     if len(np.where(lblslice)[0]) > 49:
                    #         fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '.png'
                    #         imsave(os.path.join(output_lbldir,fname),lblslice,check_contrast=False)
                    #         # cv2.imwrite(os.path.join(output_lbldir,fname),lblslice)
                    #         for ktag,ik in zip(('0003','0001'),(rtag+'flair+',rtag+'t1+')):
                    #             fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '.png'
                    #             imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)

                    img_idx += 1

a=1
