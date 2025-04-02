# script selects slices from each radnec segmented volume
# and copies to a nnunet file and directory format for model training

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

def load_dataset(cpath,type='t1c'):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if type in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

# main

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

if os.name == 'posix':
    uname = platform.uname()
    if 'dellxps' in uname.node:
        datadir = "/media/jbishop/WD4/brainmets/sunnybrook/radnec2/"
    elif 'XPS-8950' in uname.node:
        datadir = "/home/jbishop/data/radnec2/"
else:
    datadir = "D:\\data\\radnec2\\"

segdir = os.path.join(datadir,'seg')
nnunetdir = os.path.join(datadir,'nnUNet_raw','Dataset140_RadNecClassify')

normalslice_flag = True # flag for outputting normal brain cross-sections

caselist = os.listdir(os.path.join(datadir,'seg'))
cases = {}

# pre-read the segmentation dir to tally 
# T vs RN. this is needed to stratify the train/test split
# this is done at the case level regardless of how many studies per case and lesions per study there are
# for the best possible train/test results.
tally = {}
for c in caselist:
    tally[c] = {}
    cdir = os.path.join(segdir,c)
    os.chdir(cdir)
    # filename = glob.glob(os.path.join('**','mask_T_*.nii*'))
    dirdict = {'dir':None,'T':[],'TC':[]}
    tally[c] = {'dx':0,'dirs':[]}
    for root,dirs,files in os.walk(cdir,topdown=False):
        # print('{}\n{}\n{}'.format(root,dirs,files))
        tcfiles = sorted([f for f in files if re.search('mask_TC',f)])
        if len(tcfiles):
            tally[c]['dirs'].append(copy.deepcopy(dirdict))
            tally[c]['dirs'][-1]['dir'] = root
            tally[c]['dirs'][-1]['TC'] = tcfiles
            tfiles = sorted([f for f in files if re.search('mask_T[^C\']',f)])
            if len(tfiles): # ie a single T mask amongst all studies
                tally[c]['dirs'][-1]['T'] = tfiles
                tally[c]['dx']=1
            else:
                tally[c]['dirs'][-1]['T'] = [None]*len(tcfiles)

finaldx = []
casesdx = []
for k,v in tally.items():
    finaldx.append(tally[k]['dx'])
    casesdx.append(k)
cases['casesTr'],cases['casesTs'],y_train,y_test = train_test_split(casesdx,finaldx,stratify=finaldx,test_size=0.2,random_state=42)

img_idx = 1
random.seed(42) # random numbers for normal slices
# assign numeric values for labels
lblT = 127
lblRN = 255

for ck in cases.keys():
    print(ck)

    output_imgdir = os.path.join(nnunetdir,ck.replace('cases','images'))
    output_lbldir = os.path.join(nnunetdir,ck.replace('cases','labels'))
    try:
        shutil.rmtree(output_imgdir)
        shutil.rmtree(output_lbldir)
    except FileNotFoundError:
        pass
    os.makedirs(output_imgdir,exist_ok=True)
    os.makedirs(output_lbldir,exist_ok=True)

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

    for c in cases[ck]:
        print('case = {}'.format(c))

        if False: #debugging
            if c != 'M0012':
                continue

        cdir = os.path.join(segdir,c)
        os.chdir(cdir)

        for ddict in tally[c]['dirs']:

            imgs = {}
            Rimgs = {str(i):{} for i in range(ntarget)}
            Rlbls = {str(i):None for i in range(ntarget)}
            study = os.path.split(ddict['dir'])[1]

            for ik in ['flair+','t1+']:
                filename = glob.glob(os.path.join(ddict['dir'],ik+'_processed*'))[0]
                imgs[ik],affine = loadnifti(os.path.split(filename)[1],os.path.join(cdir,os.path.split(filename)[0]),type='uint8')
                assert np.max(imgs[ik] == 255)
                for k in r_obl.keys():
                    center = np.array(np.shape(imgs[ik]))/2
                    offset = center - np.matmul(r_obl[k],center)
                    Rimgs[k][ik] = affine_transform(imgs[ik],r_obl[k],offset=offset,order=3)

            for tc_file,t_file in zip(ddict['TC'],ddict['T']):
                masks = {}
                # format for a lesion number is currently hard-coded here, underscore plus single digit
                lesion = re.search('_([1-9])',tc_file)
                if lesion:
                    lesion = lesion.group(1)
                else:
                    lesion = '1'
                masks['TC'],affine = loadnifti(tc_file,ddict['dir'],type='uint8')
                if t_file:
                    masks['T'],_ = loadnifti(t_file,ddict['dir'],type='uint8')
                else:
                    masks['T'] = np.zeros_like(masks['TC'])
                # masks[tk][masks[tk] == 255] = 0
                # no easy way to display low contrast mask in a ping for spot verification. can't use anything
                # non-continguous like 127,255. so, for spot checking the pings can temporarily run this code using 127,255
                # but otherwise use 0,1,2 for nnunet. 

                # check for error pixels. according to convention, 'T' should be entirely 
                # a subset of 'TC'
                errpixels = np.where(masks['TC'].astype(int) - masks['T'].astype(int) < 0)[0]
                if len(errpixels):
                    masks['TC'] = masks['TC'] | masks['T']
                    print('error mask pixels detected, correcting...')
                masks['lbl'] = lblT*masks['T'] + lblRN*(masks['TC'] - masks['T'])
                if np.any(masks['lbl'] > lblRN):
                    raise ValueError

                for k in r_obl.keys():
                    center = np.array(np.shape(masks['lbl']))/2
                    offset = center - np.matmul(r_obl[k],center)
                    Rlbls[k] = affine_transform(masks['lbl'],r_obl[k],offset=offset,order=0)

                # oblique,orthogonal slices
                # for rtag in ['R','']:
                for k in r_obl.keys():
                    if normalslice_flag: # use all brain volume slices. since brains are not extracted
                        # use estimate to exclude the air background
                        noise = np.max(imgs['t1+'][:,0,0])
                        pset = np.where( (Rimgs[k]['t1+'] > 10*noise) & (Rimgs[k]['flair+'] > 10*noise) )
                    else: # create only slices with a lesion cross-section.
                        pset = np.where(Rlbls[k])
                    npixels = len(pset[0])
                    for dim in range(3):
                        slices = np.unique(pset[dim])
                        for slice in slices:
                            imgslice = {}
                            lblslice = np.moveaxis(Rlbls[k],dim,0)[slice]
                            if np.max(lblslice) > lblRN:
                                raise ValueError
                            # nnunet convention is RN=2,T=1,normal=0                            
                            lblset = set(np.unique(lblslice)) 
                            if lblset == {0}:
                                lbl = [0, 0] # normal slice
                            elif lblset == {0, lblT, lblRN}:
                                lbl = [1, 1] # both
                            else:  # Check individual RN=2 T=1
                                lbl = [int(lblT in lblset), int(lblRN in lblset)]                           

                            for ik in ('flair+','t1+'):
                                imgslice[ik] = np.moveaxis(Rimgs[k][ik],dim,0)[slice]

                            if normalslice_flag:
                                # cv2.imwrite(os.path.join(output_lbldir,fname),lblslice)
                                lblfname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '.json'
                                # output a tumor slice. check minimum number of pixels in each label compartment, and total.
                                skipslice = False
                                for l in lblset:
                                    if len(np.where(lblslice==l)[0]) < 20:
                                        skipslice = True
                                if skipslice:
                                    continue
                                elif len(np.where(lblslice)[0]) > 49:
                                    for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '_' + 'r' + k + '.png'
                                        imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                                    with open(os.path.join(output_lbldir,lblfname),'w') as fp:
                                        json.dump({'dx':lbl},fp)   

                                    # create test output pngs
                                    if True:
                                        lbl_ros = np.where(lblslice)
                                        lbl_rost = np.where(lblslice == lblT)
                                        lbl_rosrn = np.where(lblslice == lblRN)

                                        lbl_ovly = skimage.color.gray2rgba(np.copy(imgslice['flair+']))/255
                                        lbl_ovly[lbl_rost] = colors.to_rgb(defcolors[0]) +(0.5,)
                                        lbl_ovly[lbl_rosrn] = colors.to_rgb(defcolors[1]) +(0.5,)
                                        lbl_ovly  = (lbl_ovly*255).astype('uint8')
                                        fname = 'ovly_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + 'r' + k + '.png'
                                        fname = 'ovly_' + str(img_idx).zfill(6) + '_' + c + '.png'
                                        cv2.putText(lbl_ovly, str(lbl), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                        cv2.imwrite(os.path.join(output_lbldir,fname),cv2.cvtColor(lbl_ovly,cv2.COLOR_BGRA2RGBA))
                                        # plt.imshow(lbl_ovly)
                                        # plt.xticks([]),plt.yticks([])
                                        # plt.text(10,10,lbl,c='w')
                                        # plt.savefig(os.path.join(output_lbldir,fname),bbox_inches='tight',pad_inches=0)
                                        # imsave(os.path.join(output_lbldir,fname),lbl_ovly,check_contrast=False)
                                        a=1

                                elif len(np.where(lblslice)[0]) > 0: # don't want to use these cross-sections at all.
                                    continue
                                # output a normal slice
                                elif random.random() < 0.05:
                                    # don't need all the normal slices just a few
                                    for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '_' + 'r' + k + '.png'
                                        imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                                    with open(os.path.join(output_lbldir,lblfname),'w') as fp:
                                        json.dump({'dx':lbl},fp)                                    
                                else:
                                    continue


                            else: # no normal slices. not updated.
                                if len(np.where(lblslice)[0]) > 49:
                                    fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '.png'
                                    imsave(os.path.join(output_lbldir,fname),lblslice,check_contrast=False)
                                    # cv2.imwrite(os.path.join(output_lbldir,fname),lblslice)
                                    for ktag,ik in zip(('0003','0001'),(rtag+'flair+',rtag+'t1+')):
                                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + study + '_' + str(slice) + '_' + lesion + '_' + ktag + '.png'
                                        imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)

                            img_idx += 1

a=1
