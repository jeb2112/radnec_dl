# quick script to parse central UA slices from the DQN sag 3D volumes
# and labels from .json params 

import argparse
import os
import glob
import sys
import pandas as pd
from struct import pack
import numpy as np
import itertools
import functools,operator
import json
import re
import nibabel as nib
import pydicom
import scipy
import scipy.misc
from scipy.interpolate import griddata
from scipy.ndimage import zoom,rotate
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.image
import operator
sys.path.append('/home/jbishop/src/ua_align/dl') # kludge until package import fixed
from datalake import DataLake

IMRES = 256
NSLICE = 5
zNSLICE = NSLICE
DOF = 3

def crop_img(I,zfactor,croporigin,cropshape): # zfactor is a arbitrary resampling factor
    I = scipy.ndimage.zoom(I,zfactor,mode='nearest')
    i1 = tuple(map(lambda x,delta : x-delta//2, croporigin, cropshape))
    i2 = tuple(map(operator.add, i1, cropshape  ))
    crop = tuple(map(slice,i1,i2))
    I = I[crop]
    I = I.copy(order='C')
    return I

def rot_img(img, angle, origin):
    imsize = np.shape(img)
    X = [imsize[1] - origin[0], origin[0]]
    Y = [imsize[0] - origin[1], origin[1]]
    img_padded = np.pad(img, [Y, X], 'constant')
    img_rotated = rotate(img_padded, angle, reshape=False)
    img_rotated  = img_rotated[Y[0]:-Y[1],X[0]:-X[1]]
    return img_rotated

# return list of source filenames from list of numbered .json files
def get_filelist(lblfiles):
    nfile = len(lblfiles)
    lblfile=[]
    for n in range(0,nfile):
        lbl = json.load(open(lblfiles[n]))
        lfile = re.sub(r'^.*?MRI\sTest\sData(\\\.)?','',lbl['sourcefile'])
        lblfile.append(lfile)
    return lblfile

# find list of unique elements comparing to a previouslist
def get_uniquelist(newfiles,oldfiles):
    newfilenames = get_filelist(newfiles)
    oldfilenames = get_filelist(oldfiles)
    newfiles = [f for (f,fname) in list(zip(newfiles,newfilenames)) if fname not in oldfilenames]
    return newfiles

# collate coord info from directory of .json files into a lbl.dat file in same dir
def write_lbls(lbldir):
    lbl_files = sorted([os.path.join(fpath,f) for (fpath,_,fnames) in os.walk(lbldir) for f in fnames if f.endswith('json')])
    n_lbls = len(lbl_files)
    lbls = np.empty((DOF,n_lbls))
    for n in range(0,n_lbls):
        lbl = json.load(open(lbl_files[n]))
        coords = np.array(lbl['coords'])
        dx = -np.diff(coords[:,1]) 
        dy = np.diff(coords[:,0]) # for pixel convention
        lbls[0,n] = np.mean(coords[:,1]) # xcoord
        lbls[1,n] = np.mean(coords[:,0]) # ycoord
        lbls[2,n] = np.arctan2 (dy,dx) * 180/np.pi
    lbls = np.reshape(np.tile(lbls,(NSLICE,1)),(n_lbls*NSLICE*DOF,),order='F') # duplicate across slice dim
    formatstr = '<{:d}f'.format(n_lbls*NSLICE*DOF)
    b_angles = pack(formatstr,*lbls) 
    with open(os.path.join(lbldir,'lbl.dat'),'wb') as fb: 
        fb.write(b_angles)
        fb.close()


# returns dicom px to mm coordinate transform matrix
def dcm_matrix(pos,orient,deltaxy,deltaz,table,model):
    # dirz cosine vector could be RHS or LHS depending on imaging sequence logical to physical
    # currently, series are sorted by SliceLocation, reverse=True
    # in this covention, by inspection for siemens AX, there is a flip from RHS.
    # TODO: find better tag or work this out properly
    M = np.zeros((4,4))
    M[0:3,0] = orient[0,:].T * deltaxy
    M[0:3,1] = orient[1,:].T * deltaxy
    if table in ['AX'] and model not in ['Ingenia','Ingenia Elition X']:
        M[0:3,2] = np.cross(orient[1,:],orient[0,:]) * (deltaz)
    else:
        M[0:3,2] = np.cross(orient[0,:],orient[1,:]) * (deltaz)
    M[0:3,3] = pos.T
    M[3,3] = 1
    return M

# any simple normalizations to be attempted go here
def do_norm(img):
    # background norm of the volume
    if True:
        nhist,nedge = np.histogram(img,bins=10)
        # check for cluster of bad high magnitude pixels, eg from 001-068
        while True:
            if 0 in nhist:
                maxval_est = nedge[-2]
                nhist,nedge = np.histogram(img,bins=10,range=(0,maxval_est))
                maxbin = np.where(nhist == np.max(nhist[1:]))[0][0]
            else:
                maxbin = np.where(nhist == np.max(nhist[1:]))[0][0]
                break
        mscale = np.mean((nedge[maxbin:maxbin+2]))
        img = img / mscale
    # simple norm to 0,1
    else:
        img = img / np.max(img)
    return img

# set the origin, crop 3d volumes, write out to file and return ua centre in-plane i,j pixel coords in the cropped images.
# img_idx is a counter for output file numbering
def do_lbls_crops(img,ua_centre_1mm_ijk,px,img_idx,output_imgdir,mixup=True,tag=None):
    label = np.zeros(2,)
    ozcrop = ua_centre_1mm_ijk[2]
    # calculate xy crop coords to account for resampling to 256 equivalent full matrix at 1mm/pixel
    if True: # crop image origin. correct for pixel scale
        oxcrop = int((np.shape(img)[0]*px)//2)
        oycrop = int((np.shape(img)[0]*px)//2)
    else: # crop UA xy origin
        oxcrop = ua_centre_1mm_ijk[0]
        oycrop = ua_centre_1mm_ijk[1]

    xcrop,ycrop = 3*IMRES//8,3*IMRES//4 # hardcoded. note yx still here
    label[0] = ua_centre_1mm_ijk[0] - (oxcrop-xcrop/2)
    label[1] = ua_centre_1mm_ijk[1] - (oycrop-ycrop/2)

    zcrop = range(ozcrop-NSLICE//2,ozcrop+NSLICE//2+1)
    img_zcrop = img[:,:,zcrop]
    if mixup: # z-interpolate for additional training images 
        f_zset = scipy.interpolate.interp1d(np.arange(0,NSLICE),img_zcrop,axis=2)
        zcrop = np.arange(0.5,NSLICE-1,0.5) # drop two edges slicwes in this case
        img_zcrop = f_zset(zcrop)

    for z in range(len(zcrop)):
        z_resample = img_zcrop[:,:,z] # display convention
        z_resample = crop_img(z_resample,px,(oycrop,oxcrop),(xcrop,ycrop) )
        if False: # mixed result with histogram norm
            z_resample = np.where(z_resample<0.0,0.0,z_resample)
            z_resample = np.where(z_resample>1.0,1.0,z_resample)
            z_resample = exposure.equalize_adapthist(z_resample,clip_limit=.03)
        filename = 'I{:05d}'.format(img_idx)
        with open(os.path.join(output_imgdir,filename),'wb') as fo:
            fo.write(z_resample)
            fo.close()
        if tag is not None:
            filename = filename + tag
        matplotlib.image.imsave(os.path.join(output_imgdir,'png',filename+'.png'),z_resample)
        img_idx = img_idx + 1
    a=1
    return label,img_idx


######
# main
######

def main(inputDir,outputDir,ycrop=2,action=None,exclude=False,mixup=True,lbldir='lbl',imgdir='img',index=None):
    output_lbldir = os.path.join(outputDir,lbldir)
    output_imgdir = os.path.join(outputDir,imgdir)
 
    # re write the lbl.dat output file.
    if action == 'write_lbls':
        write_lbls(output_lbldir)

    # based on the sql db that replaces the .nii and .json from matlab
    # process full db for data sets. eventually this method should allow for incremental updates
    elif action == 'add_dl':
        output_lbldir = os.path.join(outputDir,'lbl_dl')
        output_imgdir = os.path.join(outputDir,'img_dl')
        imglist = glob.glob(output_imgdir+'/I*')
        for i in imglist:
            os.remove(i)
        output_pngdir = os.path.join(output_imgdir,'png')
        dl = DataLake('dl',localdir=inputDir)
        ndl = dl.get_N()

        if index is not None:
            start_idx = index - 1
            end_idx = index
        else:
            pnglist = glob.glob(output_pngdir+'/*.png')
            for p in pnglist:
                os.remove(p)
            start_idx = 0
            end_idx = ndl

        lbls = []
        img_idx = 0
        # loop through all records in db, or just one for debugging
        for r in range(start_idx,end_idx):
            lbl = [0,0,0]
            # get ax volume for coords
            try:
                record_ax = dl.get_record(r,'AX')
            except KeyError as e:
                print(e)
                continue
            if record_ax['comment']: # skip any commented record
                continue
            # with perfect alignment, UA is at origin of AxT2
            row_index = float(record_ax['rows'])/2.0
            col_index = float(record_ax['cols'])/2.0
            # pixel coords of UA centre point, 1st and 12th slice
            centre1_ij = np.array([col_index,row_index,0,1]) # note dicom def here
            centre2_ij = np.array([col_index,row_index,11,1]) # 12th slice hard-coded here

            orient = np.reshape(np.array(list(map(float,record_ax['orient'].split(',')))),(2,3))
            pos = np.array(list(map(float,record_ax['slicepos'].split(','))))
            px = float(record_ax['px'])
            M_ax = dcm_matrix(pos,orient,px,5,'AX',record_ax['model']) # 5mm hard-coded for Ax T2 here

            # patient coords (mm) of UA centre points
            centre_mm = np.zeros((2,3))
            centre_mm[0,:] = np.reshape(np.matmul(M_ax,centre1_ij)[0:3],(1,3))
            centre_mm[1,:]= np.reshape(np.matmul(M_ax,centre2_ij)[0:3],(1,3))
            # for the two conventions, sort on the z coordinate to get a consistent angle
            centre_mm = centre_mm[centre_mm[:,2].argsort()]

            pcs_mm_11 = np.matmul(np.linalg.inv(M_ax),[0,0,11,1])
            delta_mm = np.diff(centre_mm,axis=0)
            angle_zy_ax = np.arctan2(delta_mm[0,1],delta_mm[0,2]) * 180/np.pi
            # record zy angle for 3 dof here, position values depend on size of sag crop below
            lbl[2] = angle_zy_ax

            # get sag volume
            # don't have proper indexing consitent bewteen both tables so look up sag from ax site/case
            try:
                record_sag = dl.get_record(r,'SAG',record_ax['site'],case=record_ax['caseid'])
            except KeyError as e:
                print(e)
                continue
            if record_sag['comment']: # skip any incomplete record
                continue
            orient = np.reshape(np.array(list(map(float,record_sag['orient'].split(',')))),(2,3))
            pos = np.array(list(map(float,record_sag['slicepos'].split(','))))
            px = float(record_sag['px'])
            pz = float(record_sag['slicethk'])
            M_sag = dcm_matrix(pos,orient,px,pz,'SAG',record_sag['model'])
            M_sag_1mm =dcm_matrix(pos,orient,1,pz,'SAG',record_sag['model'])

            vol = dl.get_dbvol('SAG')
            img = np.zeros((vol[0].Rows,vol[0].Columns,len(vol)))
            for v in range(len(vol)):
                img[:,:,v] = vol[v].pixel_array 
            img = np.transpose(img,(1,0,2)) #dicom convention displays 1st dim as rows, amke 1st dim column here
            img = do_norm(img)

            # coords for selecting a cropped 3d volume
            # UA centre point in patient coords, mm, from Ax T2
            ua_centre_mm = np.append(np.mean(centre_mm,axis=0),[1])
            # convert to pixel coords in sag
            ua_centre_ijk = np.array(list(map(int,np.matmul(np.linalg.inv(M_sag),ua_centre_mm)[0:3])))
            ua_centre_1mm_ijk = np.array(list(map(int,np.matmul(np.linalg.inv(M_sag_1mm),ua_centre_mm)[0:3])))
            print('dl index {}'.format(r+1), ua_centre_ijk, ua_centre_1mm_ijk)

            # optional rotation for sag if single oblique
            angle_zy_sag = np.arctan2(orient[1,1],orient[0,1]) * 180/np.pi
            if np.abs(angle_zy_sag) > 1:
                img_rot = np.zeros(np.shape(img))
                for s in range(np.shape(img)[2]):
                    img_rot[:,:,s] = rot_img(img[:,:,s],-angle_zy_sag,ua_centre_ijk[1::-1])
                if False:
                    plt.subplot(1,2,1)
                    plt.imshow(img[:,:,35])
                    plt.plot(ua_centre_ijk[1],ua_centre_ijk[0],'r+')
                    plt.subplot(1,2,2)
                    plt.imshow(img_rot[:,:,35])
                    plt.plot(ua_centre_ijk[1],ua_centre_ijk[0],'r+')
                    plt.show()
                img = img_rot

            # index argument is for debugging one index item from dl at a time
            if index:
                return
            else:
                # crop volumes and write out to 2d image files.
                # lbls are saved separately in one single file
                tag = '-site{}_case{}'.format(record_sag['site'],record_sag['caseid'])
                lbl[0:2],img_idx = do_lbls_crops(img, ua_centre_1mm_ijk,px,img_idx,output_imgdir,mixup=True,tag=tag)
                lbls.append(lbl)

        lbls = np.asarray(lbls).T
        ndl = np.shape(lbls)[1]
        if mixup:
            zNSLICE = 2*NSLICE-3
        lbls = np.reshape(np.tile(lbls,(zNSLICE,1)),(ndl*zNSLICE*DOF,),order='F') # duplicate across slice dim
        formatstr = '<{:d}f'.format(ndl*zNSLICE*DOF)
        b_lbls = pack(formatstr,*lbls) 
        with open(os.path.join(output_lbldir,'lbl.dat'),'wb') as fb:     # note file append here.
            fb.write(b_lbls)
            fb.close()

    # combine new data into existing data pool
    # based on the .nii and .json files that were originally created in matlab for gel test and clinical data on the network during 2019 effort.
    elif action == 'add_data':

        if mixup:
            zNSLICE = 2*NSLICE-3

       # load the lbls from the .json format to get the necessary z slice coords
        existing_lbl_files = sorted([os.path.join(fpath,f) for (fpath,_,fnames) in os.walk(output_lbldir) for f in fnames if f.endswith('json')])
        new_lbl_files = sorted([os.path.join(fpath,f) for (fpath,_,fnames) in os.walk(inputDir) for f in fnames if f.endswith('json')])
        # discard any duplicates
        new_lbl_files = get_uniquelist(new_lbl_files,existing_lbl_files)
        n_new = len(new_lbl_files)
        n_old = len(existing_lbl_files)
        angles = np.empty(n_new)
        xc = np.empty(n_new)
        yc = np.empty(n_new)
        lbls = np.empty((DOF,n_new))

        # counters for re-numbering new input files into a cumulative consecutive order
        img_idx = n_old*zNSLICE
        lbl_idx = n_old

        # extract 5 individual slices per .nii volumes
        for n in range(0,n_new):
    
            img_input_file = new_lbl_files[n].replace('lm.json','.nii')
            lbl_output_file = os.path.join(output_lbldir,'Gel{:03d}lm.json'.format(lbl_idx))
            img = nib.load(img_input_file).get_fdata()
            img = np.ascontiguousarray(img) # require C cintiguous for img read/write
            img = np.transpose(img,(1,0,2)) #.nii convention displays 1st dim as rows, amke 1st dim column here
            lbl = json.load(open(new_lbl_files[n]))

            coords = np.array(lbl['coords'])
            dx = -np.diff(coords[:,1])  # 1st coord in json is at the base end of UA
            dy = np.diff(coords[:,0]) # for a positive angle convention
            lbls[2,n] = np.arctan2 (dy,dx) * 180/np.pi
            # resave json with new numbering
            json.dump(lbl,open(lbl_output_file,'w'))
            lbl_idx = lbl_idx + 1

            img = do_norm(img)
 
            # coords for selecting a cropped 3d volume
            # ycoord,xcoord,zcoord = np.mean(coords,axis=0) #yx convention in the .json files
            xcoord,ycoord,zcoord = np.mean(coords,axis=0) #yx convention in the .json files corrected by transpose above

            # crop volumes and write out to 2d image files.
            # lbls are saved separately in one single file
            # this refactor hasn't been tested yet on the older json/nii datasets
            ua_centre_1mm_ijk = np.array([xcoord,ycoord,zcoord,1])
            lbls[0:2,n],img_idx = do_lbls_crops(img, ua_centre_1mm_ijk,lbl['pxpermm'],img_idx,output_imgdir,mixup=True)

        # angles = np.reshape(np.tile(angles,(NSLICE,1)),(n_new*NSLICE,),order='F') # duplicate across slice dim
        lbls = np.reshape(np.tile(lbls,(zNSLICE,1)),(n_new*zNSLICE*DOF,),order='F') # duplicate across slice dim
        formatstr = '<{:d}f'.format(n_new*zNSLICE*DOF)
        b_lbls = pack(formatstr,*lbls) 
        with open(os.path.join(output_lbldir,'lbl.dat'),'ab') as fb:     # note file append here.
            fb.write(b_lbls)
            fb.close()

        print('number of existing volumes {:d}'.format(n_old))
        print('number of new volumes {:d}'.format(n_new))
        print('total number of output images {:d}'.format(img_idx))
        assert(img_idx == zNSLICE*(n_new+n_old))

#   inputDir - a flat dir containing both .nii volumes and equal number of matching .json lablels
#   outputDir - parent dir for output labels (lbl), cropped volumes (img) and full volumes (vol). 
#               new input files are re-numbered to be consecutive with existing data files

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir',default=None)
    parser.add_argument('--outputDir',default=None)
    parser.add_argument('--exclude',default=False)
    parser.add_argument('--action',default=None)
    parser.add_argument('--index',default=None)
    args = parser.parse_args()
    exclude = eval(args.exclude)
    if args.index is not None:
        index = int(args.index)
    else:
        index = None

    main(args.inputDir,args.outputDir,exclude,args.action,index=index)