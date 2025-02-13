import os
import glob
import sys
from struct import pack
import numpy as np
import matplotlib.pyplot as plt
import functools,operator
import scipy
import scipy.misc
from scipy.interpolate import griddata
from scipy.ndimage import zoom,rotate
from skimage import exposure
from skimage.util import montage
import matplotlib.image
from scipy.ndimage import affine_transform
import SimpleITK as sitk

# from datalake import DataLake
from cnn_2d.other import * # not sure what to do with other module yet
from dataset import DataSet

class RegistrationData(DataSet):
    def __init__(self,name,localdir,outputdir=None,index=None,N=None,mres=(32,128,32),crop='ua',dof=5,tag=None,order=3):
        DataSet.__init__(self,name,localdir,outputdir=outputdir,index=None,N=None,mres=mres,crop=crop,dof=dof,tag=tag)
        self.xycrop=True
        if list(self.mres[0:2]) == [self.imres]*2: # ie not xycrop, a dummy crop
            self.xycrop = False
            self.crop = None # ie dummy crop origin is matrix origin
            self.local = None # no additional random crop origin shift either
        else:
            self.local = None # number of random pixels for xyz crop
            self.crop = crop # if xy crop, crop to origin, or ua.
        self.order = order
        self.N = None
        self.output_txdir = os.path.join(self.output_imgdir,'tx')
        if not os.path.exists(self.output_txdir):
            os.mkdir(self.output_txdir)

    def load_dataset(self):
        dirlist = os.listdir(self.output_imgdir)
        dirs=[]
        for d in dirlist:
            if not os.path.isdir(os.path.join(self.output_imgdir,d)) and d.startswith('I'):
                dirs.append(d)
        dirs.sort()
        self.N = len(dirs)

    # main method to process a set of registration data from the db, crops and labels
    def create_dataset(self):

       # loop through all records in db, or just one for debugging
        for r in range(self.start_idx,self.end_idx):
            lbl = np.zeros(self.dof,)
            # get ax volume for coords
            try:
                record_ax = self.get_record(r,'AX')
            except KeyError as e:
                print(e)
                continue
            if record_ax['comment']: # skip any commented record
                continue
            # with perfect alignment, UA is at origin of AxT2. rows/cols are 1-based numbering, so subtract 1
            row_index = float(record_ax['rows'])/2.0-1
            col_index = float(record_ax['cols'])/2.0-1
            # pixel coords of UA centre point, 1st and 12th slice
            centre1_ij = np.array([col_index,row_index,0,1]) # note dicom def here
            centre2_ij = np.array([col_index,row_index,11,1]) # 12th slice hard-coded here

            orient = np.reshape(np.array(list(map(float,record_ax['orient'].split(',')))),(2,3))
            pos = np.array(list(map(float,record_ax['slicepos'].split(','))))
            px = float(record_ax['px'])
            M_ax = self.dcm_matrix(pos,orient,px,5,'AX',record_ax['model']) # 5mm hard-coded for Ax T2 here

            # patient coords (mm) of UA centre points
            centre_mm = np.zeros((2,3))
            centre_mm[0,:] = np.reshape(np.matmul(M_ax,centre1_ij)[0:3],(1,3))
            centre_mm[1,:] = np.reshape(np.matmul(M_ax,centre2_ij)[0:3],(1,3))
            # for the two conventions, sort on the z coordinate to get a consistent angle
            centre_mm = centre_mm[centre_mm[:,2].argsort()]

            # get tait bryan angles from UA vector
            rx,ry = self.getPoseFromEndPoints(centre_mm[0,:],centre_mm[1,:])
            lbl[3] = rx
            lbl[4] = ry

            # get sag volume
            # don't have proper indexing consitent bewteen both tables so look up sag from ax site/case
            try:
                record_sag = self.get_record(r,'SAG',record_ax['site'],case=record_ax['caseid'])
            except KeyError as e:
                print(e)
                continue
            if record_sag['comment']: # skip any incomplete record
                continue
            orient = np.reshape(np.array(list(map(float,record_sag['orient'].split(',')))),(2,3))
            pos = np.array(list(map(float,record_sag['slicepos'].split(','))))
            px = float(record_sag['px'])
            pz = float(record_sag['slicethk'])
            M_sag = self.dcm_matrix(pos,orient,px,pz,'SAG',record_sag['model'])
            M_sag_1mm = self.dcm_matrix(pos,orient,1,pz,'SAG',record_sag['model'])

            vol = self.get_dbvol('SAG')
            img = np.zeros((vol[0].Rows,vol[0].Columns,len(vol)))
            for v in range(len(vol)):
                img[:,:,v] = vol[v].pixel_array 
            img = np.transpose(img,(1,0,2)) #dicom convention displays 1st dim as rows, amke 1st dim column here
            if self.dof==5:
                lbl[3] = -lbl[3] # this transpose also flips the sign of the Rx rotation.
            img = self.do_norm(img)

            # coords for selecting a cropped 3d volume
            # UA centre point in patient coords, mm, from Ax T2
            ua_centre_mm = np.append(np.mean(centre_mm,axis=0),[1])
            # convert to pixel coords in sag
            ua_centre_px = np.array(list(np.matmul(np.linalg.inv(M_sag),ua_centre_mm)[0:3]))
            ua_centre_ijk = np.array(list(map(round,ua_centre_px)))
            ua_centre_1mm_px = np.array(list(np.matmul(np.linalg.inv(M_sag_1mm),ua_centre_mm)[0:3]))
            print('dl index {}'.format(r+1), ua_centre_px, ua_centre_1mm_px)

            # rotate and centre the volume to the UA
            R = get_rotmat(np.deg2rad(lbl[4]),0,np.deg2rad(lbl[3])) # in dimension order, patient X (R/L) is 3rd dim, patient Y (A/P) is 1st dim
            I = get_rotmat(0,0,0)
            T = get_affine(R,ua_centre_1mm_px[0],ua_centre_1mm_px[1],ua_centre_1mm_px[2])
            img_rot = affine_transform(img,T[:3,:3],offset=T[:3,3],order=self.order)
            # recentre_offset = np.array([ua_centre_px[0]-record_sag['rows']//2,ua_centre_px[1]-record_sag['cols']//2,ua_centre_px[2]-record_sag['nslice']//2])
            # img_rot = affine_transform(img_rot,I,offset=recentre_offset,order=self.order)
            img = img_rot

            # crop volumes and write out to 2d image files.
            # lbls are saved separately in one single file
            tag = '-site{}_case{}'.format(record_sag['site'],record_sag['caseid'])
            lbl[0:3] = self.do_lbls_crops(img, ua_centre_1mm_px,px,tag=tag)
        self.N = self.img_idx

    def register(self,C):
        # having cyl as fixed breaks the registration. 
        moving = sitk.GetImageFromArray(C)
        for r in range(self.N):
            filename = os.path.join(self.output_imgdir,'I'+'{:05d}'.format(r))
            I = load_img(filename,self.mres)
            fixed = sitk.GetImageFromArray(I)

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerScalesFromPhysicalShift()             
            R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
            R.SetInitialTransform(sitk.Euler3DTransform())
            R.SetInterpolator(sitk.sitkLinear)

            R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))

            outTx = R.Execute(fixed, moving)

            print("-------")
            print(outTx)
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")

            sitk.WriteTransform(outTx, os.path.join(self.output_txdir,'T{:05d}.txt'.format(r)))

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            # because fixed cyl breaks the registration, invert transform here to make the data moving
            outTx_inv = outTx.GetInverse()
            print("outTx_inv---")
            print(outTx_inv)
            resampler.SetTransform(outTx_inv)
            out = resampler.Execute(fixed)
            moving_reg = sitk.GetArrayFromImage(out)

            if 0:
                simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
                simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)


            plt.figure(1,figsize=(8,4))
            ax=plt.subplot(2,2,1)
            plt.imshow(C[:,:,16])
            ax=plt.subplot(2,2,2)
            plt.imshow(I[:,:,16])
            ax=plt.subplot(2,2,3)
            plt.imshow(moving_reg[:,:,16])
            plt.show()

            a=1
            if 0:
                cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
                sitk.Show(fixed, "ImageRegistration1 Composition",debugOn=True)

    def command_iteration(self,method):
        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")