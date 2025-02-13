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

from datalake import DataLake

class TrainingData(DataLake):
    def __init__(self,name,localdir,outputdir=None,index=None,mixup=True,N=None,mres=(32,128,32),crop='ua',dof=5,tag=None):
        DataLake.__init__(self,name,localdir=localdir)
        self.imres = 256 # reampled output resolution equivalent full-fov
        self.mres = mres
        self.nslice = mres[2] # number of slices containing range of UA R/L positions to crop
        self.znslice = self.nslice # crop in z
        self.dof = dof
        self.xycrop=True
        if list(self.mres[0:2]) == [self.imres]*2: # ie not xycrop, a dummy crop
            self.xycrop = False
            self.crop = None # ie dummy crop origin is matrix origin
            self.local = None # no additional random crop origin shift either
        else:
            self.local = 8 # number of random pixels for xyz crop
            self.crop = crop # if xy crop, crop to origin, or ua.
        self.stride=(1,1,1)
        if self.dof==3:
            self.stride=(2,2,1)
        self.inputDir = localdir # location of the db and root dir of all the site sub-dirs ie testdata/sharepoint
        if outputdir is not None: # location of the training data img and lbl dirs, ie testdata
            self.output_lbldir = os.path.join(outputdir,'lbl_{}x{}x{}'.format(*self.mres)) # currently hard-coded
            self.output_imgdir = os.path.join(outputdir,'img_{}x{}x{}'.format(*self.mres))
            if self.crop == 'ua':
                self.output_lbldir = self.output_lbldir + '_' + self.crop
                self.output_imgdir = self.output_imgdir + '_' + self.crop
            if tag is not None:
                self.output_lbldir = self.output_lbldir + '_' + tag
                self.output_imgdir = self.output_imgdir + '_' + tag
            self.output_pngdir = os.path.join(self.output_imgdir,'png')
            if index is None: # ie keep existing img dirs for single case debugging
                self.clean_img() 
            self.output=True
        else:
            self.output = False

        # row index of the database, for debugging a specific case
        if index is not None:
            self.start_idx = index - 1
            self.end_idx = index
        elif N is not None:
            self.start_idx = 0
            self.end_idx = N
        else:
            self.start_idx = 0
            self.end_idx = self.get_N()

        self.lbls = []
        self.img_idx = 0
        self.mixup = mixup # whether to interpolate slices
 
    # remove all current image crops (binaries and pngs)
    # lbls are just one file so over-writing is reliable
    def clean_img(self):
        imglist = glob.glob(self.output_imgdir+'/I*')
        for i in imglist:
            os.remove(i)
        pnglist = glob.glob(self.output_pngdir+'/*.png')
        for p in pnglist:
            os.remove(p)

    # return the normalized unit vector
    def normalizeUnitVector(self,vector):
        abs_vector = np.linalg.norm(vector)
        if (abs_vector==0):  abs_vector = np.finfo(vector.dtype).eps
        return np.array(vector) / abs_vector

    # get pos from 2 points, ie UA coords in AX slices 1,12
    def getPoseFromEndPoints(self,point1,point2):

        # y defined along the ua
        directiony = self.normalizeUnitVector(point2-point1)
        directionangles = np.rad2deg(np.arccos(directiony))
        pose_origin = np.mean(np.vstack([point1,point2]),axis=0)#   - origin
        # pose parameters
        pose_params = directionangles.tolist()+pose_origin.tolist()
    
        # find the rotation axis and uncertainty. 
        # because the reference vector is [0 0 1], one half the terms in the propagation of error from a cross product are not filled in here
        Raxis = np.cross(np.array([0,0,1]),directiony)
        ux = np.reshape([0, -directiony[2], -directiony[1],  directiony[2], 0, -directiony[0],  -directiony[1], directiony[0], 0],(3,3))
        vx = np.reshape([0,-1,0,1,0,0,0,0,0],(3,3))
        Raxis = self.normalizeUnitVector(Raxis)

        # find the rotation angle and uncertainty
        # using the arccos with only the directiony component has poor noise property, for small angles
        #Rangle = acos(dot(directiony,[0 0 1]))

        #instead form the angle with arctan using the inplane magnitude
        arg1 = np.sqrt(sum(np.power(directiony[0:2],2)))
        arg2 = directiony[2]
        Rangle = np.arctan2(arg1,arg2)

        # convert to Tait Bryan. 
        # Note that Rxis[2]==0 for the UA geometry, so some of those terms have just been ignored in the error propagation
        arg1 = Raxis[0] * Raxis[1] * (1 - np.cos(Rangle)) + Raxis[2] * np.sin(Rangle)
        Rz = np.rad2deg( np.arcsin(arg1) )

        # for more stable propagation here, use sin(th) = th for small th
        arg1 = Raxis[1] * np.sin(Rangle) - Raxis[0] * Raxis[2] * (1 - np.cos(Rangle))
        arg2 = 1 - (np.power(Raxis[1],2) + np.power(Raxis[2],2) ) * (1 - np.cos(Rangle))
        Ry = np.rad2deg( np.arctan2(arg1,arg2) ) # since angles are very limited in UA geometry, can use arctan for easier error propagation

        arg1 = Raxis[0] * np.sin(Rangle) - Raxis[1] * Raxis[2] * (1 - np.cos(Rangle))
        arg2 = 1 - (np.power(Raxis[0],2) + np.power(Raxis[2],2)) * (1 - np.cos(Rangle))
        Rx = np.rad2deg( np.arctan2(arg1,arg2) )        
        return Rx,Ry

    # returns dicom px to mm coordinate transform matrix
    def dcm_matrix(self,pos,orient,deltaxy,deltaz,table,model):
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
    def do_norm(self,img):
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

    # TODO: separate out the zfactor rescaling from the crop
    def crop_img(self,I,zfactor,croporigin,cropshape,pad=128): # zfactor is a arbitrary resampling factor
        I = scipy.ndimage.zoom(I,zfactor,mode='nearest')
        I = np.pad(I,((pad,pad),(pad,pad)))
        croporigin = tuple(np.array(croporigin)+np.array([pad,pad]))
        i1 = tuple(map(lambda x,delta : x-delta//2, croporigin, cropshape))
        i2 = tuple(map(operator.add, i1, cropshape  ))
        crop = tuple(map(slice,i1,i2))
        I = I[crop]
        I = I.copy(order='C')
        return I

    def rot_img(self,img, angle, origin):
        imsize = np.shape(img)
        X = [imsize[1] - origin[0], origin[0]]
        Y = [imsize[0] - origin[1], origin[1]]
        img_padded = np.pad(img, [Y, X], 'constant')
        img_rotated = rotate(img_padded, angle, reshape=False)
        img_rotated  = img_rotated[Y[0]:-Y[1],X[0]:-X[1]]
        return img_rotated

    # set the origin, crop 3d volumes, write out to file and return ua centre in-plane i,j pixel coords in the cropped images.
    # img_idx is a counter for output file numbering
    def do_lbls_crops(self,img,ua_centre_1mm_px,px,mixup=True,tag=None,local=None):
        label = np.zeros(3,)
        ua_centre_1mm_ijk = np.array(list(map(round,ua_centre_1mm_px)))

        if self.dof == 5 and self.xycrop==True:
            # calculate xy crop coords to account for resampling to 256 equivalent full matrix at 1mm/pixel
            if self.crop == 'ua': # crop to the UA xyz locality
                oxcrop = ua_centre_1mm_ijk[0]
                oycrop = ua_centre_1mm_ijk[1]
                ozcrop = ua_centre_1mm_ijk[2]
                if local is not None:
                    oxcrop = oxcrop + np.random.randint(-local,local+1)
                    oycrop = oycrop + np.random.randint(-local,local+1)
                    ozcrop = ozcrop + np.random.randint(-local,local+1)
                        
            else: # crop image origin. correct for pixel scale
                oxcrop = int((np.shape(img)[0]*px)//2)-1
                oycrop = int((np.shape(img)[0]*px)//2)-1
                ozcrop = int((np.shape(img)[2])//2)-1 # not resampling in z for now since sag T2 exactly 1.0mm z res for both skyra and ingenia

            xcrop = self.mres[0]
            ycrop = self.mres[1]
            label[0] = ua_centre_1mm_px[0] - (oxcrop-xcrop/2) - 1 # xcrop/2 is 1-based numbering so subtract 1
            label[1] = ua_centre_1mm_px[1] - (oycrop-ycrop/2) - 1
            label[2] = ua_centre_1mm_px[2] - (ozcrop-self.nslice/2) - 0 # TODO: possible error in dcm slice arithmetic, setting 0 here is a kludge

            zcrop = range(ozcrop-self.nslice//2,ozcrop+self.nslice-self.nslice//2) # odd slices not supported
            img_zcrop = img[:,:,zcrop]

            if mixup: # z-interpolate for additional training images 
                f_zset = scipy.interpolate.interp1d(np.arange(0,self.nslice),img_zcrop,axis=2)
                zcrop = np.arange(0.5,self.nslice-1,0.5) # drop two edges slicwes in this case
                img_zcrop = f_zset(zcrop)

        elif (self.dof == 3 or not self.xycrop): # z-only crop, xy just resample and stride
            # awkward mix. 256 is resolution post-resampling. should separate out the px resampling
            xcrop = self.imres
            ycrop = self.imres        
            # px included here arithmetic pre-resampling   
            oxcrop = int((np.shape(img)[0]*px)//2)-1
            oycrop = int((np.shape(img)[0]*px)//2)-1
            ozcrop = int((np.shape(img)[2])//2)-1 # not resampling in z for now since sag T2 exactly 1.0mm z res for both skyra and ingenia
            label[0] = ua_centre_1mm_px[0] - (oxcrop-xcrop/2) - 1 # xcrop/2 is 1-based numbering so subtract 1
            label[1] = ua_centre_1mm_px[1] - (oycrop-ycrop/2) - 1
            label[2] = ua_centre_1mm_px[2] - (ozcrop-self.nslice/2) - 0 # TODO: possible error in dcm slice arithmetic, setting 0 here is a kludge
            # adjust labels for stride. no z slice stride yet.
            label[0] = label[0] / self.stride[0]
            label[1] = label[1] / self.stride[1]

            zcrop = range(ozcrop-self.nslice//2,ozcrop+self.nslice-self.nslice//2,self.stride[2]) # odd slices not supported
            img_zcrop = img[:,:,zcrop]

        img_output = np.zeros((xcrop//self.stride[0],ycrop//self.stride[1],len(zcrop)),dtype=np.float32)
        for z in range(len(zcrop)):
            z_resample = img_zcrop[:,:,z] # display convention
            z_resample = self.crop_img(z_resample,px,(oxcrop,oycrop),(xcrop,ycrop) )
            if self.dof==3:
                z_resample = z_resample[::self.stride[0],::self.stride[1]]
            img_output[:,:,z] = z_resample
        filename = 'I{:05d}'.format(self.img_idx)
        if self.output:
            if not os.path.exists(self.output_imgdir):
                os.mkdir(self.output_imgdir)
                os.mkdir(self.output_pngdir)
            if not os.path.exists(self.output_lbldir):
                os.mkdir(self.output_lbldir)
            with open(os.path.join(self.output_imgdir,filename),'wb') as fo:
                fo.write(img_output)
                fo.close()
        if tag is not None:
            filename = filename + tag
        if self.output:
            matplotlib.image.imsave(os.path.join(self.output_imgdir,'png',filename+'.png'),montage(np.transpose(img_output,(2,0,1)), grid_shape=(5,self.nslice/self.stride[2]//5+1)))
        self.img_idx = self.img_idx + 1

        return label


    # main method to process a set of training data from the db, crops and labels
    def create_trainingdata(self):

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
            if self.dof==5:
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

            # optional rotation for sag if single oblique
            angle_zy_sag = np.arctan2(orient[1,1],orient[0,1]) * 180/np.pi
            if np.abs(angle_zy_sag) > 1:
                img_rot = np.zeros(np.shape(img))
                for s in range(np.shape(img)[2]):
                    img_rot[:,:,s] = self.rot_img(img[:,:,s],-angle_zy_sag,ua_centre_ijk[1::-1])
                img = img_rot

            # crop volumes and write out to 2d image files.
            # lbls are saved separately in one single file
            tag = '-site{}_case{}'.format(record_sag['site'],record_sag['caseid'])
            lbl[0:3] = self.do_lbls_crops(img, ua_centre_1mm_px,px,mixup=False,tag=tag,local=3)
            self.lbls.append(lbl)

        self.lbls = np.asarray(self.lbls).T
        ndl = np.shape(self.lbls)[1]
        if self.dof == 3:
            lbls = np.reshape(self.lbls,(ndl*self.dof),order='F') # duplicate across slice dim
            formatstr = '<{:d}f'.format(ndl*self.dof)
        else:
            lbls = np.reshape(self.lbls,(ndl*self.dof,),order='F')
            formatstr = '<{:d}f'.format(ndl*self.dof)
        b_lbls = pack(formatstr,*lbls) 
        if self.output:
            with open(os.path.join(self.output_lbldir,'lbl.dat'),'wb') as fb: # over-writes any existing file
                fb.write(b_lbls)
                fb.close()


    # create simulated data for testing
    def create_testdata(self):
        self.output_lbldir = self.output_lbldir + '-test'
        self.output_imgdir = self.output_imgdir + '-test'
        self.output_pngdir = os.path.join(self.output_imgdir,'png')
        tag = 'test'
        img = np.zeros((256,256,70))
        img[127,96:160,35] = 1
        ua_centre_1mm_px = np.array([127,127,35])
        px = 1
        rx = 0
        ry = 0
        ndl = 10
        for idx in range(ndl):
            lbl = np.zeros((self.dof,))
            lbl[0:3] = self.do_lbls_crops(img, ua_centre_1mm_px,px,mixup=False,tag=tag,local=8)
            lbl[3] = rx
            lbl[4] = ry
            self.lbls.append(lbl)
 
        self.lbls = np.asarray(self.lbls).T
        lbls = np.reshape(self.lbls,(ndl*self.dof,),order='F')
        formatstr = '<{:d}f'.format(ndl*self.dof)
        b_lbls = pack(formatstr,*lbls) 
        with open(os.path.join(self.output_lbldir,'lbl.dat'),'wb') as fb: # over-writes any existing file
            fb.write(b_lbls)
            fb.close()
