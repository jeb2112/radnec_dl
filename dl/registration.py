import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from datalake import DataLake
from registrationdata import RegistrationData



def cylinder(mres,pxpermm,l=None,r=3.5,f=36):
    r_px = r / pxpermm
    cres = np.round (np.array(mres) / pxpermm)
    C = np.zeros(np.shape(cres),dtype='float32')
    xc = np.arange(-mres[0]//2*pxpermm,mres[0]//2*pxpermm,pxpermm)
    yc = np.arange(-mres[1]//2*pxpermm,mres[1]//2*pxpermm,pxpermm)
    zc = np.arange(-mres[2]//2*pxpermm,mres[2]//2*pxpermm,pxpermm)
    [xx,yy,zz] = np.meshgrid(xc,yc,zc,indexing='ij')
    # C = np.where( (np.sqrt(xx**2+yy**2) < r and np.abs(zz) < l/2, 1, 0)
    C1 = np.where(np.sqrt(xx**2+zz**2)<r,1,0)
    if l is None:
        C = C1
    else:
        C2 = np.where(np.abs(yy) < l/2,1,0)
        C = C1*C2

    F = np.where(np.sqrt(xx**2+(yy+(mres[1]//2*pxpermm-f))**2+zz**2)<2,1,0)
    C = 1 - (C - F)
    C = C.astype('float32')

    if False:
        plt.figure(1)
        plt.imshow(C[:,:,mres[2]//2])
        plt.show()

    return C


def main(mres,pxpermm,db=None,localdir=None,outputdir=None,create=False):

    # generate reference object
    C = cylinder(mres,pxpermm,r=3.5,f=20)

    # read volume from dl
    # transform to label
    # close crop and feather
    dl = RegistrationData(db,localdir,outputdir,mres=mres,crop='ua')
    if create:
        dl.clean_img()
        dl.create_dataset()
    else:
        dl.load_dataset()

    # register to cylinder
    dl.register(C)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl',default="dl"),
    parser.add_argument('--localdir',default=None)
    parser.add_argument('--outputdir',default=None)
    parser.add_argument('--mres',default="32,128,32")
    parser.add_argument('--pxpermm',default="1")
    parser.add_argument('--create',default="True")
    args = parser.parse_args()
    pxpermm = float(args.pxpermm)
    create = eval(args.create)
    if args.mres:
        mres = tuple(map(int,args.mres.split(',')))

    main(mres,pxpermm,
            db = args.dl,
            localdir=args.localdir,
            outputdir=args.outputdir,
            create=create)