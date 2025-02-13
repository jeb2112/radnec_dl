# quick script to download files from dbox

import dropbox
import os
import numpy as np
import struct
import nibabel as nb

# generate new short lived token to use this script
token = 'sl.Bz7QyqISHf24Dc1RrIRPJ0LsQTTjuui8mVPecf0HNwo_TCyJm1SiGLilFju78Ir-SgEb9sqbrpYpaXrqBlvsQ12_ktsksFkL55E5SjLfZFsq7m7oTOSOFfDNZS18lpjcd1Tvf1yozyuU'
dbox = dropbox.Dropbox(token)
a = dbox.users_get_current_account()

rootdir = '/BLAST DEVELOPMENT/RAD NEC/MANUSCRIPT'
rootdir = '/BLAST DEVELOPMENT/RAD NEC/DATA'

# datadir = os.path.join(rootdir,'RAD NEC MET DATA')
# datadir = os.path.join(rootdir,'RAD NEC MET DSC RESULTS')
datadir = os.path.join(rootdir,'RAD NEC MET')

res = dbox.files_list_folder(datadir)
rv = {}
for entry in res.entries:
    if entry.name.startswith('M00'):
        rv[entry.name] = entry

localdir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
# mridir = os.path.join(localdir,'MRI')
mridir = os.path.join(localdir,'dicom')

# flist = ['t1ce_stripped.nii','t2flair_register_stripped.nii']
# flist = ['objectmask_ET.nii']
flist = ['NAWM.nii','RELCCBV_register.nii','t1ce.nii']
for c in rv.keys():
    c_prefix = c.split('_')[0]
    print(c)
    # casedir = os.path.join(datadir,c,c+' processed')
    casedir = os.path.join(datadir,c)

    # for individual files
    if len(flist):
        for f in flist:
            try:
                md,res = dbox.files_download(os.path.join(casedir,f))
            except dropbox.exceptions.HttpError as e:
                print('HTTP error',e)
                print('skipping {}'.format(f))
                continue
            except ConnectionError as e:
                print('HTTP error',e)
                print('attempt to re-connect')
                # this isn't global scope though
                dbox = dropbox.Dropbox(token)
                md,res = dbox.files_download(os.path.join(casedir,f))
                continue
            except dropbox.exceptions.ApiError as e:
                print('API error',e)
                print('skipping {}'.format(f))
                continue

            local_casedir = os.path.join(mridir,c_prefix)
            if not os.path.exists(local_casedir):
                os.mkdir(local_casedir)
            fname = os.path.join(local_casedir,f)
            if not os.path.exists(fname):
                with open(fname,'wb') as fp:
                    fp.write(res.content)
                # optinally recast the image
                if 'mask' in fname:
                    I = nb.load(fname)
                    # to be extra sure of not overwriting data:
                    new_data = np.copy(I.get_fdata())
                    hd = I.header

                    # update data type:
                    new_data = new_data.astype(np.int8)
                    I.set_data_dtype(np.uint8)

                    # if nifty1
                    if hd['sizeof_hdr'] == 348:
                        new_I = nb.Nifti1Image(new_data, I.affine, header=hd)
                    # if nifty2
                    elif hd['sizeof_hdr'] == 540:
                        new_I = nb.Nifti2Image(new_data, I.affine, header=hd)
                    else:
                        raise IOError('Input image header problem')
                    nb.save(new_I, fname)

    # tbd for dicom dirs
    else:
        a=1

for entry in dbox.files_list_folder(rootdir).entries:
    print(entry.name)

a=1
