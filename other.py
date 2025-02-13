import numpy as np
import operator
import os
import scipy
import time
import random
import tensorflow as tf

# miscellaneous other functions that haven't been sorted out yet

def load_regression_labels(inputDir, dof=3, lbl_name='lbl.dat',N=None):
    lbl = load_matrix(os.path.join(inputDir,lbl_name),'float32',-1)
    lbl = np.reshape(lbl,(-1,dof))
    if N is not None:
        lbl = lbl[:N,:]
    indx = range(0,np.shape(lbl)[0])
    return (indx,lbl)

def load_matrix(filename,dtype,dims):
    mat = np.fromfile(filename,dtype=dtype,count=np.prod(dims))
    if dims != -1:
        mat = np.reshape(mat,dims) # 'F' mode??
    mat = np.transpose(mat) # or this: for python C-style row-major vs matlab
    return mat

def get_rotmat(r1,r2,r3):
    s1 = np.sin(r1)
    c1 = np.cos(r1)
    s2 = np.sin(r2)
    c2 = np.cos(r2)
    s3 = np.sin(r3)
    c3 = np.cos(r3)
    # tait-bryan X1Y2Z3
    Rxyz = np.reshape([c2*c3, -c2*s3, s2, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2],(3,3))
    return Rxyz

def get_affine(matrix, x, y, z):
    matrix = np.pad(matrix,((0,1),(0,1)))
    matrix[3,3] = 1
    o_x = float(x)
    o_y = float(y)
    o_z = float(z)
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0,0,1,o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0,0,1,-o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def shuffle_both(a,b,copy=False): # quick kludge to shuffle two arrays with same random sequence
    rstate = np.random.get_state()
    if copy:
        aa = a.copy()
        bb = b.copy()
        np.random.shuffle(aa)
        np.random.set_state(rstate)
        np.random.shuffle(bb)
        return (aa,bb)
    else:
        np.random.shuffle(a)
        np.random.set_state(rstate)
        np.random.shuffle(b)

def crop_img(img,cropfactor): # hard-coded for equal dims. cropfactor is a multiple > 1.0
    input_shape = img.shape
    img = scipy.ndimage.zoom(img,cropfactor,mode='nearest')
    i1 = tuple(map(lambda x,delta : x//2-delta//2, img.shape, input_shape))
    i2 = tuple(map(operator.add, i1, input_shape  ))
    crop = tuple(map(slice,i1,i2))
    img = img[crop]
    return img

# convenience functions
def get_augtransform(generator,y,dims):
    while True:
        tparams = generator.get_random_transform(dims)
        if tparams['zx'] <= 1: # no zoom outs. per-axis zoom restricted in get_random_transform
            break
    return tparams
def augment_y(y,tparams):
    aug_y = np.empty((5,))
    aug_y[0] = y[0] - tparams['tx']
    aug_y[1] = y[1] - tparams['ty']
    aug_y[2] = y[2] - tparams['tz']
    aug_y[3] = y[3] - tparams['theta'] # sign convention hard-coded here
    aug_y[4] = y[4] - tparams['phi']

    return aug_y
def augment_X(X,generator,tparams):
    aug_X = np.squeeze(generator.apply_transform(X[:,:,np.newaxis], tparams))
    aug_X = generator.preprocessing_function(aug_X)
    return aug_X
def load_img(filename,dims=(64,128),dtype='float32'):
    img = np.fromfile(filename, dtype=dtype,count=-1)
    img = np.reshape(img,dims,order='F')
    return img
def load_validation(x_indexset,y,img_path,dims=(64,128)):
    img = np.empty((len(x_indexset),dims[0],dims[1]))
    for i,x in enumerate(x_indexset):
        img[i,:,:] = load_img(os.path.join(img_path,'I{:05d}'.format(x)))
    return (img,y)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# these settings are necessary for deterministic tenflow
def set_tfdeterministic(seed=42):
    set_seeds(seed=seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# these env vars are cited as possibly needed for deterministic tenflow, but haven't been needed as yet
def set_tfenv(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
