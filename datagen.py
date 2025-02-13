from augment import RegressionAug
from other import *
from display import *
import json

# data generator allowing for separate augmentation of data and regression labels
from tensorflow.keras.utils import Sequence
class RegressionGen(Sequence):

    def __init__(self, img_indices, lbl, img_path, batch_size=1, dims = (64,128), input_dims = (256,256,48),
                do_shuffle=False,
                do_train=True, 
                do_crop=True,
                do_plot=False,
                output_size=1,
                name = None,
                outputdir=None,
                mixup=False,
                seed=None,
                local=None,
                do_labelnorm=False,
                labelvar=0, # range. added to normed lbl with std=1
                lbldist=None,
                **kwargs):

        self.input_dims = input_dims
        self.img_indices = img_indices
        self.img_path = img_path
        self.lbl = lbl
        self.batch_size = batch_size
        if kwargs: # note kwargs is only for aug dict nothing else
            self.augment = True
            self.generator = RegressionAug(noisevar=0.03,**kwargs)
        else:
            self.augment = False
        self.mixup = mixup
        self.do_train = do_train
        self.do_shuffle = do_shuffle
        self.do_crop = do_crop # real-time crop after augmentation
        if self.do_crop:
            self.dims = dims
            self.lbl_crop = np.zeros(np.shape(self.lbl))
        else:
            self.dims = self.input_dims # input dims hard-coded for now.
        self.do_plot = do_plot
        self.output_size = output_size
        self.tparams = {}
        self.name = name
        self.outputdir = outputdir
        self.plotax = None
        self.epoch=0
        self.seed = seed
        self.local = local # range for realtime augmentation xy crops
        self.do_labelnorm = do_labelnorm # label norm
        self.lbldist = lbldist
        self.labelvar = labelvar
        if labelvar:
            rstate = np.random.get_state() # save current state
            np.random.seed(77) # arbitrary seed for consistent label errors
            self.labelerror = (np.random.random_sample(np.shape(self.lbl))*2-1)*labelvar
            np.random.set_state(rstate) # restore state
        else:
            self.labelerror = None
        # debug output
        self.debug = False # hard-coded

        if self.do_labelnorm:
            # for realtime crop labels
            if self.lbldist is False:
                print('Generating label distribution')
                self.set_lbldist()
                if self.outputdir is not None:
                    self.save_lbldist()
            elif self.lbldist is True:
                self.read_lbldist()
            elif len(self.lbldist) == 0:
                raise ValueError('empty lbldist array')
            # for fixed crop labels. not done yet
            # self.lbldist = [np.mean(self.lbl,axis=0),np.std(self.lbl,axis=0)]
            # self.lbl = (self.lbl - self.lbldist[0]) / self.lbldist[1]

    def __len__(self):
        return self.lbl.shape[0] // self.batch_size

    def __getitem__(self,batch_index):
        index_set = range(batch_index*self.batch_size,(batch_index+1)*self.batch_size)
        img_set = [self.img_indices[k] for k in index_set]
        X = self.gen_X(img_set)

        if self.do_train:
            y = self.gen_y(index_set)
            if self.mixup:
                (X,y) = self.do_mixup(X,y)
            if self.do_crop:
                (X,y) = self.do_lbls_crops(X,y)
                self.lbl_crop[index_set,:] = y # record cropped labels
            # if self.do_plot: # this was a display generator variant for debugging, probably can remove
            #     if batch_index == 0:
            #         self.plotax = display_img(self.displaybatch_gen(X,y),delay=1,vmax=1,pclose=False, ax=self.plotax)
            if self.do_labelnorm and not self.do_plot: # re purpose self.do_plot flag here, in which case no label norm
                y = (y - self.lbldist[0]) / self.lbldist[1]
                if self.labelerror is not None:
                    y = y + self.labelerror[index_set,:]
            return X,y
        else:
            return X

    def gen_X(self,index_set):
        batch_X = np.empty((self.batch_size, *self.input_dims))
        batch_index = index_set[0]//self.batch_size
        for i,x in enumerate(index_set):
            img = self.load_img(os.path.join(self.img_path,'I{:05d}'.format(x)))
            # img = img / mscale # further normalization
            if self.augment:
                while True:
                    self.tparams[i] = self.generator.get_random_transform(self.dims)
                    if self.tparams[i]['zx'] < 1: # no zoom outs. per-axis zoom restricted in get_random_transform
                        break
                # documentation states it should be a (1,N,N,1) array, ie image in dims 1,2. 
                # but get error with a dummy newaxis in dim 0
                # and the rotation for example applies to dims (0,1)
                batch_X[i] = np.squeeze(self.generator.apply_transform(img[:,:,np.newaxis], self.tparams[i]))
                # didn't realize this is not contained in apply_transform
                batch_X[i] = self.generator.preprocessing_function(batch_X[i])
                if self.debug:
                   if batch_index==0:
                        fname = '{}_{}_{}.png'.format(self.name,i,self.epoch)
                        img = array_to_img(np.expand_dims(batch_X[i],2),'channels_last',scale=True) # dummy channel dim here
                        img.save(os.path.join(self.outputdir,'img',fname))
            else:
                batch_X[i] = img

        return batch_X
    
    def gen_y(self,index_set,do_augment_y=True):
        batch_y = np.empty((self.batch_size,self.output_size))
        batch_index = index_set[0]//self.batch_size
        if self.augment and do_augment_y: # extra kludge for plotting un-augmented labels.
            # manually apply requested augmentatinon if they apply to angle
            if np.shape(self.lbl)[1] == 5:
                for i,x in enumerate(index_set):
                    batch_y[i,0] = self.lbl[x,0] - self.tparams[i]['tx']
                    batch_y[i,1] = self.lbl[x,1] - self.tparams[i]['ty']
                    batch_y[i,2] = self.lbl[x,2] - self.tparams[i]['tz']
                    batch_y[i,3] = self.lbl[x,3] - self.tparams[i]['theta']
                    batch_y[i,4] = self.lbl[x,4] - self.tparams[i]['phi']
                    if self.debug:
                        if batch_index == 0:
                            lbls = dict(zip(['lx','ly','lz','theta','phi'],batch_y[i,:]))
                            fname = '{}_{}_{}.json'.format(self.name,i,self.epoch)
                            json.dump(lbls,open(os.path.join(self.outputdir,'lbl',fname),'w'))
            elif np.shape(self.lbl)[1] == 3:
                for i,x in enumerate(index_set):
                    batch_y[i,0] = self.lbl[x,0] - self.tparams[i]['tx']
                    batch_y[i,1] = self.lbl[x,1] - self.tparams[i]['ty']
                    batch_y[i,2] = self.lbl[x,2] - self.tparams[i]['tz']
                    if self.debug:
                        if batch_index == 0:
                            lbls = dict(zip(['lx','ly','lz'],batch_y[i,:]))
                            fname = '{}_{}_{}.json'.format(self.name,i,self.epoch)
                            json.dump(lbls,open(os.path.join(self.outputdir,'lbl',fname),'w'))
        else:
            batch_y = self.lbl[index_set]

        return batch_y

    def set_seed(self,N=None):
        np.random.seed(self.seed)
        if N is not None:
            np.random.randint(-1,1,N)

    # pre-calculate the label normalization distributions
    def set_lbldist(self):
        self.do_labelnorm = False
        for i,(_,y) in enumerate(self):
            if i==0:
                label = y
            else:
                label = np.concatenate((label,y),axis=0)
        self.lbldist = [np.mean(label,axis=0),np.std(label,axis=0)]
        self.do_labelnorm = True
        # self.lbl = (self.lbl - self.lbldist[0]) / self.lbldist[1]

    def save_lbldist(self):
        with open(os.path.join(self.outputdir,'lbl','lbldist.pkl'),'wb') as fp:
            pickle.dump(self.lbldist,fp)

    def read_lbldist(self):
        with open(os.path.join(self.outputdir,'lbl','lbldist.pkl'),'rb') as fp:
            self.lbldist = pickle.load(fp)

    def do_mixup(self,X,y):
        alph=0.3
        lambd = np.random.beta(alph, alph, np.shape(y)[0])
        X1, y1 = shuffle_both(X,y,copy=True)
        new_X = X * lambd[:,np.newaxis,np.newaxis] + X1 * (1-lambd[:,np.newaxis,np.newaxis])
        # new_y = y * lambd[:,np.newaxis] + y1 * (1-lambd[:,np.newaxis])
        new_y = np.where(lambd[:,np.newaxis] < 0.5,y1,y)
        return (new_X, new_y)  

    # def get_augtransform(self,i):
    #     while True:
    #         self.tparams[i] = self.generator.get_random_transform(self.dims)
    #         if self.tparams[i]['theta'] + self.lbl[i,2] > 0 and self.tparams[i]['zx'] < 1: # no negative angle, no zoom -outs
    #             self.tparams[i]['zy'] = self.tparams[i]['zx'] # axis-dependent zoom will break angle unless corrected?
    #             break

    def load_img(self,filename,dtype=np.float32):
        img = np.fromfile(filename, dtype=dtype,count=-1)
        # img = np.reshape(img,self.dims,order='F')
        img = np.reshape(img,self.input_dims,order='C')
        return img

    def on_epoch_end(self):
        self.epoch = self.epoch + 1
        if self.do_shuffle:
            shuffle_both(self.img_indices,self.lbl)
        # for validation as opposed to training reset seed for realtime crop origin so crops are reproduced each epoch
        if self.seed is not None:
            np.random.seed(self.seed)

# simple generators for data display
 
# for every n'th image, optionally display a results vector
    def displayall_gen(self,nth=10,res=None,n1st = 3,do_augment_y=True):
        for i,a in enumerate(self.img_indices):
            y = self.lbl[i]
            if (i+n1st)%nth == 0:
                X = self.load_img(os.path.join(self.img_path,'I{:05d}'.format(a)))
                if self.augment:
                    tparams = get_augtransform(self.generator,y,self.dims)
                    X = augment_X(X,self.generator,tparams)
                    if do_augment_y: # separate flag here to turn off label augmentation for plotting purposes
                        y = augment_y(y,tparams)
                if self.do_crop:
                    X = X[np.newaxis,:,:,:]
                    y = y[np.newaxis,:] # insert dummy axis for batch
                    (X,y) = self.do_lbls_crops(X,y)
                    X = np.squeeze(X)
                    y = np.squeeze(y)
                if res is not None:
                    y = (res[i],res[i]-y)
                yield X, y

# for a specific set
    def displayN_gen(self,nset):
        for i in nset:
            img = self.load_img(os.path.join(self.img_path,'I{:05d}'.format(self.img_indices[i])))
            img = img / np.max(img)
            y = self.lbl[i]
            yield img, y

# another generator to display current (augmented) batch from within the data generator. in this case images are
# already loaded into memory, the generator is just a dummy for the input to display_img
    def displaybatch_gen(self,img,lbl):
        for i in range(0,img.shape[0]):
            yield np.squeeze(img[i,:,:,:]),np.squeeze(lbl[i,:]),i

# another generator for the ImageDataGenerator .fit() method
    def xdata_gen(self):
        img = np.empty( (len(self.img_indices)) + self.dims)
        for i,x in enumerate(self.img_indices):
            img[i,:,:] = self.load_img(os.path.join(self.img_path,'I{:05d}'.format(a)))
            yield img[i,:,:]

# these wrappers haven't been checked yet
    def get_augtransform(self,i):
        return get_augtransform(self.generator,self.lbl[i],self.dims)

    def augment_y(self,i):
        return augment_y(self.lbl[i],self.tparams[i])

    def augment_X(self,X,i):
        return augment_X(X,self.generator,self.tparams[i])

    # same as trainingdata.py except no zfactor
    def crop_img(self,I,croporigin,cropshape,pad=128):
        I = np.pad(I,((pad,pad),(pad,pad)))
        croporigin = tuple(np.array(croporigin)+np.array([pad,pad]))
        i1 = tuple(map(lambda x,delta : x-delta//2, croporigin, cropshape))
        i2 = tuple(map(operator.add, i1, cropshape  ))
        crop = tuple(map(slice,i1,i2))
        I = I[crop]
        I = I.copy(order='C')
        return I


    # crop 3d volumes, return modified px,py,pz label coordinates to the cropped images.
    def do_lbls_crops(self,img,lbls):
        label = np.zeros(np.shape(lbls)) 
        img_output = np.zeros((self.batch_size,self.dims[0],self.dims[1],self.dims[2]))
        for i in range(self.batch_size):
            label[i,3] = lbls[i,3] # ry,rx will be unaltered
            label[i,4] = lbls[i,4]
            X = img[i,:,:,:]
            y = lbls[i,:3]

            # calculate xy crop coords to account for resampling to 256 equivalent full matrix at 1mm/pixel
            oxcrop = round(y[0])
            oycrop = round(y[1])
            ozcrop = round(y[2])
            if self.local is not None:
                oxcrop = oxcrop + np.random.randint(-self.local,self.local+1)
                oycrop = oycrop + np.random.randint(-self.local,self.local+1)
                if self.dims[2] < self.input_dims[2]:
                    zlocal = self.input_dims[2]
                    while ozcrop+zlocal+self.dims[2]//2 >= self.input_dims[2] or ozcrop+zlocal-self.dims[2]//2<0:
                        zlocal = np.random.randint(-self.local,self.local+1)
                    ozcrop = ozcrop + zlocal

            xcrop = self.dims[0]
            ycrop = self.dims[1]
            zcrop = self.dims[2]
            label[i,0] = y[0] - (oxcrop-xcrop/2) - 1 # xcrop/2 is 1-based numbering so subtract 1
            label[i,1] = y[1] - (oycrop-ycrop/2) - 1
            if self.dims[2] < self.input_dims[2]:
                label[i,2] = y[2] - (ozcrop-zcrop/2) - 0 # TODO: possible error in dcm slice arithmetic, setting 0 here is a kludge
            else:
                label[i,2] = y[2] # for now there is no z crop
            
            if 1 and self.dims[2] < self.input_dims[2]: # do zcrop
                zcrop = range(ozcrop-zcrop//2,ozcrop+zcrop//2) # odd slices not supported
                img_zcrop = X[:,:,zcrop]
            else: # not implementing zcrop yet
                img_zcrop = X
                zcrop = range(np.shape(X)[2])

            img_batch = np.zeros((xcrop,ycrop,len(zcrop)),dtype=np.float32)
            for z in range(len(zcrop)):
                z_resample = img_zcrop[:,:,z] # display convention
                z_resample = self.crop_img(z_resample,(oxcrop,oycrop),(xcrop,ycrop) ) # px resampling already done
                img_batch[:,:,z] = z_resample
            img_output[i,:,:,:] = img_batch

        return img_output,label
