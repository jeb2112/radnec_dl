import os
import json
import platform
import re
import datetime
import types
import collections
from copy import deepcopy

# wrapper for a dict for attribute access syntax
class AttrDict(dict):
    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(AttrDict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(AttrDict, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self):
        ch = AttrDict(self)
        return ch
       
class Config(object):
    def __init__(self,configfile=None,init=None,modeldir=None,predict=False,resume=False):
        if init is not None:
            self.p = init # for a primitive copy constructor
        elif modeldir is not None:
            self.p = AttrDict() # normal constructor
            self.ptype = dict.fromkeys(('consts_int','consts_float','strings','booleans'),[]) # reference dict of param types
            self.modeldir = modeldir
            self.readmodelconfig() # reload output from a training run
            if platform.system() == 'Linux':
                for v in self.ptype['strings']:
                    if self.p[v] is not None:
                        self.p[v] = re.sub('D:','/media',self.p[v]).replace('\\','/')
            self.p.retrain=False
            self.p.repredict=predict
            self.p.resume=resume
        else:
            self.p = AttrDict() # normal constructor
            self.ptype = dict.fromkeys(('consts_int','consts_float','strings','booleans'),[]) # reference dict of param types
            self.config_params=os.path.join(os.path.dirname(__file__),'config_params.json')
            self.loadparams() # defaults from json
            self.configfile=configfile
            if configfile is not None:
                self.readconfig() # variable changes for current run
            self.updatevals()
            self.writeconfig()

    # AttrDict.copy() above should also work, but not tested yet
    # this hack implements a shallow copy. seems to work
    def __copy__(self):
        return type(self)(init=self.p)
    # this hack for deep copy. not tested yet
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.a, memo), 
                deepcopy(self.b, memo))
            memo[id_self] = _copy 
        return _copy # or _copy.p? not testsd yet
 
    # raw json into dict of param/values, and dict of params by type
    def loadparams(self):
        with open(self.config_params) as fp:
            self.jsonparams = json.load(fp)
            for k in self.jsonparams.keys():
                plist = []
                for l in self.jsonparams[k]:
                    self.processval(l['name'],l['default'],vtype=k)
                    plist.append(l['name'])
                self.ptype[k] = plist
            fp.close()
        return

    # read the local config file with changes to default values
    def readconfig(self):
        with open(self.configfile) as fp:
            for line in fp:
                if line == '\n':
                    continue
                (var,*_,val) = line.rstrip().split() # any whitespace. no \n by itself on last line
                self.processval(var,val)
            fp.close()

    # process variable according to type
    def processval(self,var,val,vtype=None):
        if var in self.ptype['consts_int'] or vtype == 'consts_int':
            if ',' in val: # ie a csv
                val = re.sub('[\(\)]','',val)
                setattr(self.p,var,tuple(map(int,val.split(','))))
            else:
                val = None if val=="None" else int(val)
                setattr(self.p,var,val)
        elif var in self.ptype['consts_float'] or vtype == 'consts_float':
            if ',' in val:
                val = re.sub('[\(\)]','',val)
                setattr(self.p,var,tuple(map(float,val.split(','))))
            else:
                setattr(self.p,var,float(val))
        elif var in self.ptype['strings'] or vtype == 'strings':
            val = None if val=="None" else val
            setattr(self.p,var,val)
        elif var in self.ptype['booleans'] or vtype == 'booleans':
            setattr(self.p,var,eval(val,{}))


    # odd adjustments and settings
    def updatevals(self):
        # dependent params
        self.p.datadir = os.path.join(self.p.rootdir,'testdata') # sub directory for img,lbl inputs
        self.p.configdir = os.path.join(self.p.rootdir,'config') # sub directory for config files
        self.p.outputdir = os.path.join(self.p.rootdir,'models') # sub directory for training output
        self.p.imgdir = os.path.join(self.p.datadir,'img')
        self.p.tag = 'e{}-{}x{}x{}-{}'.format(self.p.epochs,self.p.mres[0],self.p.mres[1],self.p.mres[2],self.p.tag)
        self.p.modelname = self.p.model + '-' + self.p.tag
        self.p.modelpath = os.path.join(self.p.outputdir,self.p.modelname)
        if platform.system() == 'Windows':
            # bizarre path prepend only in ssh/git-bash/windows
            if self.p.datadir.startswith('C'):
                self.p.datadir = re.sub('C:/Program Files/Git','',self.p.datadir)
            if self.p.outputdir.startswith('C'):
                self.p.outputdir = re.sub('C:/Program Files/Git','',self.p.outputdir)
            self.p.datadir = re.sub('/media','D:',self.p.datadir)
            self.p.outputdir = re.sub('/media','D:',self.p.outputdir)
        elif platform.system() == 'Linux':
            self.p.datadir = re.sub('D:','/media',self.p.datadir)
            self.p.outputdir = re.sub('D:','/media',self.p.outputdir)
        return

    # write out the training configuration
    def writeconfig(self):
        if not os.path.exists(self.p.modelpath):
            os.mkdir(self.p.modelpath)
            os.mkdir(os.path.join(self.p.modelpath,'img'))
            os.mkdir(os.path.join(self.p.modelpath,'lbl'))
            os.mkdir(os.path.join(self.p.modelpath,'png'))
        timestamp = '{}'.format(datetime.datetime.now())
        timestamp = timestamp.split('.')[0].replace(' ','_')
        fname = 'config_{}'.format(timestamp)
        src = os.path.dirname(__file__)
        if platform.system() == 'Windows':
            fname = fname.replace(':','.')
            gitsha = os.popen('cd {} && git rev-parse HEAD'.format(src)).read()
        else:
            gitsha = os.popen('cd {}; git rev-parse HEAD'.format(src)).read()
        # attrs = [k for k in self.p.__dict__.keys() if (not k.startswith('_') and not isinstance(getattr(self.p,k),types.FunctionType)) ]
        with open(os.path.join(self.p.modelpath,fname),'w') as fp:
            fp.write('git rev-parse HEAD={}'.format(gitsha))
            fp.write('{}\n\n'.format('-' * 50))
            for k in self.ptype.keys():
                fp.write('{}\n\n'.format(k))
                for a in self.ptype[k]:
                    attrstr = str(getattr(self.p,a)).replace(' ','')
                    fp.write('{} {}\n'.format(a.ljust(20),attrstr))
                fp.write('{}\n\n'.format('-' * 50))
        return

    # reread the training configuration output file for postprocessing later. probably should use json
    def readmodelconfig(self):
        configlist = [c for c in os.listdir(self.modeldir) if c.startswith('config')]
        configlist.sort()
        if len(configlist) > 1:
            print('More than one model configfile, using latest')
        modelconfigfile = configlist[-1]
        with open(os.path.join(self.modeldir,modelconfigfile)) as fp:
            for ll in fp:
                # ll = next(fp)
                if ll.rstrip() in ['consts_int','consts_float','strings','booleans']:
                    vtype = ll.rstrip()
                    ll = next(fp)
                    if ll.startswith('\n'):
                        ll = next(fp)
                    varlist = []
                    while not ll.startswith('---'):
                        (var,*_,val) = ll.rstrip().split() # any whitespace. no \n by itself on last line
                        varlist.append(var)
                        self.processval(var,val,vtype=vtype)
                        ll = next(fp)                        
                    self.ptype[vtype] = varlist
        return
