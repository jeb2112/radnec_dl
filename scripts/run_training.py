import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import queue
import threading
import multiprocessing as mp
import random
import platform
import numpy as np
import tensorflow as tf
from operator import itemgetter
import time

import regression_cnn
from config import Config
from config import AttrDict
from copy import copy

SEED = 42

def addprocess(id,queue,paramzip):
    configparams = AttrDict(init=dict(paramzip)) # restore the unpickel-able dict
    regression_cnn.main(configparams,thread=id)
    queue.put('Thread {} done'.format(id))

def addqueue(id,queue,configparams):
    print ('Thread {}'.format(id))
    # regression_cnn.main(configparams)
    queue.put('Id {} done'.format(id))

# these three methods probably aren't needed in the main thread
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
def set_tfdeterministic(seed=SEED):
    set_seeds(seed=seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def set_tfenv(seed=SEED):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def dict2zip(idict):
    klist = list(idict.keys())
    vlist = list(idict.values())
    return zip(klist,vlist)


def main(configlist):
    configpath,_ = os.path.split(configlist)
    result=[]
    if 0:
        set_tfdeterministic()  # this shouldn't be needed in the main thread at all
        set_tfenv() # but if needed, probably does need to be here in main thread

    mp.set_start_method('spawn')
    with open(configlist,'r') as fp:
        q1 = mp.Queue()
        threads1 = []
        for i,f in enumerate(fp):
            c = Config(os.path.join(configpath,f.rstrip()))
            # for multprocessing modul, args must be pickle-able, and dicts are not. create an intermediary pzip here to pass
            pzip = dict2zip(c.p)
            threads1.append(mp.Process(target=addprocess, args=(i,q1,pzip), daemon=False, name='Thread{:02d}'.format(i)))
    if len(threads1) == 1:
        regression_cnn.main(c.p,thread=0) # run in main thread
    else:
        runlist = [0,1] # start first two running
        threads1[0].start()
        threads1[1].start()
        # this dict will be used to link the process ID with the monitor sentinel
        rundict = dict(zip([t.sentinel for t in itemgetter(*runlist)(threads1)],itemgetter(*runlist)(threads1)))
        while True:
            if len(runlist) > 1:
                # check sentinels for the current two threads in the runlist
                donelist = mp.connection.wait(t.sentinel for t in itemgetter(*runlist)(threads1))
            else:
                donelist = mp.connection.wait([threads1[runlist[0]].sentinel]) # awkward. *runlist doesn't work in a single element list
            for d in donelist:
                result.append(q1.get())
                dthread = int(rundict[d].name[-2::])
                if 1:
                    threads1[dthread].join() # is this needed along with use of connection.wait?
                runlist.remove(dthread)
                if dthread+2 < len(threads1):
                    runlist.append(dthread+2)
                    threads1[dthread+2].start()
            if runlist == []:
                break
            if len(runlist) > 1:
                rundict = dict(zip([t.sentinel for t in itemgetter(*runlist)(threads1)],itemgetter(*runlist)(threads1)))
            else:
                rundict = dict([(threads1[runlist[0]].sentinel,threads1[runlist[0]])]) # again awkward with 1-length runlist

        print(', '.join(result))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=None)
    args = parser.parse_args()
    if platform.system() == 'Windows':
        cpath = args.config.replace('/media','D:')
    else:
        cpath = args.config

    main(cpath)