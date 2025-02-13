import os
import requests
import requests_oauthlib
import json
import msal
import argparse
import re
import io
import pickle
import time
import concurrent.futures as futures
import inspect
import logging
import logging.config

from session import OAuthSession
from chromedriver import ChromeDriver
from datalake import DataLake
from trainingdata import TrainingData
from case import Case
from sshsession import SSHSession

# other constant
AUTHTIMEOUT = 60 # max wait for authentication

# this is an example of a kludge for missing signals implementation in python for windows
def timeout(timelimit):
    def decorator(func):
        def decorated(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timelimit)
                except futures.TimeoutError:
                    print('Timedout!')
                    raise TimeoutError from None
                else:
                    print(result)
                executor._threads.clear()
                futures.thread._threads_queues.clear()
                return result
        return decorated
    return decorator

@timeout(AUTHTIMEOUT)
def waitforlogin(driver):
    codematch = None
    while codematch == None: # wait until authentication complete. nb the code is returned as a new url, not as body of an html page
        codematch = re.search('code=(.*?)\&session_state',driver.current_url)
    code = codematch.group(1)
    return code

# generator for getting mr dicom files from sharepoint landing page
def download_dicomfilegen(dir,destdir):
    d = 1
    while True:
        zipdir = dir+'_01-{:03d} MR DICOM'.format(d)
        zipfile = dir+'_01-{:03d} DICOM.zip'.format(d)
        dirname = '/'.join([dir,zipdir,zipfile])
        # dirname = os.path.join(dir,zipdir,zipfile)
        destname = '/'.join([destdir,zipfile])
        # destname = os.path.join(destdir,zipfile)
        d = d+1
        yield dirname,destname


def main(db,action=None,flags=None,dirlist=None,drive=None,case=None,site=None,filelist=None,saveauth=True,mres=None,dof=5,
        localdir=None,outputdir=None,replace=False,rowindex=None,profile=False,tags=None,host=None,user=None,pword=None):

    spath = os.path.dirname(__file__)

    logging.basicConfig(filename=os.path.join(localdir,'dl.log'))
    logging.config.fileConfig(os.path.join(spath,'dl_logging.conf'),defaults={'logfilename':os.path.join(localdir,'dl.log')})
    log = logging.getLogger(__name__)
    rlog = logging.getLogger('urllib3')
    rlog.setLevel(logging.INFO)
    wlog = logging.getLogger('msal')
    wlog.setLevel(logging.INFO)

    # load aad config. dir location is hard-coded in src dir for now
    with open(os.path.join(spath,'aad-dl_main.json'),'r') as fp:
        aad = json.load(fp)
        
    if action == 'download_vpn':
        s1 = SSHSession(user,host,pword)
        cs = Case(db,localdir=localdir)
        cs.set_session(s1)
        cs.caseid = case
        cs.download_drive_vpn(dirlist,case=case)
        s1.close()

    if action == 'download':
        driver = ChromeDriver(aad=aad)
        driver.get_token()
        # attempt an oauth session for sharepoint
        s2 = OAuthSession(driver.token,aad=aad)
        driver.quit()
        dl = DataLake(db,localdir=localdir,replace=replace,drive=drive)
        dl.set_session(s2)
        dl.download_drive(dirlist)

    if action == 'download_case':
        driver = ChromeDriver(aad=aad)
        driver.get_token()
        # attempt an oauth session for sharepoint
        s2 = OAuthSession(driver.token,aad=aad)
        driver.quit()
        dl = DataLake(db,localdir=localdir,replace=replace,drive=drive)
        dl.caseid = case
        dl.set_session(s2)
        dl.download_drive(dirlist,tags=tags,case=case)

    if action == 'extract':
        cs = Case(db,localdir=localdir)
        cs.extract(rmzip=True)

    if action == 'get_info':
        driver = ChromeDriver(aad=aad)
        driver.get_token()
        # attempt an oauth session for sharepoint
        s2 = OAuthSession(driver.token,aad=aad)
        driver.quit()
        dl = DataLake(db,localdir=localdir,replace=replace)
        dl.set_session(s2)
        dl.get_info(dirlist,case,'TDC Session Data')

    # removes any duplicates
    if action == 'prune':
        dl = DataLake(db,localdir=localdir,replace=replace)
        dl.prune_dl('SAG')
        dl.prune_dl('AX')

    # check/fix db values for a specific case or whole db
    if action == 'fix':
        if (site and case):
            cs = Case(dl,localdir,site=site,caseid=case)
            cs.fix('AX')
            cs.fix('SAG')
        else:
            cs = Case(dl,localdir=localdir)
            cs.fixall()                  

    # create crops and labels to form a set of training data
    if action == 'create_training':
        dl = TrainingData(db,localdir,outputdir,index=rowindex,N=None,mres=mres,dof=dof,crop=None,tag='5') # hard-coded tag
        dl.create_trainingdata()

    if action == 'create_test':
        dl = TrainingData(db,localdir,outputdir,index=None,N=None,mres=mres,dof=dof,crop=None)
        dl.create_testdata()
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl', default='dl')
    parser.add_argument('--action', default=None)
    parser.add_argument('--flags', default=None)
    parser.add_argument('--dirlist', help = 'list, or fullpath text file containing list of directories to transfer',default=None)
    parser.add_argument('--filelist', help = 'list, or fullpath text file containing list of specific files to transfer', default=None)
    parser.add_argument('--localdir',default=None)
    parser.add_argument('--outputdir',default=None)
    parser.add_argument('--replace',default="False")
    parser.add_argument('--case',default=None)
    parser.add_argument('--site',default=None)
    parser.add_argument('--drive',default='MR DICOM')
    parser.add_argument('--profile',default=False)
    parser.add_argument('--tags',default='DICOM')
    parser.add_argument('--user',default=None)
    parser.add_argument('--host',default='192.168.0.108')
    parser.add_argument('--pword',default=None)
    parser.add_argument('--rowindex',default=None)
    parser.add_argument('--mres',default=None)
    parser.add_argument('--dof',default='5')
    args = parser.parse_args()
    replace = eval(args.replace)

    if args.action == 'download':
        if not bool(args.dirlist) ^ bool(args.filelist):
            raise RuntimeError('Use either dirlist or filelist')
    if args.dirlist:
        args.dirlist = args.dirlist.split(',')
    if args.filelist:
        args.filelist = args.filelist.split(',')
    if args.tags:
        args.tags = args.tags.split(',')
    if args.rowindex is None:
        rowindex = None
    else:
        rowindex = int(args.rowindex)
    if args.mres:
        args.mres = tuple(map(int,args.mres.split(',')))
    else:
        raise RuntimeError('Specify mres')
    if args.pword is not None:
        pword = eval(args.pword)
    else:
        pword = None
    args.dof = int(args.dof)
    

    main(args.dl,args.action,
        flags=args.flags,
        dirlist=args.dirlist,
        drive=args.drive,
        site=args.site,
        case=args.case,
        filelist=args.filelist,
        localdir=args.localdir,
        outputdir=args.outputdir,
        rowindex=rowindex,
        mres=args.mres,
        dof=args.dof,
        replace=replace,
        profile=args.profile,
        tags = args.tags,
        user = args.user,
        host = args.host,
        pword = pword)
