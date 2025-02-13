import logging
import os
import pandas as pd
import sqlite3 as sl
import pydicom
import uuid
import re
import zipfile
import tempfile,shutil
import difflib
from dicom_parser import Image
from dicom_parser.utils.siemens.csa.header import CsaHeader
from datalake import DataLake

class Case(DataLake):

    def __init__(self,name,localdir,table=None,site=None,caseid=None):
        DataLake.__init__(self,name,localdir = localdir)
        self.data = dict.fromkeys(['id','site','caseid','date','time','path','nslice','slicethk','rows','cols','orient','slicepos','px','model','B0','comment'], None)
        self.data['site'] = site
        self.data['caseid'] = caseid
        self.site = site
        self.caseid = caseid
        # self.db = db
        if table is not None:
            self.table = table
            self.outputdir = self.table+'_T2'
        self.log = logging.getLogger(__name__)
        # self.localdir = localdir
        self.vol = None
        self.missing = False
        self.replace = False
 
    def cleardata(self):
        self.data = dict.fromkeys(['id','site','caseid','date','time','path','nslice','slicethk','rows','cols','orient','slicepos','px','model','B0','comment'], None)

    def setvalues(self,vol=None):
        self.data['site'] = self.site
        self.data['caseid'] = self.caseid
        # TODO: is vol sorted by slice number yet?
        if vol:
            dcmseries=vol
        else:
            dcmseries=self.vol
        # couple variations on px seen so far
        if 'PixelSpacing' in dcmseries[0]:
            px = dcmseries[0].PixelSpacing
            if len(px) > 1:
                px = px[0]
            else:
                px = px.split('\\')[0]
        else:
            px = None
        # two variations for slice seen so far
        if 'SpacingBetweenSlices' in dcmseries[0]:
            slicethk = dcmseries[0].SpacingBetweenSlices
        elif 'SliceThickness' in dcmseries[0]:
            slicethk = dcmseries[0].SliceThickness
        else: # StackSequence 2001,105f has this information on some studies. may be in a raw hex string though
            slicethk = None
        # do some checks on the angles and offsets which are needed for the AX series and are sometimes 
        # found in nested private tags
        # strange kludge. some series have an extra dicom slice. it has a duplicate instanceNumber, but no SliceLocation tag
        # to filter out these extra slices, first filter on SliceLocation, then sort by InstanceNumber
        if self.table == 'AX':
            dcmseries = [v for v in dcmseries if 'SliceLocation' in v]
            if len(dcmseries) == 0:
                self.log.error('No SliceLocation tag in site {}, case {}'.format(self.site,self.caseid))
                raise KeyError('no SliceLocation tag')
            # sort on InstanceNumber tag, and record ImagePositionPatient from 0th slice in that order
            dcmseries.sort(key = lambda x:float(x.SliceLocation),reverse=True)
            if 'ImageOrientationPatient' in dcmseries[0]:
                porient = ','.join(list(map(str,dcmseries[0].ImageOrientationPatient)))
            else:
                # found this example at [0x20,0x9916]. there's another at [0x2005,0x140f][0][0x20,0x32]
                # but no obvious way to find the tag programmatically.
                if 'PerFrameFunctionalGroupsSequence' in dcmseries[0]:
                    try:
                        porient = ','.join(list(map(str,dcmseries[0].PerFrameFunctionalGroupsSequence[:][0][0x20,0x9916][0])))
                    except KeyError:
                        raise KeyError('no ImageOrientationPatient at [0x0020,0x9916]')
                else:    
                    self.log.warning('site {} case {} {} has no orient'.format(self.site,self.caseid,self.table))
                    raise AttributeError('site {} case{} {} has no orient'.format(self.site,self.caseid,self.table))
            if 'ImagePositionPatient' in dcmseries[0]:
                ppos = ','.join(list(map(str,dcmseries[0].ImagePositionPatient)))
            else:
                if 'PerFrameFunctionalGroupsSequence' in dcmseries[0]:
                    try:
                        ppos = ','.join(list(map(str,dcmseries[0].PerFrameFunctionalGroupsSequence[:][0][0x20,0x9913][0])))
                    except KeyError:
                        raise KeyError('no ImagePositionPatient at [0x0020,0x9916]')
                else:    
                    self.log.warning('site {} case {} {} has no position'.format(self.site,self.caseid,self.table))
                    raise AttributeError('site {} case{} {} has no position'.format(self.site,self.caseid,self.table))
        # TODO: check for missing pos,orient are also needed for sag
        elif self.table == 'SAG':
            # not sure why, but some images don't have SliceLocation tag??
            dcmseries = [v for v in dcmseries if 'SliceLocation' in v]
            if len(dcmseries) == 0:
                self.log.error('No SliceLocation tag in site {}, case {}'.format(self.site,self.caseid))
                raise KeyError('no SliceLocation tag')
            # philips and siemens ahve different voxel slice coordinates according to k-space traversal.
            # for purposes of cropping a 3d volume to make training data, will have all coordinates to be referenced to the
            # ImagePositionPatient of the 0th slice, InstanceNumber=1. The sign of the dircos z vector will be assigned
            # later to correct for k-space traversal in during image cropping
            dcmseries.sort(key = lambda x:float(x.SliceLocation),reverse=True)
            if 'ImageOrientationPatient' in dcmseries[0]:
                porient = ','.join(list(map(str,dcmseries[0].ImageOrientationPatient)))
            else:
                porient = None
            if 'ImagePositionPatient' in dcmseries[0]:
                ppos = ','.join(list(map(str,dcmseries[0].ImagePositionPatient)))
            else:
                ppos = None
        # rest of the tags no special checks
        try:
            self.data['date'] = dcmseries[0].InstanceCreationDate
            self.data['time'] = dcmseries[0].InstanceCreationTime
            self.data['nslice'] = max([d.InstanceNumber for d in dcmseries])
            self.data['slicethk'] = slicethk
            self.data['rows'] = dcmseries[-1].Rows
            self.data['cols'] = dcmseries[-1].Columns
            self.data['orient'] = porient
            self.data['slicepos'] = ppos
            self.data['px'] = px
            self.data['model'] = dcmseries[0].ManufacturerModelName
            self.data['B0'] = dcmseries[0].MagneticFieldStrength
            self.data['path'] = os.path.join(self.localdir,self.site,self.caseid,self.table+'_T2')
        except AttributeError:
            raise AttributeError('missing other tags')

    def createrecord(self):
        self.rec = pd.DataFrame(data=self.data,index=['0'])
        if self.replace:
            self.rec.to_sql(self.table,self.db,index=False,if_exists='append',method=self.update_record)
        else:
            self.rec.to_sql(self.table,self.db,index=False,if_exists='append')

    def update_record(self,table,conn,keys,data_iter):
        values = list(data_iter)[0] # this returns a tuple
        values = [str(v) for v in values] # any numerics to str. this turns None into str too
        vlist,klist = zip(*list(filter(lambda x: x[0] != 'None', zip(values,keys)))) # select only values that aren't 'None'
        keystr =','.join(klist)
        valstr = '\",\"'.join(vlist)
        sqlcommand = """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_site_case_{}
            ON {} (site,caseid);
            INSERT OR REPLACE INTO {} ({}) VALUES ("{}")
        """.format(self.table,self.table,self.table,keystr,valstr)
        res = self.query_dl(sqlcommand,is_script=True)

    def setvol(self,vol):
        # not sure if sort needed here or only in setvalues()
        # vol.sort(key = lambda x:int(x.SliceLocation)) # or InstanceNumber, reverse or not
        self.vol = vol
        self.check_uid()

    # read a dicom volume from db
    def get_dbvol(self):
        voldir = os.path.join(self.localdir,self.site,self.caseid,self.outputdir)
        series_filenames = [os.path.join(fpath,f) for (fpath,_,fnames) in os.walk(voldir) for f in fnames]
        vol = []
        for f in series_filenames:
            vol.append(pydicom.dcmread(f))
        self.setvol(vol)

    def setmissing(self):
        self.missing=True

    def settable(self,t):
        self.table = t
        self.outputdir = self.table + '_T2'

    def getmissing(self):
        return self.missing

    # kludge for Series UID if missing
    def check_uid(self):
        # various formats for image filenames in sharepoint
        try:
            uid = self.vol[0].FrameOfReferenceUID
            uidroot = '.'.join(uid.split('.')[0:8])+'.' # guessing 1st 8 fields are root
        except AttributeError:
            uidroot = str(uuid.getnode())
            self.log.warning('Assigning arbitrary uidroot {}'.format(uidroot))
        uidstr = str(uuid.uuid4())
        for s in self.vol:
            if 'SeriesInstanceUID' not in s:
                s.add_new('0x20000e','UI',uidroot+uidstr)

    def setdata(self,**kwargs):
        for k in kwargs:
            self.data[k] = kwargs[k]
        self.site = self.data['site'] # convenience attributes
        self.caseid = self.data['caseid']

    def write_series(self):
        mkdir = os.path.join(self.localdir,self.site,self.caseid,self.outputdir)
        try:
            os.makedirs(mkdir)
        except FileExistsError:
            self.log.info('Directory {} already exists, over-writing images'.format(mkdir))
        for s in self.vol:
            fname = '{}_01-{}.MR.{:04d}.{:04d}.{}.{}.IMA'.format(self.site,self.caseid,s.SeriesNumber,s.InstanceNumber,s.SeriesDate,s.SeriesTime)
            pydicom.dcmwrite(os.path.join(self.localdir,self.site,self.caseid,self.outputdir,fname),s)
        return True

    # add a series to db
    def add_series(self):
        if self.missing:
            self.setdata(comment='missing')
            self.setdata(site=self.site)
            self.setdata(caseid=self.caseid)
        else:
            try:
                self.setvalues()
            except (AttributeError,KeyError) as e:
                self.setdata(comment=str(e))
        self.createrecord()

    # reread record in given table in the db from dcms at the path location
    def fix(self,table):
        self.replace = True
        self.settable(table)
        self.get_dbvol()
        self.add_series()

    # fix all records in db
    def fixall(self):
        N = self.get_N()
        for t in ['SAG','AX']:
            for r in range(N): 
                try:
                    record = self.get_record(0,t) # get first record in rowid. need better index
                except KeyError:
                    break
                if record['comment']: # ie any entry in the missing column
                    self.setdata(**record)
                    self.createrecord() # rewrite as is
                    continue
                rpath = record['path']
                rsite = record['site']
                rcase = record['caseid']
                if not rpath:
                    self.setdata(**record)
                    self.setdata(comment='missing')
                    self.createrecord() # rewrite as is
                    continue
                # check for consistency path/site/case
                if rsite in rpath and rcase in rpath:
                # reload data from image dicms
                    self.cleardata()
                    self.setdata(site=rsite,caseid=rcase)
                    self.fix(t)
                else:
                    raise ValueError('site {} case{} do not match path {}'.format(rsite,rcase,rpath))
            a=1
        return True

    def get_case(self):
        # try for TDC session first
        # assumes that even if the MR DICOM folder is nested, the 2d series will be labelled
        # more like Ax T2, or indeed anything but the TDC specific Anatomy2D
        if self.table == 'AX':
            vol = self.get_seriesdir('Anatomy2D')
            if len(vol) == 0:
                vol = self.get_series('AX T2',sequence='*h2d1')
        elif self.table == 'SAG':
            vol = self.get_seriesdir('Anatomy3D')
            if len(vol) == 0:
                vol = self.get_series('SAG T2',sequence='*h2d1')
        # alternate version for tdc logs
        if len(vol) == 0:
            msg = 'No {} T2 found {}'.format(self.table,self.zfile.filename)
            self.log.warning(msg)
            self.setdata(comment=msg)
            self.setmissing()
        else:
            self.setvol(vol)
        try:
            self.add_series()
        except (KeyError,AttributeError):
            self.setmissing()
            self.log.error('site {} case {} missing tags'.format(self.site,self.caseid))
            self.add_series()
        if not self.getmissing():
            self.write_series()

    # main method for sharepoint zip files
    # for all zip files in dir, pull Sag T2 3D and Ax T2 series
    # no DICOMDIR is available from TDC export?
    def extract(self,dir='zips',case=None,zlist=None,rmzip=False):
        cwd = os.path.join(self.localdir,dir)
        if zlist is None:
            zlist = [os.path.join(cwd,f) for f in os.listdir(cwd) if f.endswith('.zip')]
        else:
            zlist = [os.path.join(cwd,f) for f in zlist]
        if len(zlist) == 0:
            raise ValueError('No .zip files found in {}'.format(cwd))
            
        for z in zlist:
            # assuming site/case are included in the filename here, as 3 digit numbers
            self.site,self.caseid,_ = self.get_sitecase_drive(z)
            if self.site==None or self.caseid==None:
                self.log.error('site and/or case not matched in {}'.format(z))
                continue
            self.currentdir = os.path.join(self.localdir,self.site,self.caseid)
            if case is not None: # get just one case
                if case != self.caseid:
                    continue

            # check for duplicate data already
            (res,comment) = self.check_dl()
            if res > 0:
                if comment == 'missing':
                    self.replace = True
                else:
                    self.log.error('site {} case {} already in database'.format(self.site,self.caseid))
                    if rmzip:
                        os.remove(z)
                    continue
            
            # process zipfile
            try:
                with zipfile.ZipFile(z) as self.zfile:
                    print('zfile={}'.format(self.zfile.filename))
                    self.ztemp = tempfile.mkdtemp()
                    self.zfile.extractall(self.ztemp)

                    # try to get the ax, sag series
                    # TODO: if ax t2 missing, can try check the Therm
                    # TODO: need to make AX and SAG are matching in FrameUID
                    self.settable('AX')
                    self.get_case()

                    self.settable('SAG')
                    self.get_case()

                    # remove zipfiles
                    shutil.rmtree(self.ztemp)
                    if rmzip:
                        os.remove(self.zfile.filename)
            except zipfile.BadZipFile as e:
                # should store this result so same broken files don't keep redownloading 
                self.log.error('{}, skipping {}'.format(e,z))

    # method for picking series when dir structure is known (ie tdc session logs)
    def get_seriesdir(self,series):
        series_filenames = [os.path.join(self.ztemp,z) for z in self.zfile.namelist() if z.find(series)>-1]
        s = []
        if len(series_filenames) > 0:
            for f in series_filenames:
                if os.path.isfile(f):
                    if pydicom.misc.is_dicom(f):
                        s.append(pydicom.dcmread(f))

        # check for multiple series and take last
        if len(s) == 0:
            self.log.error('No dcms for series {}'.format(series))
            return []
        s.sort(key = lambda x:int(x.SeriesNumber))
        Series = s[-1].SeriesNumber
        vol = [v for v in s if v.SeriesNumber==Series and 'InstanceNumber' in v]
        vol.sort(key = lambda x:int(x.InstanceNumber))
        return vol

    # klunky method for picking last of several possible series reps in a possibly flat dir
    # TODO better confirmation of match between Ax, Sag when there are multiples
    # @profiler
    def get_series(self,series,sequence=None):
        # simple first try to find labelled subdirs. need to search dir names more generally for this to be useful
        # should use get_seriesdir here instead
        axnames = [os.path.join(self.ztemp,z) for z in self.zfile.namelist() if z.upper().find(series)>-1]
        if len(axnames) > 0:
            ax=[]
            for f in axnames:
                if os.path.isfile(f):
                    if pydicom.misc.is_dicom(f):
                        ax.append(pydicom.dcmread(f))
        # for flat directory containing all series, or unmatched subdirs hierarchy. search all files by tag
        # needs tidying
        elif len(axnames) == 0:
            ax = []
            protocolnames = []
            znames = []
            useSeriesUID = None
            useCSAProtocolName = None
            for z in self.zfile.namelist():
                z = os.path.join(self.ztemp,z)
                if os.path.isfile(z) and not z.endswith('DICOMDIR'): # sometimes DICOMDIR is present, sometimes not
                    if pydicom.misc.is_dicom(z):
                        d = pydicom.dcmread(z,stop_before_pixels=True)
                        if 'ProtocolName' in d:
                            protocolnames.append(d.ProtocolName.upper())
                        elif 'SeriesDescription' in d:
                            protocolnames.append(d.SeriesDescription.upper())
                        elif 'Manufacturer' in d and d.Manufacturer == 'SIEMENS': # try for a csa
                            # first check for a current Series UID
                            if (getattr(d,'SeriesInstanceUID',0) == useSeriesUID or
                                getattr(d,'FrameOfReferenceUID',0) == useSeriesUID or 
                                getattr(d,'SeriesTime',0) == useSeriesUID or
                                getattr(d,'SeriesNumber',0) == useSeriesUID):
                                protocolnames.append(useCSAProtocolName)
                            # otherwise parse the csa
                            else:
                                useSeriesUID = None
                                dparse = Image(z)
                                try:
                                    csa = CsaHeader(dparse.header.get('CSASeriesHeaderInfo',parsed=False)).parse()
                                except AttributeError:
                                    # self.log.info('No CSA header in {}'.format(z))
                                    continue
                                if 'ProtocolName' in csa:
                                    protocolnames.append(csa['ProtocolName'])
                                    # save the series UID for all the rest of the slices in this series
                                    if 'SeriesInstanceUID' in d:
                                        useSeriesUID = d.SeriesInstanceUID
                                        useCSAProtocolName = csa['ProtocolName']
                                    elif 'SeriesNumber' in d:
                                        useSeriesUID = d.SeriesNumber
                                        useCSAProtocolName = csa['ProtocolName']
                                    elif 'FrameOfReferenceUID' in d:
                                        useSeriesUID = d.FrameOfReferenceUID
                                        useCSAProtocolName = csa['ProtocolName']
                                    elif 'Therm' in csa['ProtocolName']:
                                        if 'SeriesTime' in d:
                                            useSeriesUID = d.SeriesTime
                                            useCSAProtocolName = csa['ProtocolName']
                                else:
                                    continue
                        else:
                            # maybe try match on sequence name
                            self.log.error('No protocol name in {}'.format(z))
                            continue
                        znames.append(z)
                        # check for exact series match
                        if protocolnames[-1].upper().find(series) > -1: 
                            ax.append(pydicom.dcmread(z))
        # no exact series matches
        if len(ax) == 0:
            # process protocolnames for known variants
            protocolnames = self.resolve_pnames(protocolnames)
            closest = difflib.get_close_matches(series,protocolnames)
            if len(closest) == 0:
                self.log.warning('No close matches of {} in {}'.format(series,set(protocolnames)))
                return []
            elif len(closest) > 1:
                # TODO: need to put in a check between AX and SAG to ensure last matches last
                self.log.warning('Multiple close matches of {} in {}, taking last'.format(series,closest))
                c = closest[-1]
            else:
                c = closest[0]
            if c.find(series) > -1:
                # re-iterate the saved list of names
                for (z,p) in list(zip(znames,protocolnames)):
                    if p == c:
                        ax.append(pydicom.dcmread(z))
            else:
                self.log.error('No {} images in {}'.format(series,zfile.filename))
                return []

        # old code. could select based on sequence nmae
        if False:
            sequencenames = set([s.SequenceName for s in ax if 'SequenceName' in s])
            sequencenames = [s.split('_')[0] for s in sequencenames]
            if sequence in sequencenames:
                ax = [s for s in ax if 'SequenceName' in s and s.SequenceName.startswith(sequence)]
                if len(ax) == 0:
                    print('No {} images in {}'.format(sequence,zfile.filename))
                    # return False
            else:
                print('{} not in {}'.format(sequence,sequencenames))
                # return False

        ax.sort(key=lambda x:int(x.SeriesNumber))
        Series = ax[-1].SeriesNumber # take last series if several
        ax = [s for s in ax if s.SeriesNumber == Series and 'InstanceNumber' in s]
        # final sort by image number
        ax.sort(key=lambda x:int(x.InstanceNumber))
        return ax


    # get site/case from a dirname, for either TACT or sharepoint
    # override/duplicate datalake method which is only for sharepoint
    def get_sitecase_drive(self,dicomdir):
        pattern = ['([0-9]{3})....([0-9]{3})','([0-9]{4}).([0-9]{3})']
        for p in pattern:
            item = re.search(p,dicomdir)
            if item is not None:
                siteid,caseid = item.group(1,2)
                sitecase_name = siteid+'_01-'+caseid # still hard-coded for shapreoint
                return (siteid,caseid,sitecase_name)
        self.log.error('site/case not matched {}'.format(dicomdir))
        return
