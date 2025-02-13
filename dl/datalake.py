# class and functions for local database of customer treatments

import os
import re
import uuid
import pandas as pd
import sqlite3 as sl
import zipfile
import tempfile,shutil
import pydicom
from dicom_parser import Image
from dicom_parser.utils.siemens.csa.header import CsaHeader
import difflib
import logging
import cProfile,pstats
from chromedriver import ChromeDriver
from session import OAuthSession

# class method decorator to troubleshoot slow methods
def profiler(func):
    def oncall(*args,**kwargs):
        p = cProfile.Profile()
        p.enable()
        res = func(*args,**kwargs)
        p.disable()
        s = pstats.Stats(p).sort_stats('tottime')
        s.print_stats(10)
        return res
    return oncall

class DataLake(object):

    def __init__(self,name=None,init=False,session=None,localdir=None,replace=False,drive=None):
        self.localdir = localdir
        self.name = name
        if replace:
            self.rm_dl()
        self.db = sl.connect(os.path.join(self.localdir,self.name+'.db'))
        if init:
            self.init_datalake()
        self.session=session
        self.currentdir = None
        self.log = logging.getLogger(__name__)
        self.drive = drive
        self.site = None
        self.caseid = None
        self.folder = None
        self.colnames = self.get_colnames()


# data base sql methods

    def init_datalake(self):
        with self.db:
            self.db.execute("""
                CREATE TABLE SAG (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
                    site INTEGER
                    case INTEGER
                    date TEXT
                    time TEXT
                    path TEXT
                    nslice INTEGER
                    slicethk FLOAT
                    rows INTEGER
                    cols INTEGER
                    px FLOAT
                    orient TEXT
                    slicepos TEXT
                    model TEXT
                    B0 FLOAT
                    comment TEXT
               );
            """)
            self.db.execute("""
                CREATE TABLE AX (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
                    site INTEGER
                    case INTEGER
                    date TEXT
                    time TEXT
                    path TEXT
                    nslice INTEGER
                    slicethk FLOAT
                    rows INTEGER
                    cols INTEGER
                    px FLOAT
                    orient TEXT
                    slicepos TEXT
                    model TEXT
                    B0 FLOAT
                    comment TEXT
                );
            """)
    def rm_dl(self):
        try:
            os.remove(os.path.join(self.localdir,self.name+'.db'))
            return
        except FileNotFoundError:
            return

    # remove duplicates
    def prune_dl(self,table):
        sqlcommand = """
            DELETE FROM {}
            WHERE rowid NOT IN (
                SELECT MIN(rowid) 
                FROM {} 
                GROUP BY site,caseid
                )
        """.format(table,table)
        self.query_dl(sqlcommand)

    # list any duplicates in entire db
    def duplicates_dl(self,table):
        sqlcommand = """
            SELECT path, COUNT(*)
            FROM {}
            GROUP by path
            HAVING COUNT(*) > 1
        """.format(table)
        res = self.query_dl(sqlcommand)
        for r in res:
            print(r)

    # check for specific record already existing
    def check_dl(self,table='AX'):
        sqlcommand = """
            SELECT EXISTS(
                SELECT 1
                FROM {}
                WHERE site = "{}"
                AND caseid = "{}"
            )
            """.format(table,self.site,self.caseid)
        res = self.query_dl(sqlcommand)[0]
        if isinstance(res,tuple):
            res = res[0]
        comment = ''
        # check if missing
        if res:
            sqlcommand = """
                    SELECT rowID
                    FROM {}
                    WHERE site = "{}"
                    AND caseid = "{}"
                """.format(table,self.site,self.caseid)
            row = self.query_dl(sqlcommand)[0]
            if isinstance(row,tuple):
                row = row[0]
            sqlcommand = """
                    SELECT comment FROM {} WHERE rowID = "{}"
                    """.format(table,row)
            comment = self.query_dl(sqlcommand)[0]
            if isinstance(comment,tuple):
                comment = comment[0]

        return (res,comment)

    # get all cases for a given site
    def getcases_dl(self):
        sqlcommand = """
            SELECT caseid
            FROM AX
            WHERE site = "{}"
            UNION
            SELECT caseid
            FROM SAG
            WHERE site = "{}"
        """.format(self.site,self.site)
        res = self.query_dl(sqlcommand)
        res = [item for t in res for item in t] # list of tuples to list
        return res

    def get_colnames(self,t='AX'):
        sqlcommand = """
            PRAGMA table_info({});
            """.format(t)
        res = self.query_dl(sqlcommand)
        colnames = [r[1] for r in res]
        return colnames

    # sqlite3 query
    def query_dl(self,sqlcommand,is_script=False):
        cur = self.db.cursor()
        res = None
        try:
            if is_script:
                cur.executescript(sqlcommand)
            else:
                cur.execute(sqlcommand)
            self.db.commit()
            res = cur.fetchall()
        except Warning as e:
            print(e)
        except sl.Error as e:
            print(e)
        except res is not None:
            print(res)
        finally:
            cur.close()
            return res


# convenience methods

    def set_session(self,s):
        self.session = s
        self.session.localdir = self.localdir
        return

    def get_newsession(self):
        driver = ChromeDriver()
        driver.get_token()
        s2 = OAuthSession(driver.token,driver.aad)
        self.set_session(s2)
        driver.quit()

    def set_drive(self,d):
        self.drive = d
        return

    def get_N(self):
        sqlcommand = """
            SELECT COUNT(ALL) FROM AX
            """
        res = self.query_dl(sqlcommand)
        return int(res[0][0])


    # general data lake methods

    # get record by offset from 1st record, or by site/case lookup
    def get_record(self,record,table='AX',site=None,case=None):
        if not (site or case):
            sqlcommand = """
                SELECT * FROM {} LIMIT 1 OFFSET {};
                """.format(table,record)
        else:
            self.site = site
            self.caseid = case
            sqlcommand = """
                SELECT * FROM {} WHERE site="{}" AND caseid="{}"
                """.format(table,self.site,self.caseid)
        res = self.query_dl(sqlcommand)
        if len(res) == 0:
            self.log.info('no such record {}'.format(record))
            raise KeyError('no such record in db')
        r = dict(zip(self.colnames,res[0]))

        return r

    # read a dicom volume from db
    def get_dbvol(self,table,reverse=True):
        outputdir = table + '_T2'
        voldir = os.path.join(self.localdir,self.site,self.caseid,outputdir)
        series_filenames = [os.path.join(fpath,f) for (fpath,_,fnames) in os.walk(voldir) for f in fnames]
        vol = []
        for f in series_filenames:
            vol.append(pydicom.dcmread(f))
        # there is one extra slice in some volumes that is extra/meta in some way. need to look this up.
        # meanwhile this is just a kludge for now to reselect only the slices with the tag.
        vol = [v for v in vol if 'SliceLocation' in v]
        if len(vol) == 0:
            self.log.error('No SliceLocation tag in site {}, case {}'.format(self.site,self.caseid))
            raise KeyError('no SliceLocation tag')
        vol.sort(key = lambda x:int(x.SliceLocation),reverse=reverse)
        return vol


    # sharepoint methods

    # download one/all cases from all sites contained in dirlist
    # currently a specified case only works for 1 dir in the dirlist
    def download_drive(self,dirlist,tags=['DICOM'],case=None):
        for dir in dirlist: # COM01 etc
            self.log.info('downloading {}'.format(dir))
            if dir not in self.session.sites:
                self.log.error('site {} not found on {}'.format(dir,self.session.sitesurl_long))
                continue
            self.session.set_siteid(dir + '/?$select=id')
            self.session.set_drives()
            self.session.set_drive(self.drive)
            self.session.set_cases()
            # get list of cases already in db
            self.site = self.get_sitecase_site(dir)
            localcaselist = self.getcases_dl()
            # list of cases to download
            if case is None: # take all cases
                caselist = self.session.cases.keys()
            else: # take a specified case
                case = '{:03d}'.format(int(self.caseid))
                sitecase_name = self.site+'_01-'+case
                caselist = difflib.get_close_matches(sitecase_name,self.session.cases.keys())
                if len(caselist) == 0:
                    self.log.error('No match found for site {} case {}'.format(self.site,case))
                    continue
                elif len(caselist) > 1:
                    self.log.info('Multiple matches found for site {} case {}'.format(self.site,case))
                    
            for c in caselist:
                self.folder = self.session.cases[c]
                (_,cid,sitecase_name) = self.get_sitecase_drive(c)
                if c in localcaselist:
                    continue
                self.log.info('{}'.format(c))
                try:
                    flist = self.session.get_itemnames(self.folder,sitecase_name)
                except RuntimeError as e:
                    self.log.error(str(e)+', skipping {}'.format(c))
                    continue
                if len(flist) == 0:
                    self.log.error('No zip file for case{}, skipping'.format(c))
                    # raise ValueError('No zip file for case {}'.format(c))
                    continue
                else:
                    for k in flist.keys():
                        zipfilename = self.select_itemname(flist[k],c,sitecase_name,tags=tags)
                        self.log.debug('Using this one: {}'.format(zipfilename))

                        for i,z in enumerate(zipfilename):
                            try:
                                zipfile = self.session.get_item(z,k)
                            except TimeoutError as e:
                                self.log.error(str(e)+', skipping {}, {}'.format(c,z))
                            except RuntimeError as e:
                                # try restarting the session
                                self.get_newsession()
                                try:
                                     zipfile = self.session.get_item(z,k)
                                except RuntimeError as e:
                                    pass
                                self.log.error(str(e)+', skipping {}, {}'.format(c,z))
                            with open(os.path.join(self.localdir,'{}-{}-{}.zip'.format(c,k,i)),'wb') as dfile:
                                dfile.write(zipfile.content)       

    # try to read zip info from a remote zip file
    def get_info(self,dir,case,drive,zlist=None):
        if isinstance(dir,list):
            dir = dir[0]
        (_,_,sitecase_name) = self.get_sitecase_site(dir,case)
        self.set_sharepoint(dir,drive)
        closest = difflib.get_close_matches(sitecase_name,self.session.cases.keys())
        self.session.get_iteminfo(closest)
        True

    def set_sharepoint(self,dir,drive):
        self.session.set_siteid(dir + '/?$select=id')
        self.session.set_drives()
        self.session.set_drive(drive)
        self.session.set_cases()

    # select one zip file if several present in folder
    def select_itemname(self,zlist,folder,sitecase_name,tags=[]):
        # try the right one has a DICOM in the filename, or closest to basic site-case string, otherwise just grab the first
        zlist_tag = [z for z in zlist for tag in tags if z.find(tag)>-1]
        zlist_closest = difflib.get_close_matches(sitecase_name+'.zip',zlist)
        zlist_startswith = [z for z in zlist if z.startswith(sitecase_name)]
        zlist_endswith = [z for z in zlist if z.endswith('.zip')]
        if len(zlist_tag) == 1:
            zipfilename = zlist_tag[0:]
        elif len(zlist_closest) > 0:
            zipfilename = zlist_closest[0:]
        elif len(zlist_startswith) > 0:
            zipfilename = zlist_startswith[0:]
        elif len(zlist_endswith) > 0: # if no other matches, just return all zip files
            zipfilename = zlist_endswith
        else:
            self.log.debug('No close match in {}, choosting {}'.format(zlist,zlist[0]))
            zipfilename = zlist[0:]
        return zipfilename

    # handle variations on protocol names
    def resolve_pnames(self,pnames):
        pnames = [p.upper() for p in pnames]
        pnames_mod=[]
        lastp = []
        for p in pnames:
            # hard-coded list of known variants 
            p = p.replace('T2W','T2')
            p = p.replace('T1W','T1')
            p = re.sub('(TRA(NSVERSE){0,1}|AXIAL)','AX',p)
            p = re.sub('SAG(ITTAL){0,1}','SAG',p)
            if ' ' in p and not '_' in p: # not sure how general space/underscores should be yet
                p = p.replace(' ','_')
            else:
                p = p.replace(' ','')
            pparts = p.split('_')
            pnew = []
            for ppat in ['(SAG|AX)','(T1|T2)','(2D|3D)']:
                ps = re.search(ppat,p)
                if ps is None:
                    if p != lastp: # awkward, processing one protocol name per dicom image, massive repetition
                        print('{} not found in protocol {}'.format(ppat,p))
                else:
                    pnew.append(ps[0])
                    if ps[0] in pparts:
                        pparts.remove(ps[0])
            if len(pparts)>0:
                [pnew.append(p) for p in pparts]
            pnew = ' '.join(pnew) # space or underscores?
            pnames_mod.append(pnew)
            lastp = p
        return pnames_mod

    # get site from Com?? sharepoint site naming convention
    # optionally format a case number, and return a combo site/case string
    def get_sitecase_site(self,sitedir,sitecase=None):
        siteid = '{:03d}'.format(int(re.match('.*?([0-9]{2,3})',sitedir)[1]))
        if sitecase is None:
            return siteid
        else:
            caseid = '{:03d}'.format(int(sitecase))
            sitecase_name = siteid+'_01-'+caseid
            return (siteid,caseid,sitecase_name)

    # get site/case from a dirname in a sharepoint drive (MR DICOM)
    def get_sitecase_drive(self,dicomdir):
        siteid,caseid = re.search('([0-9]{3})....([0-9]{3})',dicomdir).group(1,2)
        sitecase_name = siteid+'_01-'+caseid
        return (siteid,caseid,sitecase_name)


    # vpn methods

    # download one/all cases from all sites contained in dirlist
    # currently a specified case only works for 1 dir in the dirlist
    # tact sites are prefixed with '9' in db due to sharepoint overlap. this kludge creates awkward
    # conversions depending on whether new downloads from network or reads from local db are being attempted
    # session.site is 3 digit, datalake.site is '9'+3digit
    def download_drive_vpn(self,dirlist,tags=['DICOM'],case=None):
        for dir in dirlist: # network drive 3 digit '005' etc
            self.log.info('downloading site {}'.format(dir))
            if dir not in self.session.sites:
                self.log.error('site {} not found on {}'.format(dir,self.session.root))
                continue
            self.session.site = dir
            self.site = '9'+dir
            self.session.set_cases()
            localcaselist = self.getcases_dl()
            if case is None: # take all cases
                caselist = self.session.cases
            else: # take a specified case
                case = '{:03d}'.format(int(self.caseid)) # redundant
                if case in self.session.cases.keys():
                    caselist = [case]
                else:
                    self.log.error('No match found for site {} case {}'.format(self.session.site,case))
                    continue
            for c in caselist:
                self.folder = self.session.dirlist[self.session.cases[c]]      
                if c in localcaselist:
                    continue
                self.log.info('case {}'.format(c))

                dcm_folder = self.session.set_folder(self.folder)
                # TODO test for zipped Session dirs on TACT same as on sharepoint, there are a few
                if dcm_folder is None:
                    self.log.error('No Anatomy folder in {}'.format(self.folder))
                    continue
                # modifying site id here due to overlap with sharepoint
                localfolder = re.sub(r'([0-9]{3})-',r'9\1-',self.folder)
                localpath = os.path.join(self.localdir,'zips','{}.zip'.format(localfolder))
                try:
                    self.session.get_folder(dcm_folder,localpath)
                except PermissionError:
                    self.log.info('skipping')
                except OSError:
                    self.log.info('skipping')
                except NotADirectoryError:
                    self.log.info('skipping')


