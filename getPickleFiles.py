# -*- coding: utf-8 -*-

"""

import getPickleFiles
mouseID = '423745'
limsDir = r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_784188286'
getPickleFiles.getPickleFiles(mouseID,limsDir)

"""

import os
import glob
import shutil
import urllib
import json


def getPickleFiles(mouseID,src=r'C:\ProgramData\camstim\output',dst=r'\\EphysRoom342\Data\behavior pickle files'):
    
    newDir = os.path.join(dst,mouseID)
    if not os.path.exists(newDir):
        os.mkdir(newDir)
     
    for f in glob.glob(os.path.join(src,'*_'+mouseID+'_*.pkl')):
        shutil.copy2(f,os.path.join(newDir,os.path.basename(f)))
     
    for i in os.listdir(src):
        ipath = os.path.join(src,i)
        if os.path.isdir(ipath):
            getPickleFiles(mouseID,ipath,dst)




#Get LIMS specimen ID 
mouse_ids =  (('429084',('07112019','07122019')),
             ('423744',('08082019','08092019')),
             ('423750',('08132019','08142019')),
             ('459521',('09052019','09062019')),
             ('461027',('09122019','09132019')),
             )

for mouse, ephysdates in mouse_ids:
    json_string = urllib.urlopen("http://lims2/specimens/isi_experiment_details/" + mouse + ".json").read()
    info = json.loads(json_string)            
    specimen_id = info[0]['id']
    
    limsDir = r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_' + specimen_id
    getPickleFiles(mouse,limsDir)
    
   