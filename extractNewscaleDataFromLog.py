# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:54:59 2019

@author: svc_ccg
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, glob

def findExcelFileForDate(date, raidPath = 'Z:', withNewScale=False):
    '''Date should be in format of pandas datetime for new scale log file (YYYY-MM-DD)
    Given date, function returns path to probePosCCF excel file on RAID'''
    
    dateParts = date.split('-')
    raidDateFormat = dateParts[1] + dateParts[2] + dateParts[0]
    
    raidFolder = glob.glob(os.path.join(raidPath, raidDateFormat + '*'))[0]
    if withNewScale:
        probeCCFExcelPath = glob.glob(os.path.join(raidFolder, 'probePosCCF*NewScale*'))[0]
    else:
        probeCCFExcelPath = glob.glob(os.path.join(raidFolder, 'probePosCCF*'))[0]
    
    return probeCCFExcelPath
    
    
def findInsertionStartStop(df):
    ''' Input: Dataframe from newscale log file for a specific date and probe serial number indexed by time stamps
            df is used to generate timeDeltas: Series datetime index of pandas dataframe (time between rows) in seconds
        OutPut: start and stop points where probe insertion is inferred to have started and stopped
        (based on pattern of many log entries at small time deltas)'''
    
    timeDeltas = tempdf.index.to_series().diff().astype('timedelta64[s]')
    #find the first time such that the next 20 deltas are all small
    try:    
        rollingDelta = timeDeltas.rolling(10, win_type='boxcar').sum().dropna()
        start = rollingDelta.where(rollingDelta<1).dropna().index[0]
        end = timeDeltas.loc[start:].where(timeDeltas.loc[start:]>1000).dropna().index[0] #find first point after start where time gap is long
        endind = timeDeltas.index.get_loc(end)
        if type(endind) == slice:
            endind = endind.start
        
        end = timeDeltas.index[endind-1] #take the time stamp right before that point as the end of insertion
        
        #now see if there are any retractions between start and end (since we may have repositioned)
        diff = df.loc[start:end, 'z']
        diff = diff.where(diff.diff()<-50).dropna()
        if len(diff)>0:
            start = diff.index[-1]
    except:
        start, end = timeDeltas.index[0], timeDeltas.index[0]
    
    return start, end

#Path to New Scale log file
logFile = r"Z:\newscale_log_08142019.txt"
logFile = r"Z:\log.txt"

#Make data frame from log file and index it by the time stamp
fulldf = pd.read_csv(logFile, header=None, names=['time', 'probeID', 'x', 'y', 'z', 'relx', 'rely', 'relz'])
fulldf['time'] = pd.to_datetime(fulldf['time'])
fulldf = fulldf.set_index('time')

#Look up table to find probe position from new scale manipulator serial number
serialToProbeDict = {' SN31212': 'A', ' SN34029': 'B', ' SN31058':'C', ' SN24272':'D', ' SN32152':'E', ' SN36800':'F'}

#Which date you want to extract in format 'YYYY-MM-DD'
plt.close('all')
dateOfInterest = '2019-08-29'
pdf = fulldf.loc[dateOfInterest]

#Get probeCCF excel file for this date as dataframe
excelPath = findExcelFileForDate(dateOfInterest)
exdf = pd.read_excel(excelPath)
exdf = exdf.set_index('Unnamed: 0')
exdf.index.names = ['']

for pSN in np.unique(pdf.probeID.values):
    pid = serialToProbeDict[pSN]
    tempdf = pdf.loc[pdf.probeID==pSN]
    
    #on one day, we had to reinsert the probes to fix the ground
    #ignore first insertion for this day
    if dateOfInterest == '2019-04-30':
        tempdf = tempdf.loc['2019-04-30 16':]
    
    fig = plt.figure(pSN + ': ' + pid, figsize=[12,5])
    ax1 = plt.subplot2grid([1,3], [0,0], colspan=2)
    tempdf.plot(y=['z', 'x', 'y'], ax=ax1)

    start, end = findInsertionStartStop(tempdf)
    ax2 = plt.subplot2grid([1,3],[0,2], colspan=1)
    tempdf.plot(y=['z', 'x', 'y'], ax=ax2)
    ax2.set_xlim([start - pd.Timedelta(minutes=1), end + pd.Timedelta(minutes=1)])
    insertiondf = tempdf.loc[start:end]
    
    for ax, title in zip([ax1,ax2], ['full day', 'insertion']):
        ax.set_title(title)
        ax.plot(start, insertiondf.iloc[0, 3], 'go')
        ax.plot(end, insertiondf.iloc[-1, 3], 'ro')
    
    print(pSN + ': ' + pid)
    print('Insertion start coords: ' + str(insertiondf.iloc[0, 0:4]))
    print('Insertion end coords: ' + str(insertiondf.iloc[-1, 0:4]))
    print('\n\n')
    
    for il,axis in enumerate(['x', 'y', 'z']):
        exdf.loc['newscale ' + axis, pid +' entry'] = insertiondf.iloc[0, il+1]
        exdf.loc['newscale ' + axis, pid+' tip'] = insertiondf.iloc[-1, il+1]
        exdf.loc['diff ' + axis, pid + ' tip'] = insertiondf.iloc[-1, il+1]-insertiondf.iloc[0, il+1]
    
    

with pd.ExcelWriter(excelPath.split('.')[0]+'_withNewScaleCoords.xlsx') as writer:
    exdf.to_excel(writer)
    writer.sheets['Sheet1'].set_column(0,0,15)



###Find correspondence between new scale coords and LFP based entry channels

datesOfInterest = [f for f in glob.glob(os.path.join('Z:', ('[0-9]'*8)+'_*'))]
zdiffs_all = []
entryChans_all = []
ccfDist_all = []
ids_all = []
for doi in datesOfInterest:
    
    excelPath = glob.glob(os.path.join(doi, 'probePosCCF*NewScale*'))
    if len(excelPath)>0:
        exdf = pd.read_excel(excelPath[0])
        exdf = exdf.set_index('Unnamed: 0')
        exdf.index.names = ['']
        
        entryChannels = exdf.loc['entryChannel', [p+' entry' for p in 'ABCDEF']]
        zdiffs = exdf.loc['diff z', [p + ' tip' for p in 'ABCDEF']]

        ids = [p+'_'+doi[2:] for p in 'ABCDEF']
        zdiffs_all.append(zdiffs)
        
        ccfCoordDiffs = [exdf[p + ' entry'].iloc[:3] - exdf[p + ' tip'].iloc[:3] for p in 'ABCDEF']
        ccfDist = [np.sum([c**2 for c in ccd])**0.5 for ccd in ccfCoordDiffs]
        ccfDist_all.append(ccfDist)
        
        entryChans_all.append(entryChannels)
        ids_all.append(ids)
    

ec = [e.values for e in entryChans_all]
dz = [d.values for d in zdiffs_all]  


disp = [d-e*10 for d,e in zip(dz, ec)]
disp_array = np.array(disp)

badMismatch = np.where(np.abs(disp_array)>400)
bad_ids = [ids_all[day][probe] for day,probe in zip(*badMismatch)]

ec_concat = []
[ec_concat.extend(10*e.values) for e in entryChans_all]

dz_concat = []
[dz_concat.extend(d.values) for d in zdiffs_all]

ccfd_concat = []
[ccfd_concat.extend(c) for c in ccfDist_all]

plt.figure()
plt.plot(ec_concat, dz_concat, 'ko')
plt.title('motor vs lfp')

plt.figure()
plt.plot(ec_concat, ccfd_concat, 'ko')
plt.title('ccf vs lfp')

plt.figure()
plt.plot(dz_concat, ccfd_concat, 'ko')
plt.title('ccf vs motor')




    