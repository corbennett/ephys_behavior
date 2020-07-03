# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:54:59 2019

@author: svc_ccg
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#import cPickle as pickle
import os


#Path to New Scale log file
logFile = r"\\10.128.50.43\sd6.3\1033388795_509652_20200630\1033388795_509652_20200630.motor-locs.csv"

#Look up table to find probe position from new scale manipulator serial number
serialToProbeDict = {' SN32148': 'A', ' SN32142': 'B', ' SN32144':'C', ' SN32149':'D', ' SN32135':'E', ' SN24273':'F'}
serialToProbeDict = {' SN34027': 'A', ' SN31056': 'B', ' SN32141':'C', ' SN32146':'D', ' SN32139':'E', ' SN32145':'F'}

#Date and time of experiment
dateOfInterest = '2020-06-30'
startTime = '0:00'  #I've set it to 12 am, only necessary to change if you did multiple insertions that day
                    #This script just finds the first insertion after the experiment time


#def findInsertionStartStop(df):
#    ''' Input: Dataframe from newscale log file for a specific date and probe serial number indexed by time stamps
#            df is used to generate timeDeltas: Series datetime index of pandas dataframe (time between rows) in seconds
#        OutPut: start and stop points where probe insertion is inferred to have started and stopped
#        (based on pattern of many log entries at small time deltas)'''
#    
#    timeDeltas = df.index.to_series().diff().astype('timedelta64[s]')
#    zDeltas = df['z'].diff().abs()
#    #find the first time such that the next 20 time and z deltas are all small
#    try:    
#        rolling_time_delta = timeDeltas.rolling(20, win_type='boxcar').mean().dropna()
#        rolling_z_delta = zDeltas.rolling(20, win_type='boxcar').mean().dropna()
#        start = rollingDelta.where(rollingDelta<5).dropna().index[0]
#        end = timeDeltas.loc[start:].where(timeDeltas.loc[start:]>1000).dropna().index[0] #find first point after start where time gap is long
#        endind = timeDeltas.index.get_loc(end)
#        if type(endind) == slice:
#            endind = endind.start
#        
#        end = timeDeltas.index[endind-1] #take the time stamp right before that point as the end of insertion
#        
#        #now see if there are any retractions between start and end (since we may have repositioned)
#        diff = df.loc[start:end, 'z']
#        diff = diff.where(diff.diff()<-50).dropna()
#        if len(diff)>0:
#            start = diff.index[-1]
#    except:
#        start, end = timeDeltas.index[0], timeDeltas.index[0]
#    
#    return start, end

def findInsertionStartStop(df):
    ''' Input: Dataframe from newscale log file for a specific date and probe serial number indexed by time stamps
            df is used to generate timeDeltas: Series datetime index of pandas dataframe (time between rows) in seconds
        OutPut: start and stop points where probe insertion is inferred to have started and stopped
        (based on pattern of many log entries at small time deltas)'''
    
    timeDeltas = df.index.to_series().diff().astype('timedelta64[s]')
    rolling_time_delta = timeDeltas.rolling(20, win_type='boxcar').mean().shift(-19).dropna()
    
    deltas = df[['z','x','y']].diff().abs() #get deltas for each axis
    rolling = deltas.rolling(20, win_type='boxcar').mean().shift(-19).dropna() #average over 20 steps and shift to left align
    #find the first time such that there are small steps in Z and no movement in X and Y AND the time steps are small
    try:    
        insertion = rolling.where((rolling['z']<10)&
                                              (rolling['z']>2)&
                                              (rolling['x']<1)&
                                              (rolling['y']<1)&
                                              (rolling_time_delta<2)).dropna()
     
        start = insertion.index[0]
        end = timeDeltas[start:][timeDeltas[start:]<10].index[-1]
        
    except:
        start, end = deltas.index[0], deltas.index[0]
    return start, end


#Make data frame from log file and index it by the time stamp
fulldf = pd.read_csv(logFile, header=None, names=['time', 'probeID', 'x', 'y', 'z', 'relx', 'rely', 'relz'])
fulldf['time'] = pd.to_datetime(fulldf['time'])
fulldf = fulldf.set_index('time')

#Find probe trajectories for this experiment
pdf = fulldf.loc[dateOfInterest]
datetimeinput = dateOfInterest.split('-')
datetimeinput.extend(startTime.split(':'))
startdatetime = pd.datetime(*[int(d) for d in datetimeinput])
pdf = pdf.loc[pdf.index>=startdatetime]
pcoordsDict = {}
for pSN in np.unique(pdf.probeID.values):
    pid = serialToProbeDict[pSN]
    tempdf = pdf.loc[pdf.probeID==pSN]
    
    fig = plt.figure(pSN + ': ' + pid, figsize=[12,5])
    ax1 = plt.subplot2grid([1,3], [0,0], colspan=2)
    tempdf.plot(y=['relz', 'relx', 'rely'], ax=ax1)

    start, end = findInsertionStartStop(tempdf)
    ax2 = plt.subplot2grid([1,3],[0,2], colspan=1)
    tempdf.plot(y=['relz', 'relx', 'rely'], ax=ax2)
    ax2.set_xlim([start - pd.Timedelta(minutes=1), end + pd.Timedelta(minutes=1)])
    insertiondf = tempdf.loc[start:end]
    
    for ax, title in zip([ax1,ax2], ['full day', 'insertion']):
        ax.set_title(title)
        ax.plot(start, insertiondf.iloc[0, 3], 'go')
        ax.plot(end, insertiondf.iloc[-1, 3], 'ro')
    
    print(pSN + ': ' + pid)
    print('Insertion start time: ' + str(insertiondf.index[0]))
    print('Insertion end time: ' + str(insertiondf.index[-1]))
    print('Insertion start coords: ' + str(insertiondf.iloc[0, 0:4]))
    print('Insertion end coords: ' + str(insertiondf.iloc[-1, 0:4]))
    print('\n\n')
    
    pcoordsDict[pid] = [insertiondf.iloc[0, 1:4].values, insertiondf.iloc[-1, 1:4].values, insertiondf.index[0], insertiondf.index[-1]]


#save coords
saveDir = r"Z:"
dateString = ''
dateString = dateString.join(dateOfInterest.split('-'))
filename = 'newScaleCoords_' + dateString + '.p'
with open(os.path.join(saveDir, filename), 'wb') as fp:
    pickle.dump(pcoordsDict, fp, protocol=pickle.HIGHEST_PROTOCOL)


#Load coords   
with open(os.path.join(saveDir, 'data.p'), 'rb') as fp:
    d = pickle.load(fp)
    