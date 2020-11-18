# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:26:01 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileIO
from sync import sync
import probeSync


syncFile = fileIO.getFile('choose sync file',fileType='*.h5')
syncDataset = sync.Dataset(syncFile)

pklFile = fileIO.getFile('choose pkl file',fileType='*.pkl')
pkl = pd.read_pickle(pklFile)

params = pkl['items']['behavior']['params']

frameRate = 60

frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'vsync_stim')
vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
monitorLag = params['laser_params']['monitor_lag']/frameRate
frameAppearTimes = vsyncTimes + monitorLag

laserRising,laserFalling = probeSync.get_sync_line_data(syncDataset,channel=11)

trialLog = pkl['items']['behavior']['trial_log']
laserLog = pkl['items']['behavior']['layzer_trials']
changeLog = pkl['items']['behavior']['stimuli']['images']['change_log']

trialStartTimes = []
changeTimes = []
catchTrials = []
laserTrials = []
laserFrameTimes = []
for trial,laser in zip(trialLog,laserLog):
    catchTrials.append(trial['trial_params']['catch'])
    for event,epoch,t,frame in trial['events']:
        if event=='trial_start':
            trialStartTimes.append(vsyncTimes[frame])
        elif event in ('stimulus_changed','sham_change'):
            changeTimes.append(frameAppearTimes[frame])
    if 'actual_layzer_frame' in laser:
        laserTrials.append(True)
        laserFrameTimes.append(vsyncTimes[laser['actual_layzer_frame']])
    else:
        laserTrials.append(False)
        laserFrameTimes.append(np.nan)
  

laserOnFromChange,laserOffFromChange = [t-np.array(changeTimes)[laserTrials] for t in (laserRising,laserFalling)]

binWidth = 0.001

for t,xlbl in zip((laserRising,laserFalling),('onset','offset')):
    offset = t-np.array(changeTimes)[laserTrials]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(offset,bins=np.arange(round(min(offset),3)-0.001,round(max(offset),3)+0.001,binWidth))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from change on monitor to laser '+xlbl+' (s)')
    ax.set_ylabel('Count')


startToChange = np.array(changeTimes)-np.array(trialStartTimes)
timeBetweenChanges = np.diff(changeTimes)

for t,xlbl in zip((startToChange,timeBetweenChanges),('Time from trial start to change','Time between changes')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(t,bins=np.arange(0,t.max()+0.75,0.17))       
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel(xlbl+' (s)')
    ax.set_ylabel('Count')












