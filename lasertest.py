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
if params['periodic_flash'] is not None:
    flashDur,grayDur = params['periodic_flash']
    flashInterval = flashDur + grayDur

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
        
        
trialStartTimes = np.full(len(trialLog),np.nan)
trialStartFrames = np.full(len(trialLog),np.nan)
trialEndTimes = np.full(len(trialLog),np.nan)
abortedTrials = np.zeros(len(trialLog),dtype=bool)
abortTimes = np.full(len(trialLog),np.nan)
scheduledChangeTimes = np.full(len(trialLog),np.nan)
changeTimes = np.full(len(trialLog),np.nan)
changeFrames = np.full(len(trialLog),np.nan)
changeTrials = np.zeros(len(trialLog),dtype=bool)
catchTrials = np.zeros(len(trialLog),dtype=bool)
preChangeImage = ['' for _ in range(len(trialLog))]
changeImage = ['' for _ in range(len(trialLog))]
rewardTimes = np.full(len(trialLog),np.nan)
autoReward = np.zeros(len(trialLog),dtype=bool)
hit = np.zeros(len(trialLog),dtype=bool)
miss = np.zeros(len(trialLog),dtype=bool)
falseAlarm = np.zeros(len(trialLog),dtype=bool)
correctReject = np.zeros(len(trialLog),dtype=bool)
laserOffset = np.full(len(trialLog),np.nan)
for i,(trial,laser) in enumerate(zip(trialLog,laserLog)):
    events = [event[0] for event in trial['events']]
    for event,epoch,t,frame in trial['events']:
        if event=='trial_start':
            trialStartTimes[i] = t
            trialStartFrames[i] = frame
        elif event=='trial_end':
            trialEndTimes[i] = t
        elif event=='stimulus_window' and epoch=='enter':
            ct = trial['trial_params']['change_time']
            if params['periodic_flash'] is not None:
                ct *= flashInterval
                ct -= params['pre_change_time']-flashInterval
            scheduledChangeTimes[i] = t + ct
        elif 'abort' in events:
            if event=='abort':
                abortedTrials[i] = True
                abortTimes[i] = t
        elif event in ('stimulus_changed','sham_change'):
            changeTimes[i] = t
            changeFrames[i] = frame
        elif event=='hit':
            hit[i] = True
        elif event=='miss':
            miss[i] = True 
        elif event=='false_alarm':
            falseAlarm[i] = True
        elif event=='rejection':
            correctReject[i] = True
    if not abortedTrials[i]:
        if trial['trial_params']['catch']:
            catchTrials[i] = True
        else:
            if len(trial['stimulus_changes'])>0:
                changeTrials[i] = True
                preChangeImage[i] = trial['stimulus_changes'][0][0][0]
                changeImage[i] = trial['stimulus_changes'][0][1][0]
    if len(trial['rewards']) > 0:
        rewardTimes[i] = trial['rewards'][0][1]
        autoReward[i] = trial['trial_params']['auto_reward']
    if 'actual_layzer_frame' in laser:
        laserOffset[i] = laser['expected_layzer_frame_offset']
        
  

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


offsets = np.unique(laserOffset)
offsets = offsets[:np.where(np.isnan(offsets))[0][0]+1]
for offset in offsets:
    i = changeTrials
    i = i & np.isnan(laserOffset) if np.isnan(offset) else i & (laserOffset==offset)
    print(offset,hit[i].sum(),i.sum())








