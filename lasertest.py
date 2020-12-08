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



def getLickLatency(lickTimes,eventTimes,offset=0):
    firstLickInd = np.searchsorted(lickTimes,eventTimes+offset)
    noLicksAfter = firstLickInd==lickTimes.size
    firstLickInd[noLicksAfter] = lickTimes.size-1
    lickLat = lickTimes[firstLickInd]-eventTimes
    lickLat[noLicksAfter] = np.nan
    return lickLat


pklFile = fileIO.getFile('choose pkl file',fileType='*.pkl')
pkl = pd.read_pickle(pklFile)

params = pkl['items']['behavior']['params']
if params['periodic_flash'] is not None:
    flashDur,grayDur = params['periodic_flash']
    flashInterval = flashDur + grayDur

frameRate = 60
monitorLag = params['laser_params']['monitor_lag']/frameRate

trialLog = pkl['items']['behavior']['trial_log']
laserLog = pkl['items']['behavior']['layzer_trials']
changeLog = pkl['items']['behavior']['stimuli']['images']['change_log']
             
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
laserTrials = np.zeros(len(trialLog),dtype=bool)
laserOffset = np.full(len(trialLog),np.nan)
laserFrames = np.full(len(trialLog),np.nan)
for i,trial in enumerate(trialLog):
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
        for laser in laserLog:
            if 'actual_change_frame' in laser and 'actual_layzer_frame' in laser:
                if laser['actual_change_frame']==changeFrames[i]:
                    laserTrials[i] = True
                    laserOffset[i] = laser['actual_layzer_frame']-laser['actual_change_frame']
                    laserFrames[i] = laser['actual_layzer_frame']
                    break
    if len(trial['rewards']) > 0:
        rewardTimes[i] = trial['rewards'][0][1]
        autoReward[i] = trial['trial_params']['auto_reward']
        
frameIntervals = pkl['items']['behavior']['intervalsms']/1000
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
frameTimes += trialStartTimes[0] - frameTimes[int(trialStartFrames[0])]

lickFrames = pkl['items']['behavior']['lick_sensors'][0]['lick_events']
lickTimes = frameTimes[lickFrames]
        

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
offsets = np.concatenate((np.unique(laserOffset[~np.isnan(laserOffset)]),[np.nan]))
x = (offsets-1)/frameRate*1000
x[-1] = x[-2]*1.5
for trials,resp,clr,lbl in zip((changeTrials,catchTrials),(hit,falseAlarm),'kr',('hit','false alarm')):
    r = []
    for offset in offsets:
        i = trials & np.isnan(laserOffset) if np.isnan(offset) else trials & (laserOffset==offset)
        r.append(resp[i].sum()/i.sum())
        print(offset,resp[i].sum(),i.sum())
    ax.plot(x,r,'o-',color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1])
ax.set_xticks(x)
ax.set_xticklabels([int(i) for i in x[:-1]]+['no opto'])
ax.set_xlabel('Laser onset relative to change (ms)')
ax.set_ylabel('Response rate')
ax.legend()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for w in params['response_window']:
    ax.plot([0,x[-1]],[w*1000]*2,'--',color='0.75')
for trials,resp,clr,lbl in zip((changeTrials,catchTrials),(hit,falseAlarm),'kr',('hit','false alarm')):
    r = []
    for j,offset in enumerate(offsets):
        i = trials & resp & np.isnan(laserOffset) if np.isnan(offset) else trials & resp & (laserOffset==offset)
        r.append(1000*getLickLatency(lickTimes,changeTimes[i],params['response_window'][0]))
        ax.plot(x[j]+np.zeros(len(r[-1])),r[-1],'o',mec=clr,mfc='none')
    ax.plot(x,[np.nanmean(y) for y in r],'o',mec=clr,mfc=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,900])
ax.set_xticks(x)
ax.set_xticklabels([int(i) for i in x[:-1]]+['no opto'])
ax.set_xlabel('Laser onset relative to change (ms)')
ax.set_ylabel('Reaction time (ms)')
ax.legend()



syncFile = fileIO.getFile('choose sync file',fileType='*.h5')
syncDataset = sync.Dataset(syncFile)

frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'vsync_stim')
vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
frameAppearTimes = vsyncTimes + monitorLag

laserRising,laserFalling = probeSync.get_sync_line_data(syncDataset,channel=11)  

laserOnFromChange,laserOffFromChange = [t-vsyncTimes[changeFrames[laserTrials].astype(int)] for t in (laserRising,laserFalling)]

binWidth = 0.001

for t,xlbl in zip((laserRising,laserFalling),('onset','offset')):
    offset = t-vsyncTimes[changeFrames[laserTrials].astype(int)]-monitorLag
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(offset,bins=np.arange(round(min(offset),3)-0.001,round(max(offset),3)+0.001,binWidth))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from change on monitor to laser '+xlbl+' (s)')
    ax.set_ylabel('Count')


ind = changeTrials | catchTrials
startToChange = changeTimes[ind]-trialStartTimes[ind]
timeBetweenChanges = np.diff(changeTimes[ind])

for t,xlbl in zip((startToChange,timeBetweenChanges),('Time from trial start to change','Time between changes')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(t,bins=np.arange(0,t.max()+0.75,0.17))       
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel(xlbl+' (s)')
    ax.set_ylabel('Count')
        




