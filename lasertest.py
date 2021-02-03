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


frameRate = 60

pklFiles = []
while True:
    f = fileIO.getFile('choose pkl file',fileType='*.pkl')
    if f!='':
        pklFiles.append(f)
    else:
        break

expDate = []
params = []   
trialStartTimes = []
trialStartFrames = []
trialEndTimes = []
abortedTrials = []
abortTimes = []
scheduledChangeTimes = []
changeTimes = []
changeFrames = []
changeTrials = []
catchTrials = []
preChangeImage = []
changeImage = []
rewardTimes = []
autoReward = []
hit = []
miss = []
falseAlarm = []
correctReject = []
laser = []
laserOffset = []
laserFrames = []
lickTimes = []
        
for f in pklFiles:       
    pkl = pd.read_pickle(f)
    expDate.append(str(pkl['start_time'].date()))

    params.append(pkl['items']['behavior']['params'])
    if params[-1]['periodic_flash'] is not None:
        flashDur,grayDur = params[-1]['periodic_flash']
        flashInterval = flashDur + grayDur
    
    monitorLag = params[-1]['laser_params']['monitor_lag']/frameRate
    
    trialLog = pkl['items']['behavior']['trial_log']
    laserLog = pkl['items']['behavior']['layzer_trials']
    changeLog = pkl['items']['behavior']['stimuli']['images']['change_log']
                 
    trialStartTimes.append(np.full(len(trialLog),np.nan))
    trialStartFrames.append(np.full(len(trialLog),np.nan))
    trialEndTimes.append(np.full(len(trialLog),np.nan))
    abortedTrials.append(np.zeros(len(trialLog),dtype=bool))
    abortTimes.append(np.full(len(trialLog),np.nan))
    scheduledChangeTimes.append(np.full(len(trialLog),np.nan))
    changeTimes.append(np.full(len(trialLog),np.nan))
    changeFrames.append(np.full(len(trialLog),np.nan))
    changeTrials.append(np.zeros(len(trialLog),dtype=bool))
    catchTrials.append(np.zeros(len(trialLog),dtype=bool))
    preChangeImage.append(['' for _ in range(len(trialLog))])
    changeImage.append(['' for _ in range(len(trialLog))])
    rewardTimes.append(np.full(len(trialLog),np.nan))
    autoReward.append(np.zeros(len(trialLog),dtype=bool))
    hit.append(np.zeros(len(trialLog),dtype=bool))
    miss.append(np.zeros(len(trialLog),dtype=bool))
    falseAlarm.append(np.zeros(len(trialLog),dtype=bool))
    correctReject.append(np.zeros(len(trialLog),dtype=bool))
    laser.append(np.full(len(trialLog),np.nan))
    laserOffset.append(np.full(len(trialLog),np.nan))
    laserFrames.append(np.full(len(trialLog),np.nan))
    
    for i,trial in enumerate(trialLog):
        events = [event[0] for event in trial['events']]
        for event,epoch,t,frame in trial['events']:
            if event=='trial_start':
                trialStartTimes[-1][i] = t
                trialStartFrames[-1][i] = frame
            elif event=='trial_end':
                trialEndTimes[-1][i] = t
            elif event=='stimulus_window' and epoch=='enter':
                ct = trial['trial_params']['change_time']
                if params[-1]['periodic_flash'] is not None:
                    ct *= flashInterval
                    ct -= params[-1]['pre_change_time']-flashInterval
                scheduledChangeTimes[-1][i] = t + ct
            elif 'abort' in events:
                if event=='abort':
                    abortedTrials[-1][i] = True
                    abortTimes[-1][i] = t
            elif event in ('stimulus_changed','sham_change'):
                changeTimes[-1][i] = t
                changeFrames[-1][i] = frame
            elif event=='hit':
                hit[-1][i] = True
            elif event=='miss':
                miss[-1][i] = True 
            elif event=='false_alarm':
                falseAlarm[-1][i] = True
            elif event=='rejection':
                correctReject[-1][i] = True
        if not abortedTrials[-1][i]:
            if trial['trial_params']['catch']:
                catchTrials[-1][i] = True
            else:
                if len(trial['stimulus_changes'])>0:
                    changeTrials[-1][i] = True
                    preChangeImage[-1][i] = trial['stimulus_changes'][0][0][0]
                    changeImage[-1][i] = trial['stimulus_changes'][0][1][0]
            for laserTrial in laserLog:
                if 'actual_change_frame' in laserTrial and 'actual_layzer_frame' in laserTrial:
                    if laserTrial['actual_change_frame']==changeFrames[-1][i]:
                        laserOffset[-1][i] = laserTrial['actual_layzer_frame']-laserTrial['actual_change_frame']
                        laserFrames[-1][i] = laserTrial['actual_layzer_frame']
                        if 'laser' in laserTrial:
                            laser[-1][i] = laserTrial['laser']
                        break
        if len(trial['rewards']) > 0:
            rewardTimes[-1][i] = trial['rewards'][0][1]
            autoReward[-1][i] = trial['trial_params']['auto_reward']
            
    frameIntervals = pkl['items']['behavior']['intervalsms']/1000
    frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
    frameTimes += trialStartTimes[-1][0] - frameTimes[int(trialStartFrames[-1][0])]
    
    lickFrames = pkl['items']['behavior']['lick_sensors'][0]['lick_events']
    lickTimes.append(frameTimes[lickFrames])
        

def plotPerformance(params,laser,laserOffset,changeTrials,catchTrials,hit,falseAlarm,changeTimes,lickTimes,date=None):
    date = '' if date is None else date+' '
    if isinstance(params,list):
        respWin = params[0]['response_window']
        laser = np.concatenate(laser)
        laserOffset = np.concatenate(laserOffset)
        changeTrials = np.concatenate(changeTrials)
        catchTrials = np.concatenate(catchTrials)
        hit = np.concatenate(hit)
        falseAlarm = np.concatenate(falseAlarm)
        lickLatency = np.concatenate([getLickLatency(lt,ct,respWin[0]) for ct,lt in zip(changeTimes,lickTimes)])
    else:
        respWin = params['response_window']
        lickLatency = getLickLatency(lickTimes,changeTimes,respWin[0])
    lasers = [np.nan] if all(np.isnan(laser)) else np.unique(laser[~np.isnan(laser)])
    for las in lasers:
        fig = plt.figure()
        fig.text(0.5,0.99,date+'laser '+str(int(las)),va='top',ha='center',fontsize=10)
        ax = fig.add_subplot(1,1,1)
        offsets = np.concatenate((np.unique(laserOffset[~np.isnan(laserOffset)]),[np.nan]))
        if len(offsets)>1:
            xticks = (offsets-1)/frameRate*1000
            xticks[-1] = xticks[-2]*1.5
        else:
            xticks = [0]
        laserTrials = np.isnan(laser) | (laser==las)
        for trials,resp,clr,lbl,txty in zip((changeTrials,catchTrials),(hit,falseAlarm),'kr',('hit','false alarm'),(1.05,1.0)):
            r = []
            for offset,x in zip(offsets,xticks):
                offsetTrials = np.isnan(laserOffset) if np.isnan(offset) else laserOffset==offset
                i = trials & laserTrials & offsetTrials
                n = i.sum()
                r.append(resp[i].sum()/n)
                fig.text(x,txty,str(n),color=clr,transform=ax.transData,va='bottom',ha='center',fontsize=8)
            ax.plot(xticks,r,'o-',color=clr,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,1])
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i) for i in xticks[:-1]]+['no opto'])
        ax.set_xlabel('Laser onset relative to change (ms)')
        ax.set_ylabel('Response rate')
        ax.legend()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for w in respWin:
            ax.plot([0,xticks[-1]],[w*1000]*2,'--',color='0.75')
        for trials,resp,clr,lbl in zip((changeTrials,catchTrials),(hit,falseAlarm),'kr',('hit','false alarm')):
            r = []
            for offset,x in zip(offsets,xticks):
                offsetTrials = np.isnan(laserOffset) if np.isnan(offset) else laserOffset==offset
                i = trials & resp & laserTrials & offsetTrials
                r.append(1000*lickLatency[i])
                ax.plot(x+np.zeros(len(r[-1])),r[-1],'o',mec=clr,mfc='none')
            ax.plot(xticks,[np.nanmean(y) for y in r],'o',mec=clr,mfc=clr,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,900])
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i) for i in xticks[:-1]]+['no opto'])
        ax.set_xlabel('Laser onset relative to change (ms)')
        ax.set_ylabel('Reaction time (ms)')
        ax.set_title('laser '+str(int(las)))
        ax.legend()


for i,_ in enumerate(pklFiles):
    plotPerformance(params[i],laser[i],laserOffset[i],changeTrials[i],catchTrials[i],hit[i],falseAlarm[i],changeTimes[i],lickTimes[i],expDate[i])

plotPerformance(params,laser,laserOffset,changeTrials,catchTrials,hit,falseAlarm,changeTimes,lickTimes)




syncFile = fileIO.getFile('choose sync file',fileType='*.h5')
syncDataset = sync.Dataset(syncFile)

frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'vsync_stim')
vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
frameAppearTimes = vsyncTimes + monitorLag

binWidth = 0.001
for i,ch in enumerate((11,1)):
    laserRising,laserFalling = probeSync.get_sync_line_data(syncDataset,channel=ch)
    if any(laserRising):
        laserOnFromChange,laserOffFromChange = [t-vsyncTimes[changeFrames[laser==i].astype(int)] for t in (laserRising,laserFalling)]
        fig = plt.figure(figsize=(6,6))
        for j,(t,xlbl) in enumerate(zip((laserRising,laserFalling),('onset','offset'))):
            offset = t-vsyncTimes[changeFrames[laser==i].astype(int)]-monitorLag
            ax = fig.add_subplot(2,1,j+1)
            ax.hist(1000*offset,bins=1000*np.arange(round(min(offset),3)-0.001,round(max(offset),3)+0.001,binWidth),color='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlabel('Time from change on monitor to laser '+xlbl+' (s)')
            ax.set_ylabel('Count')
            if j==0:
                ax.set_title('laser '+str(i))
        plt.tight_layout()


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
        




