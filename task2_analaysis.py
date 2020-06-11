# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:47:06 2020

@author: svc_ccg
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import fileIO

matplotlib.rcParams['pdf.fonttype'] = 42


f = fileIO.getFile()
pkl = pd.read_pickle(f)


params = pkl['items']['behavior']['params']
frameIntervals = pkl['items']['behavior']['intervalsms']
trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])


fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(1,1,1)

trialStartTimes = np.full(len(trialLog),np.nan)
trialEndTimes = np.full(len(trialLog),np.nan)
scheduledChangeTimes = np.full(len(trialLog),np.nan)
changeTimes = np.full(len(trialLog),np.nan)
abortedTrials = np.zeros(len(trialLog),dtype=bool)
abortTimes = np.full(len(trialLog),np.nan)
catchTrials = np.zeros(len(trialLog),dtype=bool)
rewardTimes = np.full(len(trialLog),np.nan)
autoReward = np.zeros(len(trialLog),dtype=bool)
hit = np.zeros(len(trialLog),dtype=bool)
miss = np.zeros(len(trialLog),dtype=bool)
falseAlarm = np.zeros(len(trialLog),dtype=bool)
correctReject = np.zeros(len(trialLog),dtype=bool)
for i,trial in enumerate(trialLog):
    for event,epoch,t,frame in trial['events']:
        if event=='trial_start':
            trialStartTimes[i] = t
        elif event=='trial_end':
            trialEndTimes[i] = t
        elif event=='stimulus_window' and epoch=='enter':
            scheduledChangeTimes[i] = t + trial['trial_params']['change_time']
        elif event in ('stimulus_changed','sham_change'):
            changeTimes[i] = t
        elif event=='abort':
            abortedTrials[i] = True
            abortTimes[i] = t
        elif event=='hit':
            hit[i] = True
        elif event=='miss':
            miss[i] = True 
        elif event=='false_alarm':
            falseAlarm[i] = True
        elif event=='rejection':
            correctReject[i] = True
    catchTrials[i] = trial['trial_params']['catch']
    if len(trial['rewards']) > 0:
        rewardTimes[i] = trial['rewards'][0][1]
        autoReward[i] = trial['trial_params']['auto_reward']
        
    if abortedTrials[i]:
        clr = (1,0,0)
    elif autoReward[i]:
        clr = (0,0,1)
    elif hit[i]:
        clr = (0,0.5,0)
    elif miss[i]:
        clr = np.array((144,238,144))/255
    elif falseAlarm[i]:
        clr = np.array((255,140,0))/255
    elif correctReject[i]:
        clr = (1,1,0)
    
    ct = changeTimes[i] if not np.isnan(changeTimes[i]) else scheduledChangeTimes[i]
    ax.add_patch(matplotlib.patches.Rectangle([trialStartTimes[i]-ct,i-0.5],width=trialEndTimes[i]-trialStartTimes[i],height=1,facecolor=clr,edgecolor=None,alpha=0.5,zorder=0))
    lickTimes = np.array([lick[0] for lick in trial['licks']])
    lickTimes -= ct
    ax.vlines(lickTimes,i-0.5,i+0.5,colors='k')
    
plt.tight_layout()


# task timing
timeToChange = changeTimes - trialStartTimes
interTrialInterval = trialStartTimes[1:] - trialStartTimes[:-1]
timeFromChangeToTrialEnd = trialEndTimes - changeTimes
timeFromAbortToTrialEnd = trialEndTimes - abortTimes

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(4,1,1)
ax.hist(timeToChange[~catchTrials & ~abortedTrials],bins=np.arange(0,10,0.17),color='g',label='change (n='+str(np.sum(~catchTrials & ~abortedTrials))+')')
ax.hist(timeToChange[catchTrials & ~abortedTrials],bins=np.arange(0,10,0.17),color='r',label='catch (n='+str(np.sum(catchTrials & ~abortedTrials))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,round(timeToChange[~abortedTrials].max())+1])
ax.set_xlabel('Time to change/catch from trial start (s)')
ax.set_ylabel('Trials')
ax.set_title(params['stage'])
ax.legend()

ax = fig.add_subplot(4,1,2)
ax.hist(interTrialInterval,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([round(interTrialInterval.min())-1,round(interTrialInterval.max())+1])
ax.set_xlabel('Inter-trial interval (start to start) (s)')
ax.set_ylabel('Trials')

ax = fig.add_subplot(4,1,3)
changeToEnd = timeFromChangeToTrialEnd[~abortedTrials]
abortToEnd = timeFromAbortToTrialEnd[abortedTrials]
xlim = [round(min(changeToEnd.min(),abortToEnd.min()))-1,round(max(changeToEnd.max(),abortToEnd.max()))+1]
ax.hist(changeToEnd,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xlabel('Time from change/catch to trial end (includes response window + random gray) (s)')
ax.set_ylabel('Trials')

ax = fig.add_subplot(4,1,4)
ax.hist(abortToEnd,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xlabel('Time from abort to trial end (includes timeout + random gray) (s)')
ax.set_ylabel('Trials')
plt.tight_layout()




