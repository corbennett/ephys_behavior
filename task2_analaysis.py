# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:47:06 2020

@author: svc_ccg
"""

import os
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO
import visual_behavior



def calcDprime(hits,misses,falseAlarms,correctRejects):
    hitRate = calcHitRate(hits,misses,adjusted=True)
    falseAlarmRate = calcHitRate(falseAlarms,correctRejects,adjusted=True)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate,falseAlarmRate)]
    return z[0]-z[1]


def calcHitRate(hits,misses,adjusted=False):
    n = hits+misses
    if n==0:
        return np.nan
    hitRate = hits/n
    if adjusted:
        if hitRate==0:
            hitRate = 0.5/n
        elif hitRate==1:
            hitRate = 1-0.5/n
    return hitRate



# read pkl file
f = fileIO.getFile()
pkl = pd.read_pickle(f)

params = pkl['items']['behavior']['params']
trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])

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
homeOri = np.full(len(trialLog),np.nan)
changeOri = np.full(len(trialLog),np.nan)
rewardTimes = np.full(len(trialLog),np.nan)
autoReward = np.zeros(len(trialLog),dtype=bool)
hit = np.zeros(len(trialLog),dtype=bool)
miss = np.zeros(len(trialLog),dtype=bool)
falseAlarm = np.zeros(len(trialLog),dtype=bool)
correctReject = np.zeros(len(trialLog),dtype=bool)
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
                ct *= sum(params['periodic_flash'])
            scheduledChangeTimes[i] = t + trial['trial_params']['change_time']
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
            changeTrials[i] = True
            for entry in trial['stimulus_changes'][0]:
                if isinstance(entry,tuple):
                    if entry[0]=='home':
                        homeOri[i] = entry[1]
                    elif entry[0]=='change_ori':
                        changeOri[i] = entry[1]
    if len(trial['rewards']) > 0:
        rewardTimes[i] = trial['rewards'][0][1]
        autoReward[i] = trial['trial_params']['auto_reward']
        
frameIntervals = pkl['items']['behavior']['intervalsms']/1000
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
frameTimes += trialStartTimes[0] - frameTimes[int(trialStartFrames[0])]

dx,vsig,vin = [pkl['items']['behavior']['encoders'][0][key] for key in ('dx','vsig','vin')]
runSpeed = visual_behavior.analyze.compute_running_speed(dx,frameTimes,vsig,vin)


# make figures
makeSummaryPDF = True
if makeSummaryPDF:
    pdf = PdfPages(os.path.join(os.path.dirname(f),os.path.splitext(os.path.basename(f))[0]+'_summary.pdf'))
    
trialColors = {
               'abort': (1,0,0),
               'hit': (0,0.5,0),
               'miss': np.array((144,238,144))/255,
               'false alarm': np.array((255,140,0))/255,
               'correct reject': (1,1,0)
              }


# task parameters
def printParam(ax,x,y,key,val):
    if isinstance(val,dict):
        ax.text(x,y,key+':',fontsize='small')
        x += 0.05
        y += 1.5
        for k in val:
            y = printParam(ax,x,y,k,val[k])
        return y
    else:
        ax.text(x,y,key+': '+str(val),fontsize='small')
        return y+1.5

fig = plt.figure(figsize=(6,10))
ax = fig.add_subplot(1,1,1)
x = y = 0
for key in sorted(params.keys()):
    y = printParam(ax,x,y,key,params[key])
ax.set_ylim([y+1,-1])
for side in ('left','right','top','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# task timing
timeToChange = changeTimes - trialStartTimes
interTrialInterval = trialStartTimes[1:] - trialStartTimes[:-1]
timeFromChangeToTrialEnd = trialEndTimes - changeTimes
timeFromAbortToTrialEnd = trialEndTimes - abortTimes

fig = plt.figure(figsize=(7,8))
ax = fig.add_subplot(4,1,1)
ax.hist(timeToChange[changeTrials],bins=np.arange(0,10,0.17),color='g',label='change (n='+str(changeTrials.sum())+')')
ax.hist(timeToChange[catchTrials],bins=np.arange(0,10,0.17),color='r',label='catch (n='+str(catchTrials.sum())+')')
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
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# lick raster
fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(1,1,1)
for i,trial in enumerate(trialLog):       
    if abortedTrials[i]:
        clr = trialColors['abort']
        lbl = 'abort' if abortedTrials[:i].sum()==0 else None
    elif hit[i]:
        clr = trialColors['hit']
        lbl = 'hit' if hit[:i].sum()==0 else None
    elif miss[i]:
        clr = trialColors['miss']
        lbl = 'miss' if miss[:i].sum()==0 else None
    elif falseAlarm[i]:
        clr = trialColors['false alarm']
        lbl = 'false alarm' if falseAlarm[:i].sum()==0 else None
    elif correctReject[i]:
        clr = trialColors['correct reject']
        lbl = 'correct reject' if correctReject[:i].sum()==0 else None
    ct = changeTimes[i] if not np.isnan(changeTimes[i]) else scheduledChangeTimes[i]
    ax.add_patch(matplotlib.patches.Rectangle([trialStartTimes[i]-ct,i-0.5],width=trialEndTimes[i]-trialStartTimes[i],height=1,facecolor=clr,edgecolor=None,alpha=0.5,zorder=0,label=lbl))
    lickTimes = np.array([lick[0] for lick in trial['licks']])
    lickTimes -= ct
    ax.vlines(lickTimes,i-0.5,i+0.5,colors='k')
    clr = 'b' if autoReward[i] else clr
    ax.vlines(rewardTimes[i]-ct,i-0.5,i+0.5,colors=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([len(trialLog),-1])
ax.set_xlabel('Time relative to actual or schuduled change/catch (s)')   
ax.set_ylabel('Trial')
ax.set_title('Lick raster')
ax.legend()
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# running
fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(2,1,1)
ax.plot(frameTimes/60,runSpeed,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Running speed (cm/s)')

ax = fig.add_subplot(2,1,2)
preTime = 2.75
postTime = params['response_window'][1] + params['no_stim_no_lick_randrange'][0]
runPlotTime = np.arange(-preTime,postTime+0.01,0.01)
trialSpeed = np.full((len(trialLog),len(runPlotTime)),np.nan)
for i,ct in enumerate(changeTimes):
    if not np.isnan(ct):
        ind = (frameTimes>=ct-preTime) & (frameTimes<=ct+postTime)
        trialSpeed[i] = np.interp(runPlotTime,frameTimes[ind]-ct,runSpeed[ind])
for trials,lbl in zip((hit,miss,falseAlarm,correctReject),('hit','miss','false alarm','correct reject')):
    m = trialSpeed[trials].mean(axis=0)
    n = trials.sum()
    s = trialSpeed[trials].std(axis=0)/(n**0.5)
    ax.plot(runPlotTime,m,color=trialColors[lbl],label=lbl+' (n='+str(n)+')')
    ax.fill_between(runPlotTime,m+s,m-s,color=trialColors[lbl],alpha=0.25)
if params['periodic_flash'] is not None:
    flashDur,grayDur = params['periodic_flash']
    flashInterval = flashDur + grayDur
    ylim = plt.get(ax,'ylim')
    for t in np.arange(-3*flashInterval,flashDur,flashInterval):
        ax.add_patch(matplotlib.patches.Rectangle([t,ylim[0]],width=flashDur,height=ylim[1]-ylim[0],color='0.9',alpha=0.5,zorder=0))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-preTime,postTime])
ax.set_xlabel('Time from change/catch (s)')
ax.set_ylabel('Running speed (cm/s)')
ax.legend()

plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# performance
binInterval = 60
binDuration = 600
bins = np.arange(0,60*params['max_task_duration_min']-binDuration+binInterval,binInterval).astype(int)
binCenters = bins+binDuration/2
abort = []
h = []
m = []
fa = []
cr = []
dprime = []
for binStart in bins:
    i = trialStartTimes>=binStart
    if binStart<bins.max():
        i = i & (trialStartTimes<binStart+binDuration)
    abort.append(abortedTrials[i].sum())
    h.append(hit[i].sum())
    m.append(miss[i].sum())
    fa.append(falseAlarm[i].sum())
    cr.append(correctReject[i].sum())
    dprime.append(calcDprime(h[-1],m[-1],fa[-1],cr[-1]))

  
fig = plt.figure(figsize=(7,7))
for i,d in enumerate((
                      ((abort,),('abort',)),
                      ((h,m),('hit','miss')),
                      ((fa,cr),('false alarm','correct reject')),
                     )
                    ):
    ax = fig.add_subplot(3,1,i+1)
    for y,lbl in zip(*d):
        ax.plot(binCenters/60,np.array(y)/binDuration*60,color=trialColors.get(lbl,'k'),label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    if i==0:
        ax.set_ylabel('Events per min\n(rolling 10 min bins)')
    if i==2:
        ax.set_xlabel('Time (min)')
    ax.legend()
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


fig = plt.figure(figsize=(7,5))
hitRate,falseAlarmRate = [[calcHitRate(a,b,adjusted=False) for a,b in zip(*d)] for d in ((h,m),(fa,cr))]
ax = fig.add_subplot(2,1,1)
ax.plot(binCenters/60,hitRate,color=trialColors['hit'],label='change')
ax.plot(binCenters/60,falseAlarmRate,color=trialColors['false alarm'],label='catch')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.05])
ax.set_ylabel('Response probability')
ax.legend()

ax = fig.add_subplot(2,1,2)
ax.plot(binCenters/60,dprime,color='k',label='d prime')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,np.nanmax(dprime)*1.05])
ax.set_ylabel('d prime')
ax.set_xlabel('Time (min)')
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')
    
    
# orientation  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
oris = np.unique(changeOri[changeTrials])
for ori in oris:
    oriInd = changeOri==ori
    n = oriInd.sum()
    fracCorr = np.sum(hit & oriInd)/n
    ax.plot(ori,fracCorr,'ko')
    ax.text(ori,fracCorr+0.05,str(n),ha='center')
falseAlarmRate = falseAlarm.sum()/catchTrials.sum()
ax.plot(0,falseAlarmRate,'ko')
ax.text(0,falseAlarmRate+0.05,str(catchTrials.sum()),ha='center')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
xmax = 1.05*max(abs(oris))
ax.set_xlim(-xmax,xmax)
ax.set_ylim([0,1.05])
ax.set_ylabel('Response probability',fontsize=12)
ax.set_xlabel('$\Delta$ Orientation (degrees)',fontsize=12)
ax.set_title('number of trials above each point',fontsize=10)
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# finalize pdf
if makeSummaryPDF:
    pdf.close()
    plt.close('all')




