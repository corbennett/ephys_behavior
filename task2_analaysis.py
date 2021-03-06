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
import visual_behavior.analyze



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


def getLickLatency(lickTimes,eventTimes,offset=0):
    firstLickInd = np.searchsorted(lickTimes,eventTimes+offset)
    noLicksAfter = firstLickInd==lickTimes.size
    firstLickInd[noLicksAfter] = lickTimes.size-1
    lickLat = lickTimes[firstLickInd]-eventTimes
    lickLat[noLicksAfter] = np.nan
    return lickLat



# read pkl file
f = fileIO.getFile()
pkl = pd.read_pickle(f)

params = pkl['items']['behavior']['params']
if params['periodic_flash'] is not None:
    flashDur = params['periodic_flash'][0]
    
trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])
changeLog = pkl['items']['behavior']['stimuli']['grating']['change_log']

grayDur = np.full(len(trialLog),np.nan)
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
    if params['periodic_flash'] is not None:
        grayDur[i] = trial['flash_gray_dur'] if 'flash_gray_dur' in trial else params['periodic_flash'][1]
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
                ct *= flashDur+grayDur[i]
                ct -= params['pre_change_time']-(flashDur+grayDur[i])
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
                for entry in trial['stimulus_changes'][0]:
                    if isinstance(entry,tuple):
                        if entry[0]=='home':
                            homeOri[i] = entry[1]
                        elif entry[0]=='change_ori':
                            changeOri[i] = entry[1]
    if 'home' in trial:
        homeOri[i] = trial['home']['Ori']
    if len(trial['rewards']) > 0:
        rewardTimes[i] = trial['rewards'][0][1]
        autoReward[i] = trial['trial_params']['auto_reward']
        
frameIntervals = pkl['items']['behavior']['intervalsms']/1000
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
frameTimes += trialStartTimes[0] - frameTimes[int(trialStartFrames[0])]

lickFrames = pkl['items']['behavior']['lick_sensors'][0]['lick_events']
lickTimes = frameTimes[lickFrames]

dx,vsig,vin = [pkl['items']['behavior']['encoders'][0][key] for key in ('dx','vsig','vin')]
runSpeed = visual_behavior.analyze.compute_running_speed(dx[:frameTimes.size],frameTimes,vsig,vin)



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

grayDurs = np.array([np.nan]) if all(np.isnan(grayDur)) else np.unique(grayDur[~np.isnan(grayDur)])


# task parameters
def printParam(ax,x,y,key,val):
    if isinstance(val,dict):
        ax.text(x,y,key+':',fontsize='small')
        x += 0.05
        y += 1.5
        for k in sorted(val.keys()):
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
for i,gray in enumerate(grayDurs):
    ax = fig.add_subplot(grayDurs.size+3,1,i+1)
    trials = np.ones(grayDur.size,dtype=bool) if np.isnan(gray) else grayDur==gray
    xlim = [0,round(np.nanmax(timeToChange[trials & (~abortedTrials)]))+1]
    ax.hist(timeToChange[trials & changeTrials],bins=np.arange(xlim[0],xlim[1],0.17),color='g',label='change (n='+str(changeTrials[trials].sum())+')')
    ax.hist(timeToChange[trials & catchTrials],bins=np.arange(xlim[0],xlim[1],0.17),color='r',label='catch (n='+str(catchTrials[trials].sum())+')')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_xlabel('Time to change/catch from trial start (s)')
    ax.set_ylabel('Trials')
    if not np.isnan(gray):
        ax.set_title('inter-flash gray = '+str(gray)+' s')
    ax.legend()

ax = fig.add_subplot(grayDurs.size+3,1,grayDurs.size+1)
xlim = [round(interTrialInterval.min())-1,round(interTrialInterval.max())+1]
ax.hist(interTrialInterval,bins=np.arange(xlim[0],xlim[1],0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xlabel('Inter-trial interval (start to start) (s)')
ax.set_ylabel('Trials')

ax = fig.add_subplot(grayDurs.size+3,1,grayDurs.size+2)
changeToEnd = timeFromChangeToTrialEnd[~abortedTrials]
abortToEnd = timeFromAbortToTrialEnd[abortedTrials] if abortedTrials.sum()>0 else np.full(len(trialLog),np.nan)
xlim = [round(min(np.nanmin(changeToEnd),np.nanmin(abortToEnd)))-1,round(max(np.nanmax(changeToEnd),np.nanmax(abortToEnd)))+1]
ax.hist(changeToEnd,bins=np.arange(xlim[0],xlim[1],0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xlabel('Time from change/catch to trial end (includes response window + random gray) (s)')
ax.set_ylabel('Trials')

ax = fig.add_subplot(grayDurs.size+3,1,grayDurs.size+3)
if not np.all(np.isnan(abortToEnd)):
    ax.hist(abortToEnd,bins=np.arange(xlim[0],xlim[1],0.17),color='k')
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
ax.plot([0,0],[-1,len(trialLog)],'-',color='0.5')
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
    else:
        clr = '0.5'
        lbl = 'unknown'
    ct = changeTimes[i] if not np.isnan(changeTimes[i]) else scheduledChangeTimes[i]
    ax.add_patch(matplotlib.patches.Rectangle([trialStartTimes[i]-ct,i-0.5],width=trialEndTimes[i]-trialStartTimes[i],height=1,facecolor=clr,edgecolor=None,alpha=0.5,zorder=0,label=lbl))
    trialLickTimes = np.array([lick[0] for lick in trial['licks']])
    trialLickTimes -= ct
    ax.vlines(trialLickTimes,i-0.5,i+0.5,colors='k')
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
    
    
# reaction time
if params['periodic_flash'] is not None:
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(2,1,1)
else:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
ylim = [0,params['response_window'][1]*2]
ax.plot(changeTimes[hit]/60,rewardTimes[hit]-changeTimes[hit],'o',color=trialColors['hit'],label='hit')
for resp,lbl in zip((miss,falseAlarm,correctReject),('miss','false alarm','correct reject')):
    lickLat = getLickLatency(lickTimes,changeTimes[resp],params['response_window'][0])
    i = lickLat<=ylim[1]
    ax.plot(changeTimes[resp][i]/60,lickLat[i],'o',color=trialColors[lbl],label=lbl)
    i = (lickLat>ylim[1]) | (np.isnan(lickLat))
    ax.plot(changeTimes[resp][i]/60,np.zeros(i.sum())+0.99*ylim[1],'^',color=trialColors[lbl])
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
xlim = plt.get(ax,'xlim')
for i in (0,1):
    ax.plot(xlim,[params['response_window'][i]]*2,'--',color='0.5',zorder=0)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Time in session (min)')
ax.set_ylabel('Reaction time (s)')
ax.legend()

if params['periodic_flash'] is not None:
    ax = fig.add_subplot(2,1,2)
    bins = np.arange(0.125,np.nanmax(timeToChange)+0.125,0.25)
    ctBinInd = np.digitize(timeToChange,bins)
    for i in np.unique(ctBinInd[hit]):
        ind = hit & (ctBinInd==i)
        t = timeToChange[ind].mean()
        rt = rewardTimes[ind]-changeTimes[ind]
        ax.plot([t]*rt.size,rt,'o',mec='0.5',mfc='none',alpha=0.5)
        m = rt.mean()
        s = rt.std()/(rt.size**0.5)
        ax.plot(t,m,'ko')
        ax.plot([t,t],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,plt.get(ax,'ylim')[1]])
    ax.set_xlabel('Time to change (s)')
    ax.set_ylabel('Reaction time (hits) (s)')

plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')


# running
fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(grayDurs.size+1,1,1)
ax.plot(frameTimes/60,runSpeed,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time in session (min)')
ax.set_ylabel('Running speed (cm/s)')

for ind,gray in enumerate(grayDurs):
    ax = fig.add_subplot(grayDurs.size+1,1,ind+2)
    if params['periodic_flash'] is not None:
        preTime = 2*(flashDur+gray) + gray
    else:
        preTime = 2
    postTime = params['response_window'][1] + params['no_stim_no_lick_randrange'][0]
    runPlotTime = np.arange(-preTime,postTime+0.01,0.01)
    trialSpeed = np.full((len(trialLog),len(runPlotTime)),np.nan)
    for i,ct in enumerate(changeTimes):
        if not np.isnan(ct):
            ind = (frameTimes>=ct-preTime) & (frameTimes<=ct+postTime)
            trialSpeed[i] = np.interp(runPlotTime,frameTimes[ind]-ct,runSpeed[ind])
    for trials,lbl in zip((hit,miss,falseAlarm,correctReject),('hit','miss','false alarm','correct reject')):
        if grayDurs.size>1:
            trials = trials & (grayDur==gray)
        m = trialSpeed[trials].mean(axis=0)
        n = trials.sum()
        s = trialSpeed[trials].std(axis=0)/(n**0.5)
        ax.plot(runPlotTime,m,color=trialColors[lbl],label=lbl+' (n='+str(n)+')')
        ax.fill_between(runPlotTime,m+s,m-s,color=trialColors[lbl],alpha=0.25)
    if params['periodic_flash'] is not None:
        ylim = plt.get(ax,'ylim')
        for t in np.arange(-3*(flashDur+gray),flashDur,flashDur+gray):
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
    ax.set_xlim([0,(bins[-1]+binDuration)/60])
    ylim = plt.get(ax,'ylim')
    ax.set_ylim([0,ylim[1]])
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
ymax = 1 if np.all(np.isnan(dprime)) else np.nanmax(dprime)
ax.set_ylim([0,ymax*1.05])
ax.set_ylabel('d prime')
ax.set_xlabel('Time (min)')
plt.tight_layout()
if makeSummaryPDF:
    fig.savefig(pdf,format='pdf')
    
    
# orientation
home = np.unique(homeOri[changeTrials])
oris = np.unique(changeOri[changeTrials])
if len(home)>1 or len(oris)>1:
    for gray in grayDurs:
        grayTrials = np.ones(grayDur.size,dtype=bool) if np.isnan(gray) else grayDur==gray
        for ho in home:
            trials = grayTrials & (homeOri==ho)
            oriNtrials = []
            oriDelta = []
            oriFracCorr = []
            oriReactionTime = []
            for ori in oris:
                oriTrials = trials & (changeOri==ori)
                if any(trials):
                    oriNtrials.append(oriTrials.sum())
                    oriDelta.append(ori-ho)
                    oriFracCorr.append(np.sum(oriTrials & hit)/oriNtrials[-1])
                    oriReactionTime.append((rewardTimes-changeTimes)[oriTrials & hit])
            
            falseAlarmRate = falseAlarm[trials].sum()/catchTrials[trials].sum()
            falseAlarmReactionTime = getLickLatency(lickTimes,changeTimes[trials & falseAlarm],params['response_window'][0])
                
            fig = plt.figure(figsize=(6,8))
            ax = fig.add_subplot(2,1,1)
            ax.plot([0,0],[0,1.05],'--',color='0.5',alpha=0.5)
            for ori,fracCorr,n in zip(oriDelta,oriFracCorr,oriNtrials):
                ax.plot(ori,fracCorr,'ko')
                ax.text(ori,fracCorr+0.05,str(n),ha='center')
            ax.plot(0,falseAlarmRate,'ko')
            ax.text(0,falseAlarmRate+0.05,str(catchTrials[trials].sum()),ha='center')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            xmax = 1.05*max(np.abs(oriDelta))
            ax.set_xlim(-xmax,xmax)
            ax.set_ylim([0,1.05])
            ax.set_ylabel('Response probability',fontsize=12)
            ax.set_title('inter-flash gray duration = '+str(gray)+' s'+'\n'+'home ori: '+str(ho)+' (n='+str(np.sum(trials & (changeTrials | catchTrials)))+' + '+str(np.sum(trials & abortedTrials))+' aborted trials)',fontsize=10)
                
            ax = fig.add_subplot(2,1,2)
            for ori,rt in zip(np.concatenate((oriDelta,[0])),oriReactionTime+[falseAlarmReactionTime]):
                ax.plot([ori]*rt.size,rt,'o',mec='0.5',mfc='none',alpha=0.5)
                m = rt.mean()
                s = rt.std()/(rt.size**0.5)
                ax.plot(ori,m,'ko')
                ax.plot([ori]*2,[m-s,m+s],'k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim(-xmax,xmax)
            ylim = [0,params['response_window'][1]]
            ax.plot([0,0],ylim,'--',color='0.5',alpha=0.5,zorder=0)
            ax.set_ylim(ylim)
            ax.set_ylabel('Reaction time (s)',fontsize=12)
            ax.set_xlabel('$\Delta$ Orientation (degrees)',fontsize=12)
            plt.tight_layout()
            if makeSummaryPDF:
                fig.savefig(pdf,format='pdf')


# finalize pdf
if makeSummaryPDF:
    pdf.close()
    plt.close('all')




