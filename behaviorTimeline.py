# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 14:24:20 2018

@author: svc_ccg
"""

from __future__ import division
import os
import glob
import datetime
import numpy as np
import pandas as pd
import scipy
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt


def calculateHitRate(hits,misses,adjusted=False):
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

def calculateDprime(hits,misses,falseAlarms,correctRejects):
    hitRate = calculateHitRate(hits,misses,adjusted=True)
    falseAlarmRate = calculateHitRate(falseAlarms,correctRejects,adjusted=True)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate,falseAlarmRate)]
    return z[0]-z[1]


pickleDir = r'\\EphysRoom342\Data\behavior pickle files'

mouseInfo = (('390339',('09192018','09202018','09212018')),
             ('394873',('10042018','10052018')),
             ('403472',('10312018','11012018')),
             ('403468',('11142018','11152018')),
             ('412624',('11292018','11302018')),
             ('416656',('03122019','03132019','03142019')),
             ('409096',('03212019',)),
             ('417882',('03262019','03272019')),
             ('408528',('04042019','04052019')),
             ('408527',('04102019','04112019')),
             ('421323',('04252019','04262019')),
             ('422856',('04302019','05012019')),
             ('423749',('05162019','05172019')),
             ('427937',('06062019','06072019')),
             ('423745',('06122019','06132019')),
             ('429084',('07112019','07122019')),
             ('423744',('08082019','08092019')),
             ('423750',('08132019','08142019')),
             ('459521',('09052019','09062019')),
             ('461027',('09122019','09132019')),
            )

mouseIDs = [mouse[0] for mouse in mouseInfo]
ephysDays = [[datetime.datetime.strptime(d,'%m%d%Y') for d in mouse[1]] for mouse in mouseInfo]

frameRate = 60.0

mouseWeights = pd.read_csv(os.path.join(os.path.dirname(pickleDir),'mouseWeights.csv'))
mouseWeights['date_time'] = pd.to_datetime(mouseWeights['date_time'])
mouseWeights['dayofweek'] = mouseWeights['date_time'].dt.dayofweek

params = ('trainingDate',
          'trainingStage',
          'rigID',
          'rewardsEarned',
          'dprimeOverall',
          'dprimeEngaged',
          'probEngaged')

data = {p:[[] for _ in range(len(mouseInfo))] for p in params}
data['ephys'] = []
data['weighDate'] = []
data['weight'] = []

unloadablePkl = []
for mouseInd,mouseID in enumerate(mouseIDs): 
    print('loading mouse '+mouseID)
    
    for pklFile in glob.glob(os.path.join(pickleDir,mouseID,'*.pkl')):
        try:
            core_data = data_to_change_detection_core(pd.read_pickle(pklFile))
            trials = create_extended_dataframe(
                trials=core_data['trials'],
                metadata=core_data['metadata'],
                licks=core_data['licks'],
                time=core_data['time'])
        except:
            unloadablePkl.append(pklFile)
            continue
        
        data['trainingDate'][mouseInd].append(datetime.datetime.strptime(os.path.basename(pklFile)[:6],'%y%m%d'))
        data['trainingStage'][mouseInd].append(core_data['metadata']['stage'])
        data['rigID'][mouseInd].append(core_data['metadata']['rig_id'])
        
        autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
        earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
        ignore = earlyResponse | autoRewarded
        miss = np.array(trials['response_type']=='MISS')
        hit = np.array(trials['response_type']=='HIT')
        falseAlarm = np.array(trials['response_type']=='FA')
        correctReject = np.array(trials['response_type']=='CR')
        
        data['rewardsEarned'][mouseInd].append(hit.sum())
        data['dprimeOverall'][mouseInd].append(calculateDprime(hit.sum(),miss.sum(),falseAlarm.sum(),correctReject.sum()))
        
        startFrame = int(trials['startframe'][0])
        endFrame = int(np.array(trials['endframe'])[-1])
        changeFrames = np.array(trials['change_frame'])
        hitFrames = np.zeros(endFrame,dtype=bool)
        hitFrames[changeFrames[hit].astype(int)] = True
        binSize = int(frameRate*60)
        halfBin = int(binSize/2)
        engagedThresh = 2
        rewardRate = np.zeros(hitFrames.size,dtype=int)
        rewardRate[halfBin:halfBin+hitFrames.size-binSize+1] = np.correlate(hitFrames,np.ones(binSize))
        rewardRate[:halfBin] = rewardRate[halfBin]
        rewardRate[-halfBin:] = rewardRate[-halfBin]
        data['probEngaged'][mouseInd].append(np.sum(rewardRate>engagedThresh)/rewardRate.size)
        engagedTrials = rewardRate[changeFrames[~ignore].astype(int)]>engagedThresh
        data['dprimeEngaged'][mouseInd].append(calculateDprime(*(r[~ignore][engagedTrials].sum() for r in (hit,miss,falseAlarm,correctReject))))
    
    df = mouseWeights[(mouseWeights['MID']==int(mouseID)) & (mouseWeights['dayofweek'].isin((5,6)))]
    data['weighDate'].append(df['date_time'].dt.to_pydatetime())
    data['weight'].append(np.array(df['wt_g']))
    
    
for mouseInd,(mouseID,ephysdates) in enumerate(zip(mouseIDs,ephysDays)):
    trainDate = np.array(data['trainingDate'][mouseInd])
    sortInd = np.argsort(trainDate)
    handoffReady = ['handoff_ready' in stage for stage in np.array(data['trainingStage'][mouseInd])[sortInd]].index(True)
    rig = [rigID=='NP3' for rigID in np.array(data['rigID'][mouseInd])[sortInd]].index(True)
    ephys = np.where(np.in1d(trainDate[sortInd],ephysdates))[0][0]
    weighDay = [d.days for d in data['weighDate'][mouseInd]-trainDate.min()]
    trainDay = [d.days for d in trainDate[sortInd]-trainDate.min()]
    
    fig = plt.figure(facecolor='w')
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(trainDay,np.array(data['rewardsEarned'][mouseInd])[sortInd],'ko')
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(weighDay,data['weight'][mouseInd],'ko')
    
    for i,(ax,ylbl) in enumerate(zip((ax1,ax2),('Rewards','Weight (g)'))):
        for j,lbl in zip((handoffReady,rig,ephys),('handoff\nready','on rig','ephys')):
            x = trainDay[j]-0.5
            ax.axvline(x,color='k')
            if i==0:
                ylim = plt.get(ax,'ylim')
                ax.text(x,ylim[0]+1.05*(ylim[1]-ylim[0]),lbl,horizontalalignment='center',verticalalignment='bottom')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-1,max(trainDay)+1])
        if i==1:
            ax.set_xlabel('Day')
        ax.set_ylabel(ylbl)
    


params = (rewardsEarned,dprimeEngaged,probEngaged)
labels = ('Rewards Earned','d prime','prob. engaged')
for ind,(mouseID,ephysDates) in enumerate(mouseInfo):     
    fig = plt.figure(facecolor='w')
    for i,(prm,lbl,ymax) in enumerate(zip(params,labels,(None,None,None))):
        ax = plt.subplot(len(params),1,i+1)
        for j,(d,p) in enumerate(zip(trainingDay[ind],prm[ind])):
            mec = 'r' if isEphys[ind][j] else 'k'
            mfc = mec if isRig[ind][j] else 'none'
            mrk = 'o' if isImages[ind][j] else 's'
            ax.plot(d,p,mrk,mec=mec,mfc=mfc,ms=8)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,max(trainingDay[ind])+1])
        ylimMax = np.nanmax(prm[ind]) if ymax is None else ymax
        ax.set_ylim([0,1.05*ylimMax])
        ax.set_ylabel(lbl,fontsize=14)
        if i==0:
            ax.set_title(mouseID,fontsize=14)
        if i==len(params)-1:
            ax.set_xlabel('Day',fontsize=14)
    plt.tight_layout()


numRewards = []
dpr = []
engaged = []
for day,rig,ephys,rewards,d,eng in zip(trainingDay,isRig,isEphys,rewardsEarned,dprimeEngaged,probEngaged):
    numRewards.append([])
    dpr.append([])
    engaged.append([])
    sortOrder = np.argsort(day)
    rig,ephys,rewards,d,eng = [np.array(a)[sortOrder] for a in (rig,ephys,rewards,d,eng)]
    if not all(rig):
        lastNSBDay = np.where(~rig)[0][-1]
        numRewards[-1].append(rewards[lastNSBDay])
        dpr[-1].append(d[lastNSBDay])
        engaged[-1].append(eng[lastNSBDay])
        firstRigDay = np.where(rig)[0][0]
        numRewards[-1].append(rewards[firstRigDay])
        if rewards[firstRigDay]<1:
            print(mouseInfo[len(numRewards)-1][0])
        dpr[-1].append(d[firstRigDay])
        engaged[-1].append(eng[firstRigDay])
    else:
        numRewards[-1].extend([np.nan]*2)
        dpr[-1].extend(([np.nan]*2))
        engaged[-1].extend(([np.nan]*2))
    ephysInd = np.where(ephys)[0]
    lastNonEphysDays = [ephysInd[0]-2,ephysInd[0]-1]
    numRewards[-1].extend(rewards[lastNonEphysDays])
    dpr[-1].extend(d[lastNonEphysDays])
    engaged[-1].extend(eng[lastNonEphysDays])    
    ephysDays = ephysInd[:2]
    numRewards[-1].extend(rewards[ephysDays])
    dpr[-1].extend(d[ephysDays])
    engaged[-1].extend(eng[ephysDays])
    if len(ephysDays)<2:
        numRewards[-1].append(np.nan)
        dpr[-1].append(np.nan)
        engaged[-1].append(np.nan)

params = (numRewards,engaged,dpr)
paramNames = ('Rewards Earned','Prob. Engaged','d\' (engaged)')

show = slice(0,6)
fig = plt.figure(facecolor='w',figsize=(10,10))
for i,(prm,ylab,ylim) in enumerate(zip(params,paramNames,([0,300],[0,1],[0,4]))):
    ax = plt.subplot(len(params),1,i+1)
    ymax = 0
#    for p,rig in zip(prm,isRig):
#        if not all(rig):
#            ax.plot(p[show],'o-',color='0.8',mec='0.8',ms=2)
#            ymax = max(ymax,max(p[show]))
    prm = np.array([p for p,rig in zip(prm,isRig) if not all(rig)])
    meanPrm = np.nanmean(prm,axis=0)
    n = np.sum(~np.isnan(prm),axis=0)
    print(n)
    stdPrm = np.nanstd(prm,axis=0)
    semPrm = stdPrm/n**0.5
    ax.plot(meanPrm[show],'o',mfc='k',mec='k',ms=10)
    for x,(m,s) in enumerate(zip(meanPrm[show],semPrm[show])):
        ax.plot([x]*2,m+np.array([-s,s]),'k',linewidth=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([-0.25,show.stop-show.start-0.75])
    ymax = 1.05*max(ymax,np.nanmax((meanPrm+stdPrm)[show])) #if ylim is None else ylim[1]
    ax.plot([(show.stop-show.start)-2.5]*2,[0,ymax],'k--')
    ax.set_ylim([0,ymax])
    ax.set_xticks(np.arange(show.start,show.stop))
    if i==len(params)-1:
        ax.set_xticklabels(['Last NSB','First NP3',-2,-1,1,2])
        ax.set_xlabel('Day',fontsize=16)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylab,fontsize=16)
    ax.yaxis.set_label_coords(-0.075,0.5)
    ax.locator_params(axis='y',nbins=3)
fig.text(0.38,0.95,'Training',fontsize=18,horizontalalignment='center')
fig.text(0.85,0.95,'Ephys',fontsize=18,horizontalalignment='center')
plt.tight_layout()


meanImageHitRate = []
for m,day in zip(imageHitRateEngaged,trainingDay):
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    sortOrder = np.argsort(day)
    h = np.array([m[i] for i in sortOrder if len(m[i])>0])
    meanImageHitRate.append(h[-4:].mean(axis=0))
    ax.imshow(h,clim=(0,1),cmap='gray',interpolation='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlabel('image')
    ax.set_ylabel('day')
    
    
    
plt.imshow(np.array(meanImageHitRate)[3:],clim=(0,1),cmap='gray',interpolation='none')

