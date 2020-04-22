# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import math
import os
import pickle
import time
import warnings
from collections import OrderedDict
import h5py
import numpy as np
import scipy
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import fileIO
import getData
import probeSync
import analysis_utils
import summaryPlots



def makeSummaryPlots(miceToAnalyze='all'):
    for mouseID,ephysDates,probeIDs,imageSet,passiveSession in mouseInfo:
        if miceToAnalyze!='all' and mouseID not in miceToAnalyze:
            continue
        for date,probes in zip(ephysDates,probeIDs):
            expName = date+'_'+mouseID
            print(expName)
            dataDir = os.path.join(baseDir,expName)
            obj = getData.behaviorEphys(dataDir,probes)
            obj.loadFromRawData()
            summaryPlots.all_unit_summary(obj,probes)


def getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze='all',makeSDFs=True,sdfParams={}):
    if popDataToHDF5:
        popHDF5Path = os.path.join(localDir,'popData.hdf5')
    for mouseID,ephysDates,probeIDs,imageSet,passiveSession in mouseInfo:
        if miceToAnalyze!='all' and mouseID not in miceToAnalyze:
            continue
        for date,probes in zip(ephysDates,probeIDs):
            expName = date+'_'+mouseID
            print(expName)
            dataDir = os.path.join(baseDir,expName)
            obj = getData.behaviorEphys(dataDir,probes)
            hdf5Path = os.path.join(localDir,expName+'.hdf5')
            
            if objToHDF5:
                obj.loadFromRawData()
                obj.saveHDF5(hdf5Path)
            else:
                obj.loadFromHDF5(hdf5Path)
            
            if popDataToHDF5:
                trials = ~(obj.earlyResponse | obj.autoRewarded)
                resp = np.array([None for _ in trials],dtype='object')
                resp[obj.hit] = 'hit'
                resp[obj.miss] = 'miss'
                resp[obj.falseAlarm] = 'falseAlarm'
                resp[obj.correctReject] = 'correctReject'
                
                data = {expName:{}}
                data[expName]['units'] = {}
                data[expName]['ccfRegion'] = {}
                data[expName]['inCortex'] = {}
                data[expName]['spikeTimes'] = {}
                for probe in probes:
                    units = probeSync.getOrderedUnits(obj.units[probe])
                    data[expName]['units'][probe] = units
                    data[expName]['ccfRegion'][probe] = [obj.units[probe][u]['ccfRegion'] for u in units]
                    data[expName]['inCortex'][probe] = [obj.units[probe][u]['inCortex'] for u in units]
                    data[expName]['spikeTimes'][probe] = OrderedDict()
                    for u in units:
                        data[expName]['spikeTimes'][probe][str(u)] = obj.units[probe][u]['times']
                data[expName]['isiRegion'] = {probe: obj.probeCCF[probe]['ISIRegion'] for probe in probes}
                data[expName]['flashImage'] = obj.flashImage
                data[expName]['omitFlashImage'] = obj.omittedFlashImage
                data[expName]['initialImage'] = obj.initialImage[trials]
                data[expName]['changeImage'] = obj.changeImage[trials]
                data[expName]['response'] = resp[trials]
                data[expName]['behaviorFlashTimes'] = obj.frameAppearTimes[obj.flashFrames]
                data[expName]['behaviorOmitFlashTimes'] = obj.frameAppearTimes[obj.omittedFlashFrames]
                data[expName]['behaviorChangeTimes'] = obj.frameAppearTimes[obj.changeFrames[trials]]
                data[expName]['behaviorRunTime'] = obj.behaviorRunTime
                data[expName]['behaviorRunSpeed'] = obj.behaviorRunSpeed
                data[expName]['lickTimes'] = obj.lickTimes
                data[expName]['rewardTimes'] = obj.rewardTimes[trials]
                if obj.passive_pickle_file is not None:
                    data[expName]['passiveFlashTimes'] = obj.passiveFrameAppearTimes[obj.flashFrames]
                    data[expName]['passiveOmitFlashTimes'] = obj.passiveFrameAppearTimes[obj.omittedFlashFrames]
                    data[expName]['passiveChangeTimes'] = obj.passiveFrameAppearTimes[obj.changeFrames[trials]]
                    data[expName]['passiveRunTime'] = obj.passiveRunTime
                    data[expName]['passiveRunSpeed'] = obj.passiveRunSpeed
                if makeSDFs:
                    data[expName]['sdfs'] = getSDFs(obj,probes=probes,**sdfParams)

                fileIO.objToHDF5(obj=None,saveDict=data,filePath=popHDF5Path)


def getSDFs(obj,probes='all',behaviorStates=('active','passive'),epochs=('change','preChange'),preTime=0.25,postTime=0.75,sampInt=0.001,sdfFilt='exp',sdfSigma=0.005,avg=False,psth=False):
    if probes=='all':
        probes = obj.probes_to_analyze
    
    trials = ~(obj.earlyResponse | obj.autoRewarded)
    changeFrames = np.array(obj.trials['change_frame']).astype(int)+1 #add one to correct for change frame indexing problem
    flashFrames = np.array(obj.core_data['visual_stimuli']['frame'])
    
    sdfs = {probe: {state: {epoch: [] for epoch in epochs} for state in behaviorStates} for probe in probes}
    
    for probe in probes:
        units = probeSync.getOrderedUnits(obj.units[probe])
        for state in sdfs[probe]:
            if state=='active' or obj.passive_pickle_file is not None:  
                frameTimes = obj.frameAppearTimes if state=='active' else obj.passiveFrameAppearTimes
                changeTimes = frameTimes[changeFrames[trials]]
                if 'preChange' in epochs:
                    flashTimes = frameTimes[flashFrames]
                    preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
                for u in units:
                    spikes = obj.units[probe][u]['times']
                    for epoch in epochs:
                        t = changeTimes if epoch=='change' else preChangeTimes
                        if psth:
                            s = analysis_utils.makePSTH(spikes,t-preTime,preTime+postTime,binSize=sampInt,avg=avg)
                        else:
                            s = analysis_utils.getSDF(spikes,t-preTime,preTime+postTime,sampInt=sampInt,filt=sdfFilt,sigma=sdfSigma,avg=avg)[0]
                        sdfs[probe][state][epoch].append(s)                    
    return sdfs


def findResponsiveUnits(sdfs,baseWin,respWin,thresh=5,posRespOnly=False):
    unitMeanSDFs = sdfs.mean(axis=1) if len(sdfs.shape)>2 else sdfs.copy()
    hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
    unitMeanSDFs -= unitMeanSDFs[:,baseWin].mean(axis=1)[:,None]
    hasResp = unitMeanSDFs[:,respWin].max(axis=1) > thresh*unitMeanSDFs[:,baseWin].std(axis=1)
    if posRespOnly:
        hasResp = hasResp & (unitMeanSDFs[:,respWin].mean(axis=1)>0)
    return hasSpikes,hasResp

    
def findLatency(data,baseWin=None,stimWin=None,method='rel',thresh=3,minPtsAbove=30,maxval=None):
    latency = []
    if len(data.shape)<2:
        data = data[None,:]
    if baseWin is not None:
        data = data-data[:,baseWin].mean(axis=1)[:,None]
    if stimWin is None:
        stimWin = slice(0,data.shape[1])
    for d in data:
        if method=='abs':
            pass
        elif method=='rel':
            thresh *= d[stimWin].max()
        else:
            thresh *= d[baseWin].std()
        ptsAbove = np.where(np.correlate(d[stimWin]>thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        if len(ptsAbove)<1 or (maxval is not None and ptsAbove[0]>maxval):
            latency.append(np.nan)
        else:
            latency.append(ptsAbove[0])
    return np.array(latency)
    

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



baseDir = 'Z:\\'
localDir = r'C:\Users\svc_ccg\Desktop\Analysis\Probe'

mouseInfo = (
             ('409096',('03212019',),('ABCD',),'A',(False,)),
             ('417882',('03262019','03272019'),('ABCEF','ABCF'),'AA',(False,False)),
             ('408528',('04042019','04052019'),('ABCDE','ABCDE'),'AB',(True,True)),
             ('408527',('04102019','04112019'),('BCDEF','BCDEF'),'AB',(True,True)),
             ('421323',('04252019','04262019'),('ABCDEF','ABCDEF'),'AB',(True,True)),
             ('422856',('04302019',),('ABCDEF',),'A',(True,)),
             ('423749',('05162019','05172019'),('ABCDEF','ABCDEF'),'AB',(True,True)),
             ('427937',('06072019',),('ABCDF',),'B',(True,)),
             ('423745',('06122019',),('ABCDEF',),'A',(True,)),
             ('429084',('07112019','07122019'),('ABCDEF','ABCDE'),'AB',(True,True)),
             ('423744',('08082019','08092019'),('ABCDEF','ABCDEF'),'AA',(True,True)),
             ('423750',('08132019','08142019'),('AF','AF'),'AA',(True,True)),
             ('459521',('09052019','09062019'),('ABCDEF','ABCDEF'),'AA',(True,True)),
             ('461027',('09122019','09132019'),('ABCDEF','ABCDEF'),'AA',(True,True)),
             ('479219',('11262019',),('BCD',),'A',(True,)),
             ('484106',('12132019',),('BCDEF',),'A',(True,)),
             ('474732',('12192019','12202019'),('ABCDEF','ABCDE'),'AA',(True,True)),
            )


#
makeSummaryPlots(miceToAnalyze=('484106','474732'))


# make new experiment hdf5s without updating popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=False,miceToAnalyze=('479219',))

# make new experiment hdf5s and add to existing popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True,miceToAnalyze=('484106','474732'))

# make new experiment hdf5s and popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True)

# make popData.hdf5 from existing experiment hdf5s without SDFs
getPopData(objToHDF5=False,popDataToHDF5=True,makeSDFs=False)

# append existing hdf5s to existing popData.hdf5
getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze=('479219',))



data = h5py.File(os.path.join(localDir,'popData.hdf5'),'r')



# A or B days that have passive session
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im and hasPassive] for im in 'AB']


stimWin = slice(250,500)



###### behavior analysis

exps = Aexps+Bexps

hitRate = []
falseAlarmRate = []
dprime = []
for exp in exps:
    response = data[exp]['response'][:]
    hit = response=='hit'
    changeTimes = data[exp]['behaviorChangeTimes'][:] 
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    hit,miss,fa,cr = [np.sum(response[engaged]==r) for r in ('hit','miss','falseAlarm','correctReject')]
    hitRate.append(hit/(hit+miss))
    falseAlarmRate.append(fa/(cr+fa))
    dprime.append(calcDprime(hit,miss,fa,cr))

mouseID = [exp[-6:] for exp in exps]
nMice = len(set(mouseID))

#mouseAvg = []    
#for param in (hitRate,falseAlarmRate,dprime):
#    d = []
#    for mouse in set(mouseID):
#        mouseVals = [p for p,m in zip(param,mouseID) if m==mouse]
#        d.append(sum(mouseVals)/len(mouseVals))
#    mouseAvg.append(d)
#hitRate,falseAlarmRate,dprime = mouseAvg

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for h,fa in zip(hitRate,falseAlarmRate):
    ax.plot([0,1],[h,fa],'0.5')
for x,y in enumerate((hitRate,falseAlarmRate)):
    m = np.mean(y)
    s = np.std(y)/(len(y)**0.5)
    ax.plot(x,m,'ko',ms=10,mec='k',mfc='k')
    ax.plot([x,x],[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=16)
ax.set_xticks([0,1])
ax.set_xticklabels(['Change','Catch'])
ax.set_xlim([-0.25,1.25])
ax.set_ylim([0,1])
ax.set_ylabel('Response Probability',fontsize=16)
ax.set_title('n = '+str(nMice)+' mice,\n'+str(len(exps))+' experiments',fontsize=16)

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(np.zeros(len(dprime)),dprime,'o',ms=10,mec='0.5',mfc='none')
m = np.mean(dprime)
s = np.std(dprime)/(len(dprime)**0.5)
ax.plot(0,m,'ko',ms=10,mec='k',mfc='k')
ax.plot([0,0],[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=16)
ax.set_xticks([])
ax.set_ylim([0,3.5])
ax.set_ylabel('d prime',fontsize=16)
ax.set_title('n = '+str(nMice)+' mice, '+str(len(exps))+' experiments',fontsize=16)


# compare active and passive running

# overall, engaged change times, overall distribution , distribution of overall medians
sessionLabels = ('behavior','passive')
overallRunSpeed = {session: [] for session in sessionLabels}
sortedSpeeds = {session: [] for session in sessionLabels}
cumProbSpeed = {session: [] for session in sessionLabels}
changeRunSpeed = {session: [] for session in sessionLabels}
for exp in exps:
    print(exp)
    response = data[exp]['response'][:]
    hit = response=='hit'
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    for session in sessionLabels:
        runTime = data[exp][session+'RunTime'][:]
        runSpeed = data[exp][session+'RunSpeed'][:]
        changeTimes = data[exp][session+'ChangeTimes'][:]
        overallRunSpeed[session].append(np.median(runSpeed))
        sortedSpeeds[session].append(np.sort(runSpeed))
        cumProbSpeed[session].append([np.searchsorted(runSpeed,s)/runSpeed.size for s in sortedSpeeds[session][-1]])
        s = []
        for t in changeTimes[engaged]:
            i = np.searchsorted(runTime,t)
            s.append(np.median(runSpeed[i-1:i+2]))
        changeRunSpeed[session].append(s)


for speed in (overallRunSpeed,changeRunSpeed):          
    if speed is overallRunSpeed:
        behaviorRunSpeed,passiveRunSpeed = [speed[session] for session in sessionLabels]
        lbl = 'overall'
    else:
        behaviorRunSpeed,passiveRunSpeed = [[np.median(s) for s in speed[session]] for session in sessionLabels]
        lbl = 'at change time'
    amax = 1.05*max(np.max(behaviorRunSpeed),np.max(passiveRunSpeed))
    
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    ax.plot([0,amax],[0,amax],'k--')
    ax.plot(behaviorRunSpeed,passiveRunSpeed,'o',ms=10,mec='k',mfc='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-5,amax])
    ax.set_ylim([-5,amax])
    ax.set_aspect('equal')
    ax.set_xlabel('Median behavior run speed '+lbl+' (cm/s)')
    ax.set_ylabel('Median passive run speed '+lbl+' (cm/s)')
    plt.tight_layout()
    
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    bins = np.arange(-5,100)
    ymax = -10
    for session,spd,clr in zip(sessionLabels,(behaviorRunSpeed,passiveRunSpeed),'mg'):
        n = ax.hist(spd,bins,color=clr,edgecolor='none',alpha=0.5,label=session)[0]
        ymax = max(ymax,max(n))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_yticks(np.arange(ymax+1))
    ax.set_xlim([-5,amax])
    ax.set_ylim([0,3.05])
    ax.set_xlabel('Median run speed '+lbl+' (cm/s)')
    ax.set_ylabel('Number of sessions')
    ax.legend()
    plt.tight_layout()
    
    
speedIntp = np.arange(-50,200,0.1)    
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for session,clr in zip(sessionLabels,('mg')):
    probIntp = [np.interp(speedIntp,speed,prob) for speed,prob in zip(sortedSpeeds[session],cumProbSpeed[session])]
    m = np.mean(probIntp,axis=0)
    s = np.std(probIntp,axis=0)/(len(probIntp)**0.5)
    ax.plot(speedIntp,m,color=clr,label=session)
    ax.fill_between(speedIntp,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-10,100])
ax.set_xlabel('Run speed (cm/s)')
ax.set_ylabel('Cumulative fraction of samples')
ax.legend()
plt.tight_layout()
    



###### change mod and latency analysis

exps = Aexps+Bexps

baseWin = slice(stimWin.start-250,stimWin.start)
respWin = slice(stimWin.start,stimWin.start+250)

regionNames = (
               ('LGd',('LGd',)),
               ('V1',('VISp',)),
               ('LM',('VISl',)),
               ('RL',('VISrl',)),
               ('LP',('LP',)),
               ('AL',('VISal',)),
               ('PM',('VISpm',)),
               ('AM',('VISam',)),
               ('LD',('LD',)),
               ('LGv',('LGv',)),
               ('RT',('RT',)),
               ('SCd',('SCig','SCig-a','SCig-b')),
               ('APN',('APN',)),
               ('NOT',('NOT',)),
               ('POL',('POL',)),
               ('MRN',('MRN',)),
               ('MB',('MB',)),
               ('st',('st','CP')),
               ('SUB',('SUB','PRE','POST')),
               ('hipp',('CA1','CA3','DG-mo','DG-po','DG-sg','HPF'))
              )
regionNames = regionNames[:8]
regionLabels = [r[0] for r in regionNames]

behavStates = ('active','passive')
epochLabels = ('preChange','change')
changeModMethods = ('allImages','eachImage','prefImage')
trialLabels = ('change','hit','miss')

result = {region: {'mouseIDs': [],
                   'expDates': [],
                   'unitCount': [],
                   'base': {state: {epoch: [] for epoch in epochLabels} for state in behavStates},
                   'sdfs': {state: {epoch: {method: {trials: [] for trials in trialLabels} for method in changeModMethods} for epoch in epochLabels} for state in behavStates},
                   'resp': {state: {epoch: {method: {trials: [] for trials in trialLabels} for method in changeModMethods} for epoch in epochLabels} for state in behavStates},
                   'changeMod': {state: {method: {trials: [] for trials in trialLabels} for method in changeModMethods} for state in behavStates},
                   'firstSpikeLat': []} for region in regionLabels}

for regionInd,(region,regionCCFLabels) in enumerate(regionNames):
    print(region)
    for exp in exps:
        print(exp)
        response = data[exp]['response'][:]
        hit = response=='hit'
        miss = response=='miss'
        changeTimes = data[exp]['behaviorChangeTimes'][:]
        engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
        trialIndex = {'change': engaged & (hit | miss),
                      'hit': engaged & hit,
                      'miss': engaged & miss}
        initialImage = data[exp]['initialImage'][:]
        changeImage = data[exp]['changeImage'][:]
        imageNames = np.unique(changeImage)
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,regionCCFLabels)
            if any(inRegion):
                sdfs = {}
                for state in behavStates:
                    sdfs[state] = {}
                    for epoch in epochLabels:
                        sdfs[state][epoch] = data[exp]['sdfs'][probe][state][epoch][:][inRegion]
                hasSpikesActive,hasRespActive = findResponsiveUnits(sdfs['active']['change'][:,trialIndex['change']].mean(axis=1),baseWin,respWin,thresh=5,posRespOnly=False)
                if 'passive' in behavStates:
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(sdfs['passive']['change'][:,trialIndex['change']].mean(axis=1),baseWin,respWin,thresh=5,posRespOnly=False)
                    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
                else:
                    hasResp = hasSpikesActive & hasRespActive
                if hasResp.sum()>0:
                    result[region]['mouseIDs'].append(exp[-6:])
                    result[region]['expDates'].append(exp[:8])
                    result[region]['unitCount'].append(hasResp.sum())
                    
                    resp = {}
                    for state in behavStates:
                        resp[state] = {}
                        for epoch in epochLabels:
                            base = sdfs[state][epoch][:,:,baseWin].mean(axis=(1,2))
                            sdfs[state][epoch] -= base[:,None,None]
                            result[region]['base'][state][epoch].append(base[hasResp])
                            for trials in trialLabels:
                                result[region]['sdfs'][state][epoch]['allImages'][trials].append(sdfs[state][epoch][hasResp][:,trialIndex[trials]].mean(axis=1))
                                result[region]['resp'][state][epoch]['allImages'][trials].append(sdfs[state][epoch][hasResp][:,trialIndex[trials],respWin].mean(axis=(1,2)))
                     
                    for state in behavStates:
                        for trials in trialLabels:
                            pre,change = [result[region]['resp'][state][epoch]['allImages'][trials][-1] for epoch in epochLabels]
                            result[region]['changeMod'][state]['allImages'][trials].append(np.clip((change-pre)/(change+pre),-1,1))
                        
                            imgPreSdfs = np.full((len(imageNames),hasResp.sum(),sdfs['active']['change'].shape[2]),np.nan)
                            imgChangeSdfs = imgPreSdfs.copy()
                            imgPreResp = np.full((len(imageNames),hasResp.sum()),np.nan)
                            imgChangeResp = imgPreResp.copy()
                            for i,img in enumerate(imageNames):  
                                preChangeIndex = trialIndex[trials] & (initialImage==img)
                                changeIndex = trialIndex[trials] & (changeImage==img)
                                if any(preChangeIndex) and any(changeIndex):
                                    imgPreSdfs[i] = sdfs[state]['preChange'][hasResp][:,preChangeIndex].mean(axis=1)
                                    imgChangeSdfs[i] = sdfs[state]['change'][hasResp][:,changeIndex].mean(axis=1)
                                    imgPreResp[i] = imgPreSdfs[i][:,respWin].mean(axis=1)
                                    imgChangeResp[i] = imgChangeSdfs[i][:,respWin].mean(axis=1)
                            
                            if not np.all(np.isnan(imgChangeResp)):
                                imgChangeMod = np.clip((imgChangeResp-imgPreResp)/(imgChangeResp+imgPreResp),-1,1)
                                result[region]['sdfs'][state]['preChange']['eachImage'][trials].append(np.nanmean(imgPreSdfs,axis=0))
                                result[region]['sdfs'][state]['change']['eachImage'][trials].append(np.nanmean(imgChangeSdfs,axis=0))
                                result[region]['resp'][state]['preChange']['eachImage'][trials].append(np.nanmean(imgPreResp,axis=0))
                                result[region]['resp'][state]['change']['eachImage'][trials].append(np.nanmean(imgChangeResp,axis=0))
                                result[region]['changeMod'][state]['eachImage'][trials].append(np.nanmean(imgChangeMod,axis=0))
                                
                                prefImgIndex = np.s_[np.nanargmax(imgChangeResp,axis=0),np.arange(hasResp.sum())]
                                result[region]['sdfs'][state]['preChange']['prefImage'][trials].append(imgPreSdfs[prefImgIndex])
                                result[region]['sdfs'][state]['change']['prefImage'][trials].append(imgChangeSdfs[prefImgIndex])
                                result[region]['resp'][state]['preChange']['prefImage'][trials].append(imgPreResp[prefImgIndex])
                                result[region]['resp'][state]['change']['prefImage'][trials].append(imgChangeResp[prefImgIndex])
                                result[region]['changeMod'][state]['prefImage'][trials].append(imgChangeMod[prefImgIndex])
                    
                    # first spike latency
                    for u in data[exp]['units'][probe][inRegion][hasResp]:
                        spikeTimes = data[exp]['spikeTimes'][probe][str(u)][:,0]
                        lat = []
                        for t in changeTimes:
                            firstSpike = np.where((spikeTimes > t+0.03) & (spikeTimes < t+0.15))[0]
                            if len(firstSpike)>0:
                                lat.append(spikeTimes[firstSpike[0]]-t)
                            else:
                                lat.append(np.nan)
                        result[region]['firstSpikeLat'].append(np.nanmedian(lat))

def concatDictArrays(d):
    for key in d:
        if isinstance(d[key],list) and len(d[key])>0:
            if isinstance(d[key][0],np.ndarray):
                d[key] = np.concatenate(d[key])
        elif isinstance(d[key],dict):
            concatDictArrays(d[key])

concatDictArrays(result)

                       
# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(result,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
result = pickle.load(open(pkl,'rb'))


nMice = [len(set(result[region]['mouseIDs'])) for region in result]
nExps = [len(set(result[region]['expDates'])) for region in result]
nUnits = [sum(result[region]['unitCount']) for region in result]

totalMice = len(set(exp[-6:] for exp in exps))
totalExps = len(exps)
totalUnits = sum(nUnits)

anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionNames for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a==r[1][0]] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')]    
hier = hierScore_8regions

hierColors = np.array([[217,141,194], # LGd
                       [129,116,177], # V1
                       [78,115,174], # LM
                       [101,178,201], # RL
                       [88,167,106], #LP
                       [202,183,120], # AL
                       [219,132,87], # PM
                       [194,79,84]] # AM
                     ).astype(float)
hierColors /= 255

figSaveDir = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\changeMod figure for npx platform paper'

# change mod active vs passive
#for method,ylim in zip(changeModMethods,([0,0.44],[0,0.36],[0.25,0.6])):
for method,ylim in zip(('eachImage',),([0,0.36],)):
    for trials in ('change',):#trialLabels:
        fig = plt.figure(facecolor='w',figsize=(6,6))
        ax = plt.subplot(1,1,1)
        for state,fill,fitClr in zip(behavStates,(True,False),('k','0.5')):
            d = [result[region]['changeMod'][state][method][trials]for region in result]
            mn = [np.nanmean(regionData) for regionData in d]
            ci = [np.percentile([np.nanmean(np.random.choice(regionData,len(regionData),replace=True)) for _ in range(5000)],(2.5,97.5)) for regionData in d]
            slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,mn)
            x = np.array([min(hier),max(hier)])
            ax.plot(x,slope*x+yint,'--',color=fitClr)
            r,p = scipy.stats.pearsonr(hier,mn)
            if state=='active':
                title = ''
            else:
                title +='\n'
            title += 'Pearson ('+state+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
            r,p = scipy.stats.spearmanr(hier,mn)
            title += '\nSpearman ('+state+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
            for i,(h,m,c,clr) in enumerate(zip(hier,mn,ci,hierColors)):
                mfc = clr if fill else 'none'
                lbl = state if i==0 else None
                ax.plot(h,m,'o',mec=clr,mfc=mfc,ms=6,label=lbl)
                ax.plot([h,h],c,color=clr)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim(ylim)
        ax.set_xticks([-0.4,-0.2,0,0.2,0.4])
#        ax.set_xticks(hier)
#        ax.set_xticklabels([str(round(h,2))+'\n'+r[0]+'\n'+str(nu)+'\n'+str(nm) for h,r,nu,nm in zip(hier,regionNames,nUnits,nMice)])
        ax.set_xlabel('Hierarchy Score',fontsize=12)
        ax.set_ylabel('Change Modulation Index',fontsize=12)
        ax.set_title(title,fontsize=8)
        ax.legend(loc='upper left')
        plt.tight_layout()
        
# change mod hit vs miss
#for method,ylim in zip(changeModMethods,([0,0.44],[0,0.36],[0.25,0.6])):
for method,ylim in zip(('eachImage',),([0,0.36],)):
    fig = plt.figure(facecolor='w',figsize=(6,6))
    ax = plt.subplot(1,1,1)
    for trials,fill,fitClr in zip(('hit','miss'),(True,False),('k','0.5')):
        d = [result[region]['changeMod']['active'][method][trials]for region in result]
        mn = [np.nanmean(regionData) for regionData in d]
        ci = [np.percentile([np.nanmean(np.random.choice(regionData,len(regionData),replace=True)) for _ in range(5000)],(2.5,97.5)) for regionData in d]
        slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,mn)
        x = np.array([min(hier),max(hier)])
        ax.plot(x,slope*x+yint,'--',color=fitClr)
        r,p = scipy.stats.pearsonr(hier,mn)
        if trials=='hit':
            title = ''
        else:
            title +='\n'
        title += 'Pearson ('+trials+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
        r,p = scipy.stats.spearmanr(hier,mn)
        title += '\nSpearman ('+trials+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
        for i,(h,m,c,clr) in enumerate(zip(hier,mn,ci,hierColors)):
            mfc = clr if fill else 'none'
            lbl = trials if i==0 else None
            ax.plot(h,m,'o',mec=clr,mfc=mfc,ms=6,label=lbl)
            ax.plot([h,h],c,color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim(ylim)
    ax.set_xticks([-0.4,-0.2,0,0.2,0.4])
#    ax.set_xticks(hier)
#    ax.set_xticklabels([str(round(h,2))+'\n'+r[0]+'\n'+str(nu)+'\n'+str(nm) for h,r,nu,nm in zip(hier,regionNames,nUnits,nMice)])
    ax.set_xlabel('Hierarchy Score',fontsize=12)
    ax.set_ylabel('Change Modulation Index',fontsize=12)
    ax.set_title(title,fontsize=8)
    ax.legend(loc='upper left')
    plt.tight_layout()
        
# time to first spike
fig = plt.figure(facecolor='w',figsize=(6,5))
ax = plt.subplot(1,1,1)
d = [np.array(result[region]['firstSpikeLat'])*1000 for region in result]
mn = [np.nanmean(regionData) for regionData in d]
ci = [np.percentile([np.nanmean(np.random.choice(regionData,len(regionData),replace=True)) for _ in range(5000)],(2.5,97.5)) for regionData in d]
slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,mn)
x = np.array([min(hier),max(hier)])
ax.plot(x,slope*x+yint,'--',color='k')
r,p = scipy.stats.pearsonr(hier,mn)
title = 'Pearson: r = '+str(round(r,2))+', p = '+str(round(p,3))
r,p = scipy.stats.spearmanr(hier,mn)
title += '\nSpearman: r = '+str(round(r,2))+', p = '+str(round(p,3))
for h,m,c,clr in zip(hier,mn,ci,hierColors):
    ax.plot(h,m,'o',mec=clr,mfc=clr,ms=6)
    ax.plot([h,h],c,color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks([-0.4,-0.2,0,0.2,0.4])
#ax.set_xticks(hier)
#ax.set_xticklabels([str(round(h,2))+'\n'+r[0]+'\n'+str(nu)+'\n'+str(nm) for h,r,nu,nm in zip(hier,regionNames,nUnits,nMice)])
ax.set_xlabel('Hierarchy Score',fontsize=12)
ax.set_ylabel('Time to first spike (ms)',fontsize=12)
ax.set_title(title,fontsize=8)
plt.tight_layout()

# baseline, preResp, changeResp
for method in ('eachImage',):
    for rate,epoch,ylbl in zip(('base','resp','resp'),('change','preChange','change'),('Baseline Rate','Pre-change Response','Change Response')):
        fig = plt.figure(facecolor='w',figsize=(6,5))
        ax = plt.subplot(1,1,1)
        ymax = 0
        for state,fill in zip(behavStates,(True,False)):
            if rate=='base':
                d = [result[region][rate][state][epoch] for region in result]
            else:
                d = [result[region][rate][state][epoch][method]['change'] for region in result]
            mn = [np.nanmean(regionData) for regionData in d]
            ci = [np.percentile([np.nanmean(np.random.choice(regionData,len(regionData),replace=True)) for _ in range(5000)],(2.5,97.5)) for regionData in d]
            r,p = scipy.stats.pearsonr(hier,mn)
            if state=='active':
                title = ''
            else:
                title +='\n'
            title += 'Pearson ('+state+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
            r,p = scipy.stats.spearmanr(hier,mn)
            title += '\nSpearman ('+state+'): r = '+str(round(r,2))+', p = '+str(round(p,3))
            for i,(h,m,c,clr) in enumerate(zip(hier,mn,ci,hierColors)):
                mfc = clr if fill else 'none'
                lbl = state if i==0 else None
                ax.plot(h,m,'o',mec=clr,mfc=mfc,ms=6,label=lbl)
                ax.plot([h,h],c,color=clr)
                ymax = max(ymax,c[1])
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0,1.05*ymax])
        ax.set_xticks([-0.4,-0.2,0,0.2,0.4])
        #ax.set_xticks(hier)
        #ax.set_xticklabels([str(round(h,2))+'\n'+r[0]+'\n'+str(nu)+'\n'+str(nm) for h,r,nu,nm in zip(hier,regionNames,nUnits,nMice)])
        ax.set_xlabel('Hierarchy Score',fontsize=12)
        ax.set_ylabel(ylbl+' (spikes/s)',fontsize=12)
        ax.set_title(title,fontsize=8)
        ax.legend(loc='lower right')
        plt.tight_layout()

# parameter distributions       
for param,xlbl,xticks in zip(('firstSpikeLat','changeMod'),('Time to first spike (ms)','Change Modulation Index'),((40,80,120),(-1,-0.5,0,0.5,1))):
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    if param=='firstSpikeLat':
        r = [np.array(result[region][param])*1000 for region in result]
    else:
        r = [result[region][param]['active']['eachImage']['change'] for region in result]
    for d,clr,lbl in zip(r,hierColors,regionLabels):
        d = d[~np.isnan(d)]
        sortd = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in sortd]
        ax.plot(sortd,cumProb,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(xticks)
    ax.set_yticks([0,0.5,1])
    ax.set_xlabel(xlbl,fontsize=12)
    ax.set_ylabel('Cumulative Probability',fontsize=12)
    ax.legend()
    plt.tight_layout()

# p value matrix
alpha = 0.05
for param,lbl in zip(('firstSpikeLat','changeMod'),('Time to first spike','Change Modulation Index')):
    if param=='firstSpikeLat':
        d = [np.array(result[region][param]) for region in result]
    else:
        d = [result[region][param]['active']['eachImage']['change'] for region in result]
    
    comparison_matrix = np.full((len(regionLabels),)*2,np.nan) 
    for i,r1 in enumerate(d):
        for j,r2 in enumerate(d):
            if j>i:
                z, comparison_matrix[i,j] = ranksums(r1[np.invert(np.isnan(r1))],
                                                     r2[np.invert(np.isnan(r2))])
            
    p_values = comparison_matrix.flatten()
    ok_inds = ~np.isnan(p_values)
    
    reject, p_values_corrected, alphaSidak, alphacBonf = multipletests(p_values[ok_inds], alpha=alpha, method='fdr_bh')
            
    p_values_corrected2 = np.full((len(p_values),),np.nan)
    p_values_corrected2[ok_inds] = p_values_corrected
    comparison_corrected = np.reshape(p_values_corrected2, comparison_matrix.shape)
    
    lim = (1e-5,alpha)
    clim = np.log10(lim)
    
    pmatrix = comparison_corrected
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color=np.array((255, 251, 204))/255)
    im = ax.imshow(np.log10(pmatrix),cmap=cmap,clim=clim)
    ax.set_xticks(np.arange(len(regionLabels)))
    ax.set_xticklabels(regionLabels)
    ax.set_yticks(np.arange(len(regionLabels)))
    ax.set_yticklabels(regionLabels)
    ax.set_ylim([-0.5,len(regionLabels)-0.5])
    ax.set_xlim([-0.5,len(regionLabels)-0.5])
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.set_ticks(clim)
    cb.set_ticklabels(lim)
    plt.tight_layout()

# population sdfs
xlim = [0,350]
for region in ('LGd','V1','AM'):#result:
    for state in ('active',):#behavStates:
        for method in ('eachImage',):#changeModMethods:
            for trials in ('change',):#trialLabels:
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                for epoch,clr in zip(epochLabels,('0.5','k')):
                    sdfs = result[region]['sdfs'][state][epoch][method][trials][:,stimWin.start+xlim[0]:stimWin.start+xlim[1]]
                    m = sdfs.mean(axis=0)
                    s = sdfs.std(axis=0)/(len(sdfs)**0.5)
                    ax.plot(m,color=clr,label=epoch)
                    ax.fill_between(np.arange(len(m)),m+s,m-s,color=clr,alpha=0.25) 
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks([0,100,200,300])
                ax.set_xlim(xlim)
                ax.set_ylabel('Spikes/s')
                ax.set_title(region+' '+state+' '+method+' '+trials)
                ax.legend(loc='upper right')
                plt.tight_layout()
                for ext in ('.png','.pdf'):
                    plt.savefig(os.path.join(figSaveDir,'changeMod','SDFs',region+'_'+state+'_'+method+'_'+trials+ext))
                plt.close(fig)
                
# sdfs for schematic
xlim = [0,150]
for region,clr,ylim in zip(('V1','AM'),'br',([-1.5,15],[-1.2,12])):#result:
    for i,(state,method,trials) in enumerate((('active','allImages','change'),('passive','allImages','change'),('active','eachImage','miss'))):
                for epoch in epochLabels:
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    sdfs = result[region]['sdfs'][state][epoch][method][trials][:,stimWin.start+xlim[0]:stimWin.start+xlim[1]]
                    ax.plot(sdfs.mean(axis=0),color=clr,lw=20)
                    for side in ('right','top','left','bottom'):
                        ax.spines[side].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    plt.tight_layout()
                    for ext in ('.pdf',):
                        plt.savefig(os.path.join(figSaveDir,'changeMod','SDFs','forSchematic',region+'_'+str(i+1)+'_'+epoch+ext))
                    plt.close(fig)

# make dataframe for Josh (old)
d = OrderedDict()
colLabels = ('Experiment Date','Mouse ID','Region','Change Modulation Index','Time to first spike','Baseline Rate','Pre-change Response','Change Response')
for i,(key,val) in enumerate(zip(colLabels,(expDates,mouseIDs,regionLabels,changeModActive,activeFirstSpikeLat,baseRateActive,preRespActive,changeRespActive))):
    if i<2:
        val = [[v]*n for reg,count in zip(val,unitCount) for v,n in zip(reg,count)]
    elif i==2:
        val = [[v]*n for v,n in zip(val,nUnits)]
    d[key] = np.concatenate(val)
df = pd.DataFrame(data=d)
f = fileIO.saveFile(fileType='*.hdf5')
df.to_hdf(f,'table')
    


###### adaptation

exps = Aexps+Bexps

baseWin = slice(stimWin.start-150,stimWin.start)
respWin = slice(stimWin.start,stimWin.start+150)

cortical_cmap = plt.cm.plasma
subcortical_cmap = plt.cm.Reds
regions = (('LGd','LGd',(0,0,0)),
           ('V1','VISp',cortical_cmap(0)),
           ('LM','VISl',cortical_cmap(0.1)),
           ('RL','VISrl',cortical_cmap(0.2)),
           ('AL','VISal',cortical_cmap(0.3)),
           ('PM','VISpm',cortical_cmap(0.4)),
           ('AM','VISam',cortical_cmap(0.5)),
           ('LP','LP',subcortical_cmap(0.4)))
regionLabels = [r[0] for r in regions]
regionCCFLabels = [r[1] for r in regions]
regionColors = [r[2] for r in regions]

nFlashes = 10

behavStates = ('active','passive')

betweenChangeSdfs = {region: {state:[] for state in behavStates} for region in regionLabels}
for exp in exps:
    print(exp)   
    
    response = data[exp]['response'][:]
    lickTimes = data[exp]['lickTimes'][:]
    hit = response=='hit'
    falseAlarm = response=='falseAlarm'
    behaviorChangeTimes = data[exp]['behaviorChangeTimes'][:]
    passiveChangeTimes = data[exp]['passiveChangeTimes'][:]
    engaged = np.array([np.sum(hit[(behaviorChangeTimes>t-60) & (behaviorChangeTimes<t+60)]) > 1 for t in behaviorChangeTimes])
    changeTrials = engaged & (hit | (response=='miss'))
    
    for region in regionCCFLabels:
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,region)
            if any(inRegion):
                activePre,activeChange = [data[exp]['sdfs'][probe]['active'][epoch][inRegion,:] for epoch in ('preChange','change')]
                hasSpikesActive,hasRespActive = findResponsiveUnits(activeChange[:,changeTrials],baseWin,respWin,thresh=5)
                if 'passive' in behavStates:
                    passivePre,passiveChange = [data[exp]['sdfs'][probe]['passive'][epoch][inRegion,:] for epoch in ('preChange','change')]
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChange[:,changeTrials],baseWin,respWin,thresh=5)
                    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
                else:
                    hasResp = hasSpikesActive & hasRespActive
                if hasResp.any():
                    units = data[exp]['units'][probe][inRegion][hasResp]
                    spikes = data[exp]['spikeTimes'][probe]
                    for state in behavStates:
                        changeTimes = behaviorChangeTimes if state=='active' else passiveChangeTimes
                        betweenChangeSdfs[region][state].append([analysis_utils.getSDF(spikes[u],changeTimes-0.25,0.25+nFlashes*0.75,sampInt=0.001,filt='exp',sigma=0.005,avg=True)[0] for u in units])


popSdfChange = {state:[] for state in ('active','passive')}
popSdfNonChange = {state:[] for state in ('active','passive')}
adaptMean = {state:[] for state in ('active','passive')}
adaptSem = {state:[] for state in ('active','passive')}
adaptMatrix = {state:[] for state in ('active','passive')}
t = np.arange(-0.25,nFlashes*0.75,0.001)
flashTimes = np.arange(0,nFlashes*0.75,0.75)
for state in behavStates:
    fig = plt.figure(facecolor='w',figsize=(8,6))
#    fig.suptitle(state,fontsize=14)
    ax = fig.subplots(2,1)
    for ft in flashTimes:
        ax[0].add_patch(matplotlib.patches.Rectangle([ft,-0.35],width=0.25,height=0.1,color='0.5',alpha=0.5,zorder=0))
    for region,clr,lbl in zip(regionCCFLabels,regionColors,regionLabels):
        sdfs = np.concatenate(betweenChangeSdfs[region][state])
        sdfs -= sdfs[:,:250].mean(axis=1)[:,None]
        m = sdfs.mean(axis=0)
        s = sdfs.std()/(len(sdfs)**0.5)
        s /= m.max()
        m /= m.max()
        ax[0].plot(t,m,color=clr,label=lbl)
        ax[0].fill_between(t,m+s,m-s,color='w',alpha=0.25)
        
        popSdfChange[state].append(m[250:750])
        lastNonChangeFlashStart = nFlashes*750-750+250
        popSdfNonChange[state].append(m[lastNonChangeFlashStart:lastNonChangeFlashStart+500])
        
        flashResp = []
        flashRespSem = []
        for i in np.arange(250,nFlashes*750,750):
            r = sdfs[:,i:i+250].max(axis=1)
            r -= sdfs[:,i-250:i].mean(axis=1)
            flashResp.append(r.mean())
            flashRespSem.append(r.std()/(len(r)**0.5))
        flashResp,flashRespSem = [np.array(r)/flashResp[0] for r in (flashResp,flashRespSem)]
        ax[1].plot(flashTimes+0.05,flashResp,color=clr,marker='o')
        for x,m,s in zip(flashTimes+0.05,flashResp,flashRespSem):
            ax[1].plot([x,x],[m-s,m+s],color=clr)
        adaptMean[state].append(flashResp[-1])
        adaptSem[state].append(flashRespSem[-1])
        
        binsize = 1
        chng,nonchng = [s.reshape((s.shape[0],int(s.shape[1]/binsize),-1)).mean(axis=2) for s in (sdfs[:,250:750],sdfs[:,lastNonChangeFlashStart:lastNonChangeFlashStart+500])]
        a = (chng-nonchng)/chng.max(axis=1)[:,None]
        a[np.isinf(a)] = np.nan
        adaptMatrix[state].append(np.nanmean(a,axis=0))
        
    for a in ax:
        for side in ('right','top'):
            a.spines[side].set_visible(False)
        a.tick_params(direction='out',top=False,right=False,labelsize=12)
        a.set_xlim([-0.25,7.5])
    ax[0].set_yticks([0,0.5,1])
    ax[0].set_ylabel('Normalized spike rate',fontsize=14)
#    ax[0].legend(loc='upper right',fontsize=12)
    ax[1].set_ylim([0.4,1.1])
    ax[1].set_xlabel('Time after change (s)',fontsize=14)
    ax[1].set_ylabel('Normalized peak response',fontsize=14)
    plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
for state in behavStates:
    for x,(m,s,clr) in enumerate(zip(adaptMean[state],adaptSem[state],regionColors)):
        mfc = clr if state=='active' else 'none'
        lbl = state if x==0 else None
        ax.plot(x,m,'o',mec=clr,mfc=mfc,ms=10,label=lbl)
        ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    
    ax.set_xticks(np.arange(len(regionLabels)))
    ax.set_xticklabels(regionLabels)
    ax.set_yticks([0.5,0.6,0.7,0.8])
    ax.set_xlim([-0.5,len(regionLabels)-0.5])
    ax.set_ylim([0.5,0.8])
    ax.set_ylabel('Fraction of change response',fontsize=14)
#ax.legend(fontsize=12)
plt.tight_layout()


for state in behavStates:
    popAdaptMatrix = np.array(popSdfChange[state])-np.array(popSdfNonChange[state])
    fig = plt.figure(facecolor='w',figsize=(6,8))
    for i,(d,lbl) in enumerate(zip((popSdfChange,popSdfNonChange,popAdaptMatrix,adaptMatrix),('Response to change','Response to non-change','Population adaptation','Mean adaptation'))):
        ax = plt.subplot(4,1,i+1)
        if i<2:
            im = plt.imshow(np.array(d[state]),cmap='bwr',clim=(-1,1),aspect='auto')
        elif i==2:
            im = plt.imshow(d,cmap='magma',clim=(0,d.max()),aspect='auto')
        else:
            m = np.array(d[state])
            im = plt.imshow(m,cmap='magma',clim=(0,m.max()),aspect='auto')
        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
        ax.set_yticks(np.arange(len(regionLabels)))
        ax.set_yticklabels(regionLabels)
        ax.set_ylim([len(regionLabels)-0.5,-0.5])
        if i==3:
            ax.set_xlabel('Time since change (ms)',fontsize=10)
        else:
            ax.set_xticklabels([])
        lbl += ' (spikes/s)' if i<2 else ' ((change-nonChange)/max(change))'
        ax.set_title(lbl,fontsize=10)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(labelsize=8)
    plt.tight_layout()



###### decoding analysis
    
def crossValidate(model,X,y,nsplits=5):
    nclasses = len(set(y))
    nsamples = len(y)
    samplesPerSplit = round(nsamples/nsplits) if nsplits<nsamples else 1
    randInd = np.random.permutation(nsamples)
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nsplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nsamples,np.nan)
    cv['predict_proba'] = np.full((nsamples,nclasses),np.nan)
    cv['decision_function'] = np.full((nsamples,nclasses),np.nan)
    modelMethods = dir(model)
    for k,estimator in enumerate(cv['estimator']):
        i = k*samplesPerSplit
        testInd = randInd[i:i+samplesPerSplit] if k+1<nsplits else randInd[i:]
        trainInd = np.setdiff1d(randInd,testInd)
        estimator.fit(X[trainInd],y[trainInd])
        cv['train_score'].append(estimator.score(X[trainInd],y[trainInd]))
        cv['test_score'].append(estimator.score(X[testInd],y[testInd]))
        cv['predict'][testInd] = estimator.predict(X[testInd])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                m = getattr(estimator,method)(X[testInd])
                if method=='decision_function' and nclasses<3:
                    m = np.tile(m,(2,1)).T
                cv[method][testInd] = m
    return cv


exps = Aexps+Bexps

cortical_cmap = plt.cm.plasma
subcortical_cmap = plt.cm.Reds
regionsToUse = (('LGd',('LGd',),(0,0,0)),
                ('V1',('VISp',),cortical_cmap(0)),
                ('LM',('VISl',),cortical_cmap(0.1)),
                ('RL',('VISrl',),cortical_cmap(0.2)),
                ('AL',('VISal',),cortical_cmap(0.3)),
                ('PM',('VISpm',),cortical_cmap(0.4)),
                ('AM',('VISam',),cortical_cmap(0.5)),
                ('LP',('LP',),subcortical_cmap(0.4)),
#                ('APN',('APN',),subcortical_cmap(0.5)),
#                ('SCd',('SCig','SCig-a','SCig-b'),subcortical_cmap(0.6)),
#                ('MB',('MB',),subcortical_cmap(0.7)),
#                ('MRN',('MRN',),subcortical_cmap(0.8)),
#                ('SUB',('SUB','PRE','POST'),subcortical_cmap(0.9)),
#                ('hipp',('CA1','CA3','DG-mo','DG-po','DG-sg','HPF'),subcortical_cmap(1.0))
               )
regionsToUse = regionsToUse[:8]
regionLabels = [r[0] for r in regionsToUse]
regionColors = [r[2] for r in regionsToUse]
    
unitSampleSize = [20]

nCrossVal = 5

baseWin = slice(stimWin.start-150,stimWin.start)
respWin = slice(stimWin.start,stimWin.start+150)
respWinOffset = respWin.start-stimWin.start
respWinDur = respWin.stop-respWin.start

decodeWindowSize = 10
decodeWindows = [] #np.arange(stimWin.start,stimWin.start+150,decodeWindowSize)

preImageDecodeWindowSize = 50
preImageDecodeWindows = [] #np.arange(stimWin.start,stimWin.start+750,preImageDecodeWindowSize)

# models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4),LinearDiscriminantAnalysis()))
# modelNames = ('randomForest','SVM','LDA')
models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4))
modelNames = ('randomForest','SVM')

behavStates = ('active',)


result = {exp: {region: {state: {'changeScore':{model:[] for model in modelNames},
                                 'changeScoreTrain':{model:[] for model in modelNames},
                                 'changeScoreShuffle':{model:[] for model in modelNames},
                                 'changePredict':{model:[] for model in modelNames},
                                 'changePredictProb':{model:[] for model in modelNames},
                                 'changePredictShuffle':{model:[] for model in modelNames},
                                 'changePredictProbShuffle':{model:[] for model in modelNames},
                                 'changeFeatureImportance':{model:[] for model in modelNames},
                                 'changeFeatureImportanceShuffle':{model:[] for model in modelNames},
                                 'preChangePredict':{model:[] for model in modelNames},
                                 'preChangePredictProb':{model:[] for model in modelNames},
                                 'catchPredict':{model:[] for model in modelNames},
                                 'catchPredictProb':{model:[] for model in modelNames},
                                 'nonChangePredict':{model:[] for model in modelNames},
                                 'nonChangePredictProb':{model:[] for model in modelNames},
                                 'imageScore':{model:[] for model in modelNames},
                                 'imageFeatureImportance':{model:[] for model in modelNames},
                                 'changeScoreWindows':{model:[] for model in modelNames},
                                 'changePredictWindows':{model:[] for model in modelNames},
                                 'imageScoreWindows':{model:[] for model in modelNames},
                                 'preImageScoreWindows':{model:[] for model in modelNames},
                                 'preChangeSDFs':None,
                                 'changeSDFs':None,
                                 'catchSDFs':None,
                                 'nonChangeSDFs':None,
                                 'respLatency':None} for state in behavStates} for region in regionLabels} for exp in exps}

warnings.filterwarnings('ignore')
for expInd,exp in enumerate(exps):
    print('experiment '+str(expInd+1)+' of '+str(len(exps)))
    startTime = time.clock()
    
    response = data[exp]['response'][:]
    lickTimes = data[exp]['lickTimes'][:]
    hit = response=='hit'
    falseAlarm = response=='falseAlarm'
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    changeTrials = engaged & (hit | (response=='miss'))
    catchTrials = engaged & (falseAlarm | (response=='correctReject'))
    result[exp]['responseToChange'] = hit[changeTrials]
    result[exp]['responseToCatch'] = falseAlarm[catchTrials]
    result[exp]['changeReactionTime'] = data[exp]['rewardTimes'][changeTrials] - changeTimes[changeTrials]
    
    catchLickLat = np.full(catchTrials.sum(),np.nan)
    firstLickInd = np.searchsorted(lickTimes,changeTimes[engaged & falseAlarm])
    catchLickLat[falseAlarm[catchTrials]] = lickTimes[firstLickInd]-changeTimes[engaged & falseAlarm]
    result[exp]['catchReactionTime'] = catchLickLat
    
    initialImage = data[exp]['initialImage'][:]
    changeImage = data[exp]['changeImage'][:]
    imageNames = np.unique(changeImage)
    result[exp]['preChangeImage'] = initialImage[changeTrials]
    result[exp]['changeImage'] = changeImage[changeTrials]
    result[exp]['catchImage'] = changeImage[catchTrials]
    
    flashImage = data[exp]['flashImage'][:]
    flashTimes = data[exp]['behaviorFlashTimes'][:]
    nonChangeFlashIndex = []
    nonChangeImage = []
    respToNonChange = []
    nonChangeLickLat = []
    for i,(t,img) in enumerate(zip(flashTimes,flashImage)):
        timeFromChange = changeTimes-t
        timeFromLick = lickTimes-t
        if min(abs(timeFromChange))<60 and (not any(timeFromChange<0) or max(timeFromChange[timeFromChange<0])<-4) and (not any(timeFromLick<0) or max(timeFromLick[timeFromLick<0])<-4):
            nonChangeImage.append(img)
            nonChangeFlashIndex.append(i)
            if any((timeFromLick>0.15) & (timeFromLick<0.75)):
                respToNonChange.append(True)
                nonChangeLickLat.append(timeFromLick[timeFromLick>0.15].min())
            else:
                respToNonChange.append(False)
                nonChangeLickLat.append(np.nan)
    nonChangeImage = np.array(nonChangeImage)
    minImgCount = np.unique(nonChangeImage,return_counts=True)[1].min()
    nonChangeSample = [i for a in [np.random.choice(np.where(nonChangeImage==img)[0],minImgCount,replace=False) for img in imageNames] for i in a]
    nonChangeFlashIndex = np.array(nonChangeFlashIndex)[nonChangeSample]
    nonChangeFlashTimes = flashTimes[nonChangeFlashIndex]
    if 'passive' in behavStates:
        passiveNonChangeFlashTimes = data[exp]['passiveFlashTimes'][nonChangeFlashIndex]
    result[exp]['nonChangeImage'] = nonChangeImage[nonChangeSample]
    result[exp]['responseToNonChange'] = np.array(respToNonChange)[nonChangeSample]
    result[exp]['nonChangeReactionTime'] = np.array(nonChangeLickLat)[nonChangeSample]
   
    for region,regionCCFLabels,_ in regionsToUse:
        activePreSDFs = []
        activeChangeSDFs = []
        activeNonChangeSDFs = []
        passivePreSDFs = []
        passiveChangeSDFs = []
        passiveNonChangeSDFs = []
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,regionCCFLabels)
            if any(inRegion):
                activePre,activeChange = [data[exp]['sdfs'][probe]['active'][epoch][inRegion,:] for epoch in ('preChange','change')]
                hasSpikesActive,hasRespActive = findResponsiveUnits(activeChange[:,changeTrials],baseWin,respWin,thresh=5)
                if 'passive' in behavStates:
                    passivePre,passiveChange = [data[exp]['sdfs'][probe]['passive'][epoch][inRegion,:] for epoch in ('preChange','change')]
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChange[:,changeTrials],baseWin,respWin,thresh=5)
                    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
                    passivePreSDFs.append(passivePre[hasResp])
                    passiveChangeSDFs.append(passiveChange[hasResp])
                else:
                    hasResp = hasSpikesActive & hasRespActive
                if hasResp.any():
                    activePreSDFs.append(activePre[hasResp])
                    activeChangeSDFs.append(activeChange[hasResp])
                    units = data[exp]['units'][probe][inRegion][hasResp]
                    spikes = data[exp]['spikeTimes'][probe]
                    activeNonChangeSDFs.append([analysis_utils.getSDF(spikes[u],nonChangeFlashTimes+respWinOffset,respWinDur*0.001,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0] for u in units])
                    if 'passive' in behavStates:
                        passiveNonChangeSDFs.append([analysis_utils.getSDF(spikes[u],passiveNonChangeFlashTimes+respWinOffset,respWinDur*0.001,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0] for u in units])
        if len(activePreSDFs)>0:
            activePreSDFs = np.concatenate(activePreSDFs)
            activeChangeSDFs = np.concatenate(activeChangeSDFs)
            activeNonChangeSDFs = np.concatenate(activeNonChangeSDFs)
            if 'passive' in behavStates:
                passivePreSDFs = np.concatenate(passivePreSDFs)
                passiveChangeSDFs = np.concatenate(passiveChangeSDFs)
                passiveNonChangeSDFs = np.concatenate(passiveNonChangeSDFs)
            nUnits = len(activePreSDFs)
            for sampleSize in unitSampleSize:
                if nUnits>=sampleSize:
                    if sampleSize>1:
                        if sampleSize==nUnits:
                            nsamples = 1
                            unitSamples = [np.arange(nUnits)]
                        else:
                            # >99% chance each neuron is chosen at least once
                            nsamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                            unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nsamples)]
                    else:
                        nsamples = nUnits
                        unitSamples = [[_] for _ in range(nsamples)]
                    print(nsamples)
                    for state in behavStates:
                        sdfs = (activePreSDFs,activeChangeSDFs,activeNonChangeSDFs) if state=='active' else (passivePreSDFs,passiveChangeSDFs,passiveNonChangeSDFs)
                        preChangeSDFs,changeSDFs,nonChangeSDFs = [s.transpose((1,0,2)) for s in sdfs]
                        if sampleSize==unitSampleSize[0]:
                            meanSDF = activeChangeSDFs.mean(axis=1)
                            result[exp][region][state]['preChangeSDFs'] = preChangeSDFs[changeTrials][:,:,respWin].mean(axis=1) 
                            result[exp][region][state]['changeSDFs'] = changeSDFs[changeTrials][:,:,respWin].mean(axis=1)
                            result[exp][region][state]['catchSDFs'] = changeSDFs[catchTrials][:,:,respWin].mean(axis=1)
                            result[exp][region][state]['nonChangeSDFs'] = nonChangeSDFs.mean(axis=1)
                            result[exp][region][state]['respLatency'] = findLatency(changeSDFs[changeTrials][:,:,respWin].mean(axis=(0,1))[None,:],baseWin,stimWin,method='abs',thresh=0.5)[0]
                        
                        changeScore = {model: [] for model in modelNames}
                        changeScoreTrain = {model: [] for model in modelNames}
                        changeScoreShuffle = {model: [] for model in modelNames}
                        changePredict = {model: [] for model in modelNames}
                        changePredictProb = {model: [] for model in modelNames}
                        changePredictShuffle = {model: [] for model in modelNames}
                        changePredictProbShuffle = {model: [] for model in modelNames}
                        changeFeatureImportance = {model: np.full((nsamples,nUnits,respWinDur),np.nan) for model in modelNames}
                        changeFeatureImportanceShuffle = {model: np.full((nsamples,nUnits,respWinDur),np.nan) for model in modelNames}
                        preChangePredict = {model: [] for model in modelNames}
                        preChangePredictProb = {model: [] for model in modelNames}
                        catchPredict = {model: [] for model in modelNames}
                        catchPredictProb = {model: [] for model in modelNames}
                        nonChangePredict = {model: [] for model in modelNames}
                        nonChangePredictProb = {model: [] for model in modelNames}
                        imageScore = {model: [] for model in modelNames}
                        imageFeatureImportance = {model: np.full((nsamples,nUnits,respWinDur),np.nan) for model in modelNames}
                        changeScoreWindows = {model: np.full((nsamples,len(decodeWindows)),np.nan) for model in modelNames}
                        changePredictWindows = {model: np.full((nsamples,len(decodeWindows),changeTrials.sum()),np.nan) for model in modelNames}
                        imageScoreWindows = {model: np.full((nsamples,len(decodeWindows)),np.nan) for model in modelNames}
                        preImageScoreWindows = {model: np.full((nsamples,len(preImageDecodeWindows)),np.nan) for model in modelNames}
                        
                        for i,unitSamp in enumerate(unitSamples):
                            # decode image change and identity for full respWin
                            # image change
                            X = np.concatenate([s[:,unitSamp,respWin][changeTrials].reshape((changeTrials.sum(),-1)) for s in (changeSDFs,preChangeSDFs)])
                            y = np.zeros(X.shape[0])
                            y[:changeTrials.sum()] = 1
                            Xshuffle = X.copy()
                            for u in range(len(unitSamp)):
                                uslice = slice(u*respWinDur,u*respWinDur+respWinDur)
                                for img in imageNames:
                                    np.random.shuffle(Xshuffle[:changeTrials.sum()][changeImage[changeTrials]==img,uslice])
                                    np.random.shuffle(Xshuffle[changeTrials.sum():][initialImage[changeTrials]==img,uslice])
                            Xcatch = changeSDFs[:,unitSamp,respWin][catchTrials].reshape((catchTrials.sum(),-1))
                            Xnonchange = nonChangeSDFs[:,unitSamp].reshape((nonChangeSDFs.shape[0],-1))
                            for model,name in zip(models,modelNames):
                                if name=='randomForest':
                                    probMethod,featureMethod = 'predict_proba','feature_importances_'
                                elif name=='SVM':
                                    probMethod,featureMethod = 'decision_function','coef_'
                                cv = crossValidate(model,X,y,nsplits=nCrossVal)
                                cvShuffle = crossValidate(model,Xshuffle,y,nsplits=nCrossVal)
                                changeScore[name].append(np.mean(cv['test_score']))
                                changeScoreTrain[name].append(np.mean(cv['train_score']))
                                changeScoreShuffle[name].append(np.mean(cvShuffle['test_score']))
                                changePredict[name].append(cv['predict'][:changeTrials.sum()])
                                changePredictShuffle[name].append(cvShuffle['predict'][:changeTrials.sum()])
                                changePredictProb[name].append(cv[probMethod][:changeTrials.sum(),1])
                                changePredictProbShuffle[name].append(cvShuffle[probMethod][:changeTrials.sum(),1])
                                changeFeatureImportance[name][i][unitSamp] = np.mean([np.reshape(np.absolute(getattr(estimator,featureMethod)),(sampleSize,-1)) for estimator in cv['estimator']],axis=0)
                                changeFeatureImportanceShuffle[name][i][unitSamp] = np.mean([np.reshape(np.absolute(getattr(estimator,featureMethod)),(sampleSize,-1)) for estimator in cvShuffle['estimator']],axis=0)
                                preChangePredict[name].append(cv['predict'][changeTrials.sum():])
                                preChangePredictProb[name].append(cv[probMethod][changeTrials.sum():,1])
                                catchPredict[name].append(scipy.stats.mode([estimator.predict(Xcatch) for estimator in cv['estimator']],axis=0)[0].flatten())
                                nonChangePredict[name].append(scipy.stats.mode([estimator.predict(Xnonchange) for estimator in cv['estimator']],axis=0)[0].flatten())
                                if probMethod=='decision_function':
                                    catchPredictProb[name].append(np.mean([getattr(estimator,probMethod)(Xcatch) for estimator in cv['estimator']],axis=0))
                                    nonChangePredictProb[name].append(np.mean([getattr(estimator,probMethod)(Xnonchange) for estimator in cv['estimator']],axis=0))
                                else:
                                    catchPredictProb[name].append(np.mean([getattr(estimator,probMethod)(Xcatch)[:,1] for estimator in cv['estimator']],axis=0))
                                    nonChangePredictProb[name].append(np.mean([getattr(estimator,probMethod)(Xnonchange)[:,1] for estimator in cv['estimator']],axis=0))

#                            # image identity
#                            imgSDFs = [changeSDFs[:,unitSamp,respWin][changeTrials & (changeImage==img)] for img in imageNames]
#                            X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
#                            y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
#                            for model,name in zip(models,modelNames):
#                                cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
#                                imageScore[name].append(cv['test_score'].mean())
#                                if name=='randomForest':
#                                    imageFeatureImportance[name][i][unitSamp] = np.mean([np.reshape(estimator.feature_importances_,(sampleSize,-1)) for estimator in cv['estimator']],axis=0)
                            
                            # decode image change and identity for sliding windows
                            for j,winStart in enumerate(decodeWindows):
                                # image change
                                winSlice = slice(winStart,winStart+decodeWindowSize)
                                X = np.concatenate([s[:,unitSamp,winSlice][changeTrials].reshape((changeTrials.sum(),-1)) for s in (changeSDFs,preChangeSDFs)])
                                y = np.zeros(X.shape[0])
                                y[:int(X.shape[0]/2)] = 1
                                for model,name in zip(models,modelNames):
                                    changeScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                    if name=='randomForest':
                                        changePredictWindows[name][i,j] = cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:changeTrials.sum(),1]
                                # image identity
                                imgSDFs = [changeSDFs[:,unitSamp,winSlice][changeTrials & (changeImage==img)] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                for model,name in zip(models,modelNames):
                                    imageScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                    
                            # decode pre-change image identity for sliding windows
                            for j,winStart in enumerate(preImageDecodeWindows):
                                winSlice = slice(winStart,winStart+preImageDecodeWindowSize)
                                preImgSDFs = [preChangeSDFs[:,unitSamp,winSlice][changeTrials & (initialImage==img)] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in preImgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(preImgSDFs)])
                                for model,name in zip(models,modelNames):
                                    preImageScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                        
                        # average across unit samples
                        for model in modelNames:
                            result[exp][region][state]['changeScore'][model].append(np.median(changeScore[model],axis=0))
                            result[exp][region][state]['changeScoreTrain'][model].append(np.median(changeScoreTrain[model],axis=0))
                            result[exp][region][state]['changeScoreShuffle'][model].append(np.median(changeScoreShuffle[model],axis=0))
                            result[exp][region][state]['changePredict'][model].append(scipy.stats.mode(changePredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['changePredictProb'][model].append(np.median(changePredictProb[model],axis=0))
                            result[exp][region][state]['changePredictShuffle'][model].append(scipy.stats.mode(changePredictShuffle[model],axis=0)[0].flatten())
                            result[exp][region][state]['changePredictProbShuffle'][model].append(np.median(changePredictProbShuffle[model],axis=0))
                            result[exp][region][state]['changeFeatureImportance'][model].append(np.nanmedian(changeFeatureImportance[model],axis=0))
                            result[exp][region][state]['changeFeatureImportanceShuffle'][model].append(np.nanmedian(changeFeatureImportanceShuffle[model],axis=0))
                            result[exp][region][state]['preChangePredict'][model].append(scipy.stats.mode(preChangePredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['preChangePredictProb'][model].append(np.median(preChangePredictProb[model],axis=0))
                            result[exp][region][state]['catchPredict'][model].append(scipy.stats.mode(catchPredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['catchPredictProb'][model].append(np.median(catchPredictProb[model],axis=0))
                            result[exp][region][state]['nonChangePredict'][model].append(scipy.stats.mode(nonChangePredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['nonChangePredictProb'][model].append(np.median(nonChangePredictProb[model],axis=0))
                            result[exp][region][state]['imageScore'][model].append(np.median(imageScore[model],axis=0))
                            result[exp][region][state]['imageFeatureImportance'][model].append(np.nanmedian(imageFeatureImportance[model],axis=0))
                            result[exp][region][state]['changeScoreWindows'][model].append(np.median(changeScoreWindows[model],axis=0))
                            result[exp][region][state]['changePredictWindows'][model].append(np.median(changePredictWindows[model],axis=0))
                            result[exp][region][state]['imageScoreWindows'][model].append(np.median(imageScoreWindows[model],axis=0))
                            result[exp][region][state]['preImageScoreWindows'][model].append(np.median(preImageScoreWindows[model],axis=0))
    print(time.clock()-startTime)
warnings.filterwarnings('default')


# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(result,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
result = pickle.load(open(pkl,'rb'))


# plot scores vs number of units
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    allScores = {score: [] for score in ('changeScore','imageScore')}
    for i,region in enumerate(regionLabels):
        for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.5,0.125))):
            ax = plt.subplot(gs[i,j])
            expScores = []
            for exp in result:
                s = result[exp][region]['active'][score][model]
                if len(s)>0:
                    s = s + [np.nan]*(len(unitSampleSize)-len(s))
                    expScores.append(s)
                    allScores[score].append(s)
                    ax.plot(unitSampleSize,s,'k')
            if len(expScores)>0:
                ax.plot(unitSampleSize,np.nanmean(expScores,axis=0),'r',linewidth=2)
            for side in ('right','top'):
                    ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(np.arange(0,100,10))
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,max(unitSampleSize)+5])
            ax.set_ylim([ymin,1])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])  
            if i==0:
                if j==0:
                    ax.set_title(region+', '+score[:score.find('S')])
                else:
                    ax.set_title(score[:score.find('S')])
            elif j==0:
                ax.set_title(region)
            if i==0 and j==0:
                ax.set_ylabel('Decoder Accuracy')
    ax.set_xlabel('Number of Units')
    plt.tight_layout()
    
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    for score,clr in zip(('changeScore','imageScore'),('k','0.5')):
        allMean = np.nanmean(allScores[score],axis=0)
        allSem = np.nanstd(allScores[score],axis=0)/(np.sum(~np.isnan(allScores[score]),axis=0)**0.5)
        ax.plot(unitSampleSize,allMean,'o-',color=clr,label=score[:score.find('S')])
        for x,m,s in zip(unitSampleSize,allMean,allSem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(np.arange(0,100,10))
#    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_xlim([0,max(unitSampleSize)+5])
    ax.set_ylim([0.5,1])
    ax.set_xlabel('Number of Cells',fontsize=14)
    ax.set_ylabel('Decoder Accuracy',fontsize=14)
#    ax.set_title(model)
#    ax.legend()
    plt.tight_layout()
    
    xticks = np.arange(len(regionLabels))
    xlim = [-0.5,len(regionLabels)-0.5]
    for score,ymin in zip(('changeScore','imageScore'),(0.5,0.125)):
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        for i,(n,clr) in enumerate(zip(unitSampleSize,plt.cm.plasma(np.linspace(0,1,len(unitSampleSize))))):
            for j,region in enumerate(regionLabels):
                regionData = []
                for exp in result:
                    s = result[exp][region]['active'][score][model]
                    if len(s)>i:
                        regionData.append(s[i])
                if len(regionData)>0:
                    m = np.mean(regionData)
                    s = np.std(regionData)/(len(regionData)**0.5)
                    lbl = str(n)+' cells' if j==0 else None
                    ax.plot(j,m,'o',mec=clr,mfc='none',label=lbl)
                    ax.plot([j,j],[m-s,m+s],color=clr)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(regionLabels)
        ax.set_ylim([ymin,1])
        ax.set_ylabel('Decoder Accuracy',fontsize=14)
#        ax.set_title(model+', '+score[:score.find('S')])
        ax.legend(loc='upper left',fontsize=12)
        plt.tight_layout()
        
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(xlim,[0,0],'--',color='0.5')
for i,(n,clr) in enumerate(zip(unitSampleSize,plt.cm.plasma(np.linspace(0,1,len(unitSampleSize))))):
    for j,region in enumerate(regionLabels):
        regionData = []
        for exp in result:
            behavior = result[exp]['responseToChange'][:].astype(float)
            s = result[exp][region]['active']['changePredictProb']['randomForest']
            if len(s)>i:
                regionData.append(np.corrcoef(behavior,s[i])[0,1])
        if len(regionData)>0:
            m = np.mean(regionData)
            s = np.std(regionData)/(len(regionData)**0.5)
            lbl = str(n)+' cells' if j==0 else None
            ax.plot(j,m,'o',mec=clr,mfc='none',label=lbl)
            ax.plot([j,j],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels)
ax.set_xlim(xlim)
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylim([-0.075,0.8])
ax.set_ylabel('Pearson r',fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(xlim,[0,0],'--',color='0.5')
for i,(n,clr) in enumerate(zip(unitSampleSize,plt.cm.plasma(np.linspace(0,1,len(unitSampleSize))))):
    for j,region in enumerate(regionLabels):
        regionData = []
        for exp in result:
            behavior = result[exp]['responseToChange'][:]
            s = result[exp][region]['active']['changePredictProb']['randomForest']
            if len(s)>i:
                regionData.append(np.mean(s[i][behavior])-np.mean(s[i][~behavior]))
        if len(regionData)>0:
            m = np.mean(regionData)
            s = np.std(regionData)/(len(regionData)**0.5)
            lbl = str(n)+' cells' if j==0 else None
            ax.plot(j,m,'o',mec=clr,mfc='none',label=lbl)
            ax.plot([j,j],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels)
ax.set_xlim(xlim)
#ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_yticks([0,0.05,0.1,0.15])
ax.set_ylim([-0.015,0.16])
ax.set_ylabel('$\Delta$ Change probability (hit - miss)',fontsize=14)
ax.legend(loc='upper left',fontsize=12)
plt.tight_layout()

    
# compare scores for full response window
for model in modelNames:
    for score,ymin in zip(('changeScore','imageScore'),(0.5,0.125)):
        fig = plt.figure(facecolor='w')
        xticks = np.arange(len(regionLabels))
        xlim = [xticks[0]-0.5,xticks[-1]+0.5]
        ax = fig.subplots(1)
        for state in behavStates:
            mean = np.full(len(regionLabels),np.nan)
            sem = mean.copy()
            for i,region in enumerate(regionLabels):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[-1])
                n = len(regionScore)
                if n>0:
                    mean[i] = np.mean(regionScore)
                    sem[i] = np.std(regionScore)/(n**0.5)
            for i,(x,m,s,clr) in enumerate(zip(xticks,mean,sem,regionColors)):
                mfc = clr if state=='active' else 'none'
                lbl = state if i==0 else None
                ax.plot(x,m,'o',ms=10,mec=clr,mfc=mfc,label=lbl)
                ax.plot([x,x],[m-s,m+s],color=clr)           
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        ax.set_xticklabels(regionLabels)
        ax.set_xlim(xlim)
        ax.set_yticks([ymin,0.25,0.5,0.75,1])
        ax.set_ylim([ymin,1])
        ax.set_ylabel('Decoder Accuracy')
        ax.set_title(score)
        ax.legend()
        plt.tight_layout()

    
# image and change feature importance
x = np.arange(0,respWin.stop-respWin.start)
fig = plt.figure(facecolor='w',figsize=(10,12))
gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
for i,region in enumerate(regionLabels):
    for j,model in enumerate(modelNames):
        ax = plt.subplot(gs[i,j])
        regionScore = []
        for exp in result:
            s = result[exp][region]['active']['changeFeatureImportanceShuffle'][model]
            if len(s)>0:
                regionScore.append(np.nanmean(s[0],axis=0))
                print(s[0].shape)
        n = len(regionScore)
        if n>0:
            m = np.mean(regionScore,axis=0)
            s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
            ax.plot(x,m,color='k')
            ax.fill_between(x,m+s,m-s,color='k',alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_xlim(x[[0,-1]])
        if i<len(regionLabels)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (ms)')
        if i==0:
            if j==0:
                ax.set_title(region+', '+model)
                ax.set_ylabel('Feature Importance')
            else:
                ax.set_title(model)
        elif j==0:
            ax.set_title(region) 
plt.tight_layout()    


for model in ('randomForest',):
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active',)):
            fig = plt.figure(facecolor='w')
            ax = fig.subplots(1)
            imageFeatureScore = []
            changeFeatureScore = []
            for key,featureScore in zip(('imageFeatureImportance','changeFeatureImportance'),(imageFeatureScore,changeFeatureScore)):
                for exp in result:
                    s = result[exp][region][state][key][model]
                    if len(s)>0:
                        featureScore.append(np.nanmax(s[0],axis=1))
            if len(changeFeatureScore)>0:
                ax.plot(np.concatenate(imageFeatureScore),np.concatenate(changeFeatureScore),'o',mec='k',mfc='none')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlabel('Image feature importance')   
            ax.set_ylabel('Change feature importance')
            ax.set_title(region+', '+state)

    
# image identity and change decoding for sliding windows         
for model in ('randomForest',):
    for state in ('active','passive'):
        fig = plt.figure(facecolor='w',figsize=(6,10))
        fig.text(0.55,1,'Decoder Accuracy'+' ('+state+')',fontsize=10,horizontalalignment='center',verticalalignment='top')
        gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
        for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
            for j,score in enumerate(('imageScoreWindows','changeScoreWindows')):
                ax = plt.subplot(gs[i,j])
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[-1])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(decodeWindows+decodeWindowSize/2,m,color=clr,label=score[:score.find('S')])
                    ax.fill_between(decodeWindows+decodeWindowSize/2,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=8)
                if score=='changeScoreWindows':
                    yticks = [0.5,0.75]
                    ylim = [0.475,0.9]
                    title = 'Change'
                else:
                    yticks = [0.1,0.75]
                    ylim = [0.1,0.9]
                    title = 'Image Identity'
                ax.set_xticks([250,350,450,550,650,750])
                ax.set_xticklabels([0,100,200,300,400,500])
                ax.set_xlim([decodeWindows[0],decodeWindows[-1]+decodeWindowSize])
                ax.set_yticks(yticks)
                ax.set_ylim(ylim)
                if i<len(regionLabels)-1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (ms)',fontsize=10)
                if j==0:
                    ax.set_ylabel(region,fontsize=10)
                if i==0:
                    ax.set_title(title,fontsize=10)
        plt.tight_layout()


# overlay of change and image decoding
for model in ('randomForest',):
    fig = plt.figure(facecolor='w',figsize=(4,10))
    fig.text(0.5,1,'Decoder Accuracy',fontsize=10,horizontalalignment='center',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),1)
    for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
        ax1 = plt.subplot(gs[i,0])
        ax2 = ax1.twinx()
        for j,(score,ax) in enumerate(zip(('changeScoreWindows','imageScoreWindows'),(ax1,ax2))):
            if score=='changeScoreWindows':
                lineStyle = '-'
                yticks = [0.5,0.75]
                ylim = [yticks[0]-0.05*(yticks[1]-yticks[0]),0.75]
                lbl = 'Change'
            else:
                lineStyle = '--'
                yticks = [0.125,0.75]
                ylim = [yticks[0]-0.05*(yticks[1]-yticks[0]),0.75]
                lbl = 'Image Identity'
            regionScore = []
            for exp in result:
                s = result[exp][region]['active'][score][model]
                if len(s)>0:
                    regionScore.append(s[0])
            n = len(regionScore)
            if n>0:
                m = np.mean(regionScore,axis=0)
                s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                ax.plot(decodeWindows+decodeWindowSize/2,m,lineStyle,color=clr,label=lbl)
                ax.fill_between(decodeWindows+decodeWindowSize/2,m+s,m-s,color=clr,alpha=0.25)
            for side in ('top',):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,labelsize=8)
            ax.set_xticks([250,300,350,400])
            ax.set_xticklabels([0,50,100,150])
            ax.set_xlim([250,400])
            ax.set_yticks(yticks)
            ax.set_ylim(ylim)
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (ms)',fontsize=10)
#                ax.set_ylabel(lbl,fontsize=10)
            ax.set_title(region,fontsize=10)
    plt.tight_layout()


# pre-change image decoding (sliding windows) 
for model in ('randomForest',):
    fig = plt.figure(facecolor='w',figsize=(4,10))
    fig.text(0.55,1,'Decoder Accuracy, Pre-change Image Identity',fontsize=10,horizontalalignment='center',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),1)
    for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
        clr = 'k'
        ax = plt.subplot(gs[i,0])
        ax.plot([preImageDecodeWindows[0],preImageDecodeWindows[-1]+preImageDecodeWindowSize],[0.125,0.125],'--',color='0.5')
        regionScore = []
        for exp in result:
            s = result[exp][region]['active']['preImageScoreWindows'][model]
            if len(s)>0:
                regionScore.append(s[0])
        n = len(regionScore)
        if n>0:
            m = np.mean(regionScore,axis=0)
            s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
            ax.plot(preImageDecodeWindows+preImageDecodeWindowSize/2,m,color=clr,label=score[:score.find('S')])
            ax.fill_between(preImageDecodeWindows+preImageDecodeWindowSize/2,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
        ax.set_xticks([250,500,750,1000])
        ax.set_xticklabels([0,250,500,750])
        ax.set_xlim([preImageDecodeWindows[0],preImageDecodeWindows[-1]+preImageDecodeWindowSize])
        ax.set_yticks([0.1,0.75])
        ax.set_ylim([0.1,0.775])
        if i<len(regionLabels)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time from pre-change flash onset (ms)',fontsize=10)
        ax.set_ylabel(region,fontsize=10)
    plt.tight_layout()
    
fig = plt.figure(facecolor='w')
region = 'V1'
clr = 'k'
ax = plt.subplot(1,1,1)
ax.plot([preImageDecodeWindows[0],preImageDecodeWindows[-1]+preImageDecodeWindowSize],[0.125,0.125],'--',color='0.5')
regionScore = []
for exp in result:
    s = result[exp][region]['active']['preImageScoreWindows'][model]
    if len(s)>0:
        regionScore.append(s[0])
n = len(regionScore)
if n>0:
    m = np.mean(regionScore,axis=0)
    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
    ax.plot(preImageDecodeWindows+preImageDecodeWindowSize/2,m,color=clr,label=score[:score.find('S')])
    ax.fill_between(preImageDecodeWindows+preImageDecodeWindowSize/2,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks([250,500,750,1000])
ax.set_xticklabels([0,250,500,750])
ax.set_xlim([preImageDecodeWindows[0],preImageDecodeWindows[-1]+preImageDecodeWindowSize])
ax.set_ylim([0.1,0.8])
ax.set_xlabel('Time from pre-change flash onset (ms)')
ax.set_ylabel('Decoder accuracy')
plt.tight_layout()


# plot visual response, change decoding, and image decoding latencies
latencyLabels = {'resp':'Visual Response Latency','change':'Change Decoding Latency','image':'Image Decoding Latency'}

for model in ('randomForest',):#modelNames:
    latency = {exp: {region: {state: {} for state in ('active','passive')} for region in regionLabels[1:-1]} for exp in result}
    for exp in result:
        for region in regionLabels[1:-1]:
            for state in ('active','passive'):
                s = result[exp][region][state]['respLatency']
                if len(s)>0:
                    latency[exp][region][state]['resp'] = s[0]
                for score,decodeThresh in zip(('changeScoreWindows','imageScoreWindows'),(0.5+0.1*(1-0.5),0.125+0.1*(1-0.125))):
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        intpScore = np.interp(np.arange(decodeWindows[0],decodeWindows[-1]+1)+decodeWindowSize/2,decodeWindows+decodeWindowSize/2,s[0])
                        latency[exp][region][state][score[:score.find('S')]] = findLatency(intpScore,method='abs',thresh=decodeThresh)[0]
    
    fig = plt.figure(facecolor='w',figsize=(10,10))
    gs = matplotlib.gridspec.GridSpec(3,2)
    axes = []
    latMin = 1000
    latMax = 0
    for i,(xkey,ykey) in enumerate((('resp','change'),('resp','image'),('image','change'))):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            axes.append(ax)
            ax.plot([0,1000],[0,1000],'--',color='0.5')
            for region,clr in zip(regionLabels[1:-1],regionColors[1:-1]):
                x,y = [[latency[exp][region][state][key] for exp in latency if key in latency[exp][region][state]] for key in (xkey,ykey)]
                mx,my = [np.nanmean(d) for d in (x,y)]
                latMin = min(latMin,np.nanmin(mx),np.nanmin(my))
                latMax = max(latMax,np.nanmax(mx),np.nanmax(my))
                sx,sy = [np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in (x,y)]
                ax.plot(mx,my,'o',mec=clr,mfc=clr)
                ax.plot([mx,mx],[my-sy,my+sy],color=clr)
                ax.plot([mx-sx,mx+sx],[my,my],color=clr)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlabel(latencyLabels[xkey]+' (ms)')
            ax.set_ylabel(latencyLabels[ykey])
            if i==0:
                ax.set_title(state)
    alim = [latMin-5,latMax+5]
    for ax in axes:
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
    plt.tight_layout()
    
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(3,2)
    for i,key in enumerate(('resp','image','change')):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            d = np.full((len(latency),len(regionLabels)),np.nan)
            for expInd,exp in enumerate(latency):
                z = [(r,latency[exp][region][state][key]) for r,region in enumerate(regionLabels) if key in latency[exp][region][state]]
                if len(z)>0:
                    x,y = zip(*z)
                    ax.plot(x,y,'k')
                    d[expInd,list(x)] = y
            plt.plot(np.nanmean(d,axis=0),'r',linewidth=2)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_ylim(alim)
            ax.set_xticks(np.arange(len(regionLabels)))
            if i==len(latencyLabels)-1:
                ax.set_xticklabels(regionLabels)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(latencyLabels[key])
            if i==0:
                ax.set_title(state)


# plot correlation of model prediction and mouse behavior
d = []
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(6,4))
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(regionLabels))
    xlim = [-0.5,len(regionLabels)-0.5]
    ax.plot(xlim,[0,0],'--',color='0.5')
    for score,mrk,fill,lbl in zip(('changeScore','changeScoreShuffle'),('o','d'),(False,False),('non-shuffled','shuffled')):
        for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
            regionData = []
            for exp in result:
                s = result[exp][region]['active'][score][model]
                if len(s)>0:
                    regionData.append(s[-1])
            n = len(regionData)
            if n>0:
                m = np.mean(regionData)
                s = np.std(regionData)/(n**0.5)
                mfc = clr if fill else 'none'
                lbl = lbl if i==0 else None
                ax.plot(i,m,mrk,mec=clr,mfc=mfc,label=lbl)
                ax.plot([i,i],[m-s,m+s],color=clr)
                d.append(m)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0.5,1])
    ax.set_ylabel('Decoder accuracy')
    ax.legend(loc='upper left')
    ax.set_title(model)
    plt.tight_layout()

# plot individual experiments
fig = plt.figure(facecolor='w',figsize=(6,4))
ax = plt.subplot(1,1,1)
xticks = np.arange(len(regionLabels))
xlim = [-0.5,len(regionLabels)-0.5]
ax.plot(xlim,[0,0],'--',color='0.5')
allRegionData = []
allRegionExps = []
for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
    regionData = []
    regionExps = []
    for exp in result:
        behavior = result[exp]['responseToChange']
        s = result[exp][region]['active']['changePredictProb']['randomForest']
        if len(s)>0 and any(behavior) and any(s[-1]):
            regionData.append(np.corrcoef(behavior,s[-1])[0,1])
            regionExps.append(exp)
    n = len(regionData)
    if n>0:
        ax.plot(i+np.zeros(n),regionData,'o',mec='k',mfc='none')
    allRegionData.append(regionData)
    allRegionExps.append(regionExps)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Correlation of decoder prediction and behavior')
plt.tight_layout()


for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(6,4))
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(regionLabels))
    xlim = [-0.5,len(regionLabels)-0.5]
    ax.plot(xlim,[0,0],'--',color='0.5')
    for score,shuffleBehav,mrk,fill,lbl in zip(('changePredictProb','changePredictProb','changePredictProbShuffle'),(False,True,False),('o','s','d'),(True,False,False),('','shuffle behavioral response','shuffle neural response')):
        for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
            regionData = []
            for exp in result:
                behavior = result[exp]['responseToChange']
                s = result[exp][region]['active'][score][model]
                if len(s)>0 and any(behavior) and any(s[-1]):
                    if shuffleBehav:
                        regionData.append(np.mean([np.corrcoef(np.random.permutation(behavior),s[-1])[0,1] for _ in range(10)]))
                    else:
                        regionData.append(np.corrcoef(behavior,s[-1])[0,1])
            n = len(regionData)
            if n>0:
                m = np.mean(regionData)
                s = np.std(regionData)/(n**0.5)
                mfc = clr if fill else 'none'
                lbl = lbl if i==0 else None
                ax.plot(i,m,mrk,mec=clr,mfc=mfc,label=lbl)
                ax.plot([i,i],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.legend()
    ax.set_title(model)
    plt.tight_layout()


for model in ('randomForest',):    
    fig = plt.figure(facecolor='w',figsize=(6,10))
    fig.text(0.5,1,'Correlation of decoder prediction and mouse behavior',fontsize=10,horizontalalignment='center',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
        for j,state in enumerate(behavStates):
            ax = plt.subplot(gs[i,j])
            regionData = []
            for exp in result:
                behavior = result[exp]['responseToChange'][:]
                s = result[exp][region][state]['changePredictWindows'][model]
                if len(s)>2:
                    regionData.append([np.corrcoef(behavior,p)[0,1] for p in s[-1]])
            n = len(regionData)
            if n>0:
                m = np.mean(regionData,axis=0)
                s = np.std(regionData,axis=0)/(n**0.5)
                ax.plot(decodeWindows+decodeWindowSize/2,m,color=clr)
                ax.fill_between(decodeWindows+decodeWindowSize/2,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            ax.set_xticks([250,350,450,550,650,750])
            ax.set_xticklabels([0,100,200,300,400,500])
            ax.set_xlim([decodeWindows[0],decodeWindows[-1]+decodeWindowSize])
            ax.set_ylim([-0.1,0.75])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (ms)',fontsize=10)
            if j==0:
                ax.set_ylabel(region,fontsize=10)
            if i==0:
                ax.set_title(state,fontsize=10)
    plt.tight_layout()


anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionsToUse for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a==r[1][0]] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')] 
hier = hierScore_8regions

hierColors = np.array([[217,141,194], # LGd
                       [129,116,177], # V1
                       [78,115,174], # LM
                       [101,178,201], # RL
                       [202,183,120], # AL
                       [219,132,87], # PM
                       [194,79,84], # AM
                       [88,167,106]], #LP
                     ).astype(float)
hierColors /= 255

for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(5,5))
    ax = plt.subplot(1,1,1)
    title = model
    for state,fill in zip(('active',),(True,)):
        meanRegionData = []
        regionN = []
        for i,(region,h,clr) in enumerate(zip(regionLabels,hier,hierColors)):
            regionData = []
            for exp in result:
                behavior = result[exp]['responseToChange']
                s = result[exp][region][state]['changePredictProb'][model]
                if len(s)>0 and any(behavior) and any(s[0]):
                    regionData.append(np.corrcoef(behavior,s[0])[0,1])
            n = len(regionData)
            if n>0:
                m = np.mean(regionData)
                s = np.std(regionData)/(n**0.5)
                ax.plot(h,m,'o',mec=clr,mfc=clr)
                ax.plot([h,h],[m-s,m+s],color=clr)
                meanRegionData.append(m)
                regionN.append(n)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,meanRegionData)
        x = np.array([min(hier),max(hier)])
        ax.plot(x,slope*x+yint,'--',color='0.5')
        r,p = scipy.stats.pearsonr(hier,meanRegionData)
        title += '\nPearson: r = '+str(round(r,2))+', p = '+str(round(p,3))
        r,p = scipy.stats.spearmanr(hier,meanRegionData)
        title += '\nSpearman: r = '+str(round(r,2))+', p = '+str(round(p,3))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Hierarchy score')
    ax.set_ylabel('Correlation of decoder confidence and mouse behavior')
    ax.set_title(title,fontsize=8)
    plt.tight_layout()

fig = plt.figure(facecolor='w',figsize=(4,4))
ax = plt.subplot(1,1,1)
for state,fill in zip(('active',),(True,)):
    meanRegionData = []
    for i,(region,clr,h) in enumerate(zip(regionLabels,regionColors,hier)):
        regionData = []
        for exp in result:
            s = result[exp][region][state]['changeScore']['randomForest']
            if len(s)>0:
                regionData.append(s[0])
        n = len(regionData)
        if n>0:
            m = np.mean(regionData)
            s = np.std(regionData)/(n**0.5)
            mfc = clr if fill else 'none'
            lbl = state if i==0 else None
            ax.plot(h,m,'o',mec=clr,mfc=mfc,label=lbl)
            ax.plot([h,h],[m-s,m+s],color=clr)
            meanRegionData.append(m)
        else:
            meanRegionData.append(np.nan)
    slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,meanRegionData)
    x = np.array([min(hier),max(hier)])
    ax.plot(x,slope*x+yint,'--',color='0.5')
    r,p = scipy.stats.pearsonr(hier,meanRegionData)
    title = 'Pearson: r = '+str(round(r,2))+', p = '+str(round(p,3))
    r,p = scipy.stats.spearmanr(hier,meanRegionData)
    title += '\nSpearman: r = '+str(round(r,2))+', p = '+str(round(p,3))
    ax.set_title(title,fontsize=8)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xlabel('Hierarchy score')
ax.set_ylabel('Decoder accuracy')
plt.tight_layout()




## analysis of population sdfs
    
# plot hit and false alarm rates and reaction times by image
hitRate = []
falseAlarmRate = []
hitReactionTime = []
falseAlarmReactionTime = []
for exp in Aexps:
    changeImage = result[exp]['changeImage']
    respToChange = result[exp]['responseToChange']
    changeReactionTime = result[exp]['changeReactionTime']
    nonChangeImage = result[exp]['nonChangeImage']
    respToNonChange = result[exp]['responseToNonChange']
    nonChangeReactionTime = result[exp]['nonChangeReactionTime']
    imageNames = np.unique(changeImage)
    hitRate.append([])
    falseAlarmRate.append([])
    hitReactionTime.append([])
    falseAlarmReactionTime.append([])
    for img in imageNames:
        hit = respToChange[changeImage==img]
        fa = respToNonChange[nonChangeImage==img]
        hitRate[-1].append(hit.sum()/hit.size)
        falseAlarmRate[-1].append(fa.sum()/fa.size)
        hitReactionTime[-1].append(np.median(changeReactionTime[(changeImage==img) & respToChange]))
        falseAlarmReactionTime[-1].append(np.median(nonChangeReactionTime[(nonChangeImage==img) & respToNonChange]))

imageNames[np.argsort(np.mean(hitRate,axis=0))]
np.argsort(np.mean(hitReactionTime,axis=0))

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
plt.plot(np.mean(hitRate,axis=0),np.mean(falseAlarmRate,axis=0),'ko',ms=8)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_yticks([0.15,0.2,0.25,0.3])
ax.set_xlabel('Hit rate',fontsize=14)
ax.set_ylabel('False alarm rate',fontsize=14)
plt.tight_layout()

alim = [410,480]

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
plt.plot(np.mean(hitRate,axis=0),np.mean(hitReactionTime,axis=0)*1000,'ko',ms=8,label='hit')
plt.plot(np.mean(falseAlarmRate,axis=0),np.mean(falseAlarmReactionTime,axis=0)*1000,'ko',ms=8,mfc='none',label='false alarm')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,1])
ax.set_ylim(alim)
ax.set_xlabel('Lick probability',fontsize=14)
ax.set_ylabel('Lick latency (ms)',fontsize=14)
ax.legend(loc='upper left',fontsize=12)
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot(alim,alim,'--',color='0.5')
plt.plot(np.nanmean(hitReactionTime,axis=0)*1000,np.nanmean(falseAlarmReactionTime,axis=0)*1000,'ko',ms=8)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_xlabel('Hit lick latency(ms)',fontsize=14)
ax.set_ylabel('False alarm lick latency (ms)',fontsize=14)
plt.tight_layout()


# plot mean response by image on lick and no lick trials
imgOrder = np.argsort(np.mean(hitRate,axis=0))
hit = []
change = []
catch = []
for region in regionLabels:
    hitResp = []
    missResp = []
    falseAlarmResp = []
    correctRejectResp = []
    changeCorr = []
    catchCorr = []
    for exp in Aexps:
        changeImage = result[exp]['changeImage']
        respToChange = result[exp]['responseToChange']
        changeReactionTime = result[exp]['changeReactionTime']
        nonChangeImage = result[exp]['nonChangeImage']
        respToNonChange = result[exp]['responseToNonChange']
        nonChangeReactionTime = result[exp]['nonChangeReactionTime']
        changeSDFs = result[exp][region]['active']['changeSDFs']
        if changeSDFs is not None:
            nonChangeSDFs = result[exp][region]['active']['nonChangeSDFs']
            changeSDFs = changeSDFs-changeSDFs[:,:10].mean(axis=1)[:,None]
            nonChangeSDFs = nonChangeSDFs-nonChangeSDFs[:,:10].mean(axis=1)[:,None]
            hitResp.append([])
            missResp.append([])
            falseAlarmResp.append([])
            correctRejectResp.append([])
            changeCorr.append([])
            catchCorr.append([])
            for img in imageNames[imgOrder]:
                hitResp[-1].append(changeSDFs[(changeImage==img) & respToChange].mean(axis=1).mean())
                missResp[-1].append(changeSDFs[(changeImage==img) & (~respToChange)].mean(axis=1).mean())
                falseAlarmResp[-1].append(nonChangeSDFs[(nonChangeImage==img) & respToNonChange].mean(axis=1).mean())
                correctRejectResp[-1].append(nonChangeSDFs[(nonChangeImage==img) & (~respToNonChange)].mean(axis=1).mean())
                changeCorr[-1].append(np.corrcoef(respToChange[changeImage==img],changeSDFs[changeImage==img].mean(axis=1))[0,1])
                catchCorr[-1].append(np.corrcoef(respToNonChange[nonChangeImage==img],nonChangeSDFs[nonChangeImage==img].mean(axis=1))[0,1])
    
    hit.append(np.nanmean(hitResp,axis=0))
    
    change.append(np.nanmean(changeCorr,axis=0))
    catch.append(np.nanmean(catchCorr,axis=0))
    
    plt.figure()
    ax = plt.subplot(1,1,1)
#    ax.plot(np.nanmean(hitResp,axis=0),'r')
#    ax.plot(np.nanmean(missResp,axis=0),'b')
#    ax.plot(np.nanmean(falseAlarmResp,axis=0),'r--')
#    ax.plot(np.nanmean(correctRejectResp,axis=0),'b--')
    ax.plot(np.nanmean(changeCorr,axis=0),'r')
    ax.plot(np.nanmean(catchCorr,axis=0),'r')
    ax.set_title(region)
    

# overlay of decoder accuracy and sdf    
fig = plt.figure(facecolor='w',figsize=(3,10))
popScore = []
popSdf = []
for i,region in enumerate(regionLabels):
    ax1 = plt.subplot(len(regionLabels),1,i+1)
    ax2 = ax1.twinx()
    regionScore = []
    sdfs = []
    for exp in result:
        s = result[exp][region]['active']['changeScoreWindows']['randomForest']
        if len(s)>2:
            regionScore.append(s[2])
#            behavior = result[exp]['responseToChange']
#            regionScore.append([np.corrcoef(behavior,p)[0,1] for p in s[2]])
            sdfs.append(result[exp][region]['active']['changeSDFs'].mean(axis=0))
    n = len(regionScore)
    if n>0:
        for pop,ax,clr,d,t in zip((popSdf,popScore),(ax1,ax2),'kr',(sdfs,regionScore),(np.arange(respWin.start,respWin.stop),decodeWindows+decodeWindowSize/2)):
            m = np.mean(d,axis=0)
            pop.append(m)
            s = np.std(d,axis=0)/(n**0.5)
            ax.plot(t,m,color=clr,label=score[:score.find('S')])
            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for ax in (ax1,ax2):
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,labelsize=8)
    ax1.set_xticks([250,300,350,400])
    ax1.set_xticklabels([0,50,100,150])
    ax1.set_xlim([decodeWindows[0],decodeWindows[-1]+decodeWindowSize])
    if i<len(regionLabels)-1:
        ax1.set_xticklabels([])
    else:
        ax1.set_xlabel('Time (ms)',fontsize=10)
plt.tight_layout()
    

fig = plt.figure(facecolor='w',figsize=(6,8))
ax = fig.add_subplot(2,1,1)
for r,clr in zip(('AM','MRN'),'kr'):
    s = popSdf[regionLabels.index(r)]
    plt.plot(np.arange(respWin.start,respWin.stop),s-s[:20].mean(),clr,label=r)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([250,300,350,400])
ax.set_xticklabels([0,50,100,150])
ax.set_xlim([respWin.start,respWin.stop])
ax.set_ylim([-1.9,13])
ax.set_ylabel('Spikes/s',fontsize=14)
ax.legend(loc='upper right',fontsize=12)
plt.tight_layout()

ax = fig.add_subplot(2,1,2)
for r,clr in zip(('AM','MRN'),'kr'):
    plt.plot(decodeWindows+decodeWindowSize/2,popScore[regionLabels.index(r)],clr,label=r)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([250,300,350,400])
ax.set_xticklabels([0,50,100,150])
ax.set_xlim([respWin.start,respWin.stop])
ax.set_yticks([0.5,0.6,0.7,0.8,0.9])
ax.set_ylim([0.475,0.9])
ax.set_xlabel('Time since change (ms)',fontsize=14)
ax.set_ylabel('Decoder accuracy',fontsize=14)
plt.tight_layout()


fig = plt.figure(facecolor='w',figsize=(6,8))
t = np.arange(150)
for i,region in enumerate(('AM','MRN')):
    ax = plt.subplot(2,1,i+1)
    changeLick = []
    changeNoLick = []
    nonChangeLick = []
    nonChangeNoLick = []
    for exp in result:
        changeSdfs = result[exp][region]['active']['changeSDFs']
        if changeSdfs is not None:
            respToChange = result[exp]['responseToChange']
            changeLick.append(changeSdfs[respToChange].mean(axis=0))
            changeNoLick.append(changeSdfs[~respToChange].mean(axis=0))
            nonChangeSdfs = result[exp][region]['active']['nonChangeSDFs']
            respToNonChange = result[exp]['responseToNonChange']
            nonChangeLick.append(nonChangeSdfs[respToNonChange].mean(axis=0))
            nonChangeNoLick.append(nonChangeSdfs[~respToNonChange].mean(axis=0))
    for d,lbl in zip((changeLick,changeNoLick,nonChangeLick,nonChangeNoLick),('change (lick)','change (no lick)','no change (lick)','no change (no lick)')):
        if region=='AM':
            clr = '0.7' if 'no change' in lbl else 'k'
        else:
            clr = (1,0.7,0.7) if 'no change' in lbl else 'r'
        line = ':' if 'no lick' in lbl else '-'
        m = np.mean(d,axis=0)
        ax.plot(t,m-m[:20].mean(),line,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks([0,50,100,150])
    ax.set_xlim([0,150])
    ax.set_ylim([-1.9,13])
    ax.set_ylabel('Spikes/s',fontsize=14)
    if i>0:
        ax.set_xlabel('Time since change (ms)',fontsize=14)
    ax.legend(loc='upper left',fontsize=10,frameon=False)
plt.tight_layout()




###### hit rate, lick time, and change mod correlation
                 
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im] for im in 'AB']
regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')

baseWin = slice(stimWin.start-150,stimWin.start)
respWin = slice(stimWin.start,stimWin.start+150)

exps = Aexps
imgNames = np.unique(data[exps[0]]['initialImage'])
nexps = np.zeros(len(regionLabels),dtype=int)
preSdfList,changeSdfList,preRespList,changeRespList,changeModList = [[[[[] for _ in imgNames] for _ in imgNames] for _ in regionLabels] for _ in range(5)]
hitMatrix = np.zeros((len(regionLabels),len(imgNames),len(imgNames)))
missMatrix = hitMatrix.copy()
lickLatMatrix = hitMatrix.copy()
hitMatrixAllExps = np.zeros((len(imgNames),)*2)
missMatrixAllExps = hitMatrixAllExps.copy()
lickLatMatrixAllExps = hitMatrixAllExps.copy()
for exp in exps:
    print(exp)
    initialImage = data[exp]['initialImage'][:]
    changeImage = data[exp]['changeImage'][:]
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    response = data[exp]['response'][:]
    hit = response=='hit'
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
#    changeTrials = engaged & (hit | (response=='miss'))
#    catchTrials = engaged & np.in1d(response('falseAlarm','correctReject'))
    
    lickTimes = data[exp]['lickTimes'][:]
    firstLickInd = np.searchsorted(lickTimes,changeTimes[engaged])
    lickLat = lickTimes[firstLickInd]-changeTimes[engaged]
    lickLat *= 1000
    
    for ind,trial in enumerate(np.where(engaged)[0]):
        i,j = [np.where(imgNames==img)[0][0] for img in (initialImage[trial],changeImage[trial])]
        if response[trial] in ('hit','falseAlarm'):
            hitMatrixAllExps[i,j] += 1
            lickLatMatrixAllExps[i,j] += lickLat[ind]
        else:
            missMatrixAllExps[i,j] += 1
    
    for r,region in enumerate(regionLabels):
        preSdfs = []
        changeSdfs = []
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,region)
            if any(inRegion):
                pre,change = [data[exp]['sdfs'][probe]['active'][epoch][inRegion,:][:,engaged] for epoch in ('preChange','change')]
                hasSpikes,hasResp = findResponsiveUnits(change,baseWin,respWin,thresh=5)
                preSdfs.append(pre[hasSpikes & hasResp])
                changeSdfs.append(change[hasSpikes & hasResp])
        if len(preSdfs)>0:
            nexps[r] += 1
            preSdfs = np.concatenate(preSdfs)
            changeSdfs = np.concatenate(changeSdfs)
            for ind,trial in enumerate(np.where(engaged)[0]):
                preResp,changeResp = [sdfs[:,ind,respWin].mean(axis=1)-sdfs[:,ind,baseWin].mean(axis=1) for sdfs in (preSdfs,changeSdfs)]
                i,j = [np.where(imgNames==img)[0][0] for img in (initialImage[trial],changeImage[trial])]
                preSdfList[r][i][j].append(preSdfs[:,ind])
                changeSdfList[r][i][j].append(changeSdfs[:,ind])
                preRespList[r][i][j].append(preResp)
                changeRespList[r][i][j].append(changeResp)
                changeModList[r][i][j].append(np.clip((changeResp-preResp)/(changeResp+preResp),-1,1))
                if response[trial] in ('hit','falseAlarm'):
                    hitMatrix[r,i,j] += 1
                    lickLatMatrix[r,i,j] += lickLat[ind]
                else:
                    missMatrix[r,i,j] += 1
                    
# analyze all experiments
nTrialsAllExps = hitMatrixAllExps+missMatrixAllExps
hitRateAllExps = hitMatrixAllExps/nTrialsAllExps
lickLatAllExps = lickLatMatrixAllExps/hitMatrixAllExps

diag = np.eye(len(imgNames),dtype=bool)
hitRateNonCatch = hitRateAllExps.copy()
hitRateNonCatch[diag] = np.nan
imgOrder = np.argsort(np.nanmean(hitRateNonCatch,axis=0))

for j,(d,lbl) in enumerate(zip((nTrialsAllExps,hitRateAllExps,lickLatAllExps),('Number of Trials','Hit Rate','Hit Lick Latency (ms)'))):
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(len(imgNames)))
    ax.set_yticks(np.arange(len(imgNames)))
    ax.set_xlim([-0.5,len(imgNames)-0.5])
    ax.set_ylim([len(imgNames)-0.5,-0.5])
    ax.set_xlabel('Change Image',fontsize=12)
    ax.set_ylabel('Initial Image',fontsize=12)
    ax.set_title(lbl,fontsize=10)
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    cb.ax.tick_params(labelsize=6)
    
hitRateAllExps[diag][imgOrder]
lickLatAllExps[diag][imgOrder]

np.nanmean(hitRateNonCatch,axis=0)[imgOrder]

lickLatNonCatch = lickLatAllExps.copy()
lickLatNonCatch[diag] = np.nan
np.nanmean(lickLatNonCatch,axis=0)[imgOrder]

                                    
# analyze by region           
nTrialsMatrix = hitMatrix+missMatrix
hitRate = hitMatrix/nTrialsMatrix
lickLatMatrix /= hitMatrix
imgOrder = np.argsort(np.nanmean(hitRate,axis=(0,1)))
nonDiag = ~np.eye(len(imgNames),dtype=bool)

respLatMatrix = np.full(hitMatrix.shape,np.nan)
diffLatMatrix = respLatMatrix.copy()
preRespMatrix = respLatMatrix.copy()
changeRespMatrix = respLatMatrix.copy()
changeModMatrix = respLatMatrix.copy()
for r,_ in enumerate(regionLabels):
    for i,_ in enumerate(imgNames):
        for j,_ in enumerate(imgNames):
            if len(preRespList[r][i][j])>0:
                preSdf,changeSdf = [np.nanmean(np.concatenate(s[r][i][j]),axis=0) for s in (preSdfList,changeSdfList)]
                respLatMatrix[r,i,j] = findLatency(changeSdf,baseWin,stimWin,method='abs',thresh=0.5)[0]
                diffLatMatrix[r,i,j] = findLatency(changeSdf-preSdf,baseWin,stimWin,method='abs',thresh=0.5)[0]
                preRespMatrix[r,i,j] = np.nanmean(np.concatenate(preRespList[r][i][j]))
                changeRespMatrix[r,i,j] = np.nanmean(np.concatenate(changeRespList[r][i][j]))
                changeModMatrix[r,i,j] = np.nanmean(np.concatenate(changeModList[r][i][j]))
popChangeModMatrix = np.clip((changeRespMatrix-preRespMatrix)/(changeRespMatrix+preRespMatrix),-1,1)

for region,n,ntrials,hr,lickLat,respLat,diffLat,preResp,changeResp,changeMod in zip(regionLabels,nexps,nTrialsMatrix,hitRate,lickLatMatrix,respLatMatrix,diffLatMatrix,preRespMatrix,changeRespMatrix,changeModMatrix):
    fig = plt.figure(facecolor='w',figsize=(10,5))
    fig.text(0.01,0.99,region+' ('+str(n)+' experiments)',fontsize=8,horizontalalignment='left',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(2,4)
    
    for j,(d,lbl) in enumerate(zip((ntrials,hr,lickLat),('Number of Trials','Lick Probability','Hit Lick Latency (ms)'))):
        ax = plt.subplot(gs[0,j])
        im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
        ax.tick_params(direction='out',top=False,right=False,labelsize=6)
        ax.set_xticks(np.arange(len(imgNames)))
        ax.set_yticks(np.arange(len(imgNames)))
        ax.set_xlim([-0.5,len(imgNames)-0.5])
        ax.set_ylim([len(imgNames)-0.5,-0.5])
        ax.set_xlabel('Change Image',fontsize=8)
        ax.set_ylabel('Initial Image',fontsize=8)
        ax.set_title(lbl,fontsize=10)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(labelsize=6)
    
    ax = plt.subplot(gs[0,3])
    ax.plot(hr[nonDiag],lickLat[nonDiag],'ko')
    slope,yint,rval,pval,stderr = scipy.stats.linregress(hr[nonDiag],lickLat[nonDiag])
    x = np.array([hr[nonDiag].min(),hr[nonDiag].max()])
    ax.plot(x,slope*x+yint,'0.5')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlabel('Hit Rate',fontsize=10)
    ax.set_ylabel('Hit Lick Latency (ms)',fontsize=10)
    r,p = scipy.stats.pearsonr(hr[nonDiag],lickLat[nonDiag])
    ax.set_title('r = '+str(round(r,2))+', p = '+'{0:1.1e}'.format(p),fontsize=8)
    
#    for i,(d,ylbl) in enumerate(zip((respLat,diffLat,preResp,changeResp,changeMod),('Response Latency (ms)','Change Modulaton Latency (ms)','Pre-change Response (spikes/s)','Change Response (spikes/s)','Change Modulation Index'))):
    for i,(d,ylbl) in enumerate(zip((changeResp,),('Spikes/s',))):
        ax = plt.subplot(gs[i+1,0])
        im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
        ax.tick_params(direction='out',top=False,right=False,labelsize=6)
        ax.set_xticks(np.arange(len(imgNames)))
        ax.set_yticks(np.arange(len(imgNames)))
        ax.set_xlim([-0.5,len(imgNames)-0.5])
        ax.set_ylim([len(imgNames)-0.5,-0.5])
        ax.set_xlabel('Change Image',fontsize=8)
        ax.set_ylabel('Initial Image',fontsize=8)
        ax.set_title(ylbl,fontsize=10)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(labelsize=6)
        
        for j,(b,xlbl) in enumerate(zip((hr,lickLat),('Lick Probability','Hit Lick Latency (ms)'))):   
            ax = plt.subplot(gs[i+1,j+1])
            notnan = ~np.isnan(d)
            ax.plot(b[notnan],d[notnan],'ko')
            slope,yint,rval,pval,stderr = scipy.stats.linregress(b[nonDiag],d[nonDiag])
            x = np.array([b[nonDiag].min(),b[nonDiag].max()])
            ax.plot(x,slope*x+yint,'0.5')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            if i==0:
                ax.set_xlabel(xlbl,fontsize=10)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=10)
            r,p = scipy.stats.pearsonr(b[nonDiag],d[nonDiag])
            ax.set_title('r = '+str(round(r,2))+', p = '+'{0:1.1e}'.format(p),fontsize=8)
    plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
xticks = np.arange(len(regionLabels))
xlim = [-0.5,len(regionLabels)-0.5]
for param,clr,lbl in zip((changeRespMatrix,),'k',('Change Resp',)):
    r = [scipy.stats.pearsonr(hr[nonDiag],d[nonDiag])[0] for hr,d in zip(hitRate,param)]
    mfc = 'none' if 'Latency' in lbl else clr
    ax.plot(xticks,np.absolute(r),'o',ms=10,mec=clr,mfc=mfc,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(('V1','LM','AL','RL','PM','AM'),fontsize=14)
ax.set_xlim(xlim)
ax.set_ylim([0.25,0.75])
ax.set_ylabel('Pearson r',fontsize=14)
#ax.legend(loc='upper left')
plt.tight_layout()



# run speed

frameRate = 60
preTime = 7500
postTime = 7500
sampInt = 10
plotTime = np.arange(-preTime,postTime+sampInt,sampInt)
lickBins = np.arange(-preTime-sampInt/2,postTime+sampInt,sampInt)

states = ('behavior',)
hitRunSpeed,missRunSpeed,falseAlarmRunSpeed,correctRejectRunSpeed,engagedRunSpeed,disengagedRunSpeed,omitRunSpeed,omitPreChangeRunSpeed = [{state: [] for state in states} for _ in range(8)]
hitLickProb = []
falseAlarmLickProb = []

exps = data.keys()

for exp in exps:
    print(exp)
    response = data[exp]['response'][:]
    hit = response=='hit'
    miss = response=='miss'
    falseAlarm = response=='falseAlarm'
    correctReject = response=='correctReject'
    for state in states:
        runSpeed = data[exp][state+'RunSpeed'][:]
        medianRunSpeed = np.median(runSpeed)
        if medianRunSpeed<1:
            break
        runSpeed -= medianRunSpeed
        runTime = data[exp][state+'RunTime'][:]
        flashTimes = data[exp][state+'FlashTimes'][:]
        changeTimes = data[exp][state+'ChangeTimes'][:]
        preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
        omitTimes = data[exp][state+'OmitFlashTimes'][:]
        omitChangeDiff = omitTimes-changeTimes[:,None]
        engagedChange,engagedOmit = [np.min(np.absolute(times-changeTimes[hit][:,None]),axis=0) < 60 for times in (changeTimes,omitTimes)]
        for ind,(speed,times) in enumerate(((hitRunSpeed[state], changeTimes[engagedChange & hit]),
                                            (missRunSpeed[state], changeTimes[engagedChange & miss]),
                                            (falseAlarmRunSpeed[state], changeTimes[engagedChange & falseAlarm]),
                                            (correctRejectRunSpeed[state], changeTimes[engagedChange & correctReject]),
                                            (engagedRunSpeed[state], changeTimes[engagedChange]),
                                            (disengagedRunSpeed[state], changeTimes[~engagedChange]),
                                            (omitRunSpeed[state], omitTimes[engagedOmit & np.all((omitChangeDiff<0) | (omitChangeDiff>7.5),axis=0)]),
                                            (omitPreChangeRunSpeed[state], omitTimes[engagedOmit & np.any((omitChangeDiff<0) & (omitChangeDiff>-2.5),axis=0)]),
                                          )):
            if ind in (4,5) and np.all(engagedChange):
                continue
            elif len(times)>0:
                trialSpeed = []
                for t in times:
                    i = (runTime>=t-preTime) & (runTime<=t+postTime)
                    trialSpeed.append(np.interp(plotTime,1000*(runTime[i]-t),runSpeed[i]))
                speed.append(np.mean(trialSpeed,axis=0))
        if state=='behavior':
            lickTimes = data[exp]['lickTimes'][:]
            nonChangeFlashTimes = []
            for t in flashTimes:
                timeFromChange = changeTimes-t
                timeFromLick = lickTimes-t
                if (min(abs(timeFromChange))<60 and 
                    (not any(timeFromChange<0) or max(timeFromChange[timeFromChange<0])<-4) and 
                    (not any(timeFromLick<0) or max(timeFromLick[timeFromLick<0])<-4) and
                    any((timeFromLick>0.15) & (timeFromLick<0.75))):
                        nonChangeFlashTimes.append(t)
            for lickProb,times in zip((hitLickProb,falseAlarmLickProb),(changeTimes[engagedChange & hit],nonChangeFlashTimes)):
                if len(times)>0:
                    trialLicks = []
                    for t in times:
                        licks = lickTimes[(lickTimes>t-preTime) & (lickTimes<t+postTime)]
                        jitter = np.random.random_sample(licks.size)*(1/frameRate)-(0.5/frameRate)
                        trialLicks.append(np.histogram(1000*(licks-t+jitter),lickBins)[0])
                    lickProb.append(np.mean(trialLicks,axis=0))


xlim = [-1250,1250]
ylim = [0,40]
plotFlashTimes = np.concatenate((np.arange(-750,-preTime,-750),np.arange(0,postTime,750)))
for state in states:
    fig = plt.figure(facecolor='w',figsize=(6,8))
    ax = fig.add_subplot(3,1,1)
    for t in plotFlashTimes:
        ax.add_patch(matplotlib.patches.Rectangle([t,ylim[0]],width=250,height=ylim[1]-ylim[0],color='0.9',alpha=0.5,zorder=0))
    for speed,clr,lbl in zip((hitRunSpeed[state],correctRejectRunSpeed[state]),'rk',('hit (engaged)','correct reject (engaged)')):
        m = np.mean(speed,axis=0)
        n = len(speed)
        s = np.std(speed,axis=0)/(n**0.5)
        ax.plot(plotTime,m,clr,label=lbl+', n='+str(n))
        ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Time from change (ms)',fontsize=10)
    ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=10)
    ax.legend(loc='upper right',fontsize=8)
    ax.set_title(state,fontsize=12)
    
    ax = fig.add_subplot(3,1,2)
    for t in plotFlashTimes:
        ax.add_patch(matplotlib.patches.Rectangle([t,ylim[0]],width=250,height=ylim[1]-ylim[0],color='0.9',alpha=0.5,zorder=0))
    for speed,clr,lbl in zip((engagedRunSpeed[state],disengagedRunSpeed[state]),'mg',('engaged (all changes)','disengaged (all changes)')):
        m = np.mean(speed,axis=0)
        n = len(speed)
        s = np.std(speed,axis=0)/(n**0.5)
        ax.plot(plotTime,m,clr,label=lbl+', n='+str(n))
        ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Time from change (ms)',fontsize=10)
    ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=10)
    ax.legend(loc='upper right',fontsize=8)

    ax = fig.add_subplot(3,1,3)
    for t in plotFlashTimes:
        if t==0:
            ax.plot([t,t],ylim,'--',color='0.9',zorder=0)
        else:
            ax.add_patch(matplotlib.patches.Rectangle([t,ylim[0]],width=250,height=ylim[1]-ylim[0],color='0.9',alpha=0.5,zorder=0))
    for speed,clr,lbl in zip((omitPreChangeRunSpeed[state],),'k',('<2500 ms before change (engaged)',)):
        m = np.mean(speed,axis=0)
        n = len(speed)
        s = np.std(speed,axis=0)/(n**0.5)
        ax.plot(plotTime,m,clr,label=lbl+', n='+str(n))
        ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Time from omitted flash (ms)',fontsize=10)
    ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=10)
    loc = 'lower right' if state=='behavior' else 'upper right'
    ax.legend(loc=loc,fontsize=8)
    plt.tight_layout()
    
    
xlim = [-2.75,2.75]
ylim = [0,45]
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
for t in plotFlashTimes:
    clr = '0.8' if t<0 else '0.4'
    ax.add_patch(matplotlib.patches.Rectangle([t/1000,ylim[0]],width=0.25,height=ylim[1]-ylim[0],color=clr,alpha=0.2,zorder=0))
for speed,clr,lbl in zip((hitRunSpeed['behavior'],hitRunSpeed['passive']),'mg',('active','passive')):
    m = np.mean(speed,axis=0)
    n = len(speed)
    s = np.std(speed,axis=0)/(n**0.5)
    ax.plot(plotTime/1000,m,clr,label=lbl)
    ax.fill_between(plotTime/1000,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_yticks(np.arange(0,60,10))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Time from change (s)',fontsize=12)
ax.set_ylabel('Run speed (cm/s)',fontsize=12)
ax.legend(loc='lower left',fontsize=12)


xlim = [-500,500]
ylim = [-35,10]
fig = plt.figure(facecolor='w',figsize=(6,8))
ax = fig.add_subplot(2,1,1)
for lickProb,clr,lbl in zip((hitLickProb,falseAlarmLickProb),'gm',('hit','false alarm')):
    m = np.nanmean(lickProb,axis=0)
    n = len(lickProb)
    s = np.std(lickProb,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim(xlim)
ax.set_ylim([0,0.075])
ax.set_ylabel('Lick Probability',fontsize=14)

ax = fig.add_subplot(2,1,2)
#ax.plot([130]*2,ylim,'k:')
#ax.text(130,5.1,'Reaction time\n~130 ms',horizontalalignment='center')
#for speed,clr,lbl in zip((hitRunSpeed['behavior'],falseAlarmRunSpeed['behavior']),'gm',('hit','false alarm')):
for speed,clr,lbl in zip((hitRunSpeed['behavior'],falseAlarmRunSpeed['behavior'],correctRejectRunSpeed['behavior']),'gmk',('hit','false alarm','correct reject')):
#for speed,clr,lbl in zip((hitRunSpeed['behavior'],missRunSpeed['behavior'],falseAlarmRunSpeed['behavior'],correctRejectRunSpeed['behavior']),'grmk',('hit','miss','false alarm','correct reject')):
    m = np.mean(speed,axis=0)
    n = len(speed)
    s = np.std(speed,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_yticks(np.arange(-40,10,10))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Time from change/catch (ms)',fontsize=14)
ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=14)
ax.legend(loc='lower left',fontsize=14)


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
for lickProb,clr,lbl in zip((hitLickProb,falseAlarmLickProb),'gm',('Change','Non-change')):
    m = np.nanmean(lickProb,axis=0)
    n = len(lickProb)
    s = np.std(lickProb,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,500])
ax.set_ylim([0,0.075])
ax.set_xlabel('Time from image onset (ms)',fontsize=14)
ax.set_ylabel('Lick Probability',fontsize=14)
ax.legend(fontsize=14)


xlim = [-500,500]
ylim = [-35,5]
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
ax.plot([130]*2,ylim,'k:')
ax.text(130,5.1,'Reaction time\n~130 ms',horizontalalignment='center')
for speed,clr,lbl in zip((hitRunSpeed['behavior'],missRunSpeed['behavior'],falseAlarmRunSpeed['behavior'],correctRejectRunSpeed['behavior']),'krgb',('hit','miss','falseAlarm','correct reject')):
    m = np.mean(speed,axis=0)
    n = len(speed)
    s = np.std(speed,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_yticks(np.arange(-50,50,10))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Time from change/catch (ms)',fontsize=12)
ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=12)
ax.legend(loc='lower left',fontsize=12)


###### lick prob for all changes and non-change flashes
exps = data.keys()
flashType = ('flash','change')
timeSinceChange = {event:[] for event in flashType}
timeSinceLick = {event:[] for event in flashType}
timeSinceReward = {event:[] for event in flashType}
lickInWindow = {event:[] for event in flashType}
lickLatency = {event:[] for event in flashType}
hitRate = []
falseAlarmRate = []
for exp in exps:
    print(exp)
    response = data[exp]['response'][:]
    flashTimes = data[exp]['behaviorFlashTimes'][:]
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    changeTrials = np.in1d(response,('hit','miss'))
    lickTimes = data[exp]['lickTimes'][:]
    rewardTimes = data[exp]['rewardTimes'][:]
    engagedFlash,engagedChange = [np.min(np.absolute(times-changeTimes[response=='hit',None]),axis=0) < 60 for times in (flashTimes,changeTimes)]
    firstChange = changeTimes[changeTrials][0]
    flashes = flashTimes[engagedFlash & (flashTimes>firstChange) & (~np.in1d(flashTimes,changeTimes[changeTrials]))]
    changes = changeTimes[changeTrials & engagedChange & (changeTimes>firstChange)]
    for event,times in zip(flashType,(flashes,changes)):
        timeSinceChange[event].append(times-changeTimes[changeTrials][np.searchsorted(changeTimes[changeTrials],times)-1])
        timeSinceLick[event].append(times-lickTimes[np.searchsorted(lickTimes,times)-1])
        timeSinceReward[event].append(times-rewardTimes[np.searchsorted(rewardTimes,times)-1])
        timeToLick = lickTimes-times[:,None]
        lickInWindow[event].append(np.any((timeToLick>0.15) & (timeToLick<0.75),axis=1))
        timeToLick[timeToLick<0] = np.nan
        lickLatency[event].append(np.nanmin(timeToLick,axis=1))
    hitRate.append(np.sum(response[engagedChange]=='hit')/np.sum(changeTrials & engagedChange))
    falseAlarmRate.append(np.sum(response[engagedChange]=='falseAlarm')/np.sum(~changeTrials & engagedChange))

bins = np.arange(0.375,61,0.75)   
binTimes = bins-0.375

for exp,_ in enumerate(exps):
    fig = plt.figure(facecolor='w',figsize=(6,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot([0,60],[hitRate[exp]]*2,'g--',label='hit rate')
    ax1.plot([0,60],[falseAlarmRate[exp]]*2,'r--',label='false alarm rate')
    for event,clr in zip(flashType,'rg'):
        timeToBinIndex = np.searchsorted(bins,timeSinceChange[event][exp])
        lick = np.zeros(len(bins))
        nolick = lick.copy()
        for i in range(len(bins)):
            ind = np.where(timeToBinIndex==i)[0]
            lick[i] = lickInWindow[event][exp][ind].sum()
            nolick[i] = len(ind)-lick[i]
        ax1.plot(binTimes,lick/(lick+nolick),clr,label=event)
        ax2.plot(binTimes,lick+nolick,clr)
    for ax in (ax1,ax2):
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
    ax1.set_ylabel('Lick Prob.')
    ax2.set_ylabel('N')
    ax2.set_xlabel('Time since change (s)')
    ax1.legend()


for times,xmax,xlbl in zip((timeSinceChange,timeSinceLick),(55,25),('Time since change','Time since lick')):
    fig = plt.figure(facecolor='w',figsize=(6,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for d,clr,lbl in zip((hitRate,falseAlarmRate),'kk',('hit rate','false alarm rate')):
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax1.plot([0,60],[m,m],'--',color=clr)
        ax1.fill_between([0,60],[m+s]*2,[m-s]*2,color=clr,alpha=0.25)
    for event,clr,lbl in zip(flashType[::-1],'gm',('Change','Non-change')):
        lickProb = np.zeros((len(exps),len(bins)))
        n = lickProb.copy()
        for i,(t,licks) in enumerate(zip(times[event],lickInWindow[event])):
            timeToBinIndex = np.searchsorted(bins,t)
            for j in range(len(bins)):
                ind = np.where(timeToBinIndex==j)[0]
                lickProb[i,j] = licks[ind].sum()/len(ind)
                n[i,j] = len(ind)
        for d,ax in zip((lickProb,n),(ax1,ax2)):
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(len(exps)**0.5)
            ax.plot(binTimes,m,clr,label=lbl)
            ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for ax in (ax1,ax2):
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,xmax])
    ax1.set_ylabel('Lick probability',fontsize=14)
    ax2.set_ylabel('Number of events',fontsize=14)
    ax2.set_xlabel(xlbl+' (s)',fontsize=14)
    ax2.legend(fontsize=14)
    plt.tight_layout()
    
    
for times,xmax,xlbl in zip((timeSinceChange,timeSinceLick),(55,25),('Time since change','Time since lick')):
    fig = plt.figure(facecolor='w',figsize=(6,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for d,clr,lbl in zip((hitRate,falseAlarmRate),'kk',('hit rate','false alarm rate')):
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax1.plot([0,60],[m,m],'--',color=clr)
        ax1.fill_between([0,60],[m+s]*2,[m-s]*2,color=clr,alpha=0.25)
    for event,clr,lbl in zip(flashType[::-1],'gm',('Change','Non-change')):
        lickProb = np.zeros((len(exps),len(bins)))
        n = lickProb.copy()
        for i,(t,licks) in enumerate(zip(times[event],lickInWindow[event])):
            timeToBinIndex = np.searchsorted(bins,t)
            for j in range(len(bins)):
                ind = np.where(timeToBinIndex==j)[0]
                lickProb[i,j] = licks[ind].sum()/len(ind)
                n[i,j] = len(ind)/len(t)
        for d,ax in zip((lickProb,n),(ax1,ax2)):
            if d is lickProb or event=='change':
                c = 'k' if d is n else clr
                m = np.nanmean(d,axis=0)
                s = np.nanstd(d,axis=0)/(len(exps)**0.5)
                ax.plot(binTimes,m,c,label=lbl)
                ax.fill_between(binTimes,m+s,m-s,color=c,alpha=0.25)
    for ax in (ax1,ax2):
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,xmax])
    ax2.set_yticks([0,0.05,0.1,0.15])
    ax1.set_ylabel('Lick probability',fontsize=14)
    ax2.set_ylabel('Fraction of changes',fontsize=14)
    ax2.set_xlabel(xlbl+' (s)',fontsize=14)
    ax1.legend(fontsize=14,frameon=False)
    plt.tight_layout()
    
for times,xmax,xlbl in zip((timeSinceChange,timeSinceLick),(55,25),('Time since change','Time since lick')):
    fig = plt.figure(facecolor='w',figsize=(6,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
#    for d,clr,lbl in zip((falseAlarmRate,),'k',('false alarm rate',)):
#        m = np.mean(d)
#        s = np.std(d)/(len(d)**0.5)
#        ax1.plot([0,60],[m,m],'--',color=clr)
#        ax1.fill_between([0,60],[m+s]*2,[m-s]*2,color=clr,alpha=0.25)
    for event,clr,lbl in zip(flashType[::-1],'kk',('Change','Non-change')):
        lickProb = np.zeros((len(exps),len(bins)))
        n = lickProb.copy()
        for i,(t,licks) in enumerate(zip(times[event],lickInWindow[event])):
            timeToBinIndex = np.searchsorted(bins,t)
            for j in range(len(bins)):
                ind = np.where(timeToBinIndex==j)[0]
                lickProb[i,j] = licks[ind].sum()/len(ind)
                n[i,j] = len(ind)/len(t)
        for d,ax in zip((lickProb,n),(ax1,ax2)):
            if (ax is ax1 and event=='flash') or (ax is ax2 and event=='change'):
                c = 'k' if d is n else clr
                m = np.nanmean(d,axis=0)
                s = np.nanstd(d,axis=0)/(len(exps)**0.5)
                ax.plot(binTimes,m,c,label=lbl)
                ax.fill_between(binTimes,m+s,m-s,color=c,alpha=0.25)
    for ax in (ax1,ax2):
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,xmax])
    ax2.set_yticks([0,0.05,0.1,0.15])
    ax1.set_ylabel('Lick probability',fontsize=14)
    ax2.set_ylabel('Fraction of changes',fontsize=14)
    ax2.set_xlabel(xlbl+' (s)',fontsize=14)
#    ax1.legend(fontsize=14,frameon=False)
    plt.tight_layout()



fig = plt.figure(facecolor='w')
ax = fig.add_subplot(1,1,1)
binsize = 0.05
bins = np.arange(0,7,binsize)
x = bins[:-1]+binsize/2
for event,clr in zip(flashType[::-1],'gr'):
    lickLat = []
    for lick,lat,lastChange,lastLick in zip(lickInWindow[event],lickLatency[event],timeSinceChange[event],timeSinceLick[event]):
        if event=='flash':
            lat = lat[(lastChange>4) & (lastLick>4)]
        lickLat.append(np.histogram(lat,bins)[0])
    m = np.nanmean(lickLat,axis=0)
    s = np.nanstd(lickLat,axis=0)/(len(exps)**0.5)
    ax.plot(x,m,clr,label=event)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
#ax2.set_xlabel('Time since lick (s)')
#ax.set_ylabel('Lick Prob.')
ax.legend()
plt.tight_layout()


# "Rabbit effect" analysis for Christof

hitRateAfterHit = []
hitRateAfterMiss = []
reactionTimeAfterHit = [[] for exp in exps]
reactionTimeAfterMiss = [[] for exp in exps]
reactionTimeAfterHitByImage = [[[] for exp in exps] for _ in range(8)]
reactionTimeAfterMissByImage = [[[] for exp in exps] for _ in range(8)]
for expInd,exp in enumerate(exps):
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    changeImage = data[exp]['changeImage'][:]
    imageNames = np.unique(changeImage)
    response = data[exp]['response'][:]
    hit = response=='hit'
    miss = response=='miss'
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    reactionTime = data[exp]['rewardTimes'][:]-changeTimes
    changeTrials = np.where(engaged & (hit | miss))[0]
    hitAfterHit = 0
    missAfterHit = 0
    hitAfterMiss = 0
    missAfterMiss = 0
    for prev,trial in zip(changeTrials[0:],changeTrials[1:]):
        if hit[trial]:
            imgInd = np.where(imageNames==changeImage[trial])[0][0]
            if hit[prev]:
                hitAfterHit += 1
                reactionTimeAfterHit[expInd].append(reactionTime[trial])
                reactionTimeAfterHitByImage[imgInd][expInd].append(reactionTime[trial])
            else:
                hitAfterMiss += 1
                reactionTimeAfterMiss[expInd].append(reactionTime[trial])
                reactionTimeAfterMissByImage[imgInd][expInd].append(reactionTime[trial])
        else:
            if hit[prev]:
                missAfterHit += 1
            else:
                missAfterMiss += 1
    hitRateAfterHit.append(hitAfterHit/(hitAfterHit+missAfterHit))
    hitRateAfterMiss.append(hitAfterMiss/(hitAfterMiss+missAfterMiss))


fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
alim = [0,1.05]
ax.plot(alim,alim,color='0.5',linestyle='--')
ax.plot(hitRateAfterHit,hitRateAfterMiss,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Hit rate after hit for each session')
ax.set_ylabel('Hit rate after miss for each session')
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
alim = [250,650]
ax.plot(alim,alim,color='0.5',linestyle='--')
rtHit,rtMiss = [[np.median(exp)*1000 for exp in rt] for rt in (reactionTimeAfterHit,reactionTimeAfterMiss)]
ax.plot(rtHit,rtMiss,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(100,800,100))
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Reaction time after hit for each session (ms)')
ax.set_ylabel('Reaction time after miss for each session (ms)')
plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
alim = [250,650]
ax.plot(alim,alim,color='0.5',linestyle='--')
rtHit,rtMiss = [[np.nanmedian([np.median(exp) for exp in img])*1000 for img in rt] for rt in (reactionTimeAfterHitByImage,reactionTimeAfterMissByImage)]
ax.plot(rtHit,rtMiss,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(100,800,100))
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('Reaction time after hit for each image (ms)')
ax.set_ylabel('Reaction time after miss for each image (ms)')
plt.tight_layout()



