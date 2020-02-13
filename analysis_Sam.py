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
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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


def getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze='all',sdfParams={}):
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
                data[expName]['sdfs'] = getSDFs(obj,probes=probes,**sdfParams)
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


def findResponsiveUnits(sdfs,baseWin,respWin,thresh=5):
    unitMeanSDFs = sdfs.mean(axis=1) if len(sdfs.shape)>2 else sdfs.copy()
    hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
    unitMeanSDFs -= unitMeanSDFs[:,baseWin].mean(axis=1)[:,None]
    hasResp = unitMeanSDFs[:,respWin].max(axis=1) > thresh*unitMeanSDFs[:,baseWin].std(axis=1)
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
            )


#
makeSummaryPlots(miceToAnalyze=('421323','422856','423749'))


# make new experiment hdf5s without updating popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=False,miceToAnalyze=('479219',))

# make new experiment hdf5s and add to existing popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True,miceToAnalyze=('461027',))

# make new experiment hdf5s and popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True)

# make popData.hdf5 from existing experiment hdf5s
getPopData(objToHDF5=False,popDataToHDF5=True)

# append existing hdf5s to existing popData.hdf5
getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze=('461027',))



data = h5py.File(os.path.join(localDir,'popData.hdf5'),'r')

exps = data.keys()

# A or B days that have passive session
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im and hasPassive] for im in 'AB']
exps = Aexps+Bexps

baseWin = slice(0,250)
stimWin = slice(250,500)
respWin = slice(stimWin.start,stimWin.start+151)



###### behavior analysis
exps = data.keys()
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
ax.set_title('n = '+str(nMice)+' mice,\n'+str(len(exps))+' ephys recording sessions',fontsize=16)

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
ax.set_title('n = '+str(nMice)+' mice, '+str(len(exps))+' days',fontsize=16)


# compare active and passive running
activeRunSpeed,passiveRunSpeed = [[np.median(data[exp][speed]) for exp in exps] for speed in ('behaviorRunSpeed','passiveRunSpeed')]
            
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
amax = 1.05*max(np.max(activeRunSpeed),np.max(passiveRunSpeed))
ax.plot([0,amax],[0,amax],'k--')
ax.plot(activeRunSpeed,passiveRunSpeed,'o',ms=10,mec='k',mfc='none')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,amax])
ax.set_ylim([0,amax])
ax.set_aspect('equal')
ax.set_xlabel('Median Active Run Speed (cm/s)')
ax.set_ylabel('Median Passive Run Speed (cm/s)')



###### change mod and latency analysis

behavStates = ('active','passive')

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

mouseIDs = [[] for r in regionNames]
expDates =  [[] for r in regionNames]
unitCount = [[] for r in regionNames]
activeChangeSdfs = [[] for r in regionNames]
activePreSdfs = [[] for r in regionNames]
passiveChangeSdfs = [[] for r in regionNames]
passivePreSdfs = [[] for r in regionNames]
activeFirstSpikeLat = [[] for r in regionNames]
for ind,(region,regionCCFLabels) in enumerate(regionNames):
    print(region)
    for exp in exps:
        print(exp)
        response = data[exp]['response'][:]
        hit = response=='hit'
        changeTimes = data[exp]['behaviorChangeTimes'][:]
        engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
        trials = engaged & (hit | (response=='miss'))
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,regionCCFLabels)
            if any(inRegion):
                activePre,activeChange = [data[exp]['sdfs'][probe]['active'][epoch][inRegion,:][:,trials].mean(axis=1) for epoch in ('preChange','change')]
                hasSpikesActive,hasRespActive = findResponsiveUnits(activeChange,baseWin,respWin,thresh=5)
                if 'passive' in behavStates:
                    passivePre,passiveChange = [data[exp]['sdfs'][probe]['passive'][epoch][inRegion,:][:,trials].mean(axis=1) for epoch in ('preChange','change')]
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChange,baseWin,respWin,thresh=5)
                    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
                else:
                    hasResp = hasSpikesActive & hasRespActive
                if hasResp.sum()>0:
                    mouseIDs[ind].append(exp[-6:])
                    expDates[ind].append(exp[:8])
                    unitCount[ind].append(hasResp.sum())
                    activePreSdfs[ind].append(activePre[hasResp])
                    activeChangeSdfs[ind].append(activeChange[hasResp])
                    if 'passive' in behavStates:
                        passivePreSdfs[ind].append(passivePre[hasResp])
                        passiveChangeSdfs[ind].append(passiveChange[hasResp])
                    for u in data[exp]['units'][probe][inRegion][hasResp]:
                        spikeTimes = data[exp]['spikeTimes'][probe][str(u)][:,0]
                        lat = []
                        for t in changeTimes:
                            firstSpike = np.where((spikeTimes > t+0.03) & (spikeTimes < t+0.15))[0]
                            if len(firstSpike)>0:
                                lat.append(spikeTimes[firstSpike[0]]-t)
                            else:
                                lat.append(np.nan)
                        activeFirstSpikeLat[ind].append(np.nanmedian(lat))

nMice = np.array([len(set(m)) for m in mouseIDs])
nExps = np.array([len(set(d)) for d in expDates])
nUnits = np.array([sum(n) for n in unitCount])
activePreSdfs,activeChangeSdfs = [[np.concatenate(s) for s in sdfs] for sdfs in (activePreSdfs,activeChangeSdfs)]
if 'passive' in behavStates:
    passivePreSdfs,passiveChangeSdfs = [[np.concatenate(s) for s in sdfs] for sdfs in (passivePreSdfs,passiveChangeSdfs)]


# calculate metrics    
preBase,changeBase = [[s[:,baseWin].mean(axis=1) for s in sdfs] for sdfs in (activePreSdfs,activeChangeSdfs)]
preRespActive,changeRespActive = [[s[:,respWin].mean(axis=1)-b for s,b in zip(sdfs,base)] for sdfs,base in zip((activePreSdfs,activeChangeSdfs),(preBase,changeBase))]
baseRateActive = changeBase
changeModActive = [np.clip((change-pre)/(change+pre),-1,1) for pre,change in zip(preRespActive,changeRespActive)]
changeModLatActive = [findLatency(change-pre,baseWin,stimWin,method='abs',thresh=0.5,maxval=150) for pre,change in zip(activePreSdfs,activeChangeSdfs)]
popChangeModLatActive = np.array([findLatency(np.mean(change-pre,axis=0),baseWin,stimWin,method='abs',thresh=0.5)[0] for pre,change in zip(activePreSdfs,activeChangeSdfs)])
respLatActive = [findLatency(sdfs,baseWin,stimWin,method='abs',thresh=0.5,maxval=150) for sdfs in activeChangeSdfs]
popRespLatActive = np.array([findLatency(sdfs.mean(axis=0),baseWin,stimWin,method='abs',thresh=0.5)[0] for sdfs in activeChangeSdfs])

if 'passive' in behavStates:
    preBase,changeBase = [[s[:,baseWin].mean(axis=1) for s in sdfs] for sdfs in (passivePreSdfs,passiveChangeSdfs)]
    preRespPassive,changeRespPassive = [[s[:,respWin].mean(axis=1)-b for s,b in zip(sdfs,base)] for sdfs,base in zip((passivePreSdfs,passiveChangeSdfs),(preBase,changeBase))]
    baseRatePassive = changeBase
    changeModPassive = [np.clip((change-pre)/(change+pre),-1,1) for pre,change in zip(preRespPassive,changeRespPassive)]
    behavModPre = [np.clip((active-passive)/(active+passive),-1,1) for active,passive in zip(preRespActive,preRespPassive)]
    behavModChange = [np.clip((active-passive)/(active+passive),-1,1) for active,passive in zip(changeRespActive,changeRespPassive)]

   
# plot pre and post change sdfs and their difference
for region,activePre,activeChange,passivePre,passiveChange in zip(regionLabels,activePreSdfs,activeChangeSdfs,passivePreSdfs,passiveChangeSdfs):
    activePre,activeChange,passivePre,passiveChange = [d-d[:,baseWin].mean(axis=1)[:,None] for d in (activePre,activeChange,passivePre,passiveChange)]
    fig = plt.figure(figsize=(8,8))
    ylim = None
    for i,(pre,change,clr,title) in enumerate(zip((activePre,passivePre),(activeChange,passiveChange),([1,0,0],[0,0,1]),('Active','Passive'))):
        ax = fig.add_subplot(len(behavStates),1,i+1)
        clrlight = np.array(clr).astype(float)
        clrlight[clrlight==0] = 0.7
        diff = change-pre
        for d,c,lbl in zip((pre,change,change-pre),(clrlight,clr,[0.5,0.5,0.5]),('Pre','Change','Diff')):
            m = d.mean(axis=0)
            s = d.std(axis=0)/(len(d)**0.5)
            ax.plot(m,color=c,label=lbl)
            ax.fill_between(np.arange(len(m)),m+s,m-s,color=c,alpha=0.25) 
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([250,600])
        ax.set_xticks([250,350,450,550])
        ax.set_xticklabels([0,100,200,300,400])
        if ylim is None:
            ylim = plt.get(ax,'ylim')
        else:
            ax.set_ylim(ylim)
        ax.set_ylabel('Spikes/s')
        ax.set_title(region+' '+title)
        ax.legend()

    fig = plt.figure(figsize=(8,8))
    ylim = None
    for i,(active,passive,title) in enumerate(zip((activeChange,activePre),(passiveChange,passivePre),('Change','Pre'))):
        ax = fig.add_subplot(len(behavStates),1,i+1)
        for d,clr,lbl in zip((active,passive),'rb',('Active','Passive')):
            m = d.mean(axis=0)
            s = d.std(axis=0)/(len(d)**0.5)
            ax.plot(m,color=clr,label=lbl)
            ax.fill_between(np.arange(len(m)),m+s,m-s,color=clr,alpha=0.25) 
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([250,600])
        ax.set_xticks([250,350,450,550])
        ax.set_xticklabels([0,100,200,300,400])
        if ylim is None:
            ylim = plt.get(ax,'ylim')
        else:
            ax.set_ylim(ylim)
        ax.set_ylabel('Spikes/s')
        ax.set_title(region+' '+title)
        ax.legend()
        
# another representation of sdfs
cortical_cmap = plt.cm.plasma
subcortical_cmap = plt.cm.Reds
regionsToUse = (('LGd',(0,0,0)),
                ('V1',cortical_cmap(0)),
                ('LM',cortical_cmap(0.1)),
                ('RL',cortical_cmap(0.2)),
                ('AL',cortical_cmap(0.3)),
                ('PM',cortical_cmap(0.4)),
                ('AM',cortical_cmap(0.5)),
                ('LP',subcortical_cmap(0.4)),
                ('APN',subcortical_cmap(0.5)),
                ('SCd',subcortical_cmap(0.6)),
                ('MB',subcortical_cmap(0.7)),
                ('MRN',subcortical_cmap(0.8)),
                ('SUB',subcortical_cmap(0.9)),
                ('hipp',subcortical_cmap(1.0)))
regionsToUse = regionsToUse[:8]

spacing = -0.1
for i,(sdfs,lbl) in enumerate(zip((activeChangeSdfs,activePreSdfs,passiveChangeSdfs,passivePreSdfs),('Active Change','Active Pre','Passive Change','Passive Pre'))):
    fig = plt.figure(figsize=(4.5,10))
    ax = fig.subplots(1)
    if i==0:
        y = 0
        yticks = []
        norm = []
    for j,(region,clr) in enumerate(regionsToUse[::-1]):
        ind = regionLabels.index(region)
        d = sdfs[ind]-sdfs[ind][:,baseWin].mean(axis=1)[:,None]
        m = d.mean(axis=0)
        s = d.std(axis=0)/(len(sdfs[ind])**0.5)
        if i==0:
            yticks.append(y)
            norm.append(m.max())
        m /= norm[j]
        s /= norm[j]
        m += yticks[j]
        ax.plot(m,color=clr)
        ax.fill_between(np.arange(len(m)),m+s,m-s,color=clr,alpha=0.25)
        if i==0:
            y = np.max(m+s)+spacing
            ymax = y
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([250,600])
        ax.set_xticks([250,350,450,550])
        ax.set_xticklabels([0,100,200,300,400])
        ax.set_xlabel('Time (ms)')
        ax.set_ylim([-0.1,ymax-spacing+0.1])
        ax.set_yticks(yticks)
        ax.set_yticklabels([r[0] for r in regionsToUse[::-1]])
        ax.set_title(lbl)
    plt.tight_layout()

    
fig = plt.figure(figsize=(3,10))
ax = fig.subplots(1)
y = 0
yticks = []
norm = []
xlim = [250,500]
for j,(region,clr) in enumerate(regionsToUse[::-1]):
    ind = regionLabels.index(region)
    for i,(sdfs,lineStyle) in enumerate(zip((activeChangeSdfs[ind],activePreSdfs[ind]),('-','--'))):
        d = sdfs-sdfs[:,baseWin].mean(axis=1)[:,None]
        m = d.mean(axis=0)
        s = d.std(axis=0)/(len(sdfs)**0.5)
        if i==0:
            yticks.append(y)
            norm.append(m.max())
        m /= norm[j]
        s /= norm[j]
        m += yticks[j]
        c,lw,alpha = ('0.4',3,1) if i==2 else (clr,1,1)
        z = -(i-2)
        ax.plot(m,lineStyle,color=c,lineWidth=lw,alpha=alpha,zorder=z)
        ax.fill_between(np.arange(len(m)),m+s,m-s,color=c,alpha=0.25,zorder=z+1)
        if i==0:
            y = np.max(m+s)+spacing
            ymax = y
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([250,350,450])
    ax.set_xticklabels([0,100,200])
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks(yticks)
    ax.set_yticklabels([r[0] for r in regionsToUse[::-1]])
    ax.set_ylim([-0.1,ymax-spacing+0.1])
plt.tight_layout()

fig = plt.figure(figsize=(3,10))
ax = fig.subplots(1)
y = 0
yticks = []
norm = []
xlim = [250,500]
for j,(region,clr) in enumerate(regionsToUse[::-1]):
    ind = regionLabels.index(region)
    for i,(sdfs,lineStyle) in enumerate(zip((activeChangeSdfs[ind],passiveChangeSdfs[ind]),('-','--'))):
        d = sdfs-sdfs[:,baseWin].mean(axis=1)[:,None]
        m = d.mean(axis=0)
        s = d.std(axis=0)/(len(sdfs)**0.5)
        if i==0:
            yticks.append(y)
            norm.append(m.max())
        m /= norm[j]
        s /= norm[j]
        m += yticks[j]
        c,lw,alpha = ('0.4',3,1) if i==2 else (clr,1,1)
        z = -(i-2)
        ax.plot(m,lineStyle,color=c,lineWidth=lw,alpha=alpha,zorder=z)
        ax.fill_between(np.arange(len(m)),m+s,m-s,color=c,alpha=0.25,zorder=z+1)
        if i==0:
            y = np.max(m+s)+spacing
            ymax = y
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([250,350,450])
    ax.set_xticklabels([0,100,200])
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks(yticks)
    ax.set_yticklabels([r[0] for r in regionsToUse[::-1]])
    ax.set_ylim([-0.1,ymax-spacing+0.1])
plt.tight_layout()


# plot baseline rate, change resp, change mod, and behav mod
ind = [regionLabels.index(r[0]) for r in regionsToUse]
xticks = np.arange(len(regionsToUse))
for param,ylab,lbl in zip(((baseRateActive,baseRatePassive),(changeRespActive,changeRespPassive),(changeModActive,changeModPassive),(behavModChange,behavModPre)),
                          (('Baseline Rate (spikes/s)','Change Response (spikes/s)','Change Modulation','Behavior Modulation')),
                          ((('Active','Passive'),)*3+(('Change','Pre'),))):
    fig = plt.figure(facecolor='w',figsize=(15,8))
    ax = fig.subplots(1)
    xlim = [-0.5,len(regionsToUse)-0.5]
    for p,fill,l in zip(param,[True,False],lbl):
        p = [p[i] for i in ind]
        mean = [np.nanmean(d) for d in p]
        sem = [np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in p]
        for i,(x,m,s,clr) in enumerate(zip(xticks,mean,sem,[r[1] for r in regionsToUse])):
            l = l if i==0 else None
            mfc = clr if fill else 'none'
            ax.plot(x,m,'o',mec=clr,mfc=mfc,ms=10,label=l)
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionsToUse,nUnits[ind],nExps[ind],nMice[ind])],fontsize=8)
    ylim = plt.get(ax,'ylim')
    if ylim[0]>0:
        ax.set_ylim([0,ylim[1]])
    else:
        ax.plot(xlim,[0,0],'--',color='0.5')
    ax.set_ylabel(ylab,fontsize=8)
    ax.legend()
    
# plot behav mod of change resp alone
fig = plt.figure(facecolor='w',figsize=(15,8))
ax = fig.subplots(1)
p = [behavModChange[i] for i in ind]
mean = [np.nanmean(d) for d in p]
sem = [np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in p]
for x,m,s,clr in zip(xticks,mean,sem,[r[1] for r in regionsToUse]):
    ax.plot(x,m,'o',mec=clr,mfc=clr,ms=10)
    ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionsToUse,nUnits[ind],nExps[ind],nMice[ind])],fontsize=8)
ylim = plt.get(ax,'ylim')
if ylim[0]>0:
    ax.set_ylim([0,ylim[1]])
else:
    ax.plot(xlim,[0,0],'--',color='0.5')
ax.set_ylabel('Behavior Modulation',fontsize=8)

# plot pre and change resp together
fig = plt.figure(facecolor='w',figsize=(15,8))
ax = fig.subplots(1)
for param,clr,lbl in zip((preRespActive,changeRespActive,preRespPassive,changeRespPassive),'rrbb',('Active Pre','Active Change','Passive Pre','Passive Change')):
    param = [param[i] for i in ind]
    mean = [d.mean() for d in param]
    sem = [d.std()/(d.size**0.5) for d in param]
    mfc = clr if 'Change' in lbl else 'none'
    ax.plot(xticks,mean,'o',mec=clr,mfc=mfc,ms=12,label=lbl)
    for x,m,s in zip(xticks,mean,sem): 
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionsToUse,nUnits[ind],nExps[ind],nMice[ind])],fontsize=8)
ylim = plt.get(ax,'ylim')
ax.set_ylim([0,ylim[1]])
ax.set_ylabel('Response (spikes/s)',fontsize=8)
ax.legend()

# plot resp and change mod latency
fig = plt.figure(facecolor='w',figsize=(15,8))
ax = fig.subplots(1)
for param,clr,lbl in zip((respLatActive,changeModLatActive),('k','0.5'),('Visual Response','Change Modulation')):
    param = [param[i] for i in ind]
    mean = [np.nanmean(d) for d in param]
    sem = [np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in param]
    for i,(x,m,s,clr) in enumerate(zip(xticks,mean,sem,[r[1] for r in regionsToUse])):
        mfc = 'none' if 'Change' in lbl else clr
        l = lbl if i==0 else None
        ax.plot(x,m,'o',mec=clr,mfc=mfc,ms=10,label=l)
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionsToUse,nUnits[ind],nExps[ind],nMice[ind])],fontsize=8)
ax.set_ylabel('Latency (ms)',fontsize=8)
ax.legend()
    
# plot pop resp and change mod latency
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for lat,mec,mfc,lbl in zip((popRespLatActive,popChangeModLatActive),'kk',('k','none'),('visual response','change mod')):
    ax.plot(lat[ind],'o',mec=mec,mfc=mfc,ms=10,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels([r[0] for r in regionsToUse])
ax.set_ylabel('Pop Resp Latency (ms)',fontsize=10)
ax.legend()


# make dataframe for Josh
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
    

# plot parameters vs hierarchy score
anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionLabels for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a==r] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')]
    
hier = hierScore_8regions

paramLabels = ('Change Modulation Index','Time to first spike after image change (ms)','Baseline Rate (spikes/s)','Pre-change Response (spikes/s)','Change Response (spikes/s)')
for param,lbl in zip((changeModActive,activeFirstSpikeLat,baseRateActive,preRespActive,changeRespActive),paramLabels):
    fig = plt.figure(facecolor='w',figsize=(8,6))
    ax = plt.subplot(1,1,1)
    m = [np.nanmean(reg) for reg in param]
    ci = [np.percentile([np.nanmean(np.random.choice(reg,len(reg),replace=True)) for _ in range(5000)],(2.5,97.5)) for reg in param]
    ax.plot(hier,m,'ko',ms=6)
    for h,c in zip(hier,ci):
        ax.plot([h,h],c,'k')
    slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,m)
    x = np.array([min(hier),max(hier)])
    ax.plot(x,slope*x+yint,'0.5')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xticks(hier)
    ax.set_xticklabels([str(round(h,2))+'\n'+r[0]+'\n'+str(nu)+'\n'+str(nm) for h,r,nu,nm in zip(hier,regionNames,nUnits,nMice)])
    ax.set_xlabel('Hierarchy Score',fontsize=10)
    ax.set_ylabel(lbl,fontsize=10)
    r,p = scipy.stats.pearsonr(hier,m)
    title = 'Pearson: r = '+str(round(r,2))+', p = '+str(round(p,3))
    r,p = scipy.stats.spearmanr(hier,m)
    title += '\nSpearman: r = '+str(round(r,2))+', p = '+str(round(p,3))
    ax.set_title(title,fontsize=8)
    plt.tight_layout()
    
# distributions of parameter values
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))
for param,lbl in zip((changeModActive,activeFirstSpikeLat,baseRateActive,preRespActive,changeRespActive),paramLabels):
    fig = plt.figure(facecolor='w',figsize=(8,6))
    ax = plt.subplot(1,1,1)
    for d,clr,r in zip(param,regionColors,regionLabels):
        d = d[~np.isnan(d)]
        sortd = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in sortd]
        ax.plot(sortd,cumProb,color=clr,label=r)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_yticks([0,0.5,1])
    ax.set_xlabel(lbl)
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    plt.tight_layout()

# p value matrix
alpha = 0.05
for metric,lbl in zip((cmiActive,),('Change modulation',)):
    comparison_matrix = np.zeros((len(regionLabels),)*2) 
    for i,region in enumerate(regionLabels):
        for j,region in enumerate(regionLabels):
            if j > i:
                v1 = metric[i]
                v2 = metric[j]
                z, comparison_matrix[i,j] = ranksums(v1[np.invert(np.isnan(v1))],
                                                     v2[np.invert(np.isnan(v2))])
            
    p_values = comparison_matrix.flatten()
    ok_inds = np.where(p_values > 0)[0]
    
    reject, p_values_corrected, alphaSidak, alphacBonf = multipletests(p_values[ok_inds], alpha=alpha, method='fdr_bh')
            
    p_values_corrected2 = np.zeros((len(p_values),))
    p_values_corrected2[ok_inds] = p_values_corrected
    comparison_corrected = np.reshape(p_values_corrected2, comparison_matrix.shape)
    
    sig_thresh = np.log10(alpha)
    plot_range = 10
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    im = ax.imshow(np.log10(comparison_matrix),cmap='seismic',vmin=sig_thresh-plot_range,vmax=sig_thresh+plot_range)
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(len(regionLabels)))
    ax.set_xticklabels(regionLabels)
    ax.set_yticks(np.arange(len(regionLabels)))
    ax.set_yticklabels(regionLabels)
    ax.set_ylim([-0.5,len(regionLabels)-0.5])
    ax.set_xlim([-0.5,len(regionLabels)-0.5])
    ax.set_title(lbl+' p-values')
    plt.tight_layout()
    


###### adaptation

regionLabels = ('LGd','VISp','VISl','VISal','VISrl','VISpm','VISam','LP')
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))

nFlashes = 10

betweenChangeSdfs = {region: {state:[] for state in ('active','passive')} for region in regionLabels}
for exp in exps:
    print(exp)
    behaviorChangeTimes = data[exp]['behaviorChangeTimes'][:]
    passiveChangeTimes = data[exp]['passiveChangeTimes'][:]
    for region in regionLabels:
        for probe in data[exp]['regions']:
            inRegion = data[exp]['regions'][probe][:]==region
            if any(inRegion):
                (hasSpikesActive,hasRespActive),(hasSpikesPassive,hasRespPassive) = [findResponsiveUnits(data[exp]['sdfs'][probe][state]['change'][:][inRegion].mean(axis=1),baseWin,respWin,thresh=5) for state in ('active','passive')]
                uindex = np.where(inRegion)[0][hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)]
                for u in uindex:
                    spikeTimes = data[exp]['spikeTimes'][probe][str(u)][:]
                    for state,changeTimes in zip(('active','passive'),(behaviorChangeTimes,passiveChangeTimes)):
                        betweenChangeSdfs[region][state].append(analysis_utils.getSDF(spikeTimes,changeTimes-0.25,0.25+nFlashes*0.75,sampInt=0.001,filt='exp',sigma=0.005,avg=True)[0])

adaptMean = {state:[] for state in ('active','passive')}
adaptSem = {state:[] for state in ('active','passive')}
t = np.arange(-0.25,nFlashes*0.75,0.001)
for state in ('active','passive'):
    fig = plt.figure(facecolor='w',figsize=(8,6))
    fig.suptitle(state)
    ax = fig.subplots(2,1)
    for region,clr in zip(regionLabels,regionColors):
        sdfs = np.stack(betweenChangeSdfs[region][state])
        sdfs -= sdfs[:,:250].mean(axis=1)[:,None]
        m = sdfs.mean(axis=0)
        s = sdfs.std()/(len(sdfs)**0.5)
        s /= m.max()
        m /= m.max()
        ax[0].plot(t,m,color=clr,label=region)
        ax[0].fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        
        flashResp = []
        flashRespSem = []
        for i in np.arange(250,nFlashes*750,750):
            r = sdfs[:,i:i+250].max(axis=1)
            r -= sdfs[:,i-250:i].mean(axis=1)
            flashResp.append(r.mean())
            flashRespSem.append(r.std()/(len(r)**0.5))
        flashResp,flashRespSem = [np.array(r)/flashResp[0] for r in (flashResp,flashRespSem)]
        ax[1].plot(flashResp,color=clr,marker='o')
        for x,(m,s) in enumerate(zip(flashResp,flashRespSem)):
            ax[1].plot([x,x],[m-s,m+s],color=clr)
            
        adaptMean[state].append(flashResp[-1])
        adaptSem[state].append(flashRespSem[-1])
    
    for a in ax:
        for side in ('right','top'):
            a.spines[side].set_visible(False)
        a.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax[0].set_xlabel('Time after change (s)')
    ax[0].set_ylabel('Normalized spike rate')
    ax[0].legend(loc='upper right')
    ax[1].set_xlabel('Flash after change')
    ax[1].set_ylabel('Normalized peak response')
    ax[1].set_ylim([0.41,1.09])

fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
for state,clr in zip(('active','passive'),'rb'):
    ax.plot(adaptMean[state],'o',mec=clr,mfc='none',ms=10,label=state)
    for x,(m,s) in enumerate(zip(adaptMean[state],adaptSem[state])):
        ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim([-0.5,len(regionLabels)-0.5])
    ax.set_xticks(np.arange(len(regionLabels)))
    ax.set_xticklabels(regionLabels)
    ax.set_ylabel('Adaptation (fraction of change reseponse)',fontsize=10)
ax.legend()



###### decoding analysis

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
                ('APN',('APN',),subcortical_cmap(0.5)),
                ('SCd',('SCig','SCig-a','SCig-b'),subcortical_cmap(0.6)),
#                ('MB',('MB',),subcortical_cmap(0.7)),
                ('MRN',('MRN',),subcortical_cmap(0.8)),
#                ('SUB',('SUB','PRE','POST'),subcortical_cmap(0.9)),
                ('hipp',('CA1','CA3','DG-mo','DG-po','DG-sg','HPF'),subcortical_cmap(1.0)))
regionsToUse = regionsToUse[:8]
regionLabels = [r[0] for r in regionsToUse]
regionColors = [r[2] for r in regionsToUse]
    
unitSampleSize = [20]

nCrossVal = 5

baseWin = slice(stimWin.start-150,stimWin.start)
respWin = slice(stimWin.start,stimWin.start+150)

decodeWindowSize = 10
decodeWindows = []#np.arange(stimWin.start,stimWin.start+150,decodeWindowSize)

preImageDecodeWindowSize = 50
preImageDecodeWindows = []#np.arange(stimWin.start,stimWin.start+750,preImageDecodeWindowSize)

# models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4),LinearDiscriminantAnalysis(),SVC(kernel='linear',C=1.0,probability=True)))
# modelNames = ('randomForest','supportVector','LDA')
models = (RandomForestClassifier(n_estimators=100),)
modelNames = ('randomForest',)

behavStates = ('active',)

# add catchScore, catchPrediction, reactionScore
result = {exp: {region: {state: {'changeScore':{model:[] for model in modelNames},
                                 'changePredict':{model:[] for model in modelNames},
                                 'changePredictProb':{model:[] for model in modelNames},
                                 'changeFeatureImportance':{model:[] for model in modelNames},
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
    
    initialImage = data[exp]['initialImage'][changeTrials]
    changeImage = data[exp]['changeImage'][changeTrials]
    imageNames = np.unique(changeImage)
    result[exp]['preChangeImage'] = initialImage
    result[exp]['changeImage'] = changeImage
    result[exp]['catchImage'] = data[exp]['changeImage'][catchTrials]
    
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
                    activeNonChangeSDFs.append([analysis_utils.getSDF(spikes[u],nonChangeFlashTimes,0.15,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0] for u in units])
                    if 'passive' in behavStates:
                        passiveNonChangeSDFs.append([analysis_utils.getSDF(spikes[u],passiveNonChangeFlashTimes,0.15,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0] for u in units])
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
                        if sampleSize==unitSampleSize[-1]:
                            meanSDF = activeChangeSDFs.mean(axis=1)
                            result[exp][region][state]['preChangeSDFs'] = preChangeSDFs[changeTrials][:,:,respWin].mean(axis=1) 
                            result[exp][region][state]['changeSDFs'] = changeSDFs[changeTrials][:,:,respWin].mean(axis=1)
                            result[exp][region][state]['catchSDFs'] = changeSDFs[catchTrials][:,:,respWin].mean(axis=1)
                            result[exp][region][state]['nonChangeSDFs'] = nonChangeSDFs.mean(axis=1)
                            result[exp][region][state]['respLatency'] = findLatency(changeSDFs[changeTrials][:,:,respWin].mean(axis=(0,1))[None,:],baseWin,stimWin,method='abs',thresh=0.5)[0]
                        
#                        changeScore = {model: [] for model in modelNames}
#                        changePredict = {model: [] for model in modelNames}
#                        changePredictProb = {model: [] for model in modelNames}
#                        changeFeatureImportance = {model: np.full((nsamples,nUnits,respWin.stop-respWin.start),np.nan) for model in modelNames}
#                        catchPredict = {model: [] for model in modelNames}
#                        catchPredictProb = {model: [] for model in modelNames}
#                        nonChangePredict = {model: [] for model in modelNames}
#                        nonChangePredictProb = {model: [] for model in modelNames}
#                        imageScore = {model: [] for model in modelNames}
#                        imageFeatureImportance = {model: np.full((nsamples,nUnits,respWin.stop-respWin.start),np.nan) for model in modelNames}
#                        changeScoreWindows = {model: np.zeros((nsamples,len(decodeWindows))) for model in modelNames}
#                        changePredictWindows = {model: np.zeros((nsamples,len(decodeWindows),changeTrials.sum())) for model in modelNames}
#                        imageScoreWindows = {model: np.zeros((nsamples,len(decodeWindows))) for model in modelNames}
#                        preImageScoreWindows = {model: np.zeros((nsamples,len(preImageDecodeWindows))) for model in modelNames}
                        
#                        for i,unitSamp in enumerate(unitSamples):
#                            # decode image change and identity for full respWin
#                            # image change
#                            X = np.concatenate([s[:,unitSamp,respWin][changeTrials].reshape((changeTrials.sum(),-1)) for s in (changeSDFs,preChangeSDFs)])
#                            y = np.zeros(X.shape[0])
#                            y[:int(X.shape[0]/2)] = 1
#                            Xcatch = changeSDFs[:,unitSamp,respWin][catchTrials].reshape((catchTrials.sum(),-1))
#                            Xnonchange = nonChangeSDFs[:,unitSamp].reshape((nonChangeSDFs.shape[0],-1))
#                            for model,name in zip(models,modelNames):
#                                cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
#                                changeScore[name].append(cv['test_score'].mean())
#                                changePredict[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict')[:changeTrials.sum()])
#                                catchPredict[name].append(scipy.stats.mode([estimator.predict(Xcatch) for estimator in cv['estimator']],axis=0)[0].flatten())
#                                nonChangePredict[name].append(scipy.stats.mode([estimator.predict(Xnonchange) for estimator in cv['estimator']],axis=0)[0].flatten())
#                                if name=='randomForest':
#                                    changePredictProb[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:changeTrials.sum(),1])
#                                    changeFeatureImportance[name][i][unitSamp] = np.mean([np.reshape(estimator.feature_importances_,(sampleSize,-1)) for estimator in cv['estimator']],axis=0)
#                                    catchPredictProb[name].append(np.mean([estimator.predict_proba(Xcatch)[:,1] for estimator in cv['estimator']],axis=0))
#                                    nonChangePredictProb[name].append(np.mean([estimator.predict_proba(Xnonchange)[:,1] for estimator in cv['estimator']],axis=0))
                            # image identity
#                            imgSDFs = [changeSDFs[:,unitSamp,respWin][changeTrials & (changeImage==img)] for img in imageNames]
#                            X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
#                            y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
#                            for model,name in zip(models,modelNames):
#                                cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
#                                imageScore[name].append(cv['test_score'].mean())
#                                if name=='randomForest':
#                                    imageFeatureImportance[name][i][unitSamp] = np.mean([np.reshape(estimator.feature_importances_,(sampleSize,-1)) for estimator in cv['estimator']],axis=0)
                            
                            # decode image change and identity for sliding windows
#                            for j,winStart in enumerate(decodeWindows):
#                                # image change
#                                winSlice = slice(winStart,winStart+decodeWindowSize)
#                                X = np.concatenate([s[:,unitSamp,winSlice][changeTrials].reshape((changeTrials.sum(),-1)) for s in (changeSDFs,preChangeSDFs)])
#                                y = np.zeros(X.shape[0])
#                                y[:int(X.shape[0]/2)] = 1
#                                for model,name in zip(models,modelNames):
#                                    changeScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
#                                    if name=='randomForest':
#                                        changePredictWindows[name][i,j] = cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:changeTrials.sum(),1]
                                # image identity
#                                imgSDFs = [changeSDFs[:,unitSamp,winSlice][changeTrials & (changeImage==img)] for img in imageNames]
#                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
#                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
#                                for model,name in zip(models,modelNames):
#                                    imageScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                    
                            # decode pre-change image identity for sliding windows
#                            for j,winStart in enumerate(preImageDecodeWindows):
#                                winSlice = slice(winStart,winStart+preImageDecodeWindowSize)
#                                preImgSDFs = [preChangeSDFs[:,unitSamp,winSlice][changeTrials & (initialImage==img)] for img in imageNames]
#                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in preImgSDFs])
#                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(preImgSDFs)])
#                                for model,name in zip(models,modelNames):
#                                    preImageScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                        
                        # average across unit samples
#                        for model in modelNames:
#                            result[exp][region][state]['changeScore'][model].append(np.median(changeScore[model],axis=0))
#                            result[exp][region][state]['changePredict'][model].append(scipy.stats.mode(changePredict[model],axis=0)[0].flatten())
#                            result[exp][region][state]['changePredictProb'][model].append(np.median(changePredictProb[model],axis=0))
#                            result[exp][region][state]['changeFeatureImportance'][model].append(np.nanmedian(changeFeatureImportance[model],axis=0))
#                            result[exp][region][state]['catchPredict'][model].append(scipy.stats.mode(catchPredict[model],axis=0)[0].flatten())
#                            result[exp][region][state]['catchPredictProb'][model].append(np.median(catchPredictProb[model],axis=0))
#                            result[exp][region][state]['nonChangePredict'][model].append(scipy.stats.mode(nonChangePredict[model],axis=0)[0].flatten())
#                            result[exp][region][state]['nonChangePredictProb'][model].append(np.median(nonChangePredictProb[model],axis=0))
#                            result[exp][region][state]['imageScore'][model].append(np.median(imageScore[model],axis=0))
#                            result[exp][region][state]['imageFeatureImportance'][model].append(np.nanmedian(imageFeatureImportance[model],axis=0))
#                            result[exp][region][state]['changeScoreWindows'][model].append(np.median(changeScoreWindows[model],axis=0))
#                            result[exp][region][state]['changePredictWindows'][model].append(np.median(changePredictWindows[model],axis=0))
#                            result[exp][region][state]['imageScoreWindows'][model].append(np.median(imageScoreWindows[model],axis=0))
#                            result[exp][region][state]['preImageScoreWindows'][model].append(np.median(preImageScoreWindows[model],axis=0))
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
        ax.plot(unitSampleSize,np.nanmean(allScores[score],axis=0),color=clr,label=score[:score.find('S')])
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,100,10))
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_yticklabels([0,'',0.5,'',1])
    ax.set_xlim([0,max(unitSampleSize)+5])
    ax.set_ylim([0,1])
    ax.set_xlabel('Number of Units')
    ax.set_ylabel('Decoder Accuracy')
    ax.set_title(model)
    ax.legend()
    plt.tight_layout()
    
    xticks = np.arange(len(regionLabels))
    xlim = [-0.5,len(regionLabels)-0.5]
    for score,ymin in zip(('changeScore','imageScore'),(0.5,0.125)):
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        for i,(n,clr) in enumerate(zip(unitSampleSize,plt.cm.jet(np.linspace(0,1,len(unitSampleSize))))):
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
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(regionLabels)
        ax.set_ylim([ymin,1])
        ax.set_ylabel('Decoder Accuracy')
        ax.set_title(model+', '+score[:score.find('S')])
        ax.legend()
        plt.tight_layout()
        
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(xlim,[0,0],'--',color='0.5')
for i,(n,clr) in enumerate(zip(unitSampleSize,plt.cm.jet(np.linspace(0,1,len(unitSampleSize))))):
    for j,region in enumerate(regionLabels):
        regionData = []
        for exp in result:
            behavior = result[exp]['behaviorResponse'][:].astype(float)
            s = result[exp][region]['active']['changePredict']['randomForest']
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
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Correlation of decoder prediction and mouse behavior')
ax.legend()
plt.tight_layout()

    
# compare scores for full response window
for model in modelNames:
    for score,ymin,title in zip(('changeScore','imageScore'),(0.5,0.125),('Change','Image Identity')):
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
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

    
# image and change feature importance
x = np.arange(0,respWin.stop-respWin.start)
for model in ('randomForest',):
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for score,clr in zip(('imageFeatureImportance','changeFeatureImportance'),('0.5','k')):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(np.nanmean(s[0],axis=0))
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(x,m,color=clr,label=score[:score.find('F')])
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
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
                    ax.set_title(region+', '+state)
                    ax.set_ylabel('Feature Importance')
                else:
                    ax.set_title(state)
                    ax.legend()
            elif j==0:
                ax.set_title(region)          

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
fig = plt.figure(facecolor='w',figsize=(6,4))
ax = plt.subplot(1,1,1)
xticks = np.arange(len(regionLabels))
xlim = [-0.5,len(regionLabels)-0.5]
ax.plot(xlim,[0,0],'--',color='0.5')
for state,fill in zip(('active',),(True,)):
    for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
        regionData = []
        for exp in result:
            behavior = result[exp]['responseToNonChange'].astype(float)
            s = result[exp][region][state]['changePredictProb']['randomForest']
            if len(s)>0 and any(behavior) and any(s[-1]):
                regionData.append(np.corrcoef(behavior,s[-1])[0,1])
        n = len(regionData)
        if n>0:
            m = np.mean(regionData)
            s = np.std(regionData)/(n**0.5)
            mfc = clr if fill else 'none'
            lbl = state if i==0 else None
            ax.plot(i,m,'o',mec=clr,mfc=mfc,label=lbl)
            ax.plot([i,i],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Correlation of decoder prediction and mouse behavior')
ax.legend()
plt.tight_layout()

anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionsToUse for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a==r[1][0]] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')] 
hier = hierScore_8regions

fig = plt.figure(facecolor='w',figsize=(6,4))
ax = plt.subplot(1,1,1)
for state,fill in zip(('active',),(True,)):
    meanRegionData = []
    for i,(region,clr,h) in enumerate(zip(regionLabels,regionColors,hier)):
        regionData = []
        for exp in result:
            behavior = result[exp]['responseToChange'].astype(float)
            s = result[exp][region][state]['changePredictProb']['randomForest']
            if len(s)>0 and any(behavior) and any(s[0]):
                regionData.append(np.corrcoef(behavior,s[0])[0,1])
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
    ax.plot(x,slope*x+yint,'0.5')
    r,p = scipy.stats.pearsonr(hier,meanRegionData)
    title = 'Pearson: r = '+str(round(r,2))+', p = '+str(round(p,3))
    r,p = scipy.stats.spearmanr(hier,meanRegionData)
    title += '\nSpearman: r = '+str(round(r,2))+', p = '+str(round(p,3))
    ax.set_title(title,fontsize=8)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Hierarchy score')
ax.set_ylabel('Correlation of decoder prediction and mouse behavior')
plt.tight_layout()

fig = plt.figure(facecolor='w',figsize=(6,4))
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
    ax.plot(x,slope*x+yint,'0.5')
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


for model in ('randomForest',):    
    fig = plt.figure(facecolor='w',figsize=(6,10))
    fig.text(0.5,1,'Correlation of decoder prediction and mouse behavior',fontsize=10,horizontalalignment='center',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,(region,clr) in enumerate(zip(regionLabels,regionColors)):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            regionData = []
            for exp in result:
                behavior = result[exp]['behaviorResponse'][:].astype(float)
                s = result[exp][region][state]['changePredictWindows'][model]
                if len(s)>0:
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

plt.plot(np.mean(hitRate,axis=0),np.mean(falseAlarmRate,axis=0),'ko')

plt.figure()
plt.plot(np.mean(hitReactionTime,axis=0),np.mean(falseAlarmReactionTime,axis=0),'ko')

plt.figure()
plt.plot(np.mean(hitReactionTime,axis=0),np.mean(falseAlarmReactionTime,axis=0),'ko')

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
    




###### hit rate, lick time, and change mod correlation
                 
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im] for im in 'AB']
regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')

respWin = slice(stimWin.start,stimWin.start+151)

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
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.01,0.99,region+' ('+str(n)+' experiments)',fontsize=8,horizontalalignment='left',verticalalignment='top')
    gs = matplotlib.gridspec.GridSpec(4,4)
    
    for j,(d,lbl) in enumerate(zip((ntrials,hr,lickLat),('Number of Trials','Hit Rate','Hit Lick Latency (ms)'))):
        ax = plt.subplot(gs[0,j])
        im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
        ax.tick_params(direction='out',top=False,right=False,labelsize=6)
        ax.set_xticks(np.arange(len(imgNames)))
        ax.set_yticks(np.arange(len(imgNames)))
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
    for i,(d,ylbl) in enumerate(zip((respLat,preResp,changeResp),('Response Latency (ms)','Change Modulaton Latency (ms)','Change Modulation Index'))):
        ax = plt.subplot(gs[i+1,0])
        im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
        ax.tick_params(direction='out',top=False,right=False,labelsize=6)
        ax.set_xticks(np.arange(len(imgNames)))
        ax.set_yticks(np.arange(len(imgNames)))
        ax.set_xlabel('Change Image',fontsize=8)
        ax.set_ylabel('Initial Image',fontsize=8)
        ax.set_title(ylbl,fontsize=10)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(labelsize=6)
        
        for j,(b,xlbl) in enumerate(zip((hr,lickLat),('Hit Rate','Hit Lick Latency (ms)'))):   
            ax = plt.subplot(gs[i+1,j+1])
            notnan = ~np.isnan(d)
            ax.plot(b[notnan],d[notnan],'ko')
            slope,yint,rval,pval,stderr = scipy.stats.linregress(b[notnan],d[notnan])
            x = np.array([b[notnan].min(),b[notnan].max()])
            ax.plot(x,slope*x+yint,'0.5')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            if i==2:
                ax.set_xlabel(xlbl,fontsize=10)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=10)
            r,p = scipy.stats.pearsonr(b[notnan],d[notnan])
            ax.set_title('r = '+str(round(r,2))+', p = '+'{0:1.1e}'.format(p),fontsize=8)
    plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
xticks = np.arange(len(regionLabels))
xlim = [-0.5,len(regionLabels)-0.5]
for param,clr,lbl in zip((changeModMatrix,diffLatMatrix,),'kk',('Change Modulation Index','Change Modulation Latency')):
    r = [scipy.stats.pearsonr(hr[~np.isnan(d)],d[~np.isnan(d)])[0] for hr,d in zip(hitLickLatency,param)]
    mfc = 'none' if 'Latency' in lbl else clr
    ax.plot(xticks,np.absolute(r),'o',ms=10,mec=clr,mfc=mfc,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels,fontsize=10)
ax.set_xlim(xlim)
ax.set_ylim([0.4,0.65])
ax.set_ylabel('Correlation with hit rate',fontsize=10)
ax.legend(loc='upper left')
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
ylim = [-10,5]
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(2,1,1)
for lickProb,clr,lbl in zip((hitLickProb,falseAlarmLickProb),'kg',('hit','false alarm')):
    m = np.nanmean(lickProb,axis=0)
    n = len(lickProb)
    s = np.std(lickProb,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim(xlim)
ax.set_xlabel('Time from change/catch (ms)',fontsize=12)
ax.set_ylabel('Lick Probability',fontsize=12)

ax = fig.add_subplot(2,1,2)
#ax.plot([130]*2,ylim,'k:')
#ax.text(130,5.1,'Reaction time\n~130 ms',horizontalalignment='center')
for speed,clr,lbl in zip((hitRunSpeed['behavior'],missRunSpeed['behavior'],falseAlarmRunSpeed['behavior'],correctRejectRunSpeed['behavior']),'krgb',('hit','miss','false alarm','correct reject')):
    m = np.mean(speed,axis=0)
    n = len(speed)
    s = np.std(speed,axis=0)/(n**0.5)
    ax.plot(plotTime,m,clr,label=lbl)
    ax.fill_between(plotTime,m+s,m-s,color=clr,alpha=0.1)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_yticks(np.arange(-10,10,5))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xticklabels([])
ax.set_xlabel('Time from change/catch (ms)',fontsize=12)
ax.set_ylabel('$\Delta$ Run speed (cm/s)',fontsize=12)
ax.legend(loc='lower left',fontsize=12)


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
t = bins-0.375
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
        ax1.plot(t,lick/(lick+nolick),clr,label=event)
        ax2.plot(t,lick+nolick,clr)
    for ax in (ax1,ax2):
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
    ax1.set_ylabel('Lick Prob.')
    ax2.set_ylabel('N')
    ax2.set_xlabel('Time since change (s)')
    ax1.legend()

fig = plt.figure(facecolor='w',figsize=(6,8))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
for d,clr,lbl in zip((hitRate,falseAlarmRate),'gr',('hit rate','false alarm rate')):
    m = np.mean(d)
    s = np.std(d)/(len(d)**0.5)
    ax1.plot([0,60],[m,m],'--',color=clr,label=lbl)
    ax.fill_between([0,60],[m+s]*2,[m-s]*2,color=clr,alpha=0.25)
for event,clr in zip(flashType[::-1],'gr'):
    lickProb = np.zeros((len(exps),len(bins)))
    n = lickProb.copy()
    for i,(times,licks) in enumerate(zip(timeSinceChange[event],lickInWindow[event])):
        timeToBinIndex = np.searchsorted(bins,times)
        for j in range(len(bins)):
            ind = np.where(timeToBinIndex==j)[0]
            lickProb[i,j] = licks[ind].sum()/len(ind)
            n[i,j] = len(ind)
    for d,ax in zip((lickProb,n),(ax1,ax2)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(exps)**0.5)
        ax.plot(t,m,clr,label=event)
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
for ax in (ax1,ax2):
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
ax1.set_ylabel('Lick Prob.')
ax2.set_ylabel('N')
ax2.set_xlabel('Time since change (s)')
ax1.legend()
plt.tight_layout()


fig = plt.figure(facecolor='w',figsize=(6,8))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
for d,clr,lbl in zip((hitRate,falseAlarmRate),'gr',('hit rate','false alarm rate')):
    m = np.mean(d)
    s = np.std(d)/(len(d)**0.5)
    ax1.plot([0,60],[m,m],'--',color=clr,label=lbl)
    ax.fill_between([0,60],[m+s]*2,[m-s]*2,color=clr,alpha=0.25)
for event,clr in zip(flashType[::-1],'gr'):
    lickProb = np.zeros((len(exps),len(bins)))
    n = lickProb.copy()
    for i,(times,licks) in enumerate(zip(timeSinceLick[event],lickInWindow[event])):
        timeToBinIndex = np.searchsorted(bins,times)
        for j in range(len(bins)):
            ind = np.where(timeToBinIndex==j)[0]
            lickProb[i,j] = licks[ind].sum()/len(ind)
            n[i,j] = len(ind)
    for d,ax in zip((lickProb,n),(ax1,ax2)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(exps)**0.5)
        ax.plot(t,m,clr,label=event)
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
for ax in (ax1,ax2):
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,25])
ax1.set_ylabel('Lick Prob.')
ax2.set_ylabel('N')
ax2.set_xlabel('Time since lick (s)')
ax1.legend()
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



