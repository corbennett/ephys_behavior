# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os
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
                if obj.passive_pickle_file is not None:
                    data[expName]['passiveFlashTimes'] = obj.passiveFrameAppearTimes[obj.flashFrames]
                    data[expName]['passiveOmitFlashTimes'] = obj.passiveFrameAppearTimes[obj.omittedFlashFrames]
                    data[expName]['passiveChangeTimes'] = obj.passiveFrameAppearTimes[obj.changeFrames[trials]]
                    data[expName]['passiveRunTime'] = obj.passiveRunTime
                    data[expName]['passiveRunSpeed'] = obj.passiveRunSpeed
                # reward times

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

    
def findLatency(data,baseWin=None,stimWin=None,method='rel',thresh=3,minPtsAbove=30):
    latency = []
    if len(data.shape)<2:
        data = data[None,:]
    if baseWin is not None:
        data = data-data[:,baseWin].mean(axis=1)[:,None]
    if stimWin is None:
        stimWin = slice(0,data.shape[1])
    for d in data:
        if method=='abs':
            ptsAbove = np.where(np.correlate(d[stimWin]>thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        else:
            ptsAbove = np.where(np.correlate(d[stimWin]>d[baseWin].std()*thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        if len(ptsAbove)>0:
            latency.append(ptsAbove[0])
        else:
            latency.append(np.nan)
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
makeSummaryPlots(miceToAnalyze=('461027',))


# make new experiment hdf5s without updating popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=False,miceToAnalyze=('423744',))

# make new experiment hdf5s and add to existing popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True,miceToAnalyze=('461027',))

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
respWinOffset = 30
respWin = slice(stimWin.start+respWinOffset,stimWin.stop+respWinOffset)



###### behavior analysis
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
ax.set_xticklabels(['Change','False Alarm'])
ax.set_xlim([-0.25,1.25])
ax.set_ylim([0,1])
ax.set_ylabel('Response Probability',fontsize=16)
ax.set_title('n = '+str(nMice)+' mice, '+str(len(exps))+' days',fontsize=16)

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
nUnits = np.array([sum([len(s) for s in sdfs]) for sdfs in activeChangeSdfs])
activePreSdfs,activeChangeSdfs = [[np.concatenate(s) for s in sdfs] for sdfs in (activePreSdfs,activeChangeSdfs)]
if 'passive' in behavStates:
    passivePreSdfs,passiveChangeSdfs = [[np.concatenate(s) for s in sdfs] for sdfs in (passivePreSdfs,passiveChangeSdfs)]


# calculate metrics    
preBase,changeBase = [[s[:,baseWin].mean(axis=1) for s in sdfs] for sdfs in (activePreSdfs,activeChangeSdfs)]
preRespActive,changeRespActive = [[s[:,respWin].mean(axis=1)-b for s,b in zip(sdfs,base)] for sdfs,base in zip((activePreSdfs,activeChangeSdfs),(preBase,changeBase))]
baseRateActive = changeBase
changeModActive = [np.clip((change-pre)/(change+pre),-1,1) for pre,change in zip(preRespActive,changeRespActive)]
changeModLatActive = [findLatency(change-pre,baseWin,stimWin) for pre,change in zip(activePreSdfs,activeChangeSdfs)]
popChangeModLatActive = [findLatency(np.mean(change-pre,axis=0),baseWin,stimWin,method='abs',thresh=1)[0] for pre,change in zip(activePreSdfs,activeChangeSdfs)]
respLatActive = [findLatency(sdfs,baseWin,stimWin) for sdfs in activeChangeSdfs]
popRespLatActive = [findLatency(sdfs.mean(axis=0),baseWin,stimWin,method='abs',thresh=1)[0] for sdfs in activeChangeSdfs]

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

spacing = 0.1
for i,(sdfs,lbl) in enumerate(zip((activeChangeSdfs,activePreSdfs,passiveChangeSdfs,passivePreSdfs),('Active Change','Active Pre','Passive Change','Passive Pre'))):
    fig = plt.figure(figsize=(6,10))
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
        ax.set_ylim([-spacing,ymax])
        ax.set_yticks(yticks)
        ax.set_yticklabels([r[0] for r in regionsToUse])
        ax.set_title(lbl)
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
    for p,clr,l in zip(param,'rb',lbl):
        p = [p[i] for i in ind]
        mean = [np.nanmean(d) for d in p]
        sem = [np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in p]
        ax.plot(xticks,mean,'o',mec=clr,mfc='none',ms=10,label=l)
        for x,m,s in zip(xticks,mean,sem): 
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
ax.plot(xticks,mean,'o',mec='k',mfc='none',ms=10)
for x,m,s in zip(xticks,mean,sem): 
    ax.plot([x,x],[m-s,m+s],color='k')
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
ax.plot(xlim,[0,0],'k--')
for param,clr,lbl in zip((preRespActive,changeRespActive,preRespPassive,changeRespPassive),'rrbb',('Active Pre','Active Change','Passive Pre','Passive Change')):
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
ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionNames,nUnits,nExps,nMice)],fontsize=8)
ax.set_ylabel('Response (spikes/s)',fontsize=8)
ax.legend()
    

# plot pop resp latency
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for lat,mec,mfc,lbl in zip((popRespLatActive,popChangeModLatActive),'kk',('k','none'),('visual response','change mod')):
    ax.plot(lat,'o',mec=mec,mfc=mfc,ms=10,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim([-0.5,len(regionLabels)-0.5])
ax.set_xticks(np.arange(len(regionLabels)))
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Pop Resp Latency (ms)',fontsize=10)
ax.legend()
    

# plot parameters vs hierarchy score
anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionLabels for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a==r] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')]
    
hier = hierScore_8regions

paramLabels = ('Change Modulation Index','Time to first spike after image change (ms)','Baseline Rate (spikes/s)','Pre-change Response (spikes/s)','Change Response (spikes/s)')
for param,lbl in zip((cmiActive,firstSpikeLatActive,baseRateActive,preRespActive,changeRespActive),paramLabels):
    fig = plt.figure(facecolor='w',figsize=(8,6))
    ax = plt.subplot(1,1,1)
    m = [np.nanmean(reg) for reg in param]
    ci = [np.percentile([np.nanmean(np.random.choice(reg,reg.size,replace=True)) for _ in range(5000)],(2.5,97.5)) for reg in param]
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
for param,lbl in zip((cmiActive,firstSpikeLatActive,baseRateActive,preRespActive,changeRespActive),paramLabels):
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
    
regionLabels = ('LGd','VISp','VISl','VISal','VISrl','VISpm','VISam','LP')
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))

nUnits = [20]
nUnitSamples = 5
nCrossVal = 3

truncInterval = 5
lastTrunc = 150
truncTimes = np.arange(truncInterval,lastTrunc+1,truncInterval)

preTruncTimes = np.arange(-750,0,50)

windowInterval = 10
decodeWindows = np.arange(0,281,windowInterval)

assert((len(nUnits)>=1 and len(truncTimes)==1) or (len(nUnits)==1 and len(truncTimes)>=1))

# models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4),LinearDiscriminantAnalysis(),SVC(kernel='linear',C=1.0,probability=True)))
# modelNames = ('randomForest','supportVector','LDA')
models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4))
modelNames = ('randomForest','supportVector')

behavStates = ('active','passive')
result = {exp: {region: {state: {'changeScore':{model:[] for model in modelNames},
                                'changePredict':{model:[] for model in modelNames},
                                'changePredictProb':{model:[] for model in modelNames},
                                'changePredictProbValid':{model:[] for model in modelNames},
                                'changeFeatureImportance':{model:[] for model in modelNames},
                                'imageScore':{model:[] for model in modelNames},
                                'imageFeatureImportance':{model:[] for model in modelNames},
                                'preImageScore':{model:[] for model in modelNames},
                                'imageScoreWindows':{model:[] for model in modelNames},
                                'changeScoreWindows':{model:[] for model in modelNames},
                                'respLatency':[]} for state in behavStates} for region in regionLabels} for exp in exps}

warnings.filterwarnings('ignore')
for expInd,exp in enumerate(exps):
    print('experiment '+str(expInd+1)+' of '+str(len(exps)))
    startTime = time.clock()
    
    unitRegions = []
    for probe in data[exp]['sdfs']:
        ccf = data[exp]['ccfRegion'][probe][:]
        isi = data[exp]['isiRegion'][probe][()]
        if isi:
            ccf[data[exp]['inCortex'][probe][:]] = isi
        unitRegions.append(ccf)
    unitRegions = np.concatenate(unitRegions)
    
    response = data[exp]['response'][:]
    hit = response=='hit'
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    trials = engaged & (hit | (response=='miss'))
    result[exp]['behaviorResponse'] = hit[trials]
    
    initialImage = data[exp]['initialImage'][trials]
    changeImage = data[exp]['changeImage'][trials]
    imageNames = np.unique(changeImage)
    
    (activePreSDFs,activeChangeSDFs),(passivePreSDFs,passiveChangeSDFs) = [[np.concatenate([data[exp]['sdfs'][probe][state][epoch][:,trials] for probe in data[exp]['sdfs']])  for epoch in ('preChange','change')] for state in ('active','passive')]
    hasSpikesActive,hasRespActive = findResponsiveUnits(activeChangeSDFs,baseWin,respWin)
    hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChangeSDFs,baseWin,respWin)
    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
    
    for region in regionLabels:
        inRegion = unitRegions==region
        if any(inRegion):
            units = np.where(inRegion & hasResp)[0]
            for n in nUnits:
                if len(units)>=n:
                    unitSamples = [np.random.choice(units,size=n,replace=False) for _ in range(nUnitSamples)]
                    for state in behavStates:
                        changeScore = {model: np.zeros((nUnitSamples,len(truncTimes))) for model in modelNames}
                        changePredict = {model: [] for model in modelNames}
                        changePredictProb = {model: [] for model in modelNames}
                        changePredictProbValid = {model: [] for model in modelNames}
                        changeFeatureImportance = {model: [] for model in modelNames}
                        imageScore = {model: np.zeros((nUnitSamples,len(truncTimes))) for model in modelNames}
                        imageFeatureImportance = {model: [] for model in modelNames}
                        preImageScore = {model: np.zeros((nUnitSamples,len(preTruncTimes))) for model in modelNames}
                        changeScoreWindows = {model: np.zeros((nUnitSamples,len(decodeWindows)-1)) for model in modelNames}
                        imageScoreWindows = {model: np.zeros((nUnitSamples,len(decodeWindows)-1)) for model in modelNames}
                        respLatency = []
                        sdfs = (activePreSDFs,activeChangeSDFs) if state=='active' else (passivePreSDFs,passiveChangeSDFs)
                        preChangeSDFs,changeSDFs = [s.transpose((1,0,2)) for s in sdfs]
                        for i,unitSamp in enumerate(unitSamples):
                            # decode image change and identity for increasing window lengths
                            for j,trunc in enumerate(truncTimes):
                                # image change
                                truncSlice = slice(stimWin.start,stimWin.start+trunc)
                                X = np.concatenate([s[:,unitSamp,truncSlice].reshape((s.shape[0],-1)) for s in (changeSDFs,preChangeSDFs)])
                                y = np.zeros(X.shape[0])
                                y[:int(X.shape[0]/2)] = 1
                                for model,name in zip(models,modelNames):
                                    cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
                                    changeScore[name][i,j] = cv['test_score'].mean()
                                    # get model prediction and feature importance for full length sdfs
                                    if trunc==lastTrunc:
                                        changePredict[name].append(np.mean([estimator.predict(X[:trials.sum()]) for estimator in cv['estimator']],axis=0))
                                        if name=='randomForest':
                                            changePredictProb[name].append(np.mean([estimator.predict_proba(X[:trials.sum()])[:,1] for estimator in cv['estimator']],axis=0))
                                            changePredictProbValid[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:trials.sum(),1])
                                            changeFeatureImportance[name].append(np.mean([np.reshape(estimator.feature_importances_,(n,-1)).mean(axis=0) for estimator in cv['estimator']],axis=0))
                                # image identity
                                imgSDFs = [changeSDFs[:,unitSamp,truncSlice][changeImage==img] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                for model,name in zip(models,modelNames):
                                    cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
                                    imageScore[name][i,j] = cv['test_score'].mean()
                                    if trunc==lastTrunc and name=='randomForest':
                                        imageFeatureImportance[name].append(np.mean([np.reshape(estimator.feature_importances_,(n,-1)).mean(axis=0) for estimator in cv['estimator']],axis=0))
                            
                            # decode pre-change image identity
                            for j,trunc in enumerate(preTruncTimes):
                                preImgSDFs = [preChangeSDFs[:,unitSamp,trunc:][initialImage==img] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in preImgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(preImgSDFs)])
                                for model,name in zip(models,modelNames):
                                    preImageScore[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                            
                            # decode image change and identity for sliding windows
                            for j,(winBegin,winEnd) in enumerate(zip(decodeWindows[:-1],decodeWindows[1:])):
                                # image change
                                winSlice = slice(stimWin.start+winBegin,stimWin.start+winEnd)
                                X = np.concatenate([s[:,unitSamp,winSlice].reshape((s.shape[0],-1)) for s in (changeSDFs,preChangeSDFs)])
                                y = np.zeros(X.shape[0])
                                y[:int(X.shape[0]/2)] = 1
                                for model,name in zip(models,modelNames):
                                    changeScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                # image identity
                                imgSDFs = [changeSDFs[:,unitSamp,winSlice][changeImage==img] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                for model,name in zip(models,modelNames):
                                    imageScoreWindows[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                            
                            # calculate population response latency for unit sample
                            respLatency.append(findLatency(changeSDFs.transpose((1,0,2))[unitSamp].mean(axis=(0,1))[None,:],baseWin,stimWin,method='abs',thresh=1)[0])
                        
                        for model in modelNames:
                            result[exp][region][state]['changeScore'][model].append(changeScore[model].mean(axis=0))
                            result[exp][region][state]['changePredict'][model].append(np.mean(changePredict[model],axis=0))
                            result[exp][region][state]['changePredictProb'][model].append(np.mean(changePredictProb[model],axis=0))
                            result[exp][region][state]['changePredictProbValid'][model].append(np.mean(changePredictProbValid[model],axis=0))
                            result[exp][region][state]['changeFeatureImportance'][model].append(np.mean(changeFeatureImportance[model],axis=0))
                            result[exp][region][state]['imageScore'][model].append(imageScore[model].mean(axis=0))
                            result[exp][region][state]['imageFeatureImportance'][model].append(np.mean(imageFeatureImportance[model],axis=0))
                            result[exp][region][state]['preImageScore'][model].append(preImageScore[model].mean(axis=0))
                            result[exp][region][state]['changeScoreWindows'][model].append(changeScoreWindows[model].mean(axis=0))
                            result[exp][region][state]['imageScoreWindows'][model].append(imageScoreWindows[model].mean(axis=0))
                        result[exp][region][state]['respLatency'].append(np.nanmean(respLatency))
    print(time.clock()-startTime)
warnings.filterwarnings('default')

# plot scores vs number of units
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    allScores = {score: [] for score in ('changeScore','imageScore')}
    for i,region in enumerate(regionLabels):
        for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
            ax = plt.subplot(gs[i,j])
            expScores = []
            for exp in result:
                scr = result[exp][region]['active'][score][model]
                if len(scr)>0:
                    scr = [s[0] for s in scr]
                    scr += [np.nan]*(len(nUnits)-len(scr))
                    expScores.append(scr)
                    allScores[score].append(scr)
                    ax.plot(nUnits,scr,'k')
#            ax.plot(nUnits,np.nanmean(expScores,axis=0),'r',linewidth=2)
            for side in ('right','top'):
                    ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(np.arange(0,100,10))
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,max(nUnits)+5])
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
    
    fig = plt.figure(facecolor='w')
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    ax = plt.subplot(1,1,1)
    for score,clr in zip(('changeScore','imageScore'),('k','0.5')):
        ax.plot(nUnits,np.nanmean(allScores[score],axis=0),color=clr,label=score[:score.find('S')])
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,100,10))
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_yticklabels([0,'',0.5,'',1])
    ax.set_xlim([0,max(nUnits)+5])
    ax.set_ylim([0,1])
    ax.set_xlabel('Number of Units')
    ax.set_ylabel('Decoder Accuracy')
    ax.legend()
    
    
# compare models
fig = plt.figure(facecolor='w',figsize=(10,10))
gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
for i,region in enumerate(regionLabels):
    for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
        ax = plt.subplot(gs[i,j])
        for model,clr in zip(modelNames,'kg'):
            regionScore = []
            for exp in result:
                s = result[exp][region]['active'][score][model]
                if len(s)>0:
                    regionScore.append(s[0])
            n = len(regionScore)
            if n>0:
                m = np.mean(regionScore,axis=0)
                s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                ax.plot(truncTimes,m,color=clr,label=model)
                ax.fill_between(truncTimes,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([0,'',0.5,'',1])
        ax.set_xlim([0,lastTrunc])
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
        if i==len(regionLabels)-1 and j==1:
            ax.legend()
ax.set_xlabel('Time (ms)')


# plot scores for each experiment
for model in modelNames:
    for score,ymin in zip(('changeScore','imageScore'),[0.45,0]):
        fig = plt.figure(facecolor='w',figsize=(10,10))
        fig.text(0.5,0.95,model+', '+score[:score.find('S')],fontsize=14,horizontalalignment='center')
        gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
        for i,region in enumerate(regionLabels):
            for j,state in enumerate(('active','passive')):
                ax = plt.subplot(gs[i,j])
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        ax.plot(truncTimes,s[0],'k')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks([0,50,100,150,200])
                ax.set_yticks([0,0.25,0.5,0.75,1])
                ax.set_yticklabels([0,'',0.5,'',1])
                ax.set_xlim([0,lastTrunc])
                ax.set_ylim([ymin,1])
                if i<len(regionLabels)-1:
                    ax.set_xticklabels([])
                if j>0:
                    ax.set_yticklabels([])    
                if i==0:
                    if j==0:
                        ax.set_title(region+', '+state)
                    else:
                        ax.set_title(state)
                elif j==0:
                    ax.set_title(region)
                if i==0 and j==0:
                    ax.set_ylabel('Decoder Accuracy')
        ax.set_xlabel('Time (ms)')

    
# plot avg score for each area
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,8))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(2,2)
    for i,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for region,clr in zip(regionLabels,regionColors):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(truncTimes,m,color=clr,label=region+'(n='+str(n)+')')
                    ax.fill_between(truncTimes,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,200])
            ax.set_ylim([ymin,1])
            if i==0:
                ax.set_xticklabels([])
                ax.set_title(state)
            else:
                ax.set_xlabel('Time (ms)')
            if j==0:
                ax.set_ylabel('Decoder Accuracy ('+score[:score.find('S')]+')')
            else:
                ax.set_yticklabels([])
            if i==1 and j==1:
                ax.legend()
                
# compare scores for full window
x = np.arange(len(regionLabels))
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,8))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    ax = plt.subplot(1,1,1)
    for score in ('changeScore','imageScore'):
        for state in ('active','passive'):
            m = np.full(len(regionLabels),np.nan)
            sem = m.copy()
            for i,region in enumerate(regionLabels):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0][-1])
                n = len(regionScore)
                if n>0:
                    m[i] = np.mean(regionScore)
                    sem[i] = np.std(regionScore)/(len(regionScore)**0.5)
            clr = [0]*3 if score=='changeScore' else [0.5]*3
            if state=='active':
                clr[0] = 1
            else:
                clr[2] = 1
            ax.plot(x,m,color=clr,label=score[:score.find('S')]+', '+state)
            ax.fill_between(x,m+sem,m-sem,color=clr,alpha=0.25)            
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(regionLabels)
    ax.set_yticks([0.5,0.75,1])
    ax.set_ylim([0.4,1])
    ax.set_ylabel('Decoder Accuracy')
    ax.legend()

# compare avg change and image scores for each area
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for score,clr in zip(('changeScore','imageScore'),('k','0.5')):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(truncTimes,m,color=clr,label=score[:score.find('S')])
                    ax.fill_between(truncTimes,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,lastTrunc])
            ax.set_ylim([0,1])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            if j>0:
                ax.set_yticklabels([])    
            if i==0:
                if j==0:
                    ax.set_title(region+', '+state)
                else:
                    ax.set_title(state)
            elif j==0:
                ax.set_title(region)
            if i==0 and j==0:
                ax.set_ylabel('Decoder Accuracy')
            if i==len(regionLabels)-1 and j==1:
                ax.legend()
    ax.set_xlabel('Time (ms)')

# plot active vs passive for each area and score
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
            ax = plt.subplot(gs[i,j])
            for state,clr in zip(('active','passive'),'rb'):
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(truncTimes,m,color=clr,label=state)
                    ax.fill_between(truncTimes,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,lastTrunc])
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
            if i==len(regionLabels)-1 and j==1:
                ax.legend()
    ax.set_xlabel('Time (ms)')

# plot pre-change image 
for model in modelNames:
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for exp in result:
                s = result[exp][region][state]['preImageScore'][model]
                if len(s)>0:
                    ax.plot(preTruncTimes,s[0],'k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
#            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
#            ax.set_xlim([0,200])
            ax.set_ylim([ymin,1])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            if j>0:
                ax.set_yticklabels([])    
            if i==0:
                if j==0:
                    ax.set_title(region+', '+state)
                else:
                    ax.set_title(state)
            elif j==0:
                ax.set_title(region)
            if i==0 and j==0:
                ax.set_ylabel('Decoder Accuracy')
    ax.set_xlabel('Time before change (ms)')

# plot visual response, change decoding, and image decoding latencies
latencyLabels = {'resp':'Visual Response Latency','change':'Change Decoding Latency','image':'Image Decoding Latency'}

for model in modelNames:
    latency = {exp: {region: {state: {} for state in ('active','passive')} for region in regionLabels} for exp in result}
    for exp in result:
        for region in regionLabels:
            for state in ('active','passive'):
                s = result[exp][region][state]['respLatency']
                if len(s)>0:
                    latency[exp][region][state]['resp'] = s[0]
                for score,decodeThresh in zip(('changeScore','imageScore'),(0.6,0.2)):
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        intpScore = np.interp(np.arange(truncTimes[0],truncTimes[-1]+1),truncTimes,s[0])
                        latency[exp][region][state][score[:score.find('S')]] = findLatency(intpScore,method='abs',thresh=decodeThresh)[0]
    
    fig = plt.figure(facecolor='w',figsize=(10,10))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(3,2)
    axes = []
    latMin = 1000
    latMax = 0
    for i,(xkey,ykey) in enumerate((('resp','change'),('resp','image'),('change','image'))):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            axes.append(ax)
            ax.plot([0,1000],[0,1000],'--',color='0.5')
            for region,clr in zip(regionLabels,regionColors):
                x,y = [[latency[exp][region][state][key] for exp in latency if key in latency[exp][region][state]] for key in (xkey,ykey)]
                latMin = min(latMin,min(x),min(y))
                latMax = max(latMax,max(x),max(y))
#                ax.plot(x,y,'o',mec=clr,mfc='none')
                mx,my = [np.mean(d) for d in (x,y)]
                sx,sy = [np.std(d)/(len(d)**0.5) for d in (x,y)]
                ax.plot(mx,my,'o',mec=clr,mfc=clr)
                ax.plot([mx,mx],[my-sy,my+sy],color=clr)
                ax.plot([mx-sx,mx+sx],[my,my],color=clr)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlabel(latencyLabels[xkey])
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


# plot predicted vs actual performance
for model in ('randomForest',):
    fracSame = {exp: {region: {state: np.nan for state in ('active','passive')} for region in regionLabels} for exp in exps}
    for exp in exps:        
        behavior = result[exp]['behaviorResponse'][:].astype(int)
        behavior[behavior<1] = -1
        for probe in result[exp]:
            for region in regionLabels:
                    for state in ('active','passive'):
                        p = result[exp][region][state]['changePredictProb'][model]
                        if len(p)>0:
#                            fracSame[exp][region][state] = np.corrcoef(behavior,1-p[0])[0,1]
                            predictProb = p[0]
                            predict = (predictProb>0.5).astype(int)
                            predict[predict==0] = -1
                            fracSame[exp][region][state] = np.sum((behavior*predict)==1)/behavior.size
        
    fig = plt.figure(facecolor='w',figsize=(6,4))
    fig.text(0.5,0.95,model,fontsize=14,horizontalalignment='center')
    ax = plt.subplot(1,1,1)
    x = np.arange(len(regionLabels))
    for state,clr,grayclr in zip(('active','passive'),'rb',([1,0.5,0.5],[0.5,0.5,1])):
#        for exp in fracSame:
#            y = [fracSame[exp][region][state] for region in regionLabels]
#            ax.plot(x,y,color=grayclr)
        regionData = [[fracSame[exp][region][state] for exp in fracSame] for region in regionLabels]
        m = np.array([np.nanmean(d) for d in regionData])
        s = np.array([np.nanstd(d)/(np.sum(~np.isnan(d))**0.5) for d in regionData])
        ax.plot(x,m,color=clr,linewidth=2,label=state)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(regionLabels)
    ax.set_ylabel('Fraction Predicted')
    ax.legend()


for exp in fracSame:
    y = [fracSame[exp][region]['active'] for region in regionLabels]
    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(x,y,'k')
    ax.set_xticks(x)
    ax.set_xticklabels(regionLabels)
    ax.set_title(exp)


state = 'active'
for exp in result:
    for region in ('VISam',):
        s = result[exp][region][state]['changeScoreWindows'][model]
        v = result[exp]['VISp'][state]['changeScoreWindows'][model]
        if len(s)>0 and len(v)>0:
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(s[0],'r')
            ax.plot(v[0],'k')
            
for model in modelNames:
    for score,ylim in zip(('changeScoreWindows','imageScoreWindows'),([0.4,0.8],[0,0.6])):
        fig = plt.figure(facecolor='w',figsize=(10,10))
        fig.text(0.5,0.95,model+', '+score,fontsize=14,horizontalalignment='center')
        gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
        for i,region in enumerate(regionLabels):
            for j,state in enumerate(('active','passive')):
                ax = plt.subplot(gs[i,j])
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(decodeWindows[0:-1]+5,m,color=clr,label=score[:score.find('S')])
                    ax.fill_between(decodeWindows[0:-1]+5,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks([0,50,100,150,200])
#                ax.set_yticks([0,0.25,0.5,0.75,1])
#                ax.set_yticklabels([0,'',0.5,'',1])
                ax.set_xlim([0,decodeWindows[-1]])
                ax.set_ylim(ylim)
                if i<len(regionLabels)-1:
                    ax.set_xticklabels([])
                if j>0:
                    ax.set_yticklabels([])    
                if i==0:
                    if j==0:
                        ax.set_title(region+', '+state)
                    else:
                        ax.set_title(state)
                elif j==0:
                    ax.set_title(region)
                if i==0 and j==0:
                    ax.set_ylabel('Decoder Accuracy')
                if i==len(regionLabels)-1 and j==1:
                    ax.legend()
                ax.set_xlabel('Time (ms)')


for model in ('randomForest',):
    for score,ylim in zip(('changeFeatureImportance','imageFeatureImportance'),([0.4,0.8],[0,0.6])):
        fig = plt.figure(facecolor='w',figsize=(10,10))
        fig.text(0.5,0.95,model+', '+score,fontsize=14,horizontalalignment='center')
        gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
        for i,region in enumerate(regionLabels):
            for j,state in enumerate(('active','passive')):
                ax = plt.subplot(gs[i,j])
                regionScore = []
                for exp in result:
                    s = result[exp][region][state][score][model]
                    if len(s)>0:
                        regionScore.append(s[0])
                n = len(regionScore)
                if n>0:
                    m = np.mean(regionScore,axis=0)
                    s = np.std(regionScore,axis=0)/(len(regionScore)**0.5)
                    ax.plot(np.arange(150),m,color=clr,label=score[:score.find('S')])
                    ax.fill_between(np.arange(150),m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks([0,50,100,150,200])
#                ax.set_yticks([0,0.25,0.5,0.75,1])
#                ax.set_yticklabels([0,'',0.5,'',1])
                ax.set_xlim([0,lastTrunc])
#                ax.set_ylim(ylim)
                if i<len(regionLabels)-1:
                    ax.set_xticklabels([])
                if j>0:
                    ax.set_yticklabels([])    
                if i==0:
                    if j==0:
                        ax.set_title(region+', '+state)
                    else:
                        ax.set_title(state)
                elif j==0:
                    ax.set_title(region)
                if i==0 and j==0:
                    ax.set_ylabel('Feature Importance')
                if i==len(regionLabels)-1 and j==1:
                    ax.legend()
                ax.set_xlabel('Time (ms)')



###### behavior change mod correlation
                
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im] for im in 'AB']
regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')

exps = Aexps
imgNames = np.unique(data[exps[0]]['initialImage'])
preRespList,changeRespList,changeModList = [[[[[] for _ in imgNames] for _ in imgNames] for _ in regionLabels] for _ in range(3)]
hitMatrix = np.zeros((len(regionLabels),len(imgNames),len(imgNames)))
missMatrix = hitMatrix.copy()
for exp in exps:
    print(exp)
    initialImage = data[exp]['initialImage'][:]
    changeImage = data[exp]['changeImage'][:]
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    response = data[exp]['response'][:]
    hit = response=='hit'
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    trials = engaged & (hit | (response=='miss'))
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
                pre,change = [data[exp]['sdfs'][probe]['active'][epoch][inRegion,:][:,trials] for epoch in ('preChange','change')]
                hasSpikes,hasResp = findResponsiveUnits(change,baseWin,respWin,thresh=5)
                preSdfs.append(pre[hasSpikes & hasResp])
                changeSdfs.append(change[hasSpikes & hasResp])
        if len(preSdfs)>0:
            preSdfs = np.concatenate(preSdfs)
            changeSdfs = np.concatenate(changeSdfs)
            for ind,trial in enumerate(np.where(trials)[0]):
                preResp,changeResp = [sdfs[:,ind,respWin].mean(axis=1)-sdfs[:,ind,baseWin].mean(axis=1) for sdfs in (preSdfs,changeSdfs)]
                i,j = [np.where(imgNames==img)[0][0] for img in (initialImage[trial],changeImage[trial])]
                preRespList[r][i][j].append(preResp)
                changeRespList[r][i][j].append(changeResp)
                changeModList[r][i][j].append(np.clip((changeResp-preResp)/(changeResp+preResp),-1,1))
                if response[trial]=='hit':
                    hitMatrix[r,i,j] += 1
                else:
                    missMatrix[r,i,j] += 1

preRespMatrix = np.full(hitMatrix.shape,np.nan)
changeRespMatrix = preRespMatrix.copy()
changeModMatrix = preRespMatrix.copy()
for r,_ in enumerate(regionLabels):
    for i,_ in enumerate(imgNames):
        for j,_ in enumerate(imgNames):
            if len(preRespList[r][i][j])>0:
                preRespMatrix[r,i,j] = np.nanmean(np.concatenate(preRespList[r][i][j]))
                changeRespMatrix[r,i,j] = np.nanmean(np.concatenate(changeRespList[r][i][j]))
                changeModMatrix[r,i,j] = np.nanmean(np.concatenate(changeModList[r][i][j]))
popChangeModMatrix = np.clip((changeRespMatrix-preRespMatrix)/(changeRespMatrix+preRespMatrix),-1,1)

hitRate = hitMatrix/(hitMatrix+missMatrix)
imgOrder = np.argsort(np.nanmean(hitRate,axis=(0,1)))
nonDiag = ~np.eye(len(imgNames),dtype=bool)   


for ind,(region,hr) in enumerate(zip(regionLabels,hitRate)):
    fig = plt.figure(facecolor='w',figsize=(7,9))
    fig.text(0.5,0.95,region,fontsize=12,horizontalalignment='center')
    gs = matplotlib.gridspec.GridSpec(4,2)
    
    ax = plt.subplot(gs[0,0])
    im = ax.imshow(hr[imgOrder,:][:,imgOrder],cmap='magma')
    ax.tick_params(direction='out',top=False,right=False,labelsize=6)
    ax.set_xticks(np.arange(len(imgNames)))
    ax.set_yticks(np.arange(len(imgNames)))
    ax.set_xlabel('Change Image',fontsize=10)
    ax.set_ylabel('Initial Image',fontsize=10)
    ax.set_title('Hit Rate',fontsize=10)
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    
    ax = plt.subplot(gs[0,1])
    im = ax.imshow((hitMatrix[ind]+missMatrix[ind])[imgOrder,:][:,imgOrder],cmap='magma')
    ax.tick_params(direction='out',top=False,right=False,labelsize=6)
    ax.set_xticks(np.arange(len(imgNames)))
    ax.set_yticks(np.arange(len(imgNames)))
    ax.set_xlabel('Change Image',fontsize=10)
    ax.set_ylabel('Initial Image',fontsize=10)
    ax.set_title('Number of Trials',fontsize=10)
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    
    for i,(d,lbl) in enumerate(zip((preRespMatrix,changeRespMatrix,changeModMatrix),('Pre Resp','Change Resp','Change Mod'))):
        d = d[ind]
        ax = plt.subplot(gs[i+1,0])
        im = ax.imshow(d[imgOrder,:][:,imgOrder],cmap='magma')
        ax.tick_params(direction='out',top=False,right=False,labelsize=6)
        ax.set_xticks(np.arange(len(imgNames)))
        ax.set_yticks(np.arange(len(imgNames)))
        ax.set_xlabel('Change Image',fontsize=10)
        ax.set_ylabel('Initial Image',fontsize=10)
        ax.set_title(lbl,fontsize=10)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        
        ax = plt.subplot(gs[i+1,1])
        ax.plot(hr[nonDiag],d[nonDiag],'ko')
        slope,yint,rval,pval,stderr = scipy.stats.linregress(hr[nonDiag],d[nonDiag])
        x = np.array([hr[nonDiag].min(),hr[nonDiag].max()])
        ax.plot(x,slope*x+yint,'0.5')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
        ax.set_xlabel('Hit Rate',fontsize=10)
        ax.set_ylabel(lbl,fontsize=10)
        r,p = scipy.stats.pearsonr(hr[nonDiag],d[nonDiag])
        ax.set_title('Pearson: r = '+str(round(r,2))+', p = '+str(round(p,3)),fontsize=8)
    plt.tight_layout()


fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
xticks = np.arange(len(regionLabels))
xlim = [-0.5,len(regionLabels)-0.5]
ax.plot(xlim,[0,0],'k--')
for param,clr,lbl in zip((preRespMatrix,changeRespMatrix,changeModMatrix),'brk',('Pre Resp','Change Resp','Change Mod')):
    r = [scipy.stats.pearsonr(hr[nonDiag],d[nonDiag])[0] for hr,d in zip(hitRate,param)]
    ax.plot(xticks,r,'o',mec=clr,mfc=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels(regionLabels,fontsize=10)
ax.set_ylabel('Correlation with hit rate',fontsize=10)
ax.legend()


