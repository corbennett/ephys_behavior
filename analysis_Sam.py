# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os
from collections import OrderedDict
import h5py
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
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
                data[expName]['spikeTimes'] = {}
                for probe in probes:
                    data[expName]['spikeTimes'][probe] = OrderedDict()
                    for ind,u in enumerate(probeSync.getOrderedUnits(obj.units[probe])):
                        data[expName]['spikeTimes'][probe][str(ind)] = obj.units[probe][u]['times']
                data[expName]['sdfs'] = getSDFs(obj,probes=probes,**sdfParams)
                data[expName]['regions'] = getUnitRegions(obj,probes=probes)
                data[expName]['isi'] = {probe: obj.probeCCF[probe]['ISIRegion'] for probe in probes}
                data[expName]['initialImage'] = obj.initialImage[trials]
                data[expName]['changeImage'] = obj.changeImage[trials]
                data[expName]['response'] = resp[trials]
                data[expName]['behaviorFlashTimes'] = obj.frameAppearTimes[obj.flashFrames]
                data[expName]['behaviorChangeTimes'] = obj.frameAppearTimes[obj.changeFrames[trials]]
                data[expName]['behaviorRunTime'] = obj.behaviorRunTime
                data[expName]['behaviorRunSpeed'] = obj.behaviorRunSpeed
                data[expName]['lickTimes'] = obj.lickTimes
                if obj.passive_pickle_file is not None:
                    data[expName]['passiveFlashTimes'] = obj.passiveFrameAppearTimes[obj.flashFrames]
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


def getUnitRegions(obj,probes='all'):    
    if probes=='all':
        probes = obj.probes_to_analyze
    regions = {probe: {method:[] for method in ('ccf','isi')} for probe in probes}
    for probe in probes:
        units = probeSync.getOrderedUnits(obj.units[probe])
        for u in units:
            regions[probe]['ccf'].append(obj.units[probe][u]['ccfRegion'])
            isiRegion = obj.probeCCF[probe]['ISIRegion']
            r = isiRegion if str(isiRegion)!='nan' and obj.units[probe][u]['inCortex'] else ''
            regions[probe]['isi'].append(r)
    return regions


def findResponsiveUnits(sdfs,baseWin,respWin,thresh=5):
    unitMeanSDFs = np.array([s.mean(axis=0) for s in sdfs]) if len(sdfs.shape)>2 else sdfs.copy()
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


def calcChangeMod(preChangeSDFs,changeSDFs,respWin,baseWin,stimWin):
    pre = preChangeSDFs[:,respWin].mean(axis=1)
    change = changeSDFs[:,respWin].mean(axis=1)
    changeMod = np.clip((change-pre)/(change+pre),-1,1)
#    m = np.mean(changeMod)
#    s = changeMod.std()/(changeMod.size**0.5)
    m = np.median(changeMod)
    s = np.percentile([np.median(np.random.choice(changeMod,changeMod.size,replace=True)) for _ in range(5000)],(2.5,97.5))
    lat = findLatency(changeSDFs-preChangeSDFs,baseWin,stimWin)
    return m,s,lat
    

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
             ('417882',('03262019','03272019'),('ABCEF','ABCF'),'AB',(False,False)),
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
            )


# A or B days that have passive session
Aexps,Bexps = [[mouseID+'_'+exp[0] for exp in mouseInfo for mouseID,probes,imgSet,hasPassive in zip(*exp[1:]) if imgSet==im and hasPassive] for im in 'AB']
exps = Aexps+Bexps


#
makeSummaryPlots(miceToAnalyze=('423744','423750','459521'))


# make new experiment hdf5s without updating popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=False,miceToAnalyze=('423744',))

# make new experiment hdf5s and add to existing popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True,miceToAnalyze=('459521',))

# make popData.hdf5 from existing experiment hdf5s
getPopData(objToHDF5=False,popDataToHDF5=True)

# append existing hdf5s to existing popData.hdf5
getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze=('423744',))



data = h5py.File(os.path.join(localDir,'popData.hdf5'),'r')

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
    hit,miss,fa,cr = [np.sum(response==r) for r in ('hit','miss','falseAlarm','correctReject')]
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
ax.set_ylim([0,3])
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

allPre,allChange = [[np.concatenate([data[exp]['sdfs'][probe][state][epoch][:].mean(axis=1) for exp in exps for probe in data[exp]['sdfs']]) for state in ('active','passive')] for epoch in ('preChange','change')]

expNames = np.concatenate([[exp]*len(data[exp]['sdfs'][probe]['active']['change']) for exp in exps for probe in data[exp]['sdfs']])
expDates = np.array([exp[:8] for exp in expNames])
expMouseIDs = np.array([exp[-6:] for exp in expNames])

(hasSpikesActive,hasRespActive),(hasSpikesPassive,hasRespPassive) = [findResponsiveUnits(sdfs,baseWin,respWin,thresh=5) for sdfs in allChange]
baseRate = [sdfs[:,baseWin].mean(axis=1) for sdfs in allPre+allChange]
activePre,passivePre,activeChange,passiveChange = [sdfs-sdfs[:,baseWin].mean(axis=1)[:,None] for sdfs in allPre+allChange]
hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)

unitRegions = np.concatenate([data[exp]['regions'][probe][:] for exp in exps for probe in data[exp]['sdfs']])  
  
#regionNames = sorted(list(set(unitRegions)))
regionNames = (
               ('LGN',('LGd',)),
               ('V1',('VISp',)),
               ('LM',('VISl',)),
               ('AL',('VISal',)),
               ('RL',('VISrl',)),
               ('PM',('VISpm',)),
               ('AM',('VISam',)),
               ('LP',('LP',)),
               ('SCd',('SCig','SCig-b')),
               ('APN',('APN',)),
               ('MRN',('MRN',)),
               ('hipp',('CA1','CA3','DG-mo','DG-po','DG-sg','HPF'))
              )
regionNames = regionNames[:8]
regionLabels = [r[1] for r in regionNames]

nUnits = []
nExps = []
nMice = []
cmiActive = []
cmiActiveCI = []
cmiPassive = []
cmiPassiveCI = []
bmiChange = []
bmiChangeCI = []
bmiPre = []
bmiPreCI = []
respLat = []
changeModLat = []
popRespLat = []
popChangeModLat = []
figs = [plt.figure(figsize=(12,9)) for _ in range(6)]
axes = [fig.add_subplot(1,1,1) for fig in figs]
for ind,(region,regionCCFLabels) in enumerate(regionNames):
    inRegion = np.in1d(unitRegions,regionCCFLabels) & hasResp #& np.in1d(expDates,('09052019','09062019'))
    nUnits.append(inRegion.sum())
    nExps.append(len(set(expDates[inRegion])))
    nMice.append(len(set(expMouseIDs[inRegion])))
    
    (activeChangeMod,activeChangeModCI,activeChangeModLat),(passiveChangeMod,passiveChangeModCI,passiveChangeModLat),(behModChange,behModChangeCI,behModChangeLat),(behModPre,behModPreCI,behModPreLat) = \
    [calcChangeMod(pre[inRegion],change[inRegion],respWin,baseWin,stimWin) for pre,change in zip((activePre,passivePre,passiveChange,passivePre),(activeChange,passiveChange,activeChange,activePre))]
    
    activeLat,passiveLat = [findLatency(sdfs[inRegion],baseWin,stimWin) for sdfs in (activeChange,passiveChange)]
    
    cmiActive.append(activeChangeMod)
    cmiActiveCI.append(activeChangeModCI)
    cmiPassive.append(passiveChangeMod)
    cmiPassiveCI.append(passiveChangeModCI)
    bmiChange.append(behModChange)
    bmiChangeCI.append(behModChangeCI)
    bmiPre.append(behModPre)
    bmiPreCI.append(behModPreCI)
    respLat.append(activeLat)
    changeModLat.append(activeChangeModLat)
    popRespLat.append(findLatency(activeChange[inRegion].mean(axis=0),baseWin,stimWin,method='abs',thresh=1)[0])
    popChangeModLat.append(findLatency(np.mean(activeChange[inRegion]-activePre[inRegion],axis=0),baseWin,stimWin,method='abs',thresh=1)[0])
    
    # plot baseline and response spike rates
    for sdfs,base,mec,mfc,lbl in zip((activePre,passivePre,activeChange,passiveChange),baseRate,('rbrb'),('none','none','r','b'),('Active Pre','Passive Pre','Active Change','Passive Change')):
        meanResp = sdfs[inRegion,respWin].mean(axis=1)
        peakResp = sdfs[inRegion,respWin].max(axis=1)
        for r,ax in zip((base[inRegion],meanResp,peakResp),axes[:3]):
            m = np.mean(r)
            s = r.std()/(r.size**0.5)
            lbl = None if ind>0 else lbl
            ax.plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
            ax.plot([ind,ind],[m-s,m+s],color=mec)
    
    # plot mean change mod and latencies
    for m,s,mec,mfc,lbl in zip((activeChangeMod,passiveChangeMod),(activeChangeModCI,passiveChangeModCI),'rb','rb',('Active','Passive')):
        lbl = None if ind>0 else lbl
        axes[-3].plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
        axes[-3].plot([ind,ind],s,mec)
        
    for m,s,mec,mfc,lbl in zip((behModChange,behModPre),(behModChangeCI,behModPreCI),'kk',('k','none'),('Change','Pre-change')):
        lbl = None if ind>0 else lbl
        axes[-2].plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
        axes[-2].plot([ind,ind],s,mec)
            
    for lat,mec,mfc in zip((activeLat,passiveLat,activeChangeModLat,passiveChangeModLat,behModChangeLat),'rbrbk',('none','none','r','b','k')):
        lat = lat[~np.isnan(lat)]
        m = lat.mean()
        s = lat.std()/(lat.size**0.5)
        axes[-1].plot(ind,m,'o',mec=mec,mfc=mfc,ms=12)
        axes[-1].plot([ind,ind],[m-s,m+s],mec)
    
    # plot pre and post change responses and their difference
    fig = plt.figure(figsize=(8,8))
    ylim = None
    for i,(pre,change,clr,lbl) in enumerate(zip((activePre,passivePre),(activeChange,passiveChange),([1,0,0],[0,0,1]),('Active','Passive'))):
        ax = fig.add_subplot(2,1,i+1)
        clrlight = np.array(clr).astype(float)
        clrlight[clrlight==0] = 0.7
        for d,c in zip((pre,change,change-pre),(clrlight,clr,[0.5,0.5,0.5])):
            m = np.mean(d[inRegion],axis=0)
            s = np.std(d[inRegion],axis=0)/(inRegion.sum()**0.5)
            ax.plot(m,color=c)
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
        ax.set_title(region+' '+lbl)
    
    fig = plt.figure(figsize=(8,8))
    ylim = None
    for i,(active,passive,lbl) in enumerate(zip((activeChange,activePre),(passiveChange,passivePre),('Change','Pre'))):
        ax = fig.add_subplot(2,1,i+1)
        for d,c in zip((active,passive),'rb'):
            m = np.mean(d[inRegion],axis=0)
            s = np.std(d[inRegion],axis=0)/(inRegion.sum()**0.5)
            ax.plot(m,color=c)
            ax.fill_between(np.arange(len(m)),m+s,m-s,color=c,alpha=0.25)
            ax.plot(np.arange(respWin.start,respWin.stop),np.zeros(respWin.stop-respWin.start)+np.mean(m[respWin]),'--',color=c)
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
        ax.set_title(region+' '+lbl)

for ax,ylbl in zip(axes,('Baseline (spikes/s)','Mean Resp (spikes/s)','Peak Resp (spikes/s)','Change Modulation Index','Behavior Modulation Index','Latency (ms)')):
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([-0.5,len(regionNames)-0.5])
    ax.set_xticks(np.arange(len(regionNames)))
    ax.set_xticklabels([r[0]+'\n'+str(n)+' cells\n'+str(d)+' days\n'+str(m)+' mice' for r,n,d,m in zip(regionNames,nUnits,nExps,nMice)],fontsize=16)
    if 'spikes' in ylbl:
        ax.set_ylim([0,plt.get(ax,'ylim')[1]])
    ax.set_ylabel(ylbl,fontsize=16)
    ax.legend()
    
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for lat,mec,mfc in zip((popRespLat,popChangeModLat),'kk',('k','none')):
    ax.plot(lat,'o',mec=mec,mfc=mfc,ms=10)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim([-0.5,len(regionLabels)-0.5])
ax.set_ylim([25,50])
ax.set_xticks(np.arange(len(regionLabels)))
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Latency (ms)',fontsize=10)


anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
hierScore_8regions,hierScore_allRegions = [[h for r in regionLabels for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a in r] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')]
    
hier = hierScore_8regions

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for latency,mec,mfc in zip((respLat,changeModLat),'kk',('k','none')):
    m = []
    ci = []
    for lat in latency:
        lat = lat[~np.isnan(lat)]
        m.append(np.median(lat))
        ci.append(np.percentile([np.median(np.random.choice(lat,lat.size,replace=True)) for _ in range(5000)],(2.5,97.5)))
    ax.plot(hier,m,'o',mec=mec,mfc=mfc,ms=12)
    for h,c in zip(hier,ci):
        ax.plot([h,h],c,mec)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_ylim([35,75])
ax.set_xticks(hier)
ax.set_xticklabels([str(round(h,2))+'\n'+r[0] for h,r in zip(hier,regionNames)])
ax.set_xlabel('Hierarchy Score',fontsize=10)
ax.set_ylabel('Latency (ms)',fontsize=10)


for m,ci,ylab in zip((cmiActive,cmiPassive,bmiChange,bmiPre),(cmiActiveCI,cmiPassiveCI,bmiChangeCI,bmiPreCI),('Change Mod Active','Change Mod Passive','Behav Mod Change','Behav Mod Pre')):
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(1,1,1)
    ax.plot(hier,m,'ko',ms=6)
    for h,c in zip(hier,ci):
        ax.plot([h,h],c,'k')
    slope,yint,rval,pval,stderr = scipy.stats.linregress(hier,m)
    x = np.array([min(hier),max(hier)])
    ax.plot(x,slope*x+yint,'0.5')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
#        ax.set_xticks(hier)
#        ax.set_xticklabels([str(round(h,2))+'\n'+r[0] for h,r in zip(hier,regionNames)])
    ax.set_xlabel('Hierarchy Score',fontsize=10)
    ax.set_ylabel(ylab,fontsize=10)
    r,p = scipy.stats.pearsonr(hier,m)
    title = 'Pearson: r^2 = '+str(round(r**2,2))+', p = '+str(round(p,3))
    r,p = scipy.stats.spearmanr(hier,m)
    title += '\nSpearman: r^2 = '+str(round(r**2,2))+', p = '+str(round(p,3))
    ax.set_title(title,fontsize=8)
    plt.tight_layout()



###### adaptation

regionLabels = ('LGd','VISp','VISl','VISal','VISrl','VISpm','VISam','LP')
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))

sdfs = {region:[] for region in regionLabels}
for exp in exps:
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    for region in regionLabels:
        for probe in data[exp]['regions']:
            inRegion = data[exp]['regions'][probe][:]==region
            if any(inRegion):
                hasSpikes,hasResp = findResponsiveUnits(data[exp]['sdfs'][probe]['active']['change'][inRegion].mean(axis=1),baseWin,respWin,thresh=5)
                uindex = np.where(inRegion)[0][hasSpikes & hasResp]
                for u in uindex:
                    sdfs[region].append(analysis_utils.getSDF(data[exp]['spikeTimes'][probe][u][:],changeTimes-0.25,0.25+6*0.75,sampInt=0.001,filt='exp',sigma=0.005,avg=True)[0])

fig = plt.figure(facecolor='w')
ax = fig.subplots(2,1)
adaptMean = []
adaptSem = []
for region,clr in zip(regionLabels,regionColors):
    regSdfs = np.concatenate(sdfs[region])
    regSdfs -= regSdfs[:,:250].mean(axis=1)
    m = regSdfs.mean(axis=0)
    s = regSdfs.std()/(len(regSdfs)**0.5)
    s /= m.max()
    m /= m.max()
    ax[0].plot(m,clr)
    ax[0].fill_between(np.arange(len(m)),m+s,m-s,color=clr,alpha=0.25)
    
    regResp = []
    regRespSem = []
    for i in np.arange(250,6*750,750):
        r = regSdfs[:,i:i+250].max(axis=1)
        regResp.append(r.mean())
        regRespSem.append(r.std()/(len(r)**0.5))
    regResp,regRespSem = [np.array(r)/regResp[0] for r in (regResp,regRespSem)]
    ax[1].plot(regResp,clr,m='o')
    for x,(m,s) in enumerate(zip(regResp,regRespSem)):
        ax[1].plot([x,x],[m-s,m+s],clr)
        
    adaptMean.append(regResp[-1])
    adaptSem.append(regRespSem[-1])

fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
ax.plot(adaptMean,'ko',ms=10)
for x,(m,s) in enumerate(zip(adaptMean,adaptSem)):
    ax[1].plot([x,x],[m-s,m+s],'ko')
ax.tick_params(direction='out',top=False,right=False,labelsize=8)
ax.set_xlim([-0.5,len(regionLabels)-0.5])
ax.set_xticks(np.arange(len(regionLabels)))
ax.set_xticklabels(regionLabels)
ax.set_ylabel('Adaptation (fraction of change reseponse)',fontsize=10)


###### decoding analysis
    
regionLabels = ('LGd','VISp','VISl','VISal','VISrl','VISpm','VISam','LP')
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))

for exp in data:
    print(exp)
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    for thresh in (5,):
        print(thresh)
        for probe in data[exp]['sdfs']:
            n = []
            for region in regionLabels:
                inRegion = data[exp]['regions'][probe][:]==region
                if any(inRegion):
                    sdfs = data[exp]['sdfs'][probe]['active']['change'][inRegion,:,:respWin.stop][:,trials]
                    hasSpikes,hasResp = findResponsiveUnits(sdfs,baseWin,respWin,thresh)
                    n.append(np.sum(hasSpikes & hasResp))
                else:
                    n.append(0)
            print(probe,n)

nUnits = [20]
nRepeats = 3
nCrossVal = 3

truncInterval = 5
lastTrunc = 200
truncTimes = np.arange(truncInterval,lastTrunc+1,truncInterval)

preTruncTimes = np.arange(-750,0,50)

assert((len(nUnits)>=1 and len(truncTimes)==1) or (len(nUnits)==1 and len(truncTimes)>=1))
models = (RandomForestClassifier(n_estimators=100),)# LinearSVC(C=1.0,max_iter=1e6)) # SVC(kernel='linear',C=1.0,probability=True)
modelNames = ('randomForest',)# 'supportVector')
behavStates = ('active','passive')
result = {exp: {region: {state: {'changeScore':{model:[] for model in modelNames},
                                'changePredict':{model:[] for model in modelNames},
                                'imageScore':{model:[] for model in modelNames},
                                'preImageScore':{model:[] for model in modelNames},
                                'respLatency':[]} for state in behavStates} for region in regionLabels} for exp in data}
for expInd,exp in enumerate(exps):
    print('experiment '+str(expInd+1)+' of '+str(len(exps)))
    (activePreSDFs,activeChangeSDFs),(passivePreSDFs,passiveChangeSDFs) = [[np.concatenate([data[exp]['sdfs'][probe][state][epoch][:] for probe in data[exp]['sdfs']])  for epoch in ('preChange','change')] for state in ('active','passive')]
    unitRegions = np.concatenate([data[exp]['regions'][probe][:] for probe in data[exp]['sdfs']])
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    initialImage = data[exp]['initialImage'][trials]
    changeImage = data[exp]['changeImage'][trials]
    imageNames = np.unique(changeImage)
    for region in regionLabels:
        inRegion = unitRegions==region
        if any(inRegion):
            hasSpikesActive,hasRespActive = findResponsiveUnits(activeChangeSDFs,baseWin,respWin)
            hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChangeSDFs,baseWin,respWin)
            useUnits = inRegion & hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
            units = np.where(useUnits)[0]
            for n in nUnits:
                if len(units)>=n:
                    unitSamples = [np.random.choice(units,size=n,replace=False) for _ in range(nRepeats)]
                    for state in behavStates:
                        changeScore = {model: np.zeros((nRepeats,len(truncTimes))) for model in modelNames}
                        changePredict = {model: [] for model in modelNames}
                        imageScore = {model: np.zeros((nRepeats,len(truncTimes))) for model in modelNames}
                        preImageScore = {model: np.zeros((nRepeats,len(preTruncTimes))) for model in modelNames}
                        respLatency = []
                        sdfs = (activePreSDFs,activeChangeSDFs) if state=='active' else (passivePreSDFs,passiveChangeSDFs)
                        preChangeSDFs,changeSDFs = [s.transpose((1,0,2))[trials] for s in sdfs]
                        for i,unitSamp in enumerate(unitSamples):
                            for j,trunc in enumerate(truncTimes):
                                # decode image change
                                truncSlice = slice(stimWin.start,stimWin.start+trunc)
                                X = np.concatenate([s[:,unitSamp,truncSlice].reshape((s.shape[0],-1)) for s in (changeSDFs,preChangeSDFs)])
                                y = np.zeros(X.shape[0])
                                y[:int(X.shape[0]/2)] = 1
                                for model,name in zip(models,modelNames):
                                    changeScore[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                if trunc==lastTrunc:
                                    # get model prediction probability for full length sdfs
                                    for model,name in zip(models,modelNames):
                                        if not isinstance(model,sklearn.svm.classes.LinearSVC):
                                            changePredict[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:trials.sum(),1])
                                # decode image identity
                                imgSDFs = [changeSDFs[:,unitSamp,truncSlice][changeImage==img] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                for model,name in zip(models,modelNames):
                                    imageScore[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                            # decode pre-change image identity
                            for j,trunc in enumerate(preTruncTimes):
                                preImgSDFs = [preChangeSDFs[:,unitSamp,trunc:][initialImage==img] for img in imageNames]
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in preImgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(preImgSDFs)])
                                for model,name in zip(models,modelNames):
                                    preImageScore[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                            # calculate population response latency for unit sample
                            respLatency.append(findLatency(changeSDFs.transpose((1,0,2))[unitSamp].mean(axis=(0,1))[None,:],baseWin,stimWin)[0])
                        for model in modelNames:
                            result[exp][region][state]['changeScore'][model].append(changeScore[model].mean(axis=0))
                            result[exp][region][state]['changePredict'][model].append(np.mean(changePredict[model],axis=0))
                            result[exp][region][state]['imageScore'][model].append(imageScore[model].mean(axis=0))
                            result[exp][region][state]['preImageScore'][model].append(preImageScore[model].mean(axis=0))
                        result[exp][region][state]['respLatency'].append(np.nanmean(respLatency))
                            

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
        ax.set_xlim([0,200])
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
                ax.set_xlim([0,200])
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
    ax.set_ylim([0.5,1])
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
            ax.set_xlim([0,200])
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
            ax.set_xlim([0,200])
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
                for score,decodeThresh in zip(('changeScore','imageScore'),(0.625,0.25)):
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
    fracSame = {exp: {region: {state: np.nan for state in ('active','passive')} for region in regionLabels} for exp in result}
    for exp in data:
        response = data[exp]['response'][:]
        trials = (response=='hit') | (response=='miss')
        behavior = np.ones(trials.sum())
        behavior[response[trials]=='miss'] = -1
        for probe in result[exp]:
            for region in regionLabels:
                if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                    for state in ('active','passive'):
                        p = result[exp][probe][state]['changePredict'][model]
                        if len(p)>0:
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




