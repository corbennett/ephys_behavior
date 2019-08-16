# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os
import h5py
import fileIO
import getData
import probeSync
import analysis_utils
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


def getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze='all',sdfParams={}):
    if popDataToHDF5:
        popHDF5Path = os.path.join(localDir,'popData.hdf5')
    for mouseID,ephysDates,probeIDs,imageSet,passiveSession in mouseInfo:
        if miceToAnalyze!='all' and mouseID not in miceToAnalyze:
            continue
        for date,probes in zip(ephysDates,probeIDs):
            expName = date+'_'+mouseID
            print(expName)
            dataDir = baseDir+expName
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
                data[expName]['sdfs'] = getSDFs(obj,probes=probes,**sdfParams)
                data[expName]['regions'] = getUnitRegions(obj,probes=probes)
                data[expName]['isi'] = {probe: obj.probeCCF[probe]['ISIRegion'] for probe in probes}
                data[expName]['changeImage'] = obj.changeImage[trials]
                data[expName]['response'] = resp[trials]
                # add preChange image identity, time between changes, receptive field info

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
            if state=='active' or len(obj.passive_pickle_file)>0:  
                frameTimes =obj.frameAppearTimes if state=='active' else obj.passiveFrameAppearTimes
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
    regions = {}
    for probe in probes:
        regions[probe] = []
        units = probeSync.getOrderedUnits(obj.units[probe])
        for u in units:
            r = obj.probeCCF[probe]['ISIRegion'] if obj.units[probe][u]['inCortex'] else obj.units[probe][u]['ccfRegion']
            regions[probe].append(r)
    return regions


def findResponsiveUnits(sdfs,baseWin,respWin,thresh=10):
    unitMeanSDFs = np.array([s.mean(axis=0) for s in sdfs]) if len(sdfs.shape)>2 else sdfs.copy()
    hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
    unitMeanSDFs -= unitMeanSDFs[:,baseWin].mean(axis=1)[:,None]
    hasResp = unitMeanSDFs[:,respWin].max(axis=1) > thresh*unitMeanSDFs[:,baseWin].std(axis=1)
    return hasSpikes,hasResp

    
def findLatency(data,baseWin=None,respWin=None,method='rel',thresh=3,minPtsAbove=30):
    latency = []
    if len(data.shape)<2:
        data = data[None,:]
    if baseWin is not None:
        data = data-data[:,baseWin].mean(axis=1)[:,None]
    if respWin is None:
        respWin = slice(0,data.shape[1])
    for d in data:
        if method=='abs':
            ptsAbove = np.where(np.correlate(d[respWin]>thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        else:
            ptsAbove = np.where(np.correlate(d[respWin]>d[baseWin].std()*thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        if len(ptsAbove)>0:
            latency.append(ptsAbove[0])
        else:
            latency.append(np.nan)
    return np.array(latency)


def calcChangeMod(preChangeSDFs,changeSDFs,baseWin,respWin):
    pre = preChangeSDFs[:,respWin].mean(axis=1)
    change = changeSDFs[:,respWin].mean(axis=1)
    changeMod = np.clip((change-pre)/(change+pre),-1,1)
    meanMod = np.mean(changeMod)
    semMod = changeMod.std()/(changeMod.size**0.5)
    changeLat = findLatency(changeSDFs-preChangeSDFs,baseWin,respWin)
    return meanMod, semMod, changeLat


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
            )

# make new experiment hdf5s without updating popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=False,miceToAnalyze=('429084',))

# make new experiment hdf5s and add to existing popData.hdf5
getPopData(objToHDF5=True,popDataToHDF5=True,miceToAnalyze=('423745',))

# make popData.hdf5 from existing experiment hdf5s
getPopData(objToHDF5=False,popDataToHDF5=True)



data = h5py.File(os.path.join(localDir,'popData.hdf5'),'r')

baseWin = slice(0,250)
respWin = slice(250,500)

exps = data.keys() # all experiments

# A or B days that have passive session
Aexps,Bexps = [[mouseID+'_'+exp[0] for exp in mouseInfo for mouseID,probes,imgSet,hasPassive in zip(*exp[1:]) if imgSet==im and hasPassive] for im in 'AB']
exps = Aexps+Bexps


###### behavior analysis
hitRate = []
falseAlarmRate = []
dprime = []
for exp in exps:
    response = data[exp]['response'][:]
    hit,miss,fa,cr = [np.sum(response==r) for r in ('hit','miss','falseAlarm','correctReject')]
    hitRate.append(hit/(hit+miss))
    falseAlarmRate.append(fa/(cr+fa))
    dprime.append(calculateDprime(hit,miss,fa,cr))

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
ax.set_title('n = '+str(nMice)+' mice',fontsize=16)

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
ax.set_title('n = '+str(nMice)+' mice',fontsize=16)



###### change mod and latency analysis

allPre,allChange = [[np.concatenate([data[exp]['sdfs'][probe][state][epoch][:].mean(axis=1) for exp in exps for probe in data[exp]['sdfs']]) for state in ('active','passive')] for epoch in ('preChange','change')]

expNames = np.concatenate([[exp]*len(data[exp]['sdfs'][probe]['active']['change']) for exp in exps for probe in data[exp]['sdfs']])
expDates = np.array([exp[:8] for exp in expNames])
expMouseIDs = np.array([exp[-6:] for exp in expNames])

(hasSpikesActive,hasRespActive),(hasSpikesPassive,hasRespPassive) = [findResponsiveUnits(sdfs,baseWin,respWin) for sdfs in allChange]
baseRate = [sdfs[:,baseWin].mean(axis=1) for sdfs in allPre+allChange]
activePre,passivePre,activeChange,passiveChange = [sdfs-sdfs[:,baseWin].mean(axis=1)[:,None] for sdfs in allPre+allChange]
hasResp = hasSpikesActive & hasSpikesPassive & hasRespActive

regions = np.concatenate([data[exp]['regions'][probe][:] for exp in exps for probe in data[exp]['regions']])    
#regionNames = sorted(list(set(regions)))
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
regionNames = regionNames[1:7]

anatomyData = pd.read_excel(os.path.join(localDir,'hierarchy_scores_2methods.xlsx'))
regionLabels = [r[1] for r in regionNames]
hierScore_8regions,hierScore_allRegions = [[h for r in regionLabels for a,h in zip(anatomyData['areas'],anatomyData[hier]) if a in r] for hier in ('Computed among 8 regions','Computed with ALL other cortical & thalamic regions')]

nUnits = []
nExps = []
nMice = []
changeModActive = []
changeModPassive = []
behModChange = []
behModPre = []
figs = [plt.figure(figsize=(12,6)) for _ in range(6)]
axes = [fig.add_subplot(1,1,1) for fig in figs]
for ind,(region,regionLabels) in enumerate(regionNames):
    inRegion = np.in1d(regions,regionLabels) & hasResp
    nUnits.append(inRegion.sum())
    nExps.append(len(set(expDates[inRegion])))
    nMice.append(len(set(expMouseIDs[inRegion])))
    
    # plot baseline and response spike rates
    for sdfs,base,mec,mfc,lbl in zip((activePre,passivePre,activeChange,passiveChange),baseRate,('rbrb'),('none','none','r','b'),('Active Pre','Passive Pre','Active Change','Passive Change')):
        meanResp = sdfs[inRegion,respWin].mean(axis=1)
        peakResp = sdfs[inRegion,respWin].max(axis=1)
        for r,ax in zip((base[inRegion],meanResp,peakResp),axes[:3]):
            m = r.mean()
            s = r.std()/(r.size**0.5)
            lbl = None if ind>0 else lbl
            ax.plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
            ax.plot([ind,ind],[m-s,m+s],color=mec)
    
    # plot mean change mod and latencies
    (activeChangeMean,activeChangeSem,activeChangeLat),(passiveChangeMean,passiveChangeSem,passiveChangeLat),(behChangeMean,behChangeSem,behChangeLat),(behPreMean,behPreSem,behPreLat) = \
    [calcChangeMod(pre[inRegion],change[inRegion],baseWin,respWin) for pre,change in zip((activePre,passivePre,passiveChange,passivePre),(activeChange,passiveChange,activeChange,activePre))]
    
    changeModActive.append(activeChangeMean)
    changeModPassive.append(passiveChangeMean)
    behModChange.append(behChangeMean)
    behModPre.append(behPreMean)
    
    activeLat,passiveLat = [findLatency(sdfs[inRegion],baseWin,respWin) for sdfs in (activeChange,passiveChange)]
    
    for m,s,mec,mfc,lbl in zip((activeChangeMean,passiveChangeMean),(activeChangeSem,passiveChangeSem),'rb','rb',('Active','Passive')):
        lbl = None if ind>0 else lbl
        axes[-3].plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
        axes[-3].plot([ind,ind],[m-s,m+s],mec)
        
    for m,s,mec,mfc,lbl in zip((behChangeMean,behPreMean),(behChangeSem,behPreSem),'kk',('k','none'),('Change','Pre-change')):
        lbl = None if ind>0 else lbl
        axes[-2].plot(ind,m,'o',mec=mec,mfc=mfc,ms=12,label=lbl)
        axes[-2].plot([ind,ind],[m-s,m+s],mec)
            
    for lat,mec,mfc in zip((activeLat,passiveLat,activeChangeLat,passiveChangeLat,behChangeLat),'rbrbk',('none','none','r','b','k')):
        m = np.nanmean(lat)
        s = np.nanstd(lat)/(lat.size**0.5)
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
    ax.set_xticklabels([r[0]+'\nn='+str(n) for r,n in zip(regionNames,nUnits)],fontsize=16)
    ax.set_ylabel(ylbl,fontsize=16)
    ax.legend()


fig = plt.figure(facecolor='w',figsize=(18,7))
gs = matplotlib.gridspec.GridSpec(2,4)
for j,(param,ylab) in enumerate(zip((changeModActive,changeModPassive,behModChange,behModPre),('Change Mod Active','Change Mod Passive','Behav Mod Change','Behav Mod Pre'))):
    for i,(hier,xlab) in enumerate(zip((hierScore_8regions,hierScore_allRegions),('8 Region Hierarchy','All Region Hierarchy'))):
        ax = plt.subplot(gs[i,j])
        ax.plot(hier,param,'ko',ms=6)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
        ax.set_xticks(hier)
        ax.set_xticklabels([r[0] for r in regionNames])
        ax.set_xlabel(xlab,fontsize=10)
        ax.set_ylabel(ylab,fontsize=10)
        r,p = scipy.stats.pearsonr(hier,param)
        title = 'Pearson: r^2 = '+str(round(r**2,2))+', p = '+str(round(p,3))
        r,p = scipy.stats.spearmanr(hier,param)
        title += '\nSpearman: r^2 = '+str(round(r**2,2))+', p = '+str(round(p,3))
        ax.set_title(title,fontsize=8)
line = matplotlib.lines.Line2D((0,1),(0.5,0.5),color='k',transform=fig.transFigure)
fig.lines = [line]
plt.tight_layout()

#fig = plt.figure(facecolor='w',figsize=(18,7))
#gs = matplotlib.gridspec.GridSpec(2,4)
#regionColors = plt.cm.jet(np.linspace(0,1,len(regionNames)))[:,:3]
#for j,(param,ylab) in enumerate(zip((changeModActive,changeModPassive,behModChange,behModPre),('Change Mod Active','Change Mod Passive','Behav Mod Change','Behav Mod Pre'))):
#    for i,(hier,xlab) in enumerate(zip((hierScore_8regions,hierScore_allRegions),('8 Region Hierarchy','All Region Hierarchy'))):
#        hierRepeated = [a+np.zeros(len(b)) for a,b in zip(hier,param)]
#        h = np.concatenate(hierRepeated)
#        d = np.concatenate(param)
#        ax = plt.subplot(gs[i,j])
#        for x,y,clr in zip(hierRepeated,param,regionColors):
#            ax.plot(x+(np.random.rand(x.size)-0.5)/20,y,'o',ms=2,mec=clr,mfc='none')
#        for side in ('right','top'):
#            ax.spines[side].set_visible(False)
#        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
#        ax.set_xticks(hier)
#        ax.set_xticklabels([r[0] for r in regionNames])
#        ax.set_xlabel(xlab,fontsize=10)
#        ax.set_ylabel(ylab,fontsize=10)
#        r,p = scipy.stats.pearsonr(h,d)
#        title = 'Pearson: r^2 = '+str(round(r**2,3))+', p = '+str(round(p,6))
#        r,p = scipy.stats.spearmanr(h,d)
#        title += '\nSpearman: r^2 = '+str(round(r**2,3))+', p = '+str(round(p,6))
#        ax.set_title(title,fontsize=8)
#line = matplotlib.lines.Line2D((0,1),(0.5,0.5),color='k',transform=fig.transFigure)
#fig.lines = [line]
#plt.tight_layout()



###### decoding analysis
    
regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))

for exp in data:
    print(exp)
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    for thresh in (5,10,15):
        print(thresh)
        for probe in data[exp]['regions']:
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

nUnits = [19]
nRepeats = 3
nCrossVal = 3

truncInterval = 10
lastTrunc = 200
truncTimes = np.arange(truncInterval,lastTrunc+1,truncInterval)

preTruncTimes = np.arange(-750,0,50)

assert((len(nUnits)>=1 and len(truncTimes)==1) or (len(nUnits)==1 and len(truncTimes)>=1))
models = (RandomForestClassifier(n_estimators=100), LinearSVC(C=1.0,max_iter=1e5)) # SVC(kernel='linear',C=1.0,probability=True)
modelNames = ('randomForest', 'supportVector')
behavStates = ('active','passive')
result = {exp: {probe: {state: {'changeScore':{model:[] for model in modelNames},
                                'changePredict':{model:[] for model in modelNames},
                                'imageScore':{model:[] for model in modelNames},
                                'preImageScore':{model:[] for model in modelNames},
                                'respLatency':[]} for state in behavStates} for probe in data[exp]['sdfs']} for exp in data}
for expInd,exp in enumerate(exps):
    print('experiment '+str(expInd+1)+' of '+str(len(exps)))
    if 'passive' in behavStates:
        hasPassive = len(data[exp]['sdfs'][data[exp]['sdfs'].keys()[0]]['passive']['change'])>0
        if not hasPassive:
            continue
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    changeImage = data[exp]['changeImage'][trials]
    imageNames = np.unique(changeImage)
    for probeInd,probe in enumerate(data[exp]['sdfs']):
        print('probe '+str(probeInd+1)+' of '+str(len(data[exp]['sdfs'])))
        region = data[exp]['isi'][probe].value
        result[exp][probe]['region'] = region
        if region in regionLabels:
            inRegion = data[exp]['regions'][probe][:]==region
            if any(inRegion):
                hasSpikesActive,hasRespActive = findResponsiveUnits(data[exp]['sdfs'][probe]['active']['change'][inRegion,:,:respWin.stop][:,trials],baseWin,respWin)
                useUnits = hasSpikesActive & hasRespActive
                if 'passive' in behavStates:
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(data[exp]['sdfs'][probe]['passive']['change'][inRegion,:,:respWin.stop][:,trials],baseWin,respWin)
                    useUnits = useUnits & hasSpikesPassive
                units = np.where(useUnits)[0]
                for n in nUnits:
                    if len(units)>=n:
                        unitSamples = [np.random.choice(units,size=n,replace=False) for _ in range(nRepeats)]
                        for state in behavStates:
                            if state=='active' or hasPassive:
                                changeScore = {model: np.zeros((nRepeats,len(truncTimes))) for model in modelNames}
                                changePredict = {model: [] for model in modelNames}
                                imageScore = {model: np.zeros((nRepeats,len(truncTimes))) for model in modelNames}
                                preImageScore = {model: np.zeros((nRepeats,len(preTruncTimes))) for model in modelNames}
                                respLatency = []
                                changeSDFs,preChangeSDFs = [data[exp]['sdfs'][probe][state][epoch][inRegion,:].transpose((1,0,2))[trials] for epoch in ('change','preChange')]
                                for i,unitSamp in enumerate(unitSamples):
                                    for j,trunc in enumerate(truncTimes):
                                        # decode image change
                                        truncSlice = slice(respWin.start,respWin.start+trunc)
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
                                        preImgSDFs = [preChangeSDFs[:,unitSamp,trunc:][np.concatenate(([''],changeImage[:-1]))==img] for img in imageNames]
                                        X = np.concatenate([s.reshape((s.shape[0],-1)) for s in preImgSDFs])
                                        y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(preImgSDFs)])
                                        for model,name in zip(models,modelNames):
                                            preImageScore[name][i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                    # calculate population response latency for unit sample
                                    respLatency.append(findLatency(changeSDFs.transpose((1,0,2))[unitSamp].mean(axis=(0,1))[None,:],baseWin,respWin)[0])
                                for model in modelNames:
                                    result[exp][probe][state]['changeScore'][model].append(changeScore[model].mean(axis=0))
                                    result[exp][probe][state]['changePredict'][model].append(np.mean(changePredict[model],axis=0))
                                    result[exp][probe][state]['imageScore'][model].append(imageScore[model].mean(axis=0))
                                    result[exp][probe][state]['preImageScore'][model].append(preImageScore[model].mean(axis=0))
                                result[exp][probe][state]['respLatency'].append(np.nanmean(respLatency))
                            

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
                for probe in result[exp]:
                    if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                        scr = [s[0] for s in result[exp][probe]['active'][score][model]]
                        scr += [np.nan]*(len(nUnits)-len(scr))
                        expScores.append(scr)
                        allScores[score].append(scr)
                        ax.plot(nUnits,scr,'k')
            ax.plot(nUnits,np.nanmean(expScores,axis=0),'r',linewidth=2)
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
                for probe in result[exp]:
                    if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                        s = result[exp][probe]['active'][score][model]
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


# plot scores for each probe
for model in modelNames:
    for score,ymin in zip(('changeScore','imageScore'),[0.45,0]):
        fig = plt.figure(facecolor='w',figsize=(10,10))
        fig.text(0.5,0.95,model+', '+score[:score.find('S')],fontsize=14,horizontalalignment='center')
        gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
        for i,region in enumerate(regionLabels):
            for j,state in enumerate(('active','passive')):
                ax = plt.subplot(gs[i,j])
                for exp in result:
                    for probe in result[exp]:
                        if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                            for s in result[exp][probe][state][score][model]:
                                ax.plot(truncTimes,s,'k')
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
                    for probe in result[exp]:
                        if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                            s = result[exp][probe][state][score][model]
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
                    for probe in result[exp]:
                        if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                            s = result[exp][probe][state][score][model]
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
                    for probe in result[exp]:
                        if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                            s = result[exp][probe][state][score][model]
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
                for probe in result[exp]:
                    if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                        for s in result[exp][probe][state]['preImageScore'][model]:
                            ax.plot(preTruncTimes,s,'k')
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
        for probe in result[exp]:
            for region in regionLabels:
                if 'region' in result[exp][probe] and result[exp][probe]['region']==region:
                    for state in ('active','passive'):
                        s = result[exp][probe][state]['respLatency']
                        if len(s)>0:
                            latency[exp][region][state]['resp'] = s[0]
                        for score,decodeThresh in zip(('changeScore','imageScore'),(0.625,0.25)):
                            s = result[exp][probe][state][score][model]
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




