# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:49:39 2020

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


windowDur = 0.15
binSize = 0.005
binStarts = np.arange(-0.75-windowDur,windowDur,binSize)
changeBaseWin = (binStarts >= -windowDur) & (binStarts<0)
changeRespWin = (binStarts >= 0) & (binStarts < windowDur)
preChangeBaseWin = binStarts < -0.75
preChangeRespWin = (binStarts >= -0.75) & (binStarts < -0.75+windowDur)


def getPSTH(spikes,startTimes,windowDur,binSize=0.005,avg=False):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts

def findResponsiveUnits(psth,baseWin,respWin,thresh=5):
    unitMeanPSTHs = psth.mean(axis=1)
    hasSpikes = unitMeanPSTHs.mean(axis=1)>0.1
    unitMeanPSTHs -= unitMeanPSTHs[:,baseWin].mean(axis=1)[:,None]
    hasResp = unitMeanPSTHs[:,respWin].max(axis=1) > thresh*unitMeanPSTHs[:,baseWin].std(axis=1)
    return hasSpikes,hasResp


data = h5py.File(os.path.join(localDir,'popData.hdf5'),'r')

# A or B days that have passive session
Aexps,Bexps = [[expDate+'_'+mouse[0] for mouse in mouseInfo for expDate,probes,imgSet,hasPassive in zip(*mouse[1:]) if imgSet==im and hasPassive] for im in 'AB']
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

decodeWindowSize = 10
decodeWindows = []#np.arange(stimWin.start,stimWin.start+150,decodeWindowSize)

preImageDecodeWindowSize = 50
preImageDecodeWindows = []#np.arange(stimWin.start,stimWin.start+750,preImageDecodeWindowSize)

# models = (RandomForestClassifier(n_estimators=100),LinearSVC(C=1.0,max_iter=1e4),LinearDiscriminantAnalysis(),SVC(kernel='linear',C=1.0,probability=True)))
# modelNames = ('randomForest','supportVector','LDA')
models = (RandomForestClassifier(n_estimators=100),)
modelNames = ('randomForest',)

behavStates = ('active','passive')

# add catchScore, catchPrediction, reactionScore
result = {exp: {region: {state: {'changeScore':{model:[] for model in modelNames},
                                 'changePredict':{model:[] for model in modelNames},
                                 'changePredictProb':{model:[] for model in modelNames},
                                 'changeFeatureImportance':{model:[] for model in modelNames},
                                 'catchPredict':{model:[] for model in modelNames},
                                 'catchPredictProb':{model:[] for model in modelNames},
                                 'imageScore':{model:[] for model in modelNames},
                                 'imageFeatureImportance':{model:[] for model in modelNames},
                                 'changeScoreWindows':{model:[] for model in modelNames},
                                 'changePredictWindows':{model:[] for model in modelNames},
                                 'imageScoreWindows':{model:[] for model in modelNames},
                                 'preImageScoreWindows':{model:[] for model in modelNames},
                                 'meanSDF':[],
                                 'respLatency':[]} for state in behavStates} for region in regionLabels} for exp in exps}

warnings.filterwarnings('ignore')
for expInd,exp in enumerate(exps):
    print('experiment '+str(expInd+1)+' of '+str(len(exps)))
    startTime = time.clock()
    
    response = data[exp]['response'][:]
    hit = response=='hit'
    falseAlarm = response=='falseAlarm'
    changeTimes = data[exp]['behaviorChangeTimes'][:]
    flashTimes = data[exp]['behaviorFlashTimes'][:]
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in changeTimes])
    changeTrials = engaged & (hit | (response=='miss'))
    catchTrials = engaged & (falseAlarm | (response=='correctReject'))
    result[exp]['responseToChange'] = hit[changeTrials]
    result[exp]['responseToCatch'] = falseAlarm[catchTrials]
    result[exp]['reactionTime'] = data[exp]['rewardTimes'][engaged & hit] - changeTimes[engaged & hit]
    
    nonChangeFlashes = []
    result[exp]['responseToNonChangeFlash'] = []
    lickTimes = data[exp]['lickTimes'][:]
    for i,t in enumerate(flashTimes):
        if (len(nonChangeFlashes)<1) or (t>flashTimes[nonChangeFlashes[-1]]+4):
            nearestChange = min(abs(changeTimes-t))
            if (nearestChange>4) and (nearestChange<60):
                nonChangeFlashes.append(i)
                lickLat = lickTimes-t
                result[exp]['responseToNonChangeFlash'].append(any((lickLat>0.15) & (lickLat<0.75)))
    nonChangeFlashTimes = flashTimes[nonChangeFlashes]
    print(len(flashTimes),len(nonChangeFlashes),sum(result[exp]['responseToNonChangeFlash']))
    
    
    if 'passive' in behavStates:
        passiveChangeTimes = data[exp]['passiveChangeTimes'][:]
        passiveFlashTimes = data[exp]['passiveFlashTimes'][:]
        passivePreChangeTimes = passiveFlashTimes[np.searchsorted(passiveFlashTimes,passiveChangeTimes)-1]
        passiveNonChangeFlashTimes = passiveFlashTimes[nonChangeFlashes]
    
    initialImage = data[exp]['initialImage'][changeTrials]
    changeImage = data[exp]['changeImage'][changeTrials]
    imageNames = np.unique(changeImage)
    
    for region,regionCCFLabels,_ in regionsToUse:
        activePreSDFs = []
        activeChangeSDFs = []
        passivePreSDFs = []
        passiveChangeSDFs = []
        for probe in data[exp]['sdfs']:
            ccf = data[exp]['ccfRegion'][probe][:]
            isi = data[exp]['isiRegion'][probe][()]
            if isi:
                ccf[data[exp]['inCortex'][probe][:]] = isi
            inRegion = np.in1d(ccf,regionCCFLabels)
            if any(inRegion):
                units = data[exp]['units'][probe][inRegion]
                spikes = data[exp]['spikeTimes'][probe]
                activePre,activeChange = np.array([getPSTH(spikes[u],times-windowDur,windowDur*2,binSize) for u in units]
                hasSpikesActive,hasRespActive = findResponsiveUnits(activeChange[:,changeTrials],baseWin,respWin,thresh=5)
                if 'passive' in behavStates:
                    passivePre,passiveChange = [np.array([getPSTH(spikes[u],times-windowDur,windowDur*2,binSize) for u in units]) for times in (passivePreChangeTimes,passiveChangeTimes)]
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(passiveChange[:,changeTrials],baseWin,respWin,thresh=5)
                    hasResp = hasSpikesActive & hasSpikesPassive & (hasRespActive | hasRespPassive)
                    passivePreSDFs.append(passivePre[hasResp])
                    passiveChangeSDFs.append(passiveChange[hasResp])
                else:
                    hasResp = hasSpikesActive & hasRespActive
                activePreSDFs.append(activePre[hasResp])
                activeChangeSDFs.append(activeChange[hasResp])
        if len(activePreSDFs)>0:
            activePreSDFs = np.concatenate(activePreSDFs)
            activeChangeSDFs = np.concatenate(activeChangeSDFs)
            if 'passive' in behavStates:
                passivePreSDFs = np.concatenate(passivePreSDFs)
                passiveChangeSDFs = np.concatenate(passiveChangeSDFs)
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
                        changeScore = {model: [] for model in modelNames}
                        changePredict = {model: [] for model in modelNames}
                        changePredictProb = {model: [] for model in modelNames}
                        changeFeatureImportance = {model: np.full((nsamples,nUnits,respWin.sum()),np.nan) for model in modelNames}
                        catchPredict = {model: [] for model in modelNames}
                        catchPredictProb = {model: [] for model in modelNames}
                        imageScore = {model: [] for model in modelNames}
                        imageFeatureImportance = {model: np.full((nsamples,nUnits,respWin.sum()),np.nan) for model in modelNames}
                        changeScoreWindows = {model: np.zeros((nsamples,len(decodeWindows))) for model in modelNames}
                        changePredictWindows = {model: np.zeros((nsamples,len(decodeWindows),changeTrials.sum())) for model in modelNames}
                        imageScoreWindows = {model: np.zeros((nsamples,len(decodeWindows))) for model in modelNames}
                        preImageScoreWindows = {model: np.zeros((nsamples,len(preImageDecodeWindows))) for model in modelNames}
                        
                        sdfs = (activePreSDFs,activeChangeSDFs) if state=='active' else (passivePreSDFs,passiveChangeSDFs)
                        preChangeSDFs,changeSDFs = [s.transpose((1,0,2)) for s in sdfs]
                        
                        for i,unitSamp in enumerate(unitSamples):
                            # decode image change and identity for full respWin
                            # image change
                            X = np.concatenate([s[changeTrials][:,unitSamp][:,:,respWin].reshape((changeTrials.sum(),-1)) for s in (changeSDFs,preChangeSDFs)])
                            y = np.zeros(X.shape[0])
                            y[:int(X.shape[0]/2)] = 1
                            Xcatch = changeSDFs[catchTrials][:,unitSamp][:,:,respWin].reshape((catchTrials.sum(),-1))
                            for model,name in zip(models,modelNames):
                                cv = cross_validate(model,X,y,cv=nCrossVal,return_estimator=True)
                                changeScore[name].append(cv['test_score'].mean())
                                changePredict[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict')[:changeTrials.sum()])
                                catchPredict[name].append(scipy.stats.mode([estimator.predict(Xcatch) for estimator in cv['estimator']],axis=0)[0].flatten())
                                if name=='randomForest':
                                    changePredictProb[name].append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:changeTrials.sum(),1])
                                    changeFeatureImportance[name][i][unitSamp] = np.mean([np.reshape(estimator.feature_importances_,(sampleSize,-1)) for estimator in cv['estimator']],axis=0)
                                    catchPredictProb[name].append(np.mean([estimator.predict_proba(Xcatch)[:,1] for estimator in cv['estimator']],axis=0))
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
                        for model in modelNames:
                            result[exp][region][state]['changeScore'][model].append(np.median(changeScore[model],axis=0))
                            result[exp][region][state]['changePredict'][model].append(scipy.stats.mode(changePredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['changePredictProb'][model].append(np.median(changePredictProb[model],axis=0))
                            result[exp][region][state]['changeFeatureImportance'][model].append(np.nanmedian(changeFeatureImportance[model],axis=0))
                            result[exp][region][state]['catchPredict'][model].append(scipy.stats.mode(catchPredict[model],axis=0)[0].flatten())
                            result[exp][region][state]['catchPredictProb'][model].append(np.median(catchPredictProb[model],axis=0))
                            result[exp][region][state]['imageScore'][model].append(np.median(imageScore[model],axis=0))
                            result[exp][region][state]['imageFeatureImportance'][model].append(np.nanmedian(imageFeatureImportance[model],axis=0))
                            result[exp][region][state]['changeScoreWindows'][model].append(np.median(changeScoreWindows[model],axis=0))
                            result[exp][region][state]['changePredictWindows'][model].append(np.median(changePredictWindows[model],axis=0))
                            result[exp][region][state]['imageScoreWindows'][model].append(np.median(imageScoreWindows[model],axis=0))
                            result[exp][region][state]['preImageScoreWindows'][model].append(np.median(preImageScoreWindows[model],axis=0))
                        
                        meanSDF = activeChangeSDFs.mean(axis=(0,1))
                        result[exp][region][state]['meanSDF'].append(meanSDF)
#                        result[exp][region][state]['respLatency'].append(findLatency(meanSDF[None,:],baseWin,stimWin,method='abs',thresh=0.5)[0])
    print(time.clock()-startTime)
warnings.filterwarnings('default')


# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(result,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
result = pickle.load(open(pkl,'rb'))


















































