# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:51:37 2019

@author: svc_ccg
"""
from __future__ import division
import numpy as np
import h5py, os
from matplotlib import pyplot as plt
from matplotlib import cm
from analysis_utils import formatFigure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

popFile = r"Z:\analysis\popData.hdf5"
data = h5py.File(popFile)
expname = '05162019_423749'

###make experiment tensor
d = data[expname]
probes = d['units'].keys()
includePassive = False

totalActiveFlashes = d['behaviorFlashTimes'][()].size + d['behaviorOmitFlashTimes'][()].size
totalPassiveFlashes = d['passiveFlashTimes'][()].size + d['passiveOmitFlashTimes'][()].size
trialTime = 750
neuronNumber = np.sum([len(d['units'][p]) for p in probes])
tensor_active = np.zeros((neuronNumber, trialTime, totalActiveFlashes), dtype='bool')
if includePassive:
    tensor_passive = np.zeros((neuronNumber, trialTime, totalActiveFlashes), dtype='bool')


behaviorFlashTimes = np.concatenate((d['behaviorFlashTimes'][()],d['behaviorOmitFlashTimes'][()]))
passiveFlashTimes = np.concatenate((d['passiveFlashTimes'][()],d['passiveOmitFlashTimes'][()]))
indsWithOmittedInterleaved = np.argsort(behaviorFlashTimes)
imageID = np.concatenate((d['flashImage'][()], d['omitFlashImage'][()]))

behaviorFlashTimes = behaviorFlashTimes[indsWithOmittedInterleaved]
passiveFlashTimes = passiveFlashTimes[indsWithOmittedInterleaved]
imageID = imageID[indsWithOmittedInterleaved]

finalTimePoint = int(round(d['passiveFlashTimes'][()][-1]*1000))+750
unitRegion = []
unitcounter = 0
for p in probes:
    for inCortex, region, u in zip(d['inCortex'][p][()], d['ccfRegion'][p][()], d['units'][p][()]):
        if inCortex and not d['isiRegion'][p][()]=='':
            unitRegion.append(d['isiRegion'][p][()])
        else:
            unitRegion.append(region)
        
        spikeTimes = np.round(d['spikeTimes'][p][u][()]*1000).astype(np.int)
        spikeVector = np.zeros(finalTimePoint, dtype='bool')
        spikeVector[spikeTimes[spikeTimes<finalTimePoint]] = 1
        for ifl, (aflashTime, pflashTime) in enumerate(zip(behaviorFlashTimes, passiveFlashTimes)):
            aflashTime_ms = int(round(aflashTime*1000))
            tensor_active[unitcounter, :, ifl] = spikeVector[aflashTime_ms:aflashTime_ms+750]
            
            if includePassive:
                pflashTime_ms = int(round(pflashTime*1000))
                tensor_passive[unitcounter, :, ifl] = spikeVector[pflashTime_ms:pflashTime_ms+750]
            
        
        unitcounter+=1
        
       

#make raster across all neurons for change
changeTrialIndex = np.isin(behaviorFlashTimes, d['behaviorChangeTimes'][()])
omitTrialIndex = np.isin(behaviorFlashTimes, d['behaviorOmitFlashTimes'][()])
lickTimes = d['lickTimes'][()]
changeTimes = d['behaviorChangeTimes'][()]
omitTimes = d['behaviorOmitFlashTimes'][()]

cortical_cmap = cm.plasma
#cortical_cmap = cm.viridis
subcortical_cmap = cm.Reds
areas = (
         ('VISp', ('VISp'), cortical_cmap(0)),
         ('VISl', ('VISl'), cortical_cmap(0.1)),
         ('VISrl', ('VISrl'), cortical_cmap(0.2)),
         ('VISal', ('VISal'), cortical_cmap(0.3)),
         ('VISpm', ('VISpm'), cortical_cmap(0.4)),
         ('VISam', ('VISam'), cortical_cmap(0.5)),
         ('LGd', ('LGd'), subcortical_cmap(0)),
         ('LP', ('LP'), subcortical_cmap(0)),
         ('hipp', ('CA1', 'CA3', 'DG-mo', 'DG-sg', 'DG-po'), subcortical_cmap(0.6)),
         ('SUB', ('PRE', 'POST', 'SUB'), subcortical_cmap(0.7)),
         ('APN', ('APN'), subcortical_cmap(1.0)),
         ('MB', ('MB', 'MRN'), subcortical_cmap(0.8)),
#         ('MRN', ('MRN')),
         ('SCd', ('SCig', 'SCig-b'), subcortical_cmap(0.9))
         
        )

#cmap= cm.gnuplot2_r
#cmap = cm.brg
pretrialwin = 3
posttrialwin= 7
for changeTrial in np.where(changeTrialIndex)[0][:5]:
    sessionStartTime = behaviorFlashTimes[changeTrial-pretrialwin]
    sessionEndTime = behaviorFlashTimes[changeTrial+posttrialwin]
    trialsToRaster = slice(changeTrial-pretrialwin, changeTrial+posttrialwin)
    trialTensor = tensor_active[:, :, trialsToRaster]
    #trialTensor = np.reshape(trialTensor, (trialTensor.shape[0], -1))
    trialTensor = np.hstack([trialTensor[:,:,i] for i in np.arange(trialTensor.shape[2])])
    unitCounter = 0
    fig, ax = plt.subplots(figsize=(12,12))
    for ia, (area, parts,regioncolor) in enumerate(areas[::-1]):
#        regioncolor = cmap(0.8*ia/(len(areas)-1))
        areaCounter=0
        for iu, ur in enumerate(unitRegion):
            if ur in parts:
                spikes = trialTensor[iu]
                spiketimes = np.where(spikes)[0]
                if len(spiketimes)>0:
                    ax.plot(spiketimes, unitCounter*np.ones(len(spiketimes)), '|', ms=1, color=regioncolor)
                    unitCounter+=1
                    areaCounter+=1
        if areaCounter>0:
            ax.text(1.05*ax.get_xlim()[1], unitCounter - areaCounter/2, area, color=regioncolor)
    windowLicks = 1000*lickTimes[(sessionStartTime<lickTimes)&(lickTimes<sessionEndTime)] - 1000*sessionStartTime
    ax.plot(windowLicks, -20*np.ones(windowLicks.size), 'ko')
    formatFigure(fig,ax, no_spines=True)
    
    axylims = ax.get_ylim()
    windowChanges = 1000*changeTimes[(sessionStartTime<changeTimes)&(changeTimes<sessionEndTime)] - 1000*sessionStartTime
    ax.vlines(windowChanges, *axylims, color='k', alpha=0.4)
    
    windowOmits = 1000*omitTimes[(sessionStartTime<omitTimes)&(omitTimes<sessionEndTime)] - 1000*sessionStartTime
    ax.vlines(windowOmits, *axylims, color='g', alpha=0.4)
    
    for i, flashx in enumerate(np.arange(0, 750*(pretrialwin+posttrialwin+1), 750)):
        if i<pretrialwin:
            rect = Rectangle((flashx, unitCounter*1.1), 250, 100, facecolor='k', alpha=0.5)
        else:
            rect = Rectangle((flashx, unitCounter*1.1), 250, 100, facecolor='g', alpha=0.5)
        ax.add_artist(rect)











