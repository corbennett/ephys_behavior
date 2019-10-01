# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:36:30 2019

@author: svc_ccg
"""

import numpy as np
import getData
import summaryPlots
from matplotlib import pyplot as plt

#using our class for parsing the data (https://github.com/corbennett/ephys_behavior)
b = getData.behaviorEphys('Z:\\08092019_423744')
b.loadFromHDF5(r"Z:\analysis\08092019_423744.hdf5") 

#get the change times for this recording (when the image identity changed and the mouse should have licked)
selectedTrials = (b.hit | b.miss)&(~b.ignore)   #Omit "ignore" trials (aborted trials when the mouse licked too early or catch trials when the image didn't actually change)
active_changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one here to correct for a one frame shift in frame times from camstim

#Since the passive session is identical to the active session, you can index the passive session frame times the same way
passive_changeTimes = b.passiveFrameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1]

#Here are the pre-change and change image ids for each trial
change_image_id = b.changeImage[selectedTrials]
pre_change_image_id = b.initialImage[selectedTrials]


#get times for active and passive sessions
firstActiveTimePoint = b.frameAppearTimes[0]
lastActiveTimePoint = b.lastBehaviorTime
getActiveSpikes = lambda x: (x>firstActiveTimePoint)&(x<=lastActiveTimePoint)

firstPassiveTimePoint = b.passiveFrameAppearTimes[0]
lastPassiveTimePoint = b.passiveFrameAppearTimes[-1]
getPassiveSpikes = lambda x: (x>firstPassiveTimePoint)&(x<=lastPassiveTimePoint)


def plotRaster(spikes, alignTimes, ax=None, preTime = 1.5, postTime = 1.5, color='k'):
    if ax is None:
        fig, ax = plt.subplots()
    
    spikeTrain, _ = np.histogram(spikes, bins=np.arange(0, alignTimes[-1]+postTime, 0.001))

    for i,t in enumerate(alignTimes):
        t = int(round(t*1000))
        spikeTimesThisTrial = np.where(spikeTrain[t-int(1000*preTime):t+int(1000*postTime)])[0]
        if len(spikeTimesThisTrial)>0:
            ax.vlines(spikeTimesThisTrial, i, i+1, color=color)


minTrial = 0
maxTrial = 200

V1probes, V1units = b.getUnitsByArea('VISp')
for p,u in zip(V1probes, V1units):
    spikes = b.units[p][u]['times']
    if getActiveSpikes(spikes).sum() < 1000 or getPassiveSpikes(spikes).sum < 1000:
        continue
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12,5)
    fig.suptitle(p+u)
    for it, (times, color) in enumerate(zip([active_changeTimes, passive_changeTimes], 'rb')):
        plotRaster(spikes, times, ax[it], color=color)
        ax[it].set_ylim([minTrial, maxTrial])


AMprobes, AMunits = b.getUnitsByArea('VISam')
for p,u in zip(AMprobes, AMunits):
    spikes = b.units[p][u]['times']
    if getActiveSpikes(spikes).sum() < 1000 or getPassiveSpikes(spikes).sum < 1000:
        continue
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12,5)
    fig.suptitle(p+u)
    for it, (times, color) in enumerate(zip([active_changeTimes, passive_changeTimes], 'rb')):
        plotRaster(spikes, times, ax[it], color=color)
        ax[it].set_ylim([minTrial, maxTrial])

areasOfInterest = ['LGd']
for area in areasOfInterest:
    cortexFlag = area in ['VISp', 'VISl', 'VISal', 'VISrl', 'VISam', 'VISpm']
    probes, units = b.getUnitsByArea(area, cortexFlag)
    for p,u in zip(probes, units):
        spikes = b.units[p][u]['times']
        if getActiveSpikes(spikes).sum() < 1000 or getPassiveSpikes(spikes).sum() < 1000:
            continue
        if not 0.8<getActiveSpikes(spikes).sum()/getPassiveSpikes(spikes).sum()<1.2:
            continue
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(12,5)
        fig.suptitle(area + ': ' + p + u)
        for it, (times, color) in enumerate(zip([active_changeTimes, passive_changeTimes], 'rb')):
            plotRaster(spikes, times, ax[it], color=color)
            ax[it].set_ylim([minTrial, maxTrial])









