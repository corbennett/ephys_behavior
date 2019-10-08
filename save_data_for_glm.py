# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:13:18 2019

@author: svc_ccg
"""

from __future__ import division
import getData
from matplotlib import pyplot as plt
import probeSync
import licking_model
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import copy
import numpy as npo
from numba import njit
import scipy.stats
import time
import os
import pickle

#using our class for parsing the data (https://github.com/corbennett/ephys_behavior)
#expNames = ('05162019_423749','07112019_429084', '04042019_408528', '08082019_423744', '05172019_423749')
expName = '05162019_423749'
b = getData.behaviorEphys(os.path.join('Z:\\',expName))
b.loadFromHDF5(os.path.join('Z:\\analysis',expName+'.hdf5')) 

saveDir=r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\glm_files"

def binVariable(var, binwidth=0.001):
    return np.histogram(var, bins=np.arange(0, b.lastBehaviorTime, binwidth))[0]

def binRunning(runSig, runTime, binwidth=0.001):
    return np.interp(np.arange(0, b.lastBehaviorTime, binwidth), runTime, runSig)[:-1]
    
@njit
def getPSTH(binned_spikes, binned_times, binwidth, preTime, postTime):
    preTime_bin = int(preTime/binwidth)
    postTime_bin = int(postTime/binwidth)
    eventTimes = npo.where(binned_times)[0]
    psth = npo.zeros(preTime_bin+postTime_bin)
    traces_included = 0
    for i, t in enumerate(eventTimes):
        trace = binned_spikes[t-preTime_bin:t+postTime_bin]
        if len(trace)==len(psth):
            psth = psth + trace
            traces_included += 1
    
    return psth/traces_included

def find_initial_guess(psth, ffilter):
    def compute_error(latent, inputs):
        return np.sum((latent-inputs)**2)
    
    def to_min(params):
        ffilter.set_params(params)
        latent = ffilter.build_filter()
        return compute_error(latent, np.log(psth))
    
    params=ffilter.params
    g = grad(to_min)
    res = minimize(to_min, params, jac=g)
    
    return res['x']

def getChangeBins(binned_changeTimes, binwidth, preChangeTime = 2, postChangeTime=2):
    preBins = int(preChangeTime/binwidth)
    postBins = int(postChangeTime/binwidth)
    changeBins = npo.convolve(binned_changeTimes, npo.ones(preBins+postBins), 'same').astype(npo.bool)
    return changeBins
    

binwidth = 0.05
restrictToChange=False

image_id = np.array(b.core_data['visual_stimuli']['image_name'])
selectedTrials = (b.hit | b.miss)&(~b.ignore)   #Omit "ignore" trials (aborted trials when the mouse licked too early or catch trials when the image didn't actually change)
active_changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one here to correct for a one frame shift in frame times from camstim
binned_activeChangeTimes = binVariable(active_changeTimes, binwidth)
if restrictToChange:
    changeBins = getChangeBins(binned_activeChangeTimes, binwidth)
else:
    changeBins = npo.ones(binned_activeChangeTimes.size).astype(npo.bool)


lick_times = b.lickTimes
first_lick_times = lick_times[np.insert(np.diff(lick_times)>=0.5, 0, True)]

reward_frames = b.core_data['rewards']['frame'].values
reward_times = b.vsyncTimes[reward_frames] 

flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    
eventsToInclude = [('change', [active_changeTimes, 8, 0.8, 0.1, -0.2]),
                   ('licks', [lick_times, 5, 0.6, 0.1, -0.3]),
                   ('first_licks', [first_lick_times, 10, 2, 0.2, -1]),
                   ('running', [[b.behaviorRunSpeed.values, b.behaviorRunTime], 5, 2, 0.4, 0]),
                   ('reward', [reward_times, 10, 2, 0.2, -1])]

for img in np.unique(image_id):
    eventsToInclude.append((img, [flash_times[image_id==img], 8, 0.8, 0.1, -0.2]))

np.save(os.path.join(saveDir, expName+'_events.npy'), eventsToInclude)



###RUN for entire experiment###
spikeRateThresh=0.5
unit_data = []
for pID in b.probes_to_analyze:
    for u in probeSync.getOrderedUnits(b.units[pID]):
        spikes = b.units[pID][u]['times']
        binned_spikes = binVariable(spikes[spikes<b.lastBehaviorTime], binwidth)[changeBins]
        if np.sum(spikes<b.lastBehaviorTime)>spikeRateThresh*b.lastBehaviorTime:
            uid = pID+'_'+u
            if b.units[pID][u]['inCortex']:
                ccfRegion=b.probeCCF[pID]['ISIRegion']
            else:
                ccfRegion= b.units[pID][u]['ccfRegion']
        
            unit_data.append((uid,ccfRegion, binned_spikes))
        
        
np.save(os.path.join(saveDir, expName+'_unitdata.npy'), unit_data)     

np.savez(os.path.join(saveDir, expName+'_params.npz'), lastBehaviorTime=b.lastBehaviorTime, binwidth=binwidth, spikeRateThresh=spikeRateThresh)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        