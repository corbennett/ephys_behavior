# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:32:28 2019

@author: svc_ccg
"""

import numpy as np
import h5py, os

popFile = r"Z:\analysis\popData.hdf5"
data = h5py.File(popFile)

d = data['05162019_423749']

totalActiveFlashes = d['behaviorFlashTimes'][()].size
totalPassiveFlashes = d['passiveFlashTimes'][()].size
trialTime = 750
neuronNumber = np.sum([len(d['units'][p]) for p in 'ABCDEF'])
tensor_active = np.zeros((neuronNumber, trialTime, totalActiveFlashes), dtype='bool')
tensor_passive = np.zeros((neuronNumber, trialTime, totalActiveFlashes), dtype='bool')

imageID = d['flashImage'][()]

finalTimePoint = int(round(d['passiveFlashTimes'][()][-1]*1000))+750
unitRegion = []
unitcounter = 0
for p in 'ABCDEF':
    for inCortex, region, u in zip(d['inCortex'][p][()], d['ccfRegion'][p][()], d['units'][p][()]):
        if inCortex:
            unitRegion.append(d['isiRegion'][p][()])
        else:
            unitRegion.append(region)
        
        spikeTimes = np.round(d['spikeTimes'][p][u][()]*1000).astype(np.int)
        spikeVector = np.zeros(finalTimePoint, dtype='bool')
        spikeVector[spikeTimes[spikeTimes<finalTimePoint]] = 1
        for ifl, (aflashTime, pflashTime) in enumerate(zip(d['behaviorFlashTimes'][()], d['passiveFlashTimes'][()])):
            aflashTime_ms = int(round(aflashTime*1000))
            tensor_active[unitcounter, :, ifl] = spikeVector[aflashTime_ms:aflashTime_ms+750]
            
            pflashTime_ms = int(round(pflashTime*1000))
            tensor_passive[unitcounter, :, ifl] = spikeVector[pflashTime_ms:pflashTime_ms+750]
            
        
        unitcounter+=1
        
        
        
        
# TODO SELECT GOOD TRIALS (OMIT CHANGE and REWARD CONSUMPTION?)
changeTrialIndex = np.isin(d['behaviorFlashTimes'][()], d['behaviorChangeTimes'][()])
postChangeTrialIndex = np.zeros(changeTrialIndex.size, dtype='bool')
postTrialExclusionPeriod = 4
for ct in np.flatnonzero(changeTrialIndex):
    postChangeTrialIndex[ct:ct+postTrialExclusionPeriod] = 1


includeTrialIndex = ~postChangeTrialIndex
tensor_active = tensor_active[:, :, includeTrialIndex]
tensor_passive = tensor_passive[:, :, includeTrialIndex]
imageID = imageID[includeTrialIndex]


saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\ccg tensors"
saveDir = r"C:\Users\svc_ccg\Desktop\Data"
np.savez_compressed(os.path.join(saveDir, '05162019_423749.npz'), imageID=imageID, active_tensor=tensor_active, passive_tensor=tensor_passive, ccfRegions=np.array(unitRegion))
