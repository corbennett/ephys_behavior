# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:22:35 2019

@author: svc_ccg
"""

from __future__ import division
import os
from collections import OrderedDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
import fileIO
import analysis_utils


dirPath = r'C:\Users\svc_ccg\Desktop\Analysis\naive_mice'

mouseIDs = ('460098','463499','472123','472124')


# make population data hdf5
# use nwb for spike times and running data
# use sync and pkl files for flash/change times
for mouse in mouseIDs:
    print(mouse)
    
    syncFile = glob.glob(os.path.join(dirPath,'*_'+mouse+'_*.sync'))[0]
    syncData = sync.Dataset(syncFile)
    frameRising,frameFalling = probeSync.get_sync_line_data(syncData,'stim_vsync')
    vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
    monitorLag = 0.036
    frameAppearTimes = vsyncTimes + monitorLag
    firstFrame = np.where(np.diff(frameAppearTimes)>5)[0][0]+1 # replace this with vsync count from earlier script(s)
    
    pkl = pd.read_pickle(os.path.join(dirPath,'mouse'+mouse+'_replay_script.pkl'))
    vsyncCount = pkl['vsynccount']
    stim = pkl['stimuli'][0]
    frameImages = np.array(stim['sweep_params']['ReplaceImage'][0])
    imageNames = [img for img in np.unique(frameImages) if img is not None]
    nonGrayFrames = np.in1d(frameImages,imageNames)
    flashFrames = np.where(np.diff(nonGrayFrames.astype(int))>0)[0]+1
    flashImages = frameImages[flashFrames]
    changeFrames = np.array([frame for i,frame in enumerate(flashFrames[1:]) if frameImages[frame]!=frameImages[flashFrames[i]]])
    changeImages = frameImages[changeFrames]
    
    flashTimes = frameAppearTimes[flashFrames+firstFrame]
    changeTimes = frameAppearTimes[changeFrames+firstFrame]
    preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
    
    nwb = h5py.File(os.path.join(dirPath,'mouse'+mouse+'.spikes.nwb'))
    
    data = {mouse:{}}
    data[mouse]['flashTimes'] = flashTimes
    data[mouse]['flashImages'] = flashImages
    data[mouse]['changeTimes'] = changeTimes
    data[mouse]['changeImages']= changeImages
    data[mouse]['runTime'] = nwb['acquisition']['timeseries']['RunningSpeed']['timestamps'][:]
    data[mouse]['runSpeed'] = nwb['acquisition']['timeseries']['RunningSpeed']['data'][:]
    data[mouse]['units'] = {}
    data[mouse]['ccfRegions'] = {}
    data[mouse]['spikeTimes'] = {}
    data[mouse]['sdfs'] = {}
    for probe in nwb['processing']:
        print(probe)
        units = nwb['processing'][probe]['unit_list'][:]
        snr,ccf = [np.array([nwb['processing'][probe]['UnitTimes'][str(u)][param].value for u in units]) for param in ('snr','ccf_structure')]
        goodUnits = snr>1
        data[mouse]['units'][probe] = units[goodUnits]
        data[mouse]['ccfRegions'][probe] = ccf[goodUnits]
        data[mouse]['spikeTimes'][probe] = OrderedDict()
        data[mouse]['sdfs'][probe] = {epoch:[] for epoch in ('preChange','change')}
        for u in units[goodUnits]:
            spikes = nwb['processing'][probe]['UnitTimes'][str(u)]['times'][:]
            data[mouse]['spikeTimes'][probe][str(u)] = spikes
            for epoch,times in zip(('preChange','change'),(preChangeTimes,changeTimes)):
                data[mouse]['sdfs'][probe][epoch].append(analysis_utils.getSDF(spikes,times-0.25,1,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0])
    fileIO.objToHDF5(obj=None,saveDict=data,filePath=os.path.join(dirPath,'naiveMiceData2.hdf5'))
    nwb.close()


# make population data hdf5 from experiment nwb files
#for mouse in mouseIDs:
#    print(mouse)
#    nwb = h5py.File(os.path.join(dirPath,'mouse'+mouse+'.spikes.nwb'))
#    data = {mouse:{}}
#    data[mouse]['runTime'] = nwb['acquisition']['timeseries']['RunningSpeed']['timestamps'][:]
#    data[mouse]['runSpeed'] = nwb['acquisition']['timeseries']['RunningSpeed']['data'][:]
#    data[mouse]['flashTimes'] = nwb['stimulus']['presentation']['change_detection_2']['timestamps'][:,0]
#    data[mouse]['flashImages'] = nwb['stimulus']['presentation']['change_detection_2']['data'][:,0].astype(int)
#    changeIndex = np.where(np.diff(data[mouse]['flashImages'])!=0)[0]+1
#    data[mouse]['changeTimes'] = data[mouse]['flashTimes'][changeIndex]
#    data[mouse]['changeImages']= data[mouse]['flashImages'][changeIndex]
#    preChangeTimes = data[mouse]['flashTimes'][changeIndex-1]
#    data[mouse]['units'] = {}
#    data[mouse]['ccfRegions'] = {}
#    data[mouse]['spikeTimes'] = {}
#    data[mouse]['sdfs'] = {}
#    for probe in nwb['processing']:
#        print(probe)
#        units = nwb['processing'][probe]['unit_list'][:]
#        snr,ccf = [np.array([nwb['processing'][probe]['UnitTimes'][str(u)][param].value for u in units]) for param in ('snr','ccf_structure')]
#        goodUnits = snr>1
#        data[mouse]['units'][probe] = units[goodUnits]
#        data[mouse]['ccfRegions'][probe] = ccf[goodUnits]
#        data[mouse]['spikeTimes'][probe] = OrderedDict()
#        data[mouse]['sdfs'][probe] = {epoch:[] for epoch in ('preChange','change')}
#        for u in units[goodUnits]:
#            spikes = nwb['processing'][probe]['UnitTimes'][str(u)]['times'][:]
#            data[mouse]['spikeTimes'][probe][str(u)] = spikes
#            for epoch,times in zip(('preChange','change'),(preChangeTimes,data[mouse]['changeTimes'])):
#                data[mouse]['sdfs'][probe][epoch].append(analysis_utils.getSDF(spikes,times-0.25,1,sampInt=0.001,filt='exp',sigma=0.005,avg=False)[0])
#    fileIO.objToHDF5(obj=None,saveDict=data,filePath=os.path.join(dirPath,'naiveMiceData.hdf5'))
#    nwb.close()


# analysis from population hdf5 data
data = h5py.File(os.path.join(dirPath,'naiveMiceData.hdf5'),'r')

baseWin = slice(0,250)
stimWin = slice(250,500)
respWinOffset = 30
respWin = slice(stimWin.start+respWinOffset,stimWin.stop+respWinOffset)

regionNames = (
               ('LGN',('LGd',)),
               ('V1',('VISp',)),
               ('LM',('VISl',)),
               ('AL',('VISal',)),
               ('RL',('VISrl',)),
               ('PM',('VISpm',)),
               ('AM',('VISam',)),
               ('LP',('LP',)),
              )
regionLabels = [r[1] for r in regionNames]

exps = data.keys() #[data.keys()[2]]

allSdfs = [np.concatenate([data[exp]['sdfs'][probe][epoch][:].mean(axis=1) for exp in exps for probe in data[exp]['sdfs']]) for epoch in ('preChange','change')]

unitRegions = np.concatenate([data[exp]['ccfRegions'][probe][:] for exp in exps for probe in data[exp]['sdfs']])

hasSpikes,hasResp = findResponsiveUnits(allSdfs[1],baseWin,respWin,thresh=5)
preBaseRate,changeBaseRate = [sdfs[:,baseWin].mean(axis=1) for sdfs in allSdfs]
preChangeSdfs,changeSdfs = [sdfs-sdfs[:,baseWin].mean(axis=1)[:,None] for sdfs in allSdfs]

nUnits = []
baseRate= []
cmi = []
cmiCI = []
respLat = []
changeModLat = []
popRespLat = []
popChangeModLat = []
for ind,(region,regionCCFLabels) in enumerate(regionNames):
    inRegion = np.in1d(unitRegions,regionCCFLabels) & hasSpikes & hasResp
    nUnits.append(inRegion.sum())
    
    if nUnits[-1]<1:
        cmi.append(np.nan)
        cmiCI.append([np.nan,np.nan])
        continue
    
    baseRate.append(np.mean(changeBaseRate[inRegion]))
    
    cmod,cmodCI,cmodLat = calcChangeMod(preChangeSdfs[inRegion],changeSdfs[inRegion],respWin,baseWin,stimWin)
    
    cmi.append(cmod)
    cmiCI.append(cmodCI)
    respLat.append(findLatency(changeSdfs[inRegion],baseWin,stimWin))
    changeModLat.append(cmodLat)
    popRespLat.append(findLatency(changeSdfs[inRegion].mean(axis=0),baseWin,stimWin,method='abs',thresh=1)[0])
    popChangeModLat.append(findLatency(np.mean(changeSdfs[inRegion]-preChangeSdfs[inRegion],axis=0),baseWin,stimWin,method='abs',thresh=1)[0])

    fig = plt.figure(figsize=(8,8))
    ax = fig.subplots(1)
    for d,clr in zip((preChangeSdfs[inRegion],changeSdfs[inRegion],changeSdfs[inRegion]-preChangeSdfs[inRegion]),'brk'):
        m = np.mean(d,axis=0)
        s = np.std(d,axis=0)/(inRegion.sum()**0.5)
        ax.plot(m,color=clr)
        ax.fill_between(np.arange(len(m)),m+s,m-s,color=clr,alpha=0.25) 
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([250,600])
    ax.set_xticks([250,350,450,550])
    ax.set_xticklabels([0,100,200,300,400])
    ax.set_ylabel('Spikes/s')
    ax.set_title(region)


np.savez(os.path.join(dirPath,'naiveMiceChangeMod.npz'),cmiNaive=cmi,cmiNaiveCI=cmiCI)



