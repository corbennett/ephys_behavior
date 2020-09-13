# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:52:39 2020

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
import fileIO


probeDataDir = fileIO.getDir()

sampleRate = 30000

nChannels = 384

rawData = np.memmap(os.path.join(probeDataDir,'continuous.dat'),dtype='int16',mode='r')    
rawData = np.reshape(rawData,(int(rawData.size/nChannels),-1)).T
totalSamples = rawData.shape[1]

kilosortData = {key: np.load(os.path.join(probeDataDir,key+'.npy')) for key in ('spike_clusters',
                                                                                'spike_times',
                                                                                'templates',
                                                                                'spike_templates',
                                                                                'channel_positions',
                                                                                'amplitudes')}

clusterIDs = pd.read_csv(os.path.join(probeDataDir,'cluster_KSLabel.tsv'),sep='\t')

unitIDs = np.unique(kilosortData['spike_clusters'])

units = {}
for u in unitIDs:
    units[u] = {}
    units[u]['label'] = clusterIDs[clusterIDs['cluster_id']==u]['KSLabel'].tolist()[0]
    
    uind = np.where(kilosortData['spike_clusters']==u)[0]
    
    units[u]['samples'] = kilosortData['spike_times'][uind]
    
    units[u]['rate'] = units[u]['samples'].size/totalSamples*sampleRate
    
    #choose 1000 spikes with replacement, then average their templates together
    chosen_spikes = np.random.choice(uind,1000)
    chosen_templates = kilosortData['spike_templates'][chosen_spikes].flatten()
    units[u]['template'] = np.mean(kilosortData['templates'][chosen_templates],axis=0).T
    
    peakChan = np.unravel_index(np.argmin(units[u]['template']),units[u]['template'].shape)[0]
    units[u]['peakChan'] = peakChan
    units[u]['position'] = kilosortData['channel_positions'][peakChan]
    units[u]['amplitudes'] = kilosortData['amplitudes'][uind]
    
    peakTemplate = units[u]['template'][peakChan]
    if any(np.isnan(peakTemplate)):
        units[u]['peakToTrough'] = np.nan
    else:
        peakInd = np.argmin(peakTemplate)
        units[u]['peakToTrough'] = np.argmax(peakTemplate[peakInd:])/(sampleRate/1000)
    
    #check if this unit is noise
    tempNorm = peakTemplate/np.max(np.absolute(peakTemplate))
    units[u]['normTempIntegral'] = tempNorm.sum()
    if abs(tempNorm.sum())>4:
        units[u]['label'] = 'noise'


goodUnits = np.array([u for u in units if units[u]['label']=='good' and units[u]['rate']>0.1])
goodUnits = goodUnits[np.argsort([units[u]['peakChan'] for u in goodUnits])]

probeCols = 8
probeRows = 48
channelSpacing = 6 # microns
probeX = np.arange(probeCols)*channelSpacing
probeY = np.arange(probeRows)*channelSpacing
for u in goodUnits:
    fig = plt.figure(figsize=(12,12))
    gs = matplotlib.gridspec.GridSpec(probeRows,probeCols)
    template = units[u]['template']
    ymin = template.min()
    ymax = template.max()
    for ind,ch in enumerate(template):
        chX,chY = kilosortData['channel_positions'][ind]
        i = np.where(probeY==chY)[0][0]
        j = np.where(probeX==chX)[0][0]
        ax = fig.add_subplot(gs[i,j])
        clr = 'r' if ind==units[u]['peakChan'] else 'k'
        ax.plot(ch,color=clr,lw=2)
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([31,52])
        ax.set_ylim([ymin,ymax])
    plt.tight_layout()




