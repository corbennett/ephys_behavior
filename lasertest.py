# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:26:01 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileIO
from sync import sync
import probeSync
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core


dataDir = 'Z:\\'

syncFile = fileIO.getFile('choose sync file',dataDir,'*.h5')
syncDataset = sync.Dataset(syncFile)

pklFile = fileIO.getFile('choose pkl file',dataDir,'*.pkl')
pkl = pd.read_pickle(pklFile)


core_data = data_to_change_detection_core(pkl)
        
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'])


frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'stim_vsync')
vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
monitorLag = 0.036
frameAppearTimes = vsyncTimes + monitorLag  
laserStartTimes = probeSync.get_sync_line_data(syncDataset, 'opto_sweep')[0]


trialLog = pkl['items']['behavior']['trial_log']
laserTrials = pkl['items']['behavior']['layzer_trials']

laserFrameTimes = vsyncTimes[[trial['actual_layzer_frame'] for trial in laserTrials if 'actual_layzer_frame' in trial]]

changeTimes = frameAppearTimes[[trial['stimulus_changes'][0][-1] for trial in trialLog if len(trial['stimulus_changes'])>0]]

changeLog = pkl['items']['behavior']['stimuli']['images']['change_log']

changeTimes = []
laserFrameTimes = []
for trial,laser in zip(trialLog,laserTrials):
    if len(trial['stimulus_changes'])>0:
        changeTimes.append(frameAppearTimes[trial['stimulus_changes'][0][-1]])
        if 'actual_layzer_frame' in laser:
            laser['expected(layzer_flash']-laser['expected_change_flash']
            laserFrameTimes.append(vsyncTimes[laser['actual_layzer_frame']])
        else:
            laserFrameTimes.append(np.nan)
  
changeFrame = []          
for trial in trialLog:
    if len(trial['stimulus_changes'])>0:
        changeFrame.append(trial['stimulus_changes'][0][-1])
    else:
        changeFrame.append(np.nan)
        
actualChangeFrame = [trial['actual_change_frame'] for trial in laserTrials]

[(a,b) for a,b in zip(changeFrame,actualChangeFrame)]
            
            
plt.hist(np.array(laserStartTimes)-changeTimes)



        













