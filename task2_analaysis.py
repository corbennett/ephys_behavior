# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:47:06 2020

@author: svc_ccg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileIO


f = fileIO.getFile()
pkl = pd.read_pickle(f)

params = pkl['items']['behavior']['params']

#sorted(params.items())



trialLog = pkl['items']['behavior']['trial_log']
changeLog = pkl['items']['behavior']['stimuli']['grating']['change_log']


changeTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0] in ('stimulus_changed','sham_change')])

trialStartTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0]=='trial_start'])[:len(changeTimes)]

changeTrials = np.array(['stimulus_changed' in [event[0] for event in trial['events']] for trial in trialLog])[:len(changeTimes)]

catchTrials = np.array(['sham_change' in [event[0] for event in trial['events']] for trial in trialLog])[:len(changeTimes)]

assert(changeTrials.sum()+catchTrials.sum()==len(changeTimes))


timeToChange = changeTimes-trialStartTimes

interTrialInterval = trialStartTimes[1:]-changeTimes[:-1]



fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(2,1,1)
ax.hist(timeToChange[changeTrials],bins=np.arange(0,10,0.17),color='g',label='change (n='+str(changeTrials.sum())+')')
ax.hist(timeToChange[catchTrials],bins=np.arange(0,10,0.17),color='r',label='catch (n='+str(catchTrials.sum())+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,7])
ax.set_xlabel('Time to change/catch from trial start (s)')
ax.set_ylabel('Number of trials')
ax.set_title(params['stage'])
ax.legend()

ax = fig.add_subplot(2,1,2)
ax.hist(interTrialInterval,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([interTrialInterval.min()-1,interTrialInterval.max()+1])
ax.set_xlabel('Time from change/catch to start of next trial (s)')
ax.set_ylabel('Number of trials')
plt.tight_layout()




