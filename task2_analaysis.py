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

changeLog = pkl['items']['behavior']['stimuli']['grating']['change_log']

licks = pkl['items']['behavior']['lick_sensors'][0]['lick_event']

frameIntervals = pkl['items']['behavior']['intervalsms']



trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])



trialStartTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0]=='trial_start'])

trialEndTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0]=='trial_end'])

scheduledChangeTimes = np.array([event[2] + trial['trial_params']['change_time'] for trial in trialLog for event in trial['events'] if event[0]=='stimulus_window' and event[1]=='enter'])

changeTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0] in ('stimulus_changed','sham_change')])

changeTrials = np.array(['stimulus_changed' in [event[0] for event in trial['events']] for trial in trialLog])

catchTrials = np.array(['sham_change' in [event[0] for event in trial['events']] for trial in trialLog])

abortedTrials = np.array(['abort' in [event[0] for event in trial['events']] for trial in trialLog])

abortTimes = np.array([event[2] for trial in trialLog for event in trial['events'] if event[0]=='abort'])

incompleteTrials = np.array([not any(np.in1d(('stimulus_changed','sham_change'),[event[0] for event in trial['events']])) for trial in trialLog])




# loop through trials and add value or nan for each param



timeToChange = changeTimes-trialStartTimes[~incompleteTrials]

interTrialInterval = trialStartTimes[1:]-trialStartTimes[:-1]

timeFromChangeToTrialEnd = trialEndTimes[~incompleteTrials]-changeTimes

timeFromAbortToNextTrial = trialStartTimes[abortedTrials][1:]-abortTimes[:-1]



fig = plt.figure(figsize=(5,6))
ax = fig.add_subplot(3,1,1)
ax.hist(timeToChange[(changeTrials & ~abortedTrials)[~incompleteTrials]],bins=np.arange(0,10,0.17),color='g',label='change (n='+str(np.sum(changeTrials & ~abortedTrials))+')')
ax.hist(timeToChange[(catchTrials & ~abortedTrials)[~incompleteTrials]],bins=np.arange(0,10,0.17),color='r',label='catch (n='+str(np.sum(catchTrials & ~abortedTrials))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,7])
ax.set_xlabel('Time to change/catch from trial start (s)')
ax.set_ylabel('Number of trials')
ax.set_title(params['stage'])
ax.legend()

ax = fig.add_subplot(3,1,2)
ax.hist(interTrialInterval,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([interTrialInterval.min()-1,interTrialInterval.max()+1])
ax.set_xlabel('Inter-trial interval (s)')
ax.set_ylabel('Number of trials')
plt.tight_layout()

ax = fig.add_subplot(3,1,3)
t = timeFromChangeToTrialEnd[~abortedTrials[~incompleteTrials]]
ax.hist(t,bins=np.arange(0,10,0.17),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([t.min()-1,t.max()+1])
ax.set_xlabel('Time from change/catch to trial end (includes random gray) (s)')
ax.set_ylabel('Number of trials')
plt.tight_layout()




