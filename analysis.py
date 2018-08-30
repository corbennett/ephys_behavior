# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import glob
from sync import sync
import probeSync
import behavSync
import numpy as np
import matplotlib.pyplot as plt
from probeData import formatFigure
from visual_behavior.visualization.extended_trials.daily import make_daily_figure


dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\08152018_385531"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)

units = probeSync.getUnitData(dataDir,syncDataset)

pkl_file = glob.glob(os.path.join(dataDir, '*.pkl'))[0]
trials, frameRising, frameFalling, runTime, runSpeed = behavSync.getBehavData(pkl_file,syncDataset)

make_daily_figure(trials)

#align trials to sync
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameRising[trial_start_frames]
trial_end_times = frameFalling[trial_end_frames]

trial_ori = np.array(trials['initial_ori'])
    
abortedTrials = trials['change_frame'].isnull()
change_frames = np.array(trials['change_frame'][~abortedTrials]).astype(int)
change_times = frameRising[change_frames]
change_ori = np.array(trials['change_ori'])[~abortedTrials]


#make psth for units

def makePSTH(spike_times, trial_start_times, trial_duration, bin_size = 0.1):
    counts = np.zeros(int(trial_duration/bin_size))    
    for ts in trial_start_times:
        for ib, b in enumerate(np.arange(ts, ts+trial_duration, bin_size)):
            c = np.sum((spike_times>=b) & (spike_times<b+bin_size))
            counts[ib] += c
    return counts/len(trial_start_times)/bin_size

traceTime = np.linspace(-2, 10, 120)
goodUnits = probeSync.getOrderedUnits(units)
for u in goodUnits:
    spikes = units[u]['times']
    psthVert = makePSTH(spikes, change_times[np.logical_or(change_ori==90, change_ori==270)]-2, 12)
    psthHorz = makePSTH(spikes, change_times[np.logical_or(change_ori==0, change_ori==180)]-2, 12)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(str(u) + ': ' + str(units[u]['peakChan']))
    ax[0].plot(traceTime, psthVert)
    ax[1].plot(traceTime, psthHorz)
    for a in ax:    
        formatFigure(fig, a, '', 'time, s', 'FR, Hz')
    

#Make summary pdf of unit responses    
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
multipage(os.path.join(dataDir, 'behaviorPSTHs_08022018.pdf'))




def get_ccg(spikes1, spikes2, auto=False, width=0.1, bin_width=0.0005, plot=True):

    d = []                   # Distance between any two spike times
    n_sp = len(spikes2)  # Number of spikes in the input spike train

    
    i, j = 0, 0
    for t in spikes1:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spikes2[i] < t - width:
            i += 1
        while j < n_sp and spikes2[j] < t + width:
            j += 1

        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spikes2[i:j] - t)

    
    d = np.array(d)
    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b, endpoint=True)
    [h, hb] = np.histogram(d, bins=b)
    hh = h.astype(np.float)/(len(spikes1)*len(spikes2))**0.5
    
    if auto:
        hh[n_b-1] = 0
    if plot:          
        fig,ax = plt.subplots()
        ax.bar(hb[:-1], hh, bin_width)
        ax.set_xlim([-width,width])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
        
    return hh, hb
    
    
    

