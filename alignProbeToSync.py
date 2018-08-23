# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 13:08:37 2018

@author: svc_ccg
"""

from __future__ import division
import ecephys
from sync import sync
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.validation.extended_trials import *
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from load_phy_template import load_phy_template
import fileIO
from probeData import formatFigure


def get_sync_line_data(dataset, line_label=None, channel=None):
    ''' Get rising and falling edge times for a particular line from the sync h5 file
        
        Parameters
        ----------
        dataset: sync file dataset generated by sync.Dataset
        line_label: string specifying which line to read, if that line was labelled during acquisition
        channel: integer specifying which channel to read in line wasn't labelled
        
        Returns
        ----------
        rising: npy array with rising edge times for specified line
        falling: falling edge times
    '''
    if isinstance(line_label, str):
        try:
            channel = syncDataset.line_labels.index(line_label)
        except:
            print('Invalid line label')
            return
    elif channel is None:
        print('Must specify either line label or channel id')
        return
    
    sample_freq = syncDataset.meta_data['ni_daq']['counter_output_freq']
    rising = syncDataset.get_rising_edges(channel)/sample_freq
    falling = syncDataset.get_falling_edges(channel)/sample_freq
    
    return rising, falling


def load_spike_info(spike_data_dir, p_sampleRate, shift):
    ''' Make dictionary with spike times, templates, sorting label and peak channel for all units
    
        Parameters
        -----------
        spike_data_dir: path to directory with clustering output files
        p_sampleRate: probe sampling rate according to master clock
        shift: time shift between master and probe clock
        p_sampleRate and shift are outputs from 'get_probe_time_offset' function
        
        Returns
        ----------
        units: dictionary with spike info for all units
            each unit is integer key, so units[0] is a dictionary for spike cluster 0 with keys
            'label': sorting label for unit, eg 'good', 'mua', or 'noise'
            'times': spike times in seconds according to master clock
            'template': spike template, should be replaced by waveform extracted from raw data
                averaged over 1000 randomly chosen spikes
            'peakChan': channel where spike template has minimum, used to approximate unit location
    '''
    
    spike_clusters = np.load(os.path.join(spike_data_dir, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(spike_data_dir, 'spike_times.npy'))
    cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_groups.csv'), sep='\t')
    templates = np.load(os.path.join(spike_data_dir, 'templates.npy'))
    spike_templates = np.load(os.path.join(spike_data_dir, 'spike_templates.npy'))
    unit_ids = np.unique(spike_clusters)
    
    units = {}
    for u in unit_ids:
        units[u] = {}
        units[u]['label'] = cluster_ids[cluster_ids['cluster_id']==u]['group'].tolist()[0]
        
        unit_idx = np.where(spike_clusters==u)[0]
        unit_sp_times = spike_times[unit_idx]/p_sampleRate - shift
        
        units[u]['times'] = unit_sp_times
        
        #choose 1000 spikes with replacement, then average their templates together
        chosen_spikes = np.random.choice(unit_idx, 1000)
        chosen_templates = spike_templates[chosen_spikes].flatten()
        units[u]['template'] = np.mean(templates[chosen_templates], axis=0)
        units[u]['peakChan'] = np.unravel_index(np.argmin(units[u]['template']), units[u]['template'].shape)[1]
    return units


def getOrderedUnits(units, label=['good']):
    '''Returns unit ids according to sorting label (default is only the 'good'
    units) and probe position (tip first)
    
    Parameters
    ----------
    units: unit dictionary
    label: list of labels to include
    
    Returns
    ---------
    ordered list of unit ids
    '''
    goodUnits = np.array([key for key in units if units[key]['label'] in label])
    peakChans = [units[u]['peakChan'] for u in goodUnits]
    return goodUnits[np.argsort(peakChans)]
    
    
def makePSTH(spike_times, trial_start_times, trial_duration, bin_size = 0.1):
    counts = np.zeros(int(trial_duration/bin_size))    
    for ts in trial_start_times:
        for ib, b in enumerate(np.arange(ts, ts+trial_duration, bin_size)):
            c = np.sum((spike_times>=b) & (spike_times<b+bin_size))
            counts[ib] += c
    return counts/len(trial_start_times)/bin_size



#dataDir = fileIO.getDir()
dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\08152018_385531"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)

#Get barcodes from sync file
bRising, bFalling = get_sync_line_data(syncDataset, 'barcode')
bs_t, bs = ecephys.extract_barcodes_from_times(bRising, bFalling)

#Get barcodes from ephys data
channel_states = np.load(os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'events\\Neuropix-3a-100.0\\TTL_1\\channel_states.npy'))
event_times = np.load(os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'events\\Neuropix-3a-100.0\\TTL_1\\event_timestamps.npy'))

beRising = event_times[channel_states>0]/30000.
beFalling = event_times[channel_states<0]/30000.
be_t, be = ecephys.extract_barcodes_from_times(beRising, beFalling)

#Compute time shift between ephys and sync
shift, p_sampleRate, m_endpoints = ecephys.get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
be_t_shifted = (be_t/(p_sampleRate/30000)) - shift #just to check that the shift and scale are right

#Get unit spike times 
spike_data_dir = os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'continuous\\Neuropix-3a-100.0')
units = load_spike_info(spike_data_dir, p_sampleRate, shift)

#Get frame times from sync file
frameRising, frameFalling = get_sync_line_data(syncDataset, 'stim_vsync')

#Get frame times from pkl behavior file
pkl_file = glob.glob(os.path.join(dataDir, '*.pkl'))[0]
behaviordata = pd.read_pickle(pkl_file)
core_data = data_to_change_detection_core(behaviordata)
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'])

#align trials to clock
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameRising[trial_start_frames]
trial_end_times = frameFalling[trial_end_frames]
trial_ori = np.array(trials['initial_ori'])

notNullTrials = trials['change_frame'].notnull()
change_frames = np.array(trials['change_frame'][notNullTrials]).astype(int)
change_times = frameRising[change_frames]
change_ori = np.array(trials['change_ori'])[notNullTrials]


#make psth for units
traceTime = np.linspace(-2, 10, 120)
goodUnits = getOrderedUnits(units)
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




