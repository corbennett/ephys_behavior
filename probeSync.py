# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 13:08:37 2018

@author: svc_ccg
"""

from __future__ import division
from matplotlib import pyplot as plt
import ecephys
import pandas as pd
import numpy as np
import glob
import os


def getUnitData(dataDir,syncDataset,probeID, probePXIDict, probeGen = '3b'):
    if probeGen is '3a':
        probeDir =  glob.glob(os.path.join(dataDir,'*Probe'+probeID+'_sorted'))[0]   
        probeTTLDir = os.path.join(probeDir,'events\\Neuropix-3a-100.0\\TTL_1')
        probeSpikeDir = os.path.join(probeDir,'continuous\\Neuropix-3a-100.0')
        
    elif probeGen is '3b':
        eventsDir = os.path.join(dataDir, 'events')
        probeTTLDir = os.path.join(os.path.join(eventsDir,'Neuropix-PXI-' + probePXIDict[probeID]), 'TTL_1')
        probeSpikeDir = os.path.join(dataDir, 'Neuropix-PXI-' + probePXIDict[probeID] + '-AP_sortingResults')
    
    print(probeTTLDir)
    print(probeSpikeDir)
    
    #Get barcodes from sync file
    bRising, bFalling = get_sync_line_data(syncDataset, 'barcode')
    bs_t, bs = ecephys.extract_barcodes_from_times(bRising, bFalling)
    
    #Get barcodes from ephys data
    if '03122019' in dataDir and 'slot3' in probeTTLDir: 
        #files on slot3 for this day saved extra bytes at beginning, must skip them to get the right time stamps
        channel_states = np.load(r"Z:\03122019_416656\events\Neuropix-PXI-slot2-probe1\TTL_1\channel_states.npy")
        event_times_file = open(os.path.join(probeTTLDir, 'event_timestamps.npy'), 'rb')
        event_times_file.seek(8*22+1)
        event_times = np.fromfile(event_times_file, dtype='<u8')[:channel_states.size]
    
    elif '06122019' in dataDir:
        good_channel_states = np.load(r"Z:\06122019_423745\events\Neuropix-PXI-slot3-probe1\TTL_1\channel_states.npy")
        good_event_times = np.load(r"Z:\06122019_423745\events\Neuropix-PXI-slot3-probe1\TTL_1\event_timestamps.npy")
        
        channel_states = np.load(os.path.join(probeTTLDir, 'channel_states.npy'))[:good_channel_states.size]
        event_times = np.load(os.path.join(probeTTLDir, 'event_timestamps.npy'))[:good_event_times.size]
            
    else:   
         channel_states = np.load(os.path.join(probeTTLDir, 'channel_states.npy'))
         event_times = np.load(os.path.join(probeTTLDir, 'event_timestamps.npy'))
    
    beRising = event_times[channel_states>0]/30000.
    beFalling = event_times[channel_states<0]/30000.
    be_t, be = ecephys.extract_barcodes_from_times(beRising, beFalling)
    
    if '03212019' in dataDir:
        be_t = be_t[5:]
        be = be[5:]
        
    #Compute time shift between ephys and sync
    shift, p_sampleRate, m_endpoints = ecephys.get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
    
    if '03212019' in dataDir:
        shift = -3.6950408520530686
    #be_t_shifted = (be_t/(p_sampleRate/30000)) - shift #just to check that the shift and scale are right
    
    #Get unit spike times 
    units = load_spike_info(probeSpikeDir, p_sampleRate, shift)
    
    return units


def get_sync_line_data(syncDataset, line_label=None, channel=None):
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
        sortMode: if KS, read in automatically generated labels from Kilosort; if phy read in phy labels
        
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
    print(p_sampleRate)
    print(shift)
    spike_clusters = np.load(os.path.join(spike_data_dir, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(spike_data_dir, 'spike_times.npy'))
    try:
        cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_groups.csv'), sep='\t')
        group = 'group'
    except:
        cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_KSLabel.tsv'), sep='\t')            
        group = 'KSLabel'
#    if sortMode is 'KS':
#        cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_KSLabel.tsv'), sep='\t')
#    elif sortMode is 'phy':
#        cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_groups.csv'), sep='\t')
    templates = np.load(os.path.join(spike_data_dir, 'templates.npy'))
    spike_templates = np.load(os.path.join(spike_data_dir, 'spike_templates.npy'))
    channel_positions = np.load(os.path.join(spike_data_dir, 'channel_positions.npy'))
    amplitudes = np.load(os.path.join(spike_data_dir, 'amplitudes.npy'))
    unit_ids = np.unique(spike_clusters)
    
    units = {}
    for u in unit_ids:
        ukey = str(u)
        units[ukey] = {}
        units[ukey]['label'] = cluster_ids[cluster_ids['cluster_id']==u][group].tolist()[0]
        
        unit_idx = np.where(spike_clusters==u)[0]
        unit_sp_times = spike_times[unit_idx]/p_sampleRate - shift
        
        units[ukey]['times'] = unit_sp_times
        
        #choose 1000 spikes with replacement, then average their templates together
        chosen_spikes = np.random.choice(unit_idx, 1000)
        chosen_templates = spike_templates[chosen_spikes].flatten()
        units[ukey]['template'] = np.mean(templates[chosen_templates], axis=0)
        units[ukey]['peakChan'] = np.unravel_index(np.argmin(units[ukey]['template']), units[ukey]['template'].shape)[1]
        units[ukey]['position'] = channel_positions[units[ukey]['peakChan']]
        units[ukey]['amplitudes'] = amplitudes[unit_idx]
        
        #check if this unit is noise
        peakChan = units[ukey]['peakChan']
        temp = units[ukey]['template'][:, peakChan]
        pt = findPeakToTrough(temp, plot=False)
        units[ukey]['peakToTrough'] = pt
        tempNorm = temp/np.max(np.abs([temp.min(), temp.max()]))
        units[ukey]['normTempIntegral'] = tempNorm.sum()
        if abs(tempNorm.sum())>4:
            units[ukey]['label'] = 'noise'
#            plt.figure(u)
#            plt.plot(temp)
        
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
    
    
def getLFPData(dataDir, pid, syncDataset, probePXIDict, probeGen = '3b', num_channels=384):
    
    if '3a' in probeGen:
        lfp_data_dir =  glob.glob(os.path.join(dataDir,'*probe'+pid+'_sorted','continuous','Neuropix-3a-100.1'))[0]
        events_dir = glob.glob(os.path.join(dataDir,'*probe'+pid+'_sorted','events','Neuropix-3a-100.0','TTL_1'))[0]
    elif '3b' in probeGen:
        probeDirName = 'Neuropix-PXI-'+probePXIDict[pid]
        lfp_data_dir = os.path.join(dataDir,probeDirName+'-LFP')
        events_dir = os.path.join(dataDir,'events',probeDirName,'TTL_1')
    
    lfp_data_file = os.path.join(lfp_data_dir, 'continuous.dat') 
        
    if not os.path.exists(lfp_data_file):
        print('Could not find LFP data at ' + lfp_data_file)
        return None,None
    
    lfp_data = np.memmap(lfp_data_file, dtype='int16', mode='r')    
    lfp_data_reshape = np.reshape(lfp_data, [int(lfp_data.size/num_channels), -1])
    
    time_stamps = np.load(os.path.join(lfp_data_dir, 'lfp_timestamps.npy'))    
    
    #Get barcodes from sync file
    bRising, bFalling = get_sync_line_data(syncDataset, 'barcode')
    bs_t, bs = ecephys.extract_barcodes_from_times(bRising, bFalling)
    
    #Get barcodes from ephys data
    #Get barcodes from ephys data
    if '03122019' in dataDir and 'slot3' in events_dir: 
        #files on slot3 for this day saved extra bytes at beginning, must skip them to get the right time stamps
        channel_states = np.load(r"Z:\03122019_416656\events\Neuropix-PXI-slot2-probe1\TTL_1\channel_states.npy")
        event_times_file = open(os.path.join(events_dir, 'event_timestamps.npy'), 'rb')
        event_times_file.seek(8*22+1)
        event_times = np.fromfile(event_times_file, dtype='<u8')[:channel_states.size]
        lfp_data_reshape = lfp_data_reshape[:time_stamps.size]
    elif '06122019' in dataDir:
        good_channel_states = np.load(r"Z:\06122019_423745\events\Neuropix-PXI-slot3-probe1\TTL_1\channel_states.npy")
        good_event_times = np.load(r"Z:\06122019_423745\events\Neuropix-PXI-slot3-probe1\TTL_1\event_timestamps.npy")
        
        channel_states = np.load(os.path.join(events_dir, 'channel_states.npy'))[:good_channel_states.size]
        event_times = np.load(os.path.join(events_dir, 'event_timestamps.npy'))[:good_event_times.size]
    else:   
         channel_states = np.load(os.path.join(events_dir, 'channel_states.npy'))
         event_times = np.load(os.path.join(events_dir, 'event_timestamps.npy'))
    
    beRising = event_times[channel_states>0]/30000.
    beFalling = event_times[channel_states<0]/30000.
    be_t, be = ecephys.extract_barcodes_from_times(beRising, beFalling)
    
    if '03212019' in dataDir:
        be_t = be_t[5:]
        be = be[5:]
    
    #Compute time shift between ephys and sync
    shift, p_sampleRate, m_endpoints = ecephys.get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
    
    if '03212019' in dataDir:
        shift = -3.6950408520530686    
    
    
    time_stamps_shifted = (time_stamps/p_sampleRate) - shift
    
    return lfp_data_reshape, time_stamps_shifted


def findPeakToTrough(waveformArray, sampleRate=30000, plot=True):
    #waveform array should be units x samples
    if waveformArray.ndim==1:
        waveformArray=waveformArray[None,:]
    
    peakToTrough = np.zeros(len(waveformArray))       
    for iw, w in enumerate(waveformArray):
#        peakInd = np.argmax(np.absolute(w))
#        peakToTrough[iw] = (np.argmin(w[peakInd:]) if w[peakInd]>0 else np.argmax(w[peakInd:]))/(sampleRate/1000.0)
        if any(np.isnan(w)):
            peakToTrough[iw] = np.nan
        else:
            peakInd = np.argmin(w)
            peakToTrough[iw] = np.argmax(w[peakInd:])/(sampleRate/1000.0)       
    
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.hist(peakToTrough[~np.isnan(peakToTrough)],np.arange(0,1.2,0.05),color='k')
        ax.plot([0.35]*2,[0,180],'k--')
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(which='both',direction='out',top=False,right=False,labelsize=18)
        ax.set_xlabel('Spike peak-to-trough (ms)',fontsize=20)
        ax.set_ylabel('# Units',fontsize=20)
        plt.tight_layout()
        
    if len(peakToTrough)==1:
        peakToTrough = peakToTrough[0]
     
    return peakToTrough
    
    
