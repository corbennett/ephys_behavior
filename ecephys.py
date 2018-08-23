# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:40:36 2018

@author: svc_ccg

from the ecephys repo (https://github.com/AllenInstitute/ecephys_pipeline
"""

import numpy as np


def extract_barcodes_from_times(on_times, off_times, inter_barcode_interval=10, 
                                bar_duration=0.03, barcode_duration_ceiling=2, 
                                nbits=32):
    #from ecephys repo
    '''Read barcodes from timestamped rising and falling edges.
    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional 
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode
    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.
    Notes
    -----
    ignores first code in prod (ok, but not intended)
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)
    '''


    start_indices = np.diff(on_times)
    a = np.where(start_indices > inter_barcode_interval)[0]
    barcode_start_times = on_times[a+1]
    
    barcodes = []
    
    for i, t in enumerate(barcode_start_times):
        
        oncode = on_times[np.where(np.logical_and( on_times > t, on_times < t + barcode_duration_ceiling ))[0]]
        offcode = off_times[np.where(np.logical_and( off_times > t, off_times < t + barcode_duration_ceiling ))[0]]
        
        currTime = offcode[0]
        
        bits = np.zeros((nbits,))
        
        for bit in range(0, nbits):
            
            nextOn = np.where(oncode > currTime)[0]
            nextOff = np.where(offcode > currTime)[0]
            
            if nextOn.size > 0:
                nextOn = oncode[nextOn[0]]
            else:
                nextOn = t + inter_barcode_interval
            
            if nextOff.size > 0:
                nextOff = offcode[nextOff[0]]
            else:
                nextOff = t + inter_barcode_interval
            
            if nextOn < nextOff:
                bits[bit] = 1
            
            currTime += bar_duration
            
        barcode = 0        
        
        # least sig left
        for bit in range(0, nbits):
            barcode += bits[bit]*pow(2,bit)
        
        barcodes.append(barcode)
                    
    return barcode_start_times, barcodes

def match_barcodes(master_times, master_barcodes, probe_times, probe_barcodes):
    #from ecephys repo
    '''Given sequences of barcode values and (local) times on a probe line and a master 
    line, find the time points on each clock corresponding to the first and last shared 
    barcode.
    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line. 
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line. 
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    Returns
    -------
    probe_interval : np.ndarray
        Start and end times of the matched interval according to the probe_clock.
    master_interval : np.ndarray
        Start and end times of the matched interval according to the master clock
    '''

    if abs( len(probe_barcodes) - len(master_barcodes) ) < 3:

        if probe_barcodes[0] == master_barcodes[0]:
            t_p_start = probe_times[0]
            t_m_start = master_times[0]
        else:
            t_p_start = probe_times[2]
            t_m_start = master_times[np.where(master_barcodes == probe_barcodes[2])]

        if probe_barcodes[-1] == master_barcodes[-1]:
            t_p_end = probe_times[-1]
            t_m_end = master_times[-1]
        else:
            t_p_end = probe_times[-2]
            t_m_end = master_times[np.where(master_barcodes == probe_barcodes[-2])]

    else:

        for idx, item in enumerate(master_barcodes):

            if item == probe_barcodes[0]:
                print('probe dropped initial barcodes. Start from ' + str(idx))
                t_p_start = probe_times[0]
                t_m_start = master_times[idx]
                
                if probe_barcodes[-1] == master_barcodes[-1]:
                    t_p_end = probe_times[-1]
                    t_m_end = master_times[-1]
                else:
                    t_p_end = probe_times[-2]
                    t_m_end = master_times[np.where(master_barcodes == probe_barcodes[-2])]

                break

    return np.array([t_p_start, t_p_end]), np.array([t_m_start, t_m_end])


def linear_transform_from_intervals(master, probe):
    #from ecephys repo
    '''Find a scale and translation which aligns two 1d segments
    Parameters
    ----------
    master : iterable
        Pair of floats defining the master interval. Order is [start, end].
    probe : iterable
        Pair of floats defining the probe interval. Order is [start, end].
    Returns
    -------
    scale : float
        Scale factor. If > 1.0, the probe clock is running fast compared to the 
        master clock. If < 1.0, the probe clock is running slow.
    translation : float
        If > 0, the probe clock started before the master clock. If > 0, after.
    Notes
    -----
    solves 
        (master + translation) * scale = probe
    for scale and translation
    '''

    scale = (probe[1] - probe[0]) / (master[1] - master[0])
    translation = probe[0] / scale - master[0]

    return scale, translation
    

def get_probe_time_offset(master_times, master_barcodes, 
                          probe_times, probe_barcodes, 
                          acq_start_index, local_probe_rate):
    #from ecephys repo
    """Time offset between master clock and recording probes. For converting probe time to master clock.
    
    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line. 
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line. 
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    acq_start_index : int
        sample index of probe acquisition start time
    local_probe_rate : float
        the probe's apparent sampling rate
    
    Returns
    -------
    total_time_shift : float
        Time at which the probe started acquisition, assessed on 
        the master clock. If < 0, the probe started earlier than the master line.
    probe_rate : float
        The probe's sampling rate, assessed on the master clock
    master_endpoints : iterable
        Defines the start and end times of the sync interval on the master clock
    
    """

    probe_endpoints, master_endpoints = match_barcodes(master_times, master_barcodes, probe_times, probe_barcodes)
    rate_scale, time_offset = linear_transform_from_intervals(master_endpoints, probe_endpoints)

    probe_rate = local_probe_rate * rate_scale
    acq_start_time = acq_start_index / probe_rate

    total_time_shift = time_offset - acq_start_time

    return total_time_shift, probe_rate, master_endpoints
    