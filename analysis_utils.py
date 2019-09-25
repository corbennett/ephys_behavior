# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:40:38 2018

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
from numba import njit


#@njit
def getSDF(spikes,startTimes,windowDur,sampInt=0.001,filt='gaussian',sigma=0.02,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
            
        if filt in ('exp','exponential'):
            filtPts = int(5*sigma/sampInt)
            expFilt = np.zeros(filtPts*2)
            expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=sigma/sampInt,sym=False)
            expFilt /= expFilt.sum()
            sdf = scipy.ndimage.filters.convolve1d(counts,expFilt,axis=1)
        else:
            sdf = scipy.ndimage.filters.gaussian_filter1d(counts,sigma/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf,t[:-1]

@njit     
def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001, convolution_kernel=0.05, avg=True):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel/2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0,windowDur+binSize,binSize)
    convkernel = np.ones(int(convolution_kernel/binSize))
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    counts = np.convolve(counts, convkernel)/(binSize*convkernel.size)
    return counts[convkernel.size-1:-convkernel.size], bins[:-convkernel.size-1]


def makePSTH(spikes,startTimes,windowDur,binSize=0.01, avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]

    if avg:
        return counts.mean(axis=0)/binSize

    else:
        return np.array(counts)/binSize

    


@njit
def get_ccg(spikes1, spikes2, width=0.1, bin_width=0.001, num_jitter=5, jitter_win=0.02):

    d = []
    djit = []             # Distance between any two spike times
    n_sp = len(spikes2)     # Number of spikes in the input spike train
    
    jitter = np.random.random((num_jitter+1, spikes1.size))*(2*jitter_win) - jitter_win
    jitter[0] = np.zeros(spikes1.size)

    for jit in xrange(num_jitter):
        spikes1_j = spikes1+jitter[jit]
        i, j = 0, 0
        for t in spikes1_j:
            # For each spike we only consider those spikes times that are at most
            # at a 'width' time lag. This requires finding the indices
            # associated with the limiting spikes.
            while i < n_sp and spikes2[i] < t - width:
                i += 1
            while j < n_sp and spikes2[j] < t + width:
                j += 1
    
            # Once the relevant spikes are found, add the time differences
            # to the list
            if jit==0:
                d.extend(spikes2[i:j] - t)
            else:
                djit.extend(spikes2[i:j] - t)
                
    return d, djit
    
@njit
def get_ccg_corr(s1, s2, width=1, bin_width=0.001):
    num_steps = np.int(width/bin_width)
    shifts = np.linspace(-num_steps, num_steps, 2*num_steps+1)
    
    corr = np.zeros(shifts.size)
    for i,shift in enumerate(shifts):
#        corr[i] = np.dot(s1, np.roll(s2,np.int(shift)))
        corr[i] = (s1*np.roll(s2,np.int(shift))).sum()
    
    return corr
    
    
def plot_ccg(spikes1, spikes2, auto=False, width=0.1, bin_width=0.001, plot=False):
    spikes1 = spikes1.flatten() 
    spikes2 = spikes2.flatten()
    d = np.array(get_ccg(spikes1, spikes2, width=width, bin_width=bin_width)[0])
    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b+1, endpoint=True)
    [h, hb] = np.histogram(d, bins=b)
    hh = h.astype(np.float)/(len(spikes1)*len(spikes2))**0.5
    
    if auto:
        hh[n_b-1] = 0 #mask the 0 bin for autocorrelations
    if plot:          
        fig,ax = plt.subplots()
        ax.bar(hb[:-1], hh, bin_width)
        ax.set_xlim([-width,width])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
        
    return hh, hb

def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    tsinds = np.searchsorted(spikes[:,0], trial_starts)
    teinds = np.searchsorted(spikes[:,0], trial_ends)
    
    return teinds - tsinds


def find_run_transitions(run_signal, run_time, thresh = [1,5], smooth_kernel = 0.5, inter_run_interval = 2, min_run_duration = 3, sample_freq = 60):
    smooth_kernel = round(smooth_kernel*sample_freq)
    smooth_kernel = smooth_kernel if np.mod(smooth_kernel, 2) == 1 else smooth_kernel + 1 #must be an odd number for median filter
    run_speed_smooth =  scipy.signal.medfilt(run_signal, int(smooth_kernel))
    run_samples = np.where(run_speed_smooth>=thresh[1])[0]
    if len(run_samples)==0:
        run_start_times = []
    else:
        run_starts = run_samples[np.insert(np.diff(run_samples)>=inter_run_interval*sample_freq, 0, True)]
        
        adjusted_rs = []
        for rs in run_starts:
            last_stat_points = np.where(run_speed_smooth[:rs]<=thresh[0])[0]
            if len(last_stat_points)>0:
                adjusted = (last_stat_points[-1])
            else:
                adjusted = rs
            
            if np.median(run_speed_smooth[adjusted:adjusted+min_run_duration*sample_freq]) > thresh[1]:
                adjusted_rs.append(adjusted)
        
        adjusted_rs = np.array(adjusted_rs).astype(np.int)
        run_start_times = run_time[adjusted_rs]
    
    return run_start_times
    
    
def find_latency(signal, baseline_end = 100, stdev_thresh = 3, min_points_above_thresh=30):
    try:
        thresh = stdev_thresh*np.std(signal[:baseline_end]) + np.mean(signal[:baseline_end])   
        over_std = np.where(signal>thresh)[0]
    
        if len(over_std)==0:
            return np.nan
        
        counter = 1
        cand = over_std[0]
        while any(signal[cand:cand+min_points_above_thresh]<thresh):
            cand = over_std[counter]
            counter += 1
            if counter==len(over_std):
                return np.nan
        
        return cand
    except:
        return np.nan
    
    
def get_trial_by_time(times, trial_start_times, trial_end_times):
    trials = []
    for time in times:
        if trial_start_times[0]<=time<trial_end_times[-1]:
            trial = np.where((trial_start_times<=time)&(trial_end_times>time))[0][0]
        else:
            trial = -1
        trials.append(trial)
    
    return np.array(trials)
    

def calculate_lifetime_sparseness(mean_response_vector):
    '''lifetime sparseness as used in marina's biorxiv paper (defined by Gallant)
    mean_response_vector (len n) should contain the trial mean of a cell's response 
    (however defined) over n conditions'''
    
    sumsquared = float(np.sum(mean_response_vector)**2)
    sum_of_squares = float(np.sum(mean_response_vector**2))
    n = float(mean_response_vector.size)
    
    try:
        num = 1 - (1/n)*(sumsquared/sum_of_squares)
        denom = 1 - (1/n)
        
        ls = num/denom
    except:
        ls = np.nan
    
    return ls


def formatFigure(fig, ax, title=None, xLabel=None, yLabel=None, xTickLabels=None, yTickLabels=None, blackBackground=False, saveName=None):
    fig.set_facecolor('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
        
    if blackBackground:
        ax.set_axis_bgcolor('k')
        ax.tick_params(labelcolor='w', color='w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        for side in ('left','bottom'):
            ax.spines[side].set_color('w')

        fig.set_facecolor('k')
        fig.patch.set_facecolor('k')
    if saveName is not None:
        fig.savefig(saveName, facecolor=fig.get_facecolor())

    