# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:30:18 2018

@author: svc_ccg
"""

from __future__ import division
import os
import probeSync, analysis_utils
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from analysis_utils import formatFigure
import scipy.stats


def behavior_summary(obj, tag=''):
    preTime = 0.25
    postTime = 1.5
    dataDir = obj.dataDir
    autoRewarded = obj.autoRewarded
    ignore = obj.ignore
    miss = obj.miss
    hit = obj.hit
    falseAlarm = obj.falseAlarm
    correctReject = obj.correctReject
    
    startFrame = int(obj.trials['startframe'][0])
    startTime = obj.frameAppearTimes[startFrame]
    prePostGrayTime = obj.core_data['metadata']['params']['start_stop_padding']
    endFrame = int(obj.trials['endframe'][obj.hit.size-1])
    endTime = obj.frameAppearTimes[endFrame]
    frameTimes = obj.frameAppearTimes
    
    fig = plt.figure(facecolor='w',figsize=(18,10))
    ax = plt.subplot(4,1,1)
    ax.add_patch(patches.Rectangle([startTime-prePostGrayTime,-preTime],width=prePostGrayTime,height=preTime+postTime,color='0.9',alpha=0.5))
    selectedTrials = ~obj.earlyResponse
    changeTimes = frameTimes[np.array(obj.trials['change_frame'][selectedTrials]).astype(int)]
    for trialIndex,t in zip(np.where(selectedTrials)[0],changeTimes):
        licks = frameTimes[np.array(obj.trials['lick_frames'][trialIndex]).astype(int)]-t
        ax.plot(t+np.zeros(licks.size),licks,'o',mec='0.5',mfc='none',ms=3)
        reward = frameTimes[np.array(obj.trials['reward_frames'][trialIndex]).astype(int)]-t
        m = 's' if autoRewarded[trialIndex] else 'o'
        ax.plot(t+np.zeros(reward.size),reward,m,mec='0.5',mfc='0.5',ms=3)
    for resp,clr in zip((hit,miss,falseAlarm,correctReject),'bkrg'):
        ax.plot(changeTimes[resp[selectedTrials]],-preTime/2+np.zeros(resp.sum()),'s',mec=clr,mfc='none',ms=3)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,endTime])
    ax.set_ylim([-preTime,postTime])
    ax.set_ylabel('Time to image change (s)',fontsize=12)
    
    ax = plt.subplot(4,1,2)
    ax.plot(obj.behaviorRunTime,obj.behaviorRunSpeed,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,endTime])
    ax.set_ylabel('Speed',fontsize=12)
    
    ax = plt.subplot(4,1,3)
    for resp,clr,lbl in zip((hit,miss,falseAlarm,correctReject),'bkrg',('hit','miss','false alarm','correct reject')):
        ax.plot(changeTimes,np.cumsum(resp[selectedTrials]),clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,endTime])
    ax.set_ylabel('Count',fontsize=12)
    ax.legend()
    
    ax = plt.subplot(4,1,4)
    window = 300*60
    interval = 60*60
    dframes = np.arange(int(startFrame+window),int(endFrame),int(interval))
    hitProb = np.full(dframes.size,np.nan)
    falseAlarmProb = np.full(dframes.size,np.nan)
    d = np.full(dframes.size,np.nan)
    for i,f in enumerate(dframes):
        h,m,fa,cr = [np.sum((obj.trials['change_frame'][r & (~ignore)]>=f-window) & (obj.trials['change_frame'][r & (~ignore)]<f)) for r in (hit,miss,falseAlarm,correctReject)]
        hitProb[i] = h/(h+m)
        if hitProb[i]==1:
            hitProb[i] = 1-0.5/(h+m)
        elif hitProb[i]==0:
            hitProb[i] = 0.5/(h+m)
        falseAlarmProb[i] = fa/(fa+cr)
        if falseAlarmProb[i]==1:
            falseAlarmProb[i] = 1-0.5/(fa+cr)
        elif falseAlarmProb[i]==0:
            falseAlarmProb[i] = 0.5/(fa+cr)
        d[i] = scipy.stats.norm.ppf(hitProb[i])-scipy.stats.norm.ppf(falseAlarmProb[i])
    ax.plot(frameTimes[dframes],hitProb,'b',label='hit')
    ax.plot(frameTimes[dframes],falseAlarmProb,'r',label='false alarm')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,endTime])
    ax.set_ylim([0,1])
    ax.set_ylabel('Probability',fontsize=12)
    ax.set_xlabel('Time (s)',fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    multipageObj = PdfPages(os.path.join(dataDir, 'BehaviorSummary' + tag + '.pdf'))
    try:
        fig.savefig(multipageObj, format='pdf')
        plt.close(fig)
    finally:
        multipageObj.close()


def all_unit_summary(obj, probesToAnalyze=None, name_tag = ''):
    plt.close('all')
    if probesToAnalyze is None:
        probesToAnalyze = obj.probes_to_analyze
    for pid in probesToAnalyze:
        multipageObj = PdfPages(os.path.join(obj.dataDir, 'SummaryPlots_' + pid + name_tag + '.pdf'))
        try:
            orderedUnits = probeSync.getOrderedUnits(obj.units[pid])
            for u in orderedUnits:
                plot_unit_summary(obj, pid, u, multipageObj)
                plot_unit_behavior_summary(obj, pid, u, multipageObj)
        finally:
            multipageObj.close()

       
def plot_unit_summary(obj, pid, uid, multipageObj=None):
    units = obj.units
    
    spikes = units[pid][uid]['times']
    fig = plt.figure(facecolor='w', figsize=(16,10))
    figtitle = 'Probe ' + str(pid) + ', unit ' + str(uid)
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle += ' , ' + units[pid][uid]['ccfRegion']
    fig.suptitle(figtitle,fontsize=14)
    
    if len(obj.omittedFlashFrames)==0:
        gs = gridspec.GridSpec(9, 7)
        gs.update(top=0.95, bottom = 0.05, left=0.05, right=0.95, wspace=0.3)
        
        imgAxes = [plt.subplot(gs[i,0]) for i in range(8)]
        plot_images(obj, imgAxes)
        
        rfAxes = [plt.subplot(gs[i,1]) for i in range(9)]
        plot_rf(obj, spikes, rfAxes, resp_latency=0.05)
            
        allflashax = [plt.subplot(gs[i,2]) for i in range(9)]
        plot_psth_all_flashes(obj, spikes, allflashax)
        
        hitmissax = [plt.subplot(gs[i,3:]) for i in range(9)]
        plot_psth_hits_vs_misses(obj, spikes, hitmissax)
    
    else:
        gs = gridspec.GridSpec(9, 10)
        gs.update(top=0.95, bottom = 0.05, left=0.05, right=0.95, wspace=0.3)
        
        imgAxes = [plt.subplot(gs[i,0]) for i in range(8)]
        plot_images(obj, imgAxes)
        
        rfAxes = [plt.subplot(gs[i,1]) for i in range(9)]
        plot_rf(obj, spikes, rfAxes, resp_latency=0.05)
            
        allflashax = [plt.subplot(gs[i,2:4]) for i in range(9)]
        sdfs = plot_psth_all_flashes(obj, spikes, allflashax, returnSDFs=True)
        units[pid][uid]['allFlashPSTH'] = sdfs[-1]
        
        omittedflashax = [plt.subplot(gs[i,4:6]) for i in range(9)]
        plot_psth_omitted_flashes(obj, spikes, omittedflashax)        
        
        hitmissax = [plt.subplot(gs[i,6:]) for i in range(9)]
        plot_psth_hits_vs_misses(obj, spikes, hitmissax)
    
    if multipageObj is not None:
        fig.savefig(multipageObj, format='pdf')
        plt.close(fig)
    
    
def plot_unit_behavior_summary(obj, pid, uid, multipageObj=None):
    units = obj.units
    
    spikes = units[pid][uid]['times'] 
    fig = plt.figure(facecolor='w', figsize=(16,12))
    figtitle = 'Probe ' + str(pid) + ', unit ' + str(uid)
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle += ' , ' + units[pid][uid]['ccfRegion']   
    fig.suptitle(figtitle)    
    
    gs_waveform = gridspec.GridSpec(3, 3)
    gs_waveform.update(top=0.95, bottom = 0.35, left=0.05, right=0.95, wspace=0.3)

    plot_spike_template(obj, pid, uid, gs=gs_waveform)
    
    amp_ax = plt.subplot(gs_waveform[:, 2])
    plot_spike_amplitudes(obj, pid, uid, amp_ax)
    
    gs_behavior = gridspec.GridSpec(1, 3)
    gs_behavior.update(top=0.25, bottom = 0.05, left=0.05, right=0.95, wspace=0.3)
    
    lickax = plt.subplot(gs_behavior[0, 0])
    plot_lick_triggered_fr(obj, spikes, lickax)
    
    runax = plt.subplot(gs_behavior[0, 1])
    plot_run_triggered_fr(obj, spikes, runax)
    
    saccadeax = plt.subplot(gs_behavior[0,2])
    plot_saccade_triggered_fr(obj, spikes, saccadeax)
    if multipageObj is not None:
        fig.savefig(multipageObj, format='pdf')
        plt.close(fig)


def plot_images(obj, axes=None):
    if axes is None:
        fig, axes = plt.subplots(8)
        
    for ax,img,imname in zip(axes,obj.imagesDownsampled,obj.imageNames):
        ax.imshow(img,cmap='gray',clim=[0,255])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(imname,fontsize=12)


def plot_rf(obj, spikes, axes=None, resp_latency=0.025, plot=True, returnMat=False):
    if axes is None and plot:
        fig, ax = plt.subplots()
        axes = [ax]
    
    #extract trial stim info (xpos, ypos, ori)
    sweep_table = np.array(obj.rfStimParams['sweep_table'])   #table with rfstim parameters, indexed by sweep order to give stim for each trial
    sweep_order = np.array(obj.rfStimParams['sweep_order'])   #index of stimuli for sweep_table for each trial
    trialRepeats = int(len(sweep_order)/float(len(sweep_table)))
    
    trial_xpos = np.array([pos[0] for pos in sweep_table[sweep_order, 0]])
    trial_ypos = np.array([pos[1] for pos in sweep_table[sweep_order, 0]])
    trial_ori = sweep_table[sweep_order, 3]
    
    xpos = np.unique(trial_xpos)
    ypos = np.unique(trial_ypos)
    ori = np.unique(trial_ori)
    
    respInds = tuple([(np.where(ypos==y)[0][0], np.where(xpos==x)[0][0], np.where(ori==o)[0][0]) for (y,x,o) in zip(trial_ypos, trial_xpos, trial_ori)])
    trial_spikes = analysis_utils.find_spikes_per_trial(spikes, obj.rf_trial_start_times+resp_latency, obj.rf_trial_start_times+resp_latency+0.2)
    respMat = np.zeros([ypos.size, xpos.size, ori.size])
    for (respInd, tspikes) in zip(respInds, trial_spikes):
        respMat[respInd] += tspikes
    
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]

    nullRepeats = 1000
    if returnMat:
        null = [trial_spikes[randinds].sum() for randinds in [np.random.randint(0, trial_spikes.size, trialRepeats) for n in np.arange(nullRepeats)]]
        pMat = np.reshape([scipy.stats.percentileofscore(null, ind) for ind in respMat.flatten()], respMat.shape)
    
    gridSpacingDeg = xpos[1]-xpos[0]
    gridSpacingPix = int(round(obj.imageDownsamplePixPerDeg*gridSpacingDeg))
    r = respMat[:,:,bestOri].copy()
    r -= r.min()
    r /= r.max()
    r = np.repeat(np.repeat(r,gridSpacingPix,axis=0),gridSpacingPix,axis=1)
    rmap = np.zeros(obj.imagesDownsampled[0].shape)
    i,j = (int(rmap.shape[s]/2-r.shape[s]/2) for s in (0,1))
    rmap[i:i+r.shape[0],j:j+r.shape[1]] = r[::-1]
    rmapColor = plt.cm.magma(rmap)[:,:,:3]
    rmapColor *= rmap[:,:,None]
    
    if plot:
        for ax,img,imname in zip(axes[:-1],obj.imagesDownsampled,obj.imageNames):
            img = img.astype(float)
            img /= 255
            img *= 1-rmap
            ax.imshow(rmapColor+img[:,:,None])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(imname,fontsize=12)
            
        axes[-1].imshow(rmap,cmap='magma')
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
    if returnMat:
        return respMat, pMat


def plot_psth_all_flashes(obj, spikes, axes=None, preTime = 0.05, postTime = 0.55, sdfSigma=0.005, returnSDFs=False):
    if axes is None:
        fig, axes = plt.subplots(9)
    
    image_flash_times = obj.frameAppearTimes[np.array(obj.core_data['visual_stimuli']['frame'])]
    image_id = np.array(obj.core_data['visual_stimuli']['image_name'])
    
    sdfs = []
    latencies = []
    for i,img in enumerate(obj.imageNames):
        this_image_times = image_flash_times[image_id==img]
        sdf, t = analysis_utils.getSDF(spikes,this_image_times-preTime,preTime+postTime, sigma=sdfSigma)
        latency = analysis_utils.find_latency(sdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
        latencies.append(latency)
        sdfs.append(sdf)
    
    #plot mean response to all flashes at end
    allsdf, t = analysis_utils.getSDF(spikes,image_flash_times-preTime,preTime+postTime, sigma=sdfSigma)
    latency = analysis_utils.find_latency(allsdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
    latencies.append(latency)
    sdfs.append(allsdf)
        
    ymax = round(max([s.max() for s in sdfs])+0.5)
    for ax,sdf,lat in zip(axes,sdfs,latencies):
        ax.add_patch(patches.Rectangle([0,0],width=obj.behaviorStimDur.mean(),height=ymax,color='0.9',alpha=0.5))
        ax.plot(t-preTime, sdf, 'k')
        if not np.isnan(lat):
            ax.plot(lat/1000-preTime, sdf[lat], 'ro')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0, ymax])
        ax.set_xlim([-preTime,postTime])
        ax.set_yticks([0,ymax])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Time to flash (s)',fontsize=12)
    
    if returnSDFs:
        return sdfs

def plot_psth_omitted_flashes(obj, spikes, axes=None, preTime = 0.05, postTime = 0.55, sdfSigma=0.005):
    if axes is None:
        fig, axes = plt.subplots(9)
    
    image_flash_times = obj.frameAppearTimes[obj.omittedFlashFrames]
    image_id = np.array(obj.omittedFlashImage)
    
    sdfs = []
    latencies = []
    for i,img in enumerate(obj.imageNames):
        this_image_times = image_flash_times[image_id==img]
        sdf, t = analysis_utils.getSDF(spikes,this_image_times-preTime,preTime+postTime, sigma=sdfSigma)
        latency = analysis_utils.find_latency(sdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
        latencies.append(latency)
        sdfs.append(sdf)
    
    #plot mean response to all flashes at end
    allsdf, t = analysis_utils.getSDF(spikes,image_flash_times-preTime,preTime+postTime, sigma=sdfSigma)
    latency = analysis_utils.find_latency(allsdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
    latencies.append(latency)
    sdfs.append(allsdf)
    
    ymax = round(max([s.max() for s in sdfs])+0.5)
    for ax,sdf,lat in zip(axes,sdfs,latencies):
        ax.add_patch(patches.Rectangle([0,0],width=obj.behaviorStimDur.mean(),height=ymax,color='0.9',alpha=0.5))
        ax.plot(t-preTime, sdf, 'k')
        if not np.isnan(lat):
            ax.plot(lat/1000-preTime, sdf[lat], 'ro')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0, ymax])
        ax.set_xlim([-preTime,postTime])
        ax.set_yticks([0,ymax])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Time to omitted flash (s)',fontsize=12)
    
    
    
def plot_psth_hits_vs_misses(obj, spikes, axes=None, preTime=1.55, postTime=4.5, sdfSigma=0.02):
    if axes is None:
        fig, axes = plt.subplots(9)
    
    hitSDFs = []
    missSDFs = []
    for resp,s in zip((obj.hit, obj.miss),(hitSDFs,missSDFs)):
        for img in obj.imageNames:
            selectedTrials = resp & (obj.changeImage==img) & (~obj.ignore)
            changeTimes = obj.frameAppearTimes[np.array(obj.trials['change_frame'][selectedTrials]).astype(int)]
            sdf,t = analysis_utils.getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            s.append(sdf)
        
        #plot mean across images at end
        selectedTrials = resp & (~obj.ignore)
        changeTimes = obj.frameAppearTimes[np.array(obj.trials['change_frame'][selectedTrials]).astype(int)]
        sdf,t = analysis_utils.getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma) 
        s.append(sdf)
    
    ymax = round(max([max(h.max(),m.max()) for h,m in zip(hitSDFs,missSDFs)])+0.5)
    stimdur = obj.behaviorStimDur.mean()
    stimint = stimdur+obj.preGrayDur[0,0]
    stimStarts = np.concatenate((np.arange(-stimint,-preTime,-stimint),np.arange(0,postTime,stimint)))
    for ax,h,m in zip(axes,hitSDFs,missSDFs):
        for s,clr in zip(stimStarts,['0.9']*2+['0.5']*6):
            ax.add_patch(patches.Rectangle([s,0],width=stimdur,height=ymax,color=clr,alpha=0.5))
        ax.plot(t-preTime, h, 'b', linewidth=2)
        ax.plot(t-preTime, m, 'k', linewidth=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0, ymax])
        ax.set_xlim([-preTime,postTime])
        ax.set_yticks([0,ymax])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Time from change (s)',fontsize=12)
    

def plot_spike_amplitudes(obj, pid, uid, axis):
    units = obj.units
    frameTimes = obj.frameAppearTimes
    last_behavior_time = obj.lastBehaviorTime  
    
    
    spikes = units[pid][uid]['times']
    amplitudes = units[pid][uid]['amplitudes']    
    
    num_spikes_to_plot = 1000.
    if spikes.size>num_spikes_to_plot:
        num_spikes_to_skip = int(spikes.size/num_spikes_to_plot)
    else:
        num_spikes_to_skip = 1
        
    axis.plot(amplitudes[::num_spikes_to_skip], frameTimes[-1] - spikes[::num_spikes_to_skip], 'ko', alpha=0.2)
    
    
    axis.set_ylim([0, int(frameTimes[-1])])
    axis.set_yticks([0, int(frameTimes[-1])])
    axis.set_yticklabels([int(frameTimes[-1]), 0])
    axis.plot([axis.get_xlim()[0], axis.get_xlim()[1]], [frameTimes[-1]-last_behavior_time, frameTimes[-1]-last_behavior_time], 'k--')
    formatFigure(plt.gcf(), axis, xLabel='Template Scale Factor', yLabel='Time (s)')
    axis.yaxis.labelpad = -20
    

def plot_spike_template(obj, pid, uid, gs=None):
    units = obj.units    
    
    if gs is None:
        gs = gridspec.GridSpec(3,3)
    template = units[pid][uid]['template']
    tempax = plt.subplot(gs[:, 0])    
    tempax.imshow(template.T, cmap='gray', origin='lower')
    tempax.set_xlim([0, 80])
    tempax.set_xticks([0, 80])
    tempax.set_xticklabels([0, 2.67]) 
    tempax.set_xlabel('Time (ms)')
    tempax.set_ylabel('Channel')
    
    temp_channels = np.where(template>0)
    first_chan = temp_channels[1].min()
    last_chan = temp_channels[1].max()
    temp_inset = template[:, first_chan:last_chan]
    insetax = plt.subplot(gs[0, 1])
    insetax.patch.set_alpha(0.0)
    insetax.imshow(temp_inset.T, cmap='gray', origin='lower')    
    insetax.set_yticks([0, temp_inset.shape[1]])
    insetax.set_yticklabels([first_chan, last_chan])
    insetax.set_xlim([0, 80])
    insetax.set_xticks([0, 80])
    insetax.set_xticklabels([0, 2.67])    
    
    peak_channel_waveform = template[:, units[pid][uid]['peakChan']]
    peakchanax = plt.subplot(gs[1,1])
    peakchanax.plot(peak_channel_waveform, 'k')
    peakchanax.set_xlim([0, 80])
    peakchanax.set_xticks([0, 80])
    peakchanax.set_xticklabels([0, 2.67])     
    
    spike_times = units[pid][uid]['times']
    d = []                   # Distance between any two spike times
    n_sp = len(spike_times)  # Number of spikes in the input spike train
    
    bin_width=0.0005
    width=0.1
    i, j = 0, 0
    for t in spike_times:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spike_times[i] < t - width:
            i += 1
        while j < n_sp and spike_times[j] < t + width:
            j += 1
        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spike_times[i:j] - t)
        
    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b+1, endpoint=True)
    [h, hb] = np.histogram(d, bins=b)
    h[np.ceil(len(h)/2).astype(int)] = 0
                    
    acgax = plt.subplot(gs[2,1])
    acgax.bar(hb[:-1], h, bin_width)
    acgax.set_xlim([-width,width])
    acgax.spines['right'].set_visible(False)
    acgax.spines['top'].set_visible(False)
    acgax.tick_params(direction='out',top=False,right=False)
    acgax.set_yticks([0, h.max()])
    acgax.set_xlabel('Time (s)')
    acgax.set_ylabel('Spike count')


def plot_lick_triggered_fr(obj, spikes, axis=None, min_inter_lick_time = 0.5, preTime=1, postTime=2, plot=True, sdfSigma=0.01, returnSDF=False):
    if axis is None and plot:
        fig, axis = plt.subplots()
    
    frameTimes = obj.frameAppearTimes    
    
    trial_start_frames = np.array(obj.trials['startframe'])
    trial_end_frames = np.array(obj.trials['endframe'])
    trial_start_times = frameTimes[trial_start_frames]
    trial_end_times = frameTimes[trial_end_frames]
    
    lick_times = probeSync.get_sync_line_data(obj.syncDataset, 'lick_sensor')[0]
    first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]
    first_lick_trials = analysis_utils.get_trial_by_time(first_lick_times, trial_start_times, trial_end_times)
    
#    hit = np.array(obj.trials['response_type']=='HIT')
#    earlyResponse = np.array(obj.trials['response_type']=='EARLY_RESPONSE')
#    falseAlarm = np.array(obj.trials['response_type']=='FA')
    hit = obj.hit
    earlyResponse = obj.earlyResponse
    falseAlarm = obj.falseAlarm
    
    hit_lick_times = first_lick_times[np.where(hit[first_lick_trials])[0]]
    bad_lick_times = first_lick_times[np.where(falseAlarm[first_lick_trials] | earlyResponse[first_lick_trials])[0]]
   
    hit_psth, t = analysis_utils.getSDF(spikes,hit_lick_times-preTime,preTime+postTime, sigma=sdfSigma)
    bad_psth, t  = analysis_utils.getSDF(spikes,bad_lick_times-preTime,preTime+postTime, sigma=sdfSigma)
    
    if plot:
        hit, = axis.plot(t-1,hit_psth, 'k')
        bad, = axis.plot(t-1, bad_psth, 'r')
        axis.legend((hit, bad), ('hit', 'aborted/FA'), loc='best', prop={'size':8})
        formatFigure(plt.gcf(), axis, xLabel='Time to lick (s)',  yLabel='Lick-Trig. FR (Hz)')
        axis.plot([0,0], axis.get_ylim(), 'k--')
        
    if returnSDF:
        return hit_psth, bad_psth
    
    
def plot_run_triggered_fr(obj, spikes, axis=None, preTime=1, postTime=2): 
    if axis is None:
        fig, axis = plt.subplots()
        
    if len(obj.behaviorRunStartTimes)>0:
        run_psth, t = analysis_utils.getSDF(spikes,obj.behaviorRunStartTimes-preTime,preTime+postTime)
        axis.plot(t-1,run_psth, 'k')
        axis.plot([0,0], axis.get_ylim(), 'k--')
        formatFigure(plt.gcf(), axis, xLabel='Time to run (s)', yLabel='Run-Trig. FR (Hz)')
    

def plot_saccade_triggered_fr(obj, spikes, axis=None, preTime=2, postTime=2, sdfSigma=0.02, latThresh=5, minPtsAboveThresh=50):
    if obj.eyeData is None:
        return    
    if axis is None:
        fig, axis = plt.subplots()
    
    latFilt = np.ones(minPtsAboveThresh)

    axis.plot([0,0],[0,1000],'k--')
    ymax = 0
    plotlines = []
    for j,(saccades,clr) in enumerate(zip((obj.negSaccades,obj.posSaccades),'rb')):
        saccadeTimes = obj.eyeFrameTimes[saccades]
        sdf,t = analysis_utils.getSDF(spikes,saccadeTimes-preTime,preTime+postTime,sigma=sdfSigma)
        plotline, = axis.plot(t-preTime,sdf,clr)
        plotlines.append(plotline)
        ymax = max(ymax,sdf.max())
        z = sdf-sdf[t<1].mean()
        z /= sdf[t<1].std()
        posLat = np.where(np.correlate(z>latThresh,latFilt,mode='valid')==minPtsAboveThresh)[0]
        posLat = posLat[0] if posLat.size>0 else None
        negLat = np.where(np.correlate(z<-latThresh,latFilt,mode='valid')==minPtsAboveThresh)[0]
        negLat = negLat[0] if negLat.size>0 else None
#            posLat = np.where(z[:np.argmax(z)]<latencyThresh)[0][-1]+1 if z.max()>latencyThresh else None
#            negLat = np.where(z[:np.argmin(z)]>-latencyThresh)[0][-1]+1 if z.min()<-latencyThresh else None
        if posLat is not None or negLat is not None:
            if posLat is None:
                latInd = negLat
            elif negLat is None:
                latInd = posLat
            else:
                latInd = min(posLat,negLat)
            axis.plot(t[latInd]-preTime,sdf[latInd],'o',mfc=clr,mec=clr,ms=10)

    for side in ('right','top'):
        axis.spines[side].set_visible(False)
    axis.tick_params(direction='out',top=False,right=False,labelsize=12)
    axis.set_xlim([-preTime,postTime])
    axis.set_ylim([0,1.02*ymax])
    axis.set_xlabel('Time relative to saccade (s)',fontsize=14)
    axis.set_ylabel('Spike/s',fontsize=14)
    axis.legend((plotlines[0], plotlines[1]), ('temporal', 'nasal'), loc='best', prop={'size':8})
    
    
def plotLFPNoise(obj,agarChRange=None,timeRange=[0,10]):
    for pid in obj.probes_to_analyze:
        t = obj.lfp[pid]['time']
        d = obj.lfp[pid]['data']
        i = (t>=t[0]+timeRange[0]) & (t<=t[0]+timeRange[1])
        if agarChRange is None:
            d = d[i]
        else:
            agar = np.median(d[i,agarChRange[0]:agarChRange[1]],axis=1)
            d = d[i]-agar[:,None]
            
        peakChan = [obj.units[pid][u]['position'][1]/10 for u in probeSync.getOrderedUnits(obj.units[pid])]
        unitCount,bins = np.histogram(peakChan,range(d.shape[1]+1))
        
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax2 = ax.twinx()
        ax2.plot(unitCount,'b')
        ax2.set_ylabel('Unit Count',color='b')
        ax.plot(d.std(axis=0),'k')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Noise')
        
        

