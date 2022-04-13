# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 12:48:27 2019

@author: svc_ccg
"""
import getData
import glob, os
import numpy as np
from analysis_utils import *
from matplotlib import pyplot as plt
import probeSync
import summaryPlots
import pandas as pd

experiments = ['03212019_409096', '03262019_417882', '03272019_417882', '04042019_408528', '04052019_408528', '04102019_408527','04112019_408527', 
    '04252019_421323', '04262019_421323', '04302019_422856', '05162019_423749', '05172019_423749']

hdf5dir = r"C:\Users\svc_ccg\Desktop\Data\analysis"
rds = []
for exp in experiments:
    print(exp)
    b = getData.behaviorEphys('Z:\\' + exp)
    
    h5FilePath = glob.glob(os.path.join(hdf5dir, exp+'*'))[0]
    b.loadFromHDF5(h5FilePath) 
    
    frameAppearTimes = b.passiveFrameAppearTimes
    
    selectedTrials = (b.hit | b.miss)&(~b.ignore)
    changeTimes = frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one to correct for change frame indexing problem
    image_flash_times = frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    image_id = np.array(b.core_data['visual_stimuli']['image_name'])
    preChangeIndices = np.searchsorted(image_flash_times, changeTimes)-1
    preChangeTimes = image_flash_times[preChangeIndices]

    
    regionsOfInterest = ['VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl', 'VISpor', 'VISli']
    
    preTime = 0.25
    postTime = 0.5
    regionDict = {r: {a:[] for a in ['changePSTH', 'preChangePSTH']} for r in regionsOfInterest}
    regionDict['experiment'] = exp
    for region in regionsOfInterest:
        pids, us = b.getUnitsByArea(region)
        
        if len(pids)>0:
            for pid, u in zip(pids,us):
                spikes = b.units[pid][u]['times']
                
                if np.sum(spikes<3600)<3600: #exclude cells that fire below 1Hz during task
                    continue
    
                changesdf, time = getSDF(spikes, changeTimes-preTime, preTime+postTime, sigma=0.001)            
                regionDict[region]['changePSTH'].append(changesdf)
    
                prechangesdf, time = getSDF(spikes, preChangeTimes-preTime, preTime+postTime, sigma=0.001)            
                regionDict[region]['preChangePSTH'].append(prechangesdf)
    
    rds.append(regionDict)

plt.figure()
allregionDict = {r: {a:[] for a in ['changePSTH', 'preChangePSTH']} for r in regionsOfInterest}   
for region in regionsOfInterest:
    meanChangeModPerExperiment = []
    for ir, r in enumerate(rds):
        allregionDict[region]['changePSTH'].extend(r[region]['changePSTH'])
        allregionDict[region]['preChangePSTH'].extend(r[region]['preChangePSTH'])    
#        
#        print(region + ': ' + str(len(r[region]['changePSTH'])))
#        if len(r[region]['changePSTH'])>0:
#            change = np.array(r[region]['changePSTH'])
#            prechange = np.array(r[region]['preChangePSTH'])    
#            
#            change_sub = change - np.mean(change[:, :200], axis=1)[:, None]
#            change_norm = change_sub/np.max(change_sub, axis=1)[:, None]
#            
#            prechange_sub = prechange - np.mean(prechange[:, :200], axis=1)[:, None]
#            prechange_norm = prechange_sub/np.max(change_sub, axis=1)[:, None]  
#            
#            diff = change_sub - prechange_sub
#            changeMod = np.log2(diff.max(axis=1)/prechange_sub.max(axis=1))
#            
#            meanChangeModPerExperiment.append(changeMod.mean())
#            print('change mod: ' + str(changeMod.mean()))
#            
#            changeMean = np.mean(change_sub, axis=0)
#            preChangeMean = np.mean(prechange_sub, axis=0)
#            diffMean = np.mean(diff, axis=0)
#            plt.figure(region + ' ' + str(ir))
#            plt.plot(changeMean)
#            plt.plot(preChangeMean)
#            plt.plot(diffMean)
#        
#    print('region ' + region + ' MEAN: ' + str(np.mean(meanChangeModPerExperiment)))
            

    change = np.array(allregionDict[region]['changePSTH'])
    prechange = np.array(allregionDict[region]['preChangePSTH'])    
    
    change_sub = change - np.mean(change[:, :250], axis=1)[:, None]
    change_norm = change_sub/np.max(change_sub, axis=1)[:, None]
    
    prechange_sub = prechange - np.mean(prechange[:, :250], axis=1)[:, None]
    prechange_norm = prechange_sub/np.max(change_sub, axis=1)[:, None]  
    
    diff = change_sub - prechange_sub
    changeMod = np.log2(diff[:, 280:530].max(axis=1)/prechange_sub[:, 280:530].max(axis=1))
#    plt.figure(region)
#    plt.hist(changeMod, bins=np.arange(-0.5, 0.5, 0.05))
    
#    plt.figure(region + 'changeMod vs Response amp')
#    plt.plot(prechange_sub.max(axis=1), changeMod, 'ko', alpha=0.5)
    print(region + ': ' + str(np.median(changeMod[~np.isnan(changeMod)])))
    print('n = ' + str(np.sum(~np.isnan(changeMod))))
    
    
    changeMean = np.mean(change_sub, axis=0)
    preChangeMean = np.mean(prechange_sub, axis=0)
    diffMean = np.mean(diff, axis=0)

            
#    plt.figure(region)
    plt.plot(changeMean)
#    plt.plot(preChangeMean)
#    plt.plot(diffMean)
    plt.legend(regionsOfInterest)
    
    
for ir, r in enumerate(rds):
    fig, axes = plt.subplots(3,2)
    fig.suptitle(experiments[ir])
    for ir, region in enumerate(regionsOfInterest):   
        axrow = ir%3
        axcol = int(ir>=3)
        ax = axes[axrow, axcol]
        if len(r[region]['changePSTH'])>0:
            change = np.array(r[region]['changePSTH'])
            prechange = np.array(r[region]['preChangePSTH'])    
            
            change_sub = change - np.mean(change[:, :250], axis=1)[:, None]
            prechange_sub = prechange - np.mean(prechange[:, :250], axis=1)[:, None]
            
            diff = change_sub - prechange_sub
            changeMod = np.log2(diff.max(axis=1)/prechange_sub.max(axis=1))
            
            changeMean = np.mean(change_sub, axis=0)
            preChangeMean = np.mean(prechange_sub, axis=0)
            diffMean = np.mean(diff, axis=0)
            ax.set_title(region)
            ax.plot(changeMean)
            ax.plot(preChangeMean)
            ax.plot(diffMean)
            ax.text(100, 5, 'n= ' + str(len(r[region]['changePSTH'])))



#splitting up change and prechange sdfs by image
def getValueByRegion(regionDict, key, region):
    inRegion = np.array(regionDict['region'])==region
    values = np.array(regionDict[key])[inRegion]
    return values

experiments = ['03212019_409096', '03262019_417882', '03272019_417882', '04042019_408528', '04052019_408528', '04102019_408527','04112019_408527', 
    '04252019_421323', '04262019_421323', '04302019_422856', '05162019_423749', '05172019_423749']

hdf5dir = r"C:\Users\svc_ccg\Desktop\Data\analysis"

rfFilter=False #if True, only take cells with significant RFs
regionDict = {r: [] for r in ['changePSTH', 'preChangePSTH', 'hitLickPSTH', 'badLickPSTH', 'region', 'experiment', 'unitID', 'probeID']}
for exp in experiments:
    print(exp)
    b = getData.behaviorEphys('Z:\\' + exp)
    
    h5FilePath = glob.glob(os.path.join(hdf5dir, exp+'*'))[0]
    b.loadFromHDF5(h5FilePath) 
    
    selectedTrials = (b.hit | b.miss)&(~b.ignore)
    changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one to correct for change frame indexing problem
    image_flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    image_id = np.array(b.core_data['visual_stimuli']['image_name'])
    preChangeIndices = np.searchsorted(image_flash_times, changeTimes)-1
    preChangeTimes = image_flash_times[preChangeIndices]
    preChangeIDs = image_id[preChangeIndices]
    changeIDs = image_id[preChangeIndices+1] #double check to make sure this worked
    
    regionsOfInterest = ['VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl']
    preTime = 0.25
    postTime = 0.5

    for pid in b.probes_to_analyze:
        for u in probeSync.getOrderedUnits(b.units[pid]):
            spikes = b.units[pid][u]['times']
            
            #exclude cells that fire below 1Hz during task
            if np.sum(spikes<3600)<3600: 
                continue
            
            #check if RF
            if rfFilter:
                rfmat = summaryPlots.plot_rf(b, spikes, plot=False, returnMat=True)
                if rfmat[1].max() < 99.9:
                    continue
                
            changesdfs = []
            presdfs = []            
            for im in np.unique(preChangeIDs):
                imPrechangeTimes = preChangeTimes[preChangeIDs==im]
                imChangeTimes = changeTimes[changeIDs==im]

                imPrechangeSDF, time = getSDF(spikes, imPrechangeTimes-preTime, preTime+postTime, sigma=0.001)
                imChangeSDF, time = getSDF(spikes, imChangeTimes-preTime, preTime+postTime, sigma=0.001)

                changesdfs.append(imChangeSDF)
                presdfs.append(imPrechangeSDF)
                
            #lick psth
            hitLicks, badLicks = summaryPlots.plot_lick_triggered_fr(b, spikes, plot=False, returnSDF=True, sdfSigma=0.001)
            
            region = b.probeCCF[pid]['ISIRegion'] if b.units[pid][u]['inCortex'] else b.units[pid][u]['ccfRegion']
            regionDict['region'].append(region)
            regionDict['experiment'].append(exp)
            regionDict['preChangePSTH'].append(presdfs)
            regionDict['changePSTH'].append(changesdfs)
            regionDict['unitID'].append(u)
            regionDict['probeID'].append(pid)
            regionDict['hitLickPSTH'].append(hitLicks)
            regionDict['badLickPSTH'].append(badLicks)


responseWindow = slice(270,580)
changeModDict = {r:{'pref':[], 'all':[]} for r in np.unique(regionDict['region'])}
for r in np.unique(regionDict['region']):
for r in ['MRN', 'MB']:
    inRegion = np.array(regionDict['region'])==r
    print(r + ': ' + str(np.sum(inRegion)))
    
    thischange = np.array(regionDict['changePSTH'])[inRegion]
    thispre = np.array(regionDict['preChangePSTH'])[inRegion]
    
    maxChangeInd = np.array([np.unravel_index(np.argmax(c[:, responseWindow]), c[:, responseWindow].shape)[0] for c in thischange])
    ch_preferred = np.array([c[m] for c,m in zip(thischange, maxChangeInd)])
    pre_preferred = np.array([c[m] for c,m in zip(thispre, maxChangeInd)])
    
    ch_preferred = ch_preferred - np.mean(ch_preferred[:, :250], axis=1)[:, None]
    pre_preferred = pre_preferred - np.mean(pre_preferred[:, :250], axis=1)[:, None]
    diff_preferred = ch_preferred - pre_preferred
    #changeMod_preferred = np.log2(diff_preferred[:, responseWindow].max(axis=1)/pre_preferred[:, responseWindow].max(axis=1))
    changeMod_preferred = np.log2(np.mean(ch_preferred[:, responseWindow], axis=1)/np.mean(pre_preferred[:, responseWindow], axis=1))
    changeModDict[r]['pref'].append(changeMod_preferred)
    
    ch_all = np.mean(thischange, axis=1)
    pre_all = np.mean(thispre, axis=1)
    ch_all = ch_all - np.mean(ch_all[:, :250], axis=1)[:, None]
    pre_all = pre_all - np.mean(pre_all[:, :250], axis=1)[:, None]
    diff_all = ch_all - pre_all
    #changeMod_all = np.log2(diff_all[:, responseWindow].max(axis=1)/pre_all[:, responseWindow].max(axis=1))
    changeMod_all = np.log2(np.mean(ch_all[:, responseWindow], axis=1)/np.mean(pre_all[:, responseWindow], axis=1))
    changeModDict[r]['all'].append(changeMod_all)
    
    plt.figure(r + '_preferred')
    plt.plot(np.mean(ch_preferred, axis=0))
    plt.plot(np.mean(pre_preferred, axis=0))
    plt.plot(np.mean(diff_preferred, axis=0))
    
    plt.figure(r + '_all')
    plt.plot(np.mean(ch_all, axis=0))
    plt.plot(np.mean(pre_all, axis=0))
    plt.plot(np.mean(diff_all, axis=0))
    
    
    ch_peaktuning = np.max(thischange[:, :, responseWindow], axis=2)
    pre_peaktuning = np.max(thispre[:, :, responseWindow], axis=2)

    sortedPeaks = np.argsort(ch_peaktuning, axis=1)
    
#    plt.figure(r + '_tuning')
#    plt.plot(np.mean([a[s] for a,s in zip(ch_peaktuning, sortedPeaks)], axis=0))
#    plt.plot(np.mean([a[s] for a,s in zip(pre_peaktuning, sortedPeaks)], axis=0))
#    ax = plt.gca()
#    ax.set_ylim([0, 21])
    
#    change_sparseness = [calculate_lifetime_sparseness(c) for c in ch_peaktuning]
#    pre_sparseness = [calculate_lifetime_sparseness(c) for c in pre_peaktuning]
#    print('change sparseness: ' + str(np.mean(change_sparseness)))
#    print('pre change sparseness: ' + str(np.mean(pre_sparseness)))
    
    

regiondf = pd.DataFrame.from_dict(regionDict)

#plot change mod hierarchy
fig, ax = plt.subplots()
for i, r in enumerate(['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']):
    all_c = np.squeeze(changeModDict[r]['all'])
    all_c[np.isinf(all_c)] = np.nan
    pref_c = np.squeeze(changeModDict[r]['pref'])
    pref_c[np.isinf(pref_c)] = np.nan
    ax.plot(i, np.nanmean(all_c), 'ko')
    ax.plot(i, np.nanmean(pref_c), 'go')
    ax.errorbar(i, np.nanmean(all_c), np.nanstd(all_c)/(np.sum(~np.isnan(all_c)))**0.5, c='k')
    ax.errorbar(i, np.nanmean(pref_c), np.nanstd(pref_c)/(np.sum(~np.isnan(pref_c)))**0.5, c='g')

ax.set_xticklabels(['', 'VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam'])


#plot response aligned to first licks
#for i, r in enumerate(['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam', 'MRN', 'MB']):
for r in np.unique(regionDict['region']):
    fig, ax = plt.subplots()
    fig.suptitle(r)
    h = getValueByRegion(regionDict, 'hitLickPSTH', r)
    b = getValueByRegion(regionDict, 'changePSTH', r)
    b = b.mean(axis=1)
    ax.plot(h.mean(axis=0))
    ax.plot(b.mean(axis=0))
    
    
    
    
    
    


