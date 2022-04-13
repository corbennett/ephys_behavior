mpo# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:58:45 2019

@author: svc_ccg
"""
import glob, os
import numpy as np
from matplotlib import pyplot as plt
from analysis_utils import formatFigure

#mouseIDs = ('408528', '421323', '423744', '423745', '423749', '429084', '408527')
dataDir = r"\\allen\programs\braintv\workgroups\neuralcoding\corbettb\ccg tensors\processed_ccgs"

mouse_dirs = [m for m in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir,m))]
areas = ('VISp', 'VISam', 'VISpm', 'VISl', 'VISrl', 'VISal')
states = ('active', 'passive')
ccg_array = [[[[] for _ in areas] for _ in areas] for _ in states]

#for mouse in mouseIDs:
for mouse in mouse_dirs:
    print('computing ccgs for mouse: ' + mouse)
    #mouseDir = glob.glob(os.path.join(dataDir, '*'+str(mouse)))[0]
    mouseDir = os.path.join(dataDir, mouse)
    
    ccfs = np.load(os.path.join(mouseDir, 'ccf_cortex.npy'))
    fr_active = np.load(os.path.join(mouseDir, 'FR_active_cortex.npy'))
    spikes_passive_rep = np.load(os.path.join(mouseDir, 'spikes_passive_cortex_rep.npy'))
    fr_passive = np.nanmean(np.nanmean(np.nansum(spikes_passive_rep[:,:,:,:250], axis=-1)/250*1000, axis=1), axis=1)
#    fr_passive = np.load(os.path.join(mouseDir, 'FR_passive_cortex.npy'))
    ccg_jc_active = np.load(glob.glob(os.path.join(mouseDir, '*jc*active*'))[0])
    ccg_jc_passive = np.load(glob.glob(os.path.join(mouseDir, '*jc*passive*'))[0])
    
    fr_passive_given_active = fr_passive[fr_active>2]
    allgoodInds = fr_passive_given_active>2
    
    ccf = ccfs[fr_active>2]
    ccg_jc = {a:k for a,k in zip(('active', 'passive'), (ccg_jc_active, ccg_jc_passive))}
    
    areas_in_exp = np.unique(ccf)
    
    for isrc, s in enumerate(areas):
        for itgt, t in enumerate(areas):
            if s in areas_in_exp and t in areas_in_exp:
                for istate, (state, color) in enumerate(zip(('active', 'passive'), ('r', 'b'))):
                    sourceInds = (ccf==s)&allgoodInds
                    targetInds = (ccf==t)&allgoodInds
                    thisccg = ccg_jc[state][sourceInds][:, targetInds]
                    ccgoverimages = np.nanmean(thisccg, axis=3)
                    
                    ccg_array[istate][isrc][itgt].append(np.reshape(ccgoverimages, (sourceInds.sum()*targetInds.sum(), 401)))
                    
                    
                    
                    
### analyze ccg_array across experiments ###
figSaveDir = r"\\allen\programs\braintv\workgroups\neuralcoding\corbettb\ccgs for paper"
preWin = slice(189, 199)
postWin = slice(200, 210)
peakWin = slice(189, 210)
lagWin = slice(179, 220)
#baseline = slice(86, 186)
baseline = [slice(99, 149), slice(249, 299)]
all_ccgs = [[],[]]
ccg_mean_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_lag_median_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_lag_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_peak_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_lag_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_count_array = [[[[] for _ in areas] for _ in areas] for _ in states]
plot=True
std_thresh=5
for isrc, s in enumerate(areas):
    for itgt, t in enumerate(areas):
        if plot:
            fig, ax = plt.subplots()
            fig.suptitle(t + ' to ' + s)
        
        #find bad cells to exclude for passive and active
        badInds = []
        for istate, _ in enumerate(['active', 'passive']):
            these_ccgs = ccg_array[istate][isrc][itgt]
            ccg_mean = np.vstack([these_ccgs[i] for i in np.arange(len(these_ccgs))])    
            ccg_mean[np.isinf(ccg_mean)] = np.nan
            nanInds = np.array([any(np.isnan(c)) for c in ccg_mean])
            badInds.append(nanInds)
        goodInds = np.where(~badInds[0]*~badInds[1])[0]     
        
        #find cells with peaks
        goodccgs = [[],[]]
        for istate, (state, color) in enumerate(zip(('active', 'passive'), ('r', 'b'))):
            these_ccgs = ccg_array[istate][isrc][itgt]
            ccg_mean = np.vstack([these_ccgs[i] for i in np.arange(len(these_ccgs))])                
            
            ccg_mean = ccg_mean[goodInds]
            #ccg_mean_conv = np.array([np.convolve(c, np.ones(5), 'same')/5 for c in ccg_mean])
            ccg_mean_conv = np.array(ccg_mean)
            
            
            for c in ccg_mean_conv:
                cbaseline = np.concatenate([c[b] for b in baseline])
                baselinemean = np.nanmean(cbaseline)
                baselinestd = np.nanstd(cbaseline)
                goodccgs[istate].append(np.nanmax(c[peakWin])>baselinemean+baselinestd*std_thresh)
        goodccgs=np.array(goodccgs)
        good = goodccgs[0] | goodccgs[1]
        
        for istate, (state, color) in enumerate(zip(('active', 'passive'), ('r', 'b'))):
            these_ccgs = ccg_array[istate][isrc][itgt]
            ccg_mean = np.vstack([these_ccgs[i] for i in np.arange(len(these_ccgs))])                
            print(str(s) + ' ' + str(t) + ' excluding ' + str(ccg_mean.shape[0]-len(goodInds)) + ' of ' + str(ccg_mean.shape[0]))
            
            ccg_mean = np.array(ccg_mean[goodInds])
            ccg_mean = ccg_mean[good]
            ccg_mean_conv = np.array([np.convolve(c, np.ones(5), 'same')/5 for c in ccg_mean])
            
            
            thismean = np.nanmean(ccg_mean, axis=0)
            thismean_conv = np.nanmean(ccg_mean_conv, axis=0)
            thissem_conv = np.std(ccg_mean_conv, axis=0)/(ccg_mean_conv.shape[0]**0.5)
            
            ccg_mean_array[istate][isrc][itgt] = thismean_conv
            
            lag = np.mean(thismean[postWin]) - np.mean(thismean[preWin])
            lags = [np.sum(c[postWin]) - np.sum(c[preWin]) for c in ccg_mean]
            
            ccg_count_array[istate][isrc][itgt] = len(lags)
            #ccg_lag_median_array[istate][isrc][itgt] = np.nanmedian(lags)
            ccg_lag_array[istate][isrc][itgt] = lag
            
            zeroBin = int((lagWin.stop - lagWin.start)/2)
            peak_pos = np.array([np.argmax(c[lagWin]) for c in ccg_mean]) - zeroBin
            
            np.save(os.path.join(figSaveDir, t + 'to' + s + '_' + state + '_lagdist.npy'), peak_pos)
            #ccg_lag_array[istate][isrc][itgt] = peak_pos
            ccg_lag_median_array[istate][isrc][itgt] = np.nanmedian(peak_pos)
            #ax.hist(peak_pos-10, bins=21)
            
            #ccg_peak_array[istate][isrc][itgt] = np.max(thismean_conv[175:225])
            all_ccgs[istate].append(thismean_conv)
            if plot:
                x = np.linspace(-200, 200, 401)
                ax.plot(x,thismean_conv, color)
                ax.fill_between(x, thismean_conv-thissem_conv, thismean_conv+thissem_conv, color=color, alpha=0.5)
                ax.set_xlim([-25, 25])
                if istate==1:
                    ax.text(-10, ax.get_ylim()[1]*0.9, 'n = ' + str(ccg_mean_conv.shape[0]))
                    formatFigure(fig, ax, xLabel='Time (ms)', yLabel='Coincidence per spike')
                    fig.savefig(os.path.join(figSaveDir, t + ' to ' + s + '.pdf'))
                
                
ccg_peak_array = np.array(ccg_peak_array)               
ccg_lag_array = np.array(ccg_lag_array)
ccg_lag_median_array = np.array(ccg_lag_median_array)
ccg_count_array = np.array(ccg_count_array)

plt.figure()
plt.plot(np.mean(all_ccgs[0],axis=0), 'r')
plt.plot(np.mean(all_ccgs[1],axis=0), 'b')

ccg_peak_array = np.array(ccg_peak_array)
diff = ccg_peak_array[0] - ccg_peak_array[1]
ratio = ccg_peak_array[0]/ccg_peak_array[1]

hier_order = [0,3,4,5,2,1]
areas_hier = [areas[i] for i in hier_order]
diff_hier = diff[hier_order][:, hier_order]
ratio_hier = ratio[hier_order][:, hier_order]


fig, ax = plt.subplots()
im = ax.imshow(diff_hier, cmap='bwr', clim=[-np.max(np.abs(diff)), np.max(np.abs(diff))])
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)
ax.set_xlim([0-0.5, len(areas)-0.5])
ax.set_ylim([0-0.5, len(areas)-0.5])
plt.colorbar(im)

fig, ax = plt.subplots()
im = ax.imshow(ratio_hier, cmap='bwr', clim=[1-np.max(np.abs(ratio_hier-1)), 1+np.max(np.abs(ratio_hier-1))])
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)
ax.set_xlim([0-0.5, len(areas)-0.5])
ax.set_ylim([0-0.5, len(areas)-0.5])
plt.colorbar(im)


fig, ax = plt.subplots()
im = ax.imshow(ccg_lag_array[0][hier_order][:, hier_order], cmap='bwr', clim=[-np.max(np.abs(ccg_lag_array)), np.max(np.abs(ccg_lag_array))])
plt.colorbar(im)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)
ax.set_xlim([0-0.5, len(areas)-0.5])
ax.set_ylim([0-0.5, len(areas)-0.5])

fig, ax = plt.subplots()
im = ax.imshow(ccg_lag_array[1][hier_order][:, hier_order], cmap='bwr', clim=[-np.max(np.abs(ccg_lag_array)), np.max(np.abs(ccg_lag_array))])
plt.colorbar(im)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier) 
ax.set_xlim([0-0.5, len(areas)-0.5])
ax.set_ylim([0-0.5, len(areas)-0.5])             


ccg_lag_median_array_ordered = [c[hier_order][:, hier_order] for c in ccg_lag_median_array]
ccg_lag_median_array_ordered = np.array(ccg_lag_median_array_ordered)
fig, ax = plt.subplots()
im = ax.imshow(ccg_lag_median_array_ordered[0], cmap='bwr', clim=[-np.max(np.abs(ccg_lag_median_array)), np.max(np.abs(ccg_lag_median_array))])
plt.colorbar(im)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier) 
ax.set_xlim([0-0.5, len(areas)-0.5])
ax.set_ylim([0-0.5, len(areas)-0.5])


fig, ax = plt.subplots()
ax.plot(ccg_lag_array[0], ccg_lag_array[1], 'ko')
ax.set_aspect('equal')
xlims = ax.get_xlim()
ax.plot(xlims, xlims, 'k--')

fig, ax = plt.subplots()
ax.plot(ccg_lag_median_array[0], ccg_lag_array[1], 'ko')
ax.set_aspect('equal')
xlims = ax.get_xlim()
ax.plot(xlims, xlims, 'k--')

import seaborn as sns
ccg_count_array_ordered = [c[hier_order][:, hier_order] for c in ccg_count_array]
ccg_count_array_ordered = np.array(ccg_count_array_ordered)
fig, ax = plt.subplots()
sns.heatmap(ccg_count_array[1][hier_order][:, hier_order], xticklabels=areas_hier, yticklabels=areas_hier,
            square=True, cbar_kws={"shrink": .5}, linewidths=0.1, annot=True)
ax.set_xlim([0, len(areas)])
ax.set_ylim([0, len(areas)])
              
 
#SAVE RESULTS ###

saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\changeMod figure for npx platform paper\ccgs"
np.save(os.path.join(saveDir, 'ccg_lags.npy'), ccg_lag_median_array_ordered)
np.save(os.path.join(saveDir, 'ccg_counts.npy'), ccg_count_array_ordered)
np.save(os.path.join(saveDir, 'ccg_meanoverwindows_onmean.npy'), np.array([c[hier_order][:, hier_order] for c in ccg_lag_array]))



               
#####Do analysis for every experiment separately##########               
expIndex = np.zeros((len(areas), len(areas)), dtype=np.int)
for exp, mouse in enumerate(mouseIDs):
    mouseDir = glob.glob(os.path.join(dataDir, '*'+str(mouse)))[0]
    ccf = np.load(os.path.join(mouseDir, 'ccf_cortex.npy'))
    areas_in_exp = np.unique(ccf)
    all_ccgs = [[],[]]
    ccg_mean_array = [[[[] for _ in areas] for _ in areas] for _ in states]
    ccg_peak_array = np.full((2, len(areas), len(areas)), np.nan)
    ccg_lag_array = np.full((2, len(areas), len(areas)), np.nan)
    plot=False
    
    for isrc, s in enumerate(areas):
        for itgt, t in enumerate(areas):
            if s in areas_in_exp and t in areas_in_exp: 
                if plot:
                    fig, ax = plt.subplots()
                    fig.suptitle(s + ' to ' + t)
                
                #find bad cells to exclude for passive and active
                badInds = []
                for istate, _ in enumerate(['active', 'passive']):
                    ccg_mean = ccg_array[istate][isrc][itgt][expIndex[isrc, itgt]]
                    ccg_mean[np.isinf(ccg_mean)] = np.nan
                    nanInds = np.array([any(np.isnan(c)) for c in ccg_mean])
                    badInds.append(nanInds)
                goodInds = np.where(~badInds[0]*~badInds[1])[0]     
                    
                for istate, (state, color) in enumerate(zip(('active', 'passive'), ('r', 'b'))):
                    ccg_mean = ccg_array[istate][isrc][itgt][expIndex[isrc, itgt]]
                                   
                    print(str(s) + ' ' + str(t) + ' excluding ' + str(ccg_mean.shape[0]-len(goodInds)) + ' of ' + str(ccg_mean.shape[0]))
                    
                    ccg_mean = ccg_mean[goodInds]
                    thismean = np.nanmean(ccg_mean, axis=0)
                    ccg_mean_array[istate][isrc][itgt] = thismean
                    
                    lag = np.sum(thismean[186:199]) - np.sum(thismean[200:213])
                    ccg_lag_array[istate,isrc,itgt] = lag
                    
                    thismean_conv = np.convolve(thismean, np.ones(5), 'same')/5
                    ccg_peak_array[istate,isrc,itgt] = np.nanmax(thismean_conv[175:225])
                    all_ccgs[istate].append(thismean_conv)
                    if plot:
                        ax.plot(thismean_conv, color)
                        ax.set_xlim([100, 300])
                expIndex[isrc, itgt] += 1
                
    ccg_peak_array = np.array(ccg_peak_array)               
    ccg_lag_array = np.array(ccg_lag_array)
                   
    plt.figure()
    plt.plot(np.mean(all_ccgs[0],axis=0), 'r')
    plt.plot(np.mean(all_ccgs[1],axis=0), 'b')
    
    ccg_peak_array = np.array(ccg_peak_array)
    diff = ccg_peak_array[0] - ccg_peak_array[1]
    ratio = ccg_peak_array[0]/ccg_peak_array[1]
    
    hier_order = [0,3,4,5,2,1]
    areas_hier = [areas[i] for i in hier_order]
    diff_hier = diff[hier_order][:, hier_order]
    ratio_hier = ratio[hier_order][:, hier_order]
    
    
    fig, ax = plt.subplots()
    fig.suptitle('active - passive ' + mouseIDs[exp])
    im = ax.imshow(diff_hier, cmap='bwr', clim=[-np.nanmax(np.abs(diff)), np.nanmax(np.abs(diff))])
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas_hier)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas_hier)
    plt.colorbar(im)
    
    fig, ax = plt.subplots()
    fig.suptitle('active over passive ' + mouseIDs[exp])
    im = ax.imshow(ratio_hier, cmap='bwr', clim=[1-np.nanmax(np.abs(ratio_hier-1)), 1+np.nanmax(np.abs(ratio_hier-1))])
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas_hier)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas_hier)
    plt.colorbar(im)
    
    
    fig, ax = plt.subplots()
    fig.suptitle('ccg_lag, active ' + mouseIDs[exp])
    im = ax.imshow(ccg_lag_array[0][hier_order][:, hier_order], cmap='bwr', clim=[-np.nanmax(np.abs(ccg_lag_array)), np.nanmax(np.abs(ccg_lag_array))])
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas_hier)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas_hier)
    
    fig, ax = plt.subplots()
    fig.suptitle('ccg_lag, passive ' + mouseIDs[exp])
    im = ax.imshow(ccg_lag_array[1][hier_order][:, hier_order], cmap='bwr', clim=[-np.nanmax(np.abs(ccg_lag_array)), np.nanmax(np.abs(ccg_lag_array))])
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas_hier)
    ax.set_yticks(np.arange(len(areas)))
    ax.set_yticklabels(areas_hier)                
    
    
                
                
                
                
                
                
                
                
                