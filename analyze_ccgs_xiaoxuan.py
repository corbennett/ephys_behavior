mpo# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:58:45 2019

@author: svc_ccg
"""
import glob, os
import numpy as np
from matplotlib import pyplot as plt
from analysis_utils import formatFigure

mouseIDs = ('408528', '421323', '423744', '423745', '423749', '429084', '408527')
dataDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\ccg tensors\processed_ccgs"

areas = ('VISp', 'VISam', 'VISpm', 'VISl', 'VISrl', 'VISal')
states = ('active', 'passive')
ccg_array = [[[[] for _ in areas] for _ in areas] for _ in states]

for mouse in mouseIDs:
    print('computing ccgs for mouse: ' + mouse)
    mouseDir = glob.glob(os.path.join(dataDir, '*'+str(mouse)))[0]
    
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
figSaveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\SfN2019\ccg_traces"
preWin = slice(186, 199)
postWin = slice(200, 213)
peakWin = slice(186, 213)
baseline = slice(86, 186)
all_ccgs = [[],[]]
ccg_mean_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_peak_array = [[[[] for _ in areas] for _ in areas] for _ in states]
ccg_lag_array = [[[[] for _ in areas] for _ in areas] for _ in states]
plot=True
for isrc, s in enumerate(areas):
    for itgt, t in enumerate(areas):
        if plot:
            fig, ax = plt.subplots()
            fig.suptitle(s + ' to ' + t)
        
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
            ccg_mean_conv = np.array([np.convolve(c, np.ones(5), 'same')/5 for c in ccg_mean])
            
            
            for c in ccg_mean_conv:
                baselinemean = np.nanmean(c[baseline])
                baselinestd = np.nanstd(c[baseline])
                goodccgs[istate].append(np.nanmax(c[peakWin])>baselinemean+baselinestd*3)
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
            
            lag = np.sum(thismean[186:199]) - np.sum(thismean[200:213])
            ccg_lag_array[istate][isrc][itgt] = lag
            
            
            ccg_peak_array[istate][isrc][itgt] = np.max(thismean_conv[175:225])
            all_ccgs[istate].append(thismean_conv)
            if plot:
                x = np.linspace(-200, 200, 401)
                ax.plot(x,thismean_conv, color)
                ax.fill_between(x, thismean_conv-thissem_conv, thismean_conv+thissem_conv, color=color, alpha=0.5)
                ax.set_xlim([-25, 25])
                if istate==1:
                    ax.text(-10, ax.get_ylim()[1]*0.9, 'n = ' + str(ccg_mean_conv.shape[0]))
                    formatFigure(fig, ax, xLabel='Time (ms)', yLabel='Coincidence per spike')
#                    fig.savefig(os.path.join(figSaveDir, s + ' to ' + t + '.pdf'))
                
                
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
im = ax.imshow(diff_hier, cmap='bwr', clim=[-np.max(np.abs(diff)), np.max(np.abs(diff))])
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)
plt.colorbar(im)

fig, ax = plt.subplots()
im = ax.imshow(ratio_hier, cmap='bwr', clim=[1-np.max(np.abs(ratio_hier-1)), 1+np.max(np.abs(ratio_hier-1))])
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)
plt.colorbar(im)


fig, ax = plt.subplots()
im = ax.imshow(ccg_lag_array[0][hier_order][:, hier_order], cmap='bwr', clim=[-np.max(np.abs(ccg_lag_array)), np.max(np.abs(ccg_lag_array))])
plt.colorbar(im)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)

fig, ax = plt.subplots()
im = ax.imshow(ccg_lag_array[1][hier_order][:, hier_order], cmap='bwr', clim=[-np.max(np.abs(ccg_lag_array)), np.max(np.abs(ccg_lag_array))])
plt.colorbar(im)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas_hier)
ax.set_yticks(np.arange(len(areas)))
ax.set_yticklabels(areas_hier)                


fig, ax = plt.subplots()
ax.plot(ccg_lag_array[0], ccg_lag_array[1], 'ko')
ax.set_aspect('equal')
xlims = ax.get_xlim()
ax.plot(xlims, xlims, 'k--')
                
                
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
    
    
                
                
                
                
                
                
                
                
                