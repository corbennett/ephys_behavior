# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:33:41 2019

@author: svc_ccg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:18:26 2019

@author: svc_ccg
"""
from __future__ import division
import getData
from matplotlib import pyplot as plt
import probeSync
import licking_model
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import copy
import numpy as npo
from numba import njit
import scipy.stats
import time
import os

#using our class for parsing the data (https://github.com/corbennett/ephys_behavior)
b = getData.behaviorEphys('Z:\\05162019_423749')
b.loadFromHDF5(r"Z:\analysis\05162019_423749.hdf5") 

def binVariable(var, binwidth=0.001):
    return np.histogram(var, bins=np.arange(0, b.lastBehaviorTime, binwidth))[0]

def binRunning(runSig, runTime, binwidth=0.001):
    return np.interp(np.arange(0, b.lastBehaviorTime, binwidth), runTime, runSig)[:-1]
    
@njit
def getPSTH(binned_spikes, binned_times, binwidth, preTime, postTime):
    preTime_bin = int(preTime/binwidth)
    postTime_bin = int(postTime/binwidth)
    eventTimes = npo.where(binned_times)[0]
    psth = npo.zeros(preTime_bin+postTime_bin)
    traces_included = 0
    for i, t in enumerate(eventTimes):
        trace = binned_spikes[t-preTime_bin:t+postTime_bin]
        if len(trace)==len(psth):
            psth = psth + trace
            traces_included += 1
    
    return psth/traces_included

def find_initial_guess(psth, ffilter):
    def compute_error(latent, inputs):
        return np.sum((latent-inputs)**2)
    
    def to_min(params):
        ffilter.set_params(params)
        latent = ffilter.build_filter()
        return compute_error(latent, np.log(psth))
    
    params=ffilter.params
    g = grad(to_min)
    res = minimize(to_min, params, jac=g)
    
    return res['x']

def getChangeBins(binned_changeTimes, binwidth, preChangeTime = 2, postChangeTime=2):
    preBins = int(preChangeTime/binwidth)
    postBins = int(postChangeTime/binwidth)
    changeBins = npo.convolve(binned_changeTimes, npo.ones(preBins+postBins), 'same').astype(npo.bool)
    return changeBins
    

binwidth = 0.05
offset = 0.2
restrictToChange=False

image_id = np.array(b.core_data['visual_stimuli']['image_name'])
selectedTrials = (b.hit | b.miss)&(~b.ignore)   #Omit "ignore" trials (aborted trials when the mouse licked too early or catch trials when the image didn't actually change)
active_changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one here to correct for a one frame shift in frame times from camstim
binned_activeChangeTimes = binVariable(active_changeTimes, binwidth)
if restrictToChange:
    changeBins = getChangeBins(binned_activeChangeTimes, binwidth)
else:
    changeBins = npo.ones(binned_activeChangeTimes.size).astype(npo.bool)


lick_times = b.lickTimes
first_lick_times = lick_times[np.insert(np.diff(lick_times)>=0.5, 0, True)]

flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    
eventsToInclude = [('change', [active_changeTimes, 8, 0.8, 0.1, -0.2]),
                   ('licks', [lick_times, 5, 0.6, 0.1, -0.3]),
                   ('first_licks', [first_lick_times, 10, 2, 0.2, -1]),
                   ('running', [[b.behaviorRunSpeed.values, b.behaviorRunTime], 5, 2, 0.4, 0])]

for img in np.unique(image_id):
    eventsToInclude.append((img, [flash_times[image_id==img], 8, 0.8, 0.1, -0.2]))


filterList = []
for filtname, [times, nparams, duration, sigma, offset] in eventsToInclude:
    if filtname == 'running':
        binned = binRunning(times[0], times[1], binwidth)[changeBins]
    else:
        binned = binVariable(times+offset, binwidth)[changeBins]
    
    ffilter = licking_model.GaussianBasisFilter(num_params=nparams, data=binned, dt=binwidth, duration=duration, sigma=sigma)
    filterList.append((filtname, ffilter))

###RUN for entire experiment###
startTime = time.time()
spikeRateThresh=0.5
ccfRegions = []
modelParams = []
test_corr = []
uid = []
for pID in 'ABCDEF':
    for u in probeSync.getOrderedUnits(b.units[pID]):
        spikes = b.units[pID][u]['times']
#        try:
        if np.sum(spikes<b.lastBehaviorTime)>spikeRateThresh*b.lastBehaviorTime:
            binned_spikes = binVariable(spikes[spikes<b.lastBehaviorTime], binwidth)[changeBins]
            model = licking_model.Model(dt=binwidth, licks=binned_spikes)
            
            for fname, ffilter in filterList:
                model.add_filter(fname, ffilter)
            
            
#            #set initial guesses
#            for filter_name, ffilter in model.filters.items():
#                if filter_name=='run':
#                    continue
#                binned_times = ffilter.data
#                thispsth = getPSTH(binned_spikes, binned_times, binwidth, 0, ffilter.duration)[1:]/np.mean(binned_spikes) + np.finfo(float).eps
#                init_guess = find_initial_guess(thispsth, ffilter)
#                ffilter.set_params(init_guess)
#            
            
            #fit model    
            model.mean_rate_param = np.log(np.mean(binned_spikes)/binwidth)
            model.verbose=False
            model.l2=1
            model.fit()
            
            goodParams = model.param_fit_history[np.argmin(model.val_nle_history)]
            model.set_filter_params(goodParams)
            
            testlicks, testlatent = model.get_licks_and_latent(model.test_split)
            test_corr.append(scipy.stats.pearsonr(testlicks, testlatent))
            modelParams.append(goodParams)
            
            if b.units[pID][u]['inCortex']:
                ccfRegions.append(b.probeCCF[pID]['ISIRegion'])
            else:
                ccfRegions.append(b.units[pID][u]['ccfRegion'])

            uid.append(pID+'_'+u)
#        except:
#            continue

saveDir = r"C:\Users\svc_ccg\Desktop\Data\analysis"
np.savez(os.path.join(saveDir, '05162019_glmfits.npz'), modelParams=modelParams, ccfRegions=ccfRegions, fittest=test_corr, uid=uid)
elapsedTime=time.time()-startTime
print('\nelapsed time: ' + str(elapsedTime))









spikes = b.units['A']['292']['times']
binned_spikes = binVariable(spikes[spikes<b.lastBehaviorTime], binwidth)[changeBins]
model = licking_model.Model(dt=binwidth, licks=binned_spikes)

for fname, ffilter in filterList:
    model.add_filter(fname, ffilter)


#set initial guesses
for filter_name, ffilter in model.filters.items():
    if filter_name=='run':
        continue
    binned_times = ffilter.data
    thispsth = getPSTH(binned_spikes, binned_times, binwidth, 0, ffilter.duration)[1:]/np.mean(binned_spikes) + np.finfo(float).eps
    init_guess = find_initial_guess(thispsth, ffilter)
    ffilter.set_params(init_guess)


#fit model    
model.mean_rate_param = np.log(np.mean(binned_spikes)/binwidth)
model.verbose=False
model.l2=1
model.fit()
plt.figure()
model.plot_filters()


#see how close initial guesses were
init_params = []
for filter_name, ffilter in model.filters.items():
    binned_times = ffilter.data
    thispsth = getPSTH(binned_spikes, binned_times, binwidth, 0, ffilter.duration)[1:]/np.mean(binned_spikes) + np.finfo(float).eps

    init_guess_filt = copy.deepcopy(ffilter)
#    init_guess = find_initial_guess(binned_spikes, binned_times, binwidth, init_guess_filt)
    init_guess = find_initial_guess(thispsth, init_guess_filt)
    init_guess_filt.set_params(init_guess)
    init_params.extend(init_guess)
    fig, ax = plt.subplots()
    ax.plot(np.exp(init_guess_filt.build_filter()))
    ax.plot(np.exp(ffilter.build_filter()))
    ax.set_title(filter_name)

init_params.insert(0,np.log(np.mean(binned_spikes)/binwidth))
init_params = np.array(init_params)
fig, ax = plt.subplots()
ax.plot(init_params, model.get_filter_params(), 'ko')
ax.set_aspect('equal')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, 'k--')
ax.set_ylabel('final fit params')
ax.set_xlabel('initial guess params')
