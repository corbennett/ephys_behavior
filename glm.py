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
    for i, t in enumerate(eventTimes):
        trace = binned_spikes[t-preTime_bin:t+postTime_bin]
        psth = psth + trace
    
    return psth/len(eventTimes)

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


binwidth = 0.02
spikes = b.units['C']['270']['times']
binned_spikes = binVariable(spikes[spikes<b.lastBehaviorTime], binwidth)

model = licking_model.Model(dt=binwidth, licks=binned_spikes)

selectedTrials = (b.hit | b.miss)&(~b.ignore)   #Omit "ignore" trials (aborted trials when the mouse licked too early or catch trials when the image didn't actually change)
active_changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one here to correct for a one frame shift in frame times from camstim
binned_activeChangeTimes = binVariable(active_changeTimes, binwidth)
changeFilter = licking_model.GaussianBasisFilter(num_params=10, data=binned_activeChangeTimes, dt=model.dt, duration=0.6, sigma=0.05)
model.add_filter('change', changeFilter)

lick_offset = 0.25
lick_times = probeSync.get_sync_line_data(b.syncDataset, 'lick_sensor')[0] - lick_offset
binned_licks = binVariable(lick_times, binwidth)
lickFilter = licking_model.GaussianBasisFilter(num_params=5, data=binned_licks, dt=model.dt, duration=0.6, sigma=0.1)
model.add_filter('lick', lickFilter)

binned_running = binRunning(b.behaviorRunSpeed.values, b.behaviorRunTime, binwidth)
runFilter = licking_model.GaussianBasisFilter(num_params=5, data=binned_licks, dt=model.dt, duration=1, sigma=0.2)
model.add_filter('run', runFilter)

flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
image_id = np.array(b.core_data['visual_stimuli']['image_name'])
binned_flashTimes = []
for i,img in enumerate(np.unique(image_id)):
    this_image_times = flash_times[image_id==img]
    bf = binVariable(this_image_times, binwidth)
    binned_flashTimes.append(bf)
    flash_filter = licking_model.GaussianBasisFilter(num_params=10, data=bf, dt=model.dt, duration=0.6, sigma=0.05)
    model.add_filter('flash_' + img, flash_filter)

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
model.verbose=True
model.l2=10
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