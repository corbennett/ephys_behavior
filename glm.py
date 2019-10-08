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
import pickle

#using our class for parsing the data (https://github.com/corbennett/ephys_behavior)
expNames = ('05162019_423749','07112019_429084', '04042019_408528', '08082019_423744', '05172019_423749')
expName = expNames[-1]
b = getData.behaviorEphys(os.path.join('Z:\\',expName))
b.loadFromHDF5(os.path.join('Z:\\analysis',expName+'.hdf5')) 

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
for pID in b.probes_to_analyze:
    for u in probeSync.getOrderedUnits(b.units[pID]):
        spikes = b.units[pID][u]['times']
#        try:
        if np.sum(spikes<b.lastBehaviorTime)>spikeRateThresh*b.lastBehaviorTime:
            binned_spikes = binVariable(spikes[spikes<b.lastBehaviorTime], binwidth)[changeBins]
            model = licking_model.Model(dt=binwidth, licks=binned_spikes)
            
            filterList = []
            for filtname, [times, nparams, duration, sigma, offset] in eventsToInclude:
                if filtname == 'running':
                    binned = binRunning(times[0], times[1], binwidth)[changeBins]
                else:
                    binned = binVariable(times+offset, binwidth)[changeBins]
                
                ffilter = licking_model.GaussianBasisFilter(num_params=nparams, data=binned, dt=binwidth, duration=duration, sigma=sigma)
                filterList.append((filtname, ffilter))
            
            
            for fname, ffilter in filterList:
                model.add_filter(fname, ffilter)
            
            
            #fit model    
            model.mean_rate_param = np.log(np.mean(binned_spikes)/binwidth)
            model.verbose=False
            model.l2=1
            model.fit()
            
            if len(model.val_nle_history)>0:
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
            else:
                print('fit failed for ' + pID + ' ' + u)
#        except:
#            continue

saveDir = r"C:\Users\svc_ccg\Desktop\Data\analysis"
np.savez(os.path.join(saveDir,expName+'_glmfits.npz'), modelParams=modelParams, ccfRegions=ccfRegions, fittest=test_corr, uid=uid)
model.save(saveDir, expName+'_glm_model.pkl') #save the last model just to have the structure for recreating
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
model.verbose=True
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





#Analyze fit data

glmFitsFile = r"C:\Users\svc_ccg\Desktop\Data\analysis\glmfits\07112019_429084_glmfits.npz"
glmfit = np.load(glmFitsFile)

#make model with same structure as one used to fit data
modelFilename = r"C:\Users\svc_ccg\Desktop\Data\analysis\glmfits\07112019_429084_glm_model.pkl"
with open(modelFilename) as f:
    model = pickle.load(f)
temp_model = copy.deepcopy(model)

def getFiltersFromParams(mod):
    filterTraces = {}
    for ind_filter, (filter_name, filter_obj) in enumerate(mod.filters.items()):
        linear_filt = filter_obj.build_filter()
        filterTraces[filter_name]=np.exp(linear_filt)
    
    return filterTraces

def getArea(areas, ccfr):
    finalarea = None
    for area, components in areas:
        if ccfr in components:
            finalarea=area
            break
    
    return finalarea

def cumulative_dist(arr):
    sort = np.copy(arr)
    sort.sort()
    cumdist = np.array([np.sum(sort<=val) for val in sort])
    return sort, cumdist/len(sort)

areas = (
         ('VISp', ('VISp')),
         ('VISrl', ('VISrl')),
         ('VISl', ('VISl')),
         ('VISal', ('VISal')),
         ('VISam', ('VISam')),
         ('VISpm', ('VISpm')),
         ('LGd', ('LGd')),
         ('LP', ('LP')),
         ('hipp', ('CA1', 'CA3', 'DG-mo', 'DG-sg')),
         ('MB', ('MB', 'MRN')),
         ('APN', ('APN'))
)

glmfitsdir = r"C:\Users\svc_ccg\Desktop\Data\analysis\glmfits"
glmFitFiles = glob.glob(os.path.join(glmfitsdir, '*.npz'))

fitDists = {b:[] for b in temp_model.filters.keys()+['bestImageWeight']}
master_id = []
master_fit = []
master_area = []
for glmfile in glmFitFiles:
    glmfit = np.load(glmfile)
    for params, ccfr, fit, uid in zip(*[glmfit[key] for key in ['modelParams', 'ccfRegions', 'fittest', 'uid']]):    
        area = getArea(areas, ccfr)
        day = glmfile.split('\\')[-1]
        day = day[:day.rfind('_')]
        
        if area is not None and fit[0]>0.2:
            temp_model.set_filter_params(params)
            filtDict = getFiltersFromParams(temp_model)
            
            bestim = 0
            peaks = []
            for filtname, filt in filtDict.items():
                filtpeak = np.max(filt)
#                fitDists[filtname].append(filtpeak)
                if 'im' in filtname and filtpeak>bestim:
                    bestim = filtpeak
                peaks.append((filtname, filtpeak))
            
            orderOfimport = np.argsort([p[1] for p in peaks])
            for i, sortind in enumerate(orderOfimport):
                fitDists[peaks[sortind][0]].append(i) 
            
            fitDists['bestImageWeight'].append(bestim)
            master_id.append(day + '_' + uid)
            master_fit.append(fit)
            master_area.append(area)
    
master_area = np.array(master_area)       

filtname = 'change'
plt.figure(filtname)
for area, _ in areas:
#    normed = np.array(fitDists[area][filtname])/np.array(fitDists[area]['bestImageWeight'])
    x, c = cumulative_dist(np.array(fitDists[filtname])[master_area==area])
#    x,c = cumulative_dist(normed)
    plt.plot(x,c)
    
plt.legend(areas)



for area, _ in areas:
    plt.figure(area)
    mat = np.zeros((len(temp_model.filters.keys()), len(fitDists[area]['change'])))

    for i, filtpeaks in enumerate(temp_model.filters.keys()):
        mat[i] = fitDists[area][filtpeaks]
        
    plt.imshow(mat.T, aspect='auto')   
    ax=plt.gca()
    ax.set_xticks(np.arange(len(temp_model.filters.keys())))
    ax.set_xticklabels(temp_model.filters.keys())    






