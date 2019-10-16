# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:51:55 2019

@author: svc_ccg
"""
import numpy as np
import os
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
import pandas as pd
from matplotlib import pyplot as plt

pickleFileDir = r"Z:\behavior pickle files"
mouseIDs = [u'409096',
 u'417882',
 u'417882',
 u'408528',
 u'408528',
 u'408527',
 u'408527',
 u'421323',
 u'421323',
 u'422856',
 u'423749',
 u'423749',
 u'427937',
 u'423745',
 u'429084',
 u'429084',
 u'423744',
 u'423744',
 u'423750',
 u'423750',
 u'459521',
 u'459521']

imIDs = np.array(['im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077',
       'im085'], dtype='|S5')

def makeBehaviorMatrix(trials, imIDs):
    counts = np.zeros((8,8))
    hits = np.zeros((8,8))
    
    changeImages = np.array(trials.change_image_name.to_list())
    initialImages = np.array(trials.initial_image_name.to_list())
    
    autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
    earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
    ignore = earlyResponse | autoRewarded
    miss = np.array(trials['response_type']=='MISS')
    hit = np.array(trials['response_type']=='HIT')
    falseAlarm = np.array(trials['response_type']=='FA')
    correctReject = np.array(trials['response_type']=='CR')
    
    for trial in np.arange(changeImages.size):
        col = np.where(imIDs==changeImages[trial])[0]
        row = np.where(imIDs==initialImages[trial])[0]
        
        if not ignore[trial]:
            
            counts[row,col] += 1
            if hit[trial] or falseAlarm[trial]:
                hits[row,col] += 1
    
    return counts, hits
     
        
    
    

daysToInclude = 20
data = [[], [], []]
for mouse in mouseIDs:
    try:
        mousedir = os.path.join(pickleFileDir, mouse)
        pklfiles = np.sort(os.listdir(mousedir))[-daysToInclude:]
    except:
        print('could not find mouse ' + mouse + ' in ' + pickleFileDir)
        continue
    
    allcounts = np.zeros((8,8))
    allhits = np.zeros((8,8))
    for pkl in pklfiles:
        try:
            core_data = data_to_change_detection_core(pd.read_pickle(os.path.join(mousedir,pkl)))
            trials = create_extended_dataframe(
                    trials=core_data['trials'],
                    metadata=core_data['metadata'],
                    licks=core_data['licks'],
                    time=core_data['time'])
            if 'im061' in trials.change_image_name.to_list():
            
                lastRewardTime = 0
                interval = 0
                for it, tr in enumerate(trials.reward_times.values[0:]):
                    if len(tr)>0:
                        interval = tr[0] - lastRewardTime
                        lastRewardTime = tr[0]
                    if interval>120 and it>0:
                        break
                lastEngaged = it
                trials = trials.iloc[:it]
                c, h = makeBehaviorMatrix(trials, imIDs)
                print(c)
                print(h)
                allcounts = allcounts + c
                allhits = allhits + h
                
        
        except:
            print('failed to load ' + pkl)
            continue
    if np.max(allcounts)>0:
        data[0].append(mouse)
        data[1].append(allcounts)
        data[2].append(allhits)
        
        
        hitRateMat = allhits/allcounts
        fig, ax = plt.subplots()
        fig.suptitle(mouse)
        ax.imshow(hitRateMat, cmap='magma', clim=[0,1])
        [a(np.arange(8)) for a in [ax.set_yticks, ax.set_xticks]]
        [a(imIDs) for a in [ax.set_yticklabels, ax.set_xticklabels]]
    
            
            
### all mice together ####
        
all_mouse_counts = np.sum(data[1], axis=0)
all_mouse_hits = np.sum(data[2], axis=0)
all_hit_rate = all_mouse_hits/all_mouse_counts
fig, ax = plt.subplots()
ax.imshow(all_hit_rate, cmap='magma', clim=[0,1])
[a(np.arange(8)) for a in [ax.set_yticks, ax.set_xticks]]
[a(imIDs) for a in [ax.set_yticklabels, ax.set_xticklabels]]
        
        
        