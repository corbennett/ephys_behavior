# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:16:06 2019

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import h5py, os
from matplotlib import pyplot as plt
from matplotlib import cm
from analysis_utils import formatFigure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from analysis_utils import formatFigure

popFile = r"Z:\analysis\popData.hdf5"
data = h5py.File(popFile)

lickLatencies_session = []
lickLatencies_all = []
for i, (exp, d) in enumerate(data.iteritems()):
    if i < 13:
    #    d = data[exp]
        lickTimes = d['lickTimes'][()]
        changeTimes = d['behaviorChangeTimes'][()]
        
        firstLickInds = np.searchsorted(lickTimes, changeTimes)
        firstLickInds = firstLickInds[firstLickInds<len(lickTimes)]
        
        firstLickLatencies = lickTimes[firstLickInds]-changeTimes[:firstLickInds.size]
        
      
        firstLickLatencies = firstLickLatencies[firstLickLatencies<1]
        
        fig, ax = plt.subplots()
        fig.suptitle(exp)
        ax.hist(firstLickLatencies, 500)
        lickLatencies_session.append(firstLickLatencies)
        lickLatencies_all.extend(firstLickLatencies)
        
lickLatencies_all = np.array(lickLatencies_all)

fig, ax = plt.subplots()
bins = np.linspace(0, 1, 200)
ax.hist(lickLatencies_all[lickLatencies_all<0.15], bins, color='0.5')
ax.hist(lickLatencies_all[(0.15<lickLatencies_all)&(lickLatencies_all<0.75)], bins, color='g')
ax.hist(lickLatencies_all[0.75<lickLatencies_all], bins, color='r')
ax.set_xticks(np.arange(0, 1, 0.25))
ax.set_yticks(np.arange(0, 100, round(lickLatencies_all.size/100)))
ax.set_yticklabels(np.round(np.arange(0, 100, round(lickLatencies_all.size/100))/lickLatencies_all.size, 2))

formatFigure(fig, ax, xLabel='Time from Change (s)', yLabel='Fraction of licks')
