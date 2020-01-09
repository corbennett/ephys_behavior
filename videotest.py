# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:52:40 2020

@author: svc_ccg
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import fileIO
from sync import sync
import probeSync


dataDir = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\videotest'

syncFile = fileIO.getFile('choose sync file',dataDir,'*.h5')
syncDataset = sync.Dataset(syncFile)


lickTimes = probeSync.get_sync_line_data(syncDataset, channel=31)[0]

camFrameTimes = probeSync.get_sync_line_data(syncDataset,'behavior')[0]

camFramesWithLicks = np.searchsorted(camFrameTimes,lickTimes)