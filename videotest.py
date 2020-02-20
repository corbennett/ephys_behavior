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


dataDir = r'Z:'

syncFile = fileIO.getFile('choose sync file',dataDir,'*.h5')
syncDataset = sync.Dataset(syncFile)


lickTimes = probeSync.get_sync_line_data(syncDataset, channel=31)[0]

cam1FramesRising,cam1FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')

cam2FramesRising,cam2FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam2_exposure')

cam3FramesRising,cam3FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam3_exposure')

camFrameTimes = np.sort(np.concatenate((camFramesRising,camFramesFalling)))

camFramesWithLicks = np.searchsorted(camFrameTimes,lickTimes)