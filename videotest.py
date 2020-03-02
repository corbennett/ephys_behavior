# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:52:40 2020

@author: svc_ccg
"""

from __future__ import division
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
import fileIO
from sync import sync
import probeSync


dataDir = r'Z:'

syncFile = fileIO.getFile('choose sync file',dataDir,'*.h5')
syncDataset = sync.Dataset(syncFile)


lickTimes = probeSync.get_sync_line_data(syncDataset, channel=31)[0]

cam1FramesRising,cam1FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')

cam2FramesRising,cam2FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam2_exposure')

cam3FramesRising,cam3FramesFalling = probeSync.get_sync_line_data(syncDataset, channel=10)

camFrameTimes = np.sort(np.concatenate((camFramesRising,camFramesFalling)))

camFramesWithLicks = np.searchsorted(camFrameTimes,lickTimes)




filePaths = fileIO.getFiles('choose video files',dataDir,'*.avi *.mp4')

bins = np.arange(0,257)
hist = []
meanFrame = []
exampleFrame = []
label = []
for f in filePaths:
    label.append(f[[s.start() for s in re.finditer('_',f)][-1]+1:-4])
    v = cv2.VideoCapture(f)
    frames = []
    while True:
        isImage,image = v.read()
        if not isImage:
            break
        frames.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    v.release()
    meanFrame.append(np.mean(frames[15:],axis=0))
    exampleFrame.append(frames[15])
    hist.append(np.histogram(frames[15:],bins)[0])

plt.figure()
ax = plt.subplot(1,1,1)
for h,lbl,clr in zip(hist,label,('0.5','r','g','b','k')):
    ax.plot(bins[:-1],h,clr,label=lbl)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_yscale('log')
ax.set_xlim([-1,256])
ax.set_xlabel('pixel intensity',fontsize=14)
ax.set_ylabel('count',fontsize=14)
ax.legend(fontsize=14)
plt.tight_layout()

ax.set_xlim([-1,10])
ax.set_xlim([245,256])

plt.figure()
for i,(img,lbl) in enumerate(zip(exampleFrame,label)):
    ax = plt.subplot(2,2,i+1)
    ax.imshow(img,cmap='gray',clim=[0,255])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(lbl)
plt.tight_layout()












