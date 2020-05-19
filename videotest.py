# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:52:40 2020

@author: svc_ccg
"""

from __future__ import division
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import fileIO
from sync import sync
import probeSync


dataDir = r'Z:'

syncFile = fileIO.getFile('choose sync file',dataDir,'*.h5')
syncDataset = sync.Dataset(syncFile)



plt.figure(figsize=(12,10))
gs = matplotlib.gridspec.GridSpec(3,3)
channelNames = ('sync','transmission','exposure')
bins = np.arange(0,17,0.1)
for i,cam in enumerate((1,2,3)):
    if cam==1:
        channels = (8,26,30) # cam1
    elif cam==2:
        channels = (9,25,29) # cam2
    elif cam==3:
        channels = (10,22,28) # cam3
    elif cam==4:
        channels = (8,21,27) # cam1 plugged into cam4
    for j,(ch,chname) in enumerate(zip(channels,channelNames)):
        ax = plt.subplot(gs[i,j])
        rising,falling = probeSync.get_sync_line_data(syncDataset,channel=ch)
        pulseDur = falling-rising[:len(falling)]
        pulseDur *= 1000
        ax.hist(pulseDur,bins=bins,color='k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_yscale('log')
        ax.set_xlim([0,1.2*pulseDur.max()])
        if i==2:
            ax.set_xlabel('Pulse duration (falling-rising, ms)')
        if j==0:
            ax.set_ylabel('Events')
        ax.set_title('cam'+str(cam)+' '+chname+' channel '+str(ch)+', '+str(len(falling))+' falling edges',fontsize=10)
plt.tight_layout()



lickTimes = probeSync.get_sync_line_data(syncDataset, channel=31)[0]

cam1FramesRising,cam1FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')

cam2FramesRising,cam2FramesFalling = probeSync.get_sync_line_data(syncDataset,'cam2_exposure')

cam3FramesRising,cam3FramesFalling = probeSync.get_sync_line_data(syncDataset, channel=10)

camFrameTimes = np.sort(np.concatenate((camFramesRising,camFramesFalling)))

camFramesWithLicks = np.searchsorted(camFrameTimes,lickTimes)


cam = 1

if cam==1:
    channels = (8,26,30) # cam1
elif cam==2:
    channels = (9,25,29) # cam2
elif cam==3:
    channels = (10,22,28) # cam3
elif cam==4:
    channels = (8,21,27) # cam1 plugged into cam4

channelNames = ('sync','transmission','exposure')

rising = [probeSync.get_sync_line_data(syncDataset, channel=ch)[0] for ch in channels]
falling = [probeSync.get_sync_line_data(syncDataset, channel=ch)[1] for ch in channels]

pulseDur = [fall-rise[:len(fall)] for rise,fall in zip(rising,falling)]





plt.figure()
ax = plt.subplot(1,1,1)
nrise = []
nfall = []
for i,(rise,fall,name) in enumerate(zip(rising,falling,channelNames)):
    ax.vlines(rise,i-0.4,i+0.4,colors='k')
    ax.vlines(fall,i-0.4,i+0.4,colors='0.5')
    nrise.append(len(rise))
    nfall.append(len(fall))
ax.set_yticks(np.arange(len(channels)))
ax.set_yticklabels([ch+'\n'+str(nr)+' rise,'+str(nf)+' fall' for ch,nr,nf in zip(channelNames,nrise,nfall)])
ax.set_xlabel('time (s)')
plt.tight_layout()

ax.set_xlim([sync[0]-0.05,sync[10]])
ax.set_xlim([sync[-10],sync[-1]+0.05])
ax.set_xlim([5.222,5.255])


filePaths = fileIO.getFiles('choose video files',dataDir,'*.avi *.mp4')

#filePaths.append(fileIO.getFiles('choose bitmaps',dataDir,'*.bmp'))

skipFrames = 15
maxFrames = 600

bins = np.arange(0,257)
hist = []
meanFrame = []
exampleFrame = []
label = []
for f in filePaths:
    if isinstance(f,list):
        label.append('bmp')
        frames = [cv2.cvtColor(cv2.imread(b),cv2.COLOR_BGR2GRAY) for b in f]
    else:
        label.append(f[[s.start() for s in re.finditer('_',f)][-1]+1:-4]) 
        v = cv2.VideoCapture(f)
        count = 0
        frames = []
        while True:
            isImage,image = v.read()
            if not isImage or count>maxFrames:
                break
            if count>0:
                frames.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            count += 1
        v.release()
    d = np.array(frames[skipFrames:])
    meanFrame.append(np.mean(d,axis=0))
    exampleFrame.append(d[0])
    hist.append(np.histogram(d,bins)[0]/d.size)

label = ['opencv','raw', 'yuv420 crf0', 'yuv420 crf 17', 'yuv420 crf23', 'gray crf17']
clrs = plt.cm.jet(np.linspace(0,1,len(filePaths)))
plt.figure()
ax = plt.subplot(1,1,1)
for i,(h,lbl,clr) in enumerate(zip(hist,label,clrs)):
    ax.plot(bins[:-1],h,color=clr,label=lbl)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_yscale('log')
ax.set_xlim([-1,256])
ax.set_xlabel('pixel intensity',fontsize=14)
ax.set_ylabel('fraction of pixels',fontsize=14)
ax.legend(fontsize=14)
plt.tight_layout()

ax.set_xlim([-1,10])
ax.set_xlim([245,256])

plt.figure(facecolor='0.5')
for i,(img,lbl) in enumerate(zip(exampleFrame,label)):
    ax = plt.subplot(1,len(filePaths),i+1)
    ax.imshow(img,cmap='gray',clim=[0,255])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(lbl)
plt.tight_layout()



cv2.imwrite(os.path.join(os.path.dirname(filePaths[0]),'avi.tif'),exampleFrame[0])

cv2.imwrite(os.path.join(os.path.dirname(filePaths[0]),'bmp.tif'),exampleFrame[1])






