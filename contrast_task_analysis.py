# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:20:54 2022

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO


class DocData():
    
    def __init__(self):
        self.frameRate = 60
    
    def loadBehavData(self,filePath=None):
        if filePath is None:
            self.behavDataPath = fileIO.getFile('Select behavior data file',fileType='*.pkl')
        else:
            self.behavDataPath = filePath
        if len(self.behavDataPath)==0:
            return

        pkl = pd.read_pickle(self.behavDataPath)
        self.params = pkl['items']['behavior']['params']
        self.trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])
#        changeLog = pkl['items']['behavior']['stimuli']['images-params']['change_log']
#        setLog = pkl['items']['behavior']['stimuli']['images-params']['set_log']
        
        nTrials = len(self.trialLog)
        self.trialStartTimes = np.full(nTrials,np.nan)
        self.trialStartFrames = np.full(nTrials,np.nan)
        self.trialEndTimes = np.full(nTrials,np.nan)
        self.abortedTrials = np.zeros(nTrials,dtype=bool)
        self.abortTimes = np.full(nTrials,np.nan)
        self.changeTimes = np.full(nTrials,np.nan)
        self.changeFrames = np.full(nTrials,np.nan)
        self.changeTrials = np.zeros(nTrials,dtype=bool)
        self.catchTrials = np.zeros(nTrials,dtype=bool)
        self.warmupTrials = np.zeros(nTrials,dtype=bool)
        self.rewardTimes = np.full(nTrials,np.nan)
        self.autoReward = np.zeros(nTrials,dtype=bool)
        self.hit = np.zeros(nTrials,dtype=bool)
        self.miss = np.zeros(nTrials,dtype=bool)
        self.falseAlarm = np.zeros(nTrials,dtype=bool)
        self.correctReject = np.zeros(nTrials,dtype=bool)
        self.preContrast = np.full(nTrials,np.nan)
        self.postContrast = np.full(nTrials,np.nan)
        self.preImage = ['' for _ in self.trialLog]
        self.postImage = self.preImage.copy()
        self.preLabel = self.preImage.copy()
        self.postLabel = self.preImage.copy()
        for i,trial in enumerate(self.trialLog):
            events = [event[0] for event in trial['events']]
            for event,epoch,t,frame in trial['events']:
                if event=='trial_start':
                    self.trialStartTimes[i] = t
                    self.trialStartFrames[i] = frame
                elif event=='trial_end':
                    self.trialEndTimes[i] = t
                elif 'abort' in events:
                    if event=='abort':
                        self.abortedTrials[i] = True
                        self.abortTimes[i] = t
                elif event in ('stimulus_changed','sham_change'):
                    self.changeTimes[i] = t
                    self.changeFrames[i] = frame
                elif event=='hit':
                    self.hit[i] = True
                elif event=='miss':
                    self.miss[i] = True 
                elif event=='false_alarm':
                    self.falseAlarm[i] = True
                elif event=='rejection':
                    self.correctReject[i] = True
            if 'warmup' in trial['trial_params']:
                self.warmupTrials[i] = trial['trial_params']['warmup']
            if len(trial['rewards']) > 0:
                self.rewardTimes[i] = trial['rewards'][0][1]
                self.autoReward[i] = trial['trial_params']['auto_reward']
            if not self.abortedTrials[i]:
                if trial['trial_params']['catch']:
                    self.catchTrials[i] = True
                else:
                    self.changeTrials[i] = True
                if trial['stimulus_changes'][0][0][0] is not None:
                    self.preImage[i] = trial['stimulus_changes'][0][0][0]
                    self.postImage[i] = trial['stimulus_changes'][0][1][0]
                    self.preContrast[i] = trial['stimulus_changes'][0][0][1]['contrast']
                    self.postContrast[i] = trial['stimulus_changes'][0][1][1]['contrast']
                    self.preLabel[i] = self.preImage[i]+' ('+str(self.preContrast[i])+')'
                    self.postLabel[i] = self.postImage[i]+' ('+str(self.postContrast[i])+')'
                
        self.engaged = np.array([np.sum(self.hit[self.changeTrials][(self.changeTimes[self.changeTrials]>t-60) & (self.changeTimes[self.changeTrials]<t+60)]) > 1 for t in self.changeTimes])
            
        self.labels = sorted(list(set(self.preLabel+self.postLabel))[1:])
        self.trialCount = np.zeros((len(self.labels),)*2)
        self.trialCountEngaged = self.trialCount.copy()
        self.respCountEngaged = self.trialCount.copy()
        self.imageChange = self.trialCount.astype(bool)
        for i,postLbl in enumerate(self.labels):
            for j,preLbl in enumerate(self.labels):
                img = [lbl[:lbl.find(' (')] for lbl in (preLbl,postLbl)]
                if img[0] != img[1]:
                    self.imageChange[i,j] = True
                for pre,post,h,fa,wu,ar,eng in zip(self.preLabel,self.postLabel,self.hit,self.falseAlarm,self.warmupTrials,self.autoReward,self.engaged):
                    if pre==preLbl and post==postLbl and not wu:
                        self.trialCount[i,j] += 1
                        if not ar and eng:
                            self.trialCountEngaged[i,j] += 1
                            self.respCountEngaged[i,j] += h or fa
        self.respRate = self.respCountEngaged/self.trialCountEngaged
    
    def plotSummary(self):
        for d,lbl in zip((self.trialCount,self.respRate),('Trials','Response Rate')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            im = ax.imshow(d,clim=[0,d.max()],cmap='gray')
            for i,postLbl in enumerate(self.labels):
                for j,preLbl in enumerate(self.labels):
                    if not self.imageChange[i,j]:
                        ax.plot(i,j,'rx')
            ax.set_xticks(np.arange(len(self.labels)))
            ax.set_xticklabels(self.labels,rotation=90)
            ax.set_xlabel('Pre Image (contrast)')
            ax.set_xlim([-0.5,len(self.labels)-0.5])
            ax.set_yticks(np.arange(len(self.labels)))
            ax.set_yticklabels(self.labels)
            ax.set_ylabel('Change Image (contrast)')
            ax.set_ylim([len(self.labels)-0.5,-0.5])
            ax.set_title(lbl+' (x = no image identity change)')
            cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
            plt.tight_layout()
# end DocData


# get data
behavFiles = []
while True:
    files = fileIO.getFiles('choose experiments',fileType='*.pkl')
    if len(files)>0:
        behavFiles.extend(files)
    else:
        break
    
if len(behavFiles)>0:
    exps = []
    for f in behavFiles:
        obj = DocData()
        obj.loadBehavData(f)
        exps.append(obj)
        

#
trialCountAll,trialCount,respCount,imageChange = [np.array([getattr(obj,attr) for obj in exps]) for attr in ('trialCount','trialCountEngaged','respCountEngaged','imageChange')]
respRate = respCount/trialCount

labels = [obj.labels for obj in exps]

index = []
for lbls in labels:
    index.append([])
    index[-1].append(lbls.index('im115_r (0.7)'))
    index[-1].append(lbls.index('im115_r (1.0)'))
    index[-1].append(lbls.index('im047_r (1.0)'))
    index[-1] += [i for i in range(4) if i not in index[-1]]


r = respCount.copy()
r[~imageChange] = 0
c = trialCount.copy()
c[~imageChange] = 0
preImgRespRate = r.sum(axis=1)/c.sum(axis=1)
changeImgRespRate = r.sum(axis=2)/c.sum(axis=2)

imgs = np.unique(exps[0].preImage)[1:]
contrast = exps[0].preContrast
contrast = np.unique(contrast[~np.isnan(contrast)])


rr = []
for r,ind in zip(respRate,index):
    r = r[ind,:]
    r = r[:,ind]
    rr.append(r)
r = np.nanmean(rr,axis=0)
lbls = np.array(labels[0])[index[0]]
lbls[-1] = 'novel (1.0)'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(r,clim=[0,1],cmap='magma')
for i in range(len(lbls)):
    for j in range(len(lbls)):
        ax.text(j,i,str(round(r[i,j],2)),color='w',ha='center',va='center')
ax.set_xticks(np.arange(len(lbls)))
ax.set_xticklabels(lbls,rotation=90)
ax.set_xlabel('Pre Image (contrast)')
ax.set_xlim([-0.5,len(lbls)-0.5])
ax.set_yticks(np.arange(len(lbls)))
ax.set_yticklabels(lbls)
ax.set_ylabel('Change Image (contrast)')
ax.set_ylim([len(lbls)-0.5,-0.5])
ax.set_title('Response Rate')
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
plt.tight_layout()


for r,c,lbls,ind in zip(respCount,trialCount,labels,index):
    r = r[ind,:]
    r = r[:,ind]
    c = c[ind,:]
    c = c[:,ind]
    lbls = np.array(lbls)[ind]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(r/c,clim=[0,1],cmap='magma')
    for i in range(len(lbls)):
        for j in range(len(lbls)):
            ax.text(j,i,str(int(r[i,j]))+'/'+str(int(c[i,j])),color='w',ha='center',va='center')
    ax.set_xticks(np.arange(len(lbls)))
    ax.set_xticklabels(lbls,rotation=90)
    ax.set_xlabel('Pre Image (contrast)')
    ax.set_xlim([-0.5,len(lbls)-0.5])
    ax.set_yticks(np.arange(len(lbls)))
    ax.set_yticklabels(lbls)
    ax.set_ylabel('Change Image (contrast)')
    ax.set_ylim([len(lbls)-0.5,-0.5])
    ax.set_title('Response Rate')
    cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
    plt.tight_layout()


for c,lbls,ind in zip(trialCountAll,labels,index):
    c = c[ind,:]
    c = c[:,ind]    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(c,cmap='gray')
    for i in range(len(lbls)):
        for j in range(len(lbls)):
            ax.text(j,i,str(int(c[i,j])),color='r',ha='center',va='center')
    ax.set_xticks(np.arange(len(lbls)))
    ax.set_xticklabels(lbls,rotation=90)
    ax.set_xlabel('Pre Image (contrast)')
    ax.set_xlim([-0.5,len(lbls)-0.5])
    ax.set_yticks(np.arange(len(lbls)))
    ax.set_yticklabels(lbls)
    ax.set_ylabel('Change Image (contrast)')
    ax.set_ylim([len(lbls)-0.5,-0.5])
    ax.set_title('Trial Count')
    cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
    plt.tight_layout()


















fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'--',color='0.5')
for rr,mfc,lbl in zip((preImgRespRate,changeImgRespRate),('none','k'),('pre image','change image')):
    x = rr[:,1::2].mean(axis=0)
    y = rr[:,::2].mean(axis=0)
    ax.plot(x,y,'o',mec='k',mfc=mfc,label=lbl)
    for i,j,img in zip(y,x,imgs):
#        if lbl=='change image' and img=='im111_r':
#            ax.text(j,i-0.02,img[:-2],ha='center',va='top')
#        else:
        ax.text(j,i+0.02,img[:-2],ha='center',va='bottom')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Response rate, '+str(int(contrast[1]*100))+'% contrast')
ax.set_ylabel('Response rate, '+str(int(contrast[0]*100))+'% contrast')
leg = ax.legend(loc='upper left')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'--',color='0.5')
ax.plot(changeImgRespRate[:,1::2].mean(axis=1),rr[:,::2].mean(axis=1),'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,1.02])
ax.set_ylim([0,1.02])
ax.set_aspect('equal')
ax.set_xlabel('Response rate, '+str(int(contrast[1]*100))+'% contrast')
ax.set_ylabel('Response rate, '+str(int(contrast[0]*100))+'% contrast')
plt.tight_layout()



#
hitRateLow = []
hitRateHigh = []
falseAlarmRate = []
lowTrials = []
highTrials = []
catchTrials = []

hitRateLow.append([])
hitRateHigh.append([])
falseAlarmRate.append([])
lowTrials.append([])
highTrials.append([])
catchTrials.append([])
for obj in exps[::-1]:
    r = obj.respCount.copy()
    r[~obj.imageChange] = 0
    c = obj.trialCount.copy()
    c[~obj.imageChange] = 0
    hitRateLow[-1].append(r[0::2].sum()/c[0::2].sum())
    hitRateHigh[-1].append(r[1::2].sum()/c[1::2].sum())
    lowTrials[-1].append(c[0::2].sum())
    highTrials[-1].append(c[1::2].sum())
    fa = np.eye(obj.respCount.shape[0]).astype(bool)
    r = obj.respCount[fa]
    c = obj.trialCount[fa]
    falseAlarmRate[-1].append(r.sum()/c.sum())
    catchTrials[-1].append(c.sum())
    
dp = []
for hr,n in zip((hitRateHigh,hitRateLow),(highTrials,lowTrials)):
    hr = np.array(hr)
    hr[hr==0] = 0.5/np.array(n)[hr==0]
    hr[hr==1] = 1-0.5/np.array(n)[hr==1]
    fa = np.array(falseAlarmRate)
    fa[fa==0] = 0.5/np.array(catchTrials)[fa==0]
    fa[fa==1] = 1-0.5/np.array(catchTrials)[fa==1]
    z = [scipy.stats.norm.ppf(r) for r in (hr,fa)]
    dp.append(np.mean(z[0]-z[1],axis=0))

    
xlbl = ('20%','40%')+('60%',)*(len(exps)-2)

xlbl = [1,2,3]
    
fig = plt.figure(figsize=(6,9))
ax = fig.add_subplot(3,1,1)
ax.plot(np.mean(hitRateHigh,axis=0),'g',lw=2,label='high contrast')
ax.plot(np.mean(hitRateLow,axis=0),'m',lw=2,label='low contrast')
ax.plot(np.mean(falseAlarmRate,axis=0),'0.5',lw=2,label='no change')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbl)))
ax.set_xticklabels(xlbl)
ax.set_ylim([0,1])
ax.set_xlabel('Session')
ax.set_ylabel('Response Rate')
ax.legend()
plt.tight_layout()
   
ax = fig.add_subplot(3,1,2)
ax.plot(dp[0],'g',lw=2,label='high contrast')
ax.plot(dp[1],'m',lw=2,label='low contrast')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbl)))
ax.set_xticklabels(xlbl)
ax.set_ylim([0,3])
ax.set_xlabel('Session')
ax.set_ylabel('d prime')
ax.legend()
plt.tight_layout()

ax = fig.add_subplot(3,1,3)
ax.plot(100*(1-dp[1]/dp[0]),'k',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbl)))
ax.set_xticklabels(xlbl)
ax.set_ylim([35,65])
ax.set_xlabel('Session')
ax.set_ylabel('% reduction in d prime')
plt.tight_layout()








