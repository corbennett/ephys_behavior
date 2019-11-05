# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os, glob, h5py, nrrd, cv2, datetime
from xml.dom import minidom
import numpy as np
import pandas as pd
import fileIO
from sync import sync
import probeSync
import visual_behavior
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from analysis_utils import find_run_transitions
import analysis_utils

#Parent directory with sorted data, sync and pkl files
dataDir = 'Z:\\04112019'

#Which probes to run through analysis
probeIDs = 'ABCDEF'

class behaviorEphys():
    
    def __init__(self, baseDir=None, probes=None, probeGen='3b'):
        if baseDir is None:        
            self.dataDir = dataDir
        else:
            self.dataDir = baseDir
        
        if probeGen=='pipeline':
            sync_file = glob.glob(os.path.join(self.dataDir,'*.sync'))
            
        else:
            sync_file = glob.glob(os.path.join(self.dataDir,'*'+('[0-9]'*18)+'.h5'))
            
        self.sync_file = sync_file[0] if len(sync_file)>0 else None
        self.syncDataset = sync.Dataset(self.sync_file) if self.sync_file is not None else None
        if probes is None:
            self.probes_to_analyze = probeIDs
        else:
            self.probes_to_analyze = probes
        self.experimentDate = os.path.basename(self.dataDir)[:8]
        self.probeGen = probeGen
        if probeGen=='3b':
            if datetime.datetime.strptime(self.experimentDate,'%m%d%Y') < datetime.datetime(2019,3,15):
                fprobe = '4'
            else:
                fprobe = '3'
            self.PXIDict = {'A': 'slot2-probe1', 'B': 'slot2-probe2', 'C': 'slot2-probe3', 'D': 'slot3-probe1', 'E': 'slot3-probe2', 'F': 'slot3-probe'+fprobe}
        else:
            self.PXIDict = None
        
    def saveHDF5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
    
    def loadFromHDF5(self, filePath=None):
        fileIO.hdf5ToObj(self,filePath)
        self.syncDataset = sync.Dataset(self.sync_file)
        self.getBehaviorData()
        self.getRFandFlashStimInfo()
        self.getPassiveStimInfo()
    
    def loadFromRawData(self):        
        #self.getLFP()
        self.getFrameTimes()
        self.getBehaviorData()
        self.getEyeTrackingData()
        self.getRFandFlashStimInfo()
        self.getPassiveStimInfo()
        self.getUnits()
        self.getVisualResponsiveness()
        self.getCCFPositions()
        
    def getUnits(self):    
        self.units = {str(pid): probeSync.getUnitData(self.dataDir, self.syncDataset, pid, self.PXIDict, self.probeGen) for pid in self.probes_to_analyze}
    
    def getLFP(self):
        self.lfp = {}
        for pid in self.probes_to_analyze:
            self.lfp[pid] = {}
            lfpData,lfpTimestamps = probeSync.getLFPData(self.dataDir, pid, self.syncDataset, self.PXIDict, self.probeGen)
            self.lfp[pid]['data'] = lfpData
            self.lfp[pid]['time'] = lfpTimestamps
    
    def getCCFPositions(self):
        # get unit CCF positions
        self.probeCCFFile = glob.glob(os.path.join(self.dataDir,'probePosCCF_*'+('[0-9]'*8)+'_'+('[0-9]'*6)+'.xlsx'))
        #if len(self.probeCCFFile)>0:
        try:
            probeCCF = pd.read_excel(self.probeCCFFile[0])
            ccfDir = os.path.dirname(self.dataDir)
            annotationStructures = minidom.parse(os.path.join(ccfDir,'annotationStructures.xml'))
            annotationData = nrrd.read(os.path.join(ccfDir,'annotation_25.nrrd'))[0].transpose((1,2,0))
            tipLength = 201
            self.probeCCF = {}
            for pid in self.probes_to_analyze:
                entry,tip = [np.array(probeCCF[pid+' '+loc]) for loc in ('entry','tip')]
                entryChannel = entry[5]
                dx,dy,dz = [tip[i]-entry[i] for i in range(3)]
                trackLength = (dx**2+dy**2+dz**2)**0.5
                probeLength = tipLength+entryChannel*10 # length of probe in brain
                scaleFactor = trackLength/probeLength
                shift = entry[3]
                stretch = entry[4]
                self.probeCCF[pid] = {}
                self.probeCCF[pid]['entry'] = entry
                self.probeCCF[pid]['tip'] = tip
                self.probeCCF[pid]['shift'] = shift
                self.probeCCF[pid]['stretch'] = stretch
                self.probeCCF[pid]['entryChannel'] = entryChannel
                self.probeCCF[pid]['ISIRegion'] = entry[6] if isinstance(entry[6],basestring) else None
                for u in self.units[pid]:
                    distFromTip = tipLength+self.units[pid][u]['position'][1]
                    distFromEntry = probeLength-distFromTip
                    self.units[pid][u]['ccf'] = entry[:3]+(shift+distFromEntry*scaleFactor*stretch)*np.array([dx,dy,dz])/trackLength
                    self.units[pid][u]['ccfID'] = annotationData[tuple(int(self.units[pid][u]['ccf'][c]/25) for c in (1,0,2))]
                    self.units[pid][u]['ccfRegion'] = None
                    inCortex = False
                    if self.units[pid][u]['ccf'][1] >= 0:
                        graphOrder = None
                        for ind,structID in enumerate(annotationStructures.getElementsByTagName('id')):
                            if int(structID.childNodes[0].nodeValue)==self.units[pid][u]['ccfID']:
                                structure = annotationStructures.getElementsByTagName('structure')[ind]
                                acronym = structure.childNodes[7].childNodes[0].nodeValue[1:-1]
                                graphOrder = int(structure.childNodes[13].childNodes[0].nodeValue)
                                self.units[pid][u]['ccfRegion'] = acronym
                                break
                        if graphOrder is not None:
                            while graphOrder > 5:
                                structure = structure.parentNode.parentNode
                                graphOrder = int(structure.childNodes[13].childNodes[0].nodeValue)
                            if 'Isocortex' in structure.childNodes[7].childNodes[0].nodeValue[1:-1]:
                                inCortex = True
                    self.units[pid][u]['inCortex'] = inCortex
        except:
            for pid in self.probes_to_analyze:
                for u in self.units[pid]:
                    for key in ('ccf','ccfID','ccfRegion','inCortex'):
                        self.units[pid][u][key] = None               

                    
    def saveCCFPositionsAsArray(self,appendEntry=True):
        for pid in self.probes_to_analyze:
            f = os.path.join(self.dataDir,'UnitAndTipCCFPositions_probe'+pid+'.npy')
            d = np.array([self.units[pid][u]['ccf'] for u in probeSync.getOrderedUnits(self.units[pid])]).astype(float)
            if appendEntry:
                d = np.concatenate((d,self.probeCCF[pid]['entry'][None,:3])).astype(float) # add probe entry point
            d /= 25 # 25 um per ccf voxel
            d += 1 # for ImageGui
            np.save(f,d)
            
            rf = os.path.join(self.dataDir,'VisualResponsiveness_probe'+pid+'.npy')
            r = np.array([self.units[pid][u]['peakMeanVisualResponse'] for u in probeSync.getOrderedUnits(self.units[pid])]).astype(float)
            if appendEntry:
                r = np.append(r,np.median(r)).astype(float) # add probe entry point
            np.save(rf, r)      

                
    def getFrameTimes(self):
        # Get frame times from sync file
        frameRising, frameFalling = probeSync.get_sync_line_data(self.syncDataset, 'stim_vsync')
        
        #diode = probeSync.get_sync_line_data(syncDataset, 'photodiode')
        #monitorLags = diode[0][4:4+frameFalling[60::120].size][:100] - frameFalling[60::120][:100]

        # some experiments appear to have an extra frameFalling at the beginning that doesn't have a corresponding
        # frame in the behavior pkl file; this is probably caused by the DAQ starting high and being reinitialized
        # to zero a few seconds before psychopy and the normal vsyncs start
        self.vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
        
        # use vsyncTimes for all data streams except the stimulus frame times, which are subject to monitor lag
        monitorLag = 0.036
        self.frameAppearTimes = self.vsyncTimes + monitorLag    
    
    
    def getBehaviorData(self):
        # get behavior data
        if not hasattr(self, 'vsyncTimes'):
            self.getFrameTimes()
            
        self.pkl_file = glob.glob(os.path.join(self.dataDir,'*[0-9].pkl'))[0]
        behaviordata = pd.read_pickle(self.pkl_file)
        self.core_data = data_to_change_detection_core(behaviordata)
        
        self.trials = create_extended_dataframe(
            trials=self.core_data['trials'],
            metadata=self.core_data['metadata'],
            licks=self.core_data['licks'],
            time=self.core_data['time'])
        
        self.behaviorVsyncCount = self.core_data['time'].size # same as self.trials['endframe'].values[-1] + 1
        
        self.flashFrames = np.array(self.core_data['visual_stimuli']['frame'])
        self.flashImage = self.core_data['visual_stimuli']['image_name']
        self.changeFrames = np.array(self.trials['change_frame']).astype(int)+1 #add one to correct for change frame indexing problem
        self.initialImage = np.array(self.trials['initial_image_name'])
        self.changeImage = np.array(self.trials['change_image_name'])
        
        self.images = self.core_data['image_set']['images']
        newSize = tuple(int(s/10) for s in self.images[0].shape[::-1])
        self.imagesDownsampled = [cv2.resize(img,newSize,interpolation=cv2.INTER_AREA) for img in self.images]
        self.imageNames = [i['image_name'] for i in self.core_data['image_set']['image_attributes']]
        
        candidateOmittedFlashFrames = behaviordata['items']['behavior']['stimuli']['images']['flashes_omitted']
        drawlog = behaviordata['items']['behavior']['stimuli']['images']['draw_log']
        self.omittedFlashFrames = np.array([c for c in candidateOmittedFlashFrames if not drawlog[c]])
        imageFrameIndexBeforeOmitted = np.searchsorted(self.core_data['visual_stimuli']['frame'], self.omittedFlashFrames)-1
        self.omittedFlashImage = np.array(self.core_data['visual_stimuli']['image_name'])[imageFrameIndexBeforeOmitted]
        
        self.behaviorStimDur = np.array(self.core_data['visual_stimuli']['duration'])
        self.preGrayDur = np.stack(self.trials['blank_duration_range']) # where is actual gray dur
        self.lastBehaviorTime = self.frameAppearTimes[self.trials['endframe'].values[-1]]
        
        # align trials to sync
        self.trial_start_frames = np.array(self.trials['startframe'])
        self.trial_end_frames = np.array(self.trials['endframe'])
        self.trial_start_times = self.frameAppearTimes[self.trial_start_frames]
        self.trial_end_times = self.frameAppearTimes[self.trial_end_frames]
        
        # trial info
        self.autoRewarded = np.array(self.trials['auto_rewarded']).astype(bool)
        self.earlyResponse = np.array(self.trials['response_type']=='EARLY_RESPONSE')
        self.ignore = self.earlyResponse | self.autoRewarded
        self.miss = np.array(self.trials['response_type']=='MISS')
        self.hit = np.array(self.trials['response_type']=='HIT')
        self.falseAlarm = np.array(self.trials['response_type']=='FA')
        self.correctReject = np.array(self.trials['response_type']=='CR')
        
        # get running data
        self.behaviorRunTime = self.vsyncTimes[self.core_data['running'].frame]
        self.behaviorRunSpeed = self.core_data['running'].speed
        
        # get run start times
        self.behaviorRunStartTimes = find_run_transitions(self.behaviorRunSpeed, self.behaviorRunTime)
    
        #get lick and reward times
        self.lickTimes = probeSync.get_sync_line_data(self.syncDataset, 'lick_sensor')[0]
        if len(self.lickTimes)==0:
            self.lickTimes = self.vsyncTimes[np.concatenate([lf for lf in self.trials['lick_frames']]).astype(int)]
            
        self.rewardTimes = self.vsyncTimes[self.trials['reward_frames'].astype(int)]
        
    
    def getEyeTrackingData(self):
        # get eye tracking data
        self.eyeFrameTimes = probeSync.get_sync_line_data(self.syncDataset,'cam2_exposure')[0]
        
        #camPath = glob.glob(os.path.join(dataDir,'cameras','*-1.h5'))[0]
        #camData = h5py.File(camPath)
        #frameIntervals = camData['frame_intervals'][:]
        
        self.eyeDataPath = glob.glob(os.path.join(self.dataDir,'cameras','*_eyetrack_analysis.hdf5'))
        if len(self.eyeDataPath)>0:
            self.eyeData = h5py.File(self.eyeDataPath[0])
            self.pupilArea = self.eyeData['pupilArea'][:]
            self.pupilX = self.eyeData['pupilX'][:]
            self.negSaccades = self.eyeData['negSaccades'][:]
            self.posSaccades = self.eyeData['posSaccades'][:]
        else:
            self.eyeData = None
            
    
    def getRFandFlashStimInfo(self):
        rf_pickle = glob.glob(os.path.join(self.dataDir, '*brain_observatory_stimulus.pkl'))
        if len(rf_pickle)==0:
            self.rf_pickel_file = None
        else:
            self.rf_pickle_file = rf_pickle[0]
            self.rfFlashStimDict = pd.read_pickle(self.rf_pickle_file)
            self.monSizePix = self.rfFlashStimDict['monitor']['sizepix']
            self.monHeightCm = self.monSizePix[1]/self.monSizePix[0]*self.rfFlashStimDict['monitor']['widthcm']
            self.monDistCm = self.rfFlashStimDict['monitor']['distancecm']
            self.monHeightDeg = np.degrees(2*np.arctan(0.5*self.monHeightCm/self.monDistCm))
            self.imagePixPerDeg = self.images[0].shape[0]/self.monHeightDeg 
            self.imageDownsamplePixPerDeg = self.imagesDownsampled[0].shape[0]/self.monHeightDeg
            
            self.rfStimParams = self.rfFlashStimDict['stimuli'][0]
            rf_pre_blank_frames = int(self.rfFlashStimDict['pre_blank_sec']*self.rfFlashStimDict['fps'])
            first_rf_frame = self.behaviorVsyncCount + rf_pre_blank_frames
            self.rf_frameTimes = self.frameAppearTimes[first_rf_frame:]
            self.rf_trial_start_times = self.rf_frameTimes[np.array([f[0] for f in np.array(self.rfStimParams['sweep_frames'])]).astype(np.int)]
            
            self.flashStimParams = self.rfFlashStimDict['stimuli'][1]
    
        
    def getPassiveStimInfo(self):
        passivePklFiles = glob.glob(os.path.join(self.dataDir, '*-replay-script*.pkl'))
        if len(passivePklFiles)==0:
            self.passive_pickle_file = None
        else:
            if len(passivePklFiles)>1:
                vsynccount = [pd.read_pickle(f)['vsynccount'] for f in passivePklFiles]
                goodPklInd = np.argmax(vsynccount)
                self.passive_pickle_file = passivePklFiles[goodPklInd]
                abortedVsyncs = sum(vsynccount)-vsynccount[goodPklInd]
            else:
                self.passive_pickle_file = passivePklFiles[0]
                abortedVsyncs = 0
            self.passiveStimDict = pd.read_pickle(self.passive_pickle_file)
            self.passiveStimParams = self.passiveStimDict['stimuli'][0]
            self.passiveFrameImages = np.array(self.passiveStimParams['sweep_params']['ReplaceImage'][0])
            passiveImageNames = [img for img in np.unique(self.passiveFrameImages) if img is not None]
            nonGrayFrames = np.in1d(self.passiveFrameImages,passiveImageNames)
            self.passiveImageOnsetFrames = np.where(np.diff(nonGrayFrames.astype(int))>0)[0]+1
            self.passiveChangeFrames = np.array([frame for i,frame in enumerate(self.passiveImageOnsetFrames[1:]) if self.passiveFrameImages[frame]!=self.passiveFrameImages[self.passiveImageOnsetFrames[i]]])
            self.passiveChangeImages = self.passiveFrameImages[self.passiveChangeFrames]
            firstPassiveFrame = self.behaviorVsyncCount + self.rfFlashStimDict['vsynccount'] + abortedVsyncs
            self.passiveFrameAppearTimes = self.frameAppearTimes[firstPassiveFrame:]
            
            # get running data
            dx,vsig,vin = [self.passiveStimDict['items']['foraging']['encoders'][0][key] for key in ('dx','vsig','vin')]
            self.passiveRunTime = self.vsyncTimes[firstPassiveFrame:]
            self.passiveRunSpeed = visual_behavior.analyze.compute_running_speed(dx,self.passiveRunTime,vsig,vin)
            
    
    def getVisualResponsiveness(self):
        image_flash_times = self.frameAppearTimes[np.array(self.core_data['visual_stimuli']['frame'])]
        image_id = np.array(self.core_data['visual_stimuli']['image_name'])
        
        #take first 50 flashes of each image
        image_flash_times = np.array([image_flash_times[np.where(image_id==im)[0][:50]] for im in np.unique(image_id)]).flatten()
        for pid in self.probes_to_analyze:
            for u in self.units[pid]:
                spikes = self.units[pid][u]['times']
                #find mean response to all flashes
                p, t = analysis_utils.getSDF(spikes,image_flash_times-.25,0.5, sigma=0.01)

                self.units[pid][u]['peakMeanVisualResponse'] = p.max() - p[:250].mean()
                            

    def getUnitsByArea(self, area, cortex=True):
        pids = []
        us = []
        for pid in self.probes_to_analyze:
            for u in probeSync.getOrderedUnits(self.units[pid]):
                
                if cortex:
                   if self.units[pid][u]['inCortex'] and area==self.probeCCF[pid]['ISIRegion']:
                       pids.append(pid)
                       us.append(u)
                else:
                    if area==self.units[pid][u]['ccfRegion']:
                        pids.append(pid)
                        us.append(u)
        return np.array(pids), np.array(us)
            
        
    #for pid in probeIDs:
    #    plfp = lfp[pid][0]   
    #    gammapower = []
    #    thetapower = []
    #    for i in np.arange(384):
    #        f, pxx = scipy.signal.welch(plfp[:10000, i], fs=2500, nperseg=5000)
    #        gammafreq = [30<ff<55 for ff in f]
    #        gamma = np.mean(pxx[gammafreq])
    #        gammapower.append(gamma)
    #        
    #        thetafreq = [5<ff<15 for ff in f]
    #        theta = np.mean(pxx[thetafreq])
    #        thetapower.append(theta)
    #    
    #    fig, ax = plt.subplots()
    #    fig.suptitle(pid)
    #    ax.plot(gammapower/max(gammapower), 'k')    
    #    ax.plot(thetapower/max(thetapower), 'g')
    #
    #    unitchannels = [units[pid][u]['peakChan'] for u in probeSync.getOrderedUnits(units[pid])]
    #    ax.plot(max(unitchannels), 1, 'ro')
    


