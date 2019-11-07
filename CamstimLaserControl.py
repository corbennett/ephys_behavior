# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:13:48 2019

@author: svc_ccg
"""

import numpy as np
import nidaqmx


class LaserControl():
    
    def __init__(self):
        nidaqDeviceName = 'Dev2'
        aoSampleRate = 5000
        laserCh = 0
        laserAmp = 1 # volts
        laserDur = 1 # seconds
        laserRampDur = 0.1 # seconds
        
        laserBufferSize = int(laserDur * aoSampleRate) + 1
        self.laserSignal = np.zeros(laserBufferSize)
        self.laserSignal[:-1] = laserAmp
        if laserRampDur > 0:
            laserRamp = np.linspace(0,laserAmp,int(laserRampDur * aoSampleRate))
            self.laserSignal[:laserRamp.size] = laserRamp
            self.laserSignal[-(laserRamp.size+1):-1] = laserRamp[::-1]
        
        self.laserTask = nidaqmx.Task()
        self.laserTask.ao_channels.add_ao_voltage_chan(nidaqDeviceName+'/ao'+str(laserCh),min_val=0,max_val=1.5)
        self.laserTask.write(0)
        self.laserTask.timing.cfg_samp_clk_timing(aoSampleRate,samps_per_chan=laserBufferSize)

    def __del__(self):
        self.laserTask.close()
        
    def triggerLaser(self):
        self.laserTask.stop()
        self.laserTask.write(self.laserSignal,auto_start=True)
