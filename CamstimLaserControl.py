# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:13:48 2019

@author: svc_ccg
"""

import numpy as np
import nidaqmx


"""
this script looks kinda weird

run as main to test the layzer

init layzer and use a function to trigger it for deployment reasons
we want init on import
we nest in a top level function to reduce the complexity of the script calling this
"""


class LaserControl():
    
    def __init__(self,laserAmp=1.0,laserDur=1.0,laserRampDur=0.1,nidaqDeviceName='Dev2'):
        self.aoSampleRate = 5000.0
        aoBufferSize = int(laserDur * self.aoSampleRate) + 1
        self.laserRampDur = laserRampDur

        # AO0: laser input signal
        self.laserAOSignal = np.zeros((2,aoBufferSize))
        self.setLaserAmp(laserAmp)

        # A01: laser pulse timing signal
        self.laserAOSignal[1,:-1] = 5

        # initialize AO task
        self.laserAOTask = nidaqmx.Task()
        self.laserAOTask.ao_channels.add_ao_voltage_chan(nidaqDeviceName+'/ao0:1',min_val=0,max_val=5)
        self.laserAOTask.write([0,0])
        self.laserAOTask.timing.cfg_samp_clk_timing(self.aoSampleRate,samps_per_chan=aoBufferSize)

    def __del__(self):
        self.laserAOTask.close()

    def setLaserAmp(self,amp=1.0):
        assert(amp <= 1.5)
        self.laserAOSignal[0,:-1] = amp
        if self.laserRampDur > 0:
            ramp = np.linspace(0,amp,int(self.laserRampDur * self.aoSampleRate))
            self.laserAOSignal[0,:ramp.size] = ramp
            self.laserAOSignal[0,-(ramp.size+1):-1] = ramp[::-1]
   
    def triggerLaser(self):
        self.laserAOTask.stop()
        self.laserAOTask.write(self.laserAOSignal,auto_start=True)



Layzer = LaserControl()


def trigger_layzer():
    Layzer.triggerLaser()


if __name__ == "__main__":
    import time

    print("Triggering layzer test...")
    trigger_layzer()
    time.sleep(10)
    print('Ending...hopefully layzer will die at the end of this...')