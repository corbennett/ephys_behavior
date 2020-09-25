# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:42:10 2020

@author: svc_ccg
"""

from __future__ import division
import fileIO
import skvideo
skvideo.setFFmpegPath('C:\\Users\\svc_ccg\\Desktop\\ffmpeg\\bin')
import skvideo.io


savePath = fileIO.saveFile('Save movie as',fileType='*.mp4')

inputParams = {'-r': '30'}
outputParams = {'-r': '30', '-pxl_fmt': 'yuv420p', '-vcodec': 'libx264', '-crf': '23', '-preset': 'slow'}

# '-pix_fmt': 'yuv420p' important to avoid green screen on mac; number of pixels needs to be even

v = skvideo.io.FFmpegWriter(savePath,inputdict=inputParams,outputdict=outputParams)

for frame in data:
    v.writeFrame(data)

v.close()