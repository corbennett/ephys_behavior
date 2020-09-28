# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:30:25 2020

@author: svc_ccg
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sync_dataset import Dataset as sync_dataset
import probeSync
import os, glob
import json

import skvideo
skvideo.setFFmpegPath(r'C:\Users\svc_ccg\Documents\ffmpeg\bin')
import skvideo.io

annotated_video_dirs = [
r"\\10.128.50.20\sd7\habituation\1050235366_530862_20200914",
r"\\10.128.50.20\sd7\habituation\1050264010_524926_20200914",
r"\\10.128.50.20\sd7\habituation\1051845969_532246_20200921",
r"\\10.128.50.20\sd7\habituation\1051845969_532246_20200921",
r"\\10.128.50.20\sd7\habituation\1052096186_533537_20200922"
]

annotation_category_dict = {
                'no label': 0,
                'tongue': 1,
                'paw': 2,
                'groom': 3,
                'air lick': 4,
                'chin': 5,
                'air groom': 6,
                'no contact': 7,
                'tongue out': 8,
                'ambiguous': 9,
                'all labels': ''}

def find_annotation_frames(annotations, label): 
    
    labels = annotations['lickStates']
    return np.where(labels==annotation_category_dict[label])[0]


def extract_lost_frames_from_json(cam_json):
    
    lost_count = cam_json['RecordingReport']['FramesLostCount']
    if lost_count == 0:
        return []
    
    lost_string = cam_json['RecordingReport']['LostFrames'][0]
    lost_spans = lost_string.split(',')
    
    lost_frames = []
    for span in lost_spans:
        
        start_end = span.split('-')
        if len(start_end)==1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1])+1))
    
    return lost_frames
    
    
def get_frame_exposure_times(sync, cam_json):
    
    exposure_sync_line_label_dict = {
            'Eye': 'eye_cam_exposing',
            'Face': 'face_cam_exposing',
            'Behavior': 'beh_cam_exposing'}
    
    cam_label =  cam_json['RecordingReport']['CameraLabel']
    sync_line = exposure_sync_line_label_dict[cam_label]
    
    exposure_times = sync.get_rising_edges(sync_line, units='seconds')
    
    lost_frames = extract_lost_frames_from_json(cam_json)
    
    frame_times = [e for ie, e in enumerate(exposure_times) if ie not in lost_frames]
    
    return frame_times
    

side_save_path = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation\side.mp4" 
side_out = skvideo.io.FFmpegWriter(save_path, inputdict={
          '-r': r'60/1',
        },
        outputdict={
          '-vcodec': 'libx264',
          '-pix_fmt': 'yuv420p',
          '-r': r'60/1',
          '-crf': '17'
    })


face_save_path = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation\face.mp4" 
face_out = skvideo.io.FFmpegWriter(face_path, inputdict={
          '-r': r'60/1',
        },
        outputdict={
          '-vcodec': 'libx264',
          '-pix_fmt': 'yuv420p',
          '-r': r'60/1',
          '-crf': '17'
    })

for vd in annotated_video_dirs:
    
    sync_annotation_file = glob.glob(os.path.join(vd, '*annotationssync*'))[0]
    snapshot_annotation_file = glob.glob(os.path.join(vd, '*annotationssnapshot*'))
    continuous_annotation_file = glob.glob(os.path.join(vd, '*annotationsconsecutive*'))
    beh_annotation_file = snapshot_annotation_file if len(snapshot_annotation_file)>0 else continuous_annotation_file
    
    sync_annotations = np.load(sync_annotation_file)
    beh_annotations = np.load(beh_annotation_file[0])
    
    sync_file = glob.glob(os.path.join(vd, '*.sync'))[0]
    syncDataset = sync_dataset(sync_file)
    
    side_view_file = glob.glob(os.path.join(vd, '*behavior.mp4'))[0]  
    face_view_file = glob.glob(os.path.join(vd, '*face.mp4'))[0]
    
    side_view_json_file = glob.glob(os.path.join(vd, '*behavior.json'))[0]  
    face_view_json_file = glob.glob(os.path.join(vd, '*face.json'))[0]
    
    with open(side_view_json_file, 'r') as f:
        side_json = json.load(f)
        
    with open(face_view_json_file, 'r') as f:
        face_json = json.load(f)

#    side_video = cv2.VideoCapture(side_view_file)
#    face_video = cv2.VideoCapture(face_view_file)
    
    width  = side_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = side_video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    
    tongue_frame_inds = find_annotation_frames(sync_annotations, 'tongue')
    
    
    
    vid_in = skvideo.io.FFmpegReader(side_view_file)
    data = skvideo.io.ffprobe(face_view_file)['video']
    rate = data['@r_frame_rate']
    T = np.int(data['@nb_frames'])

    for idx, frame in enumerate(vid_in.nextFrame()):
        if idx in tongue_frame_inds:
            vid_out.writeFrame(frame)
        if idx>50000:
            break
         
vid_out.close()