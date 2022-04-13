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
from numba import njit

import skvideo
skvideo.setFFmpegPath(r'C:\Users\svc_ccg\Documents\ffmpeg\bin')
import skvideo.io

annotated_video_dirs = [
r"\\10.128.50.20\sd7\habituation\1050235366_530862_20200914",
r"\\10.128.50.20\sd7\habituation\1050264010_524926_20200914",
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

@njit
def find_ind_of_nearest(array, values):
    
    i=0
    inds = []
    for v in values:
        foundmatch = False
        while not foundmatch:
            if i==0:
                if abs(array[1]-v) > abs(array[0]-v):
                    inds.append(0)
                    foundmatch=True
                i+=1
            elif i == len(array):
                if abs(array[i]-v) > abs(array[i-1]-v):
                    inds.append(i-1)
                    foundmatch=True
                else:
                    inds.append(i)
                    foundmatch=True
            else:
                while (abs(array[i]-v)>abs(array[i+1]-v)):
                    i+=1
                inds.append(i)
                foundmatch=True
    return inds
            
        

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
    
    return np.array(frame_times)
    
side_save_path = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation\side.mp4" 
side_out = skvideo.io.FFmpegWriter(side_save_path, inputdict={
          '-r': r'60/1',
        },
        outputdict={
          '-vcodec': 'libx264',
          '-pix_fmt': 'yuv420p',
          '-r': r'60/1',
          '-crf': '17'
    })


face_save_path = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation\face.mp4" 
face_out = skvideo.io.FFmpegWriter(face_save_path, inputdict={
          '-r': r'60/1',
        },
        outputdict={
          '-vcodec': 'libx264',
          '-pix_fmt': 'yuv420p',
          '-r': r'60/1',
          '-crf': '17'
    })

frame_labels = []
vid_labels = []
all_side_inds = []
all_face_inds = []
for vd in annotated_video_dirs:
    
    #annotation that's 1-to-1 with sync 'lick' signals
    sync_annotation_file = glob.glob(os.path.join(vd, '*annotationssync*'))[0]
    
    #annotation that's just a block of consecutive frames. Deal with two naming conventions
    snapshot_annotation_file = glob.glob(os.path.join(vd, '*annotationssnapshot*'))
    continuous_annotation_file = glob.glob(os.path.join(vd, '*annotationsconsecutive*'))
    beh_annotation_file = snapshot_annotation_file if len(snapshot_annotation_file)>0 else continuous_annotation_file
    
    #load annotations
    sync_annotations = np.load(sync_annotation_file)
    beh_annotations = np.load(beh_annotation_file[0])
    
    #get sync data
    sync_file = glob.glob(os.path.join(vd, '*.sync'))[0]
    syncDataset = sync_dataset(sync_file)
    
    #video file paths
    side_view_file = glob.glob(os.path.join(vd, '*behavior.mp4'))[0]  
    face_view_file = glob.glob(os.path.join(vd, '*face.mp4'))[0]
    
    #video report file paths
    side_view_json_file = glob.glob(os.path.join(vd, '*behavior.json'))[0]  
    face_view_json_file = glob.glob(os.path.join(vd, '*face.json'))[0]
    
   #get video report data to find dropped frames
    with open(side_view_json_file, 'r') as f:
        side_json = json.load(f)
        
    with open(face_view_json_file, 'r') as f:
        face_json = json.load(f)


    #get frame exposure times for camera frames
    side_frame_times = get_frame_exposure_times(syncDataset, side_json)
    face_frame_times = get_frame_exposure_times(syncDataset, face_json)
    
    #frames annotated as 'lick' on the side view
    sync_labels_of_interest = ['tongue', 'paw', 'groom', 'chin']
    side_inds_of_interest = []
    temp_labels = []
    for label in sync_labels_of_interest:
        label_inds = find_annotation_frames(sync_annotations, label)
        if len(label_inds)>0:
            side_inds_of_interest.extend(label_inds)
            temp_labels.extend([label]*len(label_inds))
            print('Adding {} frames of label {}'.format(len(label_inds), label))
            
    
    beh_labels_of_interest = ['air lick', 'air groom', 'tongue out']
    for label in beh_labels_of_interest:
        label_inds = find_annotation_frames(beh_annotations, label)
        if len(label_inds)>0:
            side_inds_of_interest.extend(label_inds)
            temp_labels.extend([label]*len(label_inds))
            
            print('Adding {} frames of label {}'.format(len(label_inds), label))
    
    temp_labels = np.array(temp_labels)
    ind_order = np.argsort(side_inds_of_interest)
    temp_labels = temp_labels[ind_order]
    side_inds_of_interest= np.array(side_inds_of_interest)[ind_order]
    
    vid_labels.extend([vd]*len(side_inds_of_interest))
    frame_labels.extend(temp_labels)
    
    side_frame_times_of_interest = side_frame_times[side_inds_of_interest]
    face_inds_of_interest = find_ind_of_nearest(face_frame_times, side_frame_times_of_interest)
    
    all_side_inds.extend(side_inds_of_interest)
    all_face_inds.extend(face_inds_of_interest)
    
    #Write composite video for side view
    vid_in = skvideo.io.FFmpegReader(side_view_file)
    data = skvideo.io.ffprobe(side_view_file)['video']
    rate = data['@r_frame_rate']
    T = np.int(data['@nb_frames'])

    for idx, frame in enumerate(vid_in.nextFrame()):
        if idx in side_inds_of_interest:
            side_out.writeFrame(frame)
#        if idx>50000:
#            break
    
    #Write composite video for face view
    vid_in = skvideo.io.FFmpegReader(face_view_file)
    data = skvideo.io.ffprobe(face_view_file)['video']
    rate = data['@r_frame_rate']
    T = np.int(data['@nb_frames'])

    for idx, frame in enumerate(vid_in.nextFrame()):
        if idx in face_inds_of_interest:
            face_out.writeFrame(frame)
#        if idx>50000:
#            break

side_out.close()  
face_out.close()

save_base = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation"
np.save(os.path.join(save_base, 'frame_labels'), frame_labels)
np.save(os.path.join(save_base, 'source_videos'), vid_labels)
np.save(os.path.join(save_base, 'side_video_frame_inds'), all_side_inds)
np.save(os.path.join(save_base, 'face_video_frame_inds'), all_face_inds)





v = cv2.VideoCapture(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\eyeTrackingForPeter\frames_of_interest_for_annotation\side.mp4")

label_frames = [ig for ig,g in enumerate(frame_labels) if g=='air lick']
for g in label_frames[-10:]:
    plt.figure()
    v.set(cv2.CAP_PROP_POS_FRAMES, g)
    r, gf = v.read()
    plt.imshow(gf)
    



















