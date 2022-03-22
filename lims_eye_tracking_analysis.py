# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:47:55 2020

@author: svc_ccg
"""

import allensdk
import allensdk.brain_observatory.behavior
import allensdk.brain_observatory.behavior as sdk
import allensdk.brain_observatory.behavior.eye_tracking_processing as sdk_eye
import h5py
import pandas as pd
from sync_dataset import Dataset as sync_dataset

e = pd.read_hdf(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1023232776\ecephys_session_1050962145\eye_tracking\1050962145_ellipse.h5", key=['cr', 'eye', 'pupil])
e = pd.read_hdf(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1023232776\ecephys_session_1050962145\eye_tracking\1050962145_ellipse.h5", key=['cr', 'eye', 'pupil'])
e = sdk_eye.load_eye_tracking_hdf(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1023232776\ecephys_session_1050962145\eye_tracking\1050962145_ellipse.h5")
e.keys()

s = sync_dataset(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1023232776\ecephys_session_1050962145\1051071880\1050962145_524760_20200916.sync")
fts = s.get_rising_edges('eye_frame_received')
fts.shape
len(e)
s.line_labels()
s.line_labels
len(e.index)
fts.shape
len(e.index) - fts.shape
len(e.index) - fts.shape[0]
e = sdk_eye.load_eye_tracking_hdf(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1035469407\ecephys_session_1055240613\eye_tracking\1055240613_ellipse.h5")
s = sync_dataset(r"\\allen\programs\braintv\production\visualbehavior\prod0\specimen_1035469407\ecephys_session_1055240613\1055324726\1055240613_533539_20201007.sync")
fts = s.get_rising_edges('eye_frame_received')
fts.shape
len(e)
e = e.iloc[1:]
len(e)
fts = pd.Series(fts)
e_p = sdk_eye.process_eye_tracking_data(e, fts)
e_p.keys()
plt.plot(e_p['time'], e_p['pupil_area'])
plt.plot(e_p['time'], e_p['pupil_center_x'])
pa = e_p['pupil_area'].loc[~e_p['likely_blink']]
e_p['likely_blink'].head()
np.sum(e_p['likely_blink'])
len(e_p)
pa = e_p['pupil_area'].loc[!e_p['likely_blink']]
pa = e_p['pupil_area'].loc[e_p['likely_blink']]
plt.plot(e_p['time'], e_p['pupil_area'])
plt.plot(e_p['time'], e_p['likely_blink']*100)
plt.plot(e_p['time'], e_p['likely_blink']*100000)
pa = e_p['pupil_area'].values
pa[e_p['likely_blink']] = np.nan
pa[e_p['likely_blink'].values] = np.nan
pa[e_p['likely_blink'].values.astype('bool')] = np.nan
plt.plot(pa)
pa_mf = scipy.signal.medfilt(pa, 5)
plt.plot(pa_mf)