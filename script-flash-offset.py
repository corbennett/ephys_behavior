import time
import datetime
import os
import sys
import json
import pickle
import zipfile
import string
import logging
from collections import OrderedDict

from psychopy import visual
from pyglet.window import key
from qtpy import QtCore
Signal = QtCore.Signal
import numpy as np
import requests

from random import randrange, random

from camstim.behavior import Task, VisualObject, DummyPsycopyStimulus
from camstim.misc import get_config, check_dirs, CAMSTIM_DIR, get_platform_info,\
    save_session, ImageStimNumpyuByte  # TODO:rename these??
from camstim.experiment import EObject, ETimer
from camstim.translator import TrialTranslator
from camstim.change import epoch
from camstim.change import trial
from camstim.change import DoCTask, DoCTrialGenerator, DoCImageStimulus, \
    DoCGratingStimulus
from camstim import Window, Warp
import camstim.automation
import argparse
import yaml
from six import iteritems
import time
from camstim.sweepstim import MovieStim
from camstim.automation import PerfectDoCMouse
import CamstimLaserControl

"""
our code is buggy and wont change if offset == 0 or flash == 0 due 
to the way we're incrementing frames, TODO: fix it
"""

laser = CamstimLaserControl.LaserControl()

# LAYZER CODE GOES HERE
def trigger_layzer():
    laser.triggerLaser()


class DoCTaskWithLayzer(DoCTask):

    def __init__(self, *args, **kwargs):
        self.layzer_params = kwargs.pop('layzer_params')
        super(DoCTaskWithLayzer, self).__init__(*args, **kwargs)

        # layzer
        # self._layzer_frame = None
        self._change_flash = None
        self._layzer_flash = None
        self._layzering = False
        self._layzer_flash_frame_offset = None
        self._layzer_trial = {}
        self.layzer_trials = []

    def _clear_trial_data(self):
        logging.debug('clearing trial data')
        self._event_log = []
        self._trial_licks = []
        self._trial_rewards = []
        self._trial_changes = []
        self._current_trial_data = {}
        self._success = None

        # weird layzer stuff
        # self._layzer_frame = None
        self._change_flash = None
        self._layzer_flash = None
        self._layzering = False
        self._layzer_flash_frame_offset = None
        self._layzer_trial = {}

    def _apply_trial(self, trial):
        """Applies trial parameters.
        """
        change_time = trial.get("change_time", None)
        self._stimulus_window_epoch._change_time = change_time
        self._change_flash = change_time
        if not isinstance(change_time, int):
            raise ValueError(
                'expected to always get int change times, got %s %s' % \
                    (change_time, type(change_time), )
            )
        
        if random() <= self.layzer_params['freq']:
            self._schedule_layzer()

    def _trial_ended(self):
        """called at the end of a trial (regardless of how it ended.)
        """
        logging.debug("Trial ended.")
        self.trial_ended.emit()

        if self._layzer_trial:
        	self.layzer_trials.append(self._layzer_trial)
        self._deschedule_layzer()

        self._log_event("trial_end")
    
    def _setup_default_epochs(self):
        """ Sets up default epoch behavior and timing. See wiki for epoch definitions.
        """
        initial_blank = self._doc_config['initial_blank']
        self._blank_epoch = epoch.DoCNoStimEpoch(
            self,
            duration=initial_blank,
            name="initial_blank",
        )

        pre_change_time = self._doc_config['pre_change_time']
        self._pre_change_epoch = epoch.DoCMinPrechange(
            self,
            duration=pre_change_time,
            name="pre_change",
        )

        rw = self._doc_config['response_window']
        rw_dur = rw[1]-rw[0]
        rw_delay = rw[0]
        self._response_window_epoch = epoch.DoCResponseWindow(
            self,
            duration=rw_dur,
            delay=rw_delay,
            name="response_window",
        )

        sw_dur = self._doc_config['stimulus_window']
        change_dist = self._doc_config['change_time_dist']
        if change_dist == "geometric":
            # flash-based change timing
            self._stimulus_window_epoch = CrazyDoCStimulusWindowFlashed(
                self,
                duration=sw_dur,
                name="stimulus_window",
            )
        else:
            # time-based 
            raise NotImplementedError(
                'time based stimulus windows arent supported by this script.'
            )

        nl_dur = self._doc_config['min_no_lick_time']
        self._no_lick_epoch = epoch.DoCMinNoLickEpoch(
            self,
            duration=nl_dur,
            name="no_lick",
        )

        timeout_dur = self._doc_config['timeout_duration']
        self._timeout_epoch = epoch.DoCTimeoutEpoch(
            self,
            duration=timeout_dur,
            name="timeout",
        )

        for e in  [self._blank_epoch,
                    self._pre_change_epoch,
                    self._stimulus_window_epoch,
                    self._response_window_epoch,
                    self._no_lick_epoch,
                    self._timeout_epoch
                    ]:
            self.add_epoch(e)

        # epocs that automatically transition to each other
        self._blank_epoch.epoch_ended.connect(self._pre_change_epoch.enter)
        self._pre_change_epoch.epoch_ended.connect(self._stimulus_window_epoch.enter)
        self._stimulus_window_epoch.epoch_ended.connect(self._no_lick_epoch.enter)

        if self._doc_config['end_after_response']:
            #end stimulus window after response window instead of letting it time itself
            self._stimulus_window_epoch.exit_after_epoch(self._response_window_epoch,
                                                            delay=self._doc_config['end_after_response_sec'])

        self.set_starting_epoch(self._blank_epoch)

    def _schedule_layzer(self):
        # this is relative to start of DoCStimulusWindowFlashed so we can just go off of flashes....i think
        self._layzer_flash = self._change_flash + randrange(*self.layzer_params['offset_range_flash'])
        self._layzer_frame_offset = randrange(*self.layzer_params['offset_range_frame'])

        self._layzer_trial['trial'] = self.trial_count
        self._layzer_trial['expected_layzer_flash'] = self._layzer_flash
        self._layzer_trial['expected_layzer_frame_offset'] = self._layzer_frame_offset
        self._layzer_trial['expected_change_flash'] = self._change_flash

        logger.debug(
            'Scheduling layzer. trial: {}, expected layzer flash: {}, expected layzer frame offset: {}, expected change flash: {}'
                .format(
                    self.trial_count,
                    self._layzer_flash,
                    self._layzer_frame_offset, 
                    self._change_flash,
                )
        )

    def _deschedule_layzer(self):
        self._layzering = False
        logger.debug(
            'Descheduled layzer. trial: {}, frame: {}'
                .format(
                    self.trial_count,
                    self.update_count,
                )
        )
    
    def update(self, index=None):
        if self._layzering and self._layzer_frame_offset is not None:
            if self._layzer_frame_offset == 0:
                trigger_layzer()
                self._layzer_trial['actual_layzer_frame'] = self.update_count
                self._layzer_frame_offset = None
                logger.debug(
                    'Triggered layzer. trial: {}, frame: {}' \
                        .format(
                            self.trial_count,
                            self.update_count,
                        )
                )
            else:
                self._layzer_frame_offset -= 1

        return super(DoCTaskWithLayzer, self).update(index=index)


class CrazyDoCStimulusWindowFlashed(epoch.DoCStimulusWindowFlashed):
    
    # def _on_entry(self):
    #     super(CrazyDoCStimulusWindowFlashed, self)._on_entry()
    #     self._flash_counter = 0
    #     if not self._connected:
    #         self._task._doc_stimulus.flash_started.connect(self._handle_flash)
    #         self._connected = True
        
    #     logger.debug(
    #         'Entering stimulus flash window. Resetting stim frame count, from: {}'
    #             .format(self._task._stim_frame_count)
    #     )
    #     self._task._stim_frame_count = 0  # reset our counter

    # def _on_exit(self):
    #  	ret = super(CrazyDoCStimulusWindowFlashed, self)._on_exit()
    #  	self._task_stim_frame_count = 0
    #  	logger.debug(
    #  		'Exiting stimulus flash window. Restting stim frame count, from: {}'
    #  			.format(self._task._stim_frame_count)
    #  	)

    #  	return ret
    
    def _handle_flash(self):
        if self._active:
            self._flash_counter += 1
            if self._flash_counter == (self._change_flash - 1):
                self._disable_flash_omit()
            elif self._flash_counter == (self._change_flash + 1):
                self._enable_flash_omit()
            elif self._flash_counter == self._change_flash:
                self._disable_flash_omit()
                self._trigger_change()
                self._task._layzer_trial['actual_change_flash'] = self._flash_counter
                logger.debug(
                    'changing on flash: {}'.format(self._flash_counter)
                )
            else:
                pass

            if self._flash_counter == self._task._layzer_flash:
                self._task._layzering = True
                self._task._layzer_trial['actual_layzer_flash'] = self._flash_counter
                logger.debug(
                    'layzering on at flash: {}'.format(self._flash_counter)
                )



class CrazyDoCImageStimulus(DoCImageStimulus):
    
    def __init__(self, *args, **kwargs):
        self._task = kwargs.pop('task')
        super(CrazyDoCImageStimulus, self).__init__(*args, **kwargs)

    def _on_change(self):
        if self._scheduled_on_change:
            while self._scheduled_on_change:
                call = self._scheduled_on_change.pop()
                call()

        if self._task._layzer_trial:
            self._task._layzer_trial['actual_change_frame'] = self._task.update_count
            logger.debug('Changed! frame: {}'.format(self._task.update_count))


class CrazyDoCGratingStimulus(DoCGratingStimulus):

    def __init__(self, *args, **kwargs):
        self._task = kwargs.pop('task')
        super(CrazyDoCGratingStimulus, self).__init__(*args, **kwargs)
    
    def _on_change(self):
        if self._scheduled_on_change:
            while self._scheduled_on_change:
                call = self._scheduled_on_change.pop()
                call()

        if self._task._layzer_trial:
            self._task._layzer_trial['actual_change_frame'] = self._task.update_count
            logger.debug('Changed! frame: {}'.format(self._task.update_count))


def load_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    # print args
    with open(args.json_path, 'r') as f:
        # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
        params = yaml.load(f)
    return params


def load_stimulus_class(class_name):
    if class_name == 'grating':
        DoCStimulus = CrazyDoCGratingStimulus
    elif class_name == 'images':
        DoCStimulus = CrazyDoCImageStimulus
    else:
        raise Exception('no idea what Stimulus class to use for `{}`'.format(class_name))

    return DoCStimulus


def set_stimulus_groups(groups, stimulus_object):
    for group_name, group_params in iteritems(groups):
        for param, values in iteritems(group_params):
            stimulus_object.add_stimulus_group(group_name, param, values)


json_params = load_params()

# Configure logging level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger('LayzerDOC')

LAYZER_PARAMS = {
    'freq': 1.0,
    'offset_range_flash': [-2, 2, ],
    'offset_range_frame': [0, 20],
    'fps': 60,
}

# Create handlers
if json_params['dev'] == True:
    log_file = '%s.log' % time.time()
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)
    logger.info('logging to %s' % log_file)

    layzer_params = {}
else:
    layzer_params = json_params.get('layzer_params', {}, )

layzer_params.update(LAYZER_PARAMS)

# offset_lo_bound = layzer_params['offset_range_time'][0] * layzer_params['fps']
# offset_hi_bound = layzer_params['offset_range_time'][1] * layzer_params['fps']

# if offset_lo_bound % 1:
#     raise ValueError(
#         'computed offset range lower bound must be integer number of frames. computed: {}' \
#             .format(offset_lo_bound)
#     )

# if offset_hi_bound % 1:
#     raise ValueError(
#         'computed offset range upper bound must be integer number of frames. computed: {}' \
#             .format(offset_hi_bound)
#     )

# layzer_params['offset_range_frames'] = [offset_lo_bound, offset_hi_bound, ]

# layzer_params['flash_duration_frames'] = \
#     layzer_params['flash_duration_time'] * layzer_params['fps']

# if layzer_params['flash_duration_frames'] % 1:
#     raise ValueError(
#         'flash_duration in seconds must compute to integer number of frames.'
#     )

stimulus = json_params['stimulus']

# Set up display window (This may become unnecessary in future release)
window = Window(
    fullscr=json_params.get('dev', False) is False,  # no fullscr annoyance for dev 
    screen=1,
    monitor='Gamma1.Luminance50',
    warp=Warp.Spherical,
)

# Set up Task
params = {}
f = DoCTaskWithLayzer(
    window=window,
    auto_update=True,
    params=params,
    layzer_params=layzer_params,
)
t = DoCTrialGenerator(cfg=f.params) # This also subject to change
f.set_trial_generator(t)

# Set up our DoC stimulus
DoCStimulus = load_stimulus_class(stimulus['class'])
stimulus_object = DoCStimulus(window, task=f, **stimulus['params'])

if "groups" in stimulus:
    set_stimulus_groups(stimulus['groups'],stimulus_object)

# Add our DoC stimulus to the Task
f.set_stimulus(stimulus_object, stimulus['class'])

start_stop_padding = json_params.get('start_stop_padding', 0.5)

# add prologue to start of session
prologue = json_params.get('prologue', None)
if prologue:

    prologue_movie = MovieStim(
        window=window,
        start_time=0.0,
        stop_time=start_stop_padding,
        flip_v=True,
        **prologue['params']
    )
    f.add_static_stimulus(
        prologue_movie,
        when=0.0,
        name=prologue['name'],
    )

# add fingerprint to end of session
epilogue = json_params.get('epilogue', None)
if epilogue:
    # assert start_stop_padding > 13.0, "`start_stop_padding` must be longer than 13.0s countdown video."

    if prologue:
        assert epilogue['params']['frame_length'] == prologue['params']['frame_length'], "`frame_length` must match between prologue and epilogue"

    epilogue_movie = MovieStim(
        window=window,
        start_time=start_stop_padding,
        stop_time=None,
        flip_v=True,
        **epilogue['params']
    )
    f.add_static_stimulus(
        epilogue_movie,
        when='end',
        name=epilogue['name'],
    )

# add a test mause if dev
if json_params.get('dev'):
    mouse = PerfectDoCMouse()
    mouse.attach(f)

# Run it
f.start()