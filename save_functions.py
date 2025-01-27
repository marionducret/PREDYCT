#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:33:41 2023

@author: marionducret
"""

#%% Import libraries

import numpy as np
import mne
import os

#%%functions definition 

def save_npz_data(data, savepath, monk, date, task, channel_label):
    signal = np.transpose(data) * 0.195  # Convert to microvolts
    filename = f'{monk}_{date}_{channel_label}_raw.npz'
    np.savez_compressed(os.path.join(savepath, filename), signal=signal)
    
def save_mne_raw(path, savepath, filename, monk, date, FMA, ch_names, sampling_freq, channel_range):
    file = f"{path}{filename}_{FMA}_raw.npz"
    file2 = f"{savepath}{filename}_{FMA}_raw.fif"
    data = np.load(file)
    signal = np.transpose(data['signal']) * 1e-6  # Convert to volts for MNE
    ch_names2 = ch_names[channel_range[0]:channel_range[1]]
    info = mne.create_info(ch_names2, sfreq=sampling_freq, ch_types='ecog')
    raw = mne.io.RawArray(signal, info)
    del signal, data, ch_names2  # Remove transitory data to free some space
    raw.save(file2, fmt='double', overwrite=True)
    del raw
    
def save_mne(data, savepath, filename, monk, date, FMA, ch_names, sampling_freq, channel_range):
    file = f"{savepath}{filename}_{FMA}_raw.fif"
    signal = np.transpose(data) * 1e-6  # Convert to volts for MNE
    ch_names2 = ch_names[channel_range[0]:channel_range[1]]
    info = mne.create_info(ch_names2, sfreq=sampling_freq, ch_types='ecog')
    raw = mne.io.RawArray(signal, info)
    del signal, data, ch_names2  # Remove transitory data to free some space
    raw.save(file, fmt='double', overwrite=True)
    del raw
    
