#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:33:26 2023

@author: marionducret
"""

#%% Import libraries

import numpy as np
import posixpath
import glob
import scipy.io
import mne

#%%functions definition 

def load_raw_files(monk, task, date):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/RAW/'
    raw = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_raw.fif', preload=True)
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        raw.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_raw.fif', preload=True)])
    return raw

def load_pp_files(monk, task, date):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    pp = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_pp.fif', preload=True)
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        pp.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_pp.fif', preload=True)])
    return pp

def load_2pp_files(monk, task, date, suffix, suffix2):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    pp = mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_pp.fif', preload=True)
    pp.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix2}_pp.fif', preload=True)])
    return pp

def load_pp2_files(monk, task, date):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    pp2 = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_pp2.fif', preload=True)
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        pp2.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_pp2.fif', preload=True)])
    return pp2

def load_events(monk, date, task):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    eventpath = posixpath.join('/Volumes/PROCYK/PREDYCT', monk, 'Ephys', task, 'behav/')
    try:
        filename = monk + '_' + date + "_codes.mat"
        event = scipy.io.loadmat(eventpath + filename)
    except :
        filename = monk + '_' + date + "_codes.npy"
        event = np.load(eventpath + filename)
    a = np.around(event['CODE'][:, 0])
    a_diff = np.diff(a)
    idx_diff = np.where(a_diff == 0)[0]
    idx_diff += 1
    a[idx_diff] += 1
    b = np.zeros(len(a))
    c = event['CODE'][:, 1]
    raw = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_pp.fif', preload=True)
    times = raw[0][1]
    times = np.asarray(np.around(times * 1000))  # in ms
    d = np.transpose(np.array(np.where(np.isin(times, a) == True)))  # find sample index corresponding to time
    events = (np.rint(np.column_stack([d, b, c]))).astype(int)
    return events

#when one pp file is already loaded
def load_events2(monk, date, task, raw):
    eventpath = posixpath.join('/Volumes/PROCYK/PREDYCT', monk, 'Ephys', task, 'behav/')
    filename = monk + '_' + date + "_codes.npy"
    event = np.load(eventpath + filename)
    b = np.zeros(len(event[:,1]))
    events = np.column_stack([event[:,0]*1000, b, event[:,1]]).astype(int)
    return events


def load_eye_data(folder):
    file_list = glob.glob(folder + '/*ANALOG-IN*.dat')
    eye_data = [np.fromfile(file, dtype='uint16')[::30] for file in file_list]
    return np.array(eye_data)

def load_raw_crop(path, monk,date) :
    filename = monk+'_'+date
    file = path+filename
    raw_file = file + '_A1_raw.fif'
    raw = mne.io.read_raw_fif(raw_file, preload=False)  # automatically load split parts
    raw.crop(tmin=600, tmax=1200).load_data()  # keep only 10 minutes of signal and load data
    suffix_name = ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']
    for suffix in suffix_name:
        raw_part = mne.io.read_raw_fif(file+f'_{suffix}_raw.fif', preload=False)
        raw_part.crop(tmin=0, tmax=600).load_data()  # keep only 10 minutes of signal and load data
        raw.add_channels([raw_part])
    return raw

def load_mua_and_events(monk, date, task):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/MUA/'
    eventpath = posixpath.join('/Volumes/PROCYK/PREDYCT', monk, 'Ephys', task, 'behav/')
    mua = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_MUA.fif', preload=True)
    # try:
    #     filename = monk + '_' + date + "_codes.mat"
    #     event = scipy.io.loadmat(eventpath + filename)
    # except :
    filename = monk + '_' + date + "_codes.npy"
    event = np.load(eventpath + filename)
    a = np.zeros((len(event),1))
    events = np.column_stack([event[:,0], a, event[:,1]])
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        mua.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_MUA.fif', preload=True)])
    return events, mua

def load_pp_and_events(monk, date, task):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    eventpath = posixpath.join('/Volumes/PROCYK/PREDYCT', monk, 'Ephys', task, 'behav/')
    try:
        filename = monk + '_' + date + "_codes.mat"
        event = scipy.io.loadmat(eventpath + filename)
        a = np.around(event['CODE'][:, 0])
        c = event['CODE'][:, 1]
    except :
        filename = monk + '_' + date + "_codes.npy"
        event = np.load(eventpath + filename)
        a = np.around(event[:, 0])
        c = event[:, 1]
    a_diff = np.diff(a)
    idx_diff = np.where(a_diff == 0)[0]
    idx_diff += 1
    a[idx_diff] += 1
    b = np.zeros(len(a))
    pp = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_pp.fif', preload=True)
    times = pp[0][1]
    times = np.asarray(np.around(times * 1000))  # in ms
    d = np.squeeze(np.transpose(np.array(np.where(np.isin(times, a) == True))))  # find sample index corresponding to time
    events = (np.rint(np.column_stack([d, b, c]))).astype(int)
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        pp.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_pp.fif', preload=True)])
    return events, pp, c

def load_pp_and_events2(monk, date, task):
    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
    eventpath = posixpath.join('/Volumes/PROCYK/PREDYCT', monk, 'Ephys', task, 'behav/')
    filename = monk + '_' + date + "_codes.npy"
    event = np.load(eventpath + filename)
    b = np.zeros(len(event[:,1]))
    events = np.column_stack([event[:,0]*1000, b, event[:,1]]).astype(int)
    pp = mne.io.read_raw_fif(path + monk + '_' + date + '_A1_pp.fif', preload=True)
    for suffix in ['A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        pp.add_channels([mne.io.read_raw_fif(path + monk + '_' + date + f'_{suffix}_pp.fif', preload=True)])
    return events, pp
