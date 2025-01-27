#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:23:11 2024

@author: marionducret

This script allows to save specific epochs matrix to save time in further analysis
"""

# %% Import libraries

from datetime import datetime
import glob
import mne
import numpy as np
import pickle

# import homemade load functions
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/load_functions.py").read())

#%% Choose parameters

monk = input('Which monkey? :')
task = input('Which task? :')
event = int(input('Which event? :'))#code number in eventIDE files
event_name = input('Which event name? :')#name for saved file

base_path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
file_list = glob.glob(base_path+'/'+"*pp.fif")
date_list = np.unique(np.array([x[-16:-10] for x in file_list]))
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
date_list = [d.strftime("%d%m%y") for d in sorted_dates]

if monk == 'Badiane':
    date_list.pop(80)
      
# import electrodes name list
with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

#%% Save epochs data (2 files for 2 hemispheres)

names = [['A1', 'A2', 'B1', 'B2'], ['C1', 'C2', 'D1', 'D2']]

Epochs = []

for name_set in names:
    temp = []

    for date in date_list:
        pp = mne.io.read_raw_fif(base_path + f'{monk}_{date}_{name_set[0]}_pp.fif', preload=True)
        for suffix in name_set[1:]:
            pp.add_channels([mne.io.read_raw_fif(base_path + f'{monk}_{date}_{suffix}_pp.fif', preload=True)])
        events = load_events2(monk, date, task, pp)
        epochs = mne.Epochs(pp, events, tmin=-1, tmax=1, event_id=event, reject=None)#adapt times before and after
        del pp, events
        data = np.mean(epochs.get_data(), axis=0)
        del epochs
        temp.append(data)
        del data

    Epochs.append(temp)

with open(f'/Users/marionducret/Desktop/PREDYCT figures matrix/{monk}_{task}_{event_name}_epochs.pkl', 'wb') as file:
      pickle.dump(Epochs, file)

