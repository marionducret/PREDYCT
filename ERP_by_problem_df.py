#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:06:46 2023

@author: marionducret
"""

# %% libraries & variables

# general
from datetime import datetime
import re
import subprocess
import traceback

# data processing
from collections import defaultdict
import glob
import mne
import numpy as np
import pandas as pd
import pickle

# import homemade load functions
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/load_functions.py").read())

with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

#group ch
ch_loca = pd.read_excel("/Users/marionducret/Desktop/PREDYCT/FMA/chnames_loca.xlsx")
ch_loca['index'] = ch_loca.index
grouped_ch_loca = ch_loca.groupby(['ant_post_grp', 'hem', 'area'])['index'].apply(list).to_dict()
grouped_ch_loca_L = {f"{area}_{group}": [index for index in value] for (group, side, area), value in grouped_ch_loca.items() if side == 'L'}
grouped_ch_loca_R = {f"{area}_{group}": value for (group, side, area), value in grouped_ch_loca.items() if side == 'R'}

group_ch = [grouped_ch_loca_R, grouped_ch_loca_L]

# %% parameters

monk = input('which monkey ? ')
task = input('which task ? ')
tb = float(input('time before fb in sec ? ')) #between 0 and 1
ta = float(input('time after fb in sec ? ')) #between 0 and 1

event_dict = {'trial start': 100, 'trial end': 101, 'correct': 65, 'incorrect': 64, 'target A touch': 70, 'target B touch': 72,
              'target A hold': 74, 'target B hold': 76, 'lever touch': 40, 'lever hold': 42, 'lever release': 43, 'abort': 52, 'nogo': 50}

diff_ex = input('\n\nexecute difference ERP ? ') #yes or no
if diff_ex == 'yes':
    print("\nAvailable events and their codes: \n")
    for event, code in event_dict.items():
        print(f"{event}: {code}")
    event1 = int(input('\nfirst event (code) ? '))
    event2 = int(input('second event (code) ? '))
    diff_name = input('diff name for saving ? ')
    
ERP_ex = input('\n\nexecute event ERP ? ') #yes or no
if ERP_ex == 'yes':
    print("\nAvailable events and their codes: \n")
    for event, code in event_dict.items():
        print(f"{event}: {code}")
    code = int(input('\nwhich code ? '))
    event_name = input('event name for saving ? ')

CTO_ex = input('\n\nexecute CTO event ERP ? ') #yes or no

#run perf R script 
# script_perf_R_path = f"/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{task}task_perf.R"
# command = ["Rscript", script_perf_R_path, monk]
# subprocess.call(command)

#load pb dataframe
pb = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{monk}_pb_{task}.csv')
pb['date'] = ['0'+str(pb['date'][i]) if len(str(pb['date'][i])) == 5 else str(pb['date'][i]) for i in range(len(pb['date']))]
del pb[pb.columns[0]]

#generate date list
path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
file_list = glob.glob(path+'/'+"*pp.fif")
date_list = np.unique(np.array([x[-16:-10] for x in file_list]))
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
date_list = [d.strftime("%d%m%y") for d in sorted_dates]

# %% processing

#difference ERP 

def difference_ERP(monk, task, date_list, event1, event2, diff_name, pb, tb, ta):
    print('diff ERP execution...')
    data_temp={}
   
    for npb in pb['pb']:
        data_pb = []
        data_pb2 = []
        ch = []
        try :
            if npb == 1:
                pbm = pb[pb['pb'] == npb].reset_index()
                del pbm[pbm.columns[0]]
                last_date = pbm.at[0,'date']
                
                idx = date_list.index(last_date)+1
                dates_filt = date_list[:idx]
                
                for date in dates_filt:
                    events, pp = load_pp_and_events2(monk, date, task)
                    
                    if date == last_date:
                        t = events[events[:,2]==99][-1,0]/1000
                        
                        idx_temp = np.where(events[:, 2] == 99)[0][-1]
                        
                        temp = pp.copy().crop(tmax=t)
                        events_temp = events[:idx_temp]
                        
                        for i in chnames:
                                            
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                            b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
        
                            data_pb.append(a)
                            data_pb2.append(b)
                            ch.append(i)
                        
                    else:
                        for i in chnames:
                            
                            a = mne.Epochs(pp, events, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                            b = mne.Epochs(pp, events, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                            
                            data_pb.append(a)
                            data_pb2.append(b)
                            ch.append(i)
               
                channel_indices = defaultdict(list)
                for idx, channel_name in enumerate(ch):
                    channel_indices[channel_name].append(idx)   
                    
                evoked_by_channel = {}
                for channel_name, indices in channel_indices.items():
                    epochs_to_concat = [data_pb[i] for i in indices]
                    concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                    corr = concat_epochs.average()
                    
                    epochs_to_concat = [data_pb2[i] for i in indices]
                    concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                    inc = concat_epochs.average()
                    
                    data = np.squeeze(mne.combine_evoked([corr, inc], weights=[1, -1]).get_data())
    
                    evoked_by_channel[channel_name] = data
    
                data_temp[f'{npb}'] = evoked_by_channel
                print(f'\n problem {npb}/120 \n')
        
            else:
                pbm = pb[(pb['pb'] == npb) | (pb['pb'] == npb-1)].reset_index()
                del pbm[pbm.columns[0]]
                first_date = pbm.at[0,'date']
                last_date = pbm.at[1,'date']
                
                idx = date_list.index(first_date)
                idx2 = date_list.index(last_date)+1
                dates_filt = date_list[idx:idx2]
                
                for date in dates_filt:
                    events, pp = load_pp_and_events2(monk, date, task)
                    
                    if len(dates_filt)>1:
                    
                        if date == first_date:
                            occ=pbm['occur'][0]
                            t = events[events[:,2]==99][occ-1,0]/1000
                            
                            idx_temp = np.where(events[:, 2] == 99)[0][occ-1]
                            if idx_temp == 0:
                                idx_temp = np.where(events[:, 2] == 99)[0][occ]
        
                            temp = pp.copy().crop(tmin=t)
                            events_temp = events[idx_temp:]
                            
                            for i in chnames:
                                                
                                a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                                try:
                                    b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                                    data_pb2.append(b)
                                except:
                                    continue
                                    
                                data_pb.append(a)
                                ch.append(i)
                        
                        elif date == last_date: 
                            occ=pbm['occur'][1]
                            t = events[events[:,2]==99][occ-1,0]/1000
                            
                            idx_temp = np.where(events[:, 2] == 99)[0][occ-1]
                            if idx_temp == 0:
                                idx_temp = np.where(events[:, 2] == 99)[0][occ]
                            
                            temp = pp.copy().crop(tmax=t)
                            events_temp = events[:idx_temp]
                            
                            try:
                            
                                for i in chnames:
                                                    
                                    a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                                    try:
                                        b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                                        data_pb2.append(b)
                                    except:
                                        continue
                
                                    data_pb.append(a)
                                    ch.append(i)
                                    
                            except ValueError as e:
                                
                                match = re.match(r"tmax \(([\d.]+)\) must be less than or equal to the max time \(([\d.]+) sec\)", str(e))

                                if match:
                                    print(f"Error : {e}")
                                    adjusted_tmax = min(1, pp.times[-1])
                                
                                for i in chnames:
                                                    
                                    a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=adjusted_tmax, event_id=event1, picks=i)
                                    try:
                                        b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=adjusted_tmax, event_id=event2, picks=i)
                                        data_pb2.append(b)
                                    except:
                                        continue
                
                                    data_pb.append(a)
                                    ch.append(i)
                            
                        else:
                            try:
                            
                                for i in chnames:
                                                    
                                    a = mne.Epochs(pp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                                    try:
                                        b = mne.Epochs(pp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                                        data_pb2.append(b)
                                    except:
                                        continue
                
                                    data_pb.append(a)
                                    ch.append(i)
                                    
                            except ValueError as e:
                                
                                match = re.match(r"tmax \(([\d.]+)\) must be less than or equal to the max time \(([\d.]+) sec\)", str(e))

                                if match:
                                    print(f"Error : {e}")
                                    adjusted_tmax = min(1, pp.times[-1])
                                
                                for i in chnames:
                                                    
                                    a = mne.Epochs(pp, events_temp, tmin=-tb, tmax=adjusted_tmax, event_id=event1, picks=i)
                                    try:
                                        b = mne.Epochs(pp, events_temp, tmin=-tb, tmax=adjusted_tmax, event_id=event2, picks=i)
                                        data_pb2.append(b)
                                    except:
                                        continue
                
                                    data_pb.append(a)
                                    ch.append(i)
               
                    else:
                        occ=pbm['occur'][0]
                        t = events[events[:,2]==99][occ-1,0]/1000
                        
                        idx_temp = np.where(events[:, 2] == 99)[0][occ-1]
                        if idx_temp == 0:
                            idx_temp = np.where(events[:, 2] == 99)[0][occ]
    
                        temp = pp.copy().crop(tmin=t)
                        events_temp = events[idx_temp:]
                        
                        for i in chnames:
                                            
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                            try:
                                b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                                data_pb2.append(b)
                            except:
                                continue
                                
                            data_pb.append(a)
                            ch.append(i)
                        
                        occ=pbm['occur'][1]
                        t = events[events[:,2]==99][occ-1,0]/1000
                        
                        idx_temp = np.where(events[:, 2] == 99)[0][occ-1]
                        if idx_temp == 0:
                            idx_temp = np.where(events[:, 2] == 99)[0][occ]
                        
                        temp = pp.copy().crop(tmax=t)
                        events_temp = events[:idx_temp]
                        
                        for i in chnames:
                                            
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event1, picks=i)
                            try:
                                b = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=event2, picks=i)
                                data_pb2.append(b)
                            except:
                                continue
        
                            data_pb.append(a)
                            ch.append(i)
               
                channel_indices = defaultdict(list)
                for idx, channel_name in enumerate(ch):
                    channel_indices[channel_name].append(idx)   
                    
                evoked_by_channel = {}
                for channel_name, indices in channel_indices.items():
                    try:
                        epochs_to_concat = [data_pb[i] for i in indices]
                        concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                        corr = concat_epochs.average()
                        
                        epochs_to_concat = [data_pb2[i] for i in indices]
                        concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                        inc = concat_epochs.average()
                        
                        data = np.squeeze(mne.combine_evoked([corr, inc], weights=[1, -1]).get_data())
        
                        evoked_by_channel[channel_name] = data
                    except Exception as e:
                        print(f"Une erreur s'est produite : {e}") 
                        print(npb)
                        print(date)
                        print(f'first:{first_date}')
                        print(f'last:{last_date}')
                        break
    
                data_temp[f'{npb}'] = evoked_by_channel
                print(f'\n problem {npb}/120 \n')
        except Exception as e:
            print(f"Une erreur s'est produite : {e}") 
            print(npb)
            print(date)
            print(f'first:{first_date}')
            print(f'last:{last_date}') 
            break
            
    time = np.array(range(-(int(tb*1000)), (int(ta*1000))+1))
    
    rows = []
    
    for prob, channels in data_temp.items():
        for ch, values in channels.items():
            for t, value in zip(time, values):
                rows.append({
                    'pb': prob,
                    'ch': ch,
                    'time': t,
                    'PE': value,
                })
    
    df = pd.DataFrame(rows)

    #SAVE DATA CSV
    df.to_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_diff_{diff_name}_ERP_pb.csv', index=False)

def event_ERP(monk, task, date_list, code, event_name, pb, tb, ta):
    print('event ERP execution...')
    
    data_temp={}
   
    for npb in pb['pb']:
        data_pb = []
        ch = []
        
        if npb == 1:
            pbm = pb[pb['pb'] == npb].reset_index()
            del pbm[pbm.columns[0]]
            last_date = pbm.at[0,'date']
            
            idx = date_list.index(last_date)+1
            dates_filt = date_list[:idx]
            
            for date in dates_filt:
                events, pp = load_pp_and_events2(monk, date, task)
                
                if date == last_date:
                    t = events[events[:,2]==99][-1,0]/1000
                    
                    idx_temp = np.where(events[:, 2] == 99)[0][-1]
                    
                    temp = pp.copy().crop(tmax=t)
                    events_temp = events[:idx_temp]
                    
                    for i in chnames:
                                        
                        try:
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=code, picks=i)
                            data_pb.append(a)
                            ch.append(i)
                        except:
                            continue
    
                        data_pb.append(a)
                        ch.append(i)
                    
                else:
                    for i in chnames:
                        
                        try:
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=code, picks=i)
                            data_pb.append(a)
                            ch.append(i)
                        except:
                            continue
                        
                        data_pb.append(a)
                        ch.append(i)
           
            channel_indices = defaultdict(list)
            for idx, channel_name in enumerate(ch):
                channel_indices[channel_name].append(idx)   
                
            evoked_by_channel = {}
            for channel_name, indices in channel_indices.items():
                epochs_to_concat = [data_pb[i] for i in indices]
                concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                data = concat_epochs.average().get_data()

                evoked_by_channel[channel_name] = data

            data_temp[f'{npb}'] = evoked_by_channel
            print(f'\n problem {npb}/120 \n')
    
        else:
            pbm = pb[(pb['pb'] == npb) | (pb['pb'] == npb-1)].reset_index()
            del pbm[pbm.columns[0]]
            first_date = pbm.at[0,'date']
            last_date = pbm.at[1,'date']
            
            idx = date_list.index(first_date)
            idx2 = date_list.index(last_date)+1
            dates_filt = date_list[idx:idx2]
            
            for date in dates_filt:
                events, pp = load_pp_and_events2(monk, date, task)
                
                if date == first_date:
                    occ=pbm['occur'][0]
                    t = events[events[:,2]==99][occ-1,0]/1000
                    
                    idx_temp = np.where(events[:, 2] == 99)[0][occ]

                    temp = pp.copy().crop(tmin=t)
                    events_temp = events[idx_temp:]
                    
                    for i in chnames:
                                        
                        try:
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=code, picks=i)
                            data_pb.append(a)
                            ch.append(i)
                        except:
                            continue
                
                elif date == last_date: 
                    occ=pbm['occur'][1]
                    t = events[events[:,2]==99][occ-1,0]/1000
                    
                    idx_temp = np.where(events[:, 2] == 99)[0][occ]
                    
                    temp = pp.copy().crop(tmax=t)
                    events_temp = events[:idx_temp]
                    
                    for i in chnames:
                                        
                        try:
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=code, picks=i)
                            data_pb.append(a)
                            ch.append(i)
                        except:
                            continue

                else:
                    for i in chnames:
                        
                        try:
                            a = mne.Epochs(temp, events_temp, tmin=-tb, tmax=ta, event_id=code, picks=i)
                            data_pb.append(a)
                            ch.append(i)
                        except:
                            continue
                        
           
            channel_indices = defaultdict(list)
            for idx, channel_name in enumerate(ch):
                channel_indices[channel_name].append(idx)   
                
            evoked_by_channel = {}
            for channel_name, indices in channel_indices.items():
                epochs_to_concat = [data_pb[i] for i in indices]
                concat_epochs = mne.concatenate_epochs(epochs_to_concat)
                data = concat_epochs.average().get_data()

                evoked_by_channel[channel_name] = data

            data_temp[f'{npb}'] = evoked_by_channel
            print(f'\n problem {npb}/120 \n')

    time = np.array(range(-(int(tb*1000)), (int(ta*1000))+1))
    
    rows = []
    
    for prob, channels in data_temp.items():
        for ch, values in channels.items():
            for t, value in zip(time, values):
                rows.append({
                    'pb': prob,
                    'ch': ch,
                    'time': t,
                    'PE': value,
                })
    
    df = pd.DataFrame(rows)
    
    #SAVE DATA CSV
    df.to_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{event_name}_ERP_pb.csv', index=False)


def CTO_event_ERP(monk, task, date_list, code, event_name, pb, tb, ta):
    print('CTO ERP execution...')

    path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/CTO/PP/'
    file_list = glob.glob(path+'/'+"*pp.fif")
    
    date_list = np.unique(np.array([x[-16:-10] for x in file_list]))
    FORMAT = '%d%m%y'
    sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
    date_list = [d.strftime("%d%m%y") for d in sorted_dates]
    
    list_data = []
    
    for y, date in enumerate(date_list):
        data_pb = []
        ch = []
        sess = []
        data_temp = {}
    
        events, pp = load_pp_and_events2(monk, date, task)
        
        for i in chnames:
            data = np.squeeze(mne.Epochs(pp, events, tmin=-.2, tmax=1, event_id=65, picks=i, reject=None).get_data())
            data_pb.append(data)
            ch.append(i)
            sess.append(y)
            
            print(f'\n {date}: {i} DONE \n')
            data_temp = {'PE': data_pb, 'ch': ch, 'session': sess, 'date': date}
                
            list_of_dict = [{key: value} for key, value in data_temp.items()]
    
        list_data.append(list_of_dict)
        print(f'\n {date} DONE \n')
        
    data_meanCORR = pd.DataFrame({'PE':pd.Series(dtype=float), 'ch':pd.Series(dtype=str), 'session':pd.Series(dtype=int), 'time':pd.Series(dtype=float)})
    time = np.array(range(-(tb*1000), (ta*1000)+1))
    
    for d in range(len(list_data)): 
        ch_value = list_data[d][1]['ch']
        sess_value = list_data[d][2]['session']
        value = list_data[d][0]['PE']
        data_meanCORR2 = pd.DataFrame({'PE':pd.Series(dtype=float), 'ch':pd.Series(dtype=str), 'session':pd.Series(dtype=int), 'time':pd.Series(dtype=float)})
    
        for i in range(len(value)):
        # Mettre à jour les accumulations
            temp = pd.DataFrame({'PE':pd.Series(dtype=float), 'ch':pd.Series(dtype=str), 'session':pd.Series(dtype=int), 'time':pd.Series(dtype=float)})
            temp['PE'] = np.mean(value[i],axis=0)
            temp['time'] = time
            temp['ch'] = str(ch_value[i])
            temp['session'] = sess_value[i]
            data_meanCORR2 = pd.concat([data_meanCORR2, temp], ignore_index=True)
        
        data_meanCORR = pd.concat([data_meanCORR, data_meanCORR2])
    
    #SAVE DATA CSV 
    data_meanCORR.to_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_CTO_{event_name}_ERP.csv', index=False)
            
# %% execution

"""execute specific functions"""

if diff_ex == "yes":
   difference_ERP(monk, task, date_list, event1, event2, diff_name, pb, tb, ta)
if ERP_ex == "yes":
   event_ERP(monk, task, date_list, code, event_name, pb, tb, ta)
if CTO_ex == "yes":
   CTO_event_ERP(monk, task, date_list, code, event_name, pb, tb, ta)

