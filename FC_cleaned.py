#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:50:25 2024

@author: Marion Ducret

This script allows to generate 2 matrix for R Studio : 
1) coherence, phase-locking value and amplitude envelope correlation values with trials/sessions informations
2) granger causaliity and reversed granger causality values with trials/sessions informations
    
Statistical analysis are done with R Studio scripts
"""

# %% libraries

# general
from datetime import datetime

# data processing 
import glob
import mne
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
import pandas as pd
import pickle
from scipy.signal import hilbert, butter, filtfilt


# import homemade load functions
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/load_functions.py").read())

#%% parameters

monk = input('Which monkey? :')
task = input('Which task? :')
code = int(input('Which event? :')) #on which event aligned the epochs
tb = float(input('time before in s? :')) #time before event in epoch
ta = float(input('time after in s? :')) #time after event in epoch

base_path = f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
file_list = glob.glob(base_path+'/'+"*pp.fif")
date_list = np.unique(np.array([x[-16:-10] for x in file_list]))
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
date_list = [d.strftime("%d%m%y") for d in sorted_dates]

if monk == 'Badiane':
    date_list.pop(80)
      
with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

#antero-posterior groups
ch_loca = pd.read_excel("/Users/marionducret/Desktop/PREDYCT/FMA/chnames_loca.xlsx")
ch_loca['index'] = ch_loca.index
grouped_ch_loca = ch_loca.groupby(['ant_post_grp', 'hem', 'area'])['index'].apply(list).to_dict()
grouped_ch_loca_L = {f"{area}_{group}": [index - 128 for index in value] for (group, side, area), value in grouped_ch_loca.items() if side == 'L'}
grouped_ch_loca_R = {f"{area}_{group}": value for (group, side, area), value in grouped_ch_loca.items() if side == 'R'}

file_path = f'/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{monk}_learn_grp_pb.csv'
column_names = ["index", "pb", "group", "error"]
clusters = pd.read_csv(file_path, delimiter=",", names=column_names, skiprows=1)
clusters.rename(columns={'group': 'learn_clus_pb'}, inplace=True)
clusters.drop(clusters.columns[0], axis=1, inplace=True)
clusters2=clusters[['pb','learn_clus_pb']]
clusters2 = clusters2.drop_duplicates()

parameters = [monk, task, code, tb, ta, date_list, clusters2, grouped_ch_loca_R, grouped_ch_loca_L]

f1 = input('Do you want to run COH/PLV/ENVA matrix? :')#yes or no
f2 = input('Do you want to run GC matrix? :')#yes or no

#%% functions

def symmetrize_from_lower(matrix):
    #symmetrize data for connectivity heatmaps
    lower_triangle = np.tril(matrix)
    symmetrical_matrix = lower_triangle + lower_triangle.T - np.diag(np.diag(lower_triangle))
    return symmetrical_matrix

def bandpass_filter(data, low_freq, high_freq, fs, order=4):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

#%% processing

def COH_PLV_ENVA_matrix(parameters):
    
    freq_bands = [(2,6),(7,17),(18,35),(35,48)]
    sfreq = 1000
    
    #controlateral hemisphere
    if monk == 'Israel':
        names = ['A1','A2','B1','B2']
    else:
        names = ['C1','C2','D1','D2']
    
    methods = ['imcoh','plv']
    
    COH, PLV, ENVA = [], [], []
    
    for y, date in enumerate(date_list):
        pp = mne.io.read_raw_fif(base_path + f'{monk}_{date}_{names[0]}_pp.fif', preload=True)
        for suffix in names[1:]:
            pp.add_channels([mne.io.read_raw_fif(base_path + f'{monk}_{date}_{suffix}_pp.fif', preload=True)])
        events = load_events2(monk, date, task, pp)
        epochs = mne.Epochs(pp, events, tmin=-tb, tmax=ta, event_id=code, reject=None)
        del pp, events
        data = epochs.get_data()
        
        coh_df, plv_df, enva_df = [], [], []
        
        for fmin,fmax in freq_bands:
                
            fq = f'{fmin}-{fmax}Hz'
    
            con = spectral_connectivity_epochs(
                data, method=methods, sfreq=sfreq,
                fmin=fmin, fmax=fmax, faverage=True, n_jobs=1)
    
            coh_sess=symmetrize_from_lower(np.squeeze(con[0].get_data(output='dense')))
            plv_sess=symmetrize_from_lower(np.squeeze(con[1].get_data(output='dense')))  
            
            filtered_data = bandpass_filter(data, fmin, fmax, sfreq)
            analytic_signal = hilbert(np.mean(filtered_data, axis=0)) 
            amplitude_envelope = np.abs(analytic_signal) 
            n_channels, n_times = amplitude_envelope.shape
            enva_sess = np.empty((n_channels, n_channels))
            for i in range(n_channels):
                for j in range(n_channels):
                    enva_sess[i, j] = np.mean(np.corrcoef(amplitude_envelope[i, :], amplitude_envelope[j, :])[0, 1])
            
            coh_list, enva_list = [], []
            for n, epoch_data in enumerate(data, start=1):
                #coherence
                con = spectral_connectivity_epochs(
                    [epoch_data], method='imcoh', mode='multitaper', sfreq=sfreq,
                    fmin=fmin, fmax=fmax, faverage=True, n_jobs=1)
                coh_list.append(symmetrize_from_lower(np.squeeze(con.get_data(output='dense'))))
               
                #enva
                filtered_data = bandpass_filter(epoch_data, fmin, fmax, sfreq)
                analytic_signal = hilbert(filtered_data) 
                amplitude_envelope = np.abs(analytic_signal) 
                n_channels, n_times = amplitude_envelope.shape
                corr_matrix = np.empty((n_channels, n_channels))
                for i in range(n_channels):
                    for j in range(n_channels):
                        corr_matrix[i, j] = np.mean(np.corrcoef(amplitude_envelope[i, :], amplitude_envelope[j, :])[0, 1])
                enva_list.append(corr_matrix)
                
                print(f'{fq}: {n}/{data.shape[0]}')
                            
            coh_temp = {'fq':fq, 'date': date,'sess': coh_sess, 'trials': coh_list}
            enva_temp = {'fq':fq, 'date': date,'sess': enva_sess, 'trials': enva_list}
            plv_temp = {'fq':fq, 'date': date,'sess': plv_sess}
       
            coh_df.append(coh_temp)
            plv_df.append(plv_temp)
            enva_df.append(enva_temp)
            
            print(f'{y+1}/{len(date_list)}')
        
        COH.append(coh_df)
        PLV.append(plv_df)
        ENVA.append(enva_df)
        
        print(f'{y+1}/{len(date_list)}')
    
        #save
        with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_COH_matrix.pkl', 'wb') as file:
              pickle.dump(COH, file)
        with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_PLV_matrix.pkl', 'wb') as file:
              pickle.dump(PLV, file) 
        with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_ENVA_matrix.pkl', 'wb') as file:
          pickle.dump(ENVA, file)    
    
    #create AP group
    coh_f, plv_f, enva_f = {}, {}, {}
    
    for d in range(len(date_list)):
        print(d)
        coh_fq, plv_fq, enva_fq = [], [], []
        
        for f, fq in enumerate(freq_bands):
            fq = f'{fq[0]}-{fq[1]}'
    
            coh_gp, enva_gp = [], []
            coh_df, plv_df, enva_df = COH[d][f], PLV[d][f], ENVA[d][f]
            #sess
            coh_data, plv_data, enva_data = coh_df['sess'], plv_df['sess'], enva_df['sess']
            coh_gp_sess = {f"{FMA1}-{FMA2}": np.mean(coh_data[idx, :][:, idx2], axis=(0, 1)) for FMA1, idx in grouped_ch_loca_R.items() for FMA2, idx2 in grouped_ch_loca_R.items() if FMA1 != FMA2}
            plv_gp_sess = {f"{FMA1}-{FMA2}": np.mean(plv_data[idx, :][:, idx2], axis=(0, 1)) for FMA1, idx in grouped_ch_loca_R.items() for FMA2, idx2 in grouped_ch_loca_R.items() if FMA1 != FMA2}
            enva_gp_sess = {f"{FMA1}-{FMA2}": np.mean(enva_data[idx, :][:, idx2], axis=(0, 1)) for FMA1, idx in grouped_ch_loca_R.items() for FMA2, idx2 in grouped_ch_loca_R.items() if FMA1 != FMA2}
            #trials
            for t in range(len(coh_df['trials'])):
        
                coh_data, enva_data = coh_df['trials'][t], enva_df['trials'][t]
                coh_group = {f"{FMA1}-{FMA2}": np.mean(coh_data[idx, :][:, idx2], axis=(0, 1)) for FMA1, idx in grouped_ch_loca_R.items() for FMA2, idx2 in grouped_ch_loca_R.items() if FMA1 != FMA2}
                enva_group = {f"{FMA1}-{FMA2}": np.mean(enva_data[idx, :][:, idx2], axis=(0, 1)) for FMA1, idx in grouped_ch_loca_R.items() for FMA2, idx2 in grouped_ch_loca_R.items() if FMA1 != FMA2}
                coh_gp.append(coh_group)
                enva_gp.append(enva_group)
                
            coh_temp = {'fq': fq, 'sess': coh_gp_sess, 'trials': coh_gp, 'date': coh_df['date']}
            enva_temp = {'fq': fq, 'sess': enva_gp_sess, 'trials': enva_gp, 'date': enva_df['date']}
            plv_temp = {'fq': fq, 'sess': plv_gp_sess, 'date': plv_df['date']}
            
            coh_fq.append(coh_temp)
            plv_fq.append(plv_temp)
            enva_fq.append(enva_temp)  
        
        coh_sess={f"{d}": coh_fq} 
        plv_sess={f"{d}": plv_fq} 
        enva_sess={f"{d}": enva_fq}
    
        coh_f.update(coh_sess)
        plv_f.update(plv_sess)
        enva_f.update(enva_sess)
    
    #create big df with all informations per trial
    temp = []    
    for sess, fq_list in coh_f.items():
        print(sess)
        for fq, sess_df in enumerate(fq_list):
            print(fq)
            for trial, pair_dict in enumerate(sess_df['trials']):
                for pair in pair_dict:
                    coh = coh_f[sess][fq]['trials'][trial].get(pair)
                    enva = enva_f[sess][fq]['trials'][trial].get(pair)
                    
                    tr_temp = {
                        'fq': freq_bands[fq],
                        'sess': int(sess)+1,
                        'trial': int(trial)+1,
                        'pairs': pair,
                        'coh_tr': coh,
                        'enva_tr': enva
                    }
                    
                    coh_sess = coh_f[sess][fq]['sess'].get(pair)
                    plv_sess = plv_f[sess][fq]['sess'].get(pair)
                    enva_sess = enva_f[sess][fq]['sess'].get(pair)
                    date = coh_f[sess][fq]['date']
    
                    tr_temp.update({'date': date,'coh_sess': coh_sess, 'plv_sess': plv_sess, 'enva_sess': enva_sess})          
                    temp.append(tr_temp)
    
    connectivity_df = pd.DataFrame(temp)
    
    savepath = f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_connectivity_df.csv'
    connectivity_df.to_csv(savepath, index=False)


def GC_matrix(parameters):
    fmin=2
    fmax=48
    
    if monk == 'Israel':
        ant_post=(['A2','B1'],['A1','B2'])
    
    else:
        ant_post=(['D1','C2'],['D2','C1'])
    
    methods = ['gc','gc_tr']
    
    signals_lpfc = np.arange(0,32)
    signals_mcc = np.arange(32,64)
    
    indices_lm=(np.array([signals_lpfc]), np.array([signals_mcc]))
    indices_ml=(np.array([signals_mcc]), np.array([signals_lpfc]))
          
    GC = []
    
    for y, date in enumerate(date_list[7:]):
        
        gc_df = {}
        
        for lvl, name in zip(ant_post, ['ant','post']):
        
            pp = mne.io.read_raw_fif(base_path + f'{monk}_{date}_{lvl[0]}_pp.fif', preload=True)
            pp.add_channels([mne.io.read_raw_fif(base_path + f'{monk}_{date}_{lvl[1]}_pp.fif', preload=True)])
            events = load_events2(monk, date, task, pp)
            epochs = mne.Epochs(pp, events, tmin=-tb, tmax=ta, event_id=code, reject=None)
            del pp, events
                    
            # compute Granger causality
            gc_lm = spectral_connectivity_epochs(
                epochs,
                method=methods,
                indices=indices_lm,
                fmin=fmin,
                fmax=fmax,
                #rank=(np.array([5]), np.array([5])),
                gc_n_lags=20) 
            
            gc_ml = spectral_connectivity_epochs(
                epochs,
                method=methods,
                indices=indices_ml,
                fmin=fmin,
                fmax=fmax,
                #rank=(np.array([5]), np.array([5])),
                gc_n_lags=20)
        
            gc_df.update({f'{name}': {'date': date, 'freqs': gc_lm[0].freqs,'gc_lm': np.squeeze(gc_lm[0].get_data()), 'gc_tr_lm':  np.squeeze(gc_lm[1].get_data()),
                       'gc_ml':  np.squeeze(gc_ml[0].get_data()), 'gc_tr_ml':  np.squeeze(gc_ml[1].get_data())}})
    
        GC.append(gc_df)
        print(f'{y+1}/{len(date_list)}')
    
    #save
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_GC_matrix.pkl', 'wb') as file:
          pickle.dump(GC, file)
      
    #load
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_GC_matrix.pkl', 'rb') as file:
         GC = pickle.load(file)
    
    #create big df with all informations per trial
    granger_df = pd.DataFrame()    
    for sess, sess_dict in enumerate(GC):
        print(sess)
        for ap in sess_dict.keys():
            print(ap)
            
            gc_temp = pd.DataFrame(sess_dict[ap])
            gc_temp['ap']=ap
            gc_temp['session']=int(sess)+1
            
            granger_df = pd.concat([granger_df, gc_temp])
            
    savepath = f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_{code}_granger_df.csv'
    granger_df.to_csv(savepath, index=False)
    
# %% execution

"""execute specific functions according to the signal of interest"""

if f1 == "yes":
    COH_PLV_ENVA_matrix(parameters)
if f2 == "yes":
    GC_matrix(parameters)
