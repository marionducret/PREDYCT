#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:06:46 2023

@author: Marion Ducret

This script tests different methods for peak detection in ERP

Folders indicated in the script are those present on MacStudio de Marion

"""

# %% library & variables

"""load all libraries necessary and variables&lists with electrodes informations"""

# general
from datetime import datetime
import os

# data processing
import itertools
import numpy as np
import pandas as pd
import pickle

# plot
import matplotlib
matplotlib.use('Agg') #n'affiche pas les plots
#matplotlib.use('Qt5Agg') #affiche les plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# statistics
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu, kruskal
import statsmodels.stats.multitest as smm


# import homemade load functions
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/load_functions.py").read())

# import electrodes name list
with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

monk = input('which monkey ? :') #Israel or Badiane
task = input('which task ? :') #DL or RL 

# create antero-posterior groups 
ch_loca = pd.read_excel("/Users/marionducret/Desktop/PREDYCT/FMA/chnames_loca.xlsx")
ch_loca['index'] = ch_loca.index
grouped_ch_loca = ch_loca.groupby(['ant_post_grp', 'hem', 'area'])['index'].apply(list).to_dict()
grouped_ch_loca_L = {f"{area}_{group}": [index for index in value] for (group, side, area), value in grouped_ch_loca.items() if side == 'L'}
grouped_ch_loca_R = {f"{area}_{group}": value for (group, side, area), value in grouped_ch_loca.items() if side == 'R'}
group_ch = [grouped_ch_loca_R, grouped_ch_loca_L]

ch_loca.rename(columns={'chname': 'ch', 'ant_post_grp': 'ap_gp'}, inplace=True)
ch_loca.drop(ch_loca.columns[[1,2,6]],axis=1, inplace=True)
ch_loca['area'] = ch_loca['area'] + '_' + ch_loca['ap_gp'].astype(str)
ch_loca.drop(columns=['ap_gp'], inplace=True)

# load learning steps
file_path = f'/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{monk}_learn_grp_pb.csv'
column_names = ["index", "pb", "group", "error"]
clusters = pd.read_csv(file_path, delimiter=",", names=column_names, skiprows=1)
clusters.rename(columns={'group': 'learn_clus_pb'}, inplace=True)
clusters.drop(clusters.columns[0], axis=1, inplace=True)
clusters2=clusters[['pb','learn_clus_pb']]
clusters2 = clusters2.drop_duplicates()

# %% functions

"""create specific functions used in next chunks"""

##MATHEMATICS##

def round_down_to_nearest_tenth(value):
    return np.floor(value * 10) / 10

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

##PREPROCESSING DF##

def filter_channels(df, channels):
    return df[df['ch'].isin(channels)]

def artefact_removal(data, group) :
    curr_mean = data['PE'].mean()
    curr_sd = data['PE'].std()
    outliers_pos = curr_mean + 10*curr_sd
    outliers_neg = curr_mean - 10*curr_sd
    outliers = data[(data['PE'] < outliers_neg) | (data['PE'] > outliers_pos)]
    outlier_pb_values = outliers[group].unique()
    non_outliers = data[~data[group].isin(outlier_pb_values)]
    return non_outliers

def add_baseline(data, group):
    baseline_data = data[(data['time'] >= -100) & (data['time'] <= 0)]
    baseline_values = baseline_data.groupby([group])['PE'].mean().reset_index()
    baseline_df = data.merge(baseline_values, on=group, suffixes=('_actual', '_baseline'))
    baseline_df['PE_normalized'] = baseline_df['PE_actual'] - baseline_df['PE_baseline']
    return baseline_df

def classify_pair2(area):
    if 'MCC' in area:
        return 'MCC'
    elif 'LPFC' in area:
        return 'LPFC'
    else:
        return 'Other'

def safe_indexing(series, indices):
      # If indices are empty, return an empty array directly
      if len(indices) == 0:
          return np.array([])
      
      valid_indices = [i for i in indices if 0 <= i < len(series)]
      return np.array(series.iloc[valid_indices])
    
##PEAK DETECTION##

def detect_peaks(data, distance=5, width=5):
    height = data.mean()+data.std()
    peaks, _ = find_peaks(data, distance=distance, width=width, height=height)
   
    return peaks

def detect_peaks_via_derivative(data, min_height):
    dsignal = np.gradient(data)
    peaks = []
    for i in range(1, len(dsignal)):
        if dsignal[i-1] > 0 and dsignal[i] < 0:
            if min_height is None or data[i] > min_height:
                peaks.append(i)
    return np.array(peaks)

def refine_peaks_with_local_min(signal, peaks, search_window=20):
    peak_heights = []
    peak_lat = []
    for p in peaks:
        start = max(0, p - search_window)
        local_min = np.min(signal[start:p]) if p > 0 else signal[p]
        # Hauteur relative du pic
        relative_height = signal[p] - local_min
        peak_heights.append(relative_height)
        peak_lat.append(p)

    return np.array(peak_lat), np.array(peak_heights) 

def refine_peaks_with_window(signal, peaks, window_size=30):
    refined_peaks = []
    refined_heights = []
    for p in peaks:
        start = max(0, p - window_size)
        end = min(len(signal), p + window_size + 1)  # +1 car slicing fin non inclus
        local_segment = signal[start:end]
        # Trouver le max local dans cette fenêtre
        local_max_idx = np.argmax(local_segment)
        refined_p = start + local_max_idx
        refined_h = signal[refined_p]

        refined_peaks.append(refined_p)
        refined_heights.append(refined_h)
    
    return np.array(refined_peaks), np.array(refined_heights)

def refine_peaks_gaussian_fit(signal, peaks, fit_window=10):
    refined_centers = []
    fitted_amplitudes = []

    for p in peaks:
        start = max(0, p - fit_window)
        end = min(len(signal), p + fit_window + 1)
        x_data = np.arange(start, end)
        y_data = signal[start:end]
        
        A_init = np.max(y_data)
        x0_init = p
        sigma_init = fit_window / 2.0
        
        try:
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=[A_init, x0_init, sigma_init])
            A_fit, x0_fit, sigma_fit = popt
            refined_centers.append(x0_fit)
            fitted_amplitudes.append(A_fit)
        except RuntimeError:
            # Si l'ajustement échoue, on conserve l'indice du pic initial
            refined_centers.append(p)
            fitted_amplitudes.append(signal[p])
    
    return np.array(refined_centers), np.array(fitted_amplitudes)

def analyze_peaks(data):
    peaks = detect_peaks(data)

    lat_min, amp_min = refine_peaks_with_local_min(data, peaks, search_window=20)
    lat_wind, amp_wind = refine_peaks_with_window(data, peaks,  window_size=5)
    lat_gauss, amp_gauss = refine_peaks_gaussian_fit(data, peaks, fit_window=10)

    peaks_deriv = detect_peaks_via_derivative(data, data.mean()+data.std())
    
    lat_min_der, amp_min_der = refine_peaks_with_local_min(data.tolist(), peaks_deriv, search_window=20)
    lat_wind_der, amp_wind_der = refine_peaks_with_window(data.tolist(), peaks_deriv,  window_size=5)
    lat_gauss_der, amp_gauss_der = refine_peaks_gaussian_fit(data.tolist(), peaks_deriv, fit_window=10)

    return lat_min, amp_min, lat_wind, amp_wind,lat_gauss, amp_gauss, lat_min_der, amp_min_der, lat_wind_der, amp_wind_der, lat_gauss_der, amp_gauss_der

def summarize_peak_data(data, time_df):
    lat_min, amp_min, lat_wind, amp_wind,lat_gauss, amp_gauss, lat_min_der, amp_min_der, lat_wind_der, amp_wind_der, lat_gauss_der, amp_gauss_der = analyze_peaks(data)
    dsignal = np.gradient(data)
    peak_info = {
        'Normal': {
            'min': {'amp': amp_min, 'lat': safe_indexing(time_df, lat_min), 'idx': lat_min},
            'wind': {'amp': amp_wind, 'lat': safe_indexing(time_df, lat_wind), 'idx': lat_wind},
            'gauss': {'amp': amp_gauss, 'lat': safe_indexing(time_df, lat_gauss), 'idx': lat_gauss}
        },
        'Derivative': {
            'min': {'amp': amp_min_der, 'lat': safe_indexing(time_df, lat_min_der), 'idx': lat_min_der},
            'wind': {'amp': amp_wind_der, 'lat': safe_indexing(time_df, lat_wind_der), 'idx': lat_wind_der},
            'gauss': {'amp': amp_gauss_der, 'lat': safe_indexing(time_df, lat_gauss_der), 'idx': lat_gauss_der}
        },
        'Norm_data': data,
        'Der_data': dsignal
    }
    
    return peak_info

# %% parameters

"""define the specific parameters of the analysis : time window, peak of interest, saving directory, signal type (positive fb, negative fb, difference),
peak latency threshold"""

tb = int(input('time before fb in ms ? :')) #between 0 and 1000
ta = int(input('time after fb in ms ? :')) #between 0 and 1000

lat_th = int(input('latency threshold in ms ? :')) #between 0 and 1000

#create a folder with today date 
save_dir = '/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/ERP_analysis/'
date = datetime.now().strftime("%d%m%y") #date of the day
date_fold = os.path.join(save_dir, date)
os.makedirs(date_fold, exist_ok=True)
print(f"All files will be save in : {date_fold}")

signal_of_interest = input('which signal ? :') #positive or negative or difference

#debug
signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_PEcorr_pb.csv')
curr_signal = signal

if signal_of_interest == 'positive':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_reward_ERP_pb.csv')
    CTO_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_CTO_reward_ERP.csv')
    RL_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_RL_reward_ERP_pb.csv')
elif signal_of_interest == 'negative':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_inc_ERP_pb.csv')
    RL_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_RL_inc_ERP_pb.csv')
elif signal_of_interest == 'difference':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_diff_ERP_pb.csv')
    
curr_signal = curr_signal.merge(clusters2, on='pb')
curr_signal = curr_signal.merge(ch_loca, on='ch')

col = ['lightgreen','green','darkgreen']

# %% processing

"""compute plots et statistics"""

#debug
signal=curr_signal

#define peaks period

def chunk_peak_window(signal):
    print('Peak window execution...')
    signal = signal.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
    
    peak_periods = {}
    for hem in signal['hem'].unique():
        peak_periods2={}
        for area in signal['area'].unique():
            subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
            subset = artefact_removal(subset, 'pb')
            subset = add_baseline(subset, 'pb')
            if not subset.empty:
                pivot_table = subset.pivot(index='pb', columns='time', values='PE_normalized')
                mean = pd.DataFrame(pivot_table.mean(axis=0))
                #diff between consecutives values
                mean['diff'] = mean[0].diff()
                threshold = mean['diff'].std()
                mean['large_change'] = mean['diff'].abs() > threshold
                #extract first and last large_change == 'True' and their equivalent time index
                #filtered_mean = mean[mean.index >= lat_th]
                filtered_mean = mean
                first_idx = filtered_mean[filtered_mean['large_change']].index[0] if filtered_mean['large_change'].any() else None
                last_idx = filtered_mean[filtered_mean['large_change']].index[-1] if filtered_mean['large_change'].any() else None
                peak_periods2[area] = {'start': first_idx, 'end': last_idx}
        peak_periods[hem] = peak_periods2     
    return peak_periods

#peak detection

def chunk_peak_detection(signal, peak_periods, grouped_ch_loca_R, grouped_ch_loca_L):
    print('Peak detection execution...')
    summary = {} 
    if monk == 'Israel':
        peak_period = peak_periods['R']
        hem='RH'
        grp_ch = grouped_ch_loca_R

    if monk == 'Badiane':
        peak_period = peak_periods['L']
        hem='LH'
        grp_ch = grouped_ch_loca_L

    for group_name, channels in grp_ch.items():
        summary[group_name] = {}
        channel_names = [chnames[i] for i in channels]  
        data_temp = filter_channels(signal, channel_names)
        period = peak_period[group_name]
    
        if not data_temp.empty:
    
            data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time'])['PE'].mean().reset_index()
            data_temp_mean = data_temp_mean[(data_temp_mean['time'] >= period['start']) & (data_temp_mean['time'] <= period['end'])]
            
            summary_pos = {}
            summary_neg = {}
            for learn_clus, group in data_temp_mean.groupby('learn_clus_pb'):
                data = group['PE'].values
                time_df = group['time']
                summary_pos[learn_clus] = summarize_peak_data(data, time_df)
                summary_neg[learn_clus] = summarize_peak_data(-data, time_df)

            summary[group_name]['pos'] = summary_pos
            summary[group_name]['neg'] = summary_neg

    return(summary)

def chunk_peak_plot(summary, grouped_ch_loca_R, savepath):
    print('Peak plot execution...')
    with PdfPages(savepath) as pdf:
        for group_name, _ in grouped_ch_loca_R.items():
            for step in [1,2,3]:#learning steps
            
                curr = summary[group_name]['pos'][step]
                
                n_data = curr['Norm_data']
                n_gauss = curr['Normal']['gauss']
                n_wind = curr['Normal']['wind']
                n_min = curr['Normal']['min']
                d_gauss = curr['Derivative']['gauss']
                d_wind = curr['Derivative']['wind']
                d_min = curr['Derivative']['min']
                
                plt.figure()
                plt.plot(n_data, color='black')
                
                plt.scatter(n_gauss['idx'], n_gauss['amp'], color='blue', label='Gaussian (n)')
                plt.vlines(n_gauss['idx'], ymin=min(n_data), ymax=n_gauss['amp'], color='blue', linestyle='dashed', alpha=0.7)
                
                plt.scatter(n_wind['idx'], n_wind['amp'], color='red', label='Window (n)')
                plt.vlines(n_wind['idx'], ymin=min(n_data), ymax=n_wind['amp'], color='red', linestyle='dashed', alpha=0.7)
                
                plt.scatter(n_min['idx'], n_min['amp'], color='green', label='Minima (n)')
                plt.vlines(n_min['idx'], ymin=min(n_data), ymax=n_min['amp'], color='green', linestyle='dashed', alpha=0.7)
                
                plt.scatter(d_gauss['idx'], d_gauss['amp'], color='blue', marker='x', label='Gaussian (d)')
                plt.vlines(d_gauss['idx'], ymin=min(n_data), ymax=d_gauss['amp'], color='blue', linestyle='solid', alpha=0.7)
                
                plt.scatter(d_wind['idx'], d_wind['amp'], color='red', marker='x', label='Window (d)')
                plt.vlines(d_wind['idx'], ymin=min(n_data), ymax=d_wind['amp'], color='red', linestyle='solid', alpha=0.7)
                
                plt.scatter(d_min['idx'], d_min['amp'], color='green', marker='x', label='Minima (d)')
                plt.vlines(d_min['idx'], ymin=min(n_data), ymax=d_min['amp'], color='green', linestyle='solid', alpha=0.7)
    
                plt.legend()
                plt.title(f'Peak detection methods - {group_name} step{step}')
                pdf.savefig()
                plt.close()

# %% execution

"""execute specific functions according to the signal of interest"""

if signal_of_interest == "positive":
    peak_periods = chunk_peak_window(curr_signal)
    summary = chunk_peak_detection(signal, peak_periods, grouped_ch_loca_R, grouped_ch_loca_L)
    chunk_peak_plot(summary, grouped_ch_loca_R, f'/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/ERP_analysis/detection_methods/{monk}_{task}_{signal_of_interest}_detection_methods.pdf')
elif signal_of_interest == "negative":
    peak_periods = chunk_peak_window(curr_signal)
    summary = chunk_peak_detection(signal, peak_periods)

elif signal_of_interest == "difference":
    peak_periods = chunk_peak_window(curr_signal)
    summary = chunk_peak_detection(signal, peak_periods)

else:
    print(f"unrecognized signal : {curr_signal}")
    