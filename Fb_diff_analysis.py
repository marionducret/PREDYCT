#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:06:46 2023

@author: Marion Ducret

Event-related potentials analysis based on feedback value : positive, negative or difference
Peak detection based on the higher/lower amplitude of a specific peak window defined with the difference signal

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
#matplotlib.use('Agg') #n'affiche pas les plots
matplotlib.use('Qt5Agg') #affiche les plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# statistics
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu, kruskal
import statsmodels.stats.multitest as smm
from statsmodels.nonparametric.smoothers_lowess import lowess


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

def detect_peak_max(data, distance=5, width=5):
    height = data.mean()+data.std()
    peaks,  properties = find_peaks(data, distance=distance, width=width, height=height)
    if len(peaks) == 0:
        return None
    amplitudes = properties["peak_heights"]
    max_idx = np.argmax(amplitudes)
    return peaks[max_idx]

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

curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_diff_ERP_pb.csv')
    
curr_signal = curr_signal.merge(clusters2, on='pb')
curr_signal = curr_signal.merge(ch_loca, on='ch')
           
# %% processing

"""compute plots et statistics"""

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
                threshold = mean['diff'].std()#*1.5
                mean['large_change'] = mean['diff'].abs() > threshold
                #extract first and last large_change == 'True' and their equivalent time index
                filtered_mean = mean[mean.index >= lat_th]
                first_idx = filtered_mean[filtered_mean['large_change']].index[0] if filtered_mean['large_change'].any() else None
                last_idx = filtered_mean[filtered_mean['large_change']].index[-1] if filtered_mean['large_change'].any() else None
                peak_periods2[area] = {'start': first_idx, 'end': last_idx}
        peak_periods[hem] = peak_periods2     
    return peak_periods   

# def demean_area(signal):
#     print('Re referencing execution...')
#     signal['clean_area'] = signal['area'].str.extract(r'([A-Z]+)')  # extract "LPFC" or "MCC"
#     signal['ap'] = signal['area'].str.extract(r'_([0-9]+)$')
#     mean_pe = signal.groupby(['hem', 'clean_area'])['PE'].transform('mean')
#     signal['mean_PE'] = mean_pe
#     signal['PE_reref'] = signal['PE'] - signal['mean_PE']
#     return signal

def chunk_diff_evol(signal, peak_periods):
    print('Difference evolution execution...')
    signal = signal.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
            
    with PdfPages(date_fold+f'/Diff_evol_{monk}_{task}.pdf') as pdf:
        for hem in signal['hem'].unique():
            for area in signal['area'].unique():
                subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
                if not subset.empty:
                    period = peak_periods[hem][area]
                    group = subset[(subset['time'] >= period['start']) & (subset['time'] <= period['end'])]
                    pivot_table = group.pivot(index='pb', columns='time', values='PE')
                    
                    means = pivot_table.mean(axis=1)
                    sems = pivot_table.std(axis=1) / np.sqrt(pivot_table.shape[1])
                    smoothed = lowess(means, means.index, frac=0.1)

                    fig,ax=plt.subplots(1,1)
                    plt.errorbar(x=means.index, y=means, yerr=sems, fmt='o', capsize=5, ecolor='gray')
                    plt.plot(smoothed[:, 0], smoothed[:, 1], label="LOWESS", linewidth=2)

                    plt.xlabel('pb')
                    plt.ylabel('peak window mean')
                    plt.title(f'{area} {hem}H')
                    pdf.savefig()
                    plt.close()
                    

def chunk_diff_window(signal, peak_periods):
    print('Difference evolution execution...')
    signal = signal.groupby(['hem', 'area', 'pb', 'time'])['PE_reref'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
    with PdfPages(date_fold+f'/Diff_wind_evol_{monk}_{task}.pdf') as pdf:
        for hem in signal['hem'].unique():
            for area in signal['area'].unique():
                subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
                if not subset.empty:
                    period = peak_periods[hem][area]
                    group = subset[(subset['time'] >= period['start']) & (subset['time'] <= period['end'])]
                    peak_avg = {}
                    for p in group['pb'].unique():
                        data = group[(group['pb'] == p)].reset_index()
                        peak = detect_peak_max(-data['PE_reref'])
                        if peak is not None:
                            if peak >= 15:
                                min_peak = peak-15
                            else:
                                min_peak = 0
                            if peak <= len(data)-16:   
                                max_peak = peak+16
                            else:
                                max_peak=len(data)
                                
                            wind = data[min_peak:max_peak]['PE_reref'].values.mean()
                            peak_avg[p] = wind 
                        else:
                            continue
                        
                    df = []
                    for prob, pe in peak_avg.items():
                            df.append({
                                'pb': prob,
                                'PE': pe,
                            })
                    
                    df = pd.DataFrame(df)
                    
                    lowess_smooth = lowess(df['PE'], df['pb'], frac=0.3)  # `frac` contrôle le degré de lissage
                    
                    coeffs = Polynomial.fit(df['pb'], df['PE'], deg=1)  
                    x_lin = np.linspace(df['pb'].min(), df['pb'].max(), 300)  
                    y_lin = coeffs(x_lin)  

                    fig,ax=plt.subplots(1,1)
                    plt.scatter(df['pb'], df['PE'], color='blue')
                    plt.plot(lowess_smooth[:, 0], lowess_smooth[:, 1], color='red', label='LOWESS')
                    plt.plot(x_lin, y_lin, color='green', label='Linear regression')
                    plt.xlabel('pb')
                    plt.ylabel('peak (30ms wind) mean')
                    plt.title(f'{area} {hem}H - negative peak')
                    pdf.savefig()
                    plt.close()     
                    
                    peak_avg = {}
                    for p in group['pb'].unique():
                        data = group[(group['pb'] == p)].reset_index()
                        peak = detect_peak_max(data['PE_reref'])
                        if peak is not None:
                            if peak >= 15:
                                min_peak = peak-15
                            else:
                                min_peak = 0
                            if peak <= len(data)-16:   
                                max_peak = peak+16
                            else:
                                max_peak=len(data)
                                
                            wind = data[min_peak:max_peak]['PE_reref'].values.mean()
                            peak_avg[p] = wind 
                        else:
                            continue
                        
                    df = []
                    for prob, pe in peak_avg.items():
                            df.append({
                                'pb': prob,
                                'PE': pe,
                            })
                    
                    df = pd.DataFrame(df)
                    
                    lowess_smooth = lowess(df['PE'], df['pb'], frac=0.3)  # `frac` contrôle le degré de lissage
                    
                    coeffs = Polynomial.fit(df['pb'], df['PE'], deg=1)  
                    x_lin = np.linspace(df['pb'].min(), df['pb'].max(), 300)  
                    y_lin = coeffs(x_lin)  
                    
                    fig,ax=plt.subplots(1,1)
                    plt.scatter(df['pb'], df['PE'], color='blue')
                    plt.plot(lowess_smooth[:, 0], lowess_smooth[:, 1], color='red', label='LOWESS')
                    plt.plot(x_lin, y_lin, color='green', label='Linear regression')
                    plt.xlabel('pb')
                    plt.ylabel('peak (30ms wind) mean')
                    plt.title(f'{area} {hem}H - positive peaks')
                    pdf.savefig()
                    plt.close()     
                    
                    
                
# %% execution

"""execute specific functions according to the signal of interest"""

peak_periods = chunk_peak_window(curr_signal)
#curr_signal = demean_area(curr_signal)
chunk_diff_evol(curr_signal, peak_periods)
chunk_diff_window(curr_signal, peak_periods)


