#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:06:46 2023

@author: Marion Ducret

Event-related potentials analysis based on feedback value : positive, negative or difference
Peak detection based on the higher/lower amplitude of a specific peak window defined with the difference signal

Folders indicated in the script are those present on MacStudio de Marion

"""

# %% Israe

"""load all libraries necessary and variables&lists with electrodes informations"""

# general
from datetime import datetime
import os

# data processing
import numpy as np
import pandas as pd
import pickle

# plot
import matplotlib
matplotlib.use('Agg') #n'affiche pas les plots
#matplotlib.use('Qt5Agg') #affiche les plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# statistics
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import statsmodels.api as sm

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
#ch_loca.drop(ch_loca.columns[[1,2,6,7,8,9]],axis=1, inplace=True)
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

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

##PEAK DETECTION##

def detect_peaks(data, peak_type, distance=5, width=5):
    height=data.mean()+data.std()
    if peak_type == 'pos':
        peaks, _ = find_peaks(data, distance=distance, width=width, height=height)
    if peak_type == 'neg':
        peaks, _ = find_peaks(-data, distance=distance, width=width, height=height)
    return peaks
 
def refine_peaks_with_local_min(signal, peaks, search_window=20):

    start = max(0, peaks - search_window)
    local_min = np.min(signal[start:peaks]) if peaks > 0 else signal[peaks]
    # Hauteur relative du pic
    relative_height = signal[peaks] - local_min

    return peaks, relative_height

def refine_peaks_with_window(signal, peaks, window_size=20):

    start = max(0, peaks - window_size)
    end = min(len(signal), peaks + window_size + 1)  # +1 car slicing fin non inclus
    local_segment = signal[start:end]
    # Trouver le max local dans cette fenêtre
    local_max_idx = np.argmax(local_segment)
    refined_p = start + local_max_idx
    refined_h = signal[refined_p]
    
    return refined_p, refined_h

def refine_peaks_gaussian_fit(signal, peaks, fit_window=10):
  
    start = max(0, peaks - fit_window)
    end = min(len(signal), peaks + fit_window + 1)
    x_data = np.arange(start, end)
    y_data = signal[start:end]
    
    A_init = np.max(y_data)
    x0_init = peaks
    sigma_init = fit_window / 2.0
    
    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=[A_init, x0_init, sigma_init])
        A_fit, x0_fit, _ = popt
    except RuntimeError:
        # Si l'ajustement échoue, on conserve l'indice du pic initial
        x0_fit = peaks
        A_fit = signal[peaks]

    return x0_fit, A_fit

#peak_pos = 0 pur 1er pic et -1 pour dernier pic
def analyze_peaks(data, peak_type, peak_pos):
    peaks = detect_peaks(data, peak_type)

    if len(peaks) >= 1:
        peaks = peaks[peak_pos]        
    return peaks

def summarize_peak_data(data, lat_th, peak_type, peak_pos):
    data = np.array(data['PE'])
    peaks = analyze_peaks(data, peak_type, peak_pos)
    
    if peaks:
        print('peak found')
        lat_min, amp_min = refine_peaks_with_local_min(data, peaks, search_window=20)
        lat_wind, amp_wind = refine_peaks_with_window(data, peaks,  window_size=5)
        lat_gauss, amp_gauss = refine_peaks_gaussian_fit(data, peaks, fit_window=20)
         
        peak_info = {
            'min': {'amp': amp_min, 'lat': lat_min+lat_th, 'idx': lat_min},
            'wind': {'amp': amp_wind, 'lat': lat_wind+lat_th, 'idx': lat_wind},
            'gauss': {'amp': amp_gauss, 'lat': round(lat_gauss)+lat_th, 'idx': round(lat_gauss)}
        }
    else: 
        print('no peak')
        peak_info = {}
    
    return peak_info
   
# %% parameters

"""define the specific parameters of the analysis : time window, peak of interest, saving directory, signal type (positive fb, negative fb, difference),
peak latency threshold"""

tb = int(input('time before fb in ms ? : ')) #between 0 and 1000
ta = int(input('time after fb in ms ? : ')) #between 0 and 1000

lat_th = int(input('latency threshold in ms ? : ')) #between 0 and 1000

#create a folder with today date 
save_dir = '/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/ERP_analysis/'
date = datetime.now().strftime("%d%m%y") #date of the day
date_fold = os.path.join(save_dir, date)
os.makedirs(date_fold, exist_ok=True)
print(f"All files will be save in : {date_fold}")

signal_of_interest = input('which signal (positive/negative/difference) ? : ') #positive or negative or difference

if signal_of_interest == 'positive':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_reward_ERP_pb.csv')
elif signal_of_interest == 'negative':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_inc_ERP_pb.csv')
elif signal_of_interest == 'difference':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_diff_ERP_pb.csv')
    
curr_signal = curr_signal.merge(clusters2, on='pb')
curr_signal = curr_signal.merge(ch_loca, on='ch')
           
# %% processing

"""compute plots et statistics"""

#re referencing (demean)

def demean_area(signal):
    print('Re referencing execution...')
    signal['clean_area'] = signal['area'].str.extract(r'([A-Z]+)')  # extract "LPFC" or "MCC"
    signal['ap'] = signal['area'].str.extract(r'_([0-9]+)$')
    mean_pe = signal.groupby(['hem', 'clean_area'])['PE'].transform('mean')
    signal['mean_PE'] = mean_pe
    signal['PE_reref'] = signal['PE'] - signal['mean_PE']
    return signal
 
#define peaks period

def chunk_peak_window(signal):
    print('Peak window execution...')
    signal = signal.groupby(['hem', 'clean_area', 'pb', 'time'])['PE'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
    
    peak_periods = {}
    for hem in signal['hem'].unique():
        peak_periods2={}
        for area in signal['clean_area'].unique():
            subset = signal[(signal['hem'] == hem) & (signal['clean_area'] == area)]
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

def chunk_statistic_dataframe(signal, peak_periods, peak_type, peak_pos, lat_th):
    print('Statistic df execution...')
    temp = signal.groupby(['hem', 'clean_area', 'med_lat', 'ap', 'pb', 'time', 'learn_clus_pb'])['PE'].mean().reset_index()
    grouped = temp.groupby(['hem', 'clean_area', 'med_lat', 'ap', 'pb', 'learn_clus_pb'])

    results = []
    
    for name, group in grouped:      
        hem = name[0]
        area = name[1]
           
        period = peak_periods[hem][area]
        group = group[(group['time'] >= period['start']) & (group['time'] <= period['end'])]
        summary = summarize_peak_data(group, lat_th, peak_type, peak_pos)
        #break
            
 
        if not summary :
            results.append({
                'hem': name[0],
                'area': name[1],
                'ml':name[2],
                'ap':name[3],
                'pb': name[4],
                'clus': name[5],
                'lat_gauss': [],
                'amp_gauss': [],
                'lat_min': [],
                'amp_min': [],
                'lat_wind': [],
                'amp_wind': []})
        else:
            results.append({ 
                'hem': name[0],
                'area': name[1],
                'ml':name[2],
                'ap':name[3],
                'pb': name[4],
                'clus': name[5],
                'lat_gauss': summary['gauss']['lat'],
                'amp_gauss': summary['gauss']['amp'],
                'lat_min': summary['min']['lat'],
                'amp_min': summary['min']['amp'],
                'lat_wind': summary['wind']['lat'],
                'amp_wind': summary['wind']['amp']})
    
    stat_df = pd.DataFrame(results).dropna()
        
    return stat_df

##Statistics

def chunk_model_hm(stat_df, var, method, pdf): 
    print('Model heatmap execution...')
    
    colors = [
    (0, "mediumblue"),    
    (0.5, "white"),  
    (1, "red")       
    ]
    custom_coolwarm = LinearSegmentedColormap.from_list("CustomCoolwarm", colors)
    
    grouped = stat_df.groupby(['hem', 'area'])
    
    heatmap_data = {}
    
    for (hem, area), group in grouped:
        results = []
        ml_ap_groups = group.groupby(['ml', 'ap'])
        
        for (ml, ap), sub_group in ml_ap_groups:
     
            # Fit linear model
            y = pd.to_numeric(sub_group[f'amp_{method}'])
            valid_index = y.dropna().index  
            
            X = sm.add_constant(sub_group.loc[valid_index, var])  
            y = y.loc[valid_index]  
            
            model = sm.OLS(y, X).fit()

            slope = model.params[var]
            p_value = model.pvalues[var]
            
            if p_value < 0.05:  
                significance = slope
            else:
                significance = 0 #not significant
            
            results.append((ml, ap, significance))
        
        results_df = pd.DataFrame(results, columns=['ml', 'ap', 'significance'])
        heatmap_df = results_df.pivot(index='ml', columns='ap', values='significance').fillna(0)
        heatmap_data[(hem, area)] = heatmap_df
        
    # Plot heatmaps
    for (hem, area), heatmap_df in heatmap_data.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_df,
            annot=False,
            fmt=".2f",
            cmap=custom_coolwarm,
            center=0,
            cbar_kws={'label': 'Significance (slope)'}
        )
        plt.title(f"{area} {hem}H - {method} method")
        plt.xlabel("antero-posterior")
        plt.ylabel("medio-lateral")
        pdf.savefig()
        plt.close()

def peak_analysis(curr_signal, peak_type, peak_pos, lat_th):
    
    curr_signal = demean_area(curr_signal)
    peak_periods = chunk_peak_window(curr_signal)
    stat_df = chunk_statistic_dataframe(curr_signal, peak_periods, peak_type, peak_pos, lat_th)
    methods = ['gauss','wind','min']
    
    if peak_pos == -1:
        p ='last'
    if peak_pos == 0:
        p ='first'
    
    with PdfPages(date_fold+ f'/{monk}_{task}_{signal_of_interest}_{p}_{peak_type}_model.pdf') as pdf:
        for method in methods:
            chunk_model_hm(stat_df, 'clus', method, pdf)
    
    with PdfPages(date_fold+ f'/{monk}_{task}_{signal_of_interest}_{p}_{peak_type}_model_pb.pdf') as pdf:
        for method in methods:        
            chunk_model_hm(stat_df, 'pb', method, pdf)

# %% execution

"""execute the whole peak analysis:
    - peak_type = negative or positive 
    - peak_pos = 0 for the first one of the window and -1 for the last one
    - three different methods are used = gaussian fit, local minimum, window adjustement
"""

peak_type = input('positive (= pos) or negative (= neg) peak ? : ') 
peak_pos = int(input('first (= 0) or last (= -1) peak? : '))

peak_analysis(curr_signal, peak_type, peak_pos, lat_th)
   
    #change PE by PE_reref