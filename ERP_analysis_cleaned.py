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
matplotlib.use('Agg') #n'affiche pas les plots
#matplotlib.use('Qt5Agg') #affiche les plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# statistics
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
ch_loca.drop(ch_loca.columns[[1,2,6,7,8,9]],axis=1, inplace=True)
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


def calculate_latency_differences(df, groups_lpfc, groups_mcc, peak_type):
    # Filtrer les données pour les groupes LPFC et MCC et le type de pic spécifié
    lpfc_df = df[(df['Group'].isin(groups_lpfc)) & (df['Peak Type'] == peak_type)]
    mcc_df = df[(df['Group'].isin(groups_mcc)) & (df['Peak Type'] == peak_type)]
    
    # Calculer la latence moyenne pour chaque cluster dans chaque groupe
    lpfc_grouped = lpfc_df.groupby('Cluster')['Latency'].mean().reset_index(name='Mean Latency (LPFC)')
    mcc_grouped = mcc_df.groupby('Cluster')['Latency'].mean().reset_index(name='Mean Latency (MCC)')
    
    # Fusionner les résultats
    latency_diff = pd.merge(lpfc_grouped, mcc_grouped, on='Cluster')
    latency_diff['Latency Difference (LPFC - MCC)'] = latency_diff['Mean Latency (LPFC)'] - latency_diff['Mean Latency (MCC)']
    
    return latency_diff

def calculate_latency_differences2(df, groups_lpfc, groups_mcc):
    # Filtrer les données pour les groupes LPFC et MCC
    lpfc_df = df[(df['area'].isin(groups_lpfc))]
    mcc_df = df[(df['area'].isin(groups_mcc))]
    
    # Calculer la latence moyenne pour chaque pb dans chaque groupe
    lpfc_grouped = lpfc_df.groupby(['pb','clus'])['latency'].mean().reset_index(name='Mean Latency (LPFC)')
    mcc_grouped = mcc_df.groupby(['pb','clus'])['latency'].mean().reset_index(name='Mean Latency (MCC)')
    
    # Fusionner les résultats
    latency_diff = pd.merge(lpfc_grouped, mcc_grouped, on=['pb','clus'])
    latency_diff['Latency Difference (LPFC - MCC)'] = latency_diff['Mean Latency (LPFC)'] - latency_diff['Mean Latency (MCC)']
    
    return latency_diff

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

##PEAK DETECTION##

def summarize_peak_data(data, group_name, learn_clus, pdf, plot, c, idx_step1):
    pos_peak, neg_peak = analyze_peaks(data, group_name, learn_clus, pdf, plot, c, idx_step1)
    peak_info = {
        'Positive Peak': get_peak_info(data, pos_peak),
        'Negative Peak': get_peak_info(data, neg_peak)
    }
    if c == 'orange' and pos_peak is not None: #step1
        idx_step1 = pos_peak
    
    return peak_info, idx_step1

def get_peak_info(data, peak_indices):
    if peak_indices is None:
        return None, None
    else:
        time_peak = data['time'].iloc[peak_indices]
        amplitude_peak = data['PE'].iloc[peak_indices]
        return time_peak, amplitude_peak

def analyze_peaks(data, group_name, learn_clus, pdf, plot, c, idx_step1):
    #different parameters for peak detection as step 2 and 3 are more noisy
    if c == 'orange':
        peaks_positive, peaks_negative = detect_peaks(data, group_name, learn_clus, pdf, plot, c, 1)
    else:
        peaks_positive, peaks_negative = detect_peaks(data, group_name, learn_clus, pdf, plot, c, .8)

    if c == 'orange': #step1
        if len(peaks_positive) >= 1:
            pos_peak = peaks_positive[0]
            
            if plot=='yes':
                pos_peak = int(pos_peak)
                plt.plot(data['time'], data['PE'], color=c)
                plt.plot(data['time'].iloc[pos_peak], data['PE'].iloc[pos_peak], 'ro', label='pos')
        else:
            pos_peak = None
            
        if len(peaks_negative) >= 1:
            neg_peak = peaks_negative[0]
        else:
            neg_peak = None 
    else: # steps 2&3
        #find nearest value
        if len(peaks_positive) >= 1:
            idx = 0
            diff_min = abs(peaks_positive[0] - idx_step1)
            
            for i in range(1, len(peaks_positive)):
                diff = abs(peaks_positive[i] - idx_step1)
                if diff < diff_min:
                    diff_min = diff
                    idx = i
    
            pos_peak = peaks_positive[idx]
            
            if plot=='yes':
                pos_peak = int(pos_peak)
                plt.plot(data['time'], data['PE'], color=c)
                plt.plot(data['time'].iloc[pos_peak], data['PE'].iloc[pos_peak], 'ro', label='pos')
        else:
            pos_peak = None
            
        if len(peaks_negative) >= 1:
            neg_peak = peaks_negative[0]
        else:
            neg_peak = None 
    
    return pos_peak, neg_peak

def detect_peaks(data, group_name, learn_clus, pdf, plot, c, factor, distance=5, width=5):
    height=data['PE'].mean()+data['PE'].std()*factor
    peaks_positive, _ = find_peaks(data['PE'], distance=distance, width=width, height=height)
    peaks_negative, _ = find_peaks(-data['PE'], distance=distance, width=width, height=height)
    ##show peaks found with find_peaks function to do a visual verification
    # if plot=='yes':
    #     plt.plot(data['time'], data['PE'], color=c)
    #     plt.plot(data['time'].iloc[peaks_positive], data['PE'].iloc[peaks_positive], 'ro', label='pos')
    #     #plt.plot(data['time'].iloc[peaks_negative], data['PE'].iloc[peaks_negative], 'bo', label='neg')
    #     plt.legend()
    return peaks_positive, peaks_negative
    
##STATISTICS##

def test_stat_diff_non_param(data, group_col, time_col, value_col):
    times = data[time_col].unique()
    p_values = []
    
    for time in times:
        groups = data[data[time_col] == time].groupby(group_col)[value_col]
        group_values = [group for name, group in groups]
        
        if len(group_values) > 1:
            try:
                #u_stat, p_val = mannwhitneyu(group_values[0], group_values[1], alternative='two-sided')
                h_stat, p_val = kruskal(*group_values)  # Test de Kruskal-Wallis
                #p_val = round_down_to_nearest_tenth(p_val)
                p_val = smm.multipletests(p_val, alpha=0.5, method='fdr_bh')[1]
                p_values.append((time, p_val))
            except ValueError:
                p_values.append((time, np.nan))  # Cas où les données sont insuffisantes pour le test
        else:
            p_values.append((time, np.nan))
    
    return pd.DataFrame(p_values, columns=[time_col, 'p_value'])

def kruskal_analysis(data, var):
    kruskal_results = {}
    
    for area in data['area'].unique():
        area_df = data[data['area'] == area]
        
        cluster_groups = [area_df[area_df['clus'] == clus][var] for clus in area_df['clus'].unique()]
        
        stat, p_value = kruskal(*cluster_groups)
        
        kruskal_results[area] = {'statistic': stat, 'p-value': p_value, 'significance': significance_label(p_value)}
    
    kruskal_df = pd.DataFrame.from_dict(kruskal_results, orient='index')
    
    return kruskal_df

def post_hoc_MW(data, kk_data, var):
    
    post_hoc_results = []

    for area in kk_data.index:
        if kk_data.loc[area, 'p-value'] < 0.05:
            area_df = data[data['area'] == area]
            clusters = area_df['clus'].unique()
            pairwise_comparisons = list(itertools.combinations(clusters, 2))
            
            for (clus1, clus2) in pairwise_comparisons:
                group1 = area_df[area_df['clus'] == clus1][var]
                group2 = area_df[area_df['clus'] == clus2][var]
                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                
                median1 = group1.median()
                median2 = group2.median()
                direction = 'increase' if median2 > median1 else 'decrease'
                
                post_hoc_results.append({
                    'Area': area,
                    'comparison': f'{clus1} vs {clus2}',
                    'statistic': stat,
                    'p-value': p_value,
                    'significance': significance_label(p_value),
                    'direction': direction
                })
    
        post_hoc_df = pd.DataFrame(post_hoc_results)
        if post_hoc_df.empty:
            post_hoc_status = 'empty'
        else:
            post_hoc_status = 'ok'

    return post_hoc_df, post_hoc_status
    
    return post_hoc_df  

def stat_between_steps(df1, df2, time_column, value_column):
    unique_times = df1[time_column].unique()
    results = []

    for time in unique_times:
        df1_time = df1[df1[time_column] == time][value_column]
        df2_time = df2[df2[time_column] == time][value_column]

        stat, p_value = mannwhitneyu(df1_time, df2_time)

        results.append((time, stat, p_value))
    
    results_df = pd.DataFrame(results, columns=[time_column, 'stat', 'p_value'])
    _, corrected_p_values, _, _ = smm.multipletests(results_df['p_value'], method='fdr_bh')
    results_df['corrected_p_value'] = corrected_p_values

    return results_df

##PLOT FIGURES##

def significance_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns' 

def stat_visualisation(kk_data, post_hoc_df, status, path_pdf):
    with PdfPages (path_pdf) as pdf:
        kk_data.reset_index(inplace=True)
        kk_data.rename(columns={'index': 'Area'}, inplace=True)
        kk_data['p-value'] = kk_data['p-value'].apply(lambda x: f'{x:.2e}')
        kk_data['statistic'] = kk_data['statistic'].apply(lambda x: f'{x:.2e}')
        if status != 'empty':
            post_hoc_df['p-value'] = post_hoc_df['p-value'].apply(lambda x: f'{x:.2e}')
            post_hoc_df['statistic'] = post_hoc_df['statistic'].apply(lambda x: f'{x:.2e}')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('tight')
        ax.axis('off')
        global_table = ax.table(cellText=kk_data.values, colLabels=kk_data.columns, cellLoc='center', loc='center')
        global_table.auto_set_font_size(False)
        global_table.set_fontsize(10)
        global_table.scale(1.2, 1.2)
        for i in range(len(kk_data)):
            if kk_data.loc[i, 'significance'] != 'ns':
                for j in range(len(kk_data.columns)):
                    global_table[(i+1, j)].get_text().set_color('red')
        plt.title("Kruskal-Wallis")
        pdf.savefig()
        plt.close()
        
        if status != 'empty':
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis('tight')
            ax.axis('off')
            posthoc_table = ax.table(cellText=post_hoc_df.values, colLabels=post_hoc_df.columns, cellLoc='center', loc='center')
            posthoc_table.auto_set_font_size(False)
            posthoc_table.set_fontsize(10)
            posthoc_table.scale(1.2, 1.2)
            for i in range(len(post_hoc_df)):
                if post_hoc_df.loc[i, 'significance'] != 'ns':
                    for j in range(len(post_hoc_df.columns)):
                        posthoc_table[(i+1, j)].get_text().set_color('red')
            plt.title("Mann-Whitney U (post-hocs)")
            pdf.savefig()   
            plt.close()
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
           
# %% processing

"""compute plots et statistics"""

def chunk_diff_heatmap(signal):
    print('Difference heatmap execution...')
    signal = signal.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
    
    x_label_list = ['0', '200', '400']
        
    with PdfPages(date_fold+f'/Diff_heatmap_{monk}_{task}.pdf') as pdf:
        for hem in signal['hem'].unique():
            for area in signal['area'].unique():
                subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
                if not subset.empty:
                    pivot_table = subset.pivot(index='pb', columns='time', values='PE')
                    mean = pivot_table.mean().mean() 
                    sd = pivot_table.stack().std()
                    outliers_pos = mean + 3 * sd
                    outliers_neg = mean - 3 * sd
                    artefact_sessions = pivot_table.apply(lambda row: any((row < outliers_neg) | (row > outliers_pos)), axis=1)
                    pivot_table = pivot_table[~artefact_sessions] 
    
                    fig,ax=plt.subplots(1,1)
                    img=ax.imshow(pivot_table, cmap='viridis', aspect='auto')
                    ax.set_xticks([100,300,500])
                    ax.set_xticklabels(x_label_list)
                    fig.colorbar(img)
                    ax.set_title(f'Difference heatmap for {hem} - {area}')
                    ax.set_xlabel('times')
                    ax.set_ylabel('pb')
                    pdf.savefig()
                    plt.close()
                    
    #focus 30 first pb            
    with PdfPages(date_fold+f'/Diff_heatmap_{monk}_{task}_30pb.pdf') as pdf:
        for hem in signal['hem'].unique():
            for area in signal['area'].unique():
                subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
                subset = subset[(subset['pb'] <=30)]
                if not subset.empty:
                    pivot_table = subset.pivot(index='pb', columns='time', values='PE')
                    mean = pivot_table.mean().mean() 
                    sd = pivot_table.stack().std()
                    outliers_pos = mean + 3 * sd
                    outliers_neg = mean - 3 * sd
                    artefact_sessions = pivot_table.apply(lambda row: any((row < outliers_neg) | (row > outliers_pos)), axis=1)
                    pivot_table = pivot_table[~artefact_sessions] 
    
                    fig,ax=plt.subplots(1,1)
                    img=ax.imshow(pivot_table, cmap='viridis', aspect='auto')
                    ax.set_xticks([100,300,500])
                    ax.set_xticklabels(x_label_list)
                    fig.colorbar(img)
                    ax.set_title(f'Difference heatmap for {hem} - {area}')
                    ax.set_xlabel('times')
                    ax.set_ylabel('pb')
                    pdf.savefig()
                    plt.close()                
                
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
                threshold = mean['diff'].std()#*1.5
                mean['large_change'] = mean['diff'].abs() > threshold
                #extract first and last large_change == 'True' and their equivalent time index
                filtered_mean = mean[mean.index >= lat_th]
                first_idx = filtered_mean[filtered_mean['large_change']].index[0] if filtered_mean['large_change'].any() else None
                last_idx = filtered_mean[filtered_mean['large_change']].index[-1] if filtered_mean['large_change'].any() else None
                peak_periods2[area] = {'start': first_idx, 'end': last_idx}
        peak_periods[hem] = peak_periods2     
    return peak_periods

#plot peaks period limits    

def chunk_plot_peak_window(signal, peak_periods):
    print('Plot peak window execution...')
    signal = signal.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    signal = signal[(signal['time'] >= -tb) & (signal['time'] <= ta)]
    #x_label_list = ['0', '200', '400']
        
    with PdfPages(date_fold+f'/{monk}_{task}_{signal_of_interest}_peakwindow.pdf') as pdf:
    
        for hem in signal['hem'].unique():
            for area in signal['area'].unique():
                subset = signal[(signal['hem'] == hem) & (signal['area'] == area)]
                limits = peak_periods[hem][area]
                if not subset.empty:
                    pivot_table = subset.pivot(index='pb', columns='time', values='PE')
                    mean = pivot_table.mean().mean() 
                    sd = pivot_table.stack().std()
                    outliers_pos = mean + 3 * sd
                    outliers_neg = mean - 3 * sd
                    artefact_sessions = pivot_table.apply(lambda row: any((row < outliers_neg) | (row > outliers_pos)), axis=1)
                    pivot_table = pivot_table[~artefact_sessions] 
                    #heatmap
                    # fig,ax=plt.subplots(1,1)
                    # img=ax.imshow(pivot_table, cmap='viridis', aspect='auto')
                    # ax.set_xticks([100,300,500])
                    # ax.set_xticklabels(x_label_list)
                    # fig.colorbar(img)
                    # ax.axvline(x=limits['start']+100, color='red', linestyle='--')#+100 to adapt to xticks starting from -100
                    # ax.axvline(x=limits['end']+100, color='red', linestyle='--')
                    # ax.set_title(f'Difference heatmap for {hem} - {area}')
                    # ax.set_xlabel('times')
                    # ax.set_ylabel('pb')
                    # pdf.savefig()
                    # plt.close()
                    #mean lineplot
                    fig, ax = plt.subplots(1, 1)
                    mean_values = pivot_table.mean(axis=0)  
                    ax.plot(mean_values, color='blue', linestyle='-', label='Average')
                    #ax.set_xticks([100, 300, 500])
                    #ax.set_xticklabels(x_label_list)
                    ax.axvline(x=limits['start'], color='red', linestyle='--')
                    ax.axvline(x=limits['end'], color='red', linestyle='--')
                    ax.set_title(f'Average Line Plot for {hem} - {area}')
                    ax.set_xlabel('times')
                    ax.set_ylabel('Average Value')
                    ax.axhline(y=0, color='gray', linestyle='-')
                    ax.axvline(x=0, color='black', linestyle='--')
                    #del unuseful borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    pdf.savefig()
                    plt.close()

#clusters df

def chunk_cluster_dataframes(signal, peak_periods):
    print('Cluster dataframes execution...')
    summaryRH = {}
    peak_period = peak_periods['R']
    hem='RH'
    with PdfPages(date_fold+f'/{monk}_{task}_{hem}_{signal_of_interest}_peaks_detection.pdf') as pdf:
      
        for group_name, channels in grouped_ch_loca_R.items():
    
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names)
            period = peak_period[group_name]
        
            if not data_temp.empty:
        
                data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time'])['PE'].mean().reset_index()
                data_temp_mean = data_temp_mean[(data_temp_mean['time'] >= period['start']) & (data_temp_mean['time'] <= period['end'])]
                
                summary_temp = {}
                idx_step1=0
                y=0
                plt.figure()
                for learn_clus, group in data_temp_mean.groupby('learn_clus_pb'):
                    c = col[y]
                    summary_temp[learn_clus], idx_step1 = summarize_peak_data(group, group_name, learn_clus, pdf, 'yes', c, idx_step1)#yes = detection peak plot / no = no plot
                    y+=1
                plt.title(f'Peak detection - {group_name}')
                plt.xlabel('time')
                plt.ylabel('amplitude')
                pdf.savefig()
                plt.close()
                summaryRH[group_name] = summary_temp
    
    summaryLH = {}
    peak_period = peak_periods['L']
    hem='LH'
    
    with PdfPages(date_fold+f'/{monk}_{task}_{hem}_{signal_of_interest}_peaks_detection.pdf') as pdf:
      
        for group_name, channels in grouped_ch_loca_L.items():
    
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names)
            period = peak_period[group_name]
        
            if not data_temp.empty:
        
                data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time'])['PE'].mean().reset_index()
                data_temp_mean = data_temp_mean[(data_temp_mean['time'] >= period['start']) & (data_temp_mean['time'] <= period['end'])]
                
                summary_temp = {}
                idx_step1=0
                y=0
                plt.figure()
                for learn_clus, group in data_temp_mean.groupby('learn_clus_pb'):
                    c = col[y]
                    summary_temp[learn_clus], idx_step1 = summarize_peak_data(group, group_name, learn_clus, pdf, 'yes', c, idx_step1)#yes = detection peak plot / no = no plot
                    y+=1
                plt.title(f'Peak detection - {group_name}')
                plt.xlabel('time')
                plt.ylabel('amplitude')
                pdf.savefig()
                plt.close()
                summaryLH[group_name] = summary_temp
    return(summaryRH, summaryLH)

#plot clusters

def chunk_cluster_plot(signal, peak_periods):
    print('Plot cluster execution...')
    #fig with clus 1 to 3 #(and RL)
    with PdfPages(date_fold+'/{monk}_{task}_RH_{signal_of_interest}_learnstep.pdf') as pdf:
        
        peak_period = peak_periods['R']
    
        for group_name, channels in grouped_ch_loca_R.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()            
           
            #stat dfs
            data_temp_stat = data_temp.groupby(['learn_clus_pb', 'pb', 'time'])['PE'].mean().reset_index()
    
            #artefact removal
            non_outliers = artefact_removal(data_temp_mean, 'learn_clus_pb')
            non_outliers_stat = artefact_removal(data_temp_stat, 'learn_clus_pb')
    
            #add a baseline (-100 to 0s)
            non_outliers_normalized = add_baseline(non_outliers, 'learn_clus_pb')
            non_outliers_normalized_stat = add_baseline(non_outliers_stat, 'learn_clus_pb')
    
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -tb) & (non_outliers_normalized_stat['time'] <= ta)]
            
            # Stat between clusters
            stat_results = test_stat_diff_non_param(non_outliers_normalized_stat, 'learn_clus_pb', 'time', 'PE_normalized')
           
            #plot data
            g = sns.relplot(data=non_outliers_normalized, x='time', y='PE_normalized', hue='learn_clus_pb', facet_kws=dict(sharey=False), kind='line')
            for learn_clus in non_outliers_normalized['learn_clus_pb'].unique():
                temp_df = non_outliers_normalized[non_outliers_normalized['learn_clus_pb'] == learn_clus]
                plt.fill_between(temp_df['time'], 
                                 temp_df['PE_normalized'] - temp_df['SEM'], 
                                 temp_df['PE_normalized'] + temp_df['SEM'], 
                                 alpha=0.3)
      
            plt.axvline(x=period['start'], color='red', linestyle='--')#+100 to adapt to xticks starting from -100
            plt.axvline(x=period['end'], color='red', linestyle='--')
            plt.legend(title='learning step')
            g._legend.remove()
            plt.title(group_name)
            
            for ax in g.axes.flat:
                for time, p_value in stat_results.itertuples(index=False):
                    if p_value <= 0.05: 
                        ax.annotate('.', xy=(time, data_temp_mean['PE'].max()), 
                                    textcoords='offset points', xytext=(0,5), ha='center', 
                                    color='r', size=30)
            pdf.savefig()
            plt.close()
    
    #fig with clus 1 to 3 #(and RL)    
    with PdfPages(date_fold+'/{monk}_{task}_LH_{signal_of_interest}_learnstep.pdf') as pdf:
        
        peak_period = peak_periods['L']
    
        for group_name, channels in grouped_ch_loca_L.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()            
            
            #stat dfs
            data_temp_stat = data_temp.groupby(['learn_clus_pb', 'pb', 'time'])['PE'].mean().reset_index()
    
            #artefact removal
            non_outliers = artefact_removal(data_temp_mean, 'learn_clus_pb')
            non_outliers_stat = artefact_removal(data_temp_stat, 'learn_clus_pb')
    
            #add a baseline (-100 to 0s)
            non_outliers_normalized = add_baseline(non_outliers, 'learn_clus_pb')
            non_outliers_normalized_stat = add_baseline(non_outliers_stat, 'learn_clus_pb')
    
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -100) & (non_outliers_normalized['time'] <= 500)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -100) & (non_outliers_normalized_stat['time'] <= 500)]
           
            # Stat between clusters
            stat_results = test_stat_diff_non_param(non_outliers_normalized_stat, 'learn_clus_pb', 'time', 'PE_normalized')
           
            #plot data
            g = sns.relplot(data=non_outliers_normalized, x='time', y='PE_normalized', hue='learn_clus_pb', facet_kws=dict(sharey=False), kind='line')
            for learn_clus in non_outliers_normalized['learn_clus_pb'].unique():
                temp_df = non_outliers_normalized[non_outliers_normalized['learn_clus_pb'] == learn_clus]
                plt.fill_between(temp_df['time'], 
                                 temp_df['PE_normalized'] - temp_df['SEM'], 
                                 temp_df['PE_normalized'] + temp_df['SEM'], 
                                 alpha=0.3)
          
            plt.axvline(x=period['start'], color='red', linestyle='--')#+100 to adapt to xticks starting from -100
            plt.axvline(x=period['end'], color='red', linestyle='--')
            plt.legend(title='learning step')
            g._legend.remove()
            plt.title(group_name)
            
            for ax in g.axes.flat:
                for time, p_value in stat_results.itertuples(index=False):
                    if p_value <= 0.05: 
                        ax.annotate('.', xy=(time, data_temp_mean['PE'].max()), 
                                    textcoords='offset points', xytext=(0,5), ha='center', 
                                    color='r', size=30)
            
            pdf.savefig()
            plt.close()
    
    plt.close('all')

def chunk_statistic_dataframe(signal, peak_periods):
    print('Statistic df execution...')
    temp = signal.groupby(['hem', 'area', 'pb', 'time', 'learn_clus_pb'])['PE'].mean().reset_index()
    grouped = temp.groupby(['hem', 'area', 'pb', 'learn_clus_pb'])
    
    #positive peak
    results = []
    idx_step1=0
    for name, group in grouped:
        hem = name[0]
        area=name[1]
        period=peak_periods[hem][area]
        group = group[(group['time'] >= period['start']) & (group['time'] <= period['end'])]
        #summary = summarize_peak_data(group, name, 'none', 'none', 'no')
        summary, idx_step1 = summarize_peak_data(group, '', 'none', 'none', 'no', 'none', idx_step1)
        time, amplitude = summary['Positive Peak']
        results.append({
            'hem': name[0],
            'area': name[1],
            'pb': name[2],
            'clus': name[3],
            'latency': time,
            'amplitude': amplitude
        })
    
    pos_df = pd.DataFrame(results).dropna()
    
    pos_df.to_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{signal_of_interest}_PE.csv', index=False)
    
    #negative peak
    
    results = []
    idx_step1=0
    for name, group in grouped:
        hem = name[0]
        area=name[1]
        period=peak_periods[hem][area]
        group = group[(group['time'] >= period['start']) & (group['time'] <= period['end'])]
        #summary = summarize_peak_data(group, name, 'none', 'none', 'no')
        summary, idx_step1 = summarize_peak_data(group, '', 'none', 'none', 'no', 'none', idx_step1)
        time, amplitude = summary['Negative Peak']
        results.append({
            'hem': name[0],
            'area': name[1],
            'pb': name[2],
            'clus': name[3],
            'latency': time,
            'amplitude': amplitude
        })
    
    neg_df = pd.DataFrame(results).dropna()
    
    return pos_df, neg_df

##Statistics
##Kruskall-Wallis

def chunk_statistics(pos_df): 
    print('Statistic execution...')
    grouped = pos_df.groupby(['area', 'clus'])
    
    pos_df['mean'] = grouped['amplitude'].transform(lambda x: x.mean(skipna=True))
    pos_df['sd'] = grouped['amplitude'].transform(lambda x: x.std(skipna=True))
    
    pos_df = pos_df[
        (pos_df['amplitude'] >= pos_df['mean'] - 5 * pos_df['sd']) &
        (pos_df['amplitude'] <= pos_df['mean'] + 5 * pos_df['sd'])]
    
    pos_df = pos_df.drop(columns=['mean', 'sd'])
    
    pos_df['mean'] = grouped['latency'].transform(lambda x: x.mean(skipna=True))
    pos_df['sd'] = grouped['latency'].transform(lambda x: x.std(skipna=True))
    
    pos_df = pos_df[
        (pos_df['latency'] >= pos_df['mean'] - 5 * pos_df['sd']) &
        (pos_df['latency'] <= pos_df['mean'] + 5 * pos_df['sd'])]
    
    pos_df = pos_df.drop(columns=['mean', 'sd'])
    
    #stats
    if monk == 'Israel':
        pos_df = pos_df[pos_df["hem"] == "R"]
    else:
        pos_df = pos_df[pos_df["hem"] == "L"]
        
    amp_kk = kruskal_analysis(pos_df, 'amplitude')
    lat_kk = kruskal_analysis(pos_df, 'latency')
    
    amp_post_hoc_df, status = post_hoc_MW(pos_df, amp_kk, 'amplitude')
    lat_post_hoc_df, status = post_hoc_MW(pos_df, lat_kk, 'latency')
    
    #visualisation
    
    path_PDF = date_fold+f'/{monk}_{task}_{signal_of_interest}_amp_stats.pdf'
    stat_visualisation(amp_kk, amp_post_hoc_df, status, path_PDF)
    
    path_PDF = date_fold+f'/{monk}_{task}_{signal_of_interest}_lat_stats.pdf'
    stat_visualisation(lat_kk, lat_post_hoc_df, status, path_PDF)

# plot CTO
def chunk_plot_CTO(CTO_signal):
    print('Plot CTO execution...')
    mcc_data = []
    lpfc_data = []
    
    for group_name, channels in grouped_ch_loca_R.items():
        
        channel_names = [chnames[i] for i in channels]  
        data_temp = filter_channels(CTO_signal, channel_names).dropna()
    
        data_temp_mean = data_temp.groupby(['session', 'time']).agg(
                        PE=('PE', 'mean'),
                        SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                    ).reset_index()            
    
        # Artefact removal
        non_outliers = artefact_removal(data_temp_mean, 'session')
    
        # Ajouter une baseline (-100 à 0)
        non_outliers_normalized = add_baseline(non_outliers, 'session')
    
        non_outliers_normalized = non_outliers_normalized[
            (non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)
        ]
    
        non_outliers_normalized = non_outliers_normalized.groupby(['time']).agg(
                        PE=('PE_normalized', 'mean'),
                        SEM=('SEM', 'mean')).reset_index()   
    
        # Extraire le numéro à la fin du group_name
        group_number = int(group_name.split('_')[-1])
    
        # Ajouter les colonnes pour le group_name et le numéro associé
        non_outliers_normalized['group_name'] = group_name
        non_outliers_normalized['group_number'] = group_number
    
        # Ajouter les données au bon groupe en fonction du préfixe
        if group_name.startswith('MCC_'):
            mcc_data.append(non_outliers_normalized)
        elif group_name.startswith('LPFC_'):
            lpfc_data.append(non_outliers_normalized)
    
    # Concaténer les données de chaque groupe
    mcc_data = pd.concat(mcc_data, ignore_index=True)
    lpfc_data = pd.concat(lpfc_data, ignore_index=True)
    
    #plot data
    title_fontsize = 20
    label_fontsize = 16
    
    path_PDF = date_fold+f'/{monk}_CTO_positive'
    with PdfPages(path_PDF + '.pdf') as pdf:
        # Plot pour MCC
        plt.figure(figsize=(8, 8))
        sns.set_palette('tab10')  # Utiliser une palette standard
        ax = sns.lineplot(data=mcc_data, x='time', y='PE', hue='group_number')
        
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
        plt.title('MCC', fontsize=title_fontsize)
        ax.set_xlabel('Time (ms)', fontsize=label_fontsize)
        ax.set_ylabel('')
        plt.legend(title='')
        plt.xticks(fontsize=label_fontsize)
        plt.yticks(fontsize=label_fontsize)
        pdf.savefig()
        plt.close()
    
        # Plot pour LPFC
        plt.figure(figsize=(8, 8))
        ax = sns.lineplot(data=lpfc_data, x='time', y='PE', hue='group_number')
        
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
        plt.title('LPFC', fontsize=title_fontsize)
        ax.set_xlabel('Time (ms)', fontsize=label_fontsize)
        ax.set_ylabel('')
        plt.legend(title='')
        plt.xticks(fontsize=label_fontsize)
        plt.yticks(fontsize=label_fontsize)
        pdf.savefig()
        plt.close()
  
#plot step 3 & RL
    
def chunk_RL_comp(signal, RL_signal):
    print('RL comparison execution...')
    path_PDF = date_fold+f'/{monk}_{task}_RL_LH_{signal_of_interest}'
    
    with PdfPages(path_PDF+'.pdf') as pdf:
        
        peak_period = peak_periods['L']
    
        for group_name, channels in grouped_ch_loca_L.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
            data_RL = filter_channels(RL_signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()    
            
            data_RL_mean = data_RL.groupby(['pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()         
            
            #select step 3
            data_temp_mean = data_temp_mean[(data_temp_mean['learn_clus_pb'] ==3)]
    
            #stat dfs
            data_temp_stat = data_temp.groupby(['learn_clus_pb', 'pb', 'time'])['PE'].mean().reset_index()
            data_temp_stat = data_temp_stat[(data_temp_stat['learn_clus_pb'] ==3)]
    
            #artefact removal
            non_outliers = artefact_removal(data_temp_mean, 'learn_clus_pb')
            non_outliersRL = artefact_removal(data_RL_mean, 'pb')
            non_outliers_stat = artefact_removal(data_temp_stat, 'pb')
    
            #add a baseline (-100 to 0s)
            non_outliers_normalized = add_baseline(non_outliers, 'learn_clus_pb')
            non_outliers_normalizedRL = add_baseline(non_outliersRL, 'pb')
            non_outliers_normalized_stat = add_baseline(non_outliers_stat, 'pb')
    
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_normalizedRL = non_outliers_normalizedRL[(non_outliers_normalizedRL['time'] >= -tb) & (non_outliers_normalizedRL['time'] <= ta)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -tb) & (non_outliers_normalized_stat['time'] <= ta)]
    
            # Stat
            stat_results = stat_between_steps(non_outliers_normalized_stat, non_outliers_normalizedRL, 'time', 'PE_normalized')
            
            # Mean df
            non_outliers_normalized = non_outliers_normalized.groupby(['time']).agg(
                            PE=('PE_normalized', 'mean'),
                            SEM=('SEM','mean')).reset_index() 
            non_outliers_normalized['gp'] = 'step3'
            
            non_outliers_normalizedRL = non_outliers_normalizedRL.groupby(['time']).agg(
                            PE=('PE_normalized', 'mean'),
                            SEM=('SEM','mean')).reset_index() 
            non_outliers_normalizedRL['gp'] = 'RL'
    
            combined_df = pd.concat([non_outliers_normalized, non_outliers_normalizedRL], axis=0).reset_index(drop=True)
    
            #plot data
            g = sns.relplot(data=combined_df, x='time', y='PE', hue='gp', facet_kws=dict(sharey=False), kind='line')
            for gp in combined_df['gp'].unique():
                temp_df = combined_df[combined_df['gp'] == gp]
                plt.fill_between(temp_df['time'], 
                                 temp_df['PE'] - temp_df['SEM'], 
                                 temp_df['PE'] + temp_df['SEM'], 
                                 alpha=0.3)
            
            plt.axvline(x=period['start'], color='red', linestyle='--')#+100 to adapt to xticks starting from -100
            plt.axvline(x=period['end'], color='red', linestyle='--')      
            plt.legend(title='learning step')
            g._legend.remove()
            plt.title(group_name)
            
            for ax in g.axes.flat:
                for time, _, _, p_value in stat_results.itertuples(index=False):
                    if p_value <= 0.05: 
                        ax.annotate('.', xy=(time, data_temp_mean['PE'].max()), 
                                    textcoords='offset points', xytext=(0,5), ha='center', 
                                    color='r', size=30)
            pdf.savefig()
            plt.close()  
      
    path_PDF = date_fold+f'/{monk}_{task}_RL_RH_{signal_of_interest}'
    
    with PdfPages(path_PDF+'.pdf') as pdf:
        
        peak_period = peak_periods['R']
    
        for group_name, channels in grouped_ch_loca_R.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
            data_RL = filter_channels(RL_signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['learn_clus_pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()    
            
            data_RL_mean = data_RL.groupby(['pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()         
            
            #select step 3
            data_temp_mean = data_temp_mean[(data_temp_mean['learn_clus_pb'] ==3)]
    
            #stat dfs
            data_temp_stat = data_temp.groupby(['learn_clus_pb', 'pb', 'time'])['PE'].mean().reset_index()
            data_temp_stat = data_temp_stat[(data_temp_stat['learn_clus_pb'] ==3)]
    
            #artefact removal
            non_outliers = artefact_removal(data_temp_mean, 'learn_clus_pb')
            non_outliersRL = artefact_removal(data_RL_mean, 'pb')
            non_outliers_stat = artefact_removal(data_temp_stat, 'pb')
    
            #add a baseline (-100 to 0s)
            non_outliers_normalized = add_baseline(non_outliers, 'learn_clus_pb')
            non_outliers_normalizedRL = add_baseline(non_outliersRL, 'pb')
            non_outliers_normalized_stat = add_baseline(non_outliers_stat, 'pb')
    
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_normalizedRL = non_outliers_normalizedRL[(non_outliers_normalizedRL['time'] >= -tb) & (non_outliers_normalizedRL['time'] <= ta)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -tb) & (non_outliers_normalized_stat['time'] <= ta)]
    
            # Stat
            stat_results = stat_between_steps(non_outliers_normalized_stat, non_outliers_normalizedRL, 'time', 'PE_normalized')
            
            # Mean df
            non_outliers_normalized = non_outliers_normalized.groupby(['time']).agg(
                            PE=('PE_normalized', 'mean'),
                            SEM=('SEM','mean')).reset_index() 
            non_outliers_normalized['gp'] = 'step3'
            
            non_outliers_normalizedRL = non_outliers_normalizedRL.groupby(['time']).agg(
                            PE=('PE_normalized', 'mean'),
                            SEM=('SEM','mean')).reset_index() 
            non_outliers_normalizedRL['gp'] = 'RL'
    
            combined_df = pd.concat([non_outliers_normalized, non_outliers_normalizedRL], axis=0).reset_index(drop=True)
    
            #plot data
            g = sns.relplot(data=combined_df, x='time', y='PE', hue='gp', facet_kws=dict(sharey=False), kind='line')
            for gp in combined_df['gp'].unique():
                temp_df = combined_df[combined_df['gp'] == gp]
                plt.fill_between(temp_df['time'], 
                                 temp_df['PE'] - temp_df['SEM'], 
                                 temp_df['PE'] + temp_df['SEM'], 
                                 alpha=0.3)
            
            plt.axvline(x=period['start'], color='red', linestyle='--')#+100 to adapt to xticks starting from -100
            plt.axvline(x=period['end'], color='red', linestyle='--')      
            plt.legend(title='learning step')
            g._legend.remove()
            plt.title(group_name)
            
            for ax in g.axes.flat:
                for time, _, _, p_value in stat_results.itertuples(index=False):
                    if p_value <= 0.05: 
                        ax.annotate('.', xy=(time, data_temp_mean['PE'].max()), 
                                    textcoords='offset points', xytext=(0,5), ha='center', 
                                    color='r', size=30)
            pdf.savefig()
            plt.close()    

# temporality df
def chunk_temporality(pos_df):
    print('Temporality execution...')

    for h in ['L','R']:
        
        curr = pos_df[(pos_df['hem']== h)]
        
        #groups AP ant & post
        latency_diff_ant = calculate_latency_differences2(curr, ['LPFC_1', 'LPFC_2', 'LPFC_3'], ['MCC_1', 'MCC_2', 'MCC_3'])
        latency_diff_post = calculate_latency_differences2(curr, ['LPFC_4', 'LPFC_5', 'LPFC_6'], ['MCC_4', 'MCC_5', 'MCC_6'])
        latency_diff_ant['area'] = 'ant_gp'
        latency_diff_post['area'] = 'post_gp'
        
        #groups AP 1 to 6
        latency_diff_ant1 = calculate_latency_differences2(curr, ['LPFC_1'], ['MCC_1'])
        latency_diff_ant1['area'] = 'gp1'
        latency_diff_ant2 = calculate_latency_differences2(curr, ['LPFC_2'], ['MCC_2'])
        latency_diff_ant2['area'] = 'gp2'
        latency_diff_ant3 = calculate_latency_differences2(curr, ['LPFC_3'], ['MCC_3'])
        latency_diff_ant3['area'] = 'gp3'
        latency_diff_post4 = calculate_latency_differences2(curr, ['LPFC_4'], ['MCC_4'])
        latency_diff_post4['area'] = 'gp4'
        latency_diff_post5 = calculate_latency_differences2(curr, ['LPFC_5'], ['MCC_5'])
        latency_diff_post5['area'] = 'gp5'
        latency_diff_post6 = calculate_latency_differences2(curr, ['LPFC_6'], ['MCC_6'])
        latency_diff_post6['area'] = 'gp6'
    
        dfs = [latency_diff_ant, latency_diff_post, latency_diff_ant1, latency_diff_ant2, latency_diff_ant3, latency_diff_post4, latency_diff_post5, latency_diff_post6]
    
        latency_diff = pd.concat(dfs, ignore_index=True)
        
        latency_diff = latency_diff.drop(columns=['Mean Latency (LPFC)','Mean Latency (MCC)'])
        latency_diff = latency_diff.rename(columns={'Cluster': 'step', 'Latency Difference (LPFC - MCC)': 'lat_diff'})
        
        latency_diff.to_csv(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{h}H_{signal_of_interest}_temporality_diff.csv', index=False)
        
        #remove outliers before stats
        
        grouped = latency_diff.groupby(['area', 'clus'])
    
        latency_diff['mean'] = grouped['lat_diff'].transform(lambda x: x.mean(skipna=True))
        latency_diff['sd'] = grouped['lat_diff'].transform(lambda x: x.std(skipna=True))
        
        latency_diff = latency_diff[
            (latency_diff['lat_diff'] >= latency_diff['mean'] - 2 * latency_diff['sd']) &
            (latency_diff['lat_diff'] <= latency_diff['mean'] + 2 * latency_diff['sd'])
        ]
        
        latency_diff = latency_diff.drop(columns=['mean', 'sd'])
    
    #stats
        
        kk = kruskal_analysis(latency_diff, 'lat_diff')
        post_hoc_df, status = post_hoc_MW(latency_diff, kk, 'lat_diff')
        
        path_PDF = date_fold+f'/{monk}_{task}_{h}H_{signal_of_interest}_tempo_stats.pdf'
        stat_visualisation(kk, post_hoc_df, status, path_PDF)

# plot diff with pos and neg
def chunk_plot_diff():
    pos = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_PEcorr_pb.csv')
    pos = pos.merge(clusters2, on='pb')
    pos = pos.merge(ch_loca, on='ch')
    pos = pos.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    pos = pos[(pos['time'] >= -tb) & (pos['time'] <= ta)]
    
    neg = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_PEinc_pb.csv')
    neg = neg.merge(clusters2, on='pb')
    neg = neg.merge(ch_loca, on='ch')
    neg = neg.groupby(['hem', 'area', 'pb', 'time'])['PE'].mean().reset_index()
    neg = neg[(neg['time'] >= -tb) & (neg['time'] <= ta)]
    
    savepath = date_fold+'/{monk}_diff_fig.pdf'
        
    with PdfPages(savepath) as pdf:
        for hem in pos['hem'].unique():
            for area in pos['area'].unique():
                pos_sub = pos[(pos['hem'] == hem) & (pos['area'] == area)]
                neg_sub = neg[(neg['hem'] == hem) & (neg['area'] == area)]
                if not pos_sub.empty:  
                    pos_pivot_table = pos_sub.pivot(index='pb', columns='time', values='PE')
                    mean = pos_pivot_table.mean().mean() 
                    sd = pos_pivot_table.stack().std()
                    outliers_pos = mean + 3 * sd
                    outliers_neg = mean - 3 * sd
                    artefact_sessions = pos_pivot_table.apply(lambda row: any((row < outliers_neg) | (row > outliers_pos)), axis=1)
                    pos_pivot_table = pos_pivot_table[~artefact_sessions] 
                    pos_mean = pos_pivot_table.mean(axis=0)  
    
                    neg_pivot_table = neg_sub.pivot(index='pb', columns='time', values='PE')
                    mean = neg_pivot_table.mean().mean() 
                    sd = neg_pivot_table.stack().std()
                    outliers_pos = mean + 3 * sd
                    outliers_neg = mean - 3 * sd
                    artefact_sessions = neg_pivot_table.apply(lambda row: any((row < outliers_neg) | (row > outliers_pos)), axis=1)
                    neg_pivot_table = neg_pivot_table[~artefact_sessions] 
                    neg_mean = neg_pivot_table.mean(axis=0)  
                    
                    diff_mean = pos_mean - neg_mean
    
                    fig,ax=plt.subplots(1,1)
                    ax.plot(pos_mean, color='red', alpha=0.5, linestyle='--', label='positive')
                    ax.plot(neg_mean, color='blue', alpha=0.5, linestyle='--', label='negative')
                    ax.plot(diff_mean, color='purple', alpha=1.0, linestyle='-',linewidth=2, label='difference')
                    ax.set_title(f'{hem} - {area}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Mean Value')
                    ax.axvline(x=0, color='black', linestyle='--')
                    ax.axhline(y=0, color='gray', linestyle='-')
                    ax.legend(loc='upper right')
                    #del unuseful borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    pdf.savefig()
                    plt.close()
    
# %% execution

"""execute specific functions according to the signal of interest"""

if signal_of_interest == "positive":
    peak_periods = chunk_peak_window(curr_signal)
    chunk_cluster_dataframes(curr_signal, peak_periods)
    chunk_cluster_plot(curr_signal, peak_periods)
    pos_df, neg_df = chunk_statistic_dataframe(curr_signal, peak_periods)
    chunk_statistics(pos_df)
    chunk_plot_CTO(CTO_signal) 
    chunk_RL_comp(curr_signal, RL_signal)
    chunk_temporality(pos_df)
elif signal_of_interest == "negative":
    peak_periods = chunk_peak_window(curr_signal)
    chunk_cluster_dataframes(curr_signal, peak_periods)
    chunk_cluster_plot(curr_signal, peak_periods)
    pos_df, neg_df = chunk_statistic_dataframe(curr_signal, peak_periods)
    chunk_statistics(pos_df)
    chunk_plot_CTO(CTO_signal) 
    chunk_RL_comp(curr_signal, RL_signal)
    chunk_temporality(pos_df)
elif signal_of_interest == "difference":
    chunk_diff_heatmap(curr_signal)  
    peak_periods = chunk_peak_window(curr_signal)
    chunk_plot_peak_window(curr_signal, peak_periods)
    chunk_cluster_dataframes(curr_signal, peak_periods)
    chunk_cluster_plot(curr_signal, peak_periods)
    pos_df, neg_df = chunk_statistic_dataframe(curr_signal, peak_periods)
    chunk_statistics(pos_df)
    chunk_plot_diff()
else:
    print(f"unrecognized signal : {curr_signal}")
    