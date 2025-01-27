#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:06:46 2023

@author: Marion Ducret
"""

# %% library & variables


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

with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

#group ch
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

monk = input('which monkey ? :')
task = input('which task ? :')

file_path = f'/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{monk}_learn_grp_pb.csv'
column_names = ["index", "pb", "group", "error"]
clusters = pd.read_csv(file_path, delimiter=",", names=column_names, skiprows=1)
clusters.rename(columns={'group': 'learn_clus_pb'}, inplace=True)
clusters.drop(clusters.columns[0], axis=1, inplace=True)
clusters2=clusters[['pb','learn_clus_pb']]
clusters2 = clusters2.drop_duplicates()

file_path = f'/Users/marionducret/Desktop/PREDYCT/Analyses/Behaviour/{monk}_motiv_grp_pb.csv'
clusters_motiv = pd.read_csv(file_path, delimiter=",")
clusters_motiv.drop(clusters_motiv.columns[[0,2,3,5]],axis=1, inplace=True)
clusters_motiv.rename(columns={'changeNB': 'pb', 'motiv_clus2': 'motiv_gp'}, inplace=True)

if monk == 'Israel':
    clusters_motiv['motiv_gp'] = clusters_motiv['motiv_gp'].replace({1: 2, 2: 1})

clusters2 = clusters2.merge(clusters_motiv, on='pb')
clusters2['gp'] = clusters2['learn_clus_pb'].astype(str) + '-' + clusters2['motiv_gp'].astype(str)

# %% functions

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

def artefact_removal2(data, group, main_gp):
    cleaned_data = pd.DataFrame()  # DataFrame vide pour stocker les résultats

    # Itérer sur chaque groupe de `main_gp`
    for group_value, group_data in data.groupby(main_gp):
        # Calculez la moyenne et l'écart-type pour ce groupe spécifique
        curr_mean = group_data['PE'].mean()
        curr_sd = group_data['PE'].std()
        outliers_pos = curr_mean + 10 * curr_sd
        outliers_neg = curr_mean - 10 * curr_sd
        
        # Identifiez les outliers pour ce groupe
        outliers = group_data[(group_data['PE'] < outliers_neg) | (group_data['PE'] > outliers_pos)]
        outlier_pb_values = outliers[group].unique()
        
        # Retirez les outliers du groupe actuel
        non_outliers = group_data[~group_data[group].isin(outlier_pb_values)]
        
        # Ajouter les données nettoyées à `cleaned_data`
        cleaned_data = pd.concat([cleaned_data, non_outliers], ignore_index=True)
    
    return cleaned_data

def add_baseline(data, group):
    baseline_data = data[(data['time'] >= -100) & (data['time'] <= 0)]
    baseline_values = baseline_data.groupby([group])['PE'].mean().reset_index()
    baseline_df = data.merge(baseline_values, on=group, suffixes=('_actual', '_baseline'))
    baseline_df['PE_normalized'] = baseline_df['PE_actual'] - baseline_df['PE_baseline']
    return baseline_df

def prepare_plot_data(summary):
    plot_data = []
    for group_name, clusters in summary.items():
        for cluster, peaks in clusters.items():
            for peak_type, (latency, amplitude) in peaks.items():
                if latency is not None and amplitude is not None:
                    plot_data.append({
                        'Group': group_name,
                        'Cluster': cluster,
                        'Peak Type': peak_type,
                        'Latency': latency,
                        'Amplitude': amplitude
                    })
    return pd.DataFrame(plot_data)

def classify_pair2(area):
    if 'MCC' in area:
        return 'MCC'
    elif 'LPFC' in area:
        return 'LPFC'
    else:
        return 'Other'

##PEAK DETECTION##

def detect_peaks(data, group_name, learn_clus, pdf, plot, distance=5, width=5):
    height=data['PE'].mean()+data['PE'].std()
    peaks_positive, _ = find_peaks(data['PE'], distance=distance, width=width, height=height)
    peaks_negative, _ = find_peaks(-data['PE'], distance=distance, width=width, height=height)
    ##show peaks found with find_peaks function
    if plot=='yes':
        plt.figure()
        plt.plot(data['time'], data['PE'])
        plt.plot(data['time'].iloc[peaks_positive], data['PE'].iloc[peaks_positive], 'ro', label='pos')
        plt.plot(data['time'].iloc[peaks_negative], data['PE'].iloc[peaks_negative], 'bo', label='neg')
        plt.title(f'Peak detection - {group_name} group {learn_clus}')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.legend()
        pdf.savefig()
        plt.close()
    return peaks_positive, peaks_negative

def analyze_peaks(data, group_name, learn_clus, pdf, plot):
    peaks_positive, peaks_negative = detect_peaks(data, group_name, learn_clus, pdf, plot)
    
    if len(peaks_positive) >= 1:
        pos_peak = peaks_positive[0]
    else:
        pos_peak = None
        
    if len(peaks_negative) >= 1:
        neg_peak = peaks_negative[0]
    else:
        neg_peak = None 
    
    return pos_peak, neg_peak

def get_peak_info(data, peak_indices):
    if peak_indices is None:
        return None, None
    else:
        time_peak = data['time'].iloc[peak_indices]
        amplitude_peak = data['PE'].iloc[peak_indices]
        return time_peak, amplitude_peak

def summarize_peak_data(data, group_name, learn_clus, pdf, plot):
    pos_peak, neg_peak = analyze_peaks(data, group_name, learn_clus, pdf, plot)
    peak_info = {
        'Positive Peak': get_peak_info(data, pos_peak),
        'Negative Peak': get_peak_info(data, neg_peak)
    }
    return peak_info

##STATISTICS##

def test_stat_diff_non_param_1_level(data, group_col, time_col, value_col):
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

def test_stat_diff_non_param(data, main_group_col, sub_group_col, time_col, value_col):
    times = data[time_col].unique()
    p_values = []
    
    for time in times:
        data_at_time = data[data[time_col] == time]
        
        for main_gp, group_data in data_at_time.groupby(main_group_col):
            
            sub_groups = group_data.groupby(sub_group_col)[value_col]
            sub_group_values = [group.values for _, group in sub_groups]

            if len(sub_group_values) > 1:
                try:
                    stat, p_val = mannwhitneyu(sub_group_values[0], sub_group_values[1])
                    
                    # Correction des p-values avec FDR (False Discovery Rate)
                    p_val_corrected = smm.multipletests([p_val], alpha=0.5, method='fdr_bh')[1][0]
                    p_values.append((time, main_gp, p_val_corrected))
                except ValueError:
                    p_values.append((time, main_gp, np.nan))  # Cas où les données sont insuffisantes pour le test
            else:
                p_values.append((time, main_gp, np.nan))
    
    return pd.DataFrame(p_values, columns=[time_col, main_group_col, 'p_value'])

def compute_p_values(stat_clus1, stat_RL, time_col='time', value_col='value'):
    times = np.union1d(stat_clus1[time_col].unique(), stat_RL[time_col].unique())
    p_values = []

    for time in times:
        values_clus1 = stat_clus1[stat_clus1[time_col] == time][value_col]
        values_RL = stat_RL[stat_RL[time_col] == time][value_col]
        
        if len(values_clus1) > 0 and len(values_RL) > 0:
            try:
                u_stat, p_val = mannwhitneyu(values_clus1, values_RL, alternative='two-sided')
                p_values.append((time, p_val))
            except ValueError:
                p_values.append((time, np.nan)) 
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

##PLOT FIGURES##

def plot_latency_differences(latency_diff, title):
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(latency_diff['Cluster'], latency_diff['Latency Difference (LPFC - MCC)'], color='skyblue')
    
    for bar, diff in zip(bars, latency_diff['Latency Difference (LPFC - MCC)']):
        height = bar.get_height()
        if diff > 0:
            annotation = 'LPFC > MCC'
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, annotation, ha='center', va='bottom', fontsize=10, color='green')
        else:
            annotation = 'LPFC < MCC'
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, annotation, ha='center', va='top', fontsize=10, color='red')
    
    plt.xlabel('Learning steps')
    plt.ylabel('Latency difference (ms)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(latency_diff['Cluster'])

def custom_colormap(val):
    if val == 'increase':
        return 'lightcoral'
    elif val == 'decrease':
        return 'lightblue'
    else:
        return 'white'

def significance_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

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
save_dir = '/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/motiv_analysis/'
date = datetime.now().strftime("%d%m%y") #date of the day
date_fold = os.path.join(save_dir, date)
os.makedirs(date_fold, exist_ok=True)
print(f"All files will be save in : {date_fold}")

signal_of_interest = input('which signal ? :') #positive or negative or difference

if signal_of_interest == 'positive':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_PEcorr_pb.csv')
    CTO_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_CTO_PEcorr_pb.csv')
    RL_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_RL_PEcorr_pb.csv')
elif signal_of_interest == 'negative':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_PEinc_pb.csv')
    CTO_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_CTO_PEinc_pb.csv')
    RL_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_RL_PEinc_pb.csv')
elif signal_of_interest == 'difference':
    curr_signal = pd.read_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_{task}_diff_pb.csv')
    
curr_signal = curr_signal.merge(clusters2, on='pb')
curr_signal = curr_signal.merge(ch_loca, on='ch')
             
# %% processing

"""compute plots et statistics"""

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

#clusters df

def chunk_cluster_dataframes(signal, peak_periods):
    print('Cluster dataframes execution...')
   
    summaryRH = {}
    peak_period = peak_periods['R']
    hem='RH'
    with PdfPages(date_fold+f'/{monk}_{task}_{hem}_{signal_of_interest}_peaks_detection_motiv.pdf') as pdf:
    
        for group_name, channels in grouped_ch_loca_R.items():
    
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names)
            period = peak_period[group_name]
        
            if not data_temp.empty:
        
                data_temp_mean = data_temp.groupby(['gp', 'time'])['PE'].mean().reset_index()
                data_temp_mean = data_temp_mean[(data_temp_mean['time'] >= period['start']) & (data_temp_mean['time'] <= period['end'])]
                
                summary_temp = {}
                for clus, group in data_temp_mean.groupby('gp'):
                    summary_temp[clus] = summarize_peak_data(group, group_name, clus, pdf, 'yes')#yes = detection peak plot / no = no plot
                
                summaryRH[group_name] = summary_temp
    
    summaryLH = {}
    peak_period = peak_periods['L']
    hem='LH'
    
    with PdfPages(date_fold+f'/{monk}_{task}_{hem}_{signal_of_interest}_peaks_detection_motiv.pdf') as pdf:
    
        for group_name, channels in grouped_ch_loca_L.items():
    
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names)
            period = peak_period[group_name]
        
            if not data_temp.empty:
        
                data_temp_mean = data_temp.groupby(['gp', 'time'])['PE'].mean().reset_index()
                data_temp_mean = data_temp_mean[(data_temp_mean['time'] >= period['start']) & (data_temp_mean['time'] <= period['end'])]
                
                summary_temp = {}
                for clus, group in data_temp_mean.groupby('gp'):
                    summary_temp[clus] = summarize_peak_data(group, group_name, clus, pdf, 'yes')
                
                summaryLH[group_name] = summary_temp
    return summaryRH, summaryLH
        
#plot clusters
     
def chunk_cluster_plot (signal, peak_periods):
    print('Plot cluster execution...')
    #learning steps separated
    
    with PdfPages(date_fold+f'/{monk}_{task}_{signal_of_interest}_motiv.pdf') as pdf:
        
        if monk == 'Israel':
            peak_period = peak_periods['R']
            gp_ch = grouped_ch_loca_R
        else:
            peak_period = peak_periods['L']
            gp_ch = grouped_ch_loca_L
    
        for group_name, channels in gp_ch.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['gp', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()            
            
            #stat dfs
            data_temp_stat = data_temp.groupby(['gp', 'pb', 'time'])['PE'].mean().reset_index()
            
            #main group
            data_temp_stat['main_gp'] = data_temp_stat['gp'].str.split('-').str[0]
            data_temp_mean['main_gp'] = data_temp_mean['gp'].str.split('-').str[0]
    
            #artefact removal
            #A ADAPTEEEEER
            #non_outliers = artefact_removal(data_temp_mean, 'gp')
            #non_outliers_stat = artefact_removal(data_temp_stat, 'gp', 'main_gp')
        
            #add a baseline (-100 to 0)
            non_outliers_normalized = add_baseline(data_temp_mean, 'gp')
            non_outliers_normalized_stat = add_baseline(data_temp_stat, 'gp')
            
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -tb) & (non_outliers_normalized_stat['time'] <= ta)]
          
            # Stat between clusters
    
            stat_results = test_stat_diff_non_param(
                data=non_outliers_normalized_stat, 
                main_group_col='main_gp', 
                sub_group_col='gp', 
                time_col='time', 
                value_col='PE_normalized'
            )
                  
            #plot data
    
            g = sns.relplot(data=non_outliers_normalized, x='time', y='PE_normalized', hue='gp', col='main_gp', facet_kws=dict(sharey=False), kind='line')
            for main_gp in non_outliers_normalized['main_gp'].unique():
                temp_df = non_outliers_normalized[non_outliers_normalized['main_gp'] == main_gp]
                for ax in g.axes.flat:
                    if ax.get_title().endswith(f"main_gp = {main_gp}"):
                        for clus in temp_df['gp'].unique():
                            sub_df = temp_df[temp_df['gp'] == clus]
                            ax.fill_between(
                                sub_df['time'], 
                                sub_df['PE_normalized'] - sub_df['SEM'], 
                                sub_df['PE_normalized'] + sub_df['SEM'], 
                                alpha=0.3, 
                                label=f"SEM for {clus}"
                            )
            for ax in g.axes.flat:
                ax.axvline(x=period['start'], color='red', linestyle='--')
                ax.axvline(x=period['end'], color='red', linestyle='--')
    
            g.add_legend(title='learning step & motivation level')
            plt.suptitle(group_name, y=1.02)
            
            for main_gp in non_outliers_normalized['main_gp'].unique():
                temp_df = non_outliers_normalized[non_outliers_normalized['main_gp'] == main_gp]
                for ax in g.axes.flat:
                    if ax.get_title().endswith(f"main_gp = {main_gp}"):
                        for row in stat_results[stat_results['main_gp'] == main_gp].itertuples(index=False):
                            time = row.time
                            p_value = row.p_value
                            if p_value <= 0.05: 
                                ax.annotate('.', xy=(time, temp_df['PE_normalized'].max()), 
                                            textcoords='offset points', xytext=(0,5), ha='center', 
                                    color='r', size=30)
            pdf.savefig()
            plt.close()
    
    #learning steps 1&2 together
    
    with PdfPages(date_fold+f'/{monk}_{task}_{signal_of_interest}_motiv12.pdf') as pdf:
        
        if monk == 'Israel':
            peak_period = peak_periods['R']
            gp_ch = grouped_ch_loca_R
        else:
            peak_period = peak_periods['L']
            gp_ch = grouped_ch_loca_L
    
        for group_name, channels in gp_ch.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
    
            data_temp = data_temp[data_temp['learn_clus_pb'] != 3]
    
            data_temp_mean = data_temp.groupby(['motiv_gp','learn_clus_pb', 'gp', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()   
             
            data_temp_stat = data_temp.groupby(['motiv_gp','learn_clus_pb', 'gp', 'pb', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()            
            
            #artefact removal
            #A ADAPTEEEER
            #non_outliers = artefact_removal(data_temp_mean, 'gp')
            #non_outliers_stat = artefact_removal(data_temp_stat, 'gp')
    
            #add a baseline (-100 to 0)
            non_outliers_normalized = add_baseline(data_temp_mean, 'gp')
            non_outliers_stat_normalized = add_baseline(data_temp_stat, 'gp')
    
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_stat_normalized = non_outliers_stat_normalized[(non_outliers_stat_normalized['time'] >= -tb) & (non_outliers_stat_normalized['time'] <= ta)]
    
            #stats
            
            stat_results = test_stat_diff_non_param(data=non_outliers_stat_normalized, main_group_col='learn_clus_pb', 
                                                    sub_group_col='motiv_gp', time_col='time', value_col='PE_normalized')
            
            #plot data
    
            g = sns.lineplot(
                data=non_outliers_normalized,
                x='time',
                y='PE_normalized',
                hue='learn_clus_pb',   
                style='motiv_gp'
            )
            
            # Add SEM
            for learn_clus in non_outliers_normalized['learn_clus_pb'].unique():
                for motiv_gp in non_outliers_normalized['motiv_gp'].unique():
                    temp_df = non_outliers_normalized[(non_outliers_normalized['learn_clus_pb'] == learn_clus) & 
                                                      (non_outliers_normalized['motiv_gp'] == motiv_gp)]
                    plt.fill_between(temp_df['time'], 
                                     temp_df['PE_normalized'] - temp_df['SEM'], 
                                     temp_df['PE_normalized'] + temp_df['SEM'], 
                                     alpha=0.3)
    
            
            # Add vertical lines for periods
            plt.axvline(x=period['start'], color='red', linestyle='--')
            plt.axvline(x=period['end'], color='red', linestyle='--')
            plt.axvline(x=0, color='black', linestyle='--')
            
            # Add stats
            stat1 = stat_results[stat_results['learn_clus_pb'] == 1]
            for time, _, p_value in stat1.itertuples(index=False):
                if p_value <= 0.05: 
                    g.annotate('.', xy=(time, data_temp_mean['PE'].max()), 
                               textcoords='offset points', xytext=(0,5), ha='center', 
                               color='r', size=30)
            
            # Add stats for learn_clus_pb == 2
            stat2 = stat_results[stat_results['learn_clus_pb'] == 2]
            for time, _, p_value in stat2.itertuples(index=False):
                if p_value <= 0.05: 
                    g.annotate('.', xy=(time, data_temp_mean['PE'].min()), 
                               textcoords='offset points', xytext=(0,5), ha='center', 
                               color='b', size=30)
                        
            # Add legend and title
            plt.title(group_name)
    
            pdf.savefig()
            plt.close()
    
# statistics

def chunk_statistic_dataframe(signal, peak_periods):
    print('Statistic df execution...')

    temp = signal.groupby(['hem', 'area', 'pb', 'time', 'gp'])['PE'].mean().reset_index()
    grouped = temp.groupby(['hem', 'area', 'pb', 'gp'])
    
    #positive peak
    
    results = []
    for name, group in grouped:
        hem = name[0]
        area=name[1]
        period=peak_periods[hem][area]
        group = group[(group['time'] >= period['start']) & (group['time'] <= period['end'])]
        summary = summarize_peak_data(group, '', 'none', 'none', 'no')
        time, amplitude = summary['Positive Peak']
        results.append({
            'hem': name[0],
            'area': name[1],
            'pb': name[2],
            'clus': name[3],
            'latency': time,
            'amplitude': amplitude
        })
    
    pos_df = pd.DataFrame(results)
    
    #negative peak
    
    results = []
    for name, group in grouped:
        hem = name[0]
        area=name[1]
        period=peak_periods[hem][area]
        group = group[(group['time'] >= period['start']) & (group['time'] <= period['end'])]
        summary = summarize_peak_data(group, '', 'none', 'none', 'no')
        time, amplitude = summary['Negative Peak']
        results.append({
            'hem': name[0],
            'area': name[1],
            'pb': name[2],
            'clus': name[3],
            'latency': time,
            'amplitude': amplitude
        })
    
    neg_df = pd.DataFrame(results)
    
    return pos_df, neg_df

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


# save matrix

def chunk_save_CSV(summaryRH, summaryLH):
    print('Save CSV execution...')

    LHdata = prepare_plot_data(summaryLH)
    RHdata = prepare_plot_data(summaryRH)

    RHdata[['Area', 'AP']] = RHdata['Group'].str.split('_', expand=True)
    RHdata['AP'] = RHdata['AP'].astype(int)

    LHdata[['Area', 'AP']] = LHdata['Group'].str.split('_', expand=True)
    LHdata['AP'] = LHdata['AP'].astype(int)

    RHdata.to_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_RH_corrPE_motiv.csv', index=False)
    LHdata.to_csv(f'/Users/marionducret/Desktop/PREDYCT figures matrix/PE/{monk}_LH_corrPE_motiv.csv', index=False)
    
# motivation alone

def chunk_motiv_level(signal, peak_periods):
    print('Motivation alone execution...')

    with PdfPages(date_fold+f'/{monk}_{task}_{signal_of_interest}_motiv_alone.pdf') as pdf:
        
        if monk == 'Israel':
            peak_period = peak_periods['R']
            gp_ch = grouped_ch_loca_R
        else:
            peak_period = peak_periods['L']
            gp_ch = grouped_ch_loca_L
    
        for group_name, channels in gp_ch.items():
            
            period = peak_period[group_name]
            channel_names = [chnames[i] for i in channels]  
            data_temp = filter_channels(signal, channel_names).dropna()
    
            data_temp_mean = data_temp.groupby(['motiv_gp', 'time']).agg(
                            PE=('PE', 'mean'),
                            SEM=('PE', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                        ).reset_index()            
            
            #stat dfs
            data_temp_stat = data_temp.groupby(['motiv_gp', 'pb', 'time'])['PE'].mean().reset_index()
            
            #artefact removal
            non_outliers = artefact_removal(data_temp_mean, 'motiv_gp')
            non_outliers_stat = artefact_removal(data_temp_stat, 'motiv_gp')
        
            #add a baseline (-100 to 0)
            non_outliers_normalized = add_baseline(non_outliers, 'motiv_gp')
            non_outliers_normalized_stat = add_baseline(non_outliers_stat, 'motiv_gp')
            
            non_outliers_normalized = non_outliers_normalized[(non_outliers_normalized['time'] >= -tb) & (non_outliers_normalized['time'] <= ta)]
            non_outliers_normalized_stat = non_outliers_normalized_stat[(non_outliers_normalized_stat['time'] >= -tb) & (non_outliers_normalized_stat['time'] <= ta)]
          
            # Stat between motiv group
    
            stat_results = test_stat_diff_non_param_1_level(non_outliers_normalized_stat,'motiv_gp','time','PE_normalized')
                  
            #plot data
    
            g = sns.relplot(data=non_outliers_normalized, x='time', y='PE_normalized', hue='motiv_gp', facet_kws=dict(sharey=False), kind='line')
            for gp in non_outliers_normalized['motiv_gp'].unique():
                temp_df = non_outliers_normalized[non_outliers_normalized['motiv_gp'] == gp]
                plt.fill_between(temp_df['time'], 
                                 temp_df['PE_normalized'] - temp_df['SEM'], 
                                 temp_df['PE_normalized'] + temp_df['SEM'], 
                                 alpha=0.3)
            
            plt.axvline(x=period['start'], color='red', linestyle='--')#+100 to adapt to xticks starting from -100
            plt.axvline(x=period['end'], color='red', linestyle='--')
            plt.legend(title='motivation')
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


# %% execution

"""execute functions"""

peak_periods = chunk_peak_window(curr_signal)
summaryRH, summaryLH = chunk_cluster_dataframes(curr_signal, peak_periods)
chunk_cluster_plot(curr_signal, peak_periods)
pos_df, neg_df = chunk_statistic_dataframe(curr_signal, peak_periods)
chunk_statistics(pos_df)
chunk_save_CSV(summaryRH, summaryLH)
chunk_motiv_level(curr_signal, peak_periods)

