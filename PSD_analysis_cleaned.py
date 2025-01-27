
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2023

@author: marionducret
"""

#%% libraries


# general
from datetime import datetime
import glob
import os

# data processing 
import numpy as np
import pandas as pd
import pickle

#statistics
from fooof import FOOOF
import mne
from scipy import stats
from scipy.signal import welch
from sklearn.linear_model import LinearRegression

#plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfWriter, PdfReader

#import homemade functions
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/load_functions.py").read())
exec(open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/Scripts PYTHON/save_functions.py").read())

with open('/Users/marionducret/Desktop/PREDYCT figures matrix/chnames.pkl', 'rb') as file:
    chnames = pickle.load(file)

#%% functions

##DATA PROCESSING##

def load_and_resample_raw(path, monk, date, sfreq):
   
    try:
        raw = load_raw_crop(path, monk, date)
        
        raw.resample(sfreq, npad='auto')
        
        return raw

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def compute_psd_with_welch(raw, fmin, fmax, n_fft, sfreq):
    data = raw.get_data()
    psds = []
    freqs = None
    for channel in data:
        freqs, psd = welch(channel, fs=sfreq, nperseg=n_fft)
        psds.append(psd)
    
    psds = np.array(psds)
    
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    psds = psds[:, freq_mask]
    freqs = freqs[freq_mask]
    
    return psds, freqs

##PLOT FIGURES##

def plot_psd_comparison_method(ch_names, psd_data_lists, frequencies, titles, pdf):
    num_channels = len(ch_names)
    downsampling_factor = 200
    
    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    for j, (psd_data_list, title) in enumerate(zip(psd_data_lists, titles)):
        
        ax = axes[j]
        freq = frequencies[j]
        
        #add the mean
        concatenated_data = np.vstack(psd_data_list)
        mean_data = np.mean(concatenated_data, axis=0)
        mean_data = mean_data[::downsampling_factor]
        
        for ch_idx in range(num_channels):
            data = psd_data_list[ch_idx]  
            data = data[::downsampling_factor]
            freq_temp = freq[::downsampling_factor]
            freq_mask = (freq_temp >= 1) & (freq_temp <= 100)
          
            ax.plot(freq_temp[freq_mask], 10 * np.log10(data[freq_mask]), linewidth=.5, alpha =.5)

        #add the mean
        ax.plot(freq_temp[freq_mask], 10 * np.log10(mean_data[freq_mask]), linewidth=2, color='black')

        ax.set_title(titles[j])
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_xlabel('Frequency (Hz)')
    
    fig.suptitle(f'PSD comparisons : {ch_names[0]} to {ch_names[31]}')
    fig.tight_layout()

    pdf.savefig()
    plt.close()  

def plot_raw_psd_2sess(ch_names, raw_psd_subset, raw_psd_subset2, frequencies, title, date1, date2, pdf):
    
    mean_data1 = np.mean(raw_psd_subset, axis=0)
    mean_data2 = np.mean(raw_psd_subset2, axis=0)
    
    diff_data = np.log10(mean_data1)-np.log10(mean_data2)
    
    fig, ax = plt.subplots(2, sharex=True)

    #plot only the mean
    ax[0].plot(frequencies, 10 * np.log10(mean_data1), linewidth=2, color='blue')
    ax[0].plot(frequencies, 10 * np.log10(mean_data2), linewidth=2, color='red')
    #plot the difference
    diff_data =  (10 * np.log10(mean_data1)) - (10 * np.log10(mean_data2))
    ax[1].plot(frequencies, diff_data, linewidth=2, color='violet')
    ax[1].axhline(0, color='black', linestyle='--')
    if not title:
        ax[0].set_title(f'{date1} / {date2}')
    else:
        ax[0].set_title(f'{title} : {date1} / {date2}')
    ax[1].set_title('Difference')
    ax[0].set_ylabel('PSD (dB/Hz)')
    ax[1].set_ylabel('PSD (dB/Hz)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_xlim([1, 99])
    ax[1].set_xlim([1, 99])

    fig.tight_layout()
    pdf.savefig()
    plt.close()
    
      
#%% parameters

monk = input('which monkey ? :')
task = input('which task ? :')

#create a folder with today date 
save_dir = '/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/PSD_analysis/'
date = datetime.now().strftime("%d%m%y") #date of the day
date_fold = os.path.join(save_dir, date)
os.makedirs(date_fold, exist_ok=True)
print(f"All files will be save in : {date_fold}")

#PP list
path=f'/Volumes/Marion DATA/PREDYCT data/Ephys/{monk}/{task}/PP/'
file_list = glob.glob(path+'/'+"*pp.fif")
date_list = np.unique(np.array([x[-16:-10] for x in file_list]))
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
pp_list = [d.strftime("%d%m%y") for d in sorted_dates]

#MNE list
path=f'/Volumes/Marion_data_2/PREDYCT data/Ephys/{monk}/{task}/MNE/'
path2=f'/Volumes/Marion_data_3/PREDYCT data/Ephys/{monk}/{task}/MNE/'

file_list = glob.glob(path+'/'+"*raw.fif")
date_list = np.unique(np.array([x[-17:-11] for x in file_list]))
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
date_list = [d.strftime("%d%m%y") for d in sorted_dates]

file_list2 = glob.glob(path2+'/'+"*raw.fif")
date_list2 = np.unique(np.array([x[-17:-11] for x in file_list2]))
FORMAT = '%d%m%y'
sorted_dates2 = sorted([datetime.strptime(d, FORMAT) for d in date_list2])
date_list2 = [d.strftime("%d%m%y") for d in sorted_dates2]

date_list=date_list+date_list2
FORMAT = '%d%m%y'
sorted_dates = sorted([datetime.strptime(d, FORMAT) for d in date_list])
mne_list = [d.strftime("%d%m%y") for d in sorted_dates]

#%% processing

def compare_psd_methods(pp, chnames, pdf):

    ch_indices = [pp.ch_names.index(ch_name) for ch_name in chnames]
    
    pp_psd, frequencies_pp = mne.time_frequency.psd_array_multitaper(pp.get_data()[ch_indices], sfreq=pp.info['sfreq'])
    pp_psd2, frequencies_pp2 = mne.time_frequency.psd_array_welch(pp.get_data()[ch_indices], sfreq=pp.info['sfreq'])
    
    titles = ['multitaper', 'welch']
        
    num_channels = len(chnames)
    num_channels_subset = 32

    pp_psd_data_list = [pp_psd[i] for i in range(num_channels)]
    pp_psd_data_list2 = [pp_psd2[i] for i in range(num_channels)]
    psd_data_lists = [pp_psd_data_list, pp_psd_data_list2]

    frequencies_data = [frequencies_pp, frequencies_pp2]  # Liste de fréquences pour chaque canal

    for i in range(0, num_channels, num_channels_subset):
        chnames_subset = chnames[i:i+num_channels_subset]
        psd_data_lists_subset = [psd_data_list[i:i+num_channels_subset] for psd_data_list in psd_data_lists]
        plot_psd_comparison_method(chnames_subset, psd_data_lists_subset, frequencies_data, titles, pdf)

def compare_psd_2by2(mne_list):
 
    #Define freqs of interest
    fmin=1
    fmax=100
    
    #Define freqs resolution
    resolution = 100
    
    #Define downsampling factor
    sfreq_downsampled = 1000
    
    num_list = list(range(0,len(mne_list)))
    
    for i in num_list :
        
        if i == len(mne_list)-1 :
            print('last sess')
        else :
            date1 =  mne_list[i]
            date2 = mne_list[i+1]
            
            #RAW data
            if i == 0 : #First iteration
                
                try :
                    raw1 = load_raw_crop(path,monk,date1)
                except :
                    raw1 = load_raw_crop(path2,monk,date1)
                
                try :
                    raw2 = load_raw_crop(path,monk,date2)
                except :
                    raw2 = load_raw_crop(path2,monk,date2)
                
                #Downsampling to 1kHz
                raw1_resampled = raw1.copy().resample(sfreq_downsampled, npad='auto')
                raw2_resampled = raw2.copy().resample(sfreq_downsampled, npad='auto')
                del raw1, raw2
                
                #Extract ch names indices (one time only)
                ch_indices = [raw1_resampled.ch_names.index(ch_name) for ch_name in chnames]
            
                #Compute PSD for all channels at once
                raw_psd1, freqs = mne.time_frequency.psd_array_multitaper(raw1_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                raw_psd2, freqs  = mne.time_frequency.psd_array_multitaper(raw2_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                del raw1_resampled, raw2_resampled
                
                step = max(1, len(freqs) // resolution)
                 
                raw_psd1 = raw_psd1[:,::step]
                raw_psd2 = raw_psd2[:,::step]
                freqs = freqs[::step]
            
            else :
                raw_psd1 = raw_psd2
                try :
                    raw2 = load_raw_crop(path,monk,date2)
                except:
                    raw2 = load_raw_crop(path2,monk,date2)
            
                #Downsampling to 1kHz
                raw2_resampled = raw2.copy().resample(sfreq_downsampled, npad='auto')
                del raw2
                
                #Compute PSD for all channels at once
                raw_psd2, freqs  = mne.time_frequency.psd_array_multitaper(raw2_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                del raw2_resampled
                
                step = max(1, len(freqs) // resolution)
        
                raw_psd2 = raw_psd2[:,::step]
                freqs = freqs[::step]
        
            # Plot PSD comparison with all FMA together
            title = [] #useful for the function because we use the same function than when we separate FMA
            
            with PdfPages(date_fold+f'/{monk}_rawPSD_2by2_updated.pdf') as pdf:
                
                plot_raw_psd_2sess(chnames, raw_psd1, raw_psd2, freqs, title, date1, date2, pdf)
                 
            del raw_psd1
            
            #Add new pages to existing PDF
            pdf1_path = date_fold+f'/{monk}_rawPSD_2by2.pdf'
            pdf2_path = date_fold+f'/{monk}_rawPSD_2by2_updated.pdf'
        
            pdf1 = PdfReader(pdf1_path)
            pdf2 = PdfReader(pdf2_path)
        
            output_pdf = PdfWriter()
        
            for page_number in range(len(pdf1.pages)):
                output_pdf.add_page(pdf1.pages[page_number])
        
            for page_number in range(len(pdf2.pages)):
                output_pdf.add_page(pdf2.pages[page_number])
        
            with open(date_fold+f'/{monk}_rawPSD_2by2.pdf', "wb") as output_file:
                output_pdf.write(output_file)
    
    #save last date in text file
    text_file = open("/Users/marionducret/Desktop/PREDYCT/Analyses/Ephys/last_date.txt", "w")
    text_file.write(f'{date2}')
    text_file.close()

# PSD heatmap
 
def psd_heatmap(mne_list, chnames):

    #Define freqs of interest
    fmin=1
    fmax=100
    
    #Define freqs resolution
    resolution = 100
    
    #Define downsampling factor
    sfreq_downsampled = 1000
    
    psd_df=[]
    
    for date in mne_list :
        
        try :
            raw = load_raw_crop(path,monk,date)
        except :
            raw = load_raw_crop(path2,monk,date)
    
        #Downsampling to 1kHz
     
        raw_resampled = raw.copy().resample(sfreq_downsampled, npad='auto')
        
        del raw
        
        #Extract ch names and indices (one time only)
        ch_indices = [raw_resampled.ch_names.index(ch_name) for ch_name in chnames]
    
        #Compute PSD for all channels at once
        raw_psd, freqs = mne.time_frequency.psd_array_multitaper(raw_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
        
        del raw_resampled
        
        step = max(1, len(freqs) // resolution)
         
        raw_psd = raw_psd[:,::step]
        freqs = freqs[::step]
        mean_data = np.mean(raw_psd, axis=0)
        diff = np.log10(mean_data)
    
        psd_df.append(diff)
              
    psd_df_final = np.vstack(psd_df)
    
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD.pkl', 'wb') as file:
        pickle.dump(psd_df_final, file)
        
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD.pkl', 'rb') as file:
        psd_df_final = pickle.load(file)
        
    with PdfPages(date_fold+f'/{monk}_PSD_heatmap.pdf') as pdf:
        fig,ax=plt.subplots(1,1)
        img=ax.imshow(psd_df_final, cmap='coolwarm', aspect='auto')
        fig.colorbar(img)
        ax.set_title('PSD difference across sessions (dB/Hz)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim(1,50)
        ax.set_ylabel('sessions')
        pdf.savefig()
        plt.close()

# PSD comparison 2by2 heatmap differences

def psd_heatmap_2by2(mne_list, chnames):
    
    #Define freqs of interest
    fmin=1
    fmax=100
    
    #Define freqs resolution
    resolution = 100
    
    #Define downsampling factor
    sfreq_downsampled = 1000
    
    num_list = list(range(0,len(mne_list)))
    diff_df=[]
    
    for i in num_list :
        
        if i == len(mne_list)-1 :
            print('last sess')
        else :
            date1 =  mne_list[i]
            date2 = mne_list[i+1]
            
            #RAW data
            if i == 0 : #First iteration
            
                try :
                    raw1 = load_raw_crop(path,monk,date1)
                except :
                    raw1 = load_raw_crop(path2,monk,date1)
                
                try :
                    raw2 = load_raw_crop(path,monk,date2)
                except :
                    raw2 = load_raw_crop(path2,monk,date2)
                
    
                #Downsampling to 1kHz
                raw1_resampled = raw1.copy().resample(sfreq_downsampled, npad='auto')
                raw2_resampled = raw2.copy().resample(sfreq_downsampled, npad='auto')
                del raw1, raw2
                
                #Extract ch names and indices (one time only)
                ch_indices = [raw1_resampled.ch_names.index(ch_name) for ch_name in chnames]
            
                #Compute PSD for all channels at once
                raw_psd1, freqs = mne.time_frequency.psd_array_multitaper(raw1_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                raw_psd2, freqs  = mne.time_frequency.psd_array_multitaper(raw2_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                del raw1_resampled, raw2_resampled
                
                step = max(1, len(freqs) // resolution)
                 
                raw_psd1 = raw_psd1[:,::step]
                raw_psd2 = raw_psd2[:,::step]
                freqs = freqs[::step]
                mean_data1 = np.mean(raw_psd1, axis=0)
                mean_data2 = np.mean(raw_psd2, axis=0)
                diff = np.log10(mean_data1)-np.log10(mean_data2)
            
            else :
                
                raw_psd1 = raw_psd2
                try :
                    raw2 = load_raw_crop(path,monk,date2)
                except:
                    raw2 = load_raw_crop(path2,monk,date2)
    
                #Downsampling to 1kHz
                raw2_resampled = raw2.copy().resample(sfreq_downsampled, npad='auto')
                del raw2
                
                #Compute PSD for all channels at once
                raw_psd2, freqs  = mne.time_frequency.psd_array_multitaper(raw2_resampled.get_data()[ch_indices], sfreq=sfreq_downsampled, fmin=fmin, fmax=fmax)
                del raw2_resampled
                
                step = max(1, len(freqs) // resolution)
        
                raw_psd2 = raw_psd2[:,::step]
                freqs = freqs[::step]
                mean_data1 = np.mean(raw_psd1, axis=0)
                mean_data2 = np.mean(raw_psd2, axis=0)
                diff = np.log10(mean_data1)-np.log10(mean_data2)
                
            diff_df.append(diff)
            
    diff_df_final = np.vstack(diff_df)
    
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD_diff.pkl', 'wb') as file:
        pickle.dump(diff_df_final, file)
        
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD_diff.pkl', 'rb') as file:
        diff_df_final = pickle.load(file)
    
    with PdfPages(date_fold+f'{monk}_{task}_heatmap_2by2.pdf') as pdf:
        fig,ax=plt.subplots(1,1)
        img=ax.imshow(diff_df_final, cmap='coolwarm', aspect='auto')
        fig.colorbar(img)
        ax.set_title('PSD difference across sessions (dB/Hz)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('sessions')
        pdf.savefig()
        plt.close()

def psd_df(monk, task, mne_list):
    
    sfreq = 1000
    psd_data = []
    
    for date in mne_list:
        try:
            raw = load_and_resample_raw(path, monk, date, sfreq)
            psds, freqs = compute_psd_with_welch(raw, fmin=1, fmax=150, n_fft=1024, sfreq=sfreq)
            psd_data.append(psds.mean(axis=0))
        except:
            raw = load_and_resample_raw(path2, monk, date, sfreq)
            psds, freqs = compute_psd_with_welch(raw, fmin=1, fmax=150, n_fft=1024, sfreq=sfreq)
            psd_data.append(psds.mean(axis=0))
          
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_PSD_df.pkl', 'wb') as file:
         pickle.dump(psd_data, file) 
    
    with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_{task}_PSD_freqs.pkl', 'wb') as file:
         pickle.dump(freqs, file)
         
    del psd_data
     
def signal_loss(psd_data, freqs):

    fm = FOOOF()#initialize FOOOF model
    aperiodic_offsets = []
    snr_values = []
    
    #Process each session to extract the aperiodic offset
    for y, psd in enumerate(psd_data):
        print(y)
        fm.fit(freqs, psd)
        aperiodic_params = fm.aperiodic_params_
        offset = aperiodic_params[0]
        aperiodic_offsets.append(offset)
        
        #SNR (rapport signal/noise)
        signal_power = np.sum(psd[(freqs >= 2.5) & (freqs <= 48)])
        noise_power = np.sum(psd[(freqs > 100)])
        snr = 10 * np.log10(signal_power / noise_power)
        snr_values.append(snr)
    
    signal_loss_df = pd.DataFrame({'session': np.arange(1, len(psd_data)+1), 'offset': aperiodic_offsets, 'snr': snr_values})
    
    #fit a linear regression model to the offsets and SNR
    X = signal_loss_df[['session']]
    y_offset = signal_loss_df['offset']
    y_snr = signal_loss_df['snr']
    
    model_offset = LinearRegression()
    model_offset.fit(X, y_offset)
    slope_offset = model_offset.coef_[0]
    intercept_offset = model_offset.intercept_
    
    model_snr = LinearRegression()
    model_snr.fit(X, y_snr)
    slope_snr = model_snr.coef_[0]
    intercept_snr = model_snr.intercept_
    
    #predict offsets for visualization
    signal_loss_df['predicted_offset'] = model_offset.predict(X)
    signal_loss_df['predicted_snr'] = model_snr.predict(X)
    
    #stats
    slope_offset, intercept_offset, r_value_offset, p_value_offset, std_err_offset = stats.linregress(signal_loss_df['session'], signal_loss_df['offset'])
    slope_snr, intercept_snr, r_value_snr, p_value_snr, std_err_snr = stats.linregress(signal_loss_df['session'], signal_loss_df['snr'])
    
    #save slopes 
    with open(f'/Users/marionducret/Desktop/PREDYCT figures matrix/{monk}_signal_loss.pkl', 'wb') as file:
        pickle.dump({'slope_offset': slope_offset, 'slope_snr': slope_snr}, file)
    
    #plot the offsets, the SNR and the regression lines
    with PdfPages(date_fold+f'/{monk}_{task}_signal_loss.pdf') as pdf:
        plt.figure(figsize=(10, 6))
        plt.scatter(signal_loss_df['session'], signal_loss_df['offset'], label='Aperiodic Offset', color='blue')
        plt.plot(signal_loss_df['session'], signal_loss_df['predicted_offset'], label='Regression Line (Offset)', color='red')
        plt.xlabel('Session')
        plt.ylabel('Aperiodic Offset')
        plt.title('Progressive Signal Loss Across Sessions (Offset)')
        plt.legend()
        plt.annotate(f'Slope: {slope_offset:.4f}\np-value: {p_value_offset:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
        plt.show()
        pdf.savefig()
        plt.close()
    
        plt.figure(figsize=(10, 6))
        plt.scatter(signal_loss_df['session'], signal_loss_df['snr'], label='SNR', color='green')
        plt.plot(signal_loss_df['session'], signal_loss_df['predicted_snr'], label='Regression Line (SNR)', color='orange')
        plt.xlabel('Session')
        plt.ylabel('SNR')
        plt.title('Progressive Signal Loss Across Sessions (SNR)')
        plt.legend()
        plt.annotate(f'Slope: {slope_snr:.4f}\np-value: {p_value_snr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
        plt.show()
        pdf.savefig()
        plt.close()
        
#%%execution

with PdfPages(date_fold+f'/{monk}_PSD_comparisons_method.pdf') as pdf:

    for date in pp_list :
        
        _ , pp = load_pp_and_events2(monk, date, task)
        
        compare_psd_methods(pp, chnames, pdf)

compare_psd_2by2()
psd_heatmap(mne_list, chnames)
psd_heatmap_2by2(mne_list, chnames)
psd_df(monk, task)

# signal loss

with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD_df.pkl', 'rb') as file:
     psd_data = pickle.load(file)
   
psd_data = [arr for arr in psd_data if not np.isnan(arr).any()]
  
with open(f'/Users/marionducret/Desktop/PREDYCT_dataframes/{monk}_PSD_freqs.pkl', 'rb') as file:
     freqs = pickle.load(file)

signal_loss()
