import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

""" Sensor 1 --> ECG
    Sensor 2 --> Gastr
    Sensor 3 --> Quad
    Sensor 4 --> IMU
"""

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot\Pilot Delsys\Stylianos 05-02-2026'
os.chdir(directory)

data = pd.read_csv(r'Stylianos_25%_failed.csv', skiprows=8, header=None, low_memory=False)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.iloc[:, :12]

emg = pd.DataFrame({'Time_Gastr': data[0], 'Gastr': data[1], 'Time_ECG': data[2], 'ECG': data[3], 'Time_Quad': data[4], 'Quad': data[5],})
emg['ECG'] = np.asarray(emg['ECG']) * (-1)
imu = pd.DataFrame({'time_acc_x': data[6], 'acc_x': data[7], 'time_acc_y': data[8], 'acc_y': data[9], 'time_acc_z': data[10], 'acc_z': data[11],})
imu = imu.dropna(subset=['time_acc_x'])


fs_emg = 2148.1481
fs_imu = 370.3704

# Spectral analysis
# lib.FFT_fast(np.asarray(emg['ECG']),  fs_emg)
# lib.FFT_fast(np.asarray(emg['Gastr']),  fs_emg)
# lib.FFT_fast(np.asarray(emg['Quad']),  fs_emg)
# lib.FFT_fast(np.asarray(imu['acc_x']),  fs_imu)
# lib.FFT_fast(np.asarray(imu['acc_y']),  fs_imu)
# lib.FFT_fast(np.asarray(imu['acc_z']),  fs_imu)

# Filtering ECG - EMG bandpass
emg['ECG'] = lib.butter_bandpass_filtfilt(emg['ECG'], fs_emg, low=0.5, high=100, order=4, plot=False)
emg['Gastr'] = lib.butter_bandpass_filtfilt(emg['Gastr'], fs_emg, low=20, high=450, order=4, plot=False)
emg['Quad'] = lib.butter_bandpass_filtfilt(emg['Quad'], fs_emg, low=20, high=450, order=4, plot=False)

# Filtering ECG notch
emg['ECG'] = lib.notch_filter_with_plots(emg['ECG'], fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)

# Make the EMG time series absolute
emg['Gastr'] = abs(emg['Gastr'])
emg['Quad'] = abs(emg['Quad'])

# Linear envelope
emg['Gastr Linear Envelope'] = lib.emg_linear_envelope(emg['Gastr'], fs_emg, cutoff=12, order=4, plot=False)
emg['Quad Linear Envelope'] = lib.emg_linear_envelope(emg['Quad'], fs_emg, cutoff=12, order=4, plot=False)

peak_times_ECG, peak_amplitude_ECG = lib.interactive_find_peaks_with_sliders(
    emg['ECG'],
    emg['Time_ECG'],
    distance_init=400,
    height_init=0.02,
    distance_range=(1, fs_emg),
    height_range=(emg['ECG'].min(), emg['ECG'].max())
)