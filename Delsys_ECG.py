import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import polars as pl

""" Sensor 1 --> ECG
    Sensor 2 --> Gastr
    Sensor 3 --> Quad
    Sensor 4 --> IMU
"""



directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Prequalified Data\P1'
os.chdir(directory)


data_0_degree = pl.read_csv("0.csv", skip_rows=8, has_header=False, infer_schema_length=1000)
data_12_degree = pl.read_csv("12.csv", skip_rows=8, has_header=False, infer_schema_length=1000)
data_25_degree = pl.read_csv("25.csv", skip_rows=8, has_header=False, infer_schema_length=1000)

print(data_0_degree)

# # Convert all columns to Float64 (similar to pd.to_numeric(errors='coerce'))
# data_0_degree = data_0_degree.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in data_0_degree.columns])
# data_12_degree = data_12_degree.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in data_12_degree.columns])
# data_25_degree = data_25_degree.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in data_25_degree.columns])

# Keep only first 12 columns
data_0_degree = data_0_degree.select(data_0_degree.columns[:12])
data_12_degree = data_12_degree.select(data_12_degree.columns[:12])
data_25_degree = data_25_degree.select(data_25_degree.columns[:12])
print(data_0_degree)

##################################################
############ DATA ANALYSIS 0 DEGREE ##############
##################################################
Gastr_EMG_time = data_0_degree[cols_0[0]].to_numpy()
Gastr_EMG      = data_0_degree[cols_0[1]].to_numpy()
ECG_time       = data_0_degree[cols_0[2]].to_numpy()
ECG            = data_0_degree[cols_0[3]].to_numpy()
Quad_EMG_time  = data_0_degree[cols_0[4]].to_numpy()
Quad_EMG       = data_0_degree[cols_0[5]].to_numpy()
Acc_x_time     = data_0_degree[cols_0[6]].to_numpy()
Acc_x          = data_0_degree[cols_0[7]].to_numpy()
Acc_y_time     = data_0_degree[cols_0[8]].to_numpy()
Acc_y          = data_0_degree[cols_0[9]].to_numpy()
Acc_z_time     = data_0_degree[cols_0[10]].to_numpy()
Acc_z          = data_0_degree[cols_0[11]].to_numpy()
print(Acc_z)
fs_emg = 2148.1481
fs_imu = 370.3704

# Spectral analysis
# lib.FFT_fast(ECG,  fs_emg)
# lib.FFT_fast(Gastr_EMG,  fs_emg)
# # lib.FFT_fast(Quad_EMG,  fs_emg)
# lib.FFT_fast(Acc_x,  fs_imu)
# lib.FFT_fast(Acc_y,  fs_imu)
# lib.FFT_fast(Acc_z,  fs_imu)

# Filtering Analysis for IMU
lib.residual_analysis(Acc_x, fs_imu, 10, 170)
lib.residual_analysis(Acc_y, fs_imu, 10, 170)
lib.residual_analysis(Acc_z, fs_imu, 10, 170)



# Filtering ECG - EMG bandpass
ECG = lib.butter_bandpass_filtfilt(ECG, fs_emg, low=0.5, high=100, order=4, plot=False)
Gastr_EMG = lib.butter_bandpass_filtfilt(Gastr_EMG, fs_emg, low=20, high=450, order=4, plot=False)
Quad_EMG = lib.butter_bandpass_filtfilt(Quad_EMG, fs_emg, low=20, high=450, order=4, plot=False)

# Filtering ECG notch
ECG = lib.notch_filter_with_plots(ECG, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)

# Make the EMG time series absolute
Gastr_EMG = abs(Gastr_EMG)
Quad_EMG = abs(Quad_EMG)

# Linear envelope
Gastr_EMG_linear_envelope = lib.emg_linear_envelope(Gastr_EMG, fs_emg, cutoff=12, order=4, plot=False)
Quad_EMG_linear_envelope = lib.emg_linear_envelope(Quad_EMG, fs_emg, cutoff=12, order=4, plot=False)

peak_times_ECG, peak_amplitude_ECG = lib.interactive_find_peaks_with_sliders(
    ECG,
    ECG_time,
    distance_init=400,
    height_init=0.02,
    distance_range=(1, fs_emg),
    height_range=(np.min(ECG), np.max(ECG))
)

peak_times_Gastr, peak_amplitude_Gastr = lib.interactive_find_peaks_with_sliders(
    Gastr_EMG_linear_envelope,
    Gastr_EMG_time,
    distance_init=400,
    height_init=0.02,
    distance_range=(1, fs_emg),
    height_range=(np.min(Gastr_EMG_linear_envelope), np.max(Gastr_EMG_linear_envelope))
)

peak_times_Quad, peak_amplitude_Quad = lib.interactive_find_peaks_with_sliders(
    Quad_EMG_linear_envelope,
    Quad_EMG_time,
    distance_init=400,
    height_init=0.02,
    distance_range=(1, fs_emg),
    height_range=(np.min(Quad_EMG_linear_envelope), np.max(Quad_EMG_linear_envelope))
)


