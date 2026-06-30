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
fs_emg = 2148.1481
fs_imu = 370.3704
custom_filtering_to_EMG = True

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Valid Data\P1'
os.chdir(directory)
ID = os.path.basename(directory)
print(ID)

data_0_degree = pl.read_csv("0.csv", skip_rows=8, has_header=False, separator=",")
data_12_degree = pl.read_csv("12.csv", skip_rows=8, has_header=False, separator=",")
data_25_degree = pl.read_csv("25.csv", skip_rows=8, has_header=False, separator=",")

# Keep only first 12 columns
data_0_degree = data_0_degree.select(data_0_degree.columns[:12])
data_12_degree = data_12_degree.select(data_12_degree.columns[:12])
data_25_degree = data_25_degree.select(data_25_degree.columns[:12])

# Make it float
data_0_degree = (data_0_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_12_degree = (data_12_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_25_degree = (data_25_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))

trial = 25
if trial == 0:
    data = data_0_degree
elif trial == 12:
    data = data_12_degree
elif trial == 25:
    data = data_25_degree


##################################################
################# DATA ANALYSIS ##################
##################################################
Gastr_EMG_time = data[:, 0].to_numpy()
Gastr_EMG = data[:, 1].to_numpy()
ECG_time = data[:, 2].to_numpy()
ECG = data[:, 3].to_numpy()
Quad_EMG_time = data[:, 4].to_numpy()
Quad_EMG = data[:, 5].to_numpy()
Acc_x_time = data[:, 6].to_numpy()
Acc_x = data[:, 7].to_numpy()
Acc_y_time = data[:, 8].to_numpy()
Acc_y = data[:, 9].to_numpy()
Acc_z_time = data[:,10].to_numpy()
Acc_z = data[:,11].to_numpy()

# Remove nan from IMU data
valid_idx = ~np.isnan(Acc_x_time)
Acc_x_time = Acc_x_time[valid_idx]
Acc_x = Acc_x[valid_idx]
Acc_y_time = Acc_y_time[valid_idx]
Acc_y = Acc_y[valid_idx]
Acc_z_time = Acc_z_time[valid_idx]
Acc_z = Acc_z[valid_idx]


# Spectral analysis
# lib.FFT_fast(ECG,  fs_emg)
# lib.FFT_fast(Gastr_EMG,  fs_emg)
# lib.FFT_fast(Quad_EMG,  fs_emg)
# lib.FFT_fast(Acc_x,  fs_imu)
# lib.FFT_fast(Acc_y,  fs_imu)
# lib.FFT_fast(Acc_z,  fs_imu)

# Residuals Analysis for IMU
# lib.residual_analysis(Acc_x, fs_imu, 10, 150)
# lib.residual_analysis(Acc_y, fs_imu, 10, 150)
# lib.residual_analysis(Acc_z, fs_imu, 10, 150)

# Filtering IMU low pass
Acc_x = lib.Butterworth(fs_imu, 20, Acc_x)
Acc_y = lib.Butterworth(fs_imu, 20, Acc_y)
Acc_z = lib.Butterworth(fs_imu, 20, Acc_z)

# Calculate the sum of squares for peak finding
SS_acc = Acc_x**2 + Acc_y**2 + Acc_z**2

# Use of Teager_Kaiser
out = lib.Muscle_activity_based_on_Teager_Kaiser(
    signal=Gastr_EMG,
    fs=fs_emg,
    band_pass=(30, 300),
    lowpass=20,
    baseline_window=0.2,
    baseline_percent=20,
    h=15,
    min_activation_duration=0.025,
    plot=True,
    step_for_plot=1
)
# Filtering ECG - EMG bandpass
ECG = lib.butter_bandpass_filtfilt(ECG, fs_emg, low=0.5, high=250, order=4, plot=False)
Gastr_EMG = lib.butter_bandpass_filtfilt(Gastr_EMG, fs_emg, low=20, high=450, order=4, plot=False)
Quad_EMG = lib.butter_bandpass_filtfilt(Quad_EMG, fs_emg, low=20, high=450, order=4, plot=False)

# Filtering ECG notch
ECG = lib.notch_filter_with_plots(ECG, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)
# ECG = lib.notch_filter_with_plots(ECG, fs_emg, f_notch=148.0, bandwidth=2.5, plot=True)
# ECG = lib.notch_filter_with_plots(ECG, fs_emg, f_notch=300.0, bandwidth=2.5, plot=True)

# Make the EMG time series absolute
Gastr_EMG = abs(Gastr_EMG)
Quad_EMG = abs(Quad_EMG)


# Linear envelope
# Gastr_EMG_linear_envelope = lib.emg_linear_envelope(Gastr_EMG, fs_emg, cutoff=12, order=4, plot=False)
# Quad_EMG_linear_envelope = lib.emg_linear_envelope(Quad_EMG, fs_emg, cutoff=12, order=4, plot=False)





# Find peaks for ECG

# peak_amplitudes, peak_times, auc_values, cut_off_freq = lib.plot_emg_threshold_mountains(
#     data_series=Gastr_EMG,
#     time_series=Gastr_EMG_time,
#     sampling_freq=fs_emg,
#     cutoff_freq=12,
#     baseline_percentile=60,
#     peak_threshold_percentile=90,
#     min_duration=0.1,
#
# )
# print("Peaks ECG")
# peak_times_ECG, peak_amplitude_ECG = lib.interactive_find_peaks_with_sliders(
#     ECG,
#     ECG_time,
#     distance_init=int(fs_emg/2),
#     height_init=0.04,
#     distance_range=(1, fs_emg),
#     height_range=(0, np.max(ECG))
# )
# Peaks_ECG = pd.DataFrame({"peak_times_ECG": peak_times_ECG, "peak_amplitude_ECG": peak_amplitude_ECG})
#
# print("Peaks Gastr")
# peak_times_Gastr, peak_amplitude_Gastr, low_cut_off_Gastr = lib.interactive_find_peaks_with_sliders_low_pass(
#     Gastr_EMG,
#     Gastr_EMG_time,
#     fs=fs_emg,
#     distance_init=int(fs_emg/2),
#     height_init=0.02,
#     distance_range=(1, fs_emg),
#     height_range=(0, np.max(Gastr_EMG))
# )
# Peaks_Gastr = pd.DataFrame({"peak_times_Gastr": peak_times_Gastr, "peak_amplitude_Gastr": peak_amplitude_Gastr})
#
# print("Peaks Quad")
# peak_times_Quad, peak_amplitude_Quad, low_cut_off_Quad = lib.interactive_find_peaks_with_sliders_low_pass(
#     Quad_EMG,
#     Quad_EMG_time,
#     fs=fs_emg,
#     distance_init=int(fs_emg/2),
#     height_init=0.02,
#     distance_range=(1, fs_emg),
#     height_range=(0, np.max(Quad_EMG))
# )
# Peaks_Quad = pd.DataFrame({"peak_times_Quad": peak_times_Quad, "peak_amplitude_Quad": peak_amplitude_Quad})
#
# print("Peaks IMU")
# peak_times_IMU, peak_amplitude_IMU = lib.interactive_find_peaks_with_sliders(
#     SS_acc,
#     Acc_x_time,
#     distance_init=int(fs_imu/2),
#     height_init=5,
#     distance_range=(1, fs_imu),
#     height_range=(0, np.max(SS_acc))
# )
# Peaks_IMU = pd.DataFrame({"peak_times_IMU": peak_times_IMU, "peak_amplitude_IMU": peak_amplitude_IMU})
#
#
#
# base_dir = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Peaks Data'
# directory_save = os.path.join(base_dir, str(ID), str(trial))
# print(directory_save)
# os.chdir(directory_save)
#
# if Peaks_ECG is not None and not Peaks_ECG.empty:
#     Peaks_ECG.to_excel(f"Peaks_ECG_{trial}.xlsx")
#
# if Peaks_Gastr is not None and not Peaks_Gastr.empty:
#     if not custom_filtering_to_EMG:
#         Peaks_Gastr.to_excel(f"Peaks_Gastr_{trial}.xlsx")
#     else:
#         Peaks_Gastr.to_excel(f"Peaks_Gastr_{trial}_low_filtered_{low_cut_off_Gastr}.xlsx")
#
# if Peaks_Quad is not None and not Peaks_Quad.empty:
#     if not custom_filtering_to_EMG:
#         Peaks_Quad.to_excel(f"Peaks_Quad_{trial}.xlsx")
#     else:
#         Peaks_Quad.to_excel(f"Peaks_Quad_{trial}_low_filtered_{low_cut_off_Quad}.xlsx")
#
# if Peaks_IMU is not None and not Peaks_IMU.empty:
#     Peaks_IMU.to_excel(f"Peaks_IMU_{trial}.xlsx")
#
