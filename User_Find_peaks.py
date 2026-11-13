import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import polars as pl
import plotly.graph_objects as go


""" Sensor 1 --> ECG
    Sensor 2 --> Gastr
    Sensor 3 --> Quad
    Sensor 4 --> IMU
"""
fs_emg = 2148.1481
fs_imu = 370.3704
custom_filtering_to_EMG = True
reverse_ECG = True

directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Valid Data\P23'
os.chdir(directory)
ID = os.path.basename(directory)
print(ID)

data_0_degree = pl.read_csv("0.csv", skip_rows=8, has_header=False, separator=",")
data_12_degree = pl.read_csv("12.csv", skip_rows=8, has_header=False, separator=",")
data_25_degree = pl.read_csv("25.csv", skip_rows=8, has_header=False, separator=",")

# Keep only first 12 columns
data_0_degree = data_0_degree.select(data_0_degree.columns[:18])
data_12_degree = data_12_degree.select(data_12_degree.columns[:18])
data_25_degree = data_25_degree.select(data_25_degree.columns[:18])
print(data_12_degree.columns)

# Make it float
data_0_degree = (data_0_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_12_degree = (data_12_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_25_degree = (data_25_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))

# Finish and close each interactive plot to continue to the next analysis.
all_peak_columns = {}
for trial, inclination, data in [
    (0, "0", data_0_degree),
    (12, "12.5", data_12_degree),
    (25, "25", data_25_degree),
]:
    print(f"\nProcessing {ID}, inclination {inclination} degrees")

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
    gyro_x_time = data[:, 12].to_numpy()
    gyro_x = data[:, 13].to_numpy()
    gyro_y_time = data[:, 14].to_numpy()
    gyro_y = data[:, 15].to_numpy()
    gyro_z_time = data[:,16].to_numpy()
    gyro_z = data[:,17].to_numpy()

    if reverse_ECG == True:
        ECG = ECG*-1

    # Remove nan from IMU data
    valid_idx = ~np.isnan(gyro_x_time)
    gyro_x_time = gyro_x_time[valid_idx]
    gyro_x = gyro_x[valid_idx]

    # Filtering IMU low pass
    gyro_x = lib.Butterworth(fs_imu, 20, gyro_x)

    # Filtering ECG - EMG bandpass
    ECG = lib.butter_bandpass_filtfilt(ECG, fs_emg, low=0.5, high=250, order=4, plot=False)
    Gastr_EMG = lib.butter_bandpass_filtfilt(Gastr_EMG, fs_emg, low=20, high=450, order=4, plot=False)
    Quad_EMG = lib.butter_bandpass_filtfilt(Quad_EMG, fs_emg, low=20, high=450, order=4, plot=False)

    # Filtering ECG notch
    ECG = lib.notch_filter_with_plots(ECG, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)

    # Make the EMG time series absolute
    Gastr_EMG = abs(Gastr_EMG)
    Quad_EMG = abs(Quad_EMG)


    # Linear envelope
    # Gastr_EMG_linear_envelope = lib.emg_linear_envelope(Gastr_EMG, fs_emg, cutoff=12, order=4, plot=False)
    # Quad_EMG_linear_envelope = lib.emg_linear_envelope(Quad_EMG, fs_emg, cutoff=12, order=4, plot=False)




    ############################
    #### Find peaks for ECG ####
    ############################
    print("Peaks ECG")
    peak_times_ECG, peak_amplitude_ECG = lib.interactive_find_peaks_with_sliders(
        ECG,
        ECG_time,
        distance_init=int(fs_emg/2),
        height_init=0.04,
        distance_range=(1, fs_emg),
        height_range=(0, np.max(ECG))
    )
    Peaks_ECG = pd.DataFrame({"peak_times_ECG": peak_times_ECG, "peak_amplitude_ECG": peak_amplitude_ECG})

    ##################################
    #### Find peaks for Gastr_EMG ####
    ##################################
    print("Peaks Gastr")
    peak_times_Gast, peak_amplitude_Gast, low_cut_off_Quad = lib.interactive_find_peaks_with_sliders_low_pass(
        Gastr_EMG,
        Gastr_EMG_time,
        fs=fs_emg,
        distance_init=int(fs_emg/2),
        height_init=0.02,
        distance_range=(1, fs_emg),
        height_range=(0, np.max(Gastr_EMG))
    )
    Peaks_Gast = pd.DataFrame({"peak_times_Quad": peak_times_Gast, "peak_amplitude_Quad": peak_amplitude_Gast})

    #################################
    #### Find peaks for Quad_EMG ####
    #################################
    print("Peaks Quad")
    peak_times_Quad, peak_amplitude_Quad, low_cut_off_Quad = lib.interactive_find_peaks_with_sliders_low_pass(
        Quad_EMG,
        Quad_EMG_time,
        fs=fs_emg,
        distance_init=int(fs_emg/2),
        height_init=0.02,
        distance_range=(1, fs_emg),
        height_range=(0, np.max(Quad_EMG))
    )
    Peaks_Quad = pd.DataFrame({"peak_times_Quad": peak_times_Quad, "peak_amplitude_Quad": peak_amplitude_Quad})

    ###############################
    #### Find peaks for SS_acc ####
    ###############################
    print("Peaks IMU")
    peak_times_IMU, peak_amplitude_IMU = lib.interactive_find_peaks_with_sliders(
        gyro_x,
        gyro_x_time,
        distance_init=int(fs_imu/2),
        height_init=5,
        distance_range=(1, fs_imu),
        height_range=(0, np.max(gyro_x))
    )
    Peaks_IMU = pd.DataFrame({"peak_times_IMU": peak_times_IMU, "peak_amplitude_IMU": peak_amplitude_IMU})

    # Align by peak number, independently for each signal and condition.
    for signal_name, amplitudes, times in [
        ("ECG", peak_amplitude_ECG, peak_times_ECG),
        ("Gastrocnemius_EMG", peak_amplitude_Gast, peak_times_Gast),
        ("Quadriceps_EMG", peak_amplitude_Quad, peak_times_Quad),
        ("IMU", peak_amplitude_IMU, peak_times_IMU),
    ]:
        all_peak_columns[f"{signal_name}_Peak_Amplitude_{inclination}_deg"] = pd.Series(amplitudes, dtype=float)
        all_peak_columns[f"{signal_name}_Peak_Time_{inclination}_deg"] = pd.Series(times, dtype=float)

# Series align on their zero-based peak index; shorter series export as blank cells.
peak_results = pd.DataFrame(all_peak_columns)
output_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(directory))), "Peaks Data")
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, f"{ID}_Peaks.xlsx")
peak_results.to_excel(output_path, index=False, na_rep="", engine="openpyxl")
print(f"Saved {len(peak_results.columns)} columns to {output_path}")

