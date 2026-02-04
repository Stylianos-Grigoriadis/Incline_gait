import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.widgets import Slider, Button

directory = r'C:\Users\Stylianos Grigoriadi\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot\Pilot Anestis\twelvehalf_%_incline_2026_01_05_T_14_11_24'

os.chdir(directory)
Gastr_Med = pd.read_csv(r'KFORCEEMG20014_F0_C1_5E_B5_3E_91.csv', header=None)
Vastus_Lat = pd.read_csv(r'KFORCEEMG20207_F5_BA_D2_4E_23_B3.csv', header=None)
ECG = pd.read_csv(r'KFORCEEMG20604_FE_3C_98_9A_3E_D5.csv', header=None)
IMU = pd.read_csv(r'KFORCESens20254_C7_30_5F_BF_B4_0F.csv', header=None)

ECG = lib.prepare_ecg_df(ECG)
Gastr_Med = lib.prepare_emg_df(Gastr_Med)
Vastus_Lat = lib.prepare_emg_df(Vastus_Lat)
IMU = lib.prepare_imu_df(IMU)

print(IMU.columns)
fs = 1000
fs_IMU = 500

# lib.FFT_fast(np.asarray(IMU['Acc_X']), fs_IMU)
# lib.FFT_fast(np.asarray(IMU['Acc_Y']), fs_IMU)
# lib.FFT_fast(np.asarray(IMU['Acc_Z']), fs_IMU)



# First filtering
ECG['ECG'] = lib.butter_bandpass_filtfilt(ECG['ECG'], fs, low=0.5, high=45, order=4, plot=False)
Gastr_Med['EMG'] = lib.butter_bandpass_filtfilt(Gastr_Med['EMG'], fs, low=20, high=450, order=4, plot=False)
Vastus_Lat['EMG'] = lib.butter_bandpass_filtfilt(Vastus_Lat['EMG'], fs, low=20, high=450, order=4, plot=False)

# Notch filtering
ECG['ECG'] = lib.notch_filter_with_plots(ECG['ECG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)
Gastr_Med['EMG'] = lib.notch_filter_with_plots(Gastr_Med['EMG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)
Vastus_Lat['EMG'] = lib.notch_filter_with_plots(Vastus_Lat['EMG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)

# Make the time series absolute
# ECG['ECG'] = abs(ECG['ECG'])
Gastr_Med['EMG'] = abs(Gastr_Med['EMG'])
Vastus_Lat['EMG'] = abs(Vastus_Lat['EMG'])

# Linear envelope
# ECG['ECG Linear Envelope'] = lib.emg_linear_envelope(ECG['ECG'], fs, cutoff=12, order=4, plot=False)
Gastr_Med['EMG Linear Envelope'] = lib.emg_linear_envelope(Gastr_Med['EMG'], fs, cutoff=12, order=4, plot=False)
Vastus_Lat['EMG Linear Envelope'] = lib.emg_linear_envelope(Vastus_Lat['EMG'], fs, cutoff=12, order=4, plot=False)


# Find peaks for ECG
new_peak_times_ECG, new_peak_amplitude_EC = lib.interactive_find_peaks_with_sliders(
    ECG['ECG'],
    ECG['Time'],
    distance_init=400,
    height_init=0.02,
    distance_range=(1, 1000),
    height_range=(0.001, 1.0)
)

ECG_intervals = np.diff(new_peak_times_ECG)
df_ECG_intervals = pd.DataFrame({"Time Intervals": ECG_intervals})
df_ECG_intervals.to_excel(r'ECG_time_intervals_incl_12.5_deg.xlsx')

# Find peaks for Gastrocnemius Med
# new_peak_times_Gastr_Med, new_peak_amplitude_Gastr_Med = lib.interactive_find_peaks_with_sliders(
#     Gastr_Med['EMG Linear Envelope'],
#     Gastr_Med['Time'],
#     distance_init=600,
#     height_init=0.05,
#     distance_range=(1, 1000),
#     height_range=(0.001, 1.0)
# )
# Gastr_Med_intervals = np.diff(new_peak_times_Gastr_Med)
# df_Gastr_Med_intervals = pd.DataFrame({"Time Intervals": Gastr_Med_intervals})
# df_Gastr_Med_intervals.to_excel(r'Gastr_Med_time_intervals_incl_0_deg.xlsx')

# Find peaks for Vastus Lateralis
# new_peak_times_Vastus_Lat, new_peak_amplitude_Vastus_Lat = lib.interactive_find_peaks_with_sliders(
#     Vastus_Lat['EMG Linear Envelope'],
#     Vastus_Lat['Time'],
#     distance_init=600,
#     height_init=0.05,
#     distance_range=(1, 1000),
#     height_range=(0.001, 1.0)
# )
# Vastus_Lat_intervals = np.diff(new_peak_times_Vastus_Lat)
# df_Vastus_Lat_intervals = pd.DataFrame({"Time Intervals": Vastus_Lat_intervals})
# df_Vastus_Lat_intervals.to_excel(r'Vastus_Lat_time_intervals_incl_0_deg.xlsx')

