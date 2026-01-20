import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot\Pilot Anestis\Flat_2026_01_05_T_14_42_40'

os.chdir(directory)
Gastr_Med = pd.read_csv(r'KFORCEEMG20014_F0_C1_5E_B5_3E_91.csv', header=None)
Vastus_Lat = pd.read_csv(r'KFORCEEMG20207_F5_BA_D2_4E_23_B3.csv', header=None)
ECG = pd.read_csv(r'KFORCEEMG20604_FE_3C_98_9A_3E_D5.csv', header=None)
IMU = pd.read_csv(r'KFORCESens20254_C7_30_5F_BF_B4_0F.csv', header=None)

ECG = lib.prepare_ecg_df(ECG)
Gastr_Med = lib.prepare_emg_df(Gastr_Med)
Vastus_Lat = lib.prepare_emg_df(Vastus_Lat)
IMU = lib.prepare_imu_df(IMU)

print(Vastus_Lat)



# plt.plot(IMU['Time'], IMU['Acc_X'], label='X')
# plt.plot(IMU['Time'], IMU['Acc_Y'], label='Y')
# plt.plot(IMU['Time'], IMU['Acc_Z'], label='Z')
# plt.legend()
# plt.show()

# plt.plot(Gastr_Med['Time'], Gastr_Med['EMG'], label='X')
# plt.show()
# plt.plot(Vastus_Lat['Time'], Vastus_Lat['EMG'], label='X')
# plt.show()
# plt.plot(ECG['Time'], ECG['ECG'], label='X')
# plt.show()

fs = 1000
# lib.FFT_fast(np.asarray(ECG['ECG']), fs)
# lib.FFT_fast(np.asarray(Gastr_Med['EMG']), fs)
# lib.FFT_fast(np.asarray(Vastus_Lat['EMG']), fs)

# First filtering
ECG['ECG'] = lib.butter_bandpass_filtfilt(ECG['ECG'], fs, low=7, high=40, order=4, plot=False)
Gastr_Med['EMG'] = lib.butter_bandpass_filtfilt(Gastr_Med['EMG'], fs, low=7, high=40, order=4, plot=False)
Vastus_Lat['EMG'] = lib.butter_bandpass_filtfilt(Vastus_Lat['EMG'], fs, low=7, high=40, order=4, plot=False)

# Notch filtering
ECG['ECG'] = lib.notch_filter_with_plots(ECG['ECG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)
Gastr_Med['EMG'] = lib.notch_filter_with_plots(Gastr_Med['EMG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)
Vastus_Lat['EMG'] = lib.notch_filter_with_plots(Vastus_Lat['EMG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)

# Make the time series absolut
ECG['ECG'] = abs(ECG['ECG'])
Gastr_Med['EMG'] = abs(Gastr_Med['EMG'])
Vastus_Lat['EMG'] = abs(Vastus_Lat['EMG'])

# Linear envelope
ECG['ECG'] = lib.emg_linear_envelope(ECG['ECG'], fs, cutoff=10, order=4, plot=True)
Gastr_Med['EMG'] = lib.emg_linear_envelope(Gastr_Med['EMG'], fs, cutoff=10, order=4, plot=True)
Vastus_Lat['EMG'] = lib.emg_linear_envelope(Vastus_Lat['EMG'], fs, cutoff=10, order=4, plot=True)

