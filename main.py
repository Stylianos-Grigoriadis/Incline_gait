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

print(ECG)

plt.plot(IMU['Time'], IMU['Acc_X'], label='X')
plt.plot(IMU['Time'], IMU['Acc_Y'], label='Y')
plt.plot(IMU['Time'], IMU['Acc_Z'], label='Z')
plt.legend()
plt.show()

lib.FFT(np.asarray(ECG['ECG']), 1000)
lib.FFT(np.asarray(ECG['ECG']), 1000)