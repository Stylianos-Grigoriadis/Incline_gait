import lib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot\Pilot Delsys'

os.chdir(directory)

ECG = pd.read_csv(r'Trial_2.csv', skiprows=8, header=None)
print(ECG)
ECG = ECG.iloc[:, :2]
ECG.columns = ['Time', 'ECG']
print(ECG)

fs = 2148.1481
# lib.FFT_fast(np.asarray(ECG['ECG']),  fs)
ECG['ECG'] = np.asarray(ECG['ECG']) * (-1)
ECG['ECG'] = lib.butter_bandpass_filtfilt(ECG['ECG'], fs, low=0.5, high=100, order=4, plot=False)
ECG['ECG'] = lib.notch_filter_with_plots(ECG['ECG'], fs, f_notch=50.0, bandwidth=2.5, plot=False)


dt = ECG['Time'].diff()
print(dt)
x = np.linspace(0, len(dt), len(dt))
plt.scatter(x,dt)
plt.show()
plt.plot(ECG['Time'], ECG['ECG'])
plt.show()




