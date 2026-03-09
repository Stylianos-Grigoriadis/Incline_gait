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

# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot Data\Pilot IMU placement\Daktila'
directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot Data\Pilot IMU placement\Fterna'
# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Pilot Data\Pilot IMU placement\Knimi'

os.chdir(directory)
ID = os.path.basename(directory)
# print(ID)


data_0_degree = pl.read_csv("0.csv", skip_rows=8, has_header=False, separator=",")
data_12_degree = pl.read_csv("12.csv", skip_rows=8, has_header=False, separator=",")
data_25_degree = pl.read_csv("25.csv", skip_rows=8, has_header=False, separator=",")

# Keep only first 6 columns
data_0_degree = data_0_degree.select(data_0_degree.columns[:6])
data_12_degree = data_12_degree.select(data_12_degree.columns[:6])
data_25_degree = data_25_degree.select(data_25_degree.columns[:6])

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
Acc_x_time = data[:, 0].to_numpy()
Acc_x = data[:, 1].to_numpy()
Acc_y_time = data[:, 2].to_numpy()
Acc_y = data[:, 3].to_numpy()
Acc_z_time = data[:, 4].to_numpy()
Acc_z = data[:, 5].to_numpy()


# Remove nan from IMU data
valid_idx = ~np.isnan(Acc_x_time)
Acc_x_time = Acc_x_time[valid_idx]
Acc_x = Acc_x[valid_idx]
Acc_y_time = Acc_y_time[valid_idx]
Acc_y = Acc_y[valid_idx]
Acc_z_time = Acc_z_time[valid_idx]
Acc_z = Acc_z[valid_idx]

# Filtering IMU low pass
Acc_x = lib.Butterworth(fs_imu, 20, Acc_x)
Acc_y = lib.Butterworth(fs_imu, 20, Acc_y)
Acc_z = lib.Butterworth(fs_imu, 20, Acc_z)

plt.plot(Acc_x, label='Acc_x')
plt.plot(Acc_y, label='Acc_y')
plt.plot(Acc_z, label='Acc_z')
plt.legend()
plt.show()


# Calculate the sum of squares for peak finding
SS_acc = Acc_x**2 + Acc_y**2 + Acc_z**2

print("Peaks IMU")
peak_times_IMU, peak_amplitude_IMU = lib.interactive_find_peaks_with_sliders(
    SS_acc,
    Acc_x_time,
    distance_init=int(fs_imu/2),
    height_init=5,
    distance_range=(1, fs_imu),
    height_range=(0, np.max(SS_acc))
)
Peaks_IMU = pd.DataFrame({"peak_times_IMU": peak_times_IMU, "peak_amplitude_IMU": peak_amplitude_IMU})