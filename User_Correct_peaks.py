from pathlib import Path
import lib
import numpy as np
import pandas as pd
import polars as pl


# Select one participant, inclination (0, 12.5, or 25), and signal.
ID = "P7"
inclination = 25
Variable = "ECG"  # "ECG", "EMG" (gastrocnemius only), or "IMU"

fs_emg = 2148.1481
fs_imu = 370.3704
reverse_ECG = False

data_directory = Path(
    r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data"
)

inclination = str(inclination)
if inclination == "0.0":
    inclination = "0"
elif inclination == "25.0":
    inclination = "25"

trial_files = {"0": "0.csv", "12.5": "12.csv", "25": "25.csv"}
Variable = Variable.upper()
if inclination not in trial_files:
    raise ValueError("inclination must be 0, 12.5, or 25.")
if Variable not in ["ECG", "EMG", "IMU"]:
    raise ValueError("Variable must be ECG, EMG, or IMU.")

input_path = data_directory / "Valid Data" / ID / trial_files[inclination]
data = pl.read_csv(input_path, skip_rows=8, has_header=False, separator=",")
data = data.select(data.columns[:18])
data = (
    data.with_columns(pl.all().cast(pl.Utf8).str.strip_chars())
    .with_columns(pl.all().cast(pl.Float64, strict=False))
)

print(f"Processing {ID}, {inclination} degrees, {Variable}")
print("Adjust the sliders and edit peaks in the graph, then close it to save.")
print("Select a span: Delete/Backspace removes peaks; Space adds a peak.")

# Use the same preprocessing and interactive functions as User_Find_peaks.py.
if Variable == "ECG":
    time = data[:, 2].to_numpy()
    values = data[:, 3].to_numpy()
    if reverse_ECG:
        values = -values
    values = lib.butter_bandpass_filtfilt(
        values, fs_emg, low=0.5, high=250, order=4, plot=False
    )
    values = lib.notch_filter_with_plots(
        values, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False
    )
    peak_times, peak_amplitudes = lib.interactive_find_peaks_with_sliders(
        values, time,
        distance_init=int(fs_emg / 2), height_init=0.04,
        distance_range=(1, fs_emg), height_range=(0, np.max(values)),
    )
    signal_name = "ECG"

elif Variable == "EMG":
    time = data[:, 0].to_numpy()
    values = data[:, 1].to_numpy()
    values = lib.butter_bandpass_filtfilt(
        values, fs_emg, low=20, high=450, order=4, plot=False
    )
    values = abs(values)
    peak_times, peak_amplitudes, low_cutoff = lib.interactive_find_peaks_with_sliders_low_pass(
        values, time, fs=fs_emg,
        distance_init=int(fs_emg / 2), height_init=0.02,
        distance_range=(1, fs_emg), height_range=(0, np.max(values)),
    )
    signal_name = "Gastrocnemius_EMG"

else:
    time = data[:, 12].to_numpy()
    values = data[:, 13].to_numpy()
    valid_idx = ~np.isnan(time)
    time = time[valid_idx]
    values = values[valid_idx]
    values = lib.Butterworth(fs_imu, 20, values)
    peak_times, peak_amplitudes = lib.interactive_find_peaks_with_sliders(
        values, time,
        distance_init=int(fs_imu / 2), height_init=5,
        distance_range=(1, fs_imu), height_range=(0, np.max(values)),
    )
    signal_name = "IMU"

# Save peak amplitudes and timestamps in seconds, matching the original export.
# No first/last exclusions are applied here.
output_directory = data_directory / "Corrections of Peaks"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / f"{ID}_{signal_name}_{inclination}_deg_Peaks.xlsx"
pd.DataFrame({
    f"{signal_name}_Peak_Amplitude_{inclination}_deg": peak_amplitudes,
    f"{signal_name}_Peak_Time_{inclination}_deg": peak_times,
}).to_excel(
    output_path, index=False, engine="openpyxl"
)
print(f"Saved {len(peak_times)} peaks (amplitudes and timestamps) to {output_path}")
