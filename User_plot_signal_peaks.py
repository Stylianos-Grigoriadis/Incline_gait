"""Plot saved ECG, EMG, and IMU peaks over a selected set of strides. Press Ctrl+S to save it"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from lib import Butterworth, butter_bandpass_filtfilt, notch_filter_with_plots
from User_find_peaks_emg_joint import FS, TRIALS, read_peaks
from User_peak_interval_results import peak_directory


PARTICIPANT = "P7"
INCLINE = "25"  # "0", "12.5", or "25"
FIRST_STRIDE = 50  # 1 starts at the first saved IMU peak.
STRIDE_COUNT = 8
REVERSE_ECG = False  # Match the ECG review setting for this participant.
EMG_CUTOFF_HZ = 12  # Match the envelope setting used during peak review.
OUTPUT_PNG = "Sensor_placement_and_peak_detection.png"
DATA_DIR = peak_directory.parent
SETUP_DIR = DATA_DIR.parent / "Figures" / "Setup"
SAVE_DIR = SETUP_DIR
AVATAR_IMAGES = (
    ("Front", SETUP_DIR / "Avatar_full_body_anatomy_devices_front.png"),
    ("Rear", SETUP_DIR / "Avatar_full_body_anatomy_devices_rear.png"),
)


CHANNELS = (
    ("ECG", "ECG", "Peak ECG", "Peak ECG data", (2, 3)),
    ("Gastrocnemius medialis EMG", "Gastrocnemius_EMG", "Peak Gast", "Peak gastrocnemius data", (0, 1)),
    ("Vastus lateralis EMG", "Quadriceps_EMG", "Peak Quad", "Peak quad data", (4, 5)),
    ("IMU shank angular velocity", "IMU", "Peak IMU", "Peak IMU data", (12, 13)),
)


def process_signal(signal, channel):
    if channel == "IMU":
        return Butterworth(370.3704, 20, signal)
    if channel == "ECG":
        filtered = butter_bandpass_filtfilt(signal, FS, low=0.5, high=250, order=4)
        return notch_filter_with_plots(filtered, FS, f_notch=50.0, bandwidth=2.5, plot=False)
    filtered = butter_bandpass_filtfilt(signal, FS, low=20, high=450, order=4)
    envelope_filter = butter(4, EMG_CUTOFF_HZ, fs=FS, output="sos")
    return sosfiltfilt(envelope_filter, np.abs(filtered))


def build_figure(data, participant, incline, first_stride, stride_count):
    if incline not in TRIALS:
        raise ValueError(f"Choose an incline from {', '.join(TRIALS)}.")
    if first_stride < 1 or stride_count < 1:
        raise ValueError("FIRST_STRIDE and STRIDE_COUNT must be positive.")
    peaks = {
        channel: read_peaks(data, participant, folder, staging, f"{channel}_Peak_Time_{incline}_deg")
        for _, channel, folder, staging, _ in CHANNELS
    }
    missing = [channel for channel, times in peaks.items() if not times.size]
    if missing:
        raise ValueError(f"No saved peak times for {participant} at {incline}° in {', '.join(missing)}.")
    imu = peaks["IMU"]
    first = first_stride - 1
    if first + stride_count >= len(imu):
        raise ValueError(f"The trial has {len(imu) - 1} strides. Choose an earlier FIRST_STRIDE.")
    boundaries = imu[first:first + stride_count + 1]
    start, end = boundaries[0], boundaries[-1]

    recording = data / "Valid Data" / participant / TRIALS[incline]
    columns = [column for *_, pair in CHANNELS for column in pair]
    raw = pd.read_csv(recording, skiprows=8, header=None, usecols=columns, dtype=str)
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    grid = fig.add_gridspec(5, 3, width_ratios=(1, 1, 4), height_ratios=(1, 1, 1, 1, 0.18))
    for column, (view, image_path) in enumerate(AVATAR_IMAGES):
        image_axis = fig.add_subplot(grid[:4, column])
        image_axis.imshow(plt.imread(image_path))
        image_axis.set_title(view, fontsize=10)
        image_axis.set_axis_off()
    axes = [fig.add_subplot(grid[0, 2])]
    axes.extend(fig.add_subplot(grid[row, 2], sharex=axes[0]) for row in range(1, 4))
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    for axis, (label, channel, _, _, pair) in zip(axes, CHANNELS):
        time, signal = raw[list(pair)].apply(pd.to_numeric, errors="coerce").dropna().to_numpy().T
        if time.size < 2 or time[0] > start or time[-1] < end:
            raise ValueError(f"The selected stride window is outside the {label} recording.")
        if channel == "ECG" and REVERSE_ECG:
            signal = -signal
        processed = process_signal(signal, channel)
        axis.vlines(imu, 0, 1, transform=axis.get_xaxis_transform(),
                    color="#a6adb5", linestyle="--", linewidth=0.8, zorder=0, label="Stride boundary")
        if channel.endswith("_EMG"):
            axis.plot(time, signal, color="#416383", alpha=0.30,
                      linewidth=0.7, label="Raw signal")
        axis.plot(time, processed, color="#143e5e", linewidth=1.15, label="Filtered signal / EMG envelope")
        recorded_peaks = peaks[channel][(peaks[channel] >= time[0]) & (peaks[channel] <= time[-1])]
        axis.scatter(recorded_peaks, np.interp(recorded_peaks, time, processed), color="#b72742",
                     s=20, zorder=5, label="Saved peak")
        axis.set_title(label, loc="left", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set(xlabel="Time (s)", xlim=(start, end))
    legend_axis = fig.add_subplot(grid[4, 2])
    legend_axis.set_axis_off()
    handles, labels = axes[1].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncols=4, fontsize=8, frameon=False)
    fig.suptitle("Sensor Placement and Peak Detection During Walking", fontsize=15, fontweight="semibold")
    return fig


def save_current_view(event):
    if event.key != "ctrl+s":
        return
    directory = Path(SAVE_DIR).expanduser()
    path = directory / OUTPUT_PNG
    try:
        directory.mkdir(parents=True, exist_ok=True)
        event.canvas.figure.savefig(path, dpi=600)
    except OSError as error:
        print(f"Could not save {path}: {error}")
    else:
        print(f"Saved {path.resolve()}")


def main():
    try:
        figure = build_figure(Path(DATA_DIR), PARTICIPANT.upper(), str(INCLINE), FIRST_STRIDE, STRIDE_COUNT)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    plt.rcParams["keymap.save"] = [key for key in plt.rcParams["keymap.save"] if key != "ctrl+s"]
    figure.canvas.mpl_connect("key_press_event", save_current_view)
    print("Press Ctrl+S in the plot window to save the current view at 600 DPI.")
    plt.show()


if __name__ == "__main__":
    main()
