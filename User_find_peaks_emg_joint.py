"""Review and edit vastus lateralis and gastrocnemius peaks on one timeline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from lib import butter_bandpass_filtfilt
from User_peak_interval_results import gait_match_bounds, peak_directory


PARTICIPANT = "P18"
INCLINE = "0"  # Choose "0", "12.5", or "25".

FS = 2148.1481
TRIALS = {"0": "0.csv", "12.5": "12.csv", "25": "25.csv"}
MUSCLES = (
    ("Vastus lateralis", "Quadriceps_EMG", "Peak Quad", "Peak quad data", (4, 5)),
    ("Gastrocnemius medialis", "Gastrocnemius_EMG", "Peak Gast", "Peak gastrocnemius data", (0, 1)),
)


def read_peaks(data, participant, analysis_folder, review_folder, column):
    candidates = [
        data / "Peak Data per device" / analysis_folder / f"{participant}_Peaks.xlsx",
        data / review_folder / f"{participant}_Peaks.xlsx",
    ]
    sources = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    correction = correction_path(data, participant, analysis_folder)
    if analysis_folder != "Peak IMU" and correction.is_file():
        sources.insert(0, correction)
    for path in sources:
        workbook = pd.read_excel(path)
        if column in workbook:
            print(f"Loaded {column} from {path}")
            return np.sort(pd.to_numeric(workbook[column], errors="coerce").dropna().to_numpy())
    return np.array([])


def correction_path(data, participant, folder):
    return data / "corrections" / f"{participant}_{folder.replace(' ', '_')}.xlsx"


def save_peaks(path, prefix, incline, peaks, time, envelope):
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = pd.read_excel(path) if path.is_file() else pd.DataFrame()
    workbook = workbook.reindex(range(max(len(workbook), len(peaks))))
    workbook[f"{prefix}_Peak_Time_{incline}_deg"] = pd.Series(peaks)
    workbook[f"{prefix}_Peak_Amplitude_{incline}_deg"] = pd.Series(np.interp(peaks, time, envelope))
    workbook.to_excel(path, index=False, engine="openpyxl")


def review(data, participant, incline, cutoff):
    raw = pd.read_csv(
        data / "Valid Data" / participant / TRIALS[incline],
        skiprows=8, header=None, usecols=[0, 1, 4, 5],
    ).apply(pd.to_numeric, errors="coerce")
    imu = read_peaks(data, participant, "Peak IMU", "Peak IMU data", f"IMU_Peak_Time_{incline}_deg")
    if len(imu) < 2:
        raise SystemExit(f"At least two saved IMU peaks are required for {participant} at {incline}°.")

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 3, 1]},
    )
    fig.subplots_adjust(bottom=0.15, hspace=0.28)
    fig.suptitle(f"{participant} · {incline}° · Joint EMG peak review")
    fig.text(0.08, 0.065, "Drag a span in an EMG panel. Space adds its envelope maximum. Delete removes peaks in the span.")
    message = fig.text(0.08, 0.025, "")
    filter_sos = butter(4, cutoff, fs=FS, output="sos")
    channels = []
    selection = None

    for axis, (label, prefix, folder, staging, columns) in zip(axes, MUSCLES):
        time, signal = raw[list(columns)].dropna().to_numpy().T
        filtered = butter_bandpass_filtfilt(signal, FS, low=20, high=450, order=4)
        envelope = sosfiltfilt(filter_sos, np.abs(filtered))
        peaks = read_peaks(data, participant, folder, staging, f"{prefix}_Peak_Time_{incline}_deg")
        axis.plot(time, envelope, color="#345d95", lw=1, label="EMG envelope")
        axis.vlines(imu, 0, 1, transform=axis.get_xaxis_transform(),
                    colors="teal", linestyles="--", linewidth=1, alpha=0.35, label="IMU peaks")
        markers = axis.scatter([], [], color="#c3324c", zorder=4, label="Saved or edited EMG peaks")
        axis.set_ylabel(label)
        axis.legend(loc="upper right", fontsize=8)
        channels.append({
            "time": time, "envelope": envelope, "peaks": peaks, "markers": markers,
            "prefix": prefix, "folder": folder, "changed": False,
        })

    comparison = axes[2]
    comparison.set_xlabel("Time (s)")
    comparison.set_yticks([1, 0], ["IMU → VL", "IMU → GM"])
    comparison.set_ylim(-0.45, 1.45)
    empty_markers = comparison.scatter([], [], marker="x", color="#ba293e", label="No paired peak")
    multiple_markers = comparison.scatter([], [], marker="^", color="#a75b13", label="Multiple paired peaks")
    comparison.legend(loc="upper right", fontsize=8)
    comparison.grid(axis="y", alpha=0.2)

    def redraw():
        for channel in channels:
            peaks = channel["peaks"]
            channel["markers"].set_offsets(np.c_[peaks, np.interp(peaks, channel["time"], channel["envelope"])])
        vl, gm = (channel["peaks"] for channel in channels)
        empty, multiple, counts = [], [], []
        midpoints = (imu[:-1] + imu[1:]) / 2
        for lane, paired in zip((1, 0), (vl, gm)):
            edges, _, _ = gait_match_bounds(imu, paired)
            per_cycle = np.diff(edges)
            empty.extend((time, lane) for time in midpoints[per_cycle == 0])
            multiple.extend((time, lane) for time in midpoints[per_cycle > 1])
            counts.append(f"{sum(per_cycle == 0)} empty / {sum(per_cycle > 1)} multiple")
        empty_markers.set_offsets(np.asarray(empty).reshape(-1, 2))
        multiple_markers.set_offsets(np.asarray(multiple).reshape(-1, 2))
        comparison.set_title("   |   ".join(
            f"{label}: {count}" for label, count in zip(("IMU–VL", "IMU–GM"), counts)
        ), loc="left", fontsize=9)
        fig.canvas.draw_idle()

    def select(index, start, end):
        nonlocal selection
        selection = index, start, end
        message.set_text(f"Selected {MUSCLES[index][0]} from {start:.3f} to {end:.3f} s")
        fig.canvas.draw_idle()

    selectors = [
        SpanSelector(axis, lambda start, end, index=index: select(index, start, end),
                     "horizontal", interactive=True, props={"alpha": 0.2, "facecolor": "#4683c4"})
        for index, axis in enumerate(axes[:2])
    ]

    def edit(event):
        if selection is None or event.key not in (" ", "delete", "backspace"):
            return
        index, start, end = selection
        channel = channels[index]
        peaks = channel["peaks"]
        if event.key == " ":
            samples = np.flatnonzero((channel["time"] >= start) & (channel["time"] <= end))
            if not samples.size:
                return
            best = samples[np.argmax(channel["envelope"][samples])]
            updated = np.union1d(peaks, channel["time"][best])
        else:
            updated = peaks[(peaks < start) | (peaks > end)]
        if not np.array_equal(peaks, updated):
            channel["peaks"] = updated
            channel["changed"] = True
            message.set_text("Unsaved edits")
            redraw()

    def save(_event):
        try:
            for channel in channels:
                if channel["changed"]:
                    output = correction_path(data, participant, channel["folder"])
                    save_peaks(output, channel["prefix"], incline, channel["peaks"], channel["time"], channel["envelope"])
                    channel["changed"] = False
                    print(f"Saved {output}")
        except OSError as error:
            message.set_text(f"Save failed: {error}")
        else:
            message.set_text("Saved changed channels to Data/corrections.")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", edit)
    save_button = Button(fig.add_axes([0.78, 0.035, 0.16, 0.045]), "Save edited peaks")
    save_button.on_clicked(save)
    redraw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default=PARTICIPANT)
    parser.add_argument("--incline", choices=TRIALS, default=INCLINE)
    parser.add_argument("--cutoff", type=float, default=12)
    parser.add_argument("--data-dir", type=Path, default=peak_directory.parent)
    args = parser.parse_args()
    review(args.data_dir, args.participant.upper(), args.incline, args.cutoff)


if __name__ == "__main__":
    main()
