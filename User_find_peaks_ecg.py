"""Review ECG peaks for one participant and export all three trials."""

from pathlib import Path
import numpy as np
import pandas as pd
from lib import butter_bandpass_filtfilt
from lib import interactive_find_peaks_with_sliders, notch_filter_with_plots


ID = "P21"
data_directory = Path(
    r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data"
)
fs_emg = 2148.1481
reverse_ECG = False
height_init = 0.04


def main():
    input_directory = data_directory / "Valid Data" / ID
    trials = [("0", "0.csv"), ("12.5", "12.csv"), ("25", "25.csv")]
    peak_columns = {}

    for inclination, filename in trials:
        print(f"Reviewing {ID} ECG at {inclination}°. Close the plot to continue.")
        ecg = pd.read_csv(
            input_directory / filename, skiprows=8, header=None, usecols=[2, 3],
        ).apply(pd.to_numeric, errors="coerce").dropna()
        time = ecg[2].to_numpy()
        signal = ecg[3].to_numpy()
        if reverse_ECG:
            signal = -signal
        filtered = butter_bandpass_filtfilt(
            signal, fs_emg, low=0.5, high=250, order=4, plot=False,
        )
        filtered = notch_filter_with_plots(
            filtered, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False,
        )
        peak_times, peak_amplitudes = interactive_find_peaks_with_sliders(
            filtered,
            time,
            distance_init=int(fs_emg / 2),
            height_init=height_init,
            distance_range=(1, fs_emg),
            height_range=(0, np.max(filtered)),
        )

        # Series let trials with different peak counts share one worksheet.
        peak_columns[f"ECG_Peak_Amplitude_{inclination}_deg"] = pd.Series(
            peak_amplitudes, dtype=float,
        )
        peak_columns[f"ECG_Peak_Time_{inclination}_deg"] = pd.Series(
            peak_times, dtype=float,
        )

    output_directory = data_directory / "Peak ECG data"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{ID}_Peaks.xlsx"
    pd.DataFrame(peak_columns).to_excel(output_path, index=False, engine="openpyxl")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
