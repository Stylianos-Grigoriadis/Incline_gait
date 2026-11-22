"""Review quadriceps EMG peaks for one participant and export all three trials."""

from pathlib import Path
import numpy as np
import pandas as pd
from lib import butter_bandpass_filtfilt
from lib import test_interactive_find_peaks_with_sliders_low_pass


ID = "P11"
data_directory = Path(
    r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data"
)
fs_emg = 2148.1481
height_init = None  # Use the 90th percentile of the initially filtered signal.
autofill = True
autofill_empty_strides = True  # True uses IMU-guided filling in place of regular autofill.


def main():
    input_directory = data_directory / "Valid Data" / ID
    trials = [("0", "0.csv"), ("12.5", "12.csv"), ("25", "25.csv")]
    peak_columns = {}
    imu_workbook = pd.DataFrame()
    for folder in ("Peak IMU data", "Peak Data per device/Peak IMU"):
        imu_path = data_directory / folder / f"{ID}_Peaks.xlsx"
        if imu_path.is_file():
            imu_workbook = pd.read_excel(imu_path)
            print(f"Using IMU reference from {imu_path}")
            break

    for inclination, filename in trials:
        print(f"Reviewing {ID} quadriceps EMG at {inclination}°. Close the plot to continue.")
        quad = pd.read_csv(
            input_directory / filename, skiprows=8, header=None, usecols=[4, 5],
        ).apply(pd.to_numeric, errors="coerce").dropna()
        time = quad[4].to_numpy()
        imu_peaks = pd.to_numeric(
            imu_workbook.get(f"IMU_Peak_Time_{inclination}_deg", pd.Series(dtype=float)),
            errors="coerce",
        )
        imu_peaks = imu_peaks[imu_peaks.between(time[0], time[-1])].to_numpy()
        if not imu_peaks.size:
            print(f"No IMU reference peaks within the quadriceps recording at {inclination}°.")
        signal = quad[5].to_numpy()
        filtered = butter_bandpass_filtfilt(
            signal, fs_emg, low=20, high=450, order=4, plot=False,
        )
        rectified = np.abs(filtered)
        peak_times, peak_amplitudes, _ = test_interactive_find_peaks_with_sliders_low_pass(
            rectified,
            time,
            fs=fs_emg,
            distance_init=int(fs_emg) / fs_emg,
            height_init=height_init,
            distance_range=(1 / fs_emg, 1.0),
            # height_range=(0, np.max(rectified)),
            autofill=autofill,
            reference_peak_times=imu_peaks,
            autofill_empty_strides=autofill_empty_strides,
        )

        # Series let trials with different peak counts share one worksheet.
        peak_columns[f"Quadriceps_EMG_Peak_Amplitude_{inclination}_deg"] = pd.Series(
            peak_amplitudes, dtype=float,
        )
        peak_columns[f"Quadriceps_EMG_Peak_Time_{inclination}_deg"] = pd.Series(
            peak_times, dtype=float,
        )

    output_directory = data_directory / "Peak quad data"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{ID}_Peaks.xlsx"
    pd.DataFrame(peak_columns).to_excel(output_path, index=False, engine="openpyxl")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
