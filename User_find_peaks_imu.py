"""Review IMU peaks for one participant and export all three trials."""

from pathlib import Path
import numpy as np
import pandas as pd
from lib import Butterworth, interactive_find_peaks_with_sliders


ID = "P3"
data_directory = Path(
    r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data"
)
fs_imu = 370.3704
height_init = 5


def main():
    input_directory = data_directory / "Valid Data" / ID
    trials = [("0", "0.csv"), ("12.5", "12.csv"), ("25", "25.csv")]
    peak_columns = {}

    for inclination, filename in trials:
        print(f"Reviewing {ID} IMU at {inclination}°. Close the plot to continue.")
        imu = pd.read_csv(
            input_directory / filename, skiprows=8, header=None, usecols=[12, 13],
            dtype=str,
        ).apply(pd.to_numeric, errors="coerce").dropna(subset=[12])
        time = imu[12].to_numpy()
        signal = imu[13].to_numpy()
        filtered = Butterworth(fs_imu, 20, signal)
        peak_times, peak_amplitudes = interactive_find_peaks_with_sliders(
            filtered,
            time,
            distance_init=int(fs_imu),
            height_init=height_init,
            distance_range=(1, fs_imu),
            height_range=(0, np.max(filtered)),
        )

        # Series let trials with different peak counts share one worksheet.
        peak_columns[f"IMU_Peak_Amplitude_{inclination}_deg"] = pd.Series(
            peak_amplitudes, dtype=float,
        )
        peak_columns[f"IMU_Peak_Time_{inclination}_deg"] = pd.Series(
            peak_times, dtype=float,
        )

    output_directory = data_directory / "Peak IMU data"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{ID}_Peaks.xlsx"
    pd.DataFrame(peak_columns).to_excel(output_path, index=False, engine="openpyxl")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
