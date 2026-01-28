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

# Make the time series absolute
ECG['ECG'] = abs(ECG['ECG'])
Gastr_Med['EMG'] = abs(Gastr_Med['EMG'])
Vastus_Lat['EMG'] = abs(Vastus_Lat['EMG'])

# Linear envelope
ECG['ECG Linear Envelope'] = lib.emg_linear_envelope(ECG['ECG'], fs, cutoff=2, order=4, plot=False)
Gastr_Med['EMG Linear Envelope'] = lib.emg_linear_envelope(Gastr_Med['EMG'], fs, cutoff=10, order=4, plot=False)
Vastus_Lat['EMG Linear Envelope'] = lib.emg_linear_envelope(Vastus_Lat['EMG'], fs, cutoff=10, order=4, plot=False)
print(ECG.columns)
print(len(ECG['ECG Linear Envelope']))

# Find peaks
peak_indices_ECG, peak_amplitude_ECG = lib.peaks(ECG['ECG Linear Envelope'], 400, 0.005)
peak_indices_Gastr_Med, peak_amplitude_Gastr_Med = lib.peaks(Gastr_Med['EMG Linear Envelope'], 400, 0.005)
peak_indices_Vastus_Lat, peak_amplitude_Vastus_Lat = lib.peaks(Vastus_Lat['EMG Linear Envelope'], 400, 0.005)

peak_times_ECG = ECG['Time'].iloc[peak_indices_ECG].values
peak_times_Gastr_Med = ECG['Time'].iloc[peak_indices_Gastr_Med].values
peak_times_Vastus_Lat = ECG['Time'].iloc[peak_indices_Vastus_Lat].values

RR_intervals = np.diff(peak_times_ECG)
Gastr_Med_intervals = np.diff(peak_times_Gastr_Med)
Vastus_Lat_intervals = np.diff(peak_times_Vastus_Lat)




def plot_ecg_envelope_with_peaks_interactive(
    ECG,
    peak_times_ECG,
    peak_amplitude_ECG,
    downsample=50
):
    """
    Plot ECG linear envelope with detected peaks and derived RR intervals.

    Interaction (TOP plot only):
    - Move mouse
    - SPACE     → add black vertical line + print time
    - BACKSPACE → remove last added black vertical line
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ===============================
    # Compute RR intervals internally
    # ===============================
    RR_intervals = np.diff(peak_times_ECG)

    # ===============================
    # Create figure and axes
    # ===============================
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        gridspec_kw={'height_ratios': [4, 1]}
    )

    # ===============================
    # TOP: ECG Linear Envelope
    # ===============================
    ax1.plot(
        ECG['Time'][::downsample],
        ECG['ECG Linear Envelope'][::downsample],
        color='royalblue',
        linewidth=2,
        label='ECG Linear Envelope'
    )

    ax1.scatter(
        peak_times_ECG,
        peak_amplitude_ECG,
        color='red',
        zorder=3,
        label='Peaks'
    )

    for t in peak_times_ECG:
        ax1.axvline(
            x=t,
            color='red',
            linestyle='--',
            alpha=0.3,
            linewidth=1
        )

    ax1.set_ylabel('Amplitude')
    ax1.set_title('ECG Linear Envelope with Detected Peaks')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ===============================
    # BOTTOM: RR Intervals
    # ===============================
    ax2.plot(
        peak_times_ECG[1:],
        RR_intervals,
        color='darkorange',
        linewidth=1.5
    )

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Δ Time (s)')
    ax2.set_title('Inter-peak Intervals')
    ax2.grid(alpha=0.3)

    # ===============================
    # INTERACTION STATE
    # ===============================
    cursor_x = {"value": None}
    marked_times = []
    marked_lines = []  # store Line2D objects

    # ===============================
    # CALLBACKS
    # ===============================
    def on_mouse_move(event):
        if event.inaxes == ax1 and event.xdata is not None:
            cursor_x["value"] = event.xdata

    def on_key_press(event):
        # ---- ADD LINE ----
        if event.key == ' ' and cursor_x["value"] is not None:
            t = cursor_x["value"]

            print(f"Marked time = {t:.3f} s")
            marked_times.append(t)

            line = ax1.axvline(
                x=t,
                color='black',
                linewidth=1.5,
                alpha=0.9
            )
            marked_lines.append(line)

            fig.canvas.draw_idle()

        # ---- REMOVE LAST LINE ----
        elif event.key == 'backspace' and marked_lines:
            last_line = marked_lines.pop()
            last_time = marked_times.pop()

            last_line.remove()
            fig.canvas.draw_idle()

            print(f"Removed time = {last_time:.3f} s")

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.tight_layout()
    plt.show()

    return marked_times



plot_ecg_envelope_with_peaks_interactive(
    ECG,
    peak_times_ECG,
    peak_amplitude_ECG,
    downsample=50
)


