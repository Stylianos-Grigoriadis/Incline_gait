import polars as pl
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch
from matplotlib.widgets import Slider, TextBox
import os
import plotly.graph_objects as go


def emg_linear_envelope(signal, fs, cutoff=10, order=4, plot=False):
    """
    Compute and optionally plot the EMG linear envelope.

    Parameters
    ----------
    signal : array-like
        Rectified EMG signal
    fs : float
        Sampling frequency (Hz)
    cutoff : float
        Low-pass cutoff frequency (Hz), typically 3–6 Hz
    order : int
        Filter order
    plot : bool
        If True, plot original signal and linear envelope

    Returns
    -------
    envelope : np.ndarray
        EMG linear envelope
    """

    x = np.asarray(signal)

    nyq = fs / 2.0
    wn = cutoff / nyq

    b, a = butter(order, wn, btype='low')
    padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
    envelope = filtfilt(b, a, x, padtype='odd', padlen=padlen)

    if plot:
        t = np.arange(len(x)) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(x, label='Rectified EMG', color='black', alpha=0.4)
        plt.plot(envelope, label='Linear envelope', color='royalblue', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'EMG Linear Envelope (low-pass {cutoff} Hz)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return envelope

def notch_filter_with_plots(x, fs, f_notch=50.0, bandwidth=1.5, plot=False):
    """
    Apply notch filter and optionally plot time & frequency domain (2x2).

    Parameters
    ----------
    x : array-like
        Input signal
    fs : float
        Sampling frequency (Hz)
    f_notch : float
        Frequency to remove (Hz)
    Q : float
        Quality factor
    plot : bool
        If True, plot time & frequency domain (2x2)

    Returns
    -------
    y : np.ndarray
        Notch-filtered signal
    """

    x = np.asarray(x)

    # -------------------------
    # Notch filter
    # -------------------------
    Q = f_notch/bandwidth
    w0 = f_notch / (fs / 2)
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, x)

    # -------------------------
    # FFT helper
    # -------------------------
    def compute_fft(sig):
        freqs = fftfreq(len(sig), 1 / fs)
        mask = freqs > 0
        Y = fft(sig)
        psd = 2 * (np.abs(Y) / len(sig)) ** 2
        return freqs[mask], psd[mask]

    if plot:
        t = np.arange(len(x)) / fs

        f_x, a_x = compute_fft(x)
        f_y, a_y = compute_fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(12, 7))

        # ---- Time domain (original)
        axs[0, 0].plot(t, x, color='black', alpha=0.6)
        axs[0, 0].set_title('Time domain – Original')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Signal')

        # ---- Time domain (filtered)
        axs[0, 1].plot(t, y, color='royalblue', linewidth=1.2)
        axs[0, 1].set_title('Time domain – Notch filtered')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Signal')

        # ---- Frequency domain (original)
        axs[1, 0].plot(f_x, a_x, color='black', alpha=0.6)
        axs[1, 0].set_title('Frequency domain – Original')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Power')

        # ---- Frequency domain (filtered)
        axs[1, 1].plot(f_y, a_y, color='royalblue', linewidth=1.2)
        axs[1, 1].set_title(
            f'Frequency domain – Notch @ {f_notch} Hz (Q={round(Q,2)})'
        )
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Power')

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return y

def butter_bandpass_filtfilt(x, fs, low=0.01, high=0.30, order=4, plot=False):
    """
    Zero-phase Butterworth band-pass filter.

    If plot=False:
        → returns filtered signal immediately.

    If plot=True:
        → interactive sliders + numeric input boxes
        → returns filtered signal using final selected values.
    """

    x = np.array(x)
    t = np.arange(len(x)) / fs

    # --------------------------------------------------
    # FILTER FUNCTION
    # --------------------------------------------------
    def apply_filter(lowcut, highcut):
        nyq = fs / 2.0
        wn = [lowcut / nyq, highcut / nyq]

        if not (0 < wn[0] < wn[1] < 1):
            return None

        b, a = butter(order, wn, btype='band')
        padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
        return filtfilt(b, a, x, padtype='odd', padlen=padlen)

    # --------------------------------------------------
    # NO PLOT → SIMPLE FILTER
    # --------------------------------------------------
    if not plot:
        return apply_filter(low, high)

    # --------------------------------------------------
    # FFT FUNCTION
    # --------------------------------------------------
    def compute_fft(sig):
        freqs = fftfreq(len(sig), 1 / fs)
        mask = freqs > 0
        Y = fft(sig)
        psd = 2 * (np.abs(Y) / len(sig)) ** 2
        return freqs[mask], psd[mask]

    # Initial signal
    y = apply_filter(low, high)
    f_x, a_x = compute_fft(x)
    f_y, a_y = compute_fft(y)

    # --------------------------------------------------
    # FIGURE
    # --------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    plt.subplots_adjust(bottom=0.32)

    axs[0, 0].plot(t, x, color='black', alpha=0.6)
    axs[0, 0].set_title("Time – Original")

    line_time, = axs[0, 1].plot(t, y, color='royalblue')
    axs[0, 1].set_title("Time – Filtered")

    axs[1, 0].plot(f_x, a_x, color='black', alpha=0.6)
    axs[1, 0].set_title("Frequency – Original")

    line_freq, = axs[1, 1].plot(f_y, a_y, color='royalblue')
    axs[1, 1].set_title(f"Frequency – {low}-{high} Hz")

    for ax in axs.flat:
        ax.grid(True)

    # --------------------------------------------------
    # SLIDERS
    # --------------------------------------------------
    ax_low_slider = plt.axes([0.2, 0.18, 0.55, 0.03])
    ax_high_slider = plt.axes([0.2, 0.12, 0.55, 0.03])

    slider_low = Slider(
        ax_low_slider,
        "Low (Hz)",
        0.001,
        fs/2 - 0.01,
        valinit=low,
        valstep=0.001
    )

    slider_high = Slider(
        ax_high_slider,
        "High (Hz)",
        0.001,
        fs/2 - 0.001,
        valinit=high,
        valstep=0.001
    )

    # --------------------------------------------------
    # TEXT BOXES
    # --------------------------------------------------
    ax_low_box = plt.axes([0.80, 0.18, 0.10, 0.04])
    ax_high_box = plt.axes([0.80, 0.12, 0.10, 0.04])

    text_low = TextBox(ax_low_box, "", initial=str(low))
    text_high = TextBox(ax_high_box, "", initial=str(high))

    # --------------------------------------------------
    # UPDATE FUNCTION
    # --------------------------------------------------
    def update_filter(lowcut, highcut):
        if lowcut >= highcut:
            return

        y_new = apply_filter(lowcut, highcut)
        if y_new is None:
            return

        f_y_new, a_y_new = compute_fft(y_new)

        line_time.set_ydata(y_new)
        line_freq.set_ydata(a_y_new)
        axs[1, 1].set_title(f"Frequency – {lowcut:.3f}-{highcut:.3f} Hz")

        fig.canvas.draw_idle()

    # --------------------------------------------------
    # SLIDER CALLBACK
    # --------------------------------------------------
    def slider_update(val):
        lowcut = slider_low.val
        highcut = slider_high.val

        text_low.set_val(f"{lowcut:.3f}")
        text_high.set_val(f"{highcut:.3f}")

        update_filter(lowcut, highcut)

    slider_low.on_changed(slider_update)
    slider_high.on_changed(slider_update)

    # --------------------------------------------------
    # TEXTBOX CALLBACK
    # --------------------------------------------------
    def submit_low(text):
        try:
            value = float(text)
            slider_low.set_val(value)
        except ValueError:
            pass

    def submit_high(text):
        try:
            value = float(text)
            slider_high.set_val(value)
        except ValueError:
            pass

    text_low.on_submit(submit_low)
    text_high.on_submit(submit_high)

    plt.show()

    # --------------------------------------------------
    # RETURN FINAL FILTERED SIGNAL
    # --------------------------------------------------
    return apply_filter(slider_low.val, slider_high.val)

def Residual_analysis(var, cutoff_frequencies, fs, number_of_fit_points=10, plot=True, save=None):

    """
    Perform residual analysis for one variable.

    Parameters
    ----------
    var : array-like
        Original time series.

    cutoff_frequencies : list or array-like
        Cutoff frequencies to test.

    fs : float, default=1000
        Sampling frequency in Hz.

    number_of_fit_points : int, default=10
        Number of highest cutoff frequencies used for the linear fit.

    plot : bool, default=True
        If True, display the residual-analysis plot.

    save : tuple or None, default=None
        Tuple containing (name, directory). For example:
        ("participant_1.png", r"C:\\Results\\Residual analysis")
        If None, the plot is not saved.

    Returns
    -------
    Fc : float
        Tested cutoff frequency whose RMS residual is closest to the
        y-intercept of the fitted line.
    """

    var = np.asarray(var, dtype=float)

    # Convert the cutoff frequencies to a sorted NumPy array
    cutoff_frequencies = np.asarray(
        sorted(cutoff_frequencies),
        dtype=float
    )

    # Calculate the RMS residual for every cutoff frequency
    rms_residual = []

    for fc in cutoff_frequencies:

        var_filtered = Butterworth(
            fs,
            fc,
            var
        )

        residual = var - var_filtered

        rms_residual.append(
            np.sqrt(np.mean(residual ** 2))
        )

    rms_residual = np.asarray(rms_residual)

    # Use the highest cutoff frequencies for the linear fit
    fit_frequencies = cutoff_frequencies[-number_of_fit_points:]
    fit_residuals = rms_residual[-number_of_fit_points:]

    # Fit the straight line: y = slope*x + intercept
    slope, intercept = np.polyfit(
        fit_frequencies,
        fit_residuals,
        1
    )

    # Evaluate the fitted line across all tested cutoff frequencies
    fitted_line = np.polyval(
        [slope, intercept],
        cutoff_frequencies
    )

    # Find the tested residual closest to the y-intercept
    closest_index = np.argmin(
        np.abs(rms_residual - intercept)
    )

    Fc = cutoff_frequencies[closest_index]

    # Create the figure when it needs to be displayed or saved
    if plot or save is not None:

        fig, ax = plt.subplots(figsize=(8, 5))

        # RMS residual curve
        ax.plot(
            cutoff_frequencies,
            rms_residual,
            marker='o',
            color='#2F7FBF',
            label='RMS residual'
        )

        # Points used for the linear fit
        ax.scatter(
            fit_frequencies,
            fit_residuals,
            color='#F28C38',
            s=55,
            zorder=3,
            label=f'Last {number_of_fit_points} cutoff values'
        )

        # Extrapolated fitted line
        ax.plot(
            cutoff_frequencies,
            fitted_line,
            linestyle='--',
            color='#F28C38',
            linewidth=2,
            label='Extrapolated linear fit'
        )

        # Horizontal line at the y-intercept
        ax.axhline(
            y=intercept,
            color='black',
            linestyle='--',
            label=f'Intercept = {intercept:.4f}'
        )

        # Selected cutoff frequency
        ax.axvline(
            x=Fc,
            color='#FF1515',
            linestyle=':',
            linewidth=2,
            label=f'Fc = {Fc:g} Hz'
        )

        # Mark the selected point
        ax.scatter(
            Fc,
            rms_residual[closest_index],
            color='#FF1515',
            s=70,
            zorder=4
        )

        ax.set_xlabel('Cutoff frequency (Hz)')
        ax.set_ylabel('RMS residual')
        ax.set_title('Residual Analysis')
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        # save must have the form: (name, directory)
        if save is not None:

            if not isinstance(save, tuple) or len(save) != 2:
                raise ValueError(
                    "save must be None or a tuple: (name, directory)."
                )

            name, save_dir = save

            # Create the directory if it does not already exist
            os.makedirs(save_dir, exist_ok=True)

            # Add .png if the name does not contain an extension
            if not os.path.splitext(name)[1]:
                name += '.png'

            save_path = os.path.join(
                save_dir,
                name
            )

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight'
            )

            print(f"Plot saved to: {save_path}")

        if plot:
            plt.show()
        else:
            plt.close(fig)

    return Fc

def Butterworth(fs, fc, var):
    """ Parameter:
            fs:     sampling frequency
            fc:     cutoff frequency for example 30Hz
            var:    data series
    """

    b, a = signal.butter(N=2, Wn=fc, btype='low', fs=fs)
    return signal.filtfilt(b, a, var)

def FFT_fast(var, fs):
    dt = 1 / fs
    freqs = fftfreq(len(var), dt)
    mask = freqs > 0

    Y = fft(var)
    a = 2 * ((np.abs(Y) / len(var)) ** 2)
    f = freqs[mask]
    a = a[mask]

    plt.plot(f, a)
    plt.show()

    percA = np.cumsum(a) / np.sum(a) * 100

    index90 = np.argmin(np.abs(percA - 90))
    index95 = np.argmin(np.abs(percA - 95))
    index99 = np.argmin(np.abs(percA - 99))

    return f[index90], f[index95], f[index99]


""" Sensor 1 --> ECG
    Sensor 2 --> Gastr
    Sensor 3 --> Quad
    Sensor 4 --> IMU
"""
fs_emg = 2148.1481
fs_imu = 370.3704

ID = "P1"
print(ID)

base_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Valid Data")

directory = base_directory / ID

os.chdir(directory)

data_0_degree = pl.read_csv("0.csv", skip_rows=8, has_header=False, separator=",")
data_12_degree = pl.read_csv("12.csv", skip_rows=8, has_header=False, separator=",")
data_25_degree = pl.read_csv("25.csv", skip_rows=8, has_header=False, separator=",")

# Keep only first 12 columns
data_0_degree = data_0_degree.select(data_0_degree.columns[:18])
data_12_degree = data_12_degree.select(data_12_degree.columns[:18])
data_25_degree = data_25_degree.select(data_25_degree.columns[:18])

# Make it float
data_0_degree = (data_0_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_12_degree = (data_12_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))
data_25_degree = (data_25_degree.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64, strict=False)))

trial = 0
if trial == 0:
    data = data_0_degree
elif trial == 12:
    data = data_12_degree
elif trial == 25:
    data = data_25_degree


##################################################
################# DATA ANALYSIS ##################
##################################################
Gastr_EMG_time = data[:, 0].to_numpy()
Gastr_EMG = data[:, 1].to_numpy()
ECG_time = data[:, 2].to_numpy()
ECG = data[:, 3].to_numpy()
Quad_EMG_time = data[:, 4].to_numpy()
Quad_EMG = data[:, 5].to_numpy()
Acc_x_time = data[:, 6].to_numpy()
Acc_x = data[:, 7].to_numpy()
Acc_y_time = data[:, 8].to_numpy()
Acc_y = data[:, 9].to_numpy()
Acc_z_time = data[:,10].to_numpy()
Acc_z = data[:,11].to_numpy()
gyro_x_time = data[:, 12].to_numpy()
gyro_x = data[:, 13].to_numpy()
gyro_y_time = data[:, 14].to_numpy()
gyro_y = data[:, 15].to_numpy()
gyro_z_time = data[:,16].to_numpy()
gyro_z = data[:,17].to_numpy()




# Remove nan from IMU data
valid_idx = ~np.isnan(Acc_x_time)
Acc_x_time = Acc_x_time[valid_idx]
Acc_x = Acc_x[valid_idx]
Acc_y_time = Acc_y_time[valid_idx]
Acc_y = Acc_y[valid_idx]
Acc_z_time = Acc_z_time[valid_idx]
Acc_z = Acc_z[valid_idx]
gyro_x_time = gyro_x_time[valid_idx]
gyro_x = gyro_x[valid_idx]
gyro_y_time = gyro_y_time[valid_idx]
gyro_y = gyro_y[valid_idx]
gyro_z_time = gyro_z_time[valid_idx]
gyro_z = gyro_z[valid_idx]

print("Total duration of trial")
print(Gastr_EMG_time[-1])
signals = [
    (Gastr_EMG_time[:10000], Gastr_EMG[:10000], "Gastr_EMG"),
    (ECG_time[:10000], ECG[:10000], "ECG"),
    (Quad_EMG_time[:10000], Quad_EMG[:10000], "Quad_EMG"),
    (Acc_x_time[:10000], Acc_x[:10000], "Acc_x"),
    (Acc_y_time[:10000], Acc_y[:10000], "Acc_y"),
    (Acc_z_time[:10000], Acc_z[:10000], "Acc_z"),
    (gyro_x_time[:10000], gyro_x[:10000], "gyro_x"),
    (gyro_y_time[:10000], gyro_y[:10000], "gyro_y"),
    (gyro_z_time[:10000], gyro_z[:10000], "gyro_z"),
]

fig = go.Figure()

for time, signal, name in signals:
    fig.add_trace(go.Scattergl(
        x=time,
        y=signal,
        mode="lines",
        name=name,
    ))

fig.update_layout(
    title="Recorded signals",
    xaxis_title="Time",
    yaxis_title="Amplitude",
    template="plotly_white",
    hovermode="x unified",
    height=650,
)

fig.show()


# Spectral analysis
# FFT_fast(ECG,  fs_emg)
# FFT_fast(Gastr_EMG,  fs_emg)
FFT_fast(Quad_EMG,  fs_emg)
# FFT_fast(Acc_x,  fs_imu)
# FFT_fast(Acc_y,  fs_imu)
# FFT_fast(Acc_z,  fs_imu)

# Residuals Analysis for IMU
cutoff_frequencies = np.arange(5,70,1)
Residual_analysis(var=Acc_x, cutoff_frequencies=cutoff_frequencies, fs=fs_imu, number_of_fit_points=10, plot=True, save=None)
Residual_analysis(var=Acc_y, cutoff_frequencies=cutoff_frequencies, fs=fs_imu, number_of_fit_points=10, plot=True, save=None)
Residual_analysis(var=Acc_z, cutoff_frequencies=cutoff_frequencies, fs=fs_imu, number_of_fit_points=10, plot=True, save=None)
Residual_analysis(var=gyro_x, cutoff_frequencies=cutoff_frequencies, fs=fs_imu, number_of_fit_points=10, plot=True, save=None)

# Filtering IMU low pass
Acc_x = Butterworth(fs_imu, 20, Acc_x)
Acc_y = Butterworth(fs_imu, 20, Acc_y)
Acc_z = Butterworth(fs_imu, 20, Acc_z)
gyro_x = Butterworth(fs_imu, 20, gyro_x)

# Calculate the sum of squares for peak finding
SS_acc = Acc_x**2 + Acc_y**2 + Acc_z**2

# Filtering ECG - EMG bandpass
ECG = butter_bandpass_filtfilt(ECG, fs_emg, low=0.5, high=250, order=4, plot=False)
Gastr_EMG = butter_bandpass_filtfilt(Gastr_EMG, fs_emg, low=20, high=450, order=4, plot=False)
Quad_EMG = butter_bandpass_filtfilt(Quad_EMG, fs_emg, low=20, high=450, order=4, plot=False)
FFT_fast(Quad_EMG,  fs_emg)

# Filtering ECG notch
ECG = notch_filter_with_plots(ECG, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)

# Make the EMG time series absolute
Gastr_EMG = abs(Gastr_EMG)
Quad_EMG = abs(Quad_EMG)

# Linear envelope
Gastr_EMG_linear_envelope = emg_linear_envelope(Gastr_EMG, fs_emg, cutoff=12, order=4, plot=False)
Quad_EMG_linear_envelope = emg_linear_envelope(Quad_EMG, fs_emg, cutoff=12, order=4, plot=False)

