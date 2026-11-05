import polars as pl
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch
from matplotlib.widgets import Slider, TextBox
import os
import html
import webbrowser
from unittest.mock import patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs


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


# Report additions. The functions above and analysis below are copied unchanged.


ID = "P1"
# HTML ECG/EMG display only: 1 = every point, 2 = every second point, etc.
ecg_emg_plot_every_n_points = 10
open_html_automatically = True  # Set False to save without opening the browser.
# Choose the folder where the HTML report will be saved.
save_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Figures\Qualification reports")
base_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Valid Data")
fs_emg = 2148.1481
fs_imu = 370.3704


def process_trial(data):
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

    # Spectral analysis
    FFT_fast(ECG,  fs_emg)
    FFT_fast(Gastr_EMG,  fs_emg)
    FFT_fast(Quad_EMG,  fs_emg)
    FFT_fast(Acc_x,  fs_imu)
    FFT_fast(Acc_y,  fs_imu)
    FFT_fast(Acc_z,  fs_imu)

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

    # Filtering ECG notch
    ECG = notch_filter_with_plots(ECG, fs_emg, f_notch=50.0, bandwidth=2.5, plot=False)

    # Make the EMG time series absolute
    Gastr_EMG = abs(Gastr_EMG)
    Quad_EMG = abs(Quad_EMG)

    # Linear envelope
    Gastr_EMG_linear_envelope = emg_linear_envelope(Gastr_EMG, fs_emg, cutoff=12, order=4, plot=False)
    Quad_EMG_linear_envelope = emg_linear_envelope(Quad_EMG, fs_emg, cutoff=12, order=4, plot=False)


    return locals()


def duration_minutes(data):
    # Use recorded time spans, not row count: EMG and IMU have different rates.
    spans = []
    for column in range(0, 18, 2):
        values = data[:, column].to_numpy()
        values = values[np.isfinite(values)]
        if len(values) > 1:
            spans.append(float(values.max() - values.min()))
    if not spans:
        raise ValueError("No usable timestamps for trial duration.")
    return max(spans) / 60


def display_points(time, values, every_n_points=1):
    # Select display samples only; never change the analysis arrays.
    if isinstance(every_n_points, bool) or not isinstance(every_n_points, (int, np.integer)) or every_n_points < 1:
        raise ValueError("ecg_emg_plot_every_n_points must be a positive integer.")
    time, values = np.asarray(time), np.asarray(values)
    if len(time) != len(values):
        raise ValueError("Time and signal lengths differ.")
    return time[::every_n_points], values[::every_n_points]


def signal_plot(time, values, name, units):
    x, y = display_points(time, values, ecg_emg_plot_every_n_points)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=name,
                               line=dict(width=1, simplify=False, color="#466b91"), connectgaps=False))
    fig.update_layout(template="plotly_white", height=340,
                      margin=dict(l=60, r=20, t=25, b=55),
                      xaxis_title="Time (s)", yaxis_title=units,
                      showlegend=False)
    fig.update_layout(font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#56616e"),
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor="#edf0f3", zerolinecolor="#dce2e8", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#edf0f3", zerolinecolor="#dce2e8", title_font=dict(size=12))
    fig.update_xaxes(range=[150, 160], autorange=False)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displaylogo": False, "doubleClick": "autosize"})


def movement_plot(result):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("SS_acc", "gyro_x"), vertical_spacing=0.18)
    for row, key, time_key, unit in (
        (1, "SS_acc", "Acc_x_time", "G²"),
        (2, "gyro_x", "gyro_x_time", "deg/s"),
    ):
        x, y = display_points(result[time_key], result[key])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=key,
                                  connectgaps=False, line=dict(width=1, simplify=False, color="#466b91")), row=row, col=1)
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=540, showlegend=False,
                      margin=dict(l=60, r=20, t=45, b=55))
    fig.update_layout(font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#56616e"),
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor="#edf0f3", zerolinecolor="#dce2e8", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#edf0f3", zerolinecolor="#dce2e8", title_font=dict(size=12))
    fig.update_xaxes(range=[150, 160], autorange=False)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displaylogo": False, "doubleClick": "autosize"})


def main():
    directory = base_directory / ID
    trials = (("0", "0%"), ("12", "12.5%"), ("25", "25%"))
    sections = {name: [] for name in (
        "ECG", "EMG quadriceps", "EMG gastrocnemius", "Stride intervals")}
    durations = []
    for filename, label in trials:
        print(f"Processing {ID}: {label}", flush=True)
        data = pl.read_csv(directory / f"{filename}.csv", skip_rows=8,
                           has_header=False, separator=",")
        if data.width < 18:
            raise ValueError(f"{filename}.csv: expected at least 18 columns.")
        data = data.select(data.columns[:18])
        data = (data.with_columns(pl.all().cast(pl.Utf8).str.strip_chars())
                    .with_columns(pl.all().cast(pl.Float64, strict=False)))
        duration = duration_minutes(data)
        durations.append(f'<div><strong>{label}</strong><br>{duration:.3f} minutes</div>')
        # Keep original diagnostic calculations, without opening their plot windows.
        existing_figures = set(plt.get_fignums())
        with patch.object(plt, "show"):
            result = process_trial(data)
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
        specifications = (
            ("ECG", "ECG_time", "ECG", "mV"),
            ("EMG quadriceps", "Quad_EMG_time", "Quad_EMG_linear_envelope", "mV"),
            ("EMG gastrocnemius", "Gastr_EMG_time", "Gastr_EMG_linear_envelope", "mV"),
        )
        for section, time_key, value_key, unit in specifications:
            chart = signal_plot(result[time_key], result[value_key], section, unit)
            sections[section].append(f'<article><h3>{label}</h3>{chart}</article>')
        sections["Stride intervals"].append(
            f'<article><h3>{label}</h3>{movement_plot(result)}</article>')
        del data, result
    body = []
    for name, columns in sections.items():
        note = ('<p>Filtered SS_acc and gyro_x. Stride intervals have not been calculated.</p>'
                if name == "Stride intervals" else "")
        section_id = name.lower().replace(' ', '-')
        body.append(f'<section id="{section_id}"><h2>{name}</h2>{note}<div class="grid">'
                    + ''.join(columns) + '</div></section>')
    title = html.escape(ID) + " — Data qualification"
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{title}</title><script>' + get_plotlyjs() + '</script>'
        '<style>'
        ':root{color-scheme:light;--ink:#28313b;--muted:#6b7480;--accent:#466b91;--border:#e4e8ed}'
        '*{box-sizing:border-box}html{scroll-behavior:smooth}'
        'body{margin:0 auto;padding:48px 36px 72px;max-width:1920px;background:#fff;color:var(--ink);'
        'font-family:Georgia,"Times New Roman",serif;font-size:16px;line-height:1.65}'
        'h1,h2,h3,.eyebrow,.durations,.report-nav{font-family:"Segoe UI",Arial,sans-serif}'
        'header{padding-bottom:24px;border-bottom:1px solid var(--border);margin-bottom:30px}'
        '.eyebrow{font-size:11px;font-weight:650;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin:0 0 10px}'
        'h1{font-size:34px;font-weight:600;letter-spacing:-1px;line-height:1.2;margin:0 0 12px}'
        '.subtitle{margin:0;color:var(--muted);font-size:17px}'
        'h2{font-size:23px;line-height:1.35;font-weight:600;letter-spacing:-.4px;margin:0 0 10px}'
        'p{max-width:1050px;margin:10px 0 20px;color:var(--muted)}'
        '.grid,.durations{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}'
        '.durations{margin:22px 0 26px}'
        '.durations>div{padding:18px 22px;border:1px solid var(--border);border-radius:5px;'
        'font-size:23px;font-weight:500;letter-spacing:-.4px;background:#fafbfd}'
        '.durations strong{font-size:12px;font-weight:600;letter-spacing:.7px;color:var(--accent)}'
        'article{background:#fff;border:1px solid var(--border);border-radius:5px;min-width:0;overflow:hidden;padding:0 8px 8px}'
        'h3{margin:0 -8px 8px;padding:14px 20px;border-bottom:1px solid var(--border);'
        'font-size:14px;font-weight:600;text-align:left;color:var(--accent);background:#fcfcfd}'
        'section{margin-top:44px;padding-top:26px;border-top:1px solid var(--border)}'
        '.report-nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin:22px 0 0;font-size:13px}'
        '.report-nav a{color:var(--accent);text-decoration:none}.report-nav a:hover{text-decoration:underline}'
        '.report-note{font-size:14px;margin-top:22px}'
        '.modebar{opacity:.65}.modebar:hover{opacity:1}'
        '@media(max-width:1000px){body{padding:28px 18px}.grid{grid-template-columns:1fr}}'
        '@media(max-width:550px){.durations{grid-template-columns:1fr}h1{font-size:28px}}'
        '@media print{body{padding:0}.report-nav,.modebar{display:none}section{break-inside:avoid}}'
        '</style>'
        f'</head><body><header><p class="eyebrow">Incline gait study</p><h1>{title}</h1>'
        '<p class="subtitle">Signal review across three treadmill incline conditions</p>'
        '<nav class="report-nav" aria-label="Report sections">'
        '<a href="#ecg">ECG</a><a href="#emg-quadriceps">Quadriceps EMG</a>'
        '<a href="#emg-gastrocnemius">Gastrocnemius EMG</a>'
        '<a href="#stride-intervals">Stride intervals</a></nav></header>'
        '<h2>Recorded trial duration</h2>'
        '<p>Duration is the longest recorded channel time span in each trial.</p>'
        '<div class="durations">' + ''.join(durations) + '</div>'
        '<p class="report-note">ECG is filtered; EMG plots show linear envelopes. ECG and EMG display every '
        f'{ecg_emg_plot_every_n_points} sample(s). Acceleration and gyroscope plots show every sample. '
        'All processing uses the complete recordings. This report does not assign '
        'qualification decisions automatically.</p>'
        + ''.join(body) + '</body></html>')
    save_directory.mkdir(parents=True, exist_ok=True)
    output = save_directory / f"{ID}_prequalification.html"
    output.write_text(document, encoding="utf-8")
    print(f"Saved report: {output}")
    if open_html_automatically:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
