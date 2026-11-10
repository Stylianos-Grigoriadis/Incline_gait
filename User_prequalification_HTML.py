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

ID_list = [
    # "P1",
    # "P2",
    # "P3",
    # "P4",
    # "P5",
    # "P6",
    # "P7",
    # "P8",
    # "P9",
    # "P10",
    # "P11",
    # "P13",
    # "P14",
    # "P15",
    # "P16",
    # "P17",
    # "P18",
    # "P19",
    # "P20",
    "P21",
    "P22",
]
ID = "P1"
# Reference year for approximate age when Excel AGE contains a birth year.
age_reference_year = 2026
participants_excel = base_participants_path = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Participants' Information1.xlsx")
# HTML ECG/EMG display only: 1 = every point, 2 = every second point, etc.
ecg_emg_plot_every_n_points = 10
open_html_automatically = True  # Set False to save without opening the browser.
# Choose the folder where the HTML report will be saved.
save_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Figures\Qualification reports")
base_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Prequalified Data")
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
                               line=dict(width=1, simplify=False, color="#bd303b"), connectgaps=False))
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
                                  connectgaps=False, line=dict(width=1, simplify=False, color="#bd303b")), row=row, col=1)
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



def participant_table(durations):
    # Read only; never save or modify the participant workbook.
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError("Install openpyxl in your Python environment to read participant information.") from exc
    normalize = lambda value: ''.join(str(value or '').lower().split())
    try:
        workbook = load_workbook(participants_excel, read_only=True, data_only=True)
    except PermissionError as exc:
        raise PermissionError(f"Cannot read {participants_excel}. Close the workbook in Excel and check that it is available locally in OneDrive.") from exc
    matches = []
    try:
        for sheet in workbook.worksheets:
            headers = None
            for row in sheet.iter_rows(values_only=True):
                if headers is None:
                    candidate = [normalize(v) for v in row]
                    if 'subjectid' in candidate:
                        headers = candidate
                    continue
                record = dict(zip(headers, row))
                if str(record.get('subjectid') or '').strip().casefold() == ID.strip().casefold():
                    matches.append(record)
    finally:
        workbook.close()
    if len(matches) != 1:
        raise ValueError(f"Expected one participant row for {ID}; found {len(matches)} in {participants_excel}.")
    record = matches[0]
    def get(*columns):
        for column in columns:
            key = normalize(column)
            if key in record:
                return record[key]
        raise ValueError(f"Missing participant column: {columns[0]}")
    def number(value):
        try:
            return float(str(value).strip().replace(',', '.'))
        except (ValueError, TypeError):
            return None
    def formatted(value, unit=''):
        if value is None or str(value).strip() == '':
            return 'Not recorded'
        numeric = number(value)
        label = f'{numeric:g}' if numeric is not None else str(value).strip()
        return label + unit
    raw_age = get('AGE')
    age_value = number(raw_age)
    if age_value is not None and 1900 <= age_value <= age_reference_year:
        age = f'{age_reference_year - int(age_value)} years'
    else:
        age = formatted(raw_age, ' years')
    rows = [
        ('Sex', formatted(get('SEX'))),
        ('Age', age),
        ('Weight', formatted(get('WEIGHT'), ' kg')),
        ('Height', formatted(get('HEIGHT'), ' cm')),
        ('PWS 0%', formatted(get('PWS 0 (km/h)'), ' km/h')),
        ('PWS 12.5%', formatted(get('PWS 12.5 (km/h)', 'PWS 12 (km/h)'), ' km/h')),
        ('PWS 25%', formatted(get('PWS 25(km/h)', 'PWS 25 (km/h)'), ' km/h')),
    ]
    rows.extend((f'Recorded duration {label}', f'{minutes:.3f} minutes') for label, minutes in durations)
    cells = ''.join(f'<tr><th scope="row">{html.escape(label)}</th><td>{html.escape(value)}</td></tr>' for label, value in rows)
    return ('<aside class="participant-info"><h2>Participant information</h2>'
            f'<p class="participant-id">{html.escape(ID)}</p><table><tbody>{cells}</tbody></table>'
            + '</aside>')



def detail_plot(result, time_key, value_key, name, units, rectified_key=None, imu=False):
    # Report-only display: ECG/EMG use the configured step; IMU uses every point.
    step = 1 if imu else ecg_emg_plot_every_n_points
    fig = go.Figure()
    if rectified_key is not None:
        x, y = display_points(result[time_key], result[rectified_key], step)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Rectified EMG",
                                line=dict(color="#a5a5a5", width=1, simplify=False),
                                connectgaps=False))
    x, y = display_points(result[time_key], result[value_key], step)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                            line=dict(color="#bd303b", width=1.3, simplify=False),
                            connectgaps=False))
    fig.update_layout(template="plotly_white", height=330,
                      margin=dict(l=65, r=25, t=35, b=55),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#56616e"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis_title="Time (s)", yaxis_title=units,
                      showlegend=rectified_key is not None,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(range=[200, 250], autorange=False, gridcolor="#edf0f3")
    fig.update_yaxes(gridcolor="#edf0f3")
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True, "displaylogo": False, "doubleClick": "autosize"})


def detail_section(title, charts):
    return '<section><h2>' + title + '</h2><div class="detail-rows">' + ''.join(charts) + '</div></section>'



def spectral_plot(values, fs, name, units):
    # Unfiltered input, before any HTML point selection; FFT_fast is unchanged.
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return '<p>Spectrum unavailable: signal is too short or contains non-finite samples.</p>'
    freqs = fftfreq(len(values), 1 / fs)
    mask = freqs > 0
    transformed = fft(values)
    power = 2 * ((np.abs(transformed) / len(values)) ** 2)
    frequencies, power = freqs[mask], power[mask]
    if not np.isfinite(power.sum()) or power.sum() <= 0:
        return '<p>Spectrum unavailable: no positive-frequency power.</p>'
    existing_figures = set(plt.get_fignums())
    try:
        with patch.object(plt, "show"):
            # Explicit figure prevents adding FFT_fast's plot to an existing figure.
            plt.figure()
            thresholds = FFT_fast(values, fs)
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)
    fig = go.Figure(go.Scatter(x=frequencies, y=power, mode="lines", name=name,
                              line=dict(color="#bd303b", width=1, simplify=False)))
    for percent, frequency, color, dash, position in zip(
        (90, 95, 99), thresholds, ("#527b9e", "#b57a25", "#614a88"),
        ("dash", "dot", "dashdot"), (0.97, 0.86, 0.75),
    ):
        fig.add_shape(type="line", x0=float(frequency), x1=float(frequency),
                      y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=color, width=1.5, dash=dash))
        fig.add_annotation(x=float(frequency), y=position, xref="x", yref="paper",
                           text=f"{percent}%: {frequency:.3f} Hz", showarrow=False,
                           xanchor="left", font=dict(color=color, size=12),
                           bgcolor="rgba(255,255,255,0.9)")
    fig.update_layout(template="plotly_white", height=350,
                      margin=dict(l=75, r=35, t=35, b=55), showlegend=False,
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#56616e"),
                      xaxis_title="Frequency (Hz)",
                      yaxis_title=f"Power ({units})",
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#edf0f3", range=[0, 100], autorange=False)
    fig.update_yaxes(gridcolor="#edf0f3", rangemode="tozero")
    summary = ' · '.join(f'{percent}%: {frequency:.3f} Hz' for percent, frequency in zip((90, 95, 99), thresholds))
    return (fig.to_html(full_html=False, include_plotlyjs=False,
                        config={"responsive": True, "displaylogo": False, "doubleClick": "autosize"})
            + '<p class="report-note">' + summary + '</p>')


def spectral_section(charts):
    return ('<section><h2>Spectral analysis</h2>'
            '<p>Unfiltered recording (absolute raw EMG; SS_acc from raw acceleration). Markers indicate 90%, 95% and 99% of cumulative '
            'positive-frequency power, using FFT_fast. Zero frequency is excluded.</p>'
            '<div class="grid">' + ''.join(charts) + '</div></section>')



def residual_plot(values, fs):
    # Use the original function's calculated plot data; do not change its algorithm.
    values = np.asarray(values, dtype=float)
    if len(values) < 20 or not np.all(np.isfinite(values)):
        return '<p>Residual analysis unavailable: signal is too short or contains non-finite samples.</p>'
    cutoff_frequencies = np.arange(5, 70, 1)
    existing_figures = set(plt.get_fignums())
    try:
        with patch.object(plt, "show"):
            fc = Residual_analysis(values, cutoff_frequencies, fs,
                                   number_of_fit_points=10, plot=True, save=None)
        created = sorted(set(plt.get_fignums()) - existing_figures)
        ax = plt.figure(created[-1]).axes[0]
        residual_line, fitted_line, intercept_line = ax.lines[:3]
        frequency = np.asarray(residual_line.get_xdata())
        rms = np.asarray(residual_line.get_ydata())
        intercept = float(intercept_line.get_ydata()[0])
        fit_points = np.asarray(ax.collections[0].get_offsets())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequency, y=rms, mode="lines+markers", name="RMS residual",
                                line=dict(color="#bd303b", width=1.5), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=fit_points[:, 0], y=fit_points[:, 1], mode="markers",
                                name="Last 10 cutoff values", marker=dict(color="#b57a25", size=6)))
        fig.add_trace(go.Scatter(x=fitted_line.get_xdata(), y=fitted_line.get_ydata(),
                                mode="lines", name="Extrapolated linear fit",
                                line=dict(color="#b57a25", dash="dash")))
        fig.add_hline(y=intercept, line_color="#59616b", line_dash="dash")
        fig.add_vline(x=float(fc), line_color="#bd303b", line_dash="dot",
                      annotation_text=f"Fc = {fc:g} Hz", annotation_position="top right")
        selected = int(np.argmin(np.abs(frequency - fc)))
        fig.add_trace(go.Scatter(x=[float(fc)], y=[float(rms[selected])], mode="markers",
                                name="Selected cutoff", marker=dict(color="#bd303b", size=8)))
        fig.update_layout(template="plotly_white", height=380,
                          margin=dict(l=65, r=30, t=40, b=100),
                          font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#56616e"),
                          xaxis_title="Cutoff frequency (Hz)", yaxis_title="RMS residual",
                          legend=dict(orientation="h", y=-0.25, x=0),
                          paper_bgcolor="white", plot_bgcolor="white")
        fig.update_xaxes(gridcolor="#edf0f3")
        fig.update_yaxes(gridcolor="#edf0f3")
        return (fig.to_html(full_html=False, include_plotlyjs=False,
                            config={"responsive": True, "displaylogo": False, "doubleClick": "autosize"})
                + f'<p class="report-note">Selected cutoff: {fc:g} Hz · Intercept: {intercept:.4g}</p>')
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)


def residual_section(charts):
    return ('<section><h2>Residual analysis</h2>'
            '<p>Applied to unfiltered data: SS_acc from raw acceleration, or raw gyro_x. Tested cutoffs: 5–69 Hz; '
            'linear fit: highest 10 cutoff values. Results do not change the applied filters.</p>'
            '<div class="grid">' + ''.join(charts) + '</div></section>')



def raw_analysis_inputs(data):
    # Read directly from the CSV columns, before process_trial applies any filters.
    def column(index):
        return np.asarray(data[:, index].to_numpy(), dtype=float).copy()
    imu_rows = np.isfinite(column(6))  # Exclude the padded rows beyond IMU recording.
    acc_x, acc_y, acc_z = (column(index)[imu_rows] for index in (7, 9, 11))
    return {
        "ECG": column(3),
        "Quadriceps": np.abs(column(5)),
        "Gastrocnemius": np.abs(column(1)),
        "SS_acc": acc_x ** 2 + acc_y ** 2 + acc_z ** 2,
        "Angular velocity": column(13)[imu_rows],
    }


def main():
    for ID in ID_list:

        directory = base_directory / ID
        trials = (("0", "0%"), ("12", "12.5%"), ("25", "25%"))
        sections = {name: [] for name in (
            "ECG", "EMG quadriceps", "EMG gastrocnemius", "Stride intervals")}
        durations = []
        details = {name: [] for name in ("ECG", "Quadriceps", "Gastrocnemius", "SS_acc", "Angular velocity")}
        spectra = {name: [] for name in details}
        residuals = {name: [] for name in ("SS_acc", "Angular velocity")}
        for filename, label in trials:
            print(f"Processing {ID}: {label}", flush=True)
            data = pl.read_csv(directory / f"{filename}.csv", skip_rows=8,
                               has_header=False, separator=",")
            if data.width < 18:
                raise ValueError(f"{filename}.csv: expected at least 18 columns.")
            data = data.select(data.columns[:18])
            data = (data.with_columns(pl.all().cast(pl.Utf8).str.strip_chars())
                        .with_columns(pl.all().cast(pl.Float64, strict=False)))
            raw_inputs = raw_analysis_inputs(data)
            duration = duration_minutes(data)
            durations.append((label, duration))
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
            detail_specs = (
                ("ECG", "ECG_time", "ECG", "Filtered ECG", "mV", None, False),
                ("Quadriceps", "Quad_EMG_time", "Quad_EMG_linear_envelope", "Linear envelope", "mV", "Quad_EMG", False),
                ("Gastrocnemius", "Gastr_EMG_time", "Gastr_EMG_linear_envelope", "Linear envelope", "mV", "Gastr_EMG", False),
                ("SS_acc", "Acc_x_time", "SS_acc", "SS_acc", "G²", None, True),
                ("Angular velocity", "gyro_x_time", "gyro_x", "gyro_x", "deg/s", None, True),
            )
            for section, time_key, value_key, name, unit, rectified_key, imu in detail_specs:
                chart = detail_plot(result, time_key, value_key, name, unit, rectified_key, imu)
                details[section].append(f'<article><h3>{label} incline</h3>{chart}</article>')
                sampling_rate = fs_imu if imu else fs_emg
                power_units = "G⁴" if section == "SS_acc" else "(deg/s)²" if imu else "mV²"
                spectrum = spectral_plot(raw_inputs[section], sampling_rate, section + " — raw" + (" absolute" if section in ("Quadriceps", "Gastrocnemius") else ""), power_units)
                spectra[section].append(f'<article><h3>{label} incline</h3>{spectrum}</article>')
                if section in residuals:
                    residual_chart = residual_plot(raw_inputs[section], fs_imu)
                    residuals[section].append(f'<article><h3>{label} incline</h3>{residual_chart}</article>')
            del data, result, raw_inputs
        participant_html = participant_table(durations)
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
            ':root{color-scheme:light;--ink:#28313b;--muted:#6b7480;--accent:#ad2935;--border:#e4e8ed}'
            '*{box-sizing:border-box}html{scroll-behavior:smooth}'
            'body{margin:0 auto;padding:48px 36px 72px;max-width:1920px;background:#fff;color:var(--ink);'
            'font-family:"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.55}'
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
            'font-size:23px;font-weight:500;letter-spacing:-.4px;background:#fff}'
            '.durations strong{font-size:12px;font-weight:600;letter-spacing:.7px;color:var(--accent)}'
            'article{background:#fff;border:1px solid var(--border);border-radius:5px;min-width:0;overflow:hidden;padding:0 8px 8px}'
            'h3{margin:0 -8px 8px;padding:14px 20px;border-bottom:1px solid var(--border);'
            'font-size:14px;font-weight:600;text-align:left;color:var(--accent);background:#fff}'
            'section{margin-top:44px;padding-top:26px;border-top:1px solid var(--border)}'
            '.report-nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin:22px 0 0;font-size:13px}'
            '.report-nav a{color:var(--accent);text-decoration:none}.report-nav a:hover{text-decoration:underline}'
            '.report-note{font-size:14px;margin-top:22px}'
            '.modebar{opacity:.65}.modebar:hover{opacity:1}'
            '@media(max-width:1000px){body{padding:28px 18px}.grid{grid-template-columns:1fr}}'
            '@media(max-width:550px){.durations{grid-template-columns:1fr}h1{font-size:28px}}'
            '@media print{body{padding:0}.report-nav,.modebar{display:none}section{break-inside:avoid}}'
            'header{border-bottom:2px solid #2c3036;padding-bottom:22px}'
            'h1{font-size:30px;letter-spacing:-.6px}h2{font-size:20px}'
            'section{border-top:0;margin-top:38px;padding-top:0}'
            'section>h2{border-left:3px solid var(--accent);padding-left:12px;margin-bottom:16px}'
            'article{border-radius:2px}h3{color:#383e46;font-size:13px;background:#f8f8f8}'
            '.overview{display:grid;grid-template-columns:340px minmax(0,1fr);gap:36px;align-items:start}'
            '.participant-info{width:100%;max-width:340px;border:1px solid var(--border);padding:18px 20px;background:#fff}'
            '.participant-info h2{font-size:17px;margin:0}.participant-id{font-size:12px;margin:3px 0 12px;color:var(--accent)}'
            '.participant-info table{width:100%;border-collapse:collapse;font-size:13px}'
            '.participant-info th,.participant-info td{padding:7px 0;border-bottom:1px solid #eceef0;text-align:left}'
            '.participant-info th{font-weight:500;color:#69717a}.participant-info td{text-align:right;font-weight:600}'
            '.age-note{font-size:11px;line-height:1.45;margin:12px 0 0}'
            '.durations{gap:12px}.durations>div{font-size:22px;padding:18px;border-radius:2px}'
            '@media(max-width:1000px){.overview{grid-template-columns:1fr}}'
            '.tabs{display:flex;gap:8px;border-bottom:1px solid var(--border);margin:0 0 28px}'
            '.tab-button{font:600 14px "Segoe UI",Arial,sans-serif;border:0;border-bottom:3px solid transparent;'
            'background:white;color:#6b7480;padding:12px 22px;cursor:pointer}'
            '.tab-button[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}'
            '.tab-button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}'
            '.tab-panel[hidden]{display:none}.detail-rows{display:grid;grid-template-columns:1fr;gap:18px}'
            '.detail-rows article{width:100%}.participant-info{margin-bottom:24px}'
            '</style>'  
            f'</head><body><header><p class="eyebrow">Incline gait study</p><h1>{title}</h1>'
            '<p class="subtitle">Signal review across three treadmill incline conditions</p>'
            '</header><div class="tabs" role="tablist" aria-label="Signal views">'
            '<button id="tab-overall" class="tab-button" role="tab" aria-selected="true" aria-controls="panel-overall">Overall</button>'
            '<button id="tab-ecg" class="tab-button" role="tab" aria-selected="false" aria-controls="panel-ecg" tabindex="-1">ECG</button>'
            '<button id="tab-gastr" class="tab-button" role="tab" aria-selected="false" aria-controls="panel-gastr" tabindex="-1">EMG gastrocnemius</button>'
            '<button id="tab-quad" class="tab-button" role="tab" aria-selected="false" aria-controls="panel-quad" tabindex="-1">EMG quadriceps</button>'
            '<button id="tab-ss" class="tab-button" role="tab" aria-selected="false" aria-controls="panel-ss" tabindex="-1">IMU SS</button>'
            '<button id="tab-angular" class="tab-button" role="tab" aria-selected="false" aria-controls="panel-angular" tabindex="-1">IMU angular velocity</button></div>'
            '<div id="panel-overall" class="tab-panel" role="tabpanel" aria-labelledby="tab-overall">' + participant_html +
            '<p class="report-note">ECG is filtered; EMG plots show linear envelopes. ECG and EMG display every '
            f'{ecg_emg_plot_every_n_points} sample(s). Acceleration and gyroscope plots show every sample. '
            'All processing uses the complete recordings. This report does not assign '
            'qualification decisions automatically.</p>'
            + ''.join(body) + '</div>'
            '<div id="panel-ecg" class="tab-panel" role="tabpanel" aria-labelledby="tab-ecg" hidden>'
            + detail_section('ECG', details['ECG']) + spectral_section(spectra['ECG']) + '</div>'
            '<div id="panel-gastr" class="tab-panel" role="tabpanel" aria-labelledby="tab-gastr" hidden>'
            + detail_section('Gastrocnemius EMG', details['Gastrocnemius']) + spectral_section(spectra['Gastrocnemius']) + '</div>'
            '<div id="panel-quad" class="tab-panel" role="tabpanel" aria-labelledby="tab-quad" hidden>'
            + detail_section('Quadriceps EMG', details['Quadriceps']) + spectral_section(spectra['Quadriceps']) + '</div>'
            '<div id="panel-ss" class="tab-panel" role="tabpanel" aria-labelledby="tab-ss" hidden>'
            + detail_section('Acceleration sum of squares', details['SS_acc']) + spectral_section(spectra['SS_acc']) + residual_section(residuals['SS_acc']) + '</div>'
            '<div id="panel-angular" class="tab-panel" role="tabpanel" aria-labelledby="tab-angular" hidden>'
            + detail_section('Angular velocity — gyro_x', details['Angular velocity']) + spectral_section(spectra['Angular velocity']) + residual_section(residuals['Angular velocity']) + '</div>'
            '<script>'
            'const tabs=Array.from(document.querySelectorAll(".tab-button"));'
            'function activateTab(button){'
            'tabs.forEach(t=>{const active=t===button;t.setAttribute("aria-selected",String(active));'
            't.tabIndex=active?0:-1;document.getElementById(t.getAttribute("aria-controls")).hidden=!active;});'
            'requestAnimationFrame(()=>{document.getElementById(button.getAttribute("aria-controls"))'
            '.querySelectorAll(".plotly-graph-div").forEach(g=>{if(g._fullLayout)Plotly.Plots.resize(g);});});}'
            'tabs.forEach((button,index)=>{button.addEventListener("click",()=>activateTab(button));'
            'button.addEventListener("keydown",e=>{let target;'
            'if(e.key==="ArrowRight")target=(index+1)%tabs.length;'
            'if(e.key==="ArrowLeft")target=(index+tabs.length-1)%tabs.length;'
            'if(e.key==="Home")target=0;if(e.key==="End")target=tabs.length-1;'
            'if(target!==undefined){e.preventDefault();tabs[target].focus();activateTab(tabs[target]);}});});'
            '</script></body></html>')
        save_directory.mkdir(parents=True, exist_ok=True)
        output = save_directory / f"{ID}_prequalification.html"
        output.write_text(document, encoding="utf-8")
        print(f"Saved report: {output}")
        if open_html_automatically:
            webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
