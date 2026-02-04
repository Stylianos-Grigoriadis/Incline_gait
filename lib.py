import math
import scipy.stats
from scipy import signal
import matplotlib.pyplot as plt
import statistics
import numpy as np
from numpy.fft import fft, fftfreq
import colorednoise as cn
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch
from matplotlib.widgets import Slider, SpanSelector
from scipy.signal import find_peaks


def interactive_find_peaks_with_sliders(
    signal,
    time,
    distance_init=200,
    height_init=0.02,
    distance_range=(1, 1000),
    height_range=(0.0, 1.0)
):
    """
    Interactive peak detection and manual refinement tool.

    Features
    --------
    - Automatic peak detection using scipy.signal.find_peaks
    - Two sliders to adjust:
        * minimum peak distance
        * minimum peak height
    - Span-based manual editing on the top plot:
        * DELETE / BACKSPACE: remove peaks in span
        * SPACE: add a peak at the maximum signal value in span
    - Lower plot shows inter-peak intervals (np.diff)

    Returns
    -------
    peak_times : np.ndarray
    peak_amplitudes : np.ndarray
    """

    # ===============================
    # Initial state
    # ===============================
    state = {
        "distance": distance_init,
        "height": height_init,
        "peak_times": [],
        "peak_amp": []
    }

    # ===============================
    # Peak detection
    # ===============================
    def detect_peaks():
        idx, props = find_peaks(
            signal,
            distance=state["distance"],
            height=state["height"]
        )
        state["peak_times"] = time[idx].tolist()
        state["peak_amp"] = signal[idx].tolist()

    detect_peaks()

    # ===============================
    # Figure layout
    # ===============================
    fig = plt.figure(figsize=(13, 7))

    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[4, 1, 0.2, 0.2]
    )

    ax_sig = fig.add_subplot(gs[0])
    ax_rr = fig.add_subplot(gs[1], sharex=ax_sig)
    ax_dist = fig.add_subplot(gs[2])
    ax_height = fig.add_subplot(gs[3])

    plt.subplots_adjust(hspace=0.08)

    # --- Disable toolbar interference (CRITICAL) ---
    try:
        fig.canvas.manager.toolbar.set_visible(False)
    except Exception:
        pass

    # ===============================
    # Initial plots
    # ===============================
    ax_sig.plot(time, signal, color="royalblue", lw=1.5)
    scatter = ax_sig.scatter([], [], color="red", zorder=3)

    vlines = []

    rr_line, = ax_rr.plot([], [], color="darkorange", lw=1.5)

    ax_sig.set_title("Interactive Peak Detection")
    ax_sig.set_ylabel("Amplitude")
    ax_rr.set_ylabel("Δ Time (s)")
    ax_rr.set_xlabel("Time (s)")

    ax_sig.grid(alpha=0.3)
    ax_rr.grid(alpha=0.3)

    # ===============================
    # Span selector
    # ===============================
    span = {"xmin": None, "xmax": None}

    def on_span(xmin, xmax):
        span["xmin"] = min(xmin, xmax)
        span["xmax"] = max(xmin, xmax)
        print(f"Selected span: {span['xmin']:.3f} – {span['xmax']:.3f} s")

    span_selector = SpanSelector(
        ax_sig,
        on_span,
        direction="horizontal",
        useblit=False,      # REQUIRED with sliders
        props=dict(alpha=0.25, facecolor="gray"),
        interactive=True
    )
    span_selector.active = True

    # ===============================
    # Update plots
    # ===============================
    def update_plots():
        # Scatter
        if state["peak_times"]:
            scatter.set_offsets(
                np.c_[state["peak_times"], state["peak_amp"]]
            )
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # Vertical lines
        for ln in vlines:
            ln.remove()
        vlines.clear()

        for t in state["peak_times"]:
            vlines.append(
                ax_sig.axvline(t, color="red", ls="--", alpha=0.3)
            )

        # RR plot
        if len(state["peak_times"]) > 1:
            rr = np.diff(state["peak_times"])
            rr_line.set_data(state["peak_times"][1:], rr)
            ax_rr.relim()
            ax_rr.autoscale_view()
        else:
            rr_line.set_data([], [])

        fig.canvas.draw_idle()

    update_plots()

    # ===============================
    # Sliders
    # ===============================
    s_distance = Slider(
        ax_dist,
        "Distance",
        distance_range[0],
        distance_range[1],
        valinit=distance_init,
        valstep=1
    )

    s_height = Slider(
        ax_height,
        "Height",
        height_range[0],
        height_range[1],
        valinit=height_init
    )

    def on_slider_change(val):
        state["distance"] = int(s_distance.val)
        state["height"] = s_height.val
        detect_peaks()
        update_plots()

    s_distance.on_changed(on_slider_change)
    s_height.on_changed(on_slider_change)

    # ===============================
    # Keyboard interaction
    # ===============================
    def on_key(event):
        # DELETE peaks in span
        if event.key in ("delete", "backspace"):
            if span["xmin"] is None:
                return

            keep = [
                i for i, t in enumerate(state["peak_times"])
                if not (span["xmin"] <= t <= span["xmax"])
            ]

            if len(keep) == len(state["peak_times"]):
                return

            state["peak_times"] = [state["peak_times"][i] for i in keep]
            state["peak_amp"] = [state["peak_amp"][i] for i in keep]

            update_plots()

        # ADD peak at max in span
        elif event.key == " ":
            if span["xmin"] is None:
                return

            mask = (time >= span["xmin"]) & (time <= span["xmax"])
            if not np.any(mask):
                return

            idx_local = np.argmax(signal[mask])
            idx_global = np.where(mask)[0][idx_local]

            t_new = time[idx_global]
            a_new = signal[idx_global]

            state["peak_times"].append(t_new)
            state["peak_amp"].append(a_new)

            order = np.argsort(state["peak_times"])
            state["peak_times"] = [state["peak_times"][i] for i in order]
            state["peak_amp"] = [state["peak_amp"][i] for i in order]

            update_plots()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()

    return (
        np.array(state["peak_times"]),
        np.array(state["peak_amp"])
    )



def plot_to_check_find_peaks_algo(signal, signal_time, peak_times, peak_amplitude, downsample=None):
    """
    Interactive visualization and manual editing of detected peaks using a time-span selector.

    The upper panel displays the signal (e.g., ECG/EMG linear envelope) together with the
    currently detected peaks. A horizontal SpanSelector allows the user to mark a temporal
    region of interest directly on the signal.

    User interaction (upper plot only):
    - Click and drag to define a time span.
    - DELETE or BACKSPACE removes all detected peaks whose times fall within the selected span.
    - SPACE adds a new peak at the maximum signal value within the selected span.

    All edits are applied immediately:
    - Peak times and amplitudes are updated.
    - Inter-peak time intervals are recomputed.
    - The lower panel (inter-peak interval time series) is refreshed automatically.

    This tool enables manual correction, refinement, and inspection of peak detection results
    in a fully interactive and reproducible manner.
    """

    # ---- ensure mutable ----
    peak_times = list(peak_times)
    peak_amplitude = list(peak_amplitude)

    # ---- downsampling (for display only) ----
    if downsample is None or downsample <= 1:
        t_plot = signal_time
        sig_plot = signal
    else:
        t_plot = signal_time[::downsample]
        sig_plot = signal[::downsample]

    # ---- figure & axes ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        gridspec_kw={'height_ratios': [4, 1]},
        sharex=True
    )

    # ---- TOP: signal ----
    ax1.plot(t_plot, sig_plot, color='royalblue', linewidth=2)

    scatter = ax1.scatter(
        peak_times,
        peak_amplitude,
        color='red',
        zorder=3
    )

    vlines = [
        ax1.axvline(t, color='red', linestyle='--', alpha=0.3) for t in peak_times
    ]

    ax1.set_ylabel("Amplitude")
    ax1.set_title("Signal with editable peaks")
    ax1.grid(alpha=0.3)

    # ---- BOTTOM: inter-peak intervals ----
    def compute_intervals():
        return np.array(peak_times[1:]), np.diff(peak_times)

    x_rr, rr = compute_intervals()
    rr_line, = ax2.plot(x_rr, rr, color='darkorange', linewidth=1.5)

    ax2.set_ylabel("Δ Time (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Inter-peak intervals")
    ax2.grid(alpha=0.3)

    # ---- span state ----
    span = {"xmin": None, "xmax": None}

    def on_select(xmin, xmax):
        span["xmin"] = min(xmin, xmax)
        span["xmax"] = max(xmin, xmax)
        print(f"Selected span: {span['xmin']:.3f}–{span['xmax']:.3f} s")

    span_selector = SpanSelector(
        ax1,
        on_select,
        direction="horizontal",
        useblit=True,
        props=dict(alpha=0.2, facecolor="gray"),
        interactive=True
    )

    # ---- update plots ----
    def update_plots():
        # scatter
        if peak_times:
            scatter.set_offsets(np.c_[peak_times, peak_amplitude])
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # vertical lines
        for ln in vlines:
            ln.remove()
        vlines.clear()

        for t in peak_times:
            vlines.append(
                ax1.axvline(t, color='red', linestyle='--', alpha=0.3)
            )

        # RR plot
        x_rr, rr = compute_intervals()
        rr_line.set_data(x_rr, rr)
        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw_idle()

    # ---- key handling ----
    def on_key_press(event):
        # ---- DELETE peaks in span ----
        if event.key in ('delete', 'backspace'):
            if span["xmin"] is None:
                print("No span selected.")
                return

            keep = [
                i for i, t in enumerate(peak_times)
                if not (span["xmin"] <= t <= span["xmax"])
            ]

            if len(keep) == len(peak_times):
                print("No peaks in span.")
                return

            removed = len(peak_times) - len(keep)
            print(f"Removed {removed} peak(s).")

            peak_times[:] = [peak_times[i] for i in keep]
            peak_amplitude[:] = [peak_amplitude[i] for i in keep]

            span["xmin"] = span["xmax"] = None
            update_plots()

        # ---- ADD peak at max in span ----
        elif event.key == ' ':
            if span["xmin"] is None:
                print("No span selected.")
                return

            # indices of signal inside span
            mask = (signal_time >= span["xmin"]) & (signal_time <= span["xmax"])

            if not np.any(mask):
                print("No data in span.")
                return

            idx = np.argmax(signal[mask])
            idx_global = np.where(mask)[0][idx]

            t_peak = signal_time[idx_global]
            amp_peak = signal[idx_global]

            print(f"Added peak at {t_peak:.3f} s")

            peak_times.append(t_peak)
            peak_amplitude.append(amp_peak)

            # keep peaks sorted by time
            order = np.argsort(peak_times)
            peak_times[:] = [peak_times[i] for i in order]
            peak_amplitude[:] = [peak_amplitude[i] for i in order]

            span["xmin"] = span["xmax"] = None
            update_plots()

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.tight_layout()
    plt.show()

    return peak_times, peak_amplitude

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
    Zero-phase Butterworth band-pass for fNIRS.

    Parameters
    ----------
    x : 1D array
    fs : float         # sampling rate (Hz)
    low, high : float  # cutoffs in Hz
    order : int
    plot : bool        # if True, show 2x2 time & frequency plots

    Returns
    -------
    y : 1D array
        Filtered signal
    """

    x = np.array(x)

    nyq = fs / 2.0
    wn = [low / nyq, high / nyq]
    if not (0 < wn[0] < wn[1] < 1):
        raise ValueError("Cutoffs must satisfy 0 < low < high < fs/2.")

    b, a = butter(order, wn, btype='band')

    padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
    y = filtfilt(b, a, x, padtype='odd', padlen=padlen)

    if plot:

        def compute_fft(sig):
            freqs = fftfreq(len(sig), 1 / fs)
            mask = freqs > 0
            Y = fft(sig)
            psd = 2 * (np.abs(Y) / len(sig)) ** 2
            return freqs[mask], psd[mask]

        t = np.arange(len(x)) / fs
        f_x, a_x = compute_fft(x)
        f_y, a_y = compute_fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(12, 7))

        # ---- Time domain: original
        axs[0, 0].plot(t, x, color='black', alpha=0.6)
        axs[0, 0].set_title('Time domain – Original')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Signal')

        # ---- Time domain: filtered
        axs[0, 1].plot(t, y, color='royalblue', linewidth=1.3)
        axs[0, 1].set_title('Time domain – Band-pass filtered')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Signal')

        # ---- Frequency domain: original
        axs[1, 0].plot(f_x, a_x, color='black', alpha=0.6)
        axs[1, 0].set_title('Frequency domain – Original')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Power')

        # ---- Frequency domain: filtered
        axs[1, 1].plot(f_y, a_y, color='royalblue', linewidth=1.3)
        axs[1, 1].set_title(
            f'Frequency domain – Band-pass {low}-{high} Hz (order={order})'
        )
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Power')

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return y

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

def FFT(var,fs):
    dt = 1 / fs
    freqs = fftfreq(len(var), dt)
    mask = freqs > 0
    Y = fft(var)
    pSpec = 2 * ((abs(Y) / len(var)) ** 2)

    f = freqs[mask]
    a = pSpec[mask]
    plt.plot(f,a)
    plt.show()
    print(1)
    sumA=[0]
    for i in range(1,len(a)):
        sumA.append(a[i]+sumA[i-1])
    print(2)

    prec90= []
    prec95 = []
    prec99 = []
    percA = [(sumA[i] / sumA[-1])*100 for i in range(len(sumA))]
    for p in percA:
        prec90.append(abs(p - 90))
        prec95.append(abs(p - 95))
        prec99.append(abs(p - 99))
    print(3)

    for i in range(len(percA)):
        if prec90[i]==min(prec90):
            index90 = i
        if prec95[i]==min(prec95):
            index95 = i
        if prec99[i]==min(prec99):
            index99 = i

    print(4)

    return f[index90],f[index95],f[index99]

def q_to_ypr(q):
    if q:
        yaw = (math.atan2(2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[0] ** 2 + 2 * q[1] ** 2 - 1))
        roll = (-1 * math.asin(2 * q[1] * q[3] + 2 * q[0] * q[2]))
        pitch = (math.atan2(2 * q[2] * q[3] - 2 * q[0] * q[1], 2 * q[0] ** 2 + 2 * q[3] ** 2 - 1))
        return [yaw, pitch, roll]

def pyth2d(x1,y1,x2,y2):
    x=x2-x1
    y=y2-y1
    c=math.sqrt(x**2+y**2)
    return c

def compute_cop(X, Y, df_filtered):
    list_X_coordinates_left_plate = []
    list_Y_coordinates_left_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1L'][i] + df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i] + \
                df_filtered['CHANNEL_4L'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i])) / F_all
        list_X_coordinates_left_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3L'][i] + df_filtered['CHANNEL_4L'][i])) / F_all
        list_Y_coordinates_left_plate.append(y_coordinate)

    list_X_coordinates_right_plate = []
    list_Y_coordinates_right_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1R'][i] + df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i] + \
                df_filtered['CHANNEL_4R'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i])) / F_all
        list_X_coordinates_right_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3R'][i] + df_filtered['CHANNEL_4R'][i])) / F_all
        list_Y_coordinates_right_plate.append(y_coordinate)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    for i in range(len(list_X_coordinates_left_plate)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_left_plate[i] - X / 2)
        list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_left_plate[i] - Y / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_right_plate[i] - X / 2)
        list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_right_plate[i] - Y / 2)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms = []
    for i in range(len(list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[i] - X / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[i] + X / 2)

    list_X_coordinates_both_plates = []
    list_Y_coordinates_both_plates = []
    for i in range(len(list_X_coordinates_right_plate)):
        list_X_coordinates_both_plates.append((list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i] +
                                               list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i]) / 2)
        list_Y_coordinates_both_plates.append((list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[
                                                   i] +
                                               list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[
                                                   i]) / 2)
    return list_X_coordinates_both_plates,list_Y_coordinates_both_plates
    #return list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform

def peaks(var,distance,height):
    peaks, _ = signal.find_peaks(var, distance=distance, height=height)
    peaksAmp = [var[peak] for peak in peaks]

    return peaks, peaksAmp

def Linear_Interpolation(col, step, plus):
    n = len(col)
    newdf = []
    value = step
    while value < n - 1:
        if math.ceil(value) == math.floor(value):
            num = col[math.ceil(value) + plus]
        else:

            num = ((col[math.ceil(value) + plus] - col[math.floor(value) + plus]) * (value - math.floor(value))) / (
                    math.ceil(value) - math.floor(value)) + col[math.floor(value) + plus]

        newdf.append(num)
        value = value + step

    return newdf

def Butterworth(fs, fc, var):
    """ Parameter:
            fs:     sampling frequency
            fc:     cutoff frequency for example 30Hz
            var:    data series
    """

    b, a = signal.butter(N=2, Wn=fc, btype='low', fs=fs)
    return signal.filtfilt(b, a, var)

def Average(lst):
    return sum(lst) / len(lst)

def Remove_drift(inte,index,cut):
    inte = inte[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]
    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_sl[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Remove_drift2(inte,inte2nd,index,cut):
    inte = inte[cut:]
    inte2nd = inte2nd[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]

    slope2, intercept, r, p, stderr = scipy.stats.linregress(index, inte2nd)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl2 = [i * slope2 for i in index]
    t_slope=[a-b for a,b in zip(t_sl,t_sl2)]

    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_slope[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Bland_Altman_plot(Var1,Var2,title):
    Difference = [v - m for v, m in zip(Var1, Var2)]
    Mean = [(v + m) / 2 for v, m in zip(Var1, Var2)]
    Bias = Average(Difference)
    StanDev = statistics.stdev(Difference)
    LowerLOA = Bias - 1.96 * StanDev
    UpperLOA = Bias + 1.96 * StanDev
    inlims=0
    uplim=0
    downlim=0
    for d in Difference:
        if d>=UpperLOA:
            uplim+=1
        elif d<=LowerLOA:
            downlim+=1
        else:
            inlims+=1

    print('Total points: ', len(Difference),
          '\nInside points number: ', inlims,
          '\nUp points number: ', uplim,
          '\nDown points number: ', downlim,
          '\nUp Perc: ', (uplim/len(Difference))*100,
          '\nDown Perc: ', (downlim/len(Difference))*100,
          '\nOut Perc: ', ((downlim + uplim) / len(Difference)) * 100)

    plt.show()
    plt.title('Bland Altman Plot {name}'.format(name=title), fontsize=16)
    plt.xlabel('Average', fontsize=16)
    plt.ylabel('Difference', fontsize=16)
    plt.scatter(Mean, Difference, color='grey', linewidths=1.5)
    plt.axhline(y=Bias, color='black')
    plt.axhline(y=LowerLOA, color='black', ls=':')
    plt.axhline(y=UpperLOA, color='black', ls=':')
    plt.show()

    #OutPerc = ((downlim + uplim) / len(Difference)) * 100
    res_list = [len(Difference), inlims, uplim, downlim, (uplim / len(Difference)) * 100,
                (downlim / len(Difference)) * 100, ((downlim + uplim) / len(Difference)) * 100]
    return res_list

def intergral(span,dt):
    rects = []
    for i in range(len(span) - 1):
        rects.append(((span[i] + span[i + 1]) * dt) / 2)
    integral = [rects[0]]
    for i in range(len(rects) - 1):
        integral.append(integral[i] + rects[i + 1])

    return integral

def derivative(array,fs):
    dt = 1/fs
    der = []

    array = list(array)

    for i in range(len(array)-1):
        der.append((array[i+1]-array[i])/dt)
    return der

def Pink_noise_generator():
    pass

def residual_analysis(signal, sampling_freq, list_cutoff_freq):
    residuals_list = []
    signal = np.array(signal)
    for i in range(len(list_cutoff_freq)):
        filtered_signal = Butterworth(sampling_freq,list_cutoff_freq[i],signal)
        filtered_signal = np.array(filtered_signal)
        sum_squares = np.sum((signal - filtered_signal)**2)
        residual = np.sqrt(sum_squares/len(signal))
        residuals_list.append(residual)

    plt.plot(list_cutoff_freq, residuals_list)
    plt.show()

def correlation_analysis(x, y, method='pearson', plot=False, xlabel='X', ylabel='Y', title=None):
    """
    Perform correlation analysis between two data series.

    Parameters
    ----------
    x, y : array-like or pd.Series
        Input data
    method : str
        'pearson' or 'spearman'
    plot : bool
        If True, plot scatter with regression line
    xlabel, ylabel : str
        Axis labels for plot
    title : str
        Plot title

    Returns
    -------
    results : dict
        Dictionary with correlation coefficient, p-value, and N
    """

    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaNs pairwise
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Not enough data points for correlation.")

    # Compute correlation
    if method.lower() == 'pearson':
        r, p = pearsonr(x, y)
        corr_name = 'Pearson r'
    elif method.lower() == 'spearman':
        r, p = spearmanr(x, y)
        corr_name = 'Spearman ρ'
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    # Optional plot
    if plot:
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, color='k', alpha=0.7)

        # Regression line (for visualization)
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r', lw=2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is None:
            plt.title(f'{corr_name} = {r:.2f}, p = {p:.3g}')
        else:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    # Return results
    results = {
        'method': corr_name,
        'r': r,
        'p_value': p,
        'n': len(x)
    }

    return results

def Kinvent_convert_accel(signal):
    signal = np.asarray(signal)
    acceleration_range = np.max(signal) - np.min(signal)
    signal_converted = (signal - 32768) * (acceleration_range / 32768)

    return signal_converted

def Kinvent_convert_emg(signal):
    signal = np.asarray(signal)
    signal = (signal - 2 ** 23) / 10
    signal = signal/1000

    return signal

def prepare_emg_df(df):
    df = df.iloc[:, :-1]
    df.columns = ['Time', 'EMG']
    df['EMG'] = Kinvent_convert_emg(df['EMG'])

    return df

def prepare_ecg_df(df):
    df = df.iloc[:, :-1]
    df.columns = ['Time', 'ECG']
    df['ECG'] = Kinvent_convert_emg(df['ECG'])

    return df

def prepare_imu_df(df):

    df = df.iloc[:, [0, 5, 6, 7]].copy().set_axis(['Time', 'Acc_X', 'Acc_Y', 'Acc_Z'], axis=1)
    return df