"""Sweep EMG envelope cutoffs, using the existing Incline_gait lib functions.

Run beside lib.py in PyCharm. Outputs: one three-panel HTML figure per
participant, summary CSVs, and peak/interval CSVs for every cutoff.
Automatic detection uses User_Find_peaks.py's algorithm with estimated settings; manual edits
from its interactive windows are not reproduced. No interval outliers removed.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import find_peaks, correlate, resample_poly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lib


# SETTINGS: same sampling, filtering and peak defaults as User_Find_peaks.py.
base_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Valid Data")
save_directory = Path(__file__).resolve().parent / "EMG cutoff results"
ID_list = None  # None = all P<number> folders; or e.g. ["P6"].
fs_emg = 2148.1481
cutoff_frequencies = np.arange(3, 21, 1)
# Calibration is independent for each participant / incline / muscle.
reference_cutoff = 6
rhythm_period_range_s = (0.5, 2.0)  # Expected same-muscle activation cycle.
distance_fraction_of_period = 0.60
height_fraction_of_range = 0.20
open_html_automatically = True
trials = (("0", 0.0), ("12", 12.5), ("25", 25.0))
muscles = (("Gastrocnemius", 0, 1), ("Quadriceps", 4, 5))


def read_emg(path):
    # Read only the two EMG time/signal pairs, preserving original column order.
    data = pl.read_csv(path, skip_rows=8, has_header=False, separator=",",
                       columns=[0, 1, 4, 5], infer_schema=False)
    return (data.with_columns(pl.all().cast(pl.Utf8).str.strip_chars())
            .with_columns(pl.all().cast(pl.Float64, strict=False)))


def validate_signal(time, values):
    # Paired trailing padding is safe to trim; never join across internal gaps.
    valid = np.isfinite(time) & np.isfinite(values)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No finite EMG samples")
    first, last = indices[0], indices[-1] + 1
    if not valid[first:last].all():
        raise ValueError("Internal missing EMG samples; repair recording before filtering")
    time, values = time[first:last], values[first:last]
    if len(time) < 30 or np.any(np.diff(time) <= 0):
        raise ValueError("EMG too short or timestamps not strictly increasing")
    return time, values


def interval_metrics(intervals):
    n = len(intervals)
    result = dict(N_intervals=n, mean_interval_s=np.nan, SD_interval_s=np.nan,
                  CV=np.nan, DFA_alpha=np.nan, Ent_Samp=np.nan,
                  DFA_min_scale=16, DFA_max_scale=np.nan, status="ok")
    notes = []
    if not n:
        result["status"] = "No inter-peak intervals"
        return result
    result["mean_interval_s"] = np.mean(intervals)
    result["SD_interval_s"] = np.std(intervals)  # ddof=0, as User_analysis.py
    result["CV"] = np.std(intervals) / np.mean(intervals)
    # Exact scale convention from User_analysis.py: upper bound is excluded.
    scales = np.arange(16, n // 9, 1)
    if len(scales):
        result["DFA_max_scale"] = scales[-1]
    if len(scales) >= 2 and np.std(intervals) > 0:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, result["DFA_alpha"] = lib.dfa(intervals, scales, plot=False)
        if caught or not np.isfinite(result["DFA_alpha"]):
            notes.append("DFA numerical warning; inspect intervals")
    else:
        notes.append("DFA unavailable: need >=2 scales and nonconstant intervals")
    if n > 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            result["Ent_Samp"] = lib.Ent_Samp(intervals, 2, 0.2)
        if not np.isfinite(result["Ent_Samp"]):
            notes.append("Sample entropy undefined/infinite: insufficient matches")
    else:
        notes.append("Too few intervals for sample entropy")
    result["status"] = "; ".join(notes) or "ok"
    return result


def estimate_peak_settings(rectified):
    """Heuristic calibration, not an optimization of DFA/CV/entropy.

    Estimate rhythm on a 6 Hz reference envelope at ~107 Hz. Choose the
    strongest positive autocorrelation peak in the stated period range.
    Height is P10 + 20% of (P95-P10). Distance is 60% of the cycle period,
    permitting natural cycle variation while suppressing within-burst peaks.
    """
    envelope = lib.emg_linear_envelope(rectified, fs_emg, cutoff=reference_cutoff, order=4, plot=False)
    small = resample_poly(envelope, 1, 20)
    centered = small - np.mean(small)
    if np.std(centered) <= np.finfo(float).eps:
        raise ValueError("No varying EMG envelope for automatic rhythm estimation")
    ac = correlate(centered, centered, mode="full", method="fft")[len(centered)-1:]
    ac = ac / ac[0]
    fs_small = fs_emg / 20
    lo = int(np.ceil(rhythm_period_range_s[0] * fs_small))
    hi = min(int(np.floor(rhythm_period_range_s[1] * fs_small)), len(ac)-2)
    candidates, _ = find_peaks(ac)
    candidates = candidates[(candidates >= lo) & (candidates <= hi) & (ac[candidates] > 0)]
    if not len(candidates):
        raise ValueError("No positive rhythm peak in configured period range; inspect EMG")
    lag = candidates[np.argmax(ac[candidates])]
    period = lag / fs_small
    low, high = np.percentile(envelope, [10, 95])
    height = max(0.0, float(low + height_fraction_of_range * (high-low)))
    distance = max(1, int(distance_fraction_of_period * period * fs_emg))
    return distance, height, period, float(ac[lag]), envelope


def save_peak_check(time, envelope, distance, height, output, title):
    # A short excerpt keeps HTML manageable; calibration uses the full signal.
    mask = (time >= time[0] + 30) & (time <= time[0] + 50)
    if not mask.any():
        mask = time <= time[0] + 20
    indices, _ = find_peaks(envelope, distance=distance, height=height)
    selected = indices[mask[indices]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time[mask], y=envelope[mask], name="6 Hz reference envelope"))
    fig.add_trace(go.Scatter(x=time[selected], y=envelope[selected], mode="markers", name="Estimated peaks"))
    fig.add_hline(y=height, line_dash="dash", annotation_text="Estimated height")
    fig.update_layout(template="plotly_white", title=title, xaxis_title="Time (s)", yaxis_title="EMG amplitude")
    fig.write_html(str(output), include_plotlyjs=True, auto_open=False)


def plot_results(results, participant, output, show=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("DFA scaling exponent", "Coefficient of variation (SD/mean)",
                                        "Sample entropy (m=2, r=0.2 × SD)"),
                        vertical_spacing=0.09)
    colors = {0.0: "#2878b5", 12.5: "#e78b22", 25.0: "#388c64"}
    for (muscle, incline), group in results.groupby(["muscle", "incline_deg"], sort=False):
        group = group.sort_values("cutoff_Hz")
        name = f"{muscle}, {incline:g}°"
        for row, metric in enumerate(("DFA_alpha", "CV", "Ent_Samp"), 1):
            y = group[metric].to_numpy(dtype=float)
            y = np.where(np.isfinite(y), y, np.nan)
            fig.add_trace(go.Scatter(
                x=group["cutoff_Hz"], y=y, mode="lines+markers", name=name,
                legendgroup=name, showlegend=row == 1, connectgaps=False,
                line=dict(color=colors[incline], dash="solid" if muscle == "Gastrocnemius" else "dash"),
                customdata=group[["N_intervals", "status"]].to_numpy(),
                hovertemplate="Cutoff: %{x} Hz<br>Value: %{y:.5f}<br>N intervals: %{customdata[0]}<br>%{customdata[1]}<extra>%{fullData.name}</extra>"),
                row=row, col=1)
    for row, label in enumerate(("DFA α", "CV (ratio)", "Sample entropy"), 1):
        fig.update_yaxes(title_text=label, row=row, col=1)
    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Linear envelope cutoff (Hz)", row=3, col=1)
    fig.update_layout(template="plotly_white", height=1000,
                      title=f"{participant} — EMG inter-peak intervals vs envelope cutoff",
                      legend=dict(orientation="h", y=-0.10), margin=dict(b=160))
    fig.write_html(str(output), include_plotlyjs=True, auto_open=show)
    return fig


def main():
    if not base_directory.is_dir():
        raise FileNotFoundError(f"Data folder not found: {base_directory}")
    ids = ID_list if ID_list is not None else sorted(
        (p.name for p in base_directory.iterdir() if p.is_dir()
         and p.name.startswith("P") and p.name[1:].isdigit()), key=lambda x: int(x[1:]))
    if not ids:
        raise ValueError("No participant folders selected")
    save_directory.mkdir(parents=True, exist_ok=True)
    all_results, failures = [], []
    for participant in ids:
        rows = []
        output = save_directory / participant
        output.mkdir(parents=True, exist_ok=True)
        for filename, incline in trials:
            path = base_directory / participant / f"{filename}.csv"
            try:
                data = read_emg(path)
            except (OSError, pl.exceptions.PolarsError) as exc:
                failures.append(dict(participant=participant, file=str(path), muscle="both", error=str(exc)))
                print(f"SKIP {path}: {exc}", flush=True)
                continue
            for pair, (muscle, _, _) in enumerate(muscles):
                try:
                    time, raw = validate_signal(data[:, pair * 2].to_numpy(), data[:, pair * 2 + 1].to_numpy())
                    rectified = np.abs(lib.butter_bandpass_filtfilt(
                        raw, fs_emg, low=20, high=450, order=4, plot=False))
                    distance, height, period, rhythm_strength, reference = estimate_peak_settings(rectified)
                    save_peak_check(time, reference, distance, height,
                                    output / f"{filename}_{muscle}_peak_check.html",
                                    f"{participant}, {incline:g}°, {muscle}: period={period:.3f}s, distance={distance/fs_emg:.3f}s, height={height:.5g}")
                    del reference
                except ValueError as exc:
                    failures.append(dict(participant=participant, file=str(path), muscle=muscle, error=str(exc)))
                    print(f"SKIP {participant} {incline:g} {muscle}: {exc}", flush=True)
                    continue
                for cutoff in cutoff_frequencies:
                    envelope = lib.emg_linear_envelope(rectified, fs_emg, cutoff=cutoff, order=4, plot=False)
                    # Same automatic algorithm as lib.interactive_find_peaks_with_sliders_low_pass.
                    # Envelope is already filtered: do not low-pass it a second time.
                    indices, _ = find_peaks(envelope, distance=distance, height=height)
                    peak_times = time[indices]
                    intervals = np.diff(peak_times)
                    metrics = interval_metrics(intervals)
                    rows.append(dict(participant=participant, incline_deg=incline, muscle=muscle,
                                     cutoff_Hz=int(cutoff), N_peaks=len(indices),
                                     peak_distance_samples=distance, peak_height=height,
                                     estimated_period_s=period, rhythm_autocorrelation=rhythm_strength,
                                     reference_cutoff_Hz=reference_cutoff, **metrics))
                    pd.DataFrame(dict(peak_time_s=peak_times, peak_amplitude=envelope[indices],
                                      interval_from_previous_s=np.r_[np.nan, intervals] if len(indices) else np.array([]))).to_csv(
                        output / f"{filename}_{muscle}_{cutoff}Hz_peaks.csv", index=False)
                    print(f"{participant} {incline:g}° {muscle} {cutoff} Hz: N={len(intervals)}, {metrics['status']}", flush=True)
                # Checkpoint after each muscle/condition.
                pd.DataFrame(rows).to_csv(output / f"{participant}_cutoff_metrics.csv", index=False)
        if rows:
            results = pd.DataFrame(rows)
            all_results.extend(rows)
            plot_results(results, participant, output / f"{participant}_EMG_cutoff_comparison.html", open_html_automatically)
    if all_results:
        pd.DataFrame(all_results).to_csv(save_directory / "All_participants_cutoff_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["participant", "file", "muscle", "error"]).to_csv(save_directory / "skipped_recordings.csv", index=False)
    print(f"Finished. Results: {save_directory}. Skipped recordings: {len(failures)}")


if __name__ == "__main__":
    main()
