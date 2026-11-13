"""Analyze saved peaks for all participants; no signal filtering or redetection.

Run beside lib.py. Requires numpy, pandas, openpyxl, matplotlib and lib dependencies.
Results stay in memory. Plot functions display figures without saving files.
Source Excel files are read only. CV is a ratio; SD uses ddof=0.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib

project_directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait")
input_directory = project_directory / "Data" / "Peaks Data"
participants_excel = project_directory / "Data" / "Participants' Information1.xlsx"
exclude_not_valid_participants = True
signals = ("ECG", "Gastrocnemius_EMG", "IMU")
inclines = ("0", "12.5", "25")


def read_peak_times(column):
    """Trim export padding only; reject internal blanks and malformed values."""
    present = column.notna() & column.astype(str).str.strip().ne("")
    positions = np.flatnonzero(present.to_numpy())
    if not len(positions):
        raise ValueError("No saved peak times")
    series = column.iloc[:positions[-1]+1]
    if not present.iloc[:positions[-1]+1].all():
        raise ValueError("Missing peak time before the end of the series")
    times = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("Non-finite peak times")
    if len(times) < 2:
        raise ValueError("Need at least two peaks for an interval")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Peak times must be strictly increasing; duplicates/order need review")
    return times


def calculate_metrics(intervals):
    n = len(intervals)
    scales = np.arange(16, n // 9, 1)  # Same exclusive upper bound as User_analysis.py.
    result = dict(N_intervals=n, mean_interval_s=float(np.mean(intervals)),
                  SD_interval_s=float(np.std(intervals)),
                  CV=float(np.std(intervals)/np.mean(intervals)),
                  DFA_alpha=np.nan, Ent_Samp=np.nan, DFA_min_scale=16,
                  DFA_max_scale=int(scales[-1]) if len(scales) else np.nan,
                  status="ok")
    notes = []
    if len(scales) >= 2 and np.std(intervals) > 0:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, result["DFA_alpha"] = lib.dfa(intervals, scales, order=1, plot=False)
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


def participant_exclusions(path, participant_ids):
    """Read IDs from A and Analysis Comments from H; never modify workbook.

    Whitespace-only comments count as empty. Literal text such as 'NA' counts
    as a comment. If an ID occurs more than once, any comment excludes it.
    """
    try:
        table = pd.read_excel(path, engine="openpyxl", header=None,
                              usecols="A,H", keep_default_na=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot read participants workbook: {path}. Close it in Excel "
            "and make sure it is available locally, then run again."
        ) from exc
    if table.shape[1] != 2:
        raise ValueError("Participants workbook must contain columns A (ID) and H (Analysis Comments)")
    known, excluded = set(), set()
    for participant_id, comment in table.itertuples(index=False, name=None):
        participant_id = str(participant_id).strip().upper()
        if not (participant_id.startswith("P") and participant_id[1:].isdigit()):
            continue  # Skip header and unused rows.
        participant_id = f"P{int(participant_id[1:])}"
        known.add(participant_id)
        if pd.notna(comment) and str(comment).strip():
            excluded.add(participant_id)
    missing = set(participant_ids) - known
    if missing:
        raise ValueError(f"Peak-file IDs missing from participants column A: {', '.join(sorted(missing))}")
    return excluded & set(participant_ids)


def analyze_peaks(directory=None, verbose=True):
    """Return one metrics row per participant/signal/incline; write no files."""
    directory = Path(directory) if directory is not None else input_directory
    files = sorted((p for p in directory.glob("P*_Peaks.xlsx")
                    if p.stem.removesuffix("_Peaks")[1:].isdigit()),
                   key=lambda p: int(p.stem.split("_")[0][1:]))
    if not files:
        raise FileNotFoundError(f"No participant peak workbooks in {directory}")
    excluded = set()
    if exclude_not_valid_participants:
        excluded = participant_exclusions(
            participants_excel, [p.stem.removesuffix("_Peaks") for p in files])
        files = [p for p in files if p.stem.removesuffix("_Peaks") not in excluded]
        if verbose:
            print("Excluded (Analysis Comments in H): " +
                  (", ".join(sorted(excluded, key=lambda x: int(x[1:]))) or "none"))
        if not files:
            raise ValueError("All participants were excluded by Analysis Comments in column H")
    rows = []
    for source in files:
        participant = source.stem.removesuffix("_Peaks")
        try:
            data = pd.read_excel(source, engine="openpyxl")
            read_error = None
        except (OSError, ValueError) as exc:
            data, read_error = None, str(exc)
        for signal in signals:
            for incline in inclines:
                column = f"{signal}_Peak_Time_{incline}_deg"
                row = dict(participant=participant, signal=signal, incline_deg=float(incline),
                           source_file=str(source), source_column=column, N_peaks=np.nan,
                           N_intervals=0, mean_interval_s=np.nan, SD_interval_s=np.nan,
                           CV=np.nan, DFA_alpha=np.nan, Ent_Samp=np.nan,
                           DFA_min_scale=16, DFA_max_scale=np.nan, status="")
                try:
                    if read_error:
                        raise ValueError(f"Workbook unreadable: {read_error}")
                    if column not in data.columns:
                        raise ValueError(f"Missing column: {column}")
                    times = read_peak_times(data[column])
                    row.update(N_peaks=len(times), **calculate_metrics(np.diff(times)))
                except (ValueError, TypeError) as exc:
                    row["status"] = f"ERROR: {exc}"
                rows.append(row)
    results = pd.DataFrame(rows)
    if verbose:
        print(f"Analyzed {len(files)} participants, {len(results)} series; "
              f"{sum(results.status != 'ok')} flagged. No output files saved.")
        flagged = results.loc[results.status != "ok", ["participant", "signal", "incline_deg", "status"]]
        if not flagged.empty:
            print(flagged.to_string(index=False))
    return results


METRIC_LABELS = {"DFA_alpha": "DFA α", "CV": "CV (SD/mean)", "Ent_Samp": "Sample entropy"}
SIGNAL_LABELS = {"ECG": "ECG", "Gastrocnemius_EMG": "Gastrocnemius EMG", "IMU": "IMU"}


def _draw_boxplot(ax, results, signal, metric, show_datapoints, show_ids, show_lines):
    data = results.loc[results.signal == signal, ["participant", "incline_deg", metric]].copy()
    if data.duplicated(["participant", "incline_deg"]).any():
        raise ValueError("Expected one value per participant, signal and incline")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    wide = data.pivot(index="participant", columns="incline_deg", values=metric).reindex(columns=[0., 12.5, 25.])
    wide = wide.replace([np.inf, -np.inf], np.nan)
    ids = sorted(wide.index, key=lambda x: (int(x[1:]) if str(x).startswith("P") and str(x)[1:].isdigit() else float("inf"), str(x)))
    wide = wide.reindex(ids)
    groups = [wide[c].dropna().to_numpy() for c in wide.columns]
    colors = ["#7DA6C5", "#8EB7A4", "#D4A073"]
    # Outliers remain included in box statistics. Points are drawn separately.
    boxes = ax.boxplot(groups, positions=[0, 1, 2], widths=.48, patch_artist=True,
                       showfliers=not show_datapoints, manage_ticks=False,
                       medianprops=dict(color="#263746", linewidth=1.8),
                       whiskerprops=dict(color="#71808C", linewidth=1.1),
                       capprops=dict(color="#71808C", linewidth=1.1),
                       flierprops=dict(marker="o", markersize=4, markeredgecolor="#71808C", alpha=.6))
    for patch, color in zip(boxes["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=.45)
    # Same small horizontal offset at every incline for each participant.
    # Missing middle values break lines instead of connecting across missing data.
    offsets = np.random.default_rng(42).uniform(-.15, .15, len(ids))
    for index, participant in enumerate(ids):
        y = wide.loc[participant].to_numpy(dtype=float)
        x = np.arange(3) + offsets[index]
        valid = np.isfinite(y)
        if show_lines:
            ax.plot(x, y, color="#71808C", alpha=.30, linewidth=.85, zorder=1)
        if show_datapoints:
            ax.scatter(x[valid], y[valid], s=26, color="#344F65", alpha=.72,
                       edgecolors="white", linewidths=.45, zorder=3)
        if show_ids:
            for xx, yy in zip(x[valid], y[valid]):
                ax.annotate(str(participant), (xx, yy), xytext=(4, 3), textcoords="offset points",
                            fontsize=7, color="#344F65", zorder=4)
    ax.set_xticks([0, 1, 2], ["0°", "12.5°", "25°"])
    ax.set_xlim(-.55, 2.65 if show_ids else 2.55)
    ax.set_xlabel("Incline", fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_title(SIGNAL_LABELS[signal], loc="left", fontsize=12, fontweight="bold", pad=16)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EBEF", linewidth=.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#C8D1D9")
    ax.tick_params(labelsize=9, color="#C8D1D9")
    ax.margins(y=.15)
    for i, group in enumerate(groups):
        ax.text(i, 1.015, f"n={len(group)}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="#71808C")
    if not any(len(g) for g in groups):
        ax.text(.5, .5, "No valid values", transform=ax.transAxes, ha="center")


def plot_metric_boxplots(results, metric="DFA_alpha", signal=None,
                        show_datapoints=True, show_ids=False, show_lines=False, show=True):
    """One metric, one or all signals. Return (figure, axes); never save.

    metric: 'DFA_alpha', 'CV', or 'Ent_Samp'.
    signal: 'ECG', 'Gastrocnemius_EMG', 'IMU', or None for all three.
    The three display switches work independently, including IDs without dots.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {tuple(METRIC_LABELS)}")
    if signal is not None and signal not in signals:
        raise ValueError(f"signal must be one of {signals}, or None")
    selected = signals if signal is None else (signal,)
    fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 5), squeeze=False,
                             constrained_layout=True, facecolor="white")
    for ax, name in zip(axes[0], selected):
        _draw_boxplot(ax, results, name, metric, show_datapoints, show_ids, show_lines)
    fig.suptitle(METRIC_LABELS[metric] + " across inclines", fontsize=15, fontweight="bold")
    if show:
        plt.show()
    return fig, axes[0]


def plot_all_boxplots(results, show_datapoints=True, show_ids=False, show_lines=False, show=True):
    """3×3 overview: signals in columns, DFA/CV/sample entropy in rows."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True, facecolor="white")
    for row, metric in enumerate(METRIC_LABELS):
        for col, signal in enumerate(signals):
            _draw_boxplot(axes[row, col], results, signal, metric,
                          show_datapoints, show_ids, show_lines)
    fig.suptitle("Peak interval metrics across inclines", fontsize=17, fontweight="bold")
    if show:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    results = analyze_peaks()
    plot_all_boxplots(results, show_datapoints=True, show_ids=False, show_lines=True)
    # Examples (reuse results without recalculating):
    # plot_metric_boxplots(results, metric="CV", show_datapoints=True, show_ids=True, show_lines=True)
    # plot_metric_boxplots(results, metric="Ent_Samp", signal="ECG", show_datapoints=False)
