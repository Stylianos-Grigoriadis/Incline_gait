from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_dcca_steps(x, y, scales):
    """Vectorized DCCA demonstration with covariance and correlation curves."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Provide two aligned one-dimensional series of equal length.")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Provide nonempty series with finite values.")
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or not scales.size or not np.all(np.isfinite(scales)):
        raise ValueError("Provide a nonempty list of finite scales.")
    if np.any(scales != np.floor(scales)) or np.any(scales < 3) or np.any(scales > len(x)):
        raise ValueError("Scales must be integers from 3 to the paired series length.")
    scales = np.unique(scales.astype(int))
    profiles = np.cumsum(np.stack([x - x.mean(), y - y.mean()]), axis=1)
    results, plot_data, layouts = {}, {}, {}
    shape_count = 3 * (len(x) // int(scales[0]) + 1) + 1

    # NaNs separate independent line segments without one Plotly trace per segment.
    def segments(left, right, bottom, top):
        left, right, bottom, top = np.broadcast_arrays(left, right, bottom, top)
        gap = np.full(left.shape, np.nan)
        return (np.stack([left, right, gap], axis=-1).ravel(),
                np.stack([bottom, top, gap], axis=-1).ravel())

    for scale in scales:
        used = len(x) // scale * scale
        positions = np.arange(1, used + 1)
        windows = profiles[:, :used].reshape(2, -1, scale)
        local_x = np.arange(scale, dtype=float)
        local_x -= local_x.mean()
        # Least-squares slopes and fits for ALL windows in both signals at once.
        slopes = np.sum(windows * local_x, axis=2) / np.sum(local_x ** 2)
        fitted = windows.mean(axis=2, keepdims=True) + slopes[:, :, None] * local_x
        residuals = (windows - fitted).reshape(2, used)
        products = residuals[0] * residuals[1]
        means = products.reshape(-1, scale).mean(axis=1)
        variance_x, variance_y = np.mean(residuals ** 2, axis=1)
        covariance_at_scale = float(means.mean())
        denominator = np.sqrt(variance_x) * np.sqrt(variance_y)
        correlation = covariance_at_scale / denominator if denominator > 0 else np.nan
        results[int(scale)] = dict(residual_x=residuals[0], residual_y=residuals[1],
                                  products=products, window_means=means,
                                  covariance=covariance_at_scale,
                                  variance_x=float(variance_x), variance_y=float(variance_y),
                                  correlation=float(correlation), omitted=len(x) - used)
        traces = []
        for row in range(2):
            values, fits = profiles[row, :used], fitted[row].ravel()
            traces.append(go.Scatter(x=positions, y=values, mode="lines+markers",
                name=f"Profile {'X' if row == 0 else 'Y'}",
                line=dict(color="gray"), marker=dict(color="black", size=5)))
            # Each straight fit needs only its two endpoints.
            window_positions = positions.reshape(-1, scale)
            fx, fy = segments(window_positions[:, 0], window_positions[:, -1],
                              fitted[row, :, 0], fitted[row, :, -1])
            traces.append(go.Scatter(x=fx, y=fy, mode="lines",
                line=dict(color="black", dash="dash"), name="Local fit",
                legendgroup="fit", showlegend=(row == 0)))
            for color, mask in [("red", residuals[row] < 0), ("blue", residuals[row] >= 0)]:
                sx, sy = segments(positions[mask], positions[mask], fits[mask], values[mask])
                traces.append(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color),
                    name="Below fit" if color == "red" else "Above fit",
                    legendgroup=color, showlegend=(row == 0), hoverinfo="skip"))
        traces.append(go.Scatter(x=positions, y=products, mode="lines+markers",
                                 name="Residual X * residual Y", line=dict(color="purple")))
        starts = np.arange(len(means)) * scale + 0.5
        mx, my = segments(starts, starts + scale, means, means)
        traces.append(go.Scatter(x=mx, y=my, mode="lines", name="Window average",
                                 line=dict(color="darkorange", width=3)))
        plot_data[int(scale)] = traces
        # Build boundaries in one batch; repeated add_vline calls were very expensive.
        shapes = [dict(type="line", xref=f"x{row if row > 1 else ''}",
                       yref=f"y{row if row > 1 else ''} domain",
                       x0=float(boundary), x1=float(boundary), y0=0, y1=1,
                       line=dict(color="gray", dash="dot", width=1))
                  for row in range(1, 4) for boundary in np.arange(0.5, used + 1, scale)]
        shapes.append(dict(type="line", xref="x3 domain", yref="y3",
                           x0=0, x1=1, y0=0, y1=0, line=dict(color="gray", width=1)))
        # Plotly merges shape arrays during animation: explicitly hide unused slots.
        for shape in shapes:
            shape["visible"] = True
        shapes.extend([dict(type="line", x0=0, x1=0, y0=0, y1=0, visible=False)
                       for _ in range(shape_count - len(shapes))])
        layouts[int(scale)] = dict(shapes=shapes,
            title=f"DCCA steps: {scale}-point windows; {len(x) - used} trailing points omitted")

    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.055,
        subplot_titles=["X: cumulative profile and local fits", "Y: cumulative profile and local fits",
                        "Residual products and window averages", "Average of window averages at each scale",
                        "Normalized DCCA correlation at each scale"])
    first = int(scales[0])
    for trace, row in zip(plot_data[first], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]):
        fig.add_trace(trace, row=row, col=1)
    covariance = [results[int(s)]["covariance"] for s in scales]
    fig.add_trace(go.Scatter(x=scales, y=covariance, mode="lines+markers",
        name="Average of window averages", line=dict(color="teal")), row=4, col=1)
    fig.add_trace(go.Scatter(x=[first], y=[results[first]["covariance"]], mode="markers",
        marker=dict(color="red", size=12), name="Selected scale"), row=4, col=1)
    correlations = [results[int(s)]["correlation"] for s in scales]
    fig.add_trace(go.Scatter(x=scales, y=correlations, mode="lines+markers",
        name="DCCA correlation", line=dict(color="royalblue"),
        hovertemplate="Scale: %{x}<br>DCCA correlation: %{y:.4f}<extra></extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=[first], y=[results[first]["correlation"]], mode="markers",
        marker=dict(color="red", size=12), name="Selected correlation scale", showlegend=False), row=5, col=1)
    fig.frames = [go.Frame(name=str(int(s)), traces=list(range(10)) + [11, 13],
        data=plot_data[int(s)] + [go.Scatter(x=[int(s)], y=[results[int(s)]["covariance"]]),
                                 go.Scatter(x=[int(s)], y=[results[int(s)]["correlation"]])],
        layout=go.Layout(**layouts[int(s)])) for s in scales]
    fig.update_xaxes(matches="x3", row=1, col=1)
    fig.update_xaxes(matches="x3", row=2, col=1)
    fig.update_xaxes(title_text="Paired sample number", row=3, col=1)
    fig.update_xaxes(title_text="Window size (points)", row=4, col=1)
    fig.update_xaxes(title_text="Window size (points)", matches="x4", row=5, col=1)
    fig.update_yaxes(range=[-1.05, 1.05], zeroline=True, zerolinecolor="gray", row=5, col=1)
    for row, label in enumerate(["Cumulative X", "Cumulative Y", "Residual product", "Detrended covariance", "DCCA correlation"], 1):
        fig.update_yaxes(title_text=label, row=row, col=1)
    fig.update_layout(**layouts[first], template="plotly_white", height=1500,
        sliders=[dict(active=0, currentvalue=dict(prefix="Window size: "), steps=[
            dict(label=str(int(s)), method="animate", args=[[str(int(s))],
                 dict(mode="immediate", frame=dict(duration=0, redraw=True),
                      transition=dict(duration=0))]) for s in scales])])
    fig.show()
    return fig, results


directory = Path(r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Peaks Data")

signals = ["ECG", "Gastrocnemius_EMG", "IMU"]
inclinations = ["0", "12.5", "25"]

# Original Excel contents and peak timestamps; no exclusions or analysis applied.
participant_data = {}
peak_times_data = {}
for file in sorted(directory.glob("P*_Peaks.xlsx"), key=lambda path: int(path.stem.split("_")[0][1:])):
    participant = file.stem.replace("_Peaks", "")
    data = pd.read_excel(file)
    participant_data[participant] = data
    peak_times_data[participant] = {}
    for inclination in inclinations:
        peak_times_data[participant][inclination] = {}
        for signal in signals:
            column = f"{signal}_Peak_Time_{inclination}_deg"
            peak_times_data[participant][inclination][signal] = data[column].dropna().to_numpy()

# Demonstration using consecutive peak-time differences (interval durations in seconds).
# Pair by interval number only; temporal matching must be established before interpretation.
ID = "P10"
inclination = "0"
imu_intervals = np.diff(peak_times_data[ID][inclination]["IMU"])
emg_intervals = np.diff(peak_times_data[ID][inclination]["Gastrocnemius_EMG"])
paired_count = min(len(imu_intervals), len(emg_intervals))
print(f"{ID}, {inclination} degrees: pairing {paired_count} intervals by number for demonstration.")
scales = np.linspace(10,100,91)
print(scales)
fig_dcca, dcca_steps = plot_dcca_steps(
    imu_intervals[:paired_count], emg_intervals[:paired_count], scales=scales
)

# Previous analysis preserved below as comments.
# import os
# import numpy as np
# import pandas as pd
# import lib
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from library import analysis
# import matplotlib.pyplot as plt
# 
# fs_emg = 2148.1481
# fs_imu = 370.3704
# 
# exclude_first_seconds = 10
# exclude_last_seconds = 10
# ami_max_lag = 100  # Maximum lag in interval samples, not recording seconds.
# review_ami = False  # Set True to review AMI plots again.
# review_dfa = False  # Participant DFA review is commented out below.
# entropy_m = 2
# entropy_r = 0.2  # Library multiplies this by each interval series' SD.
# # Time windows are measured from each signal's first and last peak timestamps.
# # Keep only intervals whose BOTH endpoints lie within the retained window.
# if exclude_first_seconds < 0 or exclude_last_seconds < 0:
#     raise ValueError("Time exclusions must be non-negative.")
# 
# 
# directory = Path(r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Peaks Data")
# 
# signals = ["ECG", "Gastrocnemius_EMG", "IMU"]
# inclinations = ["0", "12.5", "25"]
# 
# intervals = {}
# 
# for file in directory.glob("P*_Peaks.xlsx"):
#     participant = file.stem.replace("_Peaks", "")
#     data = pd.read_excel(file)
#     intervals[participant] = {}
# 
#     for inclination in inclinations:
#         intervals[participant][inclination] = {}
# 
#         for signal in signals:
#             column = f"{signal}_Peak_Time_{inclination}_deg"
#             peak_times = data[column].dropna().to_numpy()
# 
#             signal_intervals = np.diff(peak_times)
#             if len(peak_times):
#                 start_time = peak_times[0] + exclude_first_seconds
#                 end_time = peak_times[-1] - exclude_last_seconds
#                 keep = (peak_times[:-1] >= start_time) & (peak_times[1:] <= end_time)
#                 signal_intervals = signal_intervals[keep]
#             intervals[participant][inclination][signal] = signal_intervals
# 
# # Example: ECG intervals for P1 at 12.5 degrees
# intervals_ECG = intervals["P1"]["12.5"]["ECG"]
# 
# 
# def create_interval_table(intervals, signal):
#     statistics = ["SD (s)", "Count", "Below Q1 - 2 IQR", "Above Q3 + 2 IQR"]
#     columns = pd.MultiIndex.from_product(
#         [inclinations, statistics], names=["Inclination", "Statistic"]
#     )
#     participants = sorted(intervals, key=lambda participant: int(participant[1:]))
#     rows = []
# 
#     for participant in participants:
#         row = []
#         for inclination in inclinations:
#             values = intervals[participant][inclination][signal]
#             count = len(values)
#             sd = np.std(values, ddof=1) if count >= 2 else np.nan
#             below_count = 0
#             above_count = 0
# 
#             if count:
#                 # Flag potential outliers without changing or removing intervals.
#                 q1, q3 = np.percentile(values, [25, 75])
#                 iqr = q3 - q1
#                 below_count = np.count_nonzero(values < q1 - 2 * iqr)
#                 above_count = np.count_nonzero(values > q3 + 2 * iqr)
# 
#             row.extend([sd, count, below_count, above_count])
#         rows.append(row)
# 
#     table = pd.DataFrame(rows, index=participants, columns=columns)
#     table.index.name = "Participant"
#     for inclination in inclinations:
#         for statistic in statistics[1:]:
#             table[(inclination, statistic)] = table[(inclination, statistic)].astype(int)
#     return table
# 
# 
# def plot_participant_intervals(intervals, participant):
#     """Show retained peak-to-peak intervals and IQR limits for one participant."""
#     if participant not in intervals:
#         raise ValueError(f"Unknown participant: {participant}")
# 
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     titles = [
#         f"{label} - {inclination} degrees"
#         for label in labels for inclination in inclinations
#     ] + [f"Summary - {inclination} degrees" for inclination in inclinations]
#     fig = make_subplots(
#         rows=4, cols=3,
#         specs=[[{"type": "xy"}] * 3 for _ in range(3)]
#         + [[{"type": "table"}] * 3],
#         subplot_titles=titles,
#         row_heights=[0.27, 0.27, 0.27, 0.19],
#         vertical_spacing=0.07,
#         horizontal_spacing=0.08,
#     )
# 
#     summaries = []
#     for col, inclination in enumerate(inclinations, start=1):
#         summary = []
#         for row, (signal, label) in enumerate(zip(signals, labels), start=1):
#             values = intervals[participant][inclination][signal]
#             count = len(values)
#             # Interval numbers refer to the retained sequence after end exclusions.
#             x = np.arange(1, count + 1)
#             fig.add_trace(go.Scatter(
#                 x=x, y=values, mode="lines+markers", name=label,
#                 line=dict(color="steelblue"), marker=dict(size=4),
#                 showlegend=False,
#                 hovertemplate="Interval %{x}<br>Duration %{y:.3f} s<extra></extra>",
#             ), row=row, col=col)
# 
#             below = above = 0
#             sd = np.std(values, ddof=1) if count >= 2 else np.nan
#             if count:
#                 q1, q3 = np.percentile(values, [25, 75])
#                 iqr = q3 - q1
#                 lower, upper = q1 - 2 * iqr, q3 + 2 * iqr
#                 mask = (values < lower) | (values > upper)
#                 below = np.count_nonzero(values < lower)
#                 above = np.count_nonzero(values > upper)
#                 for level, name, color in [
#                     (q1, "Q1", "darkorange"),
#                     (q3, "Q3", "darkorange"),
#                     (lower, "Q1 - 2 IQR", "crimson"),
#                     (upper, "Q3 + 2 IQR", "crimson"),
#                 ]:
#                     fig.add_hline(
#                         y=level, line_dash="dash", line_color=color,
#                         annotation_text=name, annotation_position="top left",
#                         row=row, col=col,
#                     )
#                 fig.add_trace(go.Scatter(
#                     x=x[mask], y=values[mask], mode="markers",
#                     marker=dict(color="crimson", size=8),
#                     name="Potential outlier", showlegend=False,
#                     hovertemplate="Interval %{x}<br>Duration %{y:.3f} s<extra>Potential outlier</extra>",
#                 ), row=row, col=col)
# 
#             summary.append([label, f"{sd:.3f}", count, below, above])
#             fig.update_xaxes(title_text="Retained interval number", row=row, col=col)
#             fig.update_yaxes(title_text="Interval (s)", row=row, col=col)
# 
#         summaries.append(summary)
# 
#     for col, summary in enumerate(summaries, start=1):
#         fig.add_trace(go.Table(
#             header=dict(values=["Signal", "SD (s)", "Count", "Below<br>Q1 - 2 IQR", "Above<br>Q3 + 2 IQR"]),
#             cells=dict(values=list(zip(*summary))),
#             columnwidth=[1.8, 1, 1, 1.4, 1.4],
#         ), row=4, col=col)
# 
#     fig.update_layout(
#         title=f"{participant}: peak-to-peak intervals (orange: quartiles; red: 2 IQR limits)",
#         height=1200, width=1600, template="plotly_white",
#     )
#     fig.show()
#     return fig
# 
# 
# def review_interval_ami(intervals, max_lag=100):
#     """Review library AMI curves; return candidates, curves, and suggested delays."""
#     if max_lag < 3:
#         raise ValueError("AMI max_lag must be at least 3.")
#     results = {}
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     for participant in sorted(intervals, key=lambda name: int(name[1:])):
#         results[participant] = {}
#         fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
#         fig.suptitle(f"{participant} - AMI of retained intervals (close to continue)")
#         for col, inclination in enumerate(inclinations):
#             results[participant][inclination] = {}
#             for row, (signal, label) in enumerate(zip(signals, labels)):
#                 ax = axes[row, col]
#                 values = intervals[participant][inclination][signal]
#                 result = {"tau": np.nan, "candidates": None, "curve": None}
#                 results[participant][inclination][signal] = result
#                 ax.set_title(f"{label} - {inclination} degrees (N={len(values)})")
#                 ax.set_xlabel("Lag (interval samples)")
#                 ax.set_ylabel("AMI (bits)")
#                 ax.grid(alpha=0.3)
#                 if len(values) < 4 or not np.all(np.isfinite(values)) or np.std(values) == 0:
#                     ax.text(0.5, 0.5, "AMI unavailable: short, constant, or nonfinite data",
#                             ha="center", va="center", transform=ax.transAxes, wrap=True)
#                     continue
# 
#                 # fs=1 maps the library's lag window directly to interval samples.
#                 # The library calculates lags 0 through L-1.
#                 lag_window = min(int(max_lag) + 1, len(values) - 1)
#                 candidates, curve = analysis.average_mutual_information(
#                     values, L_seconds=lag_window, fs=1, n_bins=0, plot=False
#                 )
#                 result["candidates"] = candidates
#                 result["curve"] = curve
#                 ax.plot(curve[0], curve[1], "o-", markersize=3)
#                 valid = candidates[
#                     np.isfinite(candidates[:, 0]) & (candidates[:, 0] > 0)
#                 ]
#                 if len(valid):
#                     delay = int(valid[0, 0])
#                     result["tau"] = delay
#                     ax.axvline(delay, color="crimson", linestyle="--",
#                                label=f"Suggested tau = {delay}")
#                     ax.plot(valid[:, 0], valid[:, 1], "o", color="darkorange",
#                             label="Library candidates")
#                     ax.legend(fontsize=8)
#                 else:
#                     ax.text(0.5, 0.9, "No positive delay candidate",
#                             ha="center", transform=ax.transAxes)
#                 print(f"AMI {participant}, {inclination} degrees, {signal}: tau={result['tau']}")
#         fig.tight_layout(rect=(0, 0, 1, 0.95))
#         plt.show(block=True)
#     return results
# 
# 
# def add_participant_lines(fig, participants):
#     """Connect matching participants across the three boxes in each signal panel."""
#     boxes = list(fig.data)
#     for col, signal in enumerate(signals, start=1):
#         signal_boxes = boxes[(col - 1) * len(inclinations):col * len(inclinations)]
#         positions = [f"{inclination} degrees" for inclination in inclinations]
#         for box, position in zip(signal_boxes, positions):
#             box.x = [position] * len(box.y)
#             box.jitter = 0
#         for number, participant in enumerate(participants):
#             values = []
#             for box in signal_boxes:
#                 value = None
#                 for identity, y in zip(box.customdata, box.y):
#                     identity = identity if isinstance(identity, str) else identity[0]
#                     if identity == participant:
#                         value = y
#                         break
#                 values.append(value)
#             if all(value is None for value in values):
#                 continue
#             fig.add_trace(go.Scatter(
#                 x=positions, y=values, mode="lines+markers",
#                 name=participant, legendgroup=f"participant_{participant}",
#                 showlegend=(col == 1), connectgaps=False,
#                 line=dict(color=f"hsl({number * 137.5 % 360},55%,45%)", width=1),
#                 marker=dict(size=5), opacity=0.65,
#                 hovertemplate=f"{participant}<br>{signal}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
#             ), row=1, col=col)
#         fig.update_xaxes(categoryorder="array", categoryarray=positions, row=1, col=col)
#     fig.update_layout(legend=dict(groupclick="togglegroup"))
# 
# 
# def plot_interval_cv(intervals):
#     """One CV (%) per participant and condition, using retained intervals."""
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
#     participants = sorted(intervals, key=lambda name: int(name[1:]))
#     colors = ["steelblue", "darkorange", "seagreen"]
# 
#     for col, signal in enumerate(signals, start=1):
#         for inclination, color in zip(inclinations, colors):
#             cv_values = []
#             ids = []
#             for participant in participants:
#                 values = intervals[participant][inclination][signal]
#                 if len(values) < 2:
#                     continue
#                 mean = np.mean(values)
#                 if not np.isfinite(mean) or mean <= 0:
#                     continue
#                 cv = 100 * np.std(values, ddof=1) / mean
#                 if np.isfinite(cv):
#                     cv_values.append(cv)
#                     ids.append(participant)
# 
#             fig.add_trace(go.Box(
#                 y=cv_values, name=f"{inclination} degrees",
#                 customdata=ids, boxpoints="all", jitter=0.3,
#                 pointpos=0, marker_color=color,
#                 legendgroup=inclination, showlegend=(col == 1),
#                 hovertemplate="%{customdata}<br>CV: %{y:.3f}%<extra>%{fullData.name}</extra>",
#             ), row=1, col=col)
#         fig.update_xaxes(title_text="Inclination", row=1, col=col)
#         fig.update_yaxes(title_text="Interval CV (%)", rangemode="tozero", row=1, col=col)
# 
#     fig.update_layout(
#         title="Interval coefficient of variation across participants (SD / mean x 100)",
#         template="plotly_white", width=1400, height=600,
#     )
#     add_participant_lines(fig, participants)
#     fig.show()
#     return fig
# 
# 
# def plot_interval_dfa(intervals):
#     """Library DFA with linear detrending, scales 10 through floor(N/10)."""
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     participants = sorted(intervals, key=lambda name: int(name[1:]))
#     columns = pd.MultiIndex.from_product(
#         [inclinations, signals], names=["Inclination", "Signal"]
#     )
#     table = pd.DataFrame(np.nan, index=participants, columns=columns)
#     table.index.name = "Participant"
#     details = {}
#     fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
#     colors = ["steelblue", "darkorange", "seagreen"]
#     for col, signal in enumerate(signals, start=1):
#         for inclination, color in zip(inclinations, colors):
#             alphas, ids = [], []
#             for participant in participants:
#                 values = intervals[participant][inclination][signal]
#                 maximum = len(values) // 10
#                 if maximum <= 10 or not np.all(np.isfinite(values)) or np.std(values) == 0:
#                     print(f"DFA unavailable: {participant}, {inclination}, {signal}")
#                     continue
#                 # Twenty logarithmically spaced integer scales, including both endpoints.
#                 scales = np.unique(np.rint(np.geomspace(10, maximum, 20)).astype(int))
#                 scales, fluctuations, alpha = analysis.dfa(values, scales, order=1, plot=False)
#                 table.loc[participant, (inclination, signal)] = alpha
#                 details[(participant, inclination, signal)] = {
#                     "scales": scales, "fluctuations": fluctuations, "alpha": alpha
#                 }
#                 if np.isfinite(alpha):
#                     alphas.append(alpha)
#                     ids.append(participant)
#                 else:
#                     print(f"Nonfinite DFA alpha: {participant}, {inclination}, {signal}")
#             fig.add_trace(go.Box(
#                 y=alphas, name=f"{inclination} degrees", customdata=ids,
#                 boxpoints="all", jitter=0.3, pointpos=0, marker_color=color,
#                 legendgroup=inclination, showlegend=(col == 1),
#                 hovertemplate="%{customdata}<br>DFA alpha: %{y:.3f}<extra>%{fullData.name}</extra>",
#             ), row=1, col=col)
#         fig.update_xaxes(title_text="Inclination", row=1, col=col)
#         fig.update_yaxes(title_text="DFA alpha", row=1, col=col)
#     fig.update_layout(
#         title="Interval DFA (linear detrending; scales 10 to floor(N/10))",
#         template="plotly_white", width=1400, height=600,
#     )
#     add_participant_lines(fig, participants)
#     fig.show()
#     return fig, table, details
# 
# 
# def review_interval_dfa(intervals, dfa_results):
#     """Review previously calculated DFA curves without recalculating DFA."""
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     for participant in sorted(intervals, key=lambda name: int(name[1:])):
#         fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
#         fig.suptitle(f"{participant} - DFA of retained intervals (close to continue)")
#         for col, inclination in enumerate(inclinations):
#             for row, (signal, label) in enumerate(zip(signals, labels)):
#                 ax = axes[row, col]
#                 ax.set_title(f"{label} - {inclination} degrees")
#                 ax.set_xlabel("Window size (interval samples)")
#                 ax.set_ylabel("Fluctuation F(n)")
#                 ax.set_xscale("log")
#                 ax.set_yscale("log")
#                 ax.grid(True, which="both", alpha=0.3)
#                 result = dfa_results.get((participant, inclination, signal))
#                 if result is None:
#                     ax.text(0.5, 0.5, "DFA unavailable", ha="center",
#                             va="center", transform=ax.transAxes)
#                     continue
#                 scales = np.asarray(result["scales"])
#                 fluctuations = np.asarray(result["fluctuations"])
#                 alpha = result["alpha"]
#                 valid = np.isfinite(fluctuations) & (fluctuations > 0)
#                 ax.plot(scales[valid], fluctuations[valid], "o-",
#                         markersize=4, label="DFA fluctuations")
#                 if np.all(valid) and np.isfinite(alpha):
#                     # Same log-log regression used inside analysis.dfa().
#                     intercept = np.mean(np.log(fluctuations) - alpha * np.log(scales))
#                     fitted = np.exp(intercept) * scales ** alpha
#                     ax.plot(scales, fitted, "--", color="crimson",
#                             label=f"Fit: alpha = {alpha:.3f}")
#                 else:
#                     ax.text(0.5, 0.9, "Fit unavailable: nonfinite or zero fluctuations",
#                             ha="center", transform=ax.transAxes)
#                 ax.legend(fontsize=8)
#         fig.tight_layout(rect=(0, 0, 1, 0.95))
#         plt.show(block=True)
# 
# 
# def plot_interval_sample_entropy(intervals, m=2, r=0.2, max_lag=100):
#     """Sample entropy using each retained series' first positive AMI candidate."""
#     labels = ["ECG", "Gastrocnemius EMG", "IMU"]
#     participants = sorted(intervals, key=lambda name: int(name[1:]))
#     columns = pd.MultiIndex.from_product(
#         [inclinations, signals], names=["Inclination", "Signal"]
#     )
#     table = pd.DataFrame(np.nan, index=participants, columns=columns)
#     table.index.name = "Participant"
#     table_tau = table.copy()
#     fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
#     colors = ["steelblue", "darkorange", "seagreen"]
# 
#     for col, signal in enumerate(signals, start=1):
#         for inclination, color in zip(inclinations, colors):
#             entropy_values = []
#             ids = []
#             delays = []
#             for participant in participants:
#                 values = intervals[participant][inclination][signal]
#                 if len(values) <= max(m + 1, 3) or not np.all(np.isfinite(values)) or np.std(values) == 0:
#                     print(f"Sample entropy unavailable: {participant}, {inclination}, {signal}")
#                     continue
#                 # fs=1 expresses AMI lags in interval samples.
#                 candidates, _ = analysis.average_mutual_information(
#                     values, L_seconds=min(int(max_lag) + 1, len(values) - 1),
#                     fs=1, n_bins=0, plot=False,
#                 )
#                 valid = candidates[np.isfinite(candidates[:, 0]) & (candidates[:, 0] > 0)]
#                 if not len(valid):
#                     print(f"No AMI delay: {participant}, {inclination}, {signal}")
#                     continue
#                 tau = int(valid[0, 0])
#                 table_tau.loc[participant, (inclination, signal)] = tau
#                 entropy = analysis.sample_entropy(values, m=m, r=r, tau=tau)
#                 table.loc[participant, (inclination, signal)] = entropy
#                 if np.isfinite(entropy):
#                     entropy_values.append(entropy)
#                     ids.append(participant)
#                     delays.append(tau)
#                 else:
#                     print(f"Nonfinite sample entropy (not plotted): {participant}, {inclination}, {signal}: {entropy}")
# 
#             fig.add_trace(go.Box(
#                 y=entropy_values, name=f"{inclination} degrees",
#                 customdata=list(zip(ids, delays)), boxpoints="all", jitter=0.3,
#                 pointpos=0, marker_color=color,
#                 legendgroup=inclination, showlegend=(col == 1),
#                 hovertemplate="%{customdata[0]}<br>Sample entropy: %{y:.3f}<br>AMI tau: %{customdata[1]}<extra>%{fullData.name}</extra>",
#             ), row=1, col=col)
#         fig.update_xaxes(title_text="Inclination", row=1, col=col)
#         fig.update_yaxes(title_text="Sample entropy (nats)", rangemode="tozero", row=1, col=col)
# 
#     fig.update_layout(
#         title=f"Interval sample entropy (m={m}, r={r} x SD, tau from each series' AMI)",
#         template="plotly_white", width=1400, height=600,
#     )
#     add_participant_lines(fig, participants)
#     fig.show()
#     return fig, table, table_tau
# 
# 
# # Example: uncomment to inspect one participant when running the script.
# # fig = plot_participant_intervals(intervals, "P1")
# 
# 
# # ECG counts are heartbeat intervals; gait counts are strides only if each peak is one stride.
# # Retained intervals stay unchanged, including zeros from repeated timestamps.
# table_ECG = create_interval_table(intervals, "ECG")
# table_Gastrocnemius = create_interval_table(intervals, "Gastrocnemius_EMG")
# table_IMU = create_interval_table(intervals, "IMU")
# 
# # Combined counts after excluding the opening and closing time windows.
# participants = sorted(intervals, key=lambda name: int(name[1:]))
# table_counts = pd.DataFrame(
#     [[len(intervals[participant][inclination][signal])
#       for inclination in inclinations for signal in signals]
#      for participant in participants],
#     index=participants,
#     columns=pd.MultiIndex.from_product(
#         [inclinations, signals], names=["Inclination", "Signal"]
#     ),
#     dtype=int,
# )
# table_counts.index.name = "Participant"
# 
# with pd.option_context(
#     "display.max_columns", None,
#     "display.max_rows", None,
#     "display.expand_frame_repr", False,
#     "display.float_format", "{:.3f}".format,
# ):
#     print(f"\nINTERVAL COUNTS - ALL PARTICIPANTS (excluded: first {exclude_first_seconds} s, last {exclude_last_seconds} s)")
#     print(table_counts)
#     print("\nECG")
#     print(table_ECG)
#     print("\nGastrocnemius EMG")
#     print(table_Gastrocnemius)
#     print("\nIMU")
#     print(table_IMU)
# 
# if review_ami:
#     ami_results = review_interval_ami(intervals, max_lag=ami_max_lag)
# 
# fig_cv = plot_interval_cv(intervals)
# fig_dfa, table_dfa, dfa_results = plot_interval_dfa(intervals)
# # if review_dfa:
# #     review_interval_dfa(intervals, dfa_results)
# fig_entropy, table_entropy, table_tau = plot_interval_sample_entropy(
#     intervals, m=entropy_m, r=entropy_r, max_lag=ami_max_lag
# )
# with pd.option_context("display.max_columns", None, "display.expand_frame_repr", False):
#     print("\nAMI delays (interval samples)")
#     print(table_tau)
# 
# # list_ID = [
# # "P1",
# # "P2",
# # "P4",
# # "P5",
# # "P7",
# # "P8",
# # "P9",
# # "P10",
# # "P12",
# # "P13",
# # "P16",
# # "P19",
# # "P20",
# # ]
# # for ID in list_ID:
# #     plot_participant_intervals(intervals, ID)
# # plot_participant_intervals(intervals, "P22")
