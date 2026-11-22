"""Compare interval variability and gait-signal coupling across inclinations."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from itertools import combinations
from pathlib import Path
import re
from time import monotonic
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from library import analysis


save_HTML = True
save_results_excel = True
save_jpeg = True
statistical_analysis = True

project = (
    Path.home() / "OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης"
    / "My Files" / "Projects" / "Inclined gait"
)
peak_directory = project / "Data" / "Peak Data per device"
devices = [
    ("ECG", "ECG", "Peak ECG"),
    ("Gastrocnemius EMG", "Gastrocnemius_EMG", "Peak Gast"),
    ("Quadriceps EMG", "Quadriceps_EMG", "Peak Quad"),
    ("IMU", "IMU", "Peak IMU"),
]
report_path = project / "Results" / "Peak_interval_results.html"
jpeg_directory = project / "Figures" / "Results"
excel_path = project / "Results" / "Peak_interval_results.xlsx"
inclinations = ("0", "12.5", "25")
dcca_ranges = ("Short", "Long")
dcca_offsets = (-0.25, 0, 0.25)
pairs = [
    ("IMU", "Gastrocnemius_EMG", "IMU–Gastrocnemius EMG"),
    ("IMU", "Quadriceps_EMG", "IMU–Quadriceps EMG"),
]
signal_names = {
    "ECG": "HR",
    "Gastrocnemius_EMG": "EMG_Gast",
    "Quadriceps_EMG": "EMG_VL",
    "IMU": "SI",
}


def dcca(x, y, scales, plot=False):
    """Calculate detrended cross-correlation for two paired series.

    Pass two aligned, one-dimensional series of equal length and a sequence
    of integer window sizes measured in paired observations. The function
    centers and integrates each series, then removes a linear trend within
    each complete, non-overlapping window. At each scale, it normalizes the
    mean product of paired residuals by the product of their RMS values.

    Returns three NumPy arrays in order: slopes, normalized correlations,
    and sorted unique scales. The slopes describe the change in correlation
    per additional observation in the window, fitted over all scales, scales
    at or below the 25th percentile, and scales at or above the 75th percentile.
    A slope is NaN when its range has fewer than two finite correlations.
    Set plot=True to display the interactive DCCA plots.
    """

    ################################
    ############ CHECKS ############
    ################################

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Provide two aligned one-dimensional series of equal length.")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Provide nonempty series with finite values.")
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or not scales.size or not np.all(np.isfinite(scales)):
        raise ValueError("Provide a nonempty list of finite scales.")
    if np.any(scales != np.floor(scales)) or np.any(scales < 3) or np.any(scales > len(x)):
        raise ValueError("Scales must be integers from 3 to the paired series length.")

    #####################################
    ############ CALCULATION ############
    #####################################

    scales = np.unique(scales.astype(int))
    # Build the cumulative profiles from the mean-centered signals.
    profiles = np.cumsum(np.stack([x - x.mean(), y - y.mean()]), axis=1)
    correlations = np.empty(scales.size)
    if plot:
        covariance = np.empty(scales.size)
        window_details = []

    for index, scale in enumerate(scales):
        # Use only complete windows at this scale.
        used = len(x) // scale * scale
        windows = profiles[:, :used].reshape(2, -1, scale)
        local_x = np.arange(scale, dtype=float)
        local_x -= local_x.mean()
        # Least-squares slopes and fits for ALL windows in both signals at once.
        slopes = np.sum(windows * local_x, axis=2) / np.sum(local_x ** 2)
        fitted = windows.mean(axis=2, keepdims=True) + slopes[:, :, None] * local_x
        residuals = (windows - fitted).reshape(2, used)

        # Normalize the average residual product by the two residual RMS values.
        products = residuals[0] * residuals[1]
        means = products.reshape(-1, scale).mean(axis=1)
        variance_x, variance_y = np.mean(residuals ** 2, axis=1)
        covariance_at_scale = float(means.mean())
        denominator = np.sqrt(variance_x) * np.sqrt(variance_y)
        correlation = covariance_at_scale / denominator if denominator > 0 else np.nan
        correlations[index] = correlation
        if plot:
            covariance[index] = covariance_at_scale
            window_details.append((fitted, residuals, products, means))

    ###################################
    ############ SLOPE FIT ############
    ###################################

    lower_quartile, upper_quartile = np.percentile(scales, [25, 75])
    finite = np.isfinite(correlations)
    fit_masks = [
        finite,
        finite & (scales <= lower_quartile),
        finite & (scales >= upper_quartile),
    ]
    fit_slopes = np.full(len(fit_masks), np.nan)
    fit_intercepts = np.full(len(fit_masks), np.nan)
    for index, mask in enumerate(fit_masks):
        fit_scales = scales[mask]
        if fit_scales.size >= 2:
            fit_slopes[index], fit_intercepts[index] = np.polyfit(
                fit_scales, correlations[mask], 1
            )

    #################################
    ############ PLOTTING ###########
    #################################

    if plot:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plot_data, layouts = {}, {}
        shape_count = 3 * (profiles.shape[1] // int(scales[0]) + 1) + 1

        # NaNs separate independent line segments in each trace.
        for scale, (fitted, residuals, products, means) in zip(scales, window_details):
            used = products.size
            positions = np.arange(1, used + 1)
            traces = []
            for row in range(2):
                values, fits = profiles[row, :used], fitted[row].ravel()
                traces.append(go.Scatter(x=positions, y=values, mode="lines+markers",
                    name=f"Profile {'X' if row == 0 else 'Y'}",
                    line=dict(color="gray"), marker=dict(color="black", size=5)))
                # Each straight fit needs only its two endpoints.
                window_positions = positions.reshape(-1, scale)
                gaps = np.full(len(window_positions), np.nan)
                fx = np.column_stack((window_positions[:, 0], window_positions[:, -1], gaps)).ravel()
                fy = np.column_stack((fitted[row, :, 0], fitted[row, :, -1], gaps)).ravel()
                traces.append(go.Scatter(x=fx, y=fy, mode="lines",
                    line=dict(color="black", dash="dash"), name="Local fit",
                    legendgroup="fit", showlegend=(row == 0)))
                for color, mask in [("red", residuals[row] < 0), ("blue", residuals[row] >= 0)]:
                    selected = positions[mask]
                    gaps = np.full(selected.size, np.nan)
                    sx = np.column_stack((selected, selected, gaps)).ravel()
                    sy = np.column_stack((fits[mask], values[mask], gaps)).ravel()
                    traces.append(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color),
                        name="Below fit" if color == "red" else "Above fit",
                        legendgroup=color, showlegend=(row == 0), hoverinfo="skip"))
            traces.append(go.Scatter(x=positions, y=products, mode="lines+markers",
                                     name="Residual X * residual Y", line=dict(color="purple")))
            starts = np.arange(len(means)) * scale + 0.5
            gaps = np.full(means.size, np.nan)
            mx = np.column_stack((starts, starts + scale, gaps)).ravel()
            my = np.column_stack((means, means, gaps)).ravel()
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
                title=f"DCCA steps: {scale}-point windows; {profiles.shape[1] - used} trailing points omitted")

        fig = make_subplots(rows=5, cols=1, vertical_spacing=0.055,
            subplot_titles=["X: cumulative profile and local fits", "Y: cumulative profile and local fits",
                            "Residual products and window averages", "Average of window averages at each scale",
                            "Normalized DCCA correlation at each scale"])
        first = int(scales[0])
        for trace, row in zip(plot_data[first], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]):
            fig.add_trace(trace, row=row, col=1)
        fig.add_trace(go.Scatter(x=scales, y=covariance, mode="lines+markers",
            name="Average of window averages", line=dict(color="teal")), row=4, col=1)
        fig.add_trace(go.Scatter(x=[first], y=[covariance[0]], mode="markers",
            marker=dict(color="red", size=12), name="Selected scale"), row=4, col=1)
        fig.add_trace(go.Scatter(x=scales, y=correlations, mode="lines+markers",
            name="DCCA correlation", line=dict(color="royalblue"),
            hovertemplate="Scale: %{x}<br>DCCA correlation: %{y:.4f}<extra></extra>"), row=5, col=1)
        fig.add_trace(go.Scatter(x=[first], y=[correlations[0]], mode="markers",
            marker=dict(color="red", size=12), name="Selected correlation scale", showlegend=False), row=5, col=1)

        fit_styles = [
            ("All scales", "black", "dash"),
            ("Scales ≤ 25th percentile", "seagreen", "dot"),
            ("Scales ≥ 75th percentile", "darkorange", "dashdot"),
        ]
        for index, (label, color, dash) in enumerate(fit_styles):
            fit_scales = scales[fit_masks[index]]
            if fit_scales.size < 2:
                continue
            slope = fit_slopes[index]
            endpoints = fit_scales[[0, -1]]
            fitted_endpoints = slope * endpoints + fit_intercepts[index]
            fig.add_trace(go.Scatter(
                x=endpoints, y=fitted_endpoints, mode="lines",
                name=f"{label} fit ({slope:+.3g} per point)",
                line=dict(color=color, dash=dash, width=3),
                hovertemplate="Scale: %{x}<br>Fitted correlation: %{y:.4f}",
            ), row=5, col=1)

        fig.frames = [go.Frame(name=str(int(s)), traces=list(range(10)) + [11, 13],
            data=plot_data[int(s)] + [go.Scatter(x=[int(s)], y=[covariance[index]]),
                                     go.Scatter(x=[int(s)], y=[correlations[index]])],
            layout=go.Layout(**layouts[int(s)])) for index, s in enumerate(scales)]
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

    #################################
    ############ OUTPUTS ############
    #################################

    return fit_slopes, correlations, scales


def gait_match_bounds(imu_peaks, emg_peaks):
    """Return paired peak boundaries and the longest run of valid cycles."""
    cycle_edges = np.searchsorted(emg_peaks, imu_peaks, side="left")
    valid_cycles = np.diff(cycle_edges) == 1

    # A cycle is the half-open time range from one IMU peak to the next.
    # Runs on either side of an empty or ambiguous cycle stay separate.
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid_cycles, False]))
    runs = zip(boundaries[::2], boundaries[1::2])
    start, end = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    return cycle_edges, start, end


def match_gait_intervals(imu_peaks, emg_peaks):
    """Return intervals from the longest run with one EMG peak per IMU cycle."""
    cycle_edges, start, end = gait_match_bounds(imu_peaks, emg_peaks)
    return np.diff(imu_peaks[start:end]), np.diff(emg_peaks[cycle_edges[start:end]])


def plot_by_inclination(results, metric, panels, units, detail_column):
    """Plot participant values for each inclination in side-by-side box plots."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=panels)
    for col, panel in enumerate(panels, start=1):
        for position, (inclination, color) in enumerate(zip(inclinations, ["steelblue", "darkorange", "seagreen"])):
            group = results.loc[(results["Panel"] == panel) &
                                (results["Inclination"] == inclination) &
                                np.isfinite(results[metric])]
            fig.add_trace(go.Box(
                x=np.full(len(group), position), y=group[metric], width=0.5, name=f"{inclination}°",
                customdata=group[["Participant", detail_column]].to_numpy(),
                boxpoints="all", jitter=0.3, pointpos=0, marker_color=color,
                legendgroup=inclination, showlegend=(col == 1),
                hovertemplate=f"%{{customdata[0]}}<br>{metric}: %{{y:.4g}}"
                              f"<br>{detail_column}: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>",
            ), row=1, col=col)
        fig.update_xaxes(
            title_text="Inclination", tickvals=list(range(len(inclinations))),
            ticktext=[f"{angle}°" for angle in inclinations], range=[-0.5, 2.5], row=1, col=col,
        )
        fig.update_yaxes(title_text=units, row=1, col=col)
    fig.update_layout(template="plotly_white", height=510)
    return fig


def build_report(panels):
    """Build the tabs and embed Plotly once so the report works offline."""
    tabs, sections = [], []
    for index, (label, figure, description) in enumerate(panels):
        panel_id = f"panel-{index}"
        tabs.append(
            f'<button type="button" role="tab" id="tab-{index}" '
            f'aria-controls="{panel_id}" aria-selected="{str(index == 0).lower()}">{label}</button>'
        )
        plot_html = figure.to_html(full_html=False, include_plotlyjs=(index == 0))
        hidden = " hidden" if index else ""
        sections.append(
            f'<section id="{panel_id}" role="tabpanel" aria-labelledby="tab-{index}"{hidden}>'
            f"<p>{description}</p>{plot_html}</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inclined gait interval analysis</title>
<style>
body {{font: 16px system-ui, sans-serif; margin: 2rem auto; max-width: 1320px; padding: 0 1rem; color: #182633;}}
[role=tablist] {{display: flex; flex-wrap: wrap; gap: .5rem; border-bottom: 1px solid #ccd5dd; padding-bottom: .75rem;}}
button {{font: inherit; cursor: pointer; border: 1px solid #aebdc9; border-radius: .4rem; padding: .5rem .9rem; background: white;}}
button[aria-selected=true] {{background: #16476a; color: white; border-color: #16476a;}}
section p {{max-width: 85ch; line-height: 1.5;}}
</style>
</head>
<body>
<h1>Inclined gait interval analysis</h1>
<nav aria-label="Analysis metric" role="tablist">{''.join(tabs)}</nav>
{''.join(sections)}
<script>
const tabs = [...document.querySelectorAll('[role="tab"]')];
for (const tab of tabs) {{
    tab.addEventListener('click', () => {{
        for (const item of tabs) {{
            const active = item === tab;
            item.setAttribute('aria-selected', active);
            const panel = document.getElementById(item.getAttribute('aria-controls'));
            panel.hidden = !active;
            if (active) {{
                requestAnimationFrame(() => Plotly.Plots.resize(panel.querySelector('.plotly-graph-div')));
            }}
        }}
    }});
}}
</script>
</body>
</html>
"""


def open_report(page):
    """Send the self-contained report to the local browser without writing a file."""
    content = page.encode("utf-8")

    class ReportHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            self.server.report_delivered = True

    with HTTPServer(("127.0.0.1", 0), ReportHandler) as server:
        server.report_delivered = False
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Opening report at {url}")
        webbrowser.open(url)
        deadline = monotonic() + 30
        while not server.report_delivered and monotonic() < deadline:
            server.timeout = max(0, deadline - monotonic())
            server.handle_request()
        if not server.report_delivered:
            raise RuntimeError("The browser did not load the report within 30 seconds.")


def results_table(variability, coupling):
    """Build one row per participant, with separate columns for every incline."""
    participant_ids = sorted(variability["Participant"].unique(), key=lambda pid: int(pid[1:]))
    results = pd.DataFrame(index=pd.Index(participant_ids, name="ID"))
    variability = variability.set_index(["Inclination", "Panel", "Participant"]).sort_index()
    coupling = coupling.set_index(["Inclination", "Pair", "Range", "Participant"]).sort_index()

    for incline in inclinations:
        for label, signal, _ in devices:
            for metric in ("CV", "DFA"):
                name = f"{metric}_{signal_names[signal]}_{incline}deg"
                results[name] = variability[metric].get((incline, label), pd.Series(dtype=float))
        for first, second, label in pairs:
            for scale_range in ("Short", "Long"):
                name = f"DCCA_{scale_range}_{signal_names[first]}_{signal_names[second]}_{incline}deg"
                results[name] = coupling["Mean DCCA"].get(
                    (incline, label, scale_range), pd.Series(dtype=float),
                )

    return results


def annotate_statistics(figure, statistics, outcomes_by_panel):
    """Draw significant mixed-model contrasts above the boxes they compare."""
    angle_positions = {f"{angle}°": index for index, angle in enumerate(inclinations)}
    grouped = len(outcomes_by_panel[0]) > 1
    for column, outcomes in enumerate(outcomes_by_panel, start=1):
        brackets = []
        for band, outcome in enumerate(outcomes):
            contrasts = statistics.loc[statistics["Outcome"].eq(outcome) & statistics["Holm p"].lt(.05)]
            band_brackets = []
            for comparison, p in contrasts[["Comparison", "Holm p"]].itertuples(index=False, name=None):
                higher, lower = comparison.split(" vs ")
                left, right = angle_positions[lower], angle_positions[higher]
                band_brackets.append((left, right, p))
            band_brackets.sort(key=lambda bracket: bracket[1] - bracket[0])
            brackets.extend((band, tier, left, right, p)
                            for tier, (left, right, p) in enumerate(band_brackets))
        if not brackets:
            continue

        values = np.asarray([value for trace in figure.select_traces(row=1, col=column)
                             for value in trace.y], dtype=float)
        values = values[np.isfinite(values)]
        low, high = (-1.05, 1.05) if grouped else (values.min(), values.max())
        span = max(high - low, 0.01)
        tip = 0.025 * span
        top = high
        for band, tier, left, right, p in brackets:
            x0, x1 = (band + dcca_offsets[left], band + dcca_offsets[right]) if grouped else (left, right)
            y = high + (0.08 + 0.12 * tier) * span
            top = max(top, y)
            figure.add_shape(
                type="path", path=f"M {x0},{y-tip} L {x0},{y} L {x1},{y} L {x1},{y-tip}",
                line=dict(color="#243447", width=1.5), row=1, col=column,
            )
            stars = "***" if p < .001 else "**" if p < .01 else "*"
            figure.add_annotation(
                x=(x0 + x1) / 2, y=y, yanchor="bottom", yshift=2,
                text=stars, showarrow=False, font=dict(size=11, color="#243447"),
                row=1, col=column,
            )
        figure.update_yaxes(range=[low - 0.05 * span, top + 0.12 * span], row=1, col=column)
    if figure.layout.shapes:
        figure.update_layout(height=600)


def load_peak_times():
    """Index available peak series by participant, inclination, and signal."""
    records = {}
    filename_pattern = re.compile(r"(P\d+)_Peaks\.xlsx", re.IGNORECASE)
    for _, signal, folder in devices:
        for file in (peak_directory / folder).glob("*.xlsx"):
            match = filename_pattern.fullmatch(file.name)
            if match is None:
                continue
            participant = match[1].upper()
            workbook = pd.read_excel(file)
            for inclination in inclinations:
                column = f"{signal}_Peak_Time_{inclination}_deg"
                if column in workbook:
                    times = pd.to_numeric(workbook[column], errors="coerce").dropna().to_numpy()
                    if times.size >= 2:
                        records[participant, inclination, signal] = times
    return records


def standardize_peak_times(records):
    """Keep the latest intervals using separate ECG and gait group minimums."""
    gait_count = min((len(times) - 1 for key, times in records.items() if key[2] != "ECG"), default=0)
    ecg_count = min((len(times) - 1 for key, times in records.items() if key[2] == "ECG"), default=0)
    trimmed = {
        key: times[-(ecg_count + 1 if key[2] == "ECG" else gait_count + 1):]
        for key, times in records.items()
    }
    return trimmed, gait_count, ecg_count


def interval_summary(records):
    """Summarize the retained intervals pooled within each signal and inclination."""
    headers = ["Signal", "Inclination", "Participants", "Intervals per participant",
               "Mean (s)", "SD (s)", "IQR (s)"]
    rows = []
    for label, signal, _ in devices:
        for inclination in inclinations:
            series = [
                np.diff(times)
                for (_, grade, device), times in records.items()
                if grade == inclination and device == signal
            ]
            if series:
                pooled = np.concatenate(series)
                sd = f"{pooled.std(ddof=1):.4f}" if pooled.size > 1 else "—"
                q25, q75 = np.percentile(pooled, [25, 75])
                summary = (len(series[0]), f"{pooled.mean():.4f}", sd, f"{q75 - q25:.4f}")
            else:
                summary = ("—", "—", "—", "—")
            rows.append((label, f"{inclination}°", len(series), *summary))

    fig = go.Figure(go.Table(
        columnwidth=[160, 90, 90, 150, 85, 85, 85],
        header=dict(values=headers, fill_color="#16476a", font=dict(color="white", size=13),
                    align=["left"] + ["right"] * 6, height=34),
        cells=dict(values=list(zip(*rows)), fill_color=[["#f1f6fa", "#ffffff"] * 6],
                   align=["left"] + ["right"] * 6, height=30),
    ))
    fig.update_layout(template="plotly_white", height=450, margin=dict(t=12, b=15, l=0, r=0))
    return fig


def variability_results(records):
    labels = {signal: label for label, signal, _ in devices}
    rows = []
    for (participant, inclination, signal), times in records.items():
        intervals = np.diff(times)
        mean = intervals.mean()
        sd = intervals.std(ddof=1) if intervals.size > 1 else np.nan
        cv = 100 * sd / mean if mean != 0 else np.nan
        maximum = intervals.size // 10
        alpha = np.nan
        if maximum > 10 and sd > 0:
            scales = np.arange(10, maximum + 1)
            _, _, alpha = analysis.dfa(intervals, scales, order=1, plot=False)
        rows.append((participant, inclination, labels[signal], intervals.size, cv, alpha))
    return pd.DataFrame(
        rows, columns=["Participant", "Inclination", "Panel", "Intervals", "CV", "DFA"],
    ).astype({"CV": float, "DFA": float})


def plot_metric_correlations(results, metric, units, metric_pairs=None):
    """Compare participant-level metrics for every signal pair and inclination."""
    labels = [label for label, _, _ in devices]
    if metric_pairs is None:
        metric_pairs = list(combinations(labels, 2))
    wide = results.pivot(index=["Participant", "Inclination"], columns="Panel", values=metric)
    wide = wide.reindex(columns=labels).replace([np.inf, -np.inf], np.nan)
    fig = make_subplots(
        rows=len(metric_pairs), cols=len(inclinations),
        column_titles=[f"{inclination}°" for inclination in inclinations],
        vertical_spacing=0.055, horizontal_spacing=0.10,
    )
    for row, (left, right) in enumerate(metric_pairs, start=1):
        complete = wide[[left, right]].dropna()
        if complete.empty:
            bounds = [0, 1]
        else:
            low, high = float(complete.min().min()), float(complete.max().max())
            padding = 0.08 * (high - low) if high > low else 0.1
            bounds = [low - padding, high + padding]

        for col, (inclination, color) in enumerate(
            zip(inclinations, ["steelblue", "darkorange", "seagreen"]), start=1,
        ):
            group = complete.loc[complete.index.get_level_values("Inclination") == inclination]
            x, y = group[left].to_numpy(), group[right].to_numpy()
            fig.add_shape(
                type="line", x0=bounds[0], y0=bounds[0], x1=bounds[1], y1=bounds[1],
                line=dict(color="#aab6c2", dash="dash", width=1), layer="below",
                row=row, col=col,
            )
            r_label = "r unavailable"
            if len(group) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                r_label = f"r = {np.corrcoef(x, y)[0, 1]:.2f}"
                slope, intercept = np.polyfit(x, y, 1)
                ends = np.array([x.min(), x.max()])
                fig.add_trace(go.Scatter(
                    x=ends, y=slope * ends + intercept, mode="lines",
                    line=dict(color=color, width=2), hoverinfo="skip",
                ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", customdata=group.index.get_level_values("Participant"),
                marker=dict(color=color, size=8, line=dict(color="white", width=1)),
                hovertemplate=f"%{{customdata}}<br>{left}: %{{x:.4g}}<br>{right}: %{{y:.4g}}<extra></extra>",
            ), row=row, col=col)
            fig.add_annotation(
                text=f"{r_label} · n = {len(group)}", x=0.03, y=0.97, xanchor="left", yanchor="top",
                xref="x domain", yref="y domain", bgcolor="rgba(255,255,255,0.85)",
                showarrow=False, row=row, col=col,
            )
            fig.update_xaxes(range=bounds, title_text=f"{left} ({units})", row=row, col=col)
            fig.update_yaxes(range=bounds, title_text=f"{right} ({units})" if col == 1 else None,
                             row=row, col=col)

    fig.update_layout(template="plotly_white", showlegend=False,
                      height=380 if len(metric_pairs) == 1 else 1760,
                      margin=dict(t=55, b=55, l=95, r=25))
    return fig


def write_manuscript_jpeg(figure, path):
    """Render a 2640-pixel-wide JPEG with 300 DPI metadata."""
    from PIL import Image

    rendered = figure.to_image(format="png", width=1320, scale=2)
    with Image.open(BytesIO(rendered)) as image:
        image.convert("RGB").save(path, format="JPEG", dpi=(300, 300), quality=95, subsampling=0)


def save_jpeg_figures(panels, variability, directory):
    """Export the report panels and each Pearson correlation row for Word."""
    directory.mkdir(parents=True, exist_ok=True)
    for title, figure, _ in panels:
        filename = title.lower().replace(" ", "_") + ".jpeg"
        write_manuscript_jpeg(figure, directory / filename)

    labels = [label for label, _, _ in devices]
    for metric, units in (("CV", "CV %"), ("DFA", "DFA α")):
        for row, pair in enumerate(combinations(labels, 2), start=1):
            figure = plot_metric_correlations(variability, metric, units, metric_pairs=[pair])
            filename = f"{metric.lower()}_correlations_row_{row}.jpeg"
            write_manuscript_jpeg(figure, directory / filename)
    print(f"Saved JPEG figures in {directory}")


def coupling_results(records, gait_count):
    maximum = gait_count // 10
    matched = []
    unavailable = 0
    trials = sorted({(participant, inclination) for participant, inclination, _ in records})
    for participant, inclination in trials:
        for reference, other, label in pairs:
            first = records.get((participant, inclination, reference))
            second = records.get((participant, inclination, other))
            if first is None or second is None:
                unavailable += 1
                continue
            first_intervals, second_intervals = match_gait_intervals(first, second)
            if first_intervals.size < max(11, maximum):
                unavailable += 1
                continue
            matched.append((participant, inclination, label, first_intervals, second_intervals))

    rows = []
    bands = "No trials had enough consecutive matched intervals for two distinct scales."
    if matched and maximum > 10:
        scales = np.arange(10, maximum + 1)
        q25, q75 = np.percentile(scales, [25, 75])
        short = scales <= q25
        long = scales >= q75
        bands = f"Short scales: {scales[short][0]}–{scales[short][-1]}. Long scales: {scales[long][0]}–{scales[long][-1]}."
        for participant, inclination, label, first, second in matched:
            _, correlations, _ = dcca(first, second, scales, plot=False)
            for name, band in [("Short", short), ("Long", long)]:
                values = correlations[band & np.isfinite(correlations)]
                if values.size:
                    rows.append((participant, inclination, label, name, len(first), float(values.mean())))

    return pd.DataFrame(rows, columns=[
        "Participant", "Inclination", "Pair", "Range", "Matched intervals", "Mean DCCA",
    ]), bands, unavailable


def plot_coupling(results):
    labels = [label for _, _, label in pairs]
    fig = make_subplots(rows=1, cols=len(labels), subplot_titles=labels)
    for col, label in enumerate(labels, start=1):
        for inclination, offset, color in zip(inclinations, dcca_offsets, ["steelblue", "darkorange", "seagreen"]):
            group = results.loc[(results["Pair"] == label) & (results["Inclination"] == inclination)]
            fig.add_trace(go.Box(
                x=group["Range"].map(dcca_ranges.index) + offset,
                y=group["Mean DCCA"], name=f"{inclination}°", width=0.2,
                customdata=group[["Participant", "Matched intervals"]].to_numpy(),
                boxpoints="all", jitter=0.3, pointpos=0, marker_color=color,
                legendgroup=inclination, showlegend=(col == 1),
                hovertemplate="%{customdata[0]}<br>Mean DCCA: %{y:.4f}"
                              "<br>Matched intervals: %{customdata[1]}<extra>%{fullData.name}</extra>",
            ), row=1, col=col)
        fig.update_xaxes(tickvals=[0, 1], ticktext=dcca_ranges, range=[-0.5, 1.5], row=1, col=col)
        fig.update_yaxes(
            title_text="Mean DCCA correlation", range=[-1.05, 1.05],
            tickvals=[-1, -0.5, 0, 0.5, 1], row=1, col=col,
        )
    fig.update_layout(template="plotly_white", height=510)
    return fig


def main():
    records, gait_count, ecg_count = standardize_peak_times(load_peak_times())
    variability = variability_results(records)
    coupling, bands, unavailable = coupling_results(records, gait_count)
    labels = [label for label, _, _ in devices]
    lengths = (
        f"Each available ECG series uses its last {ecg_count} intervals. "
        f"Each available IMU or EMG series uses its last {gait_count} intervals."
    )
    panels = [
        ("Interval summary", interval_summary(records),
         "Counts and statistics use the actual retained intervals. Mean, sample SD, and IQR "
         "are calculated from the pooled intervals for each signal and inclination."),
        ("Magnitude of variability", plot_by_inclination(variability, "CV", labels, "CV (%)", "Intervals"),
         f"{lengths} CV is 100 × sample SD / mean."),
        ("DFA", plot_by_inclination(variability, "DFA", labels, "DFA α", "Intervals"),
         f"{lengths} DFA uses linear detrending and every integer window size "
         "from 10 to one tenth of the retained interval count."),
        ("DCCA", plot_coupling(coupling),
         f"{lengths} Within each participant and inclination, the IMU defines "
         "consecutive strides. For each muscle, the longest run with exactly one EMG peak per stride "
         "supplies the matched intervals. The plotted values are signed mean correlations in the "
         f"lowest and highest scale quartiles. {bands} "
         f"Pair/trial combinations with missing data or insufficient matched intervals: {unavailable}."),
        ("CV correlations", plot_metric_correlations(variability, "CV", "CV %"),
         "Each point is one participant with both CV measurements at that inclination. "
         "Panels show Pearson r, the paired sample size, a linear fit, and a dashed identity line."),
        ("DFA correlations", plot_metric_correlations(variability, "DFA", "DFA α"),
         "Each point is one participant with both DFA estimates at that inclination. "
         "Panels show Pearson r, the paired sample size, a linear fit, and a dashed identity line."),
    ]
    if save_results_excel or statistical_analysis:
        results = results_table(variability, coupling)

    if statistical_analysis:
        from User_statistical_analysis import analyze_outcomes

        statistics, _ = analyze_outcomes(results)
        annotations = {
            "Magnitude of variability": [(f"CV_{signal_names[signal]}",) for _, signal, _ in devices],
            "DFA": [(f"DFA_{signal_names[signal]}",) for _, signal, _ in devices],
            "DCCA": [
                tuple(f"DCCA_{band}_{signal_names[first]}_{signal_names[second]}" for band in ("Short", "Long"))
                for first, second, _ in pairs
            ],
        }
        note = (f" Mixed-model p-values use Holm correction across all {len(statistics)} comparisons. "
                "Brackets mark significant differences (* p < 0.05, ** p < 0.01, *** p < 0.001).")
        for index, (label, figure, description) in enumerate(panels):
            if label in annotations:
                annotate_statistics(figure, statistics, annotations[label])
                panels[index] = (label, figure, description + note)

    if save_results_excel:
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        counts = coupling[["Participant", "Inclination", "Pair", "Matched intervals"]].drop_duplicates()
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Results", freeze_panes=(1, 1))
            counts.to_excel(writer, sheet_name="DCCA interval counts", index=False, freeze_panes=(1, 0))
        print(f"Saved {excel_path}")

    page = build_report(panels)
    if save_HTML:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(page, encoding="utf-8")
        print(f"Saved {report_path}")
    else:
        open_report(page)

    if save_jpeg:
        save_jpeg_figures(panels, variability, jpeg_directory)


if __name__ == "__main__":
    main()
