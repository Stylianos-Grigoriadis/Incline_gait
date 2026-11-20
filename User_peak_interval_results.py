"""Compare interval variability and gait-signal coupling across inclinations."""

from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

project = Path(
    r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait"
)
peak_directory = project / "Data" / "Peak Data per device"
devices = [
    ("ECG", "ECG", "Peak ECG"),
    ("Gastrocnemius EMG", "Gastrocnemius_EMG", "Peak Gast"),
    ("Quadriceps EMG", "Quadriceps_EMG", "Peak Quad"),
    ("IMU", "IMU", "Peak IMU"),
]
report_path = project / "Results" / "Peak_interval_results.html"
inclinations = ("0", "12.5", "25")
pairs = [
    ("IMU", "Gastrocnemius_EMG", "IMU–Gastrocnemius EMG"),
    ("IMU", "Quadriceps_EMG", "IMU–Quadriceps EMG"),
    ("Gastrocnemius_EMG", "Quadriceps_EMG", "Gastrocnemius–Quadriceps EMG"),
]


def dfa_alpha(data, scales):
    """Return the DFA exponent using the report's linear window detrending."""
    profile = np.cumsum(data - np.mean(data))
    fluctuations = []
    for scale in scales:
        chunks = len(data) // scale
        ms = 0.0
        positions = np.arange(scale)
        for index in range(chunks):
            window = profile[index * scale:(index + 1) * scale]
            coefficients = np.polyfit(positions, window, 1)
            fitted = np.polyval(coefficients, positions)
            ms += np.mean((window - fitted) ** 2)
        fluctuations.append(np.sqrt(ms / chunks))
    return np.polyfit(np.log(scales), np.log(fluctuations), 1)[0]


def dcca_correlations(x, y, scales):
    """Return the normalized DCCA coefficient for each sorted, unique scale."""
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
    return correlations


def match_gait_intervals(imu_peaks, emg_peaks):
    """Return intervals from the longest run with one EMG peak per IMU cycle."""
    cycle_edges = np.searchsorted(emg_peaks, imu_peaks, side="left")
    valid_cycles = np.diff(cycle_edges) == 1

    # A cycle is the half-open time range from one IMU peak to the next.
    # Runs on either side of an empty or ambiguous cycle stay separate.
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid_cycles, False]))
    runs = zip(boundaries[::2], boundaries[1::2])
    start, end = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    return np.diff(imu_peaks[start:end]), np.diff(emg_peaks[cycle_edges[start:end]])


def plot_by_inclination(results, metric, panels, units, detail_column):
    """Plot participant values for each inclination in side-by-side box plots."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=panels)
    for col, panel in enumerate(panels, start=1):
        for inclination, color in zip(inclinations, ["steelblue", "darkorange", "seagreen"]):
            group = results.loc[(results["Panel"] == panel) &
                                (results["Inclination"] == inclination) &
                                np.isfinite(results[metric])]
            fig.add_trace(go.Box(
                y=group[metric], name=f"{inclination}°",
                customdata=group[["Participant", detail_column]].to_numpy(),
                boxpoints="all", jitter=0.3, pointpos=0, marker_color=color,
                legendgroup=inclination, showlegend=(col == 1),
                hovertemplate=f"%{{customdata[0]}}<br>{metric}: %{{y:.4g}}"
                              f"<br>{detail_column}: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>",
            ), row=1, col=col)
        fig.update_xaxes(title_text="Inclination", row=1, col=col)
        fig.update_yaxes(title_text=units, row=1, col=col)
    fig.update_layout(template="plotly_white", height=510)
    return fig


def write_report(report_path, panels):
    """Write the tabs and embed Plotly once so the report works offline."""
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

    page = f"""<!doctype html>
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(page, encoding="utf-8")


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
            scales = np.unique(np.rint(np.geomspace(10, maximum, 20)).astype(int))
            alpha = dfa_alpha(intervals, scales)
        rows.append((participant, inclination, labels[signal], intervals.size, cv, alpha))
    return pd.DataFrame(
        rows, columns=["Participant", "Inclination", "Panel", "Intervals", "CV", "DFA"],
    ).astype({"CV": float, "DFA": float})


def plot_metric_correlations(results, metric, units):
    """Compare participant-level metrics for every signal pair and inclination."""
    labels = [label for label, _, _ in devices]
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

    fig.update_layout(template="plotly_white", showlegend=False, height=1760,
                      margin=dict(t=55, b=55, l=95, r=25))
    return fig


def coupling_results(records):
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
            # The same pairing rule also works with gastrocnemius as the reference.
            first_intervals, second_intervals = match_gait_intervals(first, second)
            if first_intervals.size < 11:
                unavailable += 1
                continue
            matched.append((participant, inclination, label, first_intervals, second_intervals))

    maximum = min(100, min((len(first) for _, _, _, first, _ in matched), default=0))
    rows = []
    bands = "No trials had enough consecutive matched intervals for two distinct scales."
    if matched:
        scales = np.arange(10, maximum + 1)
        q25, q75 = np.percentile(scales, [25, 75])
        short = scales <= q25
        long = scales >= q75
        bands = f"Short scales: {scales[short][0]}–{scales[short][-1]}. Long scales: {scales[long][0]}–{scales[long][-1]}."
        for participant, inclination, label, first, second in matched:
            correlations = dcca_correlations(first, second, scales)
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
        for inclination, color in zip(inclinations, ["steelblue", "darkorange", "seagreen"]):
            group = results.loc[(results["Pair"] == label) & (results["Inclination"] == inclination)]
            fig.add_trace(go.Box(
                x=group["Range"], y=group["Mean DCCA"], name=f"{inclination}°",
                customdata=group[["Participant", "Matched intervals"]].to_numpy(),
                boxpoints="all", jitter=0.3, pointpos=0, marker_color=color,
                offsetgroup=inclination, legendgroup=inclination, showlegend=(col == 1),
                hovertemplate="%{customdata[0]}<br>Mean DCCA: %{y:.4f}"
                              "<br>Matched intervals: %{customdata[1]}<extra>%{fullData.name}</extra>",
            ), row=1, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=["Short", "Long"], row=1, col=col)
        fig.update_yaxes(title_text="Mean DCCA correlation", range=[-1.05, 1.05], row=1, col=col)
    fig.update_layout(template="plotly_white", boxmode="group", height=510)
    return fig


def main():
    records, gait_count, ecg_count = standardize_peak_times(load_peak_times())
    variability = variability_results(records)
    coupling, bands, unavailable = coupling_results(records)
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
         f"{lengths} DFA uses linear detrending and up to 20 logarithmically spaced scales "
         "from 10 to one tenth of the retained interval count."),
        ("DCCA", plot_coupling(coupling),
         f"{lengths} Within each participant and inclination, the first signal in each pair defines "
         "consecutive cycles. The longest run with exactly one peak of the second signal per cycle "
         "supplies the matched intervals. The plotted values are signed mean correlations in the "
         f"lowest and highest scale quartiles. {bands} "
         f"Pair/trial combinations with missing data or fewer than 11 matched intervals: {unavailable}."),
        ("CV correlations", plot_metric_correlations(variability, "CV", "CV %"),
         "Each point is one participant with both CV measurements at that inclination. "
         "Panels show Pearson r, the paired sample size, a linear fit, and a dashed identity line."),
        ("DFA correlations", plot_metric_correlations(variability, "DFA", "DFA α"),
         "Each point is one participant with both DFA estimates at that inclination. "
         "Panels show Pearson r, the paired sample size, a linear fit, and a dashed identity line."),
    ]
    write_report(report_path, panels)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
