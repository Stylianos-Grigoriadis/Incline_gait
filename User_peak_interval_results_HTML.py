"""Compare interval variability and gait-signal coupling across inclinations."""

from itertools import combinations
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from User_analysis import analysis, dcca, match_gait_intervals, plot_by_inclination, write_report
from User_qualification_after_peak_finding import devices, peak_directory, project


report_path = project / "Results" / "Peak_interval_results.html"
inclinations = ("0", "12.5", "25")
pairs = [
    ("IMU", "Gastrocnemius_EMG", "IMU–Gastrocnemius EMG"),
    ("IMU", "Quadriceps_EMG", "IMU–Quadriceps EMG"),
    ("Gastrocnemius_EMG", "Quadriceps_EMG", "Gastrocnemius–Quadriceps EMG"),
]


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
            _, _, alpha = analysis.dfa(intervals, scales, order=1, plot=False)
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
