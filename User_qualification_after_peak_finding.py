"""Create a participant-by-participant report of intervals between exported peaks."""

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
report_path = (
    project / "Figures" / "Qualification after peak finding" / "Peak_interval_qualification.html"
)

devices = [
    ("ECG", "ECG", "Peak ECG"),
    ("Gastrocnemius EMG", "Gastrocnemius_EMG", "Peak Gast"),
    ("Quadriceps EMG", "Quadriceps_EMG", "Peak Quad"),
    ("IMU", "IMU", "Peak IMU"),
]
inclinations = [("0", "#2166ac"), ("12.5", "#a45114"), ("25", "#287456")]


def participant_results(participant):
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[label for label, _, _ in devices],
        vertical_spacing=0.21,
    )
    rows = []
    legend_shown = set()
    for index, (label, signal, folder) in enumerate(devices):
        row, col = index // 2 + 1, index % 2 + 1
        file = peak_directory / folder / f"{participant}_Peaks.xlsx"
        peaks = pd.read_excel(file) if file.is_file() else None
        missing = []

        for inclination, color in inclinations:
            column = f"{signal}_Peak_Time_{inclination}_deg"
            count, average, sd, cv, iqr = None, None, None, None, None
            if peaks is None:
                status = "Workbook missing"
            elif column not in peaks:
                status = "Peak-time column missing"
            else:
                times = pd.to_numeric(peaks[column], errors="coerce").dropna().to_numpy()
                intervals = np.diff(times)
                count = len(intervals)
                if count:
                    average = intervals.mean()
                    iqr = np.percentile(intervals, 75) - np.percentile(intervals, 25)
                    if count > 1:
                        sd = intervals.std(ddof=1)
                        cv = 100 * sd / average if average != 0 else None
                    status = "Available"
                    fig.add_trace(go.Scatter(
                        x=times[1:], y=intervals, name=f"{inclination}°",
                        mode="lines+markers", line=dict(color=color, width=1.5),
                        marker=dict(size=3),
                        legendgroup=inclination, showlegend=inclination not in legend_shown,
                        hovertemplate="%{x:.3f} s<br>Interval: %{y:.4f} s<extra>%{fullData.name}</extra>",
                    ), row=row, col=col)
                    legend_shown.add(inclination)
                else:
                    status = "Fewer than two peak times"

            rows.append((label, f"{inclination}°", count, average, sd, cv, iqr, status))
            if status != "Available":
                missing.append(f"{inclination}°: {status}")

        fig.update_xaxes(title_text="Time of later peak (s)", row=row, col=col)
        fig.update_yaxes(title_text="Interval (s)", row=row, col=col)
        if missing:
            subplot = fig.get_subplot(row, col)
            left, right = subplot.xaxis.domain
            bottom, top = subplot.yaxis.domain
            all_missing = len(missing) == len(inclinations)
            fig.add_annotation(
                text="<br>".join(missing),
                x=(left + right) / 2 if all_missing else left + 0.02 * (right - left),
                y=(bottom + top) / 2 if all_missing else top - 0.02 * (top - bottom),
                xref="paper", yref="paper",
                xanchor="center" if all_missing else "left",
                yanchor="middle" if all_missing else "top",
                showarrow=False, align="left", bgcolor="rgba(255,255,255,0.9)",
            )

    fig.update_layout(template="plotly_white", height=650, margin=dict(t=72, b=40))
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.13))
    table = pd.DataFrame(rows, columns=[
        "Signal", "Inclination", "Intervals", "Mean (s)", "SD (s)", "CV (%)", "IQR (s)", "Status",
    ])
    table["Intervals"] = table["Intervals"].astype("Int64")
    return fig, table


def main():
    workbook_pattern = re.compile(r"(P\d+)_Peaks\.xlsx", re.IGNORECASE)
    participants = set()
    for _, _, folder in devices:
        for file in (peak_directory / folder).glob("*.xlsx"):
            match = workbook_pattern.fullmatch(file.name)
            if match:
                participants.add(match[1].upper())
    participants = sorted(participants, key=lambda participant: int(participant[1:]))

    tabs, sections = [], []
    for index, participant in enumerate(participants):
        figure, table = participant_results(participant)
        selected = "true" if index == 0 else "false"
        hidden = "" if index == 0 else " hidden"
        tabs.append(
            f'<button type="button" role="tab" id="tab-{participant}" '
            f'aria-selected="{selected}" aria-controls="{participant}">{participant}</button>'
        )
        plot_html = figure.to_html(full_html=False, include_plotlyjs=(index == 0))
        table_html = table.to_html(index=False, border=0, na_rep="—", float_format="{:.4f}".format)
        sections.append(
            f'<section id="{participant}" role="tabpanel" aria-labelledby="tab-{participant}"{hidden}>'
            f"<h2>{participant}</h2>{plot_html}<h3>Interval summary</h3>"
            f'<div class="table-scroll">{table_html}</div></section>'
        )

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Peak interval qualification</title>
<style>
body { font: 16px system-ui, sans-serif; max-width: 1250px; margin: 2rem auto; padding: 0 1rem; color: #172b3b; }
[role=tablist] { display: flex; flex-wrap: wrap; gap: .45rem; padding-bottom: 1rem; border-bottom: 1px solid #c4d0da; }
button { font: inherit; cursor: pointer; border: 1px solid #a9b7c3; border-radius: .35rem; background: white; padding: .4rem .65rem; }
button[aria-selected=true] { background: #17486d; color: white; border-color: #17486d; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: .5rem .65rem; text-align: left; border-bottom: 1px solid #d9e1e8; }
th { background: #eaf1f7; }
tbody tr:nth-child(even) { background: #f7f9fb; }
</style>
</head>
<body>
<h1>Peak interval qualification</h1>
<p>Intervals are differences between consecutive exported peak times. SD is the sample standard deviation,
CV is 100 × SD / mean, and IQR is the 75th percentile minus the 25th percentile.</p>
"""
    if participants:
        page += '<nav role="tablist" aria-label="Participant">' + "".join(tabs) + "</nav>"
        page += "".join(sections)
    else:
        page += "<p>No participant workbooks found.</p>"
    page += """
<script>
const tabs = [...document.querySelectorAll('[role="tab"]')];
for (const tab of tabs) {
    tab.addEventListener('click', () => {
        for (const item of tabs) {
            const active = item === tab;
            item.setAttribute('aria-selected', active);
            const panel = document.getElementById(item.getAttribute('aria-controls'));
            panel.hidden = !active;
            if (active) {
                requestAnimationFrame(() => Plotly.Plots.resize(panel.querySelector('.plotly-graph-div')));
            }
        }
    });
}
</script>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(page, encoding="utf-8")
    print(f"Saved {len(participants)} participant tabs to {report_path}")


if __name__ == "__main__":
    main()
