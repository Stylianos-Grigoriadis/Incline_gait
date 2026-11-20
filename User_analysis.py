from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from library import analysis
from plotly.subplots import make_subplots


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


directory = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Peaks Data")
retained_peak_count = 467
trim_ecg_to_same_peak_count = False
dcca_scales = np.arange(10, 101)
entropy_m = 2
entropy_r = 0.2
ami_max_lag = 100

signals = ["ECG", "Gastrocnemius_EMG", "IMU"]
inclinations = ["0", "12.5", "25"]
signal_labels = ["ECG", "Gastrocnemius EMG", "IMU"]
dcca_band_labels = ["All scales", "Scales ≤ 25th percentile", "Scales ≥ 75th percentile"]


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
    """Write four tabs and embed Plotly once so the report works offline."""
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


def main():
    interval_results = []
    dcca_results = []
    files = sorted(directory.glob("P*_Peaks.xlsx"), key=lambda file: int(file.stem.split("_")[0][1:]))
    for file in files:
        participant = file.stem.split("_")[0]
        data = pd.read_excel(file)
        for inclination in inclinations:
            peaks = {}
            for signal, label in zip(signals, signal_labels):
                column = f"{signal}_Peak_Time_{inclination}_deg"
                times = data[column].dropna().to_numpy()
                if signal != "ECG" or trim_ecg_to_same_peak_count:
                    times = times[-retained_peak_count:]
                peaks[signal] = times
                values = np.diff(times)

                cv = 100 * np.std(values, ddof=1) / np.mean(values)
                maximum = len(values) // 10
                variable_intervals = np.std(values) > 0
                alpha = np.nan
                if maximum > 10 and variable_intervals:
                    scales = np.unique(np.rint(np.geomspace(10, maximum, 20)).astype(int))
                    _, _, alpha = analysis.dfa(values, scales, order=1, plot=False)

                entropy, tau = np.nan, np.nan
                if len(values) > entropy_m + 1 and variable_intervals:
                    candidates, _ = analysis.average_mutual_information(
                        values, L_seconds=min(ami_max_lag + 1, len(values) - 1),
                        fs=1, n_bins=0, plot=False,
                    )
                    valid_delays = candidates[np.isfinite(candidates[:, 0]) & (candidates[:, 0] > 0), 0]
                    if valid_delays.size:
                        tau = int(valid_delays[0])
                        entropy = analysis.sample_entropy(values, m=entropy_m, r=entropy_r, tau=tau)

                interval_results.append((participant, inclination, label, len(values), cv, alpha, entropy, tau))

            imu, emg = match_gait_intervals(peaks["IMU"], peaks["Gastrocnemius_EMG"])
            # Use the same scale grid for every trial included in the comparison.
            mean_correlations = np.full(len(dcca_band_labels), np.nan)
            if imu.size >= dcca_scales[-1]:
                _, correlations, scales = dcca(imu, emg, dcca_scales, plot=False)
                lower_quartile, upper_quartile = np.percentile(scales, [25, 75])
                finite = np.isfinite(correlations)
                bands = [
                    finite,
                    finite & (scales <= lower_quartile),
                    finite & (scales >= upper_quartile),
                ]
                for index, band in enumerate(bands):
                    if band.any():
                        mean_correlations[index] = correlations[band].mean()
            for label, mean_correlation in zip(dcca_band_labels, mean_correlations):
                dcca_results.append((participant, inclination, label, imu.size, mean_correlation))

    interval_table = pd.DataFrame(interval_results, columns=[
        "Participant", "Inclination", "Panel", "Intervals", "CV", "DFA", "Sample entropy", "AMI delay",
    ])
    dcca_table = pd.DataFrame(dcca_results, columns=[
        "Participant", "Inclination", "Panel", "Matched intervals", "Mean DCCA correlation",
    ])
    panels = [
        ("CV", plot_by_inclination(interval_table, "CV", signal_labels, "CV (%)", "Intervals"),
         "Sample standard deviation divided by mean interval duration, expressed as a percentage."),
        ("DFA", plot_by_inclination(interval_table, "DFA", signal_labels, "DFA α", "Intervals"),
         "Linear detrending with up to 20 logarithmically spaced window sizes from 10 intervals to one tenth of the series length."),
        ("Sample entropy", plot_by_inclination(interval_table, "Sample entropy", signal_labels, "Sample entropy (nats)", "AMI delay"),
         f"Sample entropy with m = {entropy_m}, r = {entropy_r} × SD, and the first positive delay returned by AMI."),
        ("DCCA", plot_by_inclination(dcca_table, "Mean DCCA correlation", dcca_band_labels, "Mean DCCA correlation", "Matched intervals"),
         "IMU peaks define gait cycles. The analysis uses the longest consecutive run with exactly one EMG peak per cycle. "
         "Each participant contributes the mean signed DCCA correlation over scales 10–100, 10–32, and 78–100, respectively. "
         "Trials with fewer than 100 consecutive matched intervals are omitted."),
    ]
    report_path = directory.parents[1] / "Results" / "Inclined_gait_interval_analysis.html"
    write_report(report_path, panels)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
