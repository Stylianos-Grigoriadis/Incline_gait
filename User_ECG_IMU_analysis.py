"""ECG and gyro_x interval analysis with original project filters.

Positive-peak detection is scipy.find_peaks(distance, height), as in lib.py.
Settings are estimated per recording; no manual editing or interval removal.
Run beside lib.py and User_EMG_cutoff_analysis.py. No cutoff sweep is applied.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import find_peaks, correlate, resample_poly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lib
from User_EMG_cutoff_analysis import interval_metrics, validate_signal

project = Path(r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait")
base_directory = project / "Data" / "Valid Data"
save_directory = project / "Figures"
ID_list = None
trials = (("0", 0.0), ("12", 12.5), ("25", 25.0))
settings = {
    "ECG": dict(fs=2148.1481, columns=[2, 3], period_range=(0.30, 1.50), downsample=10),
    "IMU": dict(fs=370.3704, columns=[12, 13], period_range=(0.50, 2.00), downsample=2),
}


def estimate_settings(values, fs, period_range, downsample):
    """Rhythm-based distance and amplitude-based height, per recording.

    Strongest positive autocorrelation peak within configured cycle bounds;
    distance = 60% of this period. Height = 25% of the median amplitude of
    provisional positive peaks spaced by that distance. Final detection still
    uses only find_peaks(distance=..., height=...), matching the project.
    No optimization against the outcome metrics.
    """
    small = resample_poly(values, 1, downsample)
    centered = small - np.mean(small)
    if np.std(centered) <= np.finfo(float).eps:
        raise ValueError("Signal is constant; rhythm cannot be estimated")
    ac = correlate(centered, centered, mode="full", method="fft")[len(centered)-1:]
    ac /= ac[0]
    fs_small = fs / downsample
    lags, _ = find_peaks(ac)
    lags = lags[(lags >= np.ceil(period_range[0]*fs_small)) &
                (lags <= np.floor(period_range[1]*fs_small)) & (ac[lags] > 0)]
    if not len(lags):
        raise ValueError("No positive autocorrelation peak within rhythm bounds")
    lag = lags[np.argmax(ac[lags])]
    period = lag / fs_small
    distance = max(1, int(0.60 * period * fs))
    provisional, _ = find_peaks(values, distance=distance, height=0)
    if len(provisional) < 4:
        raise ValueError("Too few positive peaks for amplitude estimation")
    height = float(0.25 * np.median(values[provisional]))
    return distance, height, period, float(ac[lag])


def save_peak_check(time, values, indices, height, output, title):
    # Display decimation only; all samples used in filtering/detection/metrics.
    every = max(1, len(time)//100000)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Filtered signal and automatic peaks", "Inter-peak intervals"))
    fig.add_trace(go.Scattergl(x=time[::every], y=values[::every], name="Filtered signal"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=time[indices], y=values[indices], mode="markers", name="Peaks", marker=dict(size=5)), row=1, col=1)
    fig.add_hline(y=height, line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scattergl(x=time[indices][1:], y=np.diff(time[indices]), name="Intervals"), row=2, col=1)
    fig.update_xaxes(range=[float(time[0]+30), float(min(time[-1],time[0]+50))], row=1,col=1)
    fig.update_yaxes(title_text="Amplitude",row=1,col=1)
    fig.update_yaxes(title_text="Interval (s)",row=2,col=1)
    fig.update_xaxes(title_text="Time (s)",row=2,col=1)
    fig.update_layout(template="plotly_white",height=750,title=title)
    fig.write_html(str(output),include_plotlyjs=True)


def metric_report(df, participant, signal, output):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("DFA scaling exponent", "Coefficient of variation (SD/mean)", "Sample entropy (m=2, r=0.2 × SD)"), vertical_spacing=0.10)
    # Reindex preserves missing incline gaps rather than connecting over them.
    df = df.set_index("incline_deg").reindex([0.,12.5,25.]).reset_index()
    for row, (metric,label) in enumerate((("DFA_alpha","DFA α"),("CV","CV (ratio)"),("Ent_Samp","Sample entropy")),1):
        y=df[metric].to_numpy(dtype=float, copy=True)
        y[~np.isfinite(y)]=np.nan
        fig.add_trace(go.Scatter(x=df.incline_deg,y=y,mode="lines+markers",name=label,connectgaps=False,
                                customdata=df[["N_intervals","status"]].to_numpy(),
                                hovertemplate="Incline: %{x}<br>Value: %{y:.5f}<br>N intervals: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"),row=row,col=1)
        fig.update_yaxes(title_text=label,row=row,col=1)
    fig.update_xaxes(tickvals=[0,12.5,25],title_text="Incline (trial label)",row=3,col=1)
    fig.update_layout(template="plotly_white",height=1000,showlegend=False,
                      title=f"{participant} — {signal if signal=='ECG' else 'IMU gyro_x'} interval metrics across inclines")
    fig.add_annotation(text="Original filters; automatic peak settings per recording. No interval exclusions. Peak-check reports require review.",
                       x=0,y=-0.12,xref="paper",yref="paper",showarrow=False)
    fig.write_html(str(output),include_plotlyjs=True)


def main():
    ids = ID_list if ID_list is not None else sorted(
        (p.name for p in base_directory.iterdir() if p.is_dir() and p.name.startswith("P") and p.name[1:].isdigit()),key=lambda x:int(x[1:]))
    for signal, config in settings.items():
        root=save_directory / f"{signal} analysis results"
        reports=root / "HTML metric reports"
        participants=root / "Participants"
        reports.mkdir(parents=True,exist_ok=True)
        participants.mkdir(parents=True,exist_ok=True)
        (root / "analysis_settings.json").write_text(json.dumps(dict(
            signal=signal, sampling_frequency=config["fs"], filters="0.5–250 Hz fourth-order bandpass; 50 Hz notch, bandwidth 2.5 Hz" if signal=="ECG" else "20 Hz second-order Butterworth low-pass, zero-phase",
            rhythm_period_range_s=config["period_range"], distance_fraction=0.60,
            height_rule="0.25 * median provisional positive peak amplitude",
            peak_algorithm="scipy.signal.find_peaks(distance, height); positive peaks",
            DFA="lib.dfa, np.arange(16, N//9, 1), order=1", CV="np.std(ddof=0)/mean",
            sample_entropy="lib.Ent_Samp(intervals, 2, 0.2)", interval_exclusions="none",
            note="Automatic heuristic settings, not guaranteed optimal; inspect peak checks. Incline labels follow input file mapping, units not inferred."),indent=2),encoding="utf-8")
        all_rows, failures=[],[]
        for participant in ids:
            output=participants/participant
            output.mkdir(exist_ok=True)
            rows=[]
            for filename,incline in trials:
                path=base_directory/participant/f"{filename}.csv"
                try:
                    data=pl.read_csv(path,skip_rows=8,has_header=False,separator=",",columns=config["columns"],infer_schema=False)
                    data=data.with_columns(pl.all().cast(pl.Utf8).str.strip_chars()).with_columns(pl.all().cast(pl.Float64,strict=False))
                    time,raw=validate_signal(data[:,0].to_numpy(),data[:,1].to_numpy())
                    fs=config["fs"]
                    if signal=="ECG":
                        values=lib.butter_bandpass_filtfilt(raw,fs,low=0.5,high=250,order=4,plot=False)
                        values=lib.notch_filter_with_plots(values,fs,f_notch=50.0,bandwidth=2.5,plot=False)
                    else:
                        values=lib.Butterworth(fs,20,raw)
                    distance,height,period,strength=estimate_settings(values,fs,config["period_range"],config["downsample"])
                    indices,_=find_peaks(values,distance=distance,height=height)
                    peak_times=time[indices]
                    metrics=interval_metrics(np.diff(peak_times))
                    if strength < 0.10:
                        note="REVIEW: weak rhythm autocorrelation (<0.10); cycle estimate may be unreliable"
                        metrics["status"] = note if metrics["status"]=="ok" else metrics["status"]+"; "+note
                    row=dict(participant=participant,signal=signal,incline_deg=incline,N_peaks=len(indices),
                             peak_distance_samples=distance,peak_distance_s=distance/fs,peak_height=height,
                             estimated_period_s=period,rhythm_autocorrelation=strength,**metrics)
                    rows.append(row)
                    pd.DataFrame(dict(peak_time_s=peak_times,peak_amplitude=values[indices],
                                      interval_from_previous_s=np.r_[np.nan,np.diff(peak_times)])).to_csv(output/f"{filename}_{signal}_peaks.csv",index=False)
                    save_peak_check(time,values,indices,height,output/f"{filename}_{signal}_peak_check.html",
                                    f"{participant}, incline {incline:g}, {signal} — automatic peak check")
                    print(f"{signal} {participant} {incline:g}: N={metrics['N_intervals']}, period={period:.3f}s, {metrics['status']}",flush=True)
                except (OSError,ValueError,pl.exceptions.PolarsError) as exc:
                    failures.append(dict(participant=participant,incline_deg=incline,error=str(exc)))
                    print(f"SKIP {signal} {participant} {incline:g}: {exc}",flush=True)
            if rows:
                df=pd.DataFrame(rows)
                df.to_csv(output/f"{participant}_{signal}_metrics.csv",index=False)
                report=output/f"{participant}_{signal}_metrics_comparison.html"
                metric_report(df,participant,signal,report)
                shutil.copy2(report,reports/report.name)
                all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(root/f"All_participants_{signal}_metrics.csv",index=False)
        if all_rows:
            combined=pd.DataFrame(all_rows)
            combined.loc[combined.status != "ok"].to_csv(root/"recordings_for_review.csv",index=False)
        pd.DataFrame(failures,columns=["participant","incline_deg","error"]).to_csv(root/"skipped_recordings.csv",index=False)
        links="\n".join(f'<li><a href="{p.name}">{p.stem}</a></li>' for p in sorted(reports.glob("P*_metrics_comparison.html"),key=lambda p:int(p.name.split("_")[0][1:])))
        (reports/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{signal} metrics</title><h1>{signal} participant metric reports</h1><ul>{links}</ul>',encoding="utf-8")
        print(f"FINISHED {signal}: {len(all_rows)} recordings, {len(failures)} skipped. {root}",flush=True)


if __name__=="__main__":
    main()
