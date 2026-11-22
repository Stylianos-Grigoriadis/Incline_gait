"""Generate an interactive audit of the existing DCCA interval selection."""

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from User_peak_interval_results import (
    gait_match_bounds,
    inclinations,
    load_peak_times,
    pairs,
    standardize_peak_times,
)


def main():
    raw = load_peak_times()
    if not raw:
        raise SystemExit("No exported peak times found.")
    records, gait_count, _ = standardize_peak_times(raw)
    maximum = gait_count // 10
    participants = sorted({key[0] for key in raw}, key=lambda name: int(name[1:]))
    cases = {}
    for participant in participants:
        for incline in inclinations:
            for index, (reference, other, _) in enumerate(pairs):
                first = records.get((participant, incline, reference))
                second = records.get((participant, incline, other))
                if first is None or second is None:
                    missing = [label for label, series in ((reference, first), (other, second)) if series is None]
                    cases[f"{participant}|{incline}|{index}"] = {"missing": missing}
                    continue
                edges, start, end = gait_match_bounds(first, second)
                cases[f"{participant}|{incline}|{index}"] = {
                    "first": first.tolist(),
                    "second": second.tolist(),
                    "edges": edges.tolist(),
                    "start": int(start),
                    "end": int(end),
                    "trimmed": [
                        raw[participant, incline, reference][: -len(first)].tolist(),
                        raw[participant, incline, other][: -len(second)].tolist(),
                    ],
                }

    data = json.dumps({
        "participants": participants, "inclinations": inclinations,
        "pairs": pairs, "maximum": maximum, "cases": cases,
    })
    output = Path(__file__).with_name("DCCA_exclusion_plot.html")
    output.write_text(
        HTML.replace("/* PLOTLY */", get_plotlyjs()).replace("/* DATA */", data),
        encoding="utf-8",
    )
    print(output)


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCCA interval review</title>
<style>
body { font: 16px system-ui, sans-serif; color: #17283b; margin: 24px auto; max-width: 1280px; padding: 0 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 16px; }
label { display: grid; gap: 4px; font-weight: 600; }
select { font: inherit; padding: 6px; }
#chart { min-height: 700px; }
</style>
<script>/* PLOTLY */</script>
</head>
<body>
<h1>DCCA interval review</h1>
<div class="controls">
  <label>Participant<select id="participant"></select></label>
  <label>Incline<select id="incline"></select></label>
  <label>Signal pair<select id="pair"></select></label>
  <label>DCCA window size<select id="scale"></select></label>
</div>
<p id="summary" aria-live="polite"></p>
<div id="chart"></div>
<p>Each interval is plotted at the time of its later peak. Colored markers show the DCCA selection status. Zoom either panel to inspect the same time range in both.</p>
<script>
const data = /* DATA */;
const controls = Object.fromEntries(
  ["participant", "incline", "pair", "scale"].map(id => [id, document.getElementById(id)])
);
const chart = document.getElementById("chart");
const summary = document.getElementById("summary");
function option(select, value, label) { select.add(new Option(label, value)); }
data.participants.forEach(value => option(controls.participant, value, value));
data.inclinations.forEach(value => option(controls.incline, value, value + "°"));
data.pairs.forEach((pair, i) => option(controls.pair, i, pair[2]));
option(controls.scale, "", "All matched intervals");
for (let size = 10; size <= data.maximum; size++) option(controls.scale, size, size + " intervals");

const colors = {
  "Included": "#168459", "Outside longest run": "#738197",
  "No paired peak": "#c43246", "Multiple paired peaks": "#aa4c08",
  "Last paired peak in run": "#7f56a0", "Incomplete window at this scale": "#ae7800",
  "Pair below DCCA minimum": "#52627a",
};

function render() {
  const { participant, incline, pair, scale } = controls;
  const key = [participant.value, incline.value, pair.value].join("|");
  const item = data.cases[key];
  if (item.missing) {
    summary.textContent = "Missing peak data for " + item.missing.join(", ") + ".";
    Plotly.purge(chart);
    chart.hidden = true;
    return;
  }
  chart.hidden = false;
  const [reference, other] = data.pairs[pair.value];
  const names = { IMU: "IMU", Gastrocnemius_EMG: "Gastrocnemius EMG", Quadriceps_EMG: "Quadriceps EMG" };
  const { first, second, edges, start, end, trimmed } = item;
  const matched = Math.max(0, end - start - 1);
  const eligible = data.maximum > 10 && matched >= Math.max(11, data.maximum);
  const size = Number(scale.value);
  const used = eligible ? (size ? Math.floor(matched / size) * size : matched) : 0;
  summary.textContent = matched + " matched intervals in the longest run. " + used +
    (size ? " used in complete windows at this scale." : " eligible for DCCA.") +
    " Peaks trimmed before matching: " + trimmed[0].length + " reference, " + trimmed[1].length + " paired.";
  const groups = {};
  const statuses = [];
  for (let i = 0; i < first.length - 1; i++) {
    const count = edges[i + 1] - edges[i];
    let status;
    if (count === 0) status = "No paired peak";
    else if (count > 1) status = "Multiple paired peaks";
    else if (i < start || i >= end) status = "Outside longest run";
    else if (i === end - 1) status = "Last paired peak in run";
    else if (!eligible) status = "Pair below DCCA minimum";
    else if (i - start >= used) status = "Incomplete window at this scale";
    else status = "Included";
    statuses.push(status);
    const group = groups[status] ??= { x: [], width: [], details: [] };
    group.x.push((first[i] + first[i + 1]) / 2);
    group.width.push(first[i + 1] - first[i]);
    group.details.push([i + 1, first[i], first[i + 1], count]);
  }
  const traces = Object.entries(colors).filter(([status]) => groups[status]).map(([status, color]) => {
    const group = groups[status];
    return {
      type: "bar", name: status, x: group.x, width: group.width,
      y: group.x.map(() => 0.26), base: -0.13, marker: { color }, customdata: group.details,
      hovertemplate: "Cycle %{customdata[0]}<br>%{customdata[1]:.3f}–%{customdata[2]:.3f} s" +
        "<br>Paired peaks: %{customdata[3]}<extra>" + status + "</extra>",
    };
  });
  function peaks(times, lane, name, color, symbol) {
    if (!times.length) return;
    traces.push({
      type: "scatter", mode: "markers", x: times, y: times.map(() => lane),
      name, marker: { symbol, color, size: 9 },
      hovertemplate: "%{x:.3f} s<extra>" + name + "</extra>",
    });
  }
  peaks(first, 2, names[reference] + " peaks", "#245493", "line-ns");
  peaks(second, 1, names[other] + " peaks", "#704486", "line-ns");
  peaks(trimmed[0], 2, "Trimmed reference peaks", "#9aa7b6", "line-ns");
  peaks(trimmed[1], 1, "Trimmed paired peaks", "#9aa7b6", "line-ns");
  function intervals(times, name, color) {
    traces.push({
      type: "scatter", mode: "lines", x: times.slice(1),
      y: times.slice(1).map((time, i) => time - times[i]), xaxis: "x2", yaxis: "y2",
      name, line: { color, width: 1.4 },
      hovertemplate: "%{x:.3f} s<br>Interval: %{y:.3f} s<extra>" + name + "</extra>",
    });
  }
  intervals([...trimmed[0], ...first], names[reference] + " intervals", "#245493");
  intervals([...trimmed[1], ...second], names[other] + " intervals", "#704486");

  const intervalGroups = {};
  function intervalMarker(times, i, status, name, symbol) {
    const group = intervalGroups[status] ??= { x: [], y: [], names: [], symbols: [] };
    group.x.push(times[i + 1]);
    group.y.push(times[i + 1] - times[i]);
    group.names.push(name);
    group.symbols.push(symbol);
  }
  statuses.forEach((status, i) => {
    intervalMarker(first, i, status, names[reference], "circle");
    // A paired interval is identifiable only when adjacent cycles each have one peak.
    if (edges[i + 1] - edges[i] === 1 && edges[i + 2] - edges[i + 1] === 1) {
      intervalMarker(second, edges[i], status, names[other], "diamond");
    }
  });
  for (const [status, group] of Object.entries(intervalGroups)) {
    traces.push({
      type: "scatter", mode: "markers", x: group.x, y: group.y, xaxis: "x2", yaxis: "y2",
      name: status, showlegend: false, text: group.names,
      marker: { color: colors[status], symbol: group.symbols, size: 7 },
      hovertemplate: "%{text}<br>Time: %{x:.3f} s<br>Interval: %{y:.3f} s<extra>" + status + "</extra>",
    });
  }
  if (used) {
    const x = [], y = [];
    for (let i = 0; i <= used; i++) {
      x.push(first[start + i], second[edges[start + i]], null);
      y.push(2, 1, null);
    }
    traces.unshift({
      type: "scatter", mode: "lines", x, y, hoverinfo: "skip", showlegend: false,
      line: { color: "rgba(90,110,130,0.22)", width: 1 },
    });
  }
  Plotly.react(chart, traces, {
    barmode: "overlay", uirevision: key, dragmode: "zoom",
    xaxis: { matches: "x2", showticklabels: false },
    xaxis2: { title: { text: "Time (s)" }, anchor: "y2" },
    yaxis: { tickvals: [0, 1, 2], ticktext: ["Cycle status", names[other], names[reference]],
      range: [-0.4, 2.4], fixedrange: true, domain: [0.61, 1] },
    yaxis2: { title: { text: "Interval duration (s)" }, domain: [0, 0.53] },
    legend: { orientation: "h", y: -0.11 },
    margin: { l: 150, r: 20, t: 15, b: 125 },
  }, { responsive: true });
}
Object.values(controls).forEach(select => select.addEventListener("change", render));
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
