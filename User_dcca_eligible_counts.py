"""Export the matched interval counts used to determine DCCA eligibility."""

import json
from pathlib import Path
import subprocess
import numpy as np

from User_peak_interval_results import (
    inclinations,
    load_peak_times,
    match_gait_intervals,
    pairs,
    standardize_peak_times,
)


def main():
    records, gait_count, _ = standardize_peak_times(load_peak_times())
    if not records:
        raise SystemExit("No peak-time records found.")

    maximum = gait_count // 10
    participants = sorted({key[0] for key in records}, key=lambda name: int(name[1:]))
    imu_pairs = [pair for pair in pairs if pair[0] == "IMU"]
    rows = []
    for participant in participants:
        for incline in inclinations:
            for reference, other, label in imu_pairs:
                first = records.get((participant, incline, reference))
                second = records.get((participant, incline, other))
                count = None
                diagnostics = [None] * 4
                if first is None or second is None:
                    status = "Missing peak data"
                    missing = [signal for signal, times in [(reference, first), (other, second)] if times is None]
                    reason = "Missing: " + ", ".join(missing)
                else:
                    matched, _ = match_gait_intervals(first, second)
                    count = len(matched)
                    peaks_per_cycle = np.diff(np.searchsorted(second, first, side="left"))
                    empty = int(np.count_nonzero(peaks_per_cycle == 0))
                    multiple = int(np.count_nonzero(peaks_per_cycle > 1))
                    valid = len(peaks_per_cycle) - empty - multiple
                    run_cycles = count + 1 if valid else 0
                    outside = valid - run_cycles
                    diagnostics = [empty, multiple, outside, run_cycles]
                    reason = (
                        f"{len(peaks_per_cycle)} reference cycles: {empty} empty, {multiple} with multiple peaks, "
                        f"{outside} matched outside the longest run. "
                        f"The longest run had {run_cycles} paired peaks, yielding {count} intervals."
                    )
                    if maximum <= 10:
                        status = "No shared scale range"
                    elif count < maximum:
                        status = "Too few matched intervals"
                    else:
                        status = "Eligible"
                rows.append([
                    participant, float(incline), label, count,
                    count if status == "Eligible" else 0, status, *diagnostics, reason,
                ])

    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    output = Path(__file__).with_name("DCCA_eligible_counts_IMU_EMG.xlsx")
    subprocess.run(
        [str(runtime / "bin/node.exe"), "--input-type=module", "-e", EXPORT_WORKBOOK,
         str(runtime), str(output)],
        input=json.dumps({"rows": rows, "gaitCount": gait_count, "maximum": maximum}),
        text=True,
        check=True,
    )
    output.with_suffix(".xlsx.inspect.ndjson").unlink(missing_ok=True)
    print(output)


EXPORT_WORKBOOK = """
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [runtime, output] = process.argv.slice(1);
const require = createRequire(join(runtime, "package.json"));
const moduleUrl = pathToFileURL(require.resolve("@oai/artifact-tool")).href;
const { Workbook, SpreadsheetFile } = await import(moduleUrl);
const { rows, gaitCount, maximum } = JSON.parse(readFileSync(0, "utf8"));
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("DCCA counts");
sheet.getRange("A1:K1").merge();
sheet.getRange("A1").values = [["DCCA eligibility by participant, incline, and signal pair"]];
sheet.getRange("A2:K2").merge();
sheet.getRange("A2").values = [[
    `Standardized gait series: ${gaitCount} intervals. Shared window sizes: ${maximum > 10 ? `10–${maximum}` : "unavailable"}.`
]];
sheet.getRange("A4:K4").values = [[
    "Participant", "Incline (°)", "Signal pair", "Matched intervals",
    "Eligible intervals", "Status", "No paired peak", "Multiple paired peaks",
    "Matched outside run", "Cycles in longest run", "Reason",
]];
sheet.getRange(`A5:K${rows.length + 4}`).values = rows;
for (const range of ["A1:K1", "A4:K4"]) {
    sheet.getRange(range).format.fill = "#1F3650";
    sheet.getRange(range).format.font.color = "#FFFFFF";
    sheet.getRange(range).format.font.bold = true;
}
for (const [column, width] of [["A", 100], ["B", 85], ["C", 250], ["D", 120], ["E", 120], ["F", 180],
    ["G", 115], ["H", 145], ["I", 145], ["J", 150], ["K", 460]]) {
    sheet.getRange(`${column}:${column}`).format.columnWidthPx = width;
}
sheet.getRange(`K5:K${rows.length + 4}`).format.wrapText = true;
sheet.getRange(`A5:K${rows.length + 4}`).format.rowHeightPx = 46;
sheet.freezePanes.freezeRows(4);
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(output);
"""


if __name__ == "__main__":
    main()
