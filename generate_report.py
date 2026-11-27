"""
FaceTrace — Day 6: Appearance report generation
-------------------------------------------------------------------
Takes the raw detection log (detections_log.csv, produced by
analyze_video.py — one row per sampled frame a person was seen in)
and turns it into an actual usable report:

  - Groups nearby detections into "appearance windows" (e.g. someone
    seen at 0:05, 0:06, 0:07 counts as ONE appearance, not three)
  - For each person: first seen, last seen, number of separate
    appearances, total time on camera, average confidence
  - Exports both a raw and summarized report to CSV and Excel

Run:
    python generate_report.py

Requires detections_log.csv to already exist — run analyze_video.py
first.
"""

import os
import pandas as pd

LOG_FILE = "detections_log.csv"
SUMMARY_CSV = "appearance_report.csv"
SUMMARY_XLSX = "appearance_report.xlsx"

# If the same person is detected again within this many seconds of their
# last detection, it's treated as the SAME appearance (not a new one).
# This accounts for someone briefly turning their head, walking behind
# something, etc. without counting it as leaving and re-entering.
SESSION_GAP_SEC = 4.0


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def group_into_appearances(person_detections: pd.DataFrame) -> list:
    """
    Given all detection rows for ONE person, sorted by time, group
    consecutive detections into appearance windows based on SESSION_GAP_SEC.
    Returns a list of dicts: start_sec, end_sec, num_detections, avg_confidence.
    """
    rows = person_detections.sort_values("timestamp_sec").to_dict("records")
    appearances = []
    current = None

    for row in rows:
        if current is None:
            current = {
                "start_sec": row["timestamp_sec"],
                "end_sec": row["timestamp_sec"],
                "confidences": [row["confidence"]],
            }
        elif row["timestamp_sec"] - current["end_sec"] <= SESSION_GAP_SEC:
            # still within the same appearance window
            current["end_sec"] = row["timestamp_sec"]
            current["confidences"].append(row["confidence"])
        else:
            # gap too large — close out the current appearance, start a new one
            appearances.append(current)
            current = {
                "start_sec": row["timestamp_sec"],
                "end_sec": row["timestamp_sec"],
                "confidences": [row["confidence"]],
            }

    if current is not None:
        appearances.append(current)

    return appearances


def main():
    if not os.path.exists(LOG_FILE):
        raise RuntimeError(
            f"'{LOG_FILE}' not found. Run analyze_video.py first to generate it."
        )

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        print("No detections found in the log — nothing to report.")
        print("(This can happen if no enrolled faces appeared in the analyzed footage.)")
        return

    summary_rows = []

    for name, group in df.groupby("name"):
        appearances = group_into_appearances(group)

        all_confidences = [c for a in appearances for c in a["confidences"]]
        total_duration_sec = sum(a["end_sec"] - a["start_sec"] for a in appearances)

        summary_rows.append({
            "Name": name,
            "First Seen": format_timestamp(appearances[0]["start_sec"]),
            "Last Seen": format_timestamp(appearances[-1]["end_sec"]),
            "Number of Appearances": len(appearances),
            "Total Detections": len(group),
            "Est. Time on Camera (sec)": round(total_duration_sec, 1),
            "Avg Confidence": round(sum(all_confidences) / len(all_confidences), 3),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("First Seen")

    print("Appearance Summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    summary_df.to_excel(SUMMARY_XLSX, index=False)

    print(f"\nSaved -> {SUMMARY_CSV}")
    print(f"Saved -> {SUMMARY_XLSX}")


if __name__ == "__main__":
    main()
