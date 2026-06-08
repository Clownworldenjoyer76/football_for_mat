#!/usr/bin/env python3
"""
06_probability_and_edge.py

Computes probability edges from calibrated props.

Logic:
- Load props_current_calibrated.csv
- Compute edge = prob_over_cal - 0.5
- Save:
    - output/edges_summary.csv (all markets combined)
    - output/edges_<market>.csv (one file per market)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

INPUT = Path("data/props/props_current_calibrated.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "edges_summary.csv"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    if "prob_over_cal" not in df.columns or "prob_under_cal" not in df.columns:
        raise ValueError("Input file missing calibrated probability columns.")

    # Compute edge relative to 50/50 fair line
    df["edge"] = df["prob_over_cal"] - 0.5
    df["pick"] = df["edge"].apply(lambda x: "over" if x > 0 else "under")

    # Add run timestamp
    df["run_ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save combined summary
    df.to_csv(SUMMARY_OUT, index=False)
    print(f"✓ Wrote {SUMMARY_OUT} with {len(df)} rows")

    # Save per-market files
    if "market" in df.columns:
        for market, g in df.groupby("market"):
            out_path = OUTDIR / f"edges_{market}.csv"
            g.to_csv(out_path, index=False)
            print(f"✓ Wrote {out_path} ({len(g)} rows)")

if __name__ == "__main__":
    main()
