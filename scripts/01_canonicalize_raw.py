#!/usr/bin/env python3
"""
01_canonicalize_raw.py

Checklist ID 28: Canonicalize raw data.

- Loads raw prediction CSVs from data/predictions/
- Standardizes column names and dtypes
- Writes canonicalized CSVs to data/processed/props/
"""

import os
import sys
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/predictions")
OUT_DIR = Path("data/processed/props")

# Standard schema we want to enforce
STANDARD_COLS = [
    "player_id", "player_name", "team", "opponent",
    "season", "week", "stat_type", "line", "projection"
]

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def canonicalize_df(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Bring df into alignment with STANDARD_COLS.
    """
    # Lowercase all columns
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Map common variants
    col_map = {
        "player": "player_name",
        "team_abbr": "team",
        "opp": "opponent",
        "stat": "stat_type",
        "prop": "line",
        "pred": "projection",
        "proj": "projection",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure all standard cols exist
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = None

    # Enforce types
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    if "line" in df.columns:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
    if "projection" in df.columns:
        df["projection"] = pd.to_numeric(df["projection"], errors="coerce")

    # Subset + order
    df = df[STANDARD_COLS]

    # Drop empty rows
    df = df.dropna(how="all")

    return df

def main():
    ensure_dirs()
    if not RAW_DIR.exists():
        print(f"[WARN] {RAW_DIR} does not exist")
        sys.exit(0)

    files = list(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"[WARN] No CSV files found in {RAW_DIR}")
        sys.exit(0)

    for f in files:
        try:
            raw = pd.read_csv(f)
            clean = canonicalize_df(raw, f.name)
            out_path = OUT_DIR / f.name
            clean.to_csv(out_path, index=False)
            print(f"[OK] Canonicalized {f} → {out_path} ({len(clean)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to process {f}: {e}")

if __name__ == "__main__":
    main()
