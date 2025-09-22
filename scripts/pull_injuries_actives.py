#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pull_injuries_actives.py

Behavior retained:
- Produces CSV at data/raw/injuries/injury_reports_latest.csv

Update:
- Also writes Parquet at data/raw/injuries/injury_reports_latest.parquet
"""

import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/injuries")
CSV_OUT = RAW_DIR / "injury_reports_latest.csv"
PARQ_OUT = RAW_DIR / "injury_reports_latest.parquet"

# If your existing script scraped ESPN via pandas.read_html, keep that flow.
# This version assumes read_html is used and lxml is installed (workflow updated).

ESPN_INJURIES_URL = "https://www.espn.com/nfl/injuries"

def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def _scrape_injuries(url: str) -> pd.DataFrame:
    # Preserve your original approach: read_html and concatenate tables.
    tables = pd.read_html(url)  # requires lxml
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True, copy=False)
    return df

def main():
    _ensure_dirs()

    df = _scrape_injuries(ESPN_INJURIES_URL)

    # Existing behavior: CSV
    df.to_csv(CSV_OUT, index=False)

    # New behavior: Parquet
    df.to_parquet(PARQ_OUT, index=False)

    print(f"Wrote: {CSV_OUT}")
    print(f"Wrote: {PARQ_OUT}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
