#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_build_weekly_latest.py

Builds player-week data for a target season into:
  data/weekly/latest.csv

Priority order for data sources (fastest + most reliable first):
1) LOCAL nflverse cache (downloaded by your refresh): data/raw/nflverse/weekly_<season>.csv(.gz)
2) REMOTE nflverse CSV (via env NFLVERSE_WEEKLY_URL_TPL), e.g.
   https://github.com/nflverse/nflverse-data/releases/download/weekly/weekly_<season>.csv.gz
3) nfl_data_py.import_weekly_data([season]) as a fallback

This avoids the parquet URL that returned 404 in your logs.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import pandas as pd

# Optional dependency: nfl_data_py (used only as a fallback)
try:
    import nfl_data_py as nfl  # type: ignore
except Exception:
    nfl = None

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_NFLVERSE_DIR = BASE_DIR / "data" / "raw" / "nflverse"
OUT_DIR = BASE_DIR / "data" / "weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "latest.csv"

NEEDED_COLS = [
    "player_id", "player_display_name", "recent_team", "opponent_team",
    "season", "week", "game_id", "position",
]
RENAME = {"player_display_name": "player_name"}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Season to pull (e.g., 2025)")
    return ap.parse_args()

def read_local_weekly_csv(season: int) -> pd.DataFrame | None:
    """
    Try local cache first: data/raw/nflverse/weekly_<season>.csv(.gz)
    """
    RAW_NFLVERSE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".csv.gz", ".csv"):
        p = RAW_NFLVERSE_DIR / f"weekly_{season}{ext}"
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                print(f"Using local nflverse cache: {p}")
                return df
            except Exception as e:
                print(f"WARNING: failed reading {p}: {e}")
    return None

def read_remote_weekly_csv(season: int) -> pd.DataFrame | None:
    """
    Try remote nflverse CSV (not parquet) using URL template if provided.
    """
    tpl = os.getenv(
        "NFLVERSE_WEEKLY_URL_TPL",
        "https://github.com/nflverse/nflverse-data/releases/download/weekly/weekly_{year}.csv.gz",
    )
    url = tpl.format(year=season)
    try:
        print(f"Fetching remote nflverse CSV: {url}")
        df = pd.read_csv(url, low_memory=False)
        return df
    except Exception as e:
        print(f"WARNING: remote weekly CSV fetch failed for {season}: {e}")
        return None

def read_weekly_via_nfl_data_py(season: int) -> pd.DataFrame | None:
    """
    Final fallback: nfl_data_py.import_weekly_data([season]).
    """
    if nfl is None:
        print("nfl_data_py is not installed; skipping fallback.")
        return None
    try:
        print(f"Fetching weekly via nfl_data_py for season {season}")
        df = nfl.import_weekly_data([season])
        return df
    except Exception as e:
        print(f"WARNING: nfl_data_py fallback failed for {season}: {e}")
        return None

def normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        # Try to repair: some datasets use 'gameday' or lack 'position'
        if "gameday" in df.columns and "game_date" not in df.columns:
            df = df.rename(columns={"gameday": "game_date"})
        # If position missing, add empty
        if "position" not in df.columns:
            df["position"] = ""
        # Re-check minimal set
        missing = [c for c in NEEDED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Weekly data missing required columns: {missing}")

    # Rename to canonical names expected elsewhere
    df = df.rename(columns=RENAME)

    # Clean & types
    for c in ["player_id", "player_name", "recent_team", "opponent_team", "position"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # enforce ints where possible
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Drop obviously bad rows
    df = df.dropna(subset=["player_id", "season", "week", "game_id"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # Stable sort
    df = df.sort_values(["season", "week", "game_id", "player_id"], kind="stable").reset_index(drop=True)
    return df

def main() -> None:
    args = parse_args()
    season = int(args.season)
    print(f"Building weekly latest for season={season}")

    # 1) Local cache (preferred)
    df = read_local_weekly_csv(season)
    # 2) Remote CSV
    if df is None:
        df = read_remote_weekly_csv(season)
    # 3) nfl_data_py fallback
    if df is None:
        df = read_weekly_via_nfl_data_py(season)

    if df is None or df.empty:
        sys.exit(f"ERROR: Could not obtain weekly data for season {season} from any source.")

    try:
        df = normalize_weekly(df)
    except Exception as e:
        sys.exit(f"ERROR: failed to normalize weekly data for {season}: {e}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Wrote {OUT_CSV} ({len(df)} rows) for season {season}")

if __name__ == "__main__":
    main()
