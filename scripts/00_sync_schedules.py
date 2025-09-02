#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sync NFL schedules for specified seasons.
Outputs:
  - data/raw/schedules/schedules_2024.csv
  - data/raw/schedules/schedules_2025.csv
  - data/processed/schedules/schedules_merged.csv
"""

from pathlib import Path
import sys
import pandas as pd

# Dependencies: nfl_data_py, pandas, pyarrow (for downstream compatibility)
try:
    import nfl_data_py as nfl
except Exception as e:
    sys.exit(f"ERROR: nfl_data_py import failed: {e}")

# Fixed target seasons per project instruction
TARGET_SEASONS = [2024, 2025]

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "schedules"
PROC_DIR = BASE_DIR / "data" / "processed" / "schedules"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
]

def fetch_season_df(season: int) -> pd.DataFrame:
    df = nfl.import_schedules([season])
    # Normalize column naming if source uses 'gameday'
    if "gameday" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"gameday": "game_date"})
    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: season {season} missing required columns: {missing}")
    # Basic canonicalization
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str).str.strip()
    # Sort for stable diffs
    df = df.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)
    return df

def validate(df_merged: pd.DataFrame) -> None:
    # Required columns
    missing = [c for c in REQUIRED_COLS if c not in df_merged.columns]
    if missing:
        sys.exit(f"ERROR: merged schedules missing required columns: {missing}")
    # Null checks
    if df_merged["game_id"].isna().any():
        sys.exit("ERROR: merged schedules contain null game_id")
    if df_merged["home_team"].eq("").any() or df_merged["away_team"].eq("").any():
        sys.exit("ERROR: merged schedules contain empty team codes")
    if df_merged["game_date"].isna().any():
        sys.exit("ERROR: merged schedules contain null game_date")
    # Uniqueness
    if df_merged.duplicated(subset=["game_id"]).any():
        dups = df_merged[df_merged.duplicated(subset=["game_id"], keep=False)]
        sys.exit(f"ERROR: duplicate game_id values detected:\n{dups[['season','week','game_id']].head(20)}")
    # Parseable dates (no exception)
    try:
        _ = pd.to_datetime(df_merged["game_date"])
    except Exception as e:
        sys.exit(f"ERROR: game_date not parseable: {e}")

def main() -> None:
    frames = []
    for season in TARGET_SEASONS:
        df = fetch_season_df(season)
        out_csv = RAW_DIR / f"schedules_{season}.csv"
        df.to_csv(out_csv, index=False)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)

    validate(merged)

    merged_out = PROC_DIR / "schedules_merged.csv"
    merged.to_csv(merged_out, index=False)

if __name__ == "__main__":
    main()
