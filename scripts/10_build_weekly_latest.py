#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_build_weekly_latest.py

Builds player-week data for a target season into:
  data/weekly/latest.csv

Inputs:
  - --season <int>  (e.g., 2025). If omitted, defaults to current year.
  - Optional: data/processed/schedules/schedules_merged.csv is used only
    for sanity (season/week existence), but is NOT required.

Output columns (common downstream needs):
  player_id, player_name, recent_team, opponent_team,
  season, week, game_id, position, player_display_name (alias for name)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import pandas as pd

# Dependency: nfl_data_py
try:
    import nfl_data_py as nfl
except Exception as e:
    sys.exit(f"ERROR: nfl_data_py import failed: {e}")

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "latest.csv"

NEEDED_COLS = [
    "player_id", "player_display_name", "recent_team", "opponent_team",
    "season", "week", "game_id", "position",
]

RENAME = {
    "player_display_name": "player_name"
}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Season to pull (e.g., 2025)")
    return ap.parse_args()

def resolve_season(arg_season: int | None) -> int:
    if arg_season:
        return int(arg_season)
    env_season = os.getenv("SEASON")
    if env_season:
        return int(env_season)
    # Fallback: use current calendar year
    from datetime import datetime
    return datetime.utcnow().year

def fetch_weekly(season: int) -> pd.DataFrame:
    # nfl.import_weekly_data returns player-week stats; we only need identifying columns
    df = nfl.import_weekly_data([season])
    missing = [c for c in ["player_id","player_display_name","recent_team","opponent_team","week","season","game_id"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: weekly data missing columns: {missing}")

    # Keep essential columns
    keep = [c for c in NEEDED_COLS if c in df.columns]
    df = df[keep].copy()

    # Rename to canonical names expected elsewhere
    df = df.rename(columns=RENAME)

    # Basic cleaning
    for c in ["player_id","player_name","recent_team","opponent_team","position"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Drop obvious junk rows
    df = df.dropna(subset=["player_id","season","week","game_id"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # Sort for stable diffs
    df = df.sort_values(["season","week","game_id","player_id"], kind="stable").reset_index(drop=True)
    return df

def main() -> None:
    args = parse_args()
    season = resolve_season(args.season)
    print(f"Building weekly latest for season={season}")

    df = fetch_weekly(season)
    if df.empty:
        sys.exit(f"ERROR: weekly data came back empty for season {season}")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Wrote {OUT_CSV} ({len(df)} rows)")

if __name__ == "__main__":
    main()
