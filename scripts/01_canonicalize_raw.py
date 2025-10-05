#!/usr/bin/env python3
"""
01_canonicalize_raw.py

Goal for this repo: make sure data/props/props_current.csv exists
and is stamped with the TARGET_SEASON so downstream steps (features/pipeline)
see season=2025.

Behavior:
1) If data/props/props_current.csv exists, overwrite/ensure its `season`
   column equals TARGET_SEASON and save in place.
2) Otherwise, build it from the latest data/odds/processed/odds_*.csv,
   ensure a `season` column set to TARGET_SEASON, and write to
   data/props/props_current.csv.

Environment:
- TARGET_SEASON (preferred) or SEASON (fallback)
"""

import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

PROPS_CUR = Path("data/props/props_current.csv")
ODDS_PROC_DIR = Path("data/odds/processed")

def _get_target_season() -> int:
    s = (os.getenv("TARGET_SEASON") or os.getenv("SEASON") or "").strip()
    if not s.isdigit():
        print("ERROR: TARGET_SEASON/SEASON must be set (e.g., 2025).", file=sys.stderr)
        sys.exit(2)
    return int(s)

def _latest_processed_odds() -> Optional[Path]:
    if not ODDS_PROC_DIR.exists():
        return None
    candidates = sorted(ODDS_PROC_DIR.glob("odds_*.csv"))
    return candidates[-1] if candidates else None

def _ensure_season_column(df: pd.DataFrame, season: int) -> pd.DataFrame:
    # Create/overwrite `season`
    df = df.copy()
    df["season"] = season
    return df

def _save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main() -> int:
    season = _get_target_season()
    print(f"[canon] TARGET_SEASON={season}")

    # Case 1: props_current.csv already exists -> just stamp season and save
    if PROPS_CUR.exists():
        try:
            df = pd.read_csv(PROPS_CUR)
        except Exception as e:
            print(f"[canon:ERROR] Could not read {PROPS_CUR}: {e}", file=sys.stderr)
            return 1

        df = _ensure_season_column(df, season)
        _save_csv(PROPS_CUR, df)

        seasons_present = sorted(set(pd.to_numeric(df.get("season"), errors="coerce").dropna().astype(int).tolist()))
        print(f"[canon] Updated existing props_current.csv, seasons present: {seasons_present}, rows={len(df)}")
        return 0

    # Case 2: build from latest processed odds CSV
    latest = _latest_processed_odds()
    if latest is None:
        print(f"[canon:ERROR] {PROPS_CUR} not found and no processed odds in {ODDS_PROC_DIR}.", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(latest)
    except Exception as e:
        print(f"[canon:ERROR] Could not read {latest}: {e}", file=sys.stderr)
        return 1

    df = _ensure_season_column(df, season)
    _save_csv(PROPS_CUR, df)

    seasons_present = sorted(set(pd.to_numeric(df.get("season"), errors="coerce").dropna().astype(int).tolist()))
    print(f"[canon] Built props_current.csv from {latest.name}, seasons present: {seasons_present}, rows={len(df)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
