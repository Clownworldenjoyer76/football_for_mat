#!/usr/bin/env python3
"""
Pull injury reports and derive actives/inactives using nfl_data_py.

Outputs:
  - data/raw/injuries/injury_reports_latest.csv
  - data/raw/injuries/injury_reports_{season}.csv  (if --season given)
Usage:
  python scripts/pull_injuries_actives.py --season 2025
  python scripts/pull_injuries_actives.py           # uses all seasons seen in schedules file if present
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import sys

import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:
    sys.stderr.write(f"ERROR: nfl_data_py not installed or failed to import: {e}\n")
    sys.exit(1)

RAW_DIR = Path("data/raw")
OUT_DIR = RAW_DIR / "injuries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULES_GLOB = RAW_DIR / "nflverse" / "schedules_*.csv.gz"

OUT_LATEST = OUT_DIR / "injury_reports_latest.csv"


def infer_seasons_from_schedules() -> list[int]:
    paths = sorted(SCHEDULES_GLOB.parent.glob(SCHEDULES_GLOB.name))
    if not paths:
        return []
    df_list = []
    for p in paths:
        try:
            df_list.append(pd.read_csv(p))
        except Exception:
            continue
    if not df_list:
        return []
    sched = pd.concat(df_list, ignore_index=True)
    if "season" not in sched.columns:
        return []
    seasons = sorted(pd.unique(sched["season"].dropna().astype(int)))
    return seasons


# Conservative "out" / "inactive" mapping
OUT_STATUSES = {
    "out", "doubtful", "injured reserve", "ir", "pup", "non football injury",
    "nfi", "suspended", "covid-19", "physically unable to perform", "did not practice",
}
QUESTIONABLE_STATUSES = {"questionable", "limited practice", "lp", "limited"}
PROBABLE_STATUSES = {"probable", "full practice", "fp"}


def normalize_status(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # compress common variations
    s = s.replace("_", " ").replace("-", " ")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None, help="Season (e.g., 2025). If omitted, infer from schedules file(s).")
    args = parser.parse_args()

    seasons = [args.season] if args.season else infer_seasons_from_schedules()
    if not seasons:
        # fall back to most recent 3 years
        current_year = datetime.now().year
        seasons = list(range(current_year - 2, current_year + 1))

    # Pull injuries
    try:
        inj = nfl.import_injury_reports(seasons)
    except Exception as e:
        sys.stderr.write(f"ERROR: nfl_data_py.import_injury_reports failed: {e}\n")
        sys.exit(1)

    # Standardize columns
    inj_cols = [c.lower().strip() for c in inj.columns]
    inj.columns = inj_cols

    # Keep useful fields if present
    keep_candidates = [
        "season", "week", "game_id",
        "player_id", "player_gsis_id", "player_name", "position", "team",
        "opponent", "report_primary_injury", "report_status", "practice_status",
        "entry_date",
    ]
    keep = [c for c in keep_candidates if c in inj.columns]
    inj = inj[keep].copy()

    # Derive status flags
    inj["report_status_norm"] = inj.get("report_status", "").apply(normalize_status) if "report_status" in inj else ""
    inj["practice_status_norm"] = inj.get("practice_status", "").apply(normalize_status) if "practice_status" in inj else ""

    # A player is considered OUT if either the official report says out-ish or practice is DNP
    is_out = inj["report_status_norm"].isin(OUT_STATUSES) | inj["practice_status_norm"].isin({"did not practice", "dnp"})
    is_q = inj["report_status_norm"].isin(QUESTIONABLE_STATUSES)
    is_prob = inj["report_status_norm"].isin(PROBABLE_STATUSES)

    inj["is_out"] = is_out.astype(int)
    inj["is_questionable"] = is_q.astype(int)
    inj["is_probable"] = is_prob.astype(int)

    # Prefer a single player_id column
    if "player_id" not in inj.columns and "player_gsis_id" in inj.columns:
        inj = inj.rename(columns={"player_gsis_id": "player_id"})

    # Deduplicate to last entry per player/game if entry_date exists
    if "entry_date" in inj.columns:
        inj["entry_dt"] = pd.to_datetime(inj["entry_date"], errors="coerce", utc=True)
        sort_cols = ["season", "week", "game_id", "player_id", "entry_dt"]
        existing = [c for c in sort_cols if c in inj.columns]
        inj = inj.sort_values(existing).drop_duplicates(
            subset=[c for c in ["season", "week", "game_id", "player_id"] if c in inj.columns],
            keep="last"
        )
        inj = inj.drop(columns=["entry_dt"], errors="ignore")

    inj["fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Write outputs
    OUT_LATEST.parent.mkdir(parents=True, exist_ok=True)
    inj.to_csv(OUT_LATEST, index=False)

    # Optional seasonal dump (useful for debugging/backtests)
    if args.season:
        seasonal_path = OUT_DIR / f"injury_reports_{args.season}.csv"
        inj.to_csv(seasonal_path, index=False)

    print(f"Wrote: {OUT_LATEST}")
    if args.season:
        print(f"Wrote: {seasonal_path}")


if __name__ == "__main__":
    main()
