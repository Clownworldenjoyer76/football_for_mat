#!/usr/bin/env python3
"""
Canonicalize latest processed odds into data/props/props_current.csv
and stamp with TARGET_SEASON.

- Maps provider market keys to your repo's canonical names.
- If props_current.csv exists, season is overwritten to TARGET_SEASON.
- Otherwise builds from latest data/odds/processed/odds_*.csv.

Env:
  TARGET_SEASON (preferred) or SEASON
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROPS_CUR = Path("data/props/props_current.csv")
ODDS_PROC_DIR = Path("data/odds/processed")

# Provider → canonical
MARKET_MAP = {
    # existing
    "player_pass_yards":        "qb_passing_yards",
    "player_rush_yards":        "rb_rushing_yards",
    "player_rec_yards":         "wr_rec_yards",
    "player_receptions":        "wrte_receptions",
    # new
    "player_pass_tds":          "qb_passing_tds",
    "player_interceptions":     "qb_interceptions",
    "player_touchdowns":        "player_tds",
    "player_sacks":             "player_sacks",
    "player_tackles":           "player_tackles",
    "player_tackle_assists":    "player_tackles_assists",
    "player_field_goals_made":  "player_field_goals_made",
}

def _get_target_season() -> int:
    s = (os.getenv("TARGET_SEASON") or os.getenv("SEASON") or "").strip()
    if not s.isdigit():
        print("ERROR: TARGET_SEASON/SEASON must be set (e.g., 2025).", file=sys.stderr)
        sys.exit(2)
    return int(s)

def _latest_processed_odds() -> Optional[Path]:
    if not ODDS_PROC_DIR.exists():
        return None
    cands = sorted(ODDS_PROC_DIR.glob("odds_*.csv"))
    return cands[-1] if cands else None

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # light schema: keep columns we know, add if missing
    base = {
        "season": None, "week": None, "game_id": None,
        "player_id": None, "player_name": None, "team": None, "opponent": None,
        "market": None, "line": None, "odds_over": None, "odds_under": None, "book": None,
    }
    for c in base:
        if c not in df.columns:
            df[c] = base[c]
    return df

def main() -> int:
    season = _get_target_season()
    print(f"[canon] TARGET_SEASON={season}")

    if PROPS_CUR.exists():
        df = pd.read_csv(PROPS_CUR)
        df["season"] = season
        # Keep as-is otherwise; just ensure baseline columns exist
        df = _ensure_cols(df)
        df.to_csv(PROPS_CUR, index=False)
        print(f"[canon] Updated existing props_current.csv rows={len(df)}")
        return 0

    latest = _latest_processed_odds()
    if latest is None:
        print(f"[canon:ERROR] No processed odds found in {ODDS_PROC_DIR}.", file=sys.stderr)
        return 1

    src = pd.read_csv(latest, low_memory=False)

    # Map provider markets → canonical
    if "market" in src.columns:
        src["market"] = src["market"].map(lambda x: MARKET_MAP.get(str(x), str(x)))
    else:
        src["market"] = None

    # Try to derive a player name from 'runner' when it doesn't look like Over/Under
    player_name = None
    if "runner" in src.columns:
        s = src["runner"].astype(str)
        player_name = s.where(~s.str.lower().isin(["over","under"]), None)

    out = pd.DataFrame({
        "season": season,
        "week": None,  # will be filled later if you have a weekly file
        "game_id": src.get("game_id"),
        "player_id": None,  # no reliable id from odds feed
        "player_name": player_name if player_name is not None else None,
        "team": None,
        "opponent": None,
        "market": src.get("market"),
        "line": src.get("point"),
        # We don't get clear split over/under prices in all feeds; store unified price as over if unknown.
        "odds_over": src.get("price_american"),
        "odds_under": None,
        "book": src.get("book"),
    })

    out = _ensure_cols(out)
    out.to_csv(PROPS_CUR, index=False)

    seasons_present = sorted(set(pd.to_numeric(out.get("season"), errors="coerce").dropna().astype(int)))
    print(f"[canon] Built props_current.csv from {latest.name}; seasons={seasons_present}; rows={len(out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
