#!/usr/bin/env python3
"""
Builds model features for four player-prop markets from the current props file.

INPUTS (must exist):
  - data/props/props_current.csv
  - (optional) data/weekly/latest.csv (used only to backfill week if missing)

OUTPUTS (must be produced with >=1 row each, else this script exits(1)):
  - data/features/qb_passing_yards.csv
  - data/features/rb_rushing_yards.csv
  - data/features/wr_rec_yards.csv
  - data/features/wrte_receptions.csv

All outputs include required metadata columns:
  season, week, game_id, player_id, player_name, team, opponent
Plus simple, stable features when available in props:
  line, odds_over, odds_under, book (if present)

Season is taken from env TARGET_SEASON or --season (required).
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Paths
PROPS_CUR = Path("data/props/props_current.csv")
WEEKLY_LATEST = Path("data/weekly/latest.csv")
OUT_DIR = Path("data/features")

# Markets to materialize: (search_term, out_filename, optional position filters)
MARKETS = [
    ("passing yards", "qb_passing_yards.csv", {"QB"}),
    ("rushing yards", "rb_rushing_yards.csv", {"RB"}),
    ("receiving yards", "wr_rec_yards.csv", {"WR"}),
    ("receptions",     "wrte_receptions.csv", {"WR","TE"}),
]

REQ_META = ["season","week","game_id","player_id","player_name","team","opponent"]
CANDIDATE_ID_COLS = ["player_id","playerId","id"]
CANDIDATE_NAME_COLS = ["player_name","player","name","playerFullName"]
CANDIDATE_TEAM_COLS = ["team","team_abbr","teamAbbr","team_short","team_id"]
CANDIDATE_OPP_COLS  = ["opponent","opp","opp_abbr","opponent_abbr","oppAbbr"]
CANDIDATE_WEEK_COLS = ["week","week_number","weekNumber"]
CANDIDATE_GAME_COLS = ["game_id","gameId","gsis_id","gameIdStr"]
CANDIDATE_POS_COLS  = ["position","pos","player_position"]

def die(msg: str) -> None:
    print(f"[features:ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def warn(msg: str) -> None:
    print(f"[features:WARN] {msg}")

def info(msg: str) -> None:
    print(f"[features] {msg}")

def pick_first_col(df: pd.DataFrame, candidates: list[str], default_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            if c != default_name and default_name in df.columns:
                # prefer exact default_name if present; else take first candidate found
                return default_name
            return c
    # ensure column exists
    df[default_name] = None
    return default_name

def read_week_fallback() -> int | None:
    if WEEKLY_LATEST.exists():
        try:
            wdf = pd.read_csv(WEEKLY_LATEST)
            # try common fields: 'week' or 'current_week'
            for c in ["week","current_week","wk","week_number"]:
                if c in wdf.columns and pd.notnull(wdf[c]).any():
                    val = int(pd.to_numeric(wdf[c], errors="coerce").dropna().iloc[0])
                    return val
        except Exception:
            pass
    return None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    env_season = os.getenv("TARGET_SEASON")
    p.add_argument("--season", type=int, default=int(env_season) if env_season and env_season.isdigit() else None)
    return p.parse_args()

def normalize_metadata(df: pd.DataFrame, season: int) -> pd.DataFrame:
    # Map common variants into required schema
    id_col   = pick_first_col(df, CANDIDATE_ID_COLS,   "player_id")
    name_col = pick_first_col(df, CANDIDATE_NAME_COLS, "player_name")
    team_col = pick_first_col(df, CANDIDATE_TEAM_COLS, "team")
    opp_col  = pick_first_col(df, CANDIDATE_OPP_COLS,  "opponent")
    week_col = pick_first_col(df, CANDIDATE_WEEK_COLS, "week")
    game_col = pick_first_col(df, CANDIDATE_GAME_COLS, "game_id")

    out = df.copy()

    # Ensure season
    out["season"] = season

    # Ensure week: keep if present; else backfill from weekly/latest.csv (if available)
    if out[week_col].isna().all():
        wk = read_week_fallback()
        if wk is not None:
            out[week_col] = wk

    # Reorder/rename to required names
    out = out.rename(columns={
        id_col: "player_id",
        name_col: "player_name",
        team_col: "team",
        opp_col: "opponent",
        week_col: "week",
        game_col: "game_id",
    })

    # Make sure all REQ_META exist
    for c in REQ_META:
        if c not in out.columns:
            out[c] = None

    # Keep duplicates minimal
    return out

def extract_features_for_market(props: pd.DataFrame, season: int, market_term: str, pos_allow: set[str] | None) -> pd.DataFrame:
    df = props.copy()

    # Filter by market string if 'market' column exists
    if "market" in df.columns:
        mask = df["market"].astype(str).str.contains(market_term, case=False, na=False)
        df = df[mask].copy()
    else:
        warn(f"'market' column not found; cannot filter by '{market_term}'. Using all rows.")

    # Position filter if available
    if pos_allow:
        pos_col = None
        for c in CANDIDATE_POS_COLS:
            if c in df.columns:
                pos_col = c
                break
        if pos_col is not None:
            df = df[df[pos_col].astype(str).str.upper().isin({p.upper() for p in pos_allow})].copy()

    # Normalize metadata
    df = normalize_metadata(df, season)

    # Attach simple, stable feature columns when present
    if "line" not in df.columns:
        df["line"] = pd.NA
    # common odds columns
    if "odds_over" not in df.columns:
        for alt in ["over_odds","odds_o","american_odds_over","overAmerican"]:
            if alt in df.columns:
                df["odds_over"] = df[alt]
                break
        else:
            df["odds_over"] = pd.NA
    if "odds_under" not in df.columns:
        for alt in ["under_odds","odds_u","american_odds_under","underAmerican"]:
            if alt in df.columns:
                df["odds_under"] = df[alt]
                break
        else:
            df["odds_under"] = pd.NA
    if "book" not in df.columns:
        for alt in ["sportsbook","book_name","book","source"]:
            if alt in df.columns:
                df["book"] = df[alt]
                break
        else:
            df["book"] = pd.NA

    # Final column order: meta + a few numeric features (others will be ignored by predict if not needed)
    cols = REQ_META + ["line","odds_over","odds_under","book"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]

    # Drop exact duplicates
    df = df.drop_duplicates()

    return df

def main() -> int:
    args = parse_args()
    if args.season is None:
        die("Season not provided. Set env TARGET_SEASON or pass --season.")

    if not PROPS_CUR.exists():
        die(f"Missing input: {PROPS_CUR}")

    # Load props
    try:
        props = pd.read_csv(PROPS_CUR)
    except Exception as e:
        die(f"Unable to read {PROPS_CUR}: {e}")

    # If props has season, prefer target season; else assign later
    if "season" in props.columns:
        props = props[(props["season"].astype(str) == str(args.season)) | (props["season"].isna())].copy()
        # If filtering dropped everything, we’ll still assign season downstream

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    for market_term, out_name, pos_allow in MARKETS:
        df_feat = extract_features_for_market(props, args.season, market_term, pos_allow)
        out_path = OUT_DIR / out_name
        df_feat.to_csv(out_path, index=False)
        info(f"Wrote {out_path} rows={len(df_feat)}")
        if len(df_feat) == 0:
            failures.append(out_name)

    if failures:
        fail_list = "\n  - " + "\n  - ".join(failures)
        die(f"No rows produced for:\n{fail_list}\nVerify props_current.csv has those markets for season {args.season}.")

    info("All feature files written successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
