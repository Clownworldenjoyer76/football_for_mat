#!/usr/bin/env python3
# scripts/02_build_features.py  (market alias aware, still strict)

import os, sys
from pathlib import Path
import argparse
import pandas as pd

PROPS_CUR = Path("data/props/props_current.csv")
WEEKLY_LATEST = Path("data/weekly/latest.csv")
OUT_DIR = Path("data/features")

# Each tuple: (aliases, out_filename, allowed_positions)
MARKETS = [
    (["passing yards","pass yards","player_pass_yards","player_passing_yards"], "qb_passing_yards.csv", {"QB"}),
    (["rushing yards","rush yards","player_rush_yards","player_rushing_yards"], "rb_rushing_yards.csv", {"RB"}),
    (["receiving yards","rec yards","player_rec_yards","player_receiving_yards"], "wr_rec_yards.csv", {"WR"}),
    (["receptions","player_receptions","player_receptions_total"], "wrte_receptions.csv", {"WR","TE"}),
]

REQ_META = ["season","week","game_id","player_id","player_name","team","opponent"]
CANDIDATE_ID_COLS = ["player_id","playerId","id"]
CANDIDATE_NAME_COLS = ["player_name","player","name","playerFullName"]
CANDIDATE_TEAM_COLS = ["team","team_abbr","teamAbbr","team_short","team_id"]
CANDIDATE_OPP_COLS  = ["opponent","opp","opp_abbr","opponent_abbr","oppAbbr"]
CANDIDATE_WEEK_COLS = ["week","week_number","weekNumber"]
CANDIDATE_GAME_COLS = ["game_id","gameId","gsis_id","gameIdStr"]
CANDIDATE_POS_COLS  = ["position","pos","player_position"]

def die(msg): print(f"[features:ERROR] {msg}", file=sys.stderr); sys.exit(1)
def warn(msg): print(f"[features:WARN] {msg}")
def info(msg): print(f"[features] {msg}")

def parse_args():
    env = os.getenv("TARGET_SEASON")
    return argparse.Namespace(season=int(env) if env and env.isdigit() else None)

def pick_first_col(df, candidates, default_name):
    for c in candidates:
        if c in df.columns: return c
    df[default_name] = None
    return default_name

def read_week_fallback():
    if WEEKLY_LATEST.exists():
        try:
            wdf = pd.read_csv(WEEKLY_LATEST)
            for c in ["week","current_week","wk","week_number"]:
                if c in wdf.columns and pd.notnull(wdf[c]).any():
                    return int(pd.to_numeric(wdf[c], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
    return None

def normalize_metadata(df, season):
    id_col   = pick_first_col(df, CANDIDATE_ID_COLS,   "player_id")
    name_col = pick_first_col(df, CANDIDATE_NAME_COLS, "player_name")
    team_col = pick_first_col(df, CANDIDATE_TEAM_COLS, "team")
    opp_col  = pick_first_col(df, CANDIDATE_OPP_COLS,  "opponent")
    week_col = pick_first_col(df, CANDIDATE_WEEK_COLS, "week")
    game_col = pick_first_col(df, CANDIDATE_GAME_COLS, "game_id")

    out = df.rename(columns={
        id_col:"player_id", name_col:"player_name", team_col:"team",
        opp_col:"opponent", week_col:"week", game_col:"game_id"
    }).copy()

    out["season"] = season
    if out["week"].isna().all():
        wk = read_week_fallback()
        if wk is not None:
            out["week"] = wk

    for c in REQ_META:
        if c not in out.columns: out[c] = None
    return out

def market_filter(df, aliases):
    if "market" not in df.columns:
        return df  # can’t filter, return all
    m = df["market"].astype(str).str.lower()
    al = [a.lower() for a in aliases]
    # match if any alias is substring OR exact
    mask = False
    for a in al:
        mask = mask | m.str.contains(a, na=False) | (m == a)
    return df[mask].copy()

def extract_features(props, season, aliases, pos_allow):
    df = props.copy()

    # Try filtering by aliases
    df_before = len(df)
    df = market_filter(df, aliases)
    info(f"Market match '{aliases[0]}' aliases={aliases}  rows_before={df_before} rows_after={len(df)}")

    # Optional position filter
    if pos_allow:
        pos_col = None
        for c in CANDIDATE_POS_COLS:
            if c in df.columns: pos_col = c; break
        if pos_col is not None:
            df = df[df[pos_col].astype(str).str.upper().isin({p.upper() for p in pos_allow})].copy()

    df = normalize_metadata(df, season)

    # Stable basic features
    if "line" not in df.columns: df["line"] = pd.NA
    if "odds_over" not in df.columns:
        for alt in ["over_odds","odds_o","american_odds_over","overAmerican","overAmericanOdds"]:
            if alt in df.columns: df["odds_over"] = df[alt]; break
        else: df["odds_over"] = pd.NA
    if "odds_under" not in df.columns:
        for alt in ["under_odds","odds_u","american_odds_under","underAmerican","underAmericanOdds"]:
            if alt in df.columns: df["odds_under"] = df[alt]; break
        else: df["odds_under"] = pd.NA
    if "book" not in df.columns:
        for alt in ["sportsbook","book_name","book","source"]:
            if alt in df.columns: df["book"] = df[alt]; break
        else: df["book"] = pd.NA

    cols = REQ_META + ["line","odds_over","odds_under","book"]
    for c in cols:
        if c not in df.columns: df[c] = pd.NA
    df = df[cols].drop_duplicates()
    return df

def main():
    args = parse_args()
    if args.season is None:
        die("Season not provided. Set env TARGET_SEASON.")

    if not PROPS_CUR.exists():
        die(f"Missing input: {PROPS_CUR}")

    try:
        props = pd.read_csv(PROPS_CUR)
    except Exception as e:
        die(f"Unable to read {PROPS_CUR}: {e}")

    # Prefer current season rows if present
    if "season" in props.columns:
        props = props[(props["season"].astype(str) == str(args.season)) | (props["season"].isna())].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    empty = []
    for aliases, out_name, pos_allow in MARKETS:
        df_feat = extract_features(props, args.season, aliases, pos_allow)
        out_path = OUT_DIR / out_name
        df_feat.to_csv(out_path, index=False)
        info(f"Wrote {out_path} rows={len(df_feat)}")
        if len(df_feat) == 0:
            empty.append(out_name)

    if empty:
        die(
            "No rows produced for:\n  - " + "\n  - ".join(empty) +
            f"\nCheck that props_current.csv has those markets (aliases supported) for season {args.season}."
        )

    info("All feature files written successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
