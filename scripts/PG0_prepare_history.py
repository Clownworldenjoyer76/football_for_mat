#!/usr/bin/env python3
# Build player & opponent rolling features from past games only (no leakage)
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/pregame"); OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2019, 2025))  # adjust as needed
ROLL_G = 8                       # rolling window

def load_weekly():
    fp = Path("data/features/weekly_clean.csv.gz")
    if fp.exists():
        return pd.read_csv(fp, low_memory=False)
    # Fallback to nfl_data_py raw (expects prior step 01 to have saved)
    # If not, user can replace path to your weekly source
    fp2 = Path("data/features/weekly.csv.gz")
    return pd.read_csv(fp2, low_memory=False)

def safe_cols(df, wanted):
    return [c for c in wanted if c in df.columns]

def main():
    df = load_weekly()
    # minimal columns (be flexible)
    base = safe_cols(df, [
        "season","week","player_id","player_name","position","team","opponent",
        "home_away","attempts","completions","passing_yards","rushing_yards",
        "receptions","receiving_yards","targets","snaps","snap_pct"
    ])
    df = df[base].copy()

    # normalize types
    for c in ["season","week"]: 
        if c in df: df[c] = df[c].astype(int)
    for c in ["passing_yards","rushing_yards","receiving_yards","receptions","attempts","targets","snaps","snap_pct"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # home flag
    if "home_away" in df.columns:
        df["is_home"] = (df["home_away"].str.upper().fillna("") != "AWAY").astype(int)
    else:
        df["is_home"] = 1

    # rolling per-player (last ROLL_G games BEFORE current)
    df = df.sort_values(["player_id","season","week"])
    def roll_player(g):
        r = g[["passing_yards","rushing_yards","receiving_yards","receptions","targets","snaps","snap_pct"]]
        rolled = r.rolling(ROLL_G, min_periods=1).mean().shift(1)
        rolled.columns = [f"plyr_{c}_ma{ROLL_G}" for c in rolled.columns]
        return pd.concat([g, rolled], axis=1)
    df = df.groupby("player_id", group_keys=False).apply(roll_player)

    # rolling opponent allowed (team defense, last ROLL_G, shift 1)
    # build opponent view: what this opponent allowed to the player's position
    pos_key = df["position"].fillna("UNK")
    allowed = df.groupby(["opponent","position"]).apply(
        lambda g: g[["passing_yards","rushing_yards","receiving_yards","receptions"]]
                  .rolling(ROLL_G, min_periods=1).mean().shift(1)
    )
    allowed = allowed.reset_index(level=[0,1]).rename(columns={
        "opponent":"opp_team","position":"opp_pos"
    })
    allowed.columns = [*allowed.columns[:-4], "opp_pass_yds_ma", "opp_rush_yds_ma", "opp_rec_yds_ma", "opp_receptions_ma"]

    # merge opponent allowed back (key by opponent & position & same row order)
    key = df[["opponent","position"]].reset_index(drop=True)
    allow_vals = allowed[[ "opp_pass_yds_ma","opp_rush_yds_ma","opp_rec_yds_ma","opp_receptions_ma" ]].reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), allow_vals], axis=1)

    # export history features for training/inference
    out = df
    out.to_csv(OUT/"history_rolling_ma.csv.gz", index=False)
    print(f"✓ wrote {OUT/'history_rolling_ma.csv.gz'} ({len(out)} rows)")

if __name__ == "__main__":
    main()
