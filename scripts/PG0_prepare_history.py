#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data/pregame"); OUT.mkdir(parents=True, exist_ok=True)
ROLL_G = 8  # rolling window

# Try these in order (first existing file wins)
CANDIDATES = [
    ("parquet", "data/raw/weekly.parquet"),
    ("csv",     "data/raw/weekly.csv"),
    ("parquet", "data/features/weekly.parquet"),
    ("parquet", "data/features/weekly_all.parquet"),
    ("csv",     "data/features/weekly.csv.gz"),
    ("csv",     "data/features/weekly.csv"),
]

def load_weekly():
    for kind, p in CANDIDATES:
        fp = Path(p)
        if fp.exists():
            print(f"[PG0] using {fp}")
            return pd.read_parquet(fp) if kind == "parquet" else pd.read_csv(fp, low_memory=False)
    raise SystemExit(f"[PG0] No weekly file found. Tried: {[p for _,p in CANDIDATES]}")

def safe(df, cols):
    return [c for c in cols if c in df.columns]

def main():
    df = load_weekly()

    base_cols = safe(df, [
        "season","week","player_id","player_name","position",
        "team","posteam","recent_team","opponent","defteam",
        "home_away","home_team","away_team","is_home",
        "attempts","completions","passing_yards","rushing_yards",
        "receptions","receiving_yards","targets","snaps","snap_pct"
    ])
    df = df[base_cols].copy()

    # Standardize team/opponent
    if "team" not in df and "posteam" in df: df = df.rename(columns={"posteam":"team"})
    if "team" not in df and "recent_team" in df: df = df.rename(columns={"recent_team":"team"})
    if "opponent" not in df and "defteam" in df: df = df.rename(columns={"defteam":"opponent"})
    if "is_home" not in df:
        if "home_away" in df:
            df["is_home"] = (df["home_away"].astype(str).str.upper() != "AWAY").astype(int)
        else:
            df["is_home"] = 1

    for c in ["season","week"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["passing_yards","rushing_yards","receiving_yards","receptions","attempts","targets","snaps","snap_pct"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Sort for rolling
    sort_keys = [k for k in ["player_id","season","week"] if k in df.columns]
    df = df.sort_values(sort_keys)

    # Rolling per-player (shifted -> no leakage)
    num_cols = [c for c in ["passing_yards","rushing_yards","receiving_yards","receptions","targets","snaps","snap_pct"] if c in df.columns]
    def roll_player(g):
        r = g[num_cols]
        rolled = r.rolling(ROLL_G, min_periods=1).mean().shift(1)
        rolled.columns = [f"plyr_{c}_ma{ROLL_G}" for c in rolled.columns]
        return pd.concat([g.reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)
    if "player_id" not in df.columns:
        raise SystemExit("[PG0] weekly data missing player_id")
    df = df.groupby("player_id", group_keys=False).apply(roll_player)

    # Opponent allowed (position-aware, shift 1)
    pos = df["position"].fillna("UNK")
    base_allowed = [c for c in ["passing_yards","rushing_yards","receiving_yards","receptions"] if c in df.columns]
    if {"opponent","position"}.issubset(df.columns) and base_allowed:
        df = df.sort_values(["opponent","position","season","week"])
        allow = df.groupby(["opponent","position"])[base_allowed] \
                  .rolling(ROLL_G, min_periods=1).mean().shift(1) \
                  .reset_index(level=[0,1], drop=False)
        allow.columns = ["opponent","position"] + [f"opp_{c}_ma" for c in base_allowed]
        df = pd.concat([df.reset_index(drop=True), allow[[c for c in allow.columns if c.startswith("opp_")]].reset_index(drop=True)], axis=1)
    else:
        for nm in ["opp_passing_yards_ma","opp_rushing_yards_ma","opp_receiving_yards_ma","opp_receptions_ma"]:
            if nm not in df: df[nm] = np.nan

    OUT.mkdir(parents=True, exist_ok=True)
    out_fp = OUT / "history_rolling_ma.csv.gz"
    df.to_csv(out_fp, index=False)
    print(f"[PG0] wrote {out_fp} ({len(df)} rows)")

if __name__ == "__main__":
    main()
