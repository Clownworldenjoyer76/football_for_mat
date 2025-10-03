#!/usr/bin/env python3
"""
Generates current-season prediction CSVs required by scripts/04_generate_props.py.

Writes (all to data/predictions/):
  - qb_passing_yards.csv
  - rb_rushing_yards.csv
  - wr_rec_yards.csv
  - wrte_receptions.csv

Behavior:
- Uses TARGET_SEASON env or --season arg (default: current year).
- If trained models are present (models/calibrators.joblib or team models), you can extend
  the `predict_value` function to load and apply them.
- If no models are found, it falls back to a baseline derived from props/odds if available,
  otherwise a simple heuristic. The goal is to ensure 2025 rows exist so the pipeline runs.

Columns expected by downstream:
  season, week, game_id, player_id, player_name, team, opponent, market, pred
"""

import argparse, os, sys, json
from pathlib import Path
from datetime import datetime
import pandas as pd

DATA_DIR = Path("data")
PRED_DIR = DATA_DIR / "predictions"
PROPS_DIR = DATA_DIR / "props"

# ---- helpers ----------------------------------------------------------------

def season_default():
    # prefer env, else current year
    s = os.getenv("TARGET_SEASON")
    if s and s.isdigit():
        return int(s)
    return datetime.utcnow().year

def load_props_current():
    # optional: use props lines as weak baseline
    f = PROPS_DIR / "props_current.csv"
    if f.exists():
        try:
            return pd.read_csv(f)
        except Exception:
            return None
    return None

def predict_value(row, market, fallback_mean):
    # Placeholder prediction logic:
    # 1) If props line exists for this player/market, use it as the prediction.
    # 2) Else, fallback to a simple heuristic constant per market.
    line_col = "line"
    if line_col in row and pd.notnull(row[line_col]):
        try:
            return float(row[line_col])
        except Exception:
            pass
    return float(fallback_mean)

def normalize_cols(df, season, market):
    # Ensure required columns exist; fill missing with sensible defaults
    need = ["season","week","game_id","player_id","player_name","team","opponent","market","pred"]
    base = {k: None for k in need}
    base["season"] = season
    base["market"] = market
    for col in base:
        if col not in df.columns:
            df[col] = base[col]
    # Order columns
    return df[need]

def write_csv(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

# ---- main per-market builders -----------------------------------------------

def build_from_props(props_df, season, market, player_pos_filter=None, fallback_mean=50.0):
    df = props_df.copy()
    # Keep same-season rows if present; otherwise we’ll overwrite season below
    df["season"] = season
    # Filter by market name if column exists
    if "market" in df.columns:
        mask = df["market"].str.contains(market, case=False, na=False)
        df = df[mask].copy()
    # Minimal player columns
    if "player_name" not in df.columns:
        df["player_name"] = df.get("name") if "name" in df.columns else None
    if "team" not in df.columns:
        df["team"] = df.get("team_abbr") if "team_abbr" in df.columns else None
    if "opponent" not in df.columns:
        df["opponent"] = df.get("opp_abbr") if "opp_abbr" in df.columns else None
    if "week" not in df.columns:
        df["week"] = df.get("week_number") if "week_number" in df.columns else None
    if "game_id" not in df.columns:
        df["game_id"] = df.get("game_id")

    # Predict column
    df["pred"] = df.apply(lambda r: predict_value(r, market, fallback_mean), axis=1)
    df = normalize_cols(df, season, market)
    return df

def build_baseline(season, market):
    # Create a tiny baseline frame to keep pipeline moving even if no props found
    # (Downstream will still validate/clip later.)
    return normalize_cols(pd.DataFrame([], columns=[]), season, market)

# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=season_default())
    args = parser.parse_args()
    season = int(args.season)

    props = load_props_current()

    outputs = {
        "qb_passing_yards.csv": ("passing yards", 245.0),
        "rb_rushing_yards.csv": ("rushing yards", 62.0),
        "wr_rec_yards.csv": ("receiving yards", 58.0),
        "wrte_receptions.csv": ("receptions", 4.5),
    }

    wrote_any = False
    for fname, (market_key, fallback_mean) in outputs.items():
        if props is not None and not props.empty:
            df = build_from_props(props, season, market_key, fallback_mean=fallback_mean)
        else:
            df = build_baseline(season, market_key)
        out_path = PRED_DIR / fname
        write_csv(df, out_path)
        wrote_any = wrote_any or (len(df) > 0)

    print(f"[predict] season={season}; wrote files to {PRED_DIR.resolve()}")
    if not wrote_any:
        # Still succeed so pipeline can continue; downstream has its own checks.
        print("[predict] No rows generated (no props found) — baseline csvs created.")

if __name__ == "__main__":
    sys.exit(main())
