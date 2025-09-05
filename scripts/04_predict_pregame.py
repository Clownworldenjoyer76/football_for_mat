#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict with trained models. Robust to:
- Missing optional ID columns (creates empty placeholders)
- Older .joblib saved as tuples (uses element [0] as estimator)

Inputs
------
data/features/weekly_clean.csv.gz
models/pregame/*.joblib

Outputs
-------
data/predictions/pregame/predictions_<season>_wk<week>.csv.gz
output/predictions/pregame/predictions_<season>_wk<week>.csv.gz
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "data" / "features" / "weekly_clean.csv.gz"
MODELS = REPO / "models" / "pregame"
OUT_DATA = REPO / "data" / "predictions" / "pregame"
OUT_MIRR = REPO / "output" / "predictions" / "pregame"

ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",     # optional; placeholder if missing
    "position",
    "season",
    "week",
    "season_type",
]

def die(msg: str):
    print(f"INSUFFICIENT INFORMATION: {msg}")
    sys.exit(1)

def load_features() -> pd.DataFrame:
    if not FEATS.exists():
        die(f"missing features file '{FEATS.as_posix()}'")
    try:
        df = pd.read_csv(FEATS)
    except Exception as e:
        die(f"cannot read '{FEATS.as_posix()}': {e}")

    # normalize optional IDs so downstream never fails on column lookups
    for c in ID_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("player_name","recent_team","position","season_type") else 0
    return df

def filter_df(df: pd.DataFrame, season: str, week: str) -> pd.DataFrame:
    if season:
        df = df[df["season"].astype(str) == str(season)]
    if week:
        df = df[df["week"].astype(str) == str(week)]
    return df

def load_estimator(p: Path):
    obj = joblib.load(p)
    # accept (estimator, meta) tuples from older runs
    if isinstance(obj, tuple) and len(obj) > 0:
        obj = obj[0]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=os.environ.get("FORECAST_SEASON", "").strip())
    ap.add_argument("--week",   default=os.environ.get("FORECAST_WEEK", "").strip())
    args = ap.parse_args()

    df = load_features()
    if args.season or args.week:
        df = filter_df(df, args.season, args.week)
        if df.empty:
            die(f"no rows to predict for season='{args.season or 'ALL'}' week='{args.week or 'ALL'}'")

    if not MODELS.exists():
        die(f"models directory not found: {MODELS.as_posix()}")

    files = sorted(MODELS.glob("*.joblib"))
    if not files:
        die(f"no models in {MODELS.as_posix()}")

    # base output
    out = df[ID_COLS].copy()
    base_X = df.drop(columns=ID_COLS, errors="ignore")

    made = 0
    for mp in files:
        est = load_estimator(mp)
        if not hasattr(est, "predict"):
            die(f"loaded object has no predict(): {mp.name}")

        # align to training-time features if available
        if hasattr(est, "feature_names_in_"):
            cols = list(est.feature_names_in_)
        else:
            cols = [c for c in base_X.columns if c not in ID_COLS]

        X = base_X.reindex(columns=cols, fill_value=0)

        # numeric only
        for c in X.columns:
            if not np.issubdtype(X[c].dtype, np.number):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        try:
            yhat = est.predict(X)
        except Exception as e:
            die(f"prediction failed for model '{mp.stem}': {e}")

        out[mp.stem] = yhat
        made += 1

    if made == 0:
        die("no predictions produced")

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_MIRR.mkdir(parents=True, exist_ok=True)

    if args.season or args.week:
        s = args.season if args.season else "ALL"
        w = args.week if args.week else "ALL"
        fname = f"predictions_{s}_wk{w}.csv.gz"
    else:
        fname = "predictions_all.csv.gz"

    p1 = OUT_DATA / fname
    p2 = OUT_MIRR / fname
    out.to_csv(p1, index=False, compression="gzip")
    out.to_csv(p2, index=False, compression="gzip")

    print(f"Wrote:\n- {p1.as_posix()}\n- {p2.as_posix()}\nRows={len(out):,} Models={made}")

if __name__ == "__main__":
    main()
