#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make pregame predictions using the trained models.

Inputs
------
- Feature matrix: data/features/weekly_clean.csv.gz
  (same columns used during training)

- Trained models: models/pregame/*.joblib
  (one per target)

Optional filters
----------------
--season SEASON (e.g., 2025)
--week   WEEK   (e.g., 1)

Outputs
-------
- data/predictions/pregame/predictions_[season]_wk[week].csv.gz  (or predictions_all.csv.gz)
- output/predictions/pregame/predictions_[season]_wk[week].csv.gz (mirrored copy)
- Prints a short summary of what was predicted.
"""

import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR    = REPO_ROOT / "models" / "pregame"
OUT_DIR_DATA  = REPO_ROOT / "data"   / "predictions" / "pregame"
OUT_DIR_OUT   = REPO_ROOT / "output" / "predictions" / "pregame"

# Must match training script
ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",
    "position",
    "season",
    "week",
    "season_type",
]

def fail(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
    sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Allow CLI or env (for GH Actions)
    p.add_argument("--season", type=str, default=os.environ.get("FORECAST_SEASON", "").strip())
    p.add_argument("--week",   type=str, default=os.environ.get("FORECAST_WEEK", "").strip())
    return p.parse_args()

def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        fail(f"missing features file '{FEATURES_FILE.as_posix()}'")
    try:
        df = pd.read_csv(FEATURES_FILE)
    except Exception as e:
        fail(f"cannot read '{FEATURES_FILE.as_posix()}': {e}")
    # verify ID cols
    for c in ID_COLS:
        if c not in df.columns:
            fail(f"required column '{c}' missing in {FEATURES_FILE}")
    return df

def filter_forecast_set(df: pd.DataFrame, season: str, week: str) -> pd.DataFrame:
    if season:
        # season col may be str or int
        df = df[df["season"].astype(str) == str(season)]
    if week:
        df = df[df["week"].astype(str) == str(week)]
    return df

def load_models() -> dict:
    if not MODELS_DIR.exists():
        fail(f"models directory not found: {MODELS_DIR.as_posix()}")
    models = {}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            m = joblib.load(p)
        except Exception as e:
            fail(f"failed loading model '{p.name}': {e}")
        target = p.stem
        models[target] = m
    if not models:
        fail(f"no models found in {MODELS_DIR.as_posix()}")
    return models

def main():
    args = parse_args()
    season = args.season
    week   = args.week

    df = load_features()
    if season or week:
        df = filter_forecast_set(df, season, week)
        if df.empty:
            sw = f"season={season or 'ALL'}, week={week or 'ALL'}"
            fail(f"no rows to predict for filter: {sw}")

    models = load_models()

    # Build base output with identifiers
    out = df[ID_COLS].copy()

    # Feature matrix we’ll align per-model
    # (Drop identifiers only; the training script dropped ID_COLS + [target],
    # so we will select exactly model.feature_names_in_ for each model below.)
    base_X = df.drop(columns=ID_COLS, errors="ignore")

    # Predict each target with its corresponding model
    preds_made = 0
    for target, model in models.items():
        # Align columns to training-time features
        if hasattr(model, "feature_names_in_"):
            feat_cols = list(model.feature_names_in_)
        else:
            # Extremely defensive: if the attribute is missing, fall back to current columns.
            feat_cols = [c for c in base_X.columns if c not in ID_COLS]

        X = base_X.reindex(columns=feat_cols, fill_value=0)

        # Coerce any non-numeric columns that might have slipped in
        for c in X.columns:
            if not np.issubdtype(X[c].dtype, np.number):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        try:
            yhat = model.predict(X)
        except Exception as e:
            fail(f"prediction failed for target '{target}': {e}")

        out[target] = yhat
        preds_made += 1

    if preds_made == 0:
        fail("no predictions produced (no models?)")

    # Destinations
    OUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OUT.mkdir(parents=True, exist_ok=True)

    if season or week:
        s = season if season else "ALL"
        w = week if week else "ALL"
        fname = f"predictions_{s}_wk{w}.csv.gz"
    else:
        fname = "predictions_all.csv.gz"

    f_data = OUT_DIR_DATA / fname
    f_out  = OUT_DIR_OUT  / fname

    out.to_csv(f_data, index=False, compression="gzip")
    out.to_csv(f_out,  index=False, compression="gzip")

    print(f"Predictions written:\n - {f_data.as_posix()}\n - {f_out.as_posix()}")
    print(f"Rows: {len(out):,} | Targets: {preds_made}")

if __name__ == "__main__":
    main()
