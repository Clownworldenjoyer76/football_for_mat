#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make pregame predictions using the trained models.

Inputs
------
- Feature matrix: data/features/weekly_clean.csv.gz

- Trained models: models/pregame/*.joblib
  (supports plain estimators, (model, meta) tuples, or {"model": estimator} dicts)

Optional filters
----------------
--season SEASON (e.g., 2025)
--week   WEEK   (e.g., 1)

Outputs
-------
- data/predictions/pregame/predictions_[season]_wk[week].csv.gz  (or predictions_all.csv.gz)
- output/predictions/pregame/predictions_[season]_wk[week].csv.gz (mirror)

This script is hardened to:
- Align inputs to the exact training feature list (model.feature_names_in_).
- Handle tuple/dict joblib artifacts.
- Tolerate missing non-essential ID columns (e.g., recent_team).
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR    = REPO_ROOT / "models" / "pregame"
OUT_DIR_DATA  = REPO_ROOT / "data"   / "predictions" / "pregame"
OUT_DIR_OUT   = REPO_ROOT / "output" / "predictions" / "pregame"

# Known identifier columns; we will only keep those that actually exist in the data.
KNOWN_ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",      # optional
    "team",             # optional, used if recent_team absent
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

    # If 'recent_team' is absent but 'team' exists, we will not fail; we’ll just
    # include whichever identifier columns are present later.
    return df

def filter_forecast_set(df: pd.DataFrame, season: str, week: str) -> pd.DataFrame:
    if season:
        df = df[df["season"].astype(str) == str(season)]
    if week:
        df = df[df["week"].astype(str) == str(week)]
    return df

def _unwrap_model(obj):
    """
    Accepts:
      - a plain estimator with .predict
      - a (model, meta) tuple
      - a dict like {"model": estimator, ...}
    Returns:
      (estimator, feature_names) where feature_names is either
      model.feature_names_in_ or None.
    """
    model = obj
    if isinstance(obj, tuple) and len(obj) >= 1:
        model = obj[0]
    elif isinstance(obj, dict) and "model" in obj:
        model = obj["model"]

    # Basic contract check
    if not hasattr(model, "predict"):
        fail("loaded artifact is not a valid estimator (no .predict)")

    # Preferred sklearn attribute
    feat_names = getattr(model, "feature_names_in_", None)
    # Some custom scripts stash a private name
    if feat_names is None:
        feat_names = getattr(model, "_feature_names_in", None)

    return model, feat_names

def load_models() -> dict:
    if not MODELS_DIR.exists():
        fail(f"models directory not found: {MODELS_DIR.as_posix()}")

    models = {}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            obj = joblib.load(p)
        except Exception as e:
            fail(f"failed loading '{p.name}': {e}")

        model, feat_names = _unwrap_model(obj)

        # If we still do not have a feature list, require retraining.
        if feat_names is None:
            fail(
                f"model '{p.name}' has no feature name list; "
                f"retrain so sklearn sets feature_names_in_, or save it explicitly."
            )

        models[p.stem] = {"model": model, "features": list(feat_names)}

    if not models:
        fail(f"no models found in {MODELS_DIR.as_posix()}")
    return models

def coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

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

    # Choose ID columns that actually exist
    id_cols = [c for c in KNOWN_ID_COLS if c in df.columns]
    if not id_cols:
        fail("no known identifier columns present in features file")

    # Base matrix for modeling (drop only the ID columns; per-model selection follows)
    base_X = df.drop(columns=id_cols, errors="ignore")

    # Load models with their feature lists
    bundle = load_models()

    # Start output with identifiers
    out = df[id_cols].copy()

    preds_made = 0
    for target, info in bundle.items():
        model = info["model"]
        feat_cols = info["features"]

        # Align strictly and in-order to training features
        X = base_X.reindex(columns=feat_cols, fill_value=0)
        X = coerce_numeric(X)

        try:
            yhat = model.predict(X)
        except Exception as e:
            fail(f"prediction failed for model '{target}': {e}")

        out[target] = yhat
        preds_made += 1

    if preds_made == 0:
        fail("no predictions produced")

    # Output locations
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
