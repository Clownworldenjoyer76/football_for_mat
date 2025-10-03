#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: scripts/04_predict_pregame.py
Purpose: Make pregame predictions using the trained models with strict
         feature alignment to the training-time columns.

Inputs
------
- data/features/weekly_clean.csv.gz            (feature matrix)
- models/pregame/*.joblib                      (one model per target)
  Each .joblib may be:
    • a fitted sklearn estimator/pipeline, or
    • a tuple/list like (estimator, feature_names) or (estimator, meta)
      where one element is the estimator and one element may be a list of
      training feature names.

Optional filters (CLI or env)
-----------------------------
--season SEASON      (or env FORECAST_SEASON or TARGET_SEASON)
--week   WEEK        (or env FORECAST_WEEK)

Outputs
-------
- data/predictions/pregame/predictions_[season]_wk[week].csv.gz  (or predictions_all.csv.gz)
- output/predictions/pregame/predictions_[season]_wk[week].csv.gz
- Prints a short summary.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple

# ----- Paths -----
REPO_ROOT     = Path(__file__).resolve().parents[1]
FEATURES_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR    = REPO_ROOT / "models" / "pregame"
OUT_DIR_DATA  = REPO_ROOT / "data"   / "predictions" / "pregame"
OUT_DIR_OUT   = REPO_ROOT / "output" / "predictions" / "pregame"

# Identifier columns to include if present
PREFERRED_ID_COLS = [
    "player_id", "player_name", "recent_team", "position",
    "season", "week", "season_type", "team", "game_id"
]

def fail(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
    sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=str,
                   default=os.environ.get("FORECAST_SEASON", os.environ.get("TARGET_SEASON", "")).strip())
    p.add_argument("--week",   type=str,
                   default=os.environ.get("FORECAST_WEEK", "").strip())
    return p.parse_args()

def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        fail(f"missing features file '{FEATURES_FILE.as_posix()}'")
    try:
        return pd.read_csv(FEATURES_FILE)
    except Exception as e:
        fail(f"cannot read '{FEATURES_FILE.as_posix()}': {e}")

def select_id_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in PREFERRED_ID_COLS if c in df.columns]
    if not cols:
        fail(f"no identifier columns found among {PREFERRED_ID_COLS}")
    return cols

def filter_rows(df: pd.DataFrame, season: str, week: str) -> pd.DataFrame:
    if season:
        df = df[df["season"].astype(str) == str(season)] if "season" in df.columns else df.iloc[0:0]
    if week:
        df = df[df["week"].astype(str) == str(week)] if "week" in df.columns else df.iloc[0:0]
    return df

def _unwrap_estimator(obj: object) -> Tuple[object, List[str]]:
    est = None
    feat_names: List[str] = []
    if hasattr(obj, "predict"):
        est = obj
    if est is None and isinstance(obj, (tuple, list)):
        for part in obj:
            if hasattr(part, "predict"):
                est = part
                break
        for part in obj:
            if isinstance(part, (list, tuple)) and all(isinstance(x, str) for x in part):
                feat_names = list(part)
                break
    if est is None:
        fail("loaded model object is not a predictor (no .predict)")
    if hasattr(est, "feature_names_in_") and len(getattr(est, "feature_names_in_")) > 0:
        feat_names = list(est.feature_names_in_)
    return est, feat_names

def load_models() -> dict:
    if not MODELS_DIR.exists():
        fail(f"models directory not found: {MODELS_DIR.as_posix()}")
    models = {}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            raw = joblib.load(p)
        except Exception as e:
            fail(f"failed loading '{p.name}': {e}")
        est, feats = _unwrap_estimator(raw)
        models[p.stem] = {"estimator": est, "features": feats}
    if not models:
        fail(f"no models found in {MODELS_DIR.as_posix()}")
    return models

def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0)

def main():
    args   = parse_args()
    season = args.season
    week   = args.week

    feat_df = load_features()
    if season or week:
        feat_df = filter_rows(feat_df, season, week)
        if feat_df.empty:
            sw = f"season={season or 'ALL'}, week={week or 'ALL'}"
            fail(f"no rows to predict for filter: {sw}")

    id_cols = select_id_cols(feat_df)
    base_X  = feat_df.drop(columns=id_cols, errors="ignore")

    model_map = load_models()

    out = feat_df[id_cols].copy()
    preds_made = 0

    for target, pack in model_map.items():
        est   = pack["estimator"]
        feats = pack["features"]
        if not feats:
            if hasattr(est, "feature_names_in_"):
                feats = list(est.feature_names_in_)
            else:
                feats = [c for c in base_X.columns if c not in id_cols]
        X = base_X.reindex(columns=feats, fill_value=0)
        X = to_numeric_frame(X)
        try:
            yhat = est.predict(X)
        except Exception as e:
            fail(f"prediction failed for model '{target}': {e}")
        out[target] = yhat
        preds_made += 1

    if preds_made == 0:
        fail("no predictions produced")

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

    print("Pregame predictions complete")
    print(f"Rows: {len(out):,} | Targets: {preds_made}")
    print(f"Wrote: {f_data.as_posix()}")
    print(f"Wrote: {f_out.as_posix()}")

if __name__ == "__main__":
    main()
