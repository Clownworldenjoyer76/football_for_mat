#!/usr/bin/env python3
# scripts/04_predict.py

import os
from pathlib import Path
import argparse
import sys
import pandas as pd
from joblib import load as joblib_load

TARGETS = {
    # out_csv                  model_filename              features_csv (expected)
    "qb_passing_yards.csv":    ("qb_passing_yards.joblib",   "qb_passing_yards.csv"),
    "rb_rushing_yards.csv":    ("rb_rushing_yards.joblib",   "rb_rushing_yards.csv"),
    "wr_rec_yards.csv":        ("wr_rec_yards.joblib",       "wr_rec_yards.csv"),
    "wrte_receptions.csv":     ("wrte_receptions.joblib",    "wrte_receptions.csv"),
}

MODELS_DIR = Path("models/pregame")
FEATS_DIR  = Path("data/features")
OUT_DIR    = Path("data/predictions")

REQ_COLS_MIN = ["season","week","game_id","player_id","player_name","team","opponent"]
# Any additional feature columns are fed to the model.

def die(msg: str) -> None:
    print(f"[predict:ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def warn(msg: str) -> None:
    print(f"[predict:WARN] {msg}")

def info(msg: str) -> None:
    print(f"[predict] {msg}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=int(os.getenv("TARGET_SEASON", "0")) or None)
    return p.parse_args()

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"Missing required columns in features: {missing}")

def main():
    args = parse_args()
    if args.season is None:
        die("Season not provided. Set env TARGET_SEASON or pass --season.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrote_any = False

    for out_csv, (model_fname, feats_fname) in TARGETS.items():
        model_path = MODELS_DIR / model_fname
        feats_path = FEATS_DIR  / feats_fname
        out_path   = OUT_DIR    / out_csv

        if not model_path.exists():
            warn(f"Model not found: {model_path}. Skipping {out_csv}.")
            continue
        if not feats_path.exists():
            warn(f"Features not found: {feats_path}. Skipping {out_csv}.")
            continue

        info(f"Loading model: {model_path.name}")
        model = joblib_load(model_path)

        info(f"Loading features: {feats_path.name}")
        feats = pd.read_csv(feats_path)

        # Basic metadata check
        ensure_cols(feats, REQ_COLS_MIN)
        # Use all non-metadata columns as features for prediction
        meta_cols = set(REQ_COLS_MIN)
        X_cols = [c for c in feats.columns if c not in meta_cols]
        if not X_cols:
            die(f"No feature columns found in {feats_fname} (only metadata present).")

        # Predict
        try:
            preds = model.predict(feats[X_cols])
        except Exception as e:
            die(f"Model.predict failed for {model_fname} on {feats_fname}: {e}")

        # Assemble output
        out = feats.copy()
        out["season"] = int(args.season)  # force current season
        out["pred"] = preds

        # Add required fields if missing
        required_for_props = ["season","week","game_id","player_id","player_name","team","opponent","pred"]
        for c in required_for_props:
            if c not in out.columns:
                out[c] = None

        # Trim/output columns
        out = out[required_for_props]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        wrote_any = True
        info(f"Wrote {out_path} rows={len(out)}")

    if not wrote_any:
        die("No predictions were written (missing models and/or features).")

    info(f"Done. Season={args.season}. Files in {OUT_DIR}")

if __name__ == "__main__":
    main()
