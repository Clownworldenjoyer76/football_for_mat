#!/usr/bin/env python3
# scripts/04_predict.py  (STRICT MODE)

import os, sys
from pathlib import Path
import argparse
import pandas as pd
from joblib import load as joblib_load

# out_csv                  model_filename              features_csv
TARGETS = {
    "qb_passing_yards.csv": ("qb_passing_yards.joblib", "qb_passing_yards.csv"),
    "rb_rushing_yards.csv": ("rb_rushing_yards.joblib", "rb_rushing_yards.csv"),
    "wr_rec_yards.csv":     ("wr_rec_yards.joblib",     "wr_rec_yards.csv"),
    "wrte_receptions.csv":  ("wrte_receptions.joblib",  "wrte_receptions.csv"),
}

MODELS_DIR = Path("models/pregame")
FEATS_DIR  = Path("data/features")
OUT_DIR    = Path("data/predictions")

META_COLS = ["season","week","game_id","player_id","player_name","team","opponent"]

def die(msg: str) -> "NoReturn":
    print(f"[predict:ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def info(msg: str) -> None:
    print(f"[predict] {msg}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=int(os.getenv("TARGET_SEASON", "0")) or None)
    return p.parse_args()

def ensure_required_files():
    missing = []
    for _, (model_fname, feats_fname) in TARGETS.items():
        if not (MODELS_DIR / model_fname).exists():
            missing.append(f"model:{MODELS_DIR / model_fname}")
        if not (FEATS_DIR / feats_fname).exists():
            missing.append(f"features:{FEATS_DIR / feats_fname}")
    if missing:
        die("Missing required inputs:\n  " + "\n  ".join(missing))

def load_features(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        die(f"Failed reading features {path.name}: {e}")
    # Determine feature columns (exclude metadata if present)
    x_cols = [c for c in df.columns if c not in set(META_COLS + ["market","pred","line"])]
    if not x_cols:
        die(f"No feature columns in {path.name} (only metadata present).")
    return df, x_cols

def ensure_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in META_COLS:
        if c not in df.columns:
            df[c] = None
    if "market" not in df.columns:
        df["market"] = None
    return df[["season","week","game_id","player_id","player_name","team","opponent","market","pred"]]

def main():
    args = parse_args()
    if args.season is None:
        die("Season not provided. Set env TARGET_SEASON or pass --season.")

    # Hard fail if any required model/features file is missing
    ensure_required_files()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    for out_csv, (model_fname, feats_fname) in TARGETS.items():
        model_path = MODELS_DIR / model_fname
        feats_path = FEATS_DIR  / feats_fname
        out_path   = OUT_DIR    / out_csv

        info(f"Loading model {model_path.name}")
        try:
            model = joblib_load(model_path)
        except Exception as e:
            die(f"Failed loading model {model_path.name}: {e}")

        info(f"Loading features {feats_path.name}")
        feats, x_cols = load_features(feats_path)

        # Predict
        try:
            preds = model.predict(feats[x_cols])
        except Exception as e:
            die(f"Model.predict failed for {model_path.name} on {feats_path.name}: {e}")

        out = feats.copy()
        out["season"] = int(args.season)  # enforce current season
        out["pred"] = preds
        out = ensure_output_cols(out)
        out.to_csv(out_path, index=False)
        info(f"Wrote {out_path} rows={len(out)}")
        wrote_any = True

    if not wrote_any:
        die("No predictions were written (unexpected).")

    info(f"Done. Season={args.season}. Outputs in {OUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
