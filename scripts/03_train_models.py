#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train models for all target columns using weekly feature data.

Input
-----
data/features/weekly_clean.csv.gz

Outputs
-------
models/pregame/{target}.joblib
output/models/metrics_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR = REPO_ROOT / "models" / "pregame"
OUTPUT_METRICS = REPO_ROOT / "output" / "models" / "metrics_summary.csv"

# ---------------- IDs & Targets ----------------
# Required ID columns present in weekly_clean.csv.gz (validated exactly).
REQUIRED_ID_COLS = [
    "player_id",
    "player_name",
    "team",          # NOTE: weekly_clean has 'team' (not 'recent_team')
    "position",
    "season",
    "week",
    "season_type",
]

# Target columns to try to train (skipped if missing).
TARGET_COLS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "pacr",
    "dakota",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "receiving_2pt_conversions",
    "racr",
    "target_share",
    "air_yards_share",
    "wopr",
    "special_teams_tds",
    "fantasy_points",
    "fantasy_points_ppr",
]

# ---------------- Helpers ----------------
def fail(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    if not INPUT_FILE.exists():
        fail(f"missing input file '{INPUT_FILE.as_posix()}'")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        fail(f"cannot read '{INPUT_FILE.as_posix()}': {e}")

    # Validate required IDs against the *actual* header
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing_ids:
        fail(f"required ID column(s) missing in {INPUT_FILE.name}: {missing_ids}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # Use only numeric feature columns to avoid strings leaking into X
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Never use numeric ID columns as features (season/week, etc.)
    numeric_feature_cols = [c for c in numeric_cols if c not in REQUIRED_ID_COLS]

    metrics_rows = []
    trained = 0
    skipped_missing = []
    skipped_constant = []

    for target in TARGET_COLS:
        if target not in df.columns:
            skipped_missing.append(target)
            continue

        # y: numeric, drop rows where target is NaN
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna()
        if valid.sum() < 2:
            skipped_constant.append(target)
            continue

        # X: numeric features excluding the target itself
        feat_cols = [c for c in numeric_feature_cols if c != target]
        if not feat_cols:
            skipped_constant.append(target)
            continue

        X = df.loc[valid, feat_cols].fillna(0.0)
        y = y.loc[valid]

        # If y has no variance, skip
        if y.nunique() <= 1:
            skipped_constant.append(target)
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        model_file = (MODELS_DIR / f"{target}.joblib")
        joblib.dump(model, model_file)

        metrics_rows.append(
            {
                "target": target,
                "rows": int(len(X)),
                "features": int(len(feat_cols)),
                "mae": float(mae),
                "r2": float(r2),
                "model_file": model_file.as_posix(),
            }
        )
        trained += 1

    if metrics_rows:
        pd.DataFrame(metrics_rows).sort_values("target").to_csv(OUTPUT_METRICS, index=False)
        print(f"Trained {trained} models. Wrote metrics summary to {OUTPUT_METRICS}")
        if skipped_missing:
            print(f"Skipped (missing): {', '.join(skipped_missing)}")
        if skipped_constant:
            print(f"Skipped (no variance / no features): {', '.join(skipped_constant)}")
    else:
        fail("no valid targets trained")

if __name__ == "__main__":
    main()
