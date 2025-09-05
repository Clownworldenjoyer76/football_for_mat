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
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- Config ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"

MODELS_DIR = REPO_ROOT / "models" / "pregame"
OUTPUT_METRICS = REPO_ROOT / "output" / "models" / "metrics_summary.csv"

ID_COLS = [
    "player_id",
    "player_name",
    "position",
    "season",
    "week",
    "season_type",
]

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

    # make sure ID columns exist
    for c in ID_COLS:
        if c not in df.columns:
            fail(f"required column '{c}' missing in {INPUT_FILE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    for target in TARGET_COLS:
        if target not in df.columns:
            continue

        X = df.drop(columns=ID_COLS + [target])
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)

        if y.nunique() <= 1:
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        # Save only the model, not a tuple
        model_file = MODELS_DIR / f"{target}.joblib"
        joblib.dump(model, model_file)

        metrics_rows.append(
            {
                "target": target,
                "rows": len(df),
                "features": X.shape[1],
                "mae": mae,
                "r2": r2,
                "model_file": model_file.as_posix(),
            }
        )

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(OUTPUT_METRICS, index=False)
        print(f"Wrote metrics summary to {OUTPUT_METRICS}")
    else:
        fail("no valid targets trained")

if __name__ == "__main__":
    main()
