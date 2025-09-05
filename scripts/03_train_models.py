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

from __future__ import annotations

import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"

MODELS_DIR = REPO_ROOT / "models" / "pregame"
OUTPUT_METRICS = REPO_ROOT / "output" / "models" / "metrics_summary.csv"

# ---------------- Columns ----------------
ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",
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

# ---------------- Utils ----------------
def fail(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
    sys.exit(1)

def ensure_numeric(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """Coerce all non-ID, non-target columns to numeric safely."""
    keep = [c for c in df.columns if c not in drop_cols]
    X = df[keep].copy()

    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Fill remaining NaNs
    X = X.fillna(0)

    # Drop any columns that are still non-numeric (rare edge)
    numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    X = X[numeric_cols]

    return X

def purge_models_dir(models_dir: Path) -> None:
    """Delete existing joblib files to prevent stale/tuple objects."""
    if models_dir.exists():
        for p in models_dir.glob("*.joblib"):
            try:
                p.unlink()
            except Exception:
                # If unlink fails, fall back to move into tmp quarantine
                qdir = models_dir / "_quarantine"
                qdir.mkdir(exist_ok=True)
                shutil.move(str(p), str(qdir / p.name))
    else:
        models_dir.mkdir(parents=True, exist_ok=True)

def validate_saved_model(path: Path) -> None:
    """Load back and ensure the object has .predict (i.e., is an estimator)."""
    obj = joblib.load(path)
    if isinstance(obj, tuple) or not hasattr(obj, "predict"):
        raise TypeError(f"invalid saved object (not estimator) at {path.name}")

# ---------------- Main ----------------
def main() -> None:
    if not INPUT_FILE.exists():
        fail(f"missing input file '{INPUT_FILE.as_posix()}'")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        fail(f"cannot read '{INPUT_FILE.as_posix()}': {e}")

    # Verify ID columns
    missing_ids = [c for c in ID_COLS if c not in df.columns]
    if missing_ids:
        fail(f"required ID column(s) missing in {INPUT_FILE.name}: {missing_ids}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    # Purge old joblib artifacts (prevents tuple/legacy files)
    purge_models_dir(MODELS_DIR)

    metrics_rows: list[dict] = []

    # Precompute numeric features matrix (without ID + target each loop)
    # We build from the full frame each time to ensure proper alignment.
    base_numeric = ensure_numeric(df, drop_cols=ID_COLS)

    for target in TARGET_COLS:
        if target not in df.columns:
            # Skip silently if this target does not exist in the feature file
            continue

        # y
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        if y.nunique(dropna=False) <= 1:
            # Not learnable
            continue

        # X = numeric features minus the target itself (if present)
        cols_to_drop = set(ID_COLS + [target])
        feat_cols = [c for c in base_numeric.columns if c not in cols_to_drop]
        if not feat_cols:
            # No usable features
            continue

        X = base_numeric[feat_cols]

        # Train/validate split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model: BaseEstimator = RandomForestRegressor(
            n_estimators=200,  # a bit stronger by default
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Eval
        preds = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        # Save ONLY the estimator
        model_file = MODELS_DIR / f"{target}.joblib"
        joblib.dump(model, model_file, compress=3)

        # Validate round-trip load is a proper estimator
        try:
            validate_saved_model(model_file)
        except Exception as e:
            # Remove the bad artifact and fail hard
            if model_file.exists():
                try:
                    model_file.unlink()
                except Exception:
                    pass
            fail(f"model save/validate failed for '{target}': {e}")

        # Record metrics
        metrics_rows.append(
            {
                "target": target,
                "rows": int(len(df)),
                "features": int(len(feat_cols)),
                "mae": mae,
                "r2": r2,
                "model_file": model_file.as_posix(),
            }
        )

    if not metrics_rows:
        fail("no valid targets trained")

    pd.DataFrame(metrics_rows).sort_values("target").to_csv(OUTPUT_METRICS, index=False)
    print(f"Wrote metrics summary to {OUTPUT_METRICS.as_posix()}")
    print(f"Models saved to {MODELS_DIR.as_posix()} (count={len(metrics_rows)})")

if __name__ == "__main__":
    main()
