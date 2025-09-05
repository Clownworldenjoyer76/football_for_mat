#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train pregame regression models on strictly numeric features and save
only the fitted estimator per target.

Inputs
------
data/features/weekly_clean.csv.gz

Outputs
-------
models/pregame/<target>.joblib
output/models/metrics_summary.csv
"""

from pathlib import Path
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR = REPO / "models" / "pregame"
OUT_DIR = REPO / "output" / "models"
SUMMARY_CSV = OUT_DIR / "metrics_summary.csv"

# Identifier columns (excluded from X)
ID_COLS = [
    "player_id",
    "player_name",
    "recent_team",     # may be missing in some builds; handled below
    "position",
    "season",
    "week",
    "season_type",
]

# Targets to train if present
TARGETS = [
    "air_yards_share","attempts","carries","completions","dakota",
    "fantasy_points","fantasy_points_ppr","interceptions","pacr",
    "passing_2pt_conversions","passing_air_yards","passing_epa",
    "passing_first_downs","passing_tds","passing_yards",
    "passing_yards_after_catch","racr","receiving_2pt_conversions",
    "receiving_air_yards","receiving_epa","receiving_first_downs",
    "receiving_fumbles","receiving_fumbles_lost","receiving_tds",
    "receiving_yards","receiving_yards_after_catch","receptions",
    "rushing_2pt_conversions","rushing_epa","rushing_first_downs",
    "rushing_fumbles","rushing_fumbles_lost","rushing_tds",
    "rushing_yards","sack_fumbles","sack_fumbles_lost","sack_yards",
    "sacks","special_teams_tds","target_share","targets","wopr",
    # add any additional targets here if your features file includes them
]

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def die(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}")
    sys.exit(1)

def load_features() -> pd.DataFrame:
    if not FEATS.exists():
        die(f"missing features file '{FEATS.as_posix()}'")
    try:
        df = pd.read_csv(FEATS)
    except Exception as e:
        die(f"cannot read '{FEATS.as_posix()}': {e}")

    # normalize optional ID columns (create empty if missing)
    for c in ID_COLS:
        if c not in df.columns:
            # do NOT fail here; create harmless placeholder to keep IDs aligned
            df[c] = "" if c in ("player_name","recent_team","position","season_type") else 0

    return df

def pick_regressor():
    """Return a robust regressor available on the runner."""
    # Avoid importing heavy libs unless installed
    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    except Exception as e:
        die(f"sklearn not available: {e}")

def numeric_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Start from all columns except IDs + target
    cols_drop = [c for c in ID_COLS if c in df.columns] + [target]
    base = df.drop(columns=cols_drop, errors="ignore")
    # Keep strictly numeric; coerce anything else to NaN then fill
    for c in base.columns:
        if not np.issubdtype(base[c].dtype, np.number):
            base[c] = pd.to_numeric(base[c], errors="coerce")
    X = base.select_dtypes(include=[np.number]).copy()
    # Guard: empty feature matrix is not useful
    if X.shape[1] == 0:
        die(f"no numeric features available after filtering for target '{target}'")
    return X.fillna(0.0)

def fit_one(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return None, None
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy()
    X = numeric_X(df, target)
    model = pick_regressor()
    model.fit(X, y)
    # Ensure feature names are attached for alignment at inference
    try:
        setattr(model, "feature_names_in_", X.columns.to_numpy())
    except Exception:
        pass
    return model, {"rows": int(X.shape[0]), "features": int(X.shape[1])}

def main():
    df = load_features()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for tgt in TARGETS:
        model, info = fit_one(df, tgt)
        if model is None:
            continue

        # Save ONLY the estimator (no tuples, no extras)
        mpath = MODELS_DIR / f"{tgt}.joblib"
        joblib.dump(model, mpath, compress=3)

        # Basic in-sample metrics (quick, deterministic)
        X = pd.DataFrame({})  # placeholder to satisfy lints
        try:
            # Recompute numeric X quickly for metrics
            X = numeric_X(df, tgt)
            pred = model.predict(X)
            y = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0).to_numpy()

            mae = float(np.mean(np.abs(y - pred)))
            # R^2 (manual) to avoid extra imports
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            mae, r2 = float("nan"), float("nan")

        rows.append({
            "target": tgt,
            "rows": info["rows"],
            "features": info["features"],
            "mae": mae,
            "r2": r2,
            "model_file": mpath.as_posix(),
        })

    # Write summary
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    print(f"Saved models -> {MODELS_DIR.as_posix()}")
    print(f"Wrote metrics -> {SUMMARY_CSV.as_posix()}")

if __name__ == "__main__":
    main()
