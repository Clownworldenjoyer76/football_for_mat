#!/usr/bin/env python3
"""
Train all stat targets and write artifacts + metrics.

Outputs:
- models/pregame/<target>_<YYYYMMDD_<gitshort>>.joblib  (timestamped)
- models/pregame/<legacy>.joblib  (only for the 4 legacy models)
- output/models/metrics_summary.csv
- output/models/metrics_summary.json
- output/logs/features_columns.txt  (for visibility)

Notes:
- Picks features = all numeric columns excluding target columns and common id/text columns.
- Skips a target if the actual column is missing or has NaNs.
"""

import os
import json
import gzip
import hashlib
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# -------------------------
# Config
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/
DATA_FEATURES = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
OUT_DIR = REPO_ROOT / "output" / "models"
LOG_DIR = REPO_ROOT / "output" / "logs"
MODELS_DIR = REPO_ROOT / "models" / "pregame"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Targets: {manifest_target_name: actual_column}
# (Includes your full list; legacy four are included at bottom for aliasing.)
TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "completions": "completions",
    "passing_tds": "passing_tds",
    "interceptions": "interceptions",
    "rushing_tds": "rushing_tds",
    "targets": "targets",
    "receiving_tds": "receiving_tds",
    "sacks": "sacks",
    "pass_attempts": "pass_attempts",
    "rush_attempts": "rush_attempts",
    "qb_sacks_taken": "qb_sacks_taken",
}

# Legacy filenames mapping → copy the newest model to these stable names too
LEGACY_LATEST = {
    "qb_passing_yards.joblib": ("passing_yards", "passing_yards"),
    "rb_rushing_yards.joblib": ("rushing_yards", "rushing_yards"),
    "wr_rec_yards.joblib": ("receiving_yards", "receiving_yards"),
    "wrte_receptions.joblib": ("receptions", "receptions"),
}

# Model hyperparams
RF_PARAMS = dict(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)

# -------------------------
# Helpers
# -------------------------
def git_short_sha() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha[:7]
    # local fallback
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "local"

def version_tag() -> str:
    return f"{datetime.utcnow():%Y%m%d}_{git_short_sha()}"

def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    if path.suffix == ".gz":
        # Pandas can read gzip directly, but we keep this for clarity
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def select_feature_columns(df: pd.DataFrame, target_cols: set) -> list:
    # Drop obvious non-features
    drop_like = {"player", "name", "team", "position", "pos", "opponent", "game", "date", "season", "week", "id"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = []
    for c in numeric_cols:
        lc = c.lower()
        if c in target_cols:
            continue
        if any(tok in lc for tok in drop_like):
            continue
        feats.append(c)
    return feats

def save_metrics_csv_json(rows: list, out_csv: Path, out_json: Path):
    # CSV
    dfm = pd.DataFrame(rows)
    dfm = dfm[
        ["target", "actual_column", "artifact_path", "artifact_sha256", "git_commit",
         "random_seed", "built_at_utc", "mae", "r2", "rmse", "rows", "version_tag"]
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(out_csv, index=False)

    # JSON (compact)
    j = {}
    for r in rows:
        j[r["target"]] = {
            "status": "ok" if r["rows"] > 0 else "skipped",
            "MAE": round(r["mae"], 4) if r["mae"] is not None else None,
            "RMSE": round(r["rmse"], 4) if r["rmse"] is not None else None,
            "rows": r["rows"],
        }
    with open(out_json, "w") as f:
        json.dump(j, f, indent=2)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------
# Train
# -------------------------
def main():
    vt = version_tag()
    commit = git_short_sha()
    built_at = datetime.utcnow().isoformat() + "Z"

    df = load_features(DATA_FEATURES)

    # figure feature columns
    target_cols = set(TARGETS.values())
    features = select_feature_columns(df, target_cols)
    (LOG_DIR / "features_columns.txt").write_text("\n".join(map(str, features)), encoding="utf-8")

    rows_for_metrics = []

    for target_name, actual_col in TARGETS.items():
        if actual_col not in df.columns:
            warnings.warn(f"[WARN] Target column missing, skipping: {target_name} (actual='{actual_col}')")
            rows_for_metrics.append({
                "target": target_name,
                "actual_column": actual_col,
                "artifact_path": "",
                "artifact_sha256": "",
                "git_commit": commit,
                "random_seed": RF_PARAMS["random_state"],
                "built_at_utc": built_at,
                "mae": None,
                "r2": None,
                "rmse": None,
                "rows": 0,
                "version_tag": vt,
            })
            continue

        y = df[actual_col]
        X = df[features].copy()

        # drop rows with NaNs in y or X
        mask = y.notna()
        for c in features:
            mask &= X[c].notna()
        Xc = X.loc[mask]
        yc = y.loc[mask]

        if len(yc) == 0:
            warnings.warn(f"[WARN] No valid rows for: {target_name}")
            rows_for_metrics.append({
                "target": target_name, "actual_column": actual_col,
                "artifact_path": "", "artifact_sha256": "",
                "git_commit": commit, "random_seed": RF_PARAMS["random_state"],
                "built_at_utc": built_at, "mae": None, "r2": None, "rmse": None,
                "rows": 0, "version_tag": vt,
            })
            continue

        est = RandomForestRegressor(**RF_PARAMS)
        est.fit(Xc, yc)

        preds = est.predict(Xc)
        mae = float(mean_absolute_error(yc, preds))
        r2 = float(r2_score(yc, preds))
        rmse = float(mean_squared_error(yc, preds, squared=False))

        # artifact paths
        fname = f"{target_name}_{vt}.joblib"
        artifact_path = MODELS_DIR / fname
        dump(est, artifact_path)

        artifact_sha = sha256_file(artifact_path)

        # record metrics row
        rows_for_metrics.append({
            "target": target_name,
            "actual_column": actual_col,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "git_commit": commit,
            "random_seed": RF_PARAMS["random_state"],
            "built_at_utc": built_at,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "rows": int(len(yc)),
            "version_tag": vt,
        })

    # Write metrics
    save_metrics_csv_json(
        rows_for_metrics,
        OUT_DIR / "metrics_summary.csv",
        REPO_ROOT / "output" / "metrics_summary.json",
    )

    # Also copy “latest” for legacy 4 (if we produced those targets)
    for legacy_filename, (target_key, _) in LEGACY_LATEST.items():
        produced = [r for r in rows_for_metrics if r["target"] == target_key and r["artifact_path"]]
        if not produced:
            continue
        # Pick the one we just built (same vt)
        latest_row = produced[-1]
        src = Path(latest_row["artifact_path"])
        dst = MODELS_DIR / legacy_filename
        # Copy bytes (avoid symlink issues on mobile clients)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

    print("[INFO] Wrote metrics ->", OUT_DIR / "metrics_summary.csv")
    print("[INFO] Models dir ->", MODELS_DIR)

if __name__ == "__main__":
    main()
