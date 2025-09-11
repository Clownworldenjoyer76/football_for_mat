#!/usr/bin/env python3
# /scripts/03_train_models.py
# -*- coding: utf-8 -*-
"""
Train pregame regression models on strictly numeric features and save ONLY the
fitted estimator per target, with versioned filenames and full metadata for
reproducibility.

Inputs
------
- config/models.yml
- data/features/weekly_clean.csv.gz

Outputs
-------
- models/pregame/<target>_<tag>.joblib           (versioned)
- models/pregame/<target>.latest.joblib          (stable pointer)
- models/pregame/<target>_<tag>.meta.json        (sidecar metadata)
- output/models/metrics_summary.csv              (includes artifact sha256)
"""

from __future__ import annotations
import json, os, random, sys, subprocess, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import yaml

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "models.yml"
FEATS = REPO / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR = REPO / "models" / "pregame"
OUT_DIR = REPO / "output" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "metrics_summary.csv"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "nogit"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_config() -> dict:
    if not CONFIG.exists():
        raise SystemExit(f"Missing config: {CONFIG}")
    with CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)

def build_version_tag(cfg: dict) -> str:
    tag_cfg = str(cfg.get("version_tag", "auto"))
    if tag_cfg != "auto":
        return tag_cfg
    # auto: use YYYYMMDD + git short
    return datetime.now(timezone.utc).strftime("%Y%m%d") + "_" + git_commit_short()

def get_features_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Features file not found: {path}")
    return pd.read_csv(path, low_memory=False)

def numeric_X(df: pd.DataFrame, target: str, id_cols: list[str]) -> pd.DataFrame:
    cols_drop = [c for c in id_cols if c in df.columns] + [target]
    base = df.drop(columns=cols_drop, errors="ignore")
    # Keep numeric only; coerce errant types to NaN then fill
    X = base.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def get_estimator(cfg: dict, seed: int):
    est_name = cfg.get("estimator", "random_forest")
    if est_name == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as e:
            raise SystemExit(f"sklearn not available: {e}")
        params = cfg.get("random_forest", {}) | {"random_state": seed}
        return RandomForestRegressor(**params), {"name": "RandomForestRegressor", "params": params}
    else:
        raise SystemExit(f"Unsupported estimator: {est_name}")

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train_all() -> None:
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    tag = build_version_tag(cfg)
    commit = git_commit_short()
    built_utc = utc_now_iso()

    targets = list(cfg.get("targets", []))
    if not targets:
        raise SystemExit("No targets specified in config/models.yml -> targets: []")

    id_cols = list(cfg.get("id_columns", []))

    df = get_features_df(FEATS)

    rows = []
    for tgt in targets:
        y = df[tgt] if tgt in df.columns else None
        if y is None:
            print(f"[WARN] Target column missing, skipping: {tgt}")
            continue

        X = numeric_X(df, tgt, id_cols)
        est, est_info = get_estimator(cfg, seed)

        # Fit
        est.fit(X, y)

        # Save model (versioned + latest)
        versioned = MODELS_DIR / f"{tgt}_{tag}.joblib"
        latest = MODELS_DIR / f"{tgt}.latest.joblib"
        joblib.dump(est, versioned, compress=3)
        # Maintain a stable pointer (copy so checksum differs; that’s OK)
        shutil.copyfile(versioned, latest)

        # Sidecar metadata for manifest
        meta = {
            "target": tgt,
            "built_at_utc": built_utc,
            "git_commit": commit,
            "random_seed": seed,
            "features_path": str(FEATS),
            "data_sha256": sha256_file(FEATS) if FEATS.exists() else None,
            "estimator": est_info["name"],
            "hyperparameters": est_info["params"],
            "feature_columns": list(X.columns),
            "artifact_path": str(versioned),
            "artifact_sha256": sha256_file(versioned),
            "version_tag": tag,
        }
        meta_path = MODELS_DIR / f"{tgt}_{tag}.meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Quick in-sample metrics
        preds = est.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((preds - y) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) if len(y) else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

        rows.append({
            "target": tgt,
            "artifact_path": str(versioned),
            "artifact_sha256": meta["artifact_sha256"],
            "git_commit": commit,
            "random_seed": seed,
            "built_at_utc": built_utc,
            "mae": mae,
            "r2": r2,
            "version_tag": tag,
        })

        print(f"[OK] Saved {versioned.name}  sha256={meta['artifact_sha256'][:10]}...  mae={mae:.3f} r2={r2:.3f}")

    # Write metrics summary
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote metrics -> {SUMMARY_CSV}")
    print(f"Models dir -> {MODELS_DIR}")

if __name__ == "__main__":
    train_all()
