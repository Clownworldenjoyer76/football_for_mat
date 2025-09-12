#!/usr/bin/env python3
# scripts/03_train_models.py
# Robust model trainer:
# - maps friendly targets -> actual columns (when needed)
# - DROPS rows with NaN/inf in the target BEFORE fit (critical fix)
# - logs row counts before/after filtering
# - writes versioned artifact names with git short sha + date
# - updates metrics summary incrementally (append-safe)
from __future__ import annotations

import json
import math
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(".")
FEATURES = ROOT / "data" / "features" / "weekly_clean.csv.gz"
CONFIG   = ROOT / "config" / "models.yml"
OUTPUT_DIR = ROOT / "output" / "models"
LOG_DIR    = ROOT / "output" / "logs"
METRICS_CSV = ROOT / "output" / "models" / "metrics_summary.csv"
MODELS_DIR  = ROOT / "models" / "pregame"

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------- small YAML loader without pyyaml hard dependency ----------
def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # ultra-minimal fallback (expects simple YAML only)
        text = path.read_text(encoding="utf-8")
        # this won’t handle complex YAML; your repo already installs pyyaml via requirements,
        # so the try branch will normally run.
        import re
        data = {}
        key = None
        for line in text.splitlines():
            if re.match(r"^\s*#", line) or not line.strip():
                continue
            if ":" in line and not line.strip().startswith("-"):
                k, v = line.split(":", 1)
                key = k.strip()
                v = v.strip()
                if v == "" or v.lower() == "null":
                    data[key] = None
                elif v.lower() in ("true","false"):
                    data[key] = (v.lower()=="true")
                else:
                    data[key] = v
            elif line.strip().startswith("-") and key:
                data.setdefault(key, [])
                data[key].append(line.strip()[1:].strip())
        return data

# --------------------- utilities ---------------------
def log(msg: str):
    print(msg)
    with (LOG_DIR / "step03.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

def git_short_sha() -> str:
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return sha or "nogit"
    except Exception:
        return "nogit"

def today_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _is_finite_series(s: pd.Series) -> pd.Series:
    return np.isfinite(pd.to_numeric(s, errors="coerce"))

# --------------------- training core ---------------------
def load_config():
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG}")
    cfg = _load_yaml(CONFIG)
    # defaults
    seed = int(cfg.get("seed", 42) or 42)
    version_tag = cfg.get("version_tag", "auto")
    est_name = (cfg.get("estimator") or "random_forest").lower()
    rf_params = cfg.get("random_forest") or {}
    targets = cfg.get("targets") or []
    features_path = cfg.get("features_path") or str(FEATURES)
    id_columns = set(cfg.get("id_columns") or [])

    return {
        "seed": seed,
        "version_tag": version_tag,
        "estimator": est_name,
        "rf_params": rf_params,
        "targets": targets,
        "features_path": features_path,
        "id_columns": id_columns,
    }

# Map friendly names to actual feature columns when needed
TARGET_ALIASES = {
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "wrte_receptions": "receptions",
    # the new ones (exact names already match columns)
    "pass_attempts": "pass_attempts",
    "rush_attempts": "rush_attempts",
    "qb_sacks_taken": "qb_sacks_taken",
}

def select_actual_column(df: pd.DataFrame, target: str) -> tuple[str, str]:
    """
    Returns (reported_target_name, actual_column_name)
    The reported name is what we log/manifest as 'target'
    """
    # exact column present?
    if target in df.columns:
        return target, target
    # alias mapping?
    if target in TARGET_ALIASES and TARGET_ALIASES[target] in df.columns:
        return target, TARGET_ALIASES[target]
    # heuristic fallbacks
    candidates = [
        target,
        TARGET_ALIASES.get(target, ""),
    ]
    candidates += [
        target.replace("qb_", "").replace("rb_", "").replace("wrte_", "").replace("wr_", ""),
        target.replace("rec_", "receiving_"),
        target.replace("pass_", "passing_"),
        target.replace("rush_", "rushing_"),
    ]
    candidates = [c for c in dict.fromkeys(candidates) if c and c in df.columns]
    if candidates:
        return target, candidates[0]
    raise KeyError(f"Target column not found for '{target}'")

def build_estimator(cfg) -> RandomForestRegressor:
    params = dict(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=cfg["seed"],
    )
    params.update({k: v for k, v in cfg["rf_params"].items() if v is not None})
    return RandomForestRegressor(**params)

def train_one(df: pd.DataFrame, cfg, target_name: str) -> dict | None:
    reported, actual = select_actual_column(df, target_name)

    # Drop rows with NaN/inf in the target — FIX for your crash
    y_raw = pd.to_numeric(df[actual], errors="coerce")
    mask = y_raw.notna() & _is_finite_series(y_raw)
    n_before = len(df)
    df_train = df.loc[mask].copy()
    y = y_raw.loc[mask]
    n_after = len(df_train)

    if n_after == 0:
        log(f"[SKIP] '{reported}' has 0 usable rows after dropping NaN/inf.")
        return None
    if n_after < 100:
        log(f"[WARN] '{reported}' has few rows after filtering (n={n_after}/{n_before}). Results may be unstable.")

    # Feature matrix: numeric columns excluding id columns and the target itself
    drop_cols = set(cfg["id_columns"]) | {actual}
    X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors="ignore")
    # keep only numeric columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols].copy()

    # if any remaining NaN in features, fill with 0 (simple, model-agnostic)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # train/test split (deterministic)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=cfg["seed"])

    est = build_estimator(cfg)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)

    mae = float(mean_absolute_error(yte, pred))
    r2  = float(r2_score(yte, pred))

    # Save artifact
    git_sha = git_short_sha()
    tag = cfg["version_tag"]
    if not tag or tag == "auto":
        tag = f"{today_tag()}_{git_sha}"

    fname = f"{reported}_{tag}.joblib"
    fpath = (MODELS_DIR / fname).resolve()

    # write model
    from joblib import dump
    dump(est, fpath)

    sha = sha256_file(fpath)
    built_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    log(f"[OK] Saved {fname} for '{reported}' (actual='{actual}') sha256={sha[:10]}...  mae={mae:.3f} r2={r2:.3f}")

    # Append metrics row (append-safe)
    row = {
        "target": reported,
        "actual_column": actual,
        "artifact_path": str(fpath),
        "artifact_sha256": sha,
        "git_commit": git_sha,
        "random_seed": cfg["seed"],
        "built_at_utc": built_at,
        "mae": mae,
        "r2": r2,
        "version_tag": tag,
    }
    _append_metrics(row)

    return row

def _append_metrics(row: dict):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if METRICS_CSV.exists():
        dfm = pd.read_csv(METRICS_CSV)
        dfm = pd.concat([dfm, pd.DataFrame([row])], ignore_index=True)
        dfm.to_csv(METRICS_CSV, index=False)
    else:
        pd.DataFrame([row]).to_csv(METRICS_CSV, index=False)
    log(f"[INFO] Wrote metrics -> {METRICS_CSV}")

def train_all():
    if not FEATURES.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES}")
    df = pd.read_csv(FEATURES, low_memory=False)

    cfg = load_config()
    targets = cfg["targets"]
    if not targets:
        log("[WARN] No targets specified in config/models.yml -> targets")
        return

    # For visibility, write out the feature column list
    (LOG_DIR / "features_columns.txt").write_text("\n".join(df.columns), encoding="utf-8")
    log(f"[INFO] Wrote feature column list -> {LOG_DIR / 'features_columns.txt'}")

    for t in targets:
        try:
            _ = train_one(df, cfg, t)
        except KeyError as e:
            log(f"[WARN] Target column missing, skipping: {t}")
        except Exception as e:
            log(f"[ERROR] Training failed for target={t}: {type(e).__name__}: {e}")
            raise

    log(f"[INFO] Models dir -> {MODELS_DIR}")

if __name__ == "__main__":
    train_all()
