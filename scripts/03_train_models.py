#!/usr/bin/env python3
"""
Train per-market models from data/features/*.csv and write artifacts + metrics.

This aligns training with the feature files produced by 02_build_features.py,
so 04_predict.py gets matching schemas.

Outputs:
- models/pregame/<market>_<YYYYMMDD_git>.joblib   (timestamped)
- models/pregame/<legacy>.joblib                  (stable names for 4 legacy markets)
- output/models/metrics_summary.csv
- output/metrics_summary.json
- output/logs/features_columns.txt  (per-market feature lists, appended)
"""

from __future__ import annotations
import os, json, hashlib, warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data" / "features"
OUT_MODELS = REPO / "models" / "pregame"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "output" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Map feature-file stem -> stable legacy filename (what 04_predict.py expects)
LEGACY_STEMS = {
    "qb_passing_yards": "qb_passing_yards.joblib",
    "rb_rushing_yards": "rb_rushing_yards.joblib",
    "wr_rec_yards":     "wr_rec_yards.joblib",
    "wrte_receptions":  "wrte_receptions.joblib",
}

RF_PARAMS = dict(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)

DROP_TOKENS = {"player", "name", "team", "position", "pos", "opponent",
               "game", "date", "season", "week", "id", "book"}  # keep odds_*; drop 'book'

def git_short() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    return sha[:7] if sha else "local"

def version_tag() -> str:
    return f"{datetime.utcnow():%Y%m%d}_{git_short()}"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def list_feature_files() -> List[Path]:
    return sorted([p for p in FEAT_DIR.glob("*.csv") if p.name != "weekly_clean.csv" and p.stat().st_size > 0])

def select_features(df: pd.DataFrame, target_col: str | None = None) -> List[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = []
    tset = {target_col} if target_col else set()
    for c in num:
        lc = c.lower()
        if c in tset:
            continue
        if any(tok in lc for tok in DROP_TOKENS):
            continue
        feats.append(c)
    return feats

def train_one(stem: str, path: Path) -> Dict[str, object]:
    df = pd.read_csv(path, low_memory=False)

    # Choose a target:
    # If a true target column exists (e.g., historical outcomes), prefer it.
    # Otherwise, regress on the offered line to learn a line-model conditioned on odds.
    preferred_targets = ["projection", "target", "actual", "result", "value"]
    target_col = next((c for c in preferred_targets if c in df.columns), None)
    if target_col is None:
        target_col = "line"
        if target_col not in df.columns:
            warnings.warn(f"[WARN] '{stem}': no target/projection/line; skipping.")
            return dict(stem=stem, rows=0, artifact_path="", mae=None, r2=None, rmse=None)

    feats = select_features(df, target_col)
    if not feats:
        warnings.warn(f"[WARN] '{stem}': no usable numeric features; skipping.")
        return dict(stem=stem, rows=0, artifact_path="", mae=None, r2=None, rmse=None)

    # Drop rows with NaNs in y or X
    y = df[target_col]
    X = df[feats].copy()
    mask = y.notna()
    for c in feats:
        mask &= X[c].notna()
    Xc, yc = X.loc[mask], y.loc[mask]
    if len(yc) == 0:
        warnings.warn(f"[WARN] '{stem}': no valid rows after NaN filtering; skipping.")
        return dict(stem=stem, rows=0, artifact_path="", mae=None, r2=None, rmse=None)

    est = RandomForestRegressor(**RF_PARAMS)
    est.fit(Xc, yc)

    preds = est.predict(Xc)
    mae = float(mean_absolute_error(yc, preds))
    r2 = float(r2_score(yc, preds))
    rmse = float(mean_squared_error(yc, preds, squared=False))

    vt = version_tag()
    fname = f"{stem}_{vt}.joblib"
    out_path = OUT_MODELS / fname
    dump(est, out_path)

    # Also write/refresh stable legacy filename if known
    legacy = LEGACY_STEMS.get(stem)
    if legacy:
        (OUT_MODELS / legacy).write_bytes(out_path.read_bytes())

    # Append the feature list for visibility
    with (LOG_DIR / "features_columns.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{stem}] {len(feats)} features\n")
        for c in feats:
            f.write(f" - {c}\n")

    return dict(
        stem=stem,
        rows=int(len(yc)),
        artifact_path=str(out_path),
        artifact_sha256=sha256_file(out_path),
        mae=mae, r2=r2, rmse=rmse,
        features=len(feats)
    )

def main():
    files = list_feature_files()
    if not files:
        print("[ERROR] No feature CSVs found in data/features/. Run 02_build_features.py first.", flush=True)
        return 1

    results = [train_one(p.stem, p) for p in files]

    # Metrics CSV + JSON
    rows = []
    for r in results:
        rows.append({
            "market": r.get("stem"),
            "rows": r.get("rows"),
            "artifact_path": r.get("artifact_path", ""),
            "artifact_sha256": r.get("artifact_sha256", ""),
            "mae": r.get("mae"),
            "r2": r.get("r2"),
            "rmse": r.get("rmse"),
            "built_at_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": git_short(),
            "version_tag": version_tag(),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    json_out = REPO / "output" / "metrics_summary.json"
    with open(json_out, "w") as f:
        json.dump({r["market"]: {k: r[k] for k in ("rows","mae","rmse")} for r in rows}, f, indent=2)

    print("[INFO] Trained models for:", ", ".join([r["market"] for r in rows if r["rows"] > 0]))
    print("[INFO] Metrics ->", OUT_DIR / "metrics_summary.csv")
    print("[INFO] Models  ->", OUT_MODELS)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
