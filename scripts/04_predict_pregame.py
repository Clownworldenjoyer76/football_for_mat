#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: scripts/04_predict_pregame.py
Purpose: Make pregame predictions using the trained models with strict
         feature alignment to the training-time columns. Robust to old
         legacy .joblib stubs: prefers freshly built timestamped models
         and skips unreadable/invalid artifacts instead of hard-failing.

Inputs
------
- data/features/weekly_clean.csv.gz            (feature matrix)
- models/manifest/models_manifest.csv          (model inventory; preferred)
- models/pregame/*.joblib                      (artifacts; timestamped + legacy)
  Each .joblib may be:
    • a fitted sklearn estimator/pipeline, or
    • a tuple/list like (estimator, feature_names) or (estimator, meta)
      where one element is the estimator and one element may be a list of
      training feature names.

Optional filters (CLI or env)
-----------------------------
--season SEASON      (or env FORECAST_SEASON)
--week   WEEK        (or env FORECAST_WEEK)

Outputs
-------
- data/predictions/pregame/predictions_[season]_wk[week].csv.gz  (or predictions_all.csv.gz)
- output/predictions/pregame/predictions_[season]_wk[week].csv.gz
- Prints a short summary.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Iterable, List, Tuple, Union, Dict

# ----- Paths -----
REPO_ROOT     = Path(__file__).resolve().parents[1]
FEATURES_FILE = REPO_ROOT / "data" / "features" / "weekly_clean.csv.gz"
MODELS_DIR    = REPO_ROOT / "models" / "pregame"
MANIFEST_CSV  = REPO_ROOT / "models" / "manifest" / "models_manifest.csv"
OUT_DIR_DATA  = REPO_ROOT / "data"   / "predictions" / "pregame"
OUT_DIR_OUT   = REPO_ROOT / "output" / "predictions" / "pregame"

# Identifier columns to include if present (do not hard-require any except that at least one exists)
PREFERRED_ID_COLS = [
    "player_id", "player_name", "recent_team", "position",
    "season", "week", "season_type", "team", "game_id"
]

# Recognized targets (used for ordering and basic sanity)
KNOWN_TARGETS_ORDER = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "completions",
    "passing_tds",
    "interceptions",
    "rushing_tds",
    "targets",
    "receiving_tds",
    "sacks",
    "pass_attempts",
    "rush_attempts",
    "qb_sacks_taken",
]

# Legacy filenames mapping → these are often committed in-repo; we treat them as lowest priority fallbacks
LEGACY_FILENAMES = {
    "qb_passing_yards.joblib": "passing_yards",
    "rb_rushing_yards.joblib": "rushing_yards",
    "wr_rec_yards.joblib": "receiving_yards",
    "wrte_receptions.joblib": "receptions",
}

def fail(msg: str) -> None:
    print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
    sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=str, default=os.environ.get("FORECAST_SEASON", "").strip())
    p.add_argument("--week",   type=str, default=os.environ.get("FORECAST_WEEK", "").strip())
    return p.parse_args()

# ----- Feature IO -----
def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        fail(f"missing features file '{FEATURES_FILE.as_posix()}'")
    try:
        df = pd.read_csv(FEATURES_FILE)
    except Exception as e:
        fail(f"cannot read '{FEATURES_FILE.as_posix()}': {e}")
    return df

def select_id_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in PREFERRED_ID_COLS if c in df.columns]
    if not cols:
        fail(f"no identifier columns found among {PREFERRED_ID_COLS}")
    return cols

def filter_rows(df: pd.DataFrame, season: str, week: str) -> pd.DataFrame:
    if season:
        df = df[df["season"].astype(str) == str(season)] if "season" in df.columns else df.iloc[0:0]
    if week:
        df = df[df["week"].astype(str) == str(week)] if "week" in df.columns else df.iloc[0:0]
    return df

# ----- Model loading with robust unwrapping -----
def _unwrap_estimator(obj: object) -> Tuple[object, List[str]]:
    """
    Return (estimator, feature_names) where:
      - estimator has .predict
      - feature_names are the training-time columns if known, else [] (caller will infer)
    Accepts:
      • estimator
      • (estimator, feature_list) or any tuple/list containing an estimator and a list of strings
    """
    est = None
    feat_names: List[str] = []

    # direct estimator
    if hasattr(obj, "predict"):
        est = obj

    # tuple/list forms
    if est is None and isinstance(obj, (tuple, list)):
        # find the estimator element
        for part in obj:
            if hasattr(part, "predict"):
                est = part
                break
        # find an explicit feature list if provided
        for part in obj:
            if isinstance(part, (list, tuple)) and all(isinstance(x, str) for x in part):
                feat_names = list(part)
                break

    if est is None:
        raise ValueError("loaded model object is not a predictor (no .predict)")

    # if estimator exposes feature_names_in_, prefer that
    if hasattr(est, "feature_names_in_"):
        fni = getattr(est, "feature_names_in_")
        try:
            if len(fni) > 0:
                feat_names = list(fni)
        except Exception:
            pass

    return est, feat_names

def _safe_joblib_load(p: Path):
    """Load a joblib artifact with clear error context."""
    try:
        return joblib.load(p)
    except Exception as e:
        raise RuntimeError(f"failed loading '{p.name}': {e}")

def _models_from_manifest() -> Dict[str, Path]:
    """
    Prefer models listed in the manifest (timestamped artifacts produced by the current pipeline).
    Returns dict[target] = path.
    """
    res: Dict[str, Path] = {}
    if not MANIFEST_CSV.exists():
        return res
    try:
        m = pd.read_csv(MANIFEST_CSV)
    except Exception:
        return res
    need = {"path", "target"}
    if not need.issubset(m.columns):
        return res
    for _, r in m.iterrows():
        tgt = str(r["target"]).strip()
        path = str(r["path"]).strip()
        if not tgt or not path:
            continue
        p = (REPO_ROOT / path).resolve()
        if p.exists() and p.suffix == ".joblib":
            res[tgt] = p
    return res

def _timestamped_models() -> Dict[str, Path]:
    """Fallback to any timestamped artifacts in models/pregame (name like <target>_YYYYMMDD_xxx.joblib)."""
    res: Dict[str, Path] = {}
    if not MODELS_DIR.exists():
        return res
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        stem = p.stem
        # detect "<target>_<yyyymmdd>_<gitshort>"
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 8:
            target = "_".join(parts[:-2])
            res[target] = p
    return res

def _legacy_models() -> Dict[str, Path]:
    """Lowest priority: legacy stable filenames that may be committed (or LFS stubs)."""
    res: Dict[str, Path] = {}
    if not MODELS_DIR.exists():
        return res
    for fname, target in LEGACY_FILENAMES.items():
        p = MODELS_DIR / fname
        if p.exists():
            res[target] = p
    return res

def load_models_prioritized() -> dict:
    """
    Load models with priority:
      1) manifest timestamped entries
      2) timestamped files in models/pregame
      3) legacy stable filenames (qb_passing_yards.joblib, etc.)
    Skip unreadable artifacts with a warning; only fail if none load.
    """
    paths_by_target: Dict[str, Path] = {}
    # Merge with priority (earlier entries win)
    for provider in (_models_from_manifest, _timestamped_models, _legacy_models):
        for tgt, p in provider().items():
            paths_by_target.setdefault(tgt, p)

    loaded = {}
    errors = []

    for tgt, p in paths_by_target.items():
        try:
            raw = _safe_joblib_load(p)
            est, feats = _unwrap_estimator(raw)
            loaded[tgt] = {"estimator": est, "features": feats, "path": p}
        except Exception as e:
            errors.append(str(e))
            print(f"[WARN] Skipping model for target '{tgt}': {e}", file=sys.stderr)

    if not loaded:
        # Give the most actionable error message
        if errors:
            fail("; ".join(errors))
        else:
            fail(f"no models found in {MODELS_DIR.as_posix()} or manifest {MANIFEST_CSV.as_posix()}")

    # Re-order for stable output
    ordered = {}
    for tgt in KNOWN_TARGETS_ORDER:
        if tgt in loaded:
            ordered[tgt] = loaded[tgt]
    # add any extras at the end
    for tgt in loaded:
        if tgt not in ordered:
            ordered[tgt] = loaded[tgt]

    return ordered

# ----- Prediction -----
def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0)

def main():
    args   = parse_args()
    season = args.season
    week   = args.week

    # load + (optional) filter
    feat_df = load_features()
    if season or week:
        feat_df = filter_rows(feat_df, season, week)
        if feat_df.empty:
            sw = f"season={season or 'ALL'}, week={week or 'ALL'}"
            fail(f"no rows to predict for filter: {sw}")

    # identifiers to carry through (whatever is present)
    id_cols = select_id_cols(feat_df)
    base_X  = feat_df.drop(columns=id_cols, errors="ignore")

    # load models (robust)
    model_map = load_models_prioritized()

    # build output
    out = feat_df[id_cols].copy()
    preds_made = 0
    used_models = []

    for target, pack in model_map.items():
        est   = pack["estimator"]
        feats = pack["features"]  # may be []
        # if no explicit feature list, align to current columns minus ids
        if not feats:
            if hasattr(est, "feature_names_in_"):
                feats = list(est.feature_names_in_)
            else:
                feats = [c for c in base_X.columns if c not in id_cols]

        # build aligned matrix
        X = base_X.reindex(columns=feats, fill_value=0)
        X = to_numeric_frame(X)

        try:
            yhat = est.predict(X)
        except Exception as e:
            print(f"[WARN] prediction failed for model '{target}': {e}", file=sys.stderr)
            continue

        out[target] = yhat
        preds_made += 1
        used_models.append(f"{target} <- {pack['path'].name}")

    if preds_made == 0:
        fail("no predictions produced (all models unreadable or incompatible)")

    OUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OUT.mkdir(parents=True, exist_ok=True)

    if season or week:
        s = season if season else "ALL"
        w = week if week else "ALL"
        fname = f"predictions_{s}_wk{w}.csv.gz"
    else:
        fname = "predictions_all.csv.gz"

    f_data = OUT_DIR_DATA / fname
    f_out  = OUT_DIR_OUT  / fname

    out.to_csv(f_data, index=False, compression="gzip")
    out.to_csv(f_out,  index=False, compression="gzip")

    print("Pregame predictions complete")
    print(f"Rows: {len(out):,} | Targets predicted: {preds_made}")
    print("Used models:")
    for m in used_models:
        print("  -", m)
    print(f"Wrote: {f_data.as_posix()}")
    print(f"Wrote: {f_out.as_posix()}")

if __name__ == "__main__":
    main()
