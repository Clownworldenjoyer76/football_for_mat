#!/usr/bin/env python3
"""
Build/refresh models manifest + lock file.

- Scans models/ (default: models/pregame/*.joblib) and records artifact metadata.
- Enriches rows with training metrics from:
   1) output/models/metrics_summary.csv (legacy: mae, r2, version_tag, etc.)
   2) output/metrics_summary.json (new: MAE, RMSE, rows)
- Writes:
   - models/manifest/models_manifest.csv
   - models/manifest/models_manifest.lock.json
"""

from __future__ import annotations
import csv
import json
import hashlib
from pathlib import Path
from datetime import timezone, datetime

import pandas as pd

# ---------- Config ----------
MODELS_GLOB = "models/pregame/**/*.joblib"
MANIFEST_CSV = Path("models/manifest/models_manifest.csv")
LOCK_JSON    = Path("models/manifest/models_manifest.lock.json")

METRICS_CSV  = Path("output/models/metrics_summary.csv")     # legacy, has r2 + artifact paths
METRICS_JSON = Path("output/metrics_summary.json")           # new, has MAE/RMSE/rows

# ---------- Helpers ----------
def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iso_utc(ts: float | int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def safe_str(x):
    return "" if x is None else str(x)

# ---------- Load metrics ----------
def load_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize expected cols
    for col in [
        "target","actual_column","artifact_path","artifact_sha256","git_commit",
        "random_seed","built_at_utc","mae","r2","version_tag"
    ]:
        if col not in df.columns:
            df[col] = ""
    return df

def load_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    # Normalize keys: {target: {"MAE": float, "RMSE": float, "rows": int, ...}}
    norm = {}
    for tgt, vals in raw.items():
        if isinstance(vals, dict):
            norm[tgt] = {
                "MAE": vals.get("MAE"),
                "RMSE": vals.get("RMSE"),
                "rows": vals.get("rows")
            }
    return norm

# ---------- Build manifest rows ----------
def build_manifest_rows() -> list[dict]:
    csv_metrics = load_metrics_csv(METRICS_CSV)
    json_metrics = load_metrics_json(METRICS_JSON)

    # Map legacy CSV metrics by artifact_path (most specific) and by target (fallback)
    by_artifact = {}
    by_target   = {}
    if not csv_metrics.empty:
        for _, r in csv_metrics.iterrows():
            by_target[safe_str(r.get("target",""))] = r
            ap = safe_str(r.get("artifact_path",""))
            if ap:
                by_artifact[ap] = r

    rows = []
    for path in sorted(Path(".").glob(MODELS_GLOB)):
        rel_path = str(path.as_posix())

        # Basic file stats
        try:
            stat = path.stat()
            filesize = stat.st_size
            mtime_iso = iso_utc(stat.st_mtime)
            file_sha = sha256_of_file(path)
        except FileNotFoundError:
            # ignore vanishing files
            continue

        filename = path.name

        # Infer target from filename if needed (strip .joblib)
        inferred_target = filename.replace(".joblib", "")

        # Pull legacy metrics (CSV), preferring artifact match
        legacy = by_artifact.get(rel_path) or by_target.get(inferred_target) or {}

        # Pull new metrics (JSON) by target key
        jsonm = json_metrics.get(inferred_target, {})

        # Compose row
        row = {
            "path": rel_path,
            "filename": filename,
            "sha256": file_sha,
            "filesize_bytes": filesize,
            "modified_time_utc": mtime_iso,

            # From legacy CSV when available
            "target": safe_str(legacy.get("target") or inferred_target),
            "version_tag": safe_str(legacy.get("version_tag")),
            "git_commit": safe_str(legacy.get("git_commit")),
            "random_seed": safe_str(legacy.get("random_seed")),
            "estimator": safe_str(legacy.get("estimator")),
            "hyperparameters_json": safe_str(legacy.get("hyperparameters_json")),
            "features_path": safe_str(legacy.get("features_path")),
            "data_sha256": safe_str(legacy.get("data_sha256")),
            "feature_count": safe_str(legacy.get("feature_count")),
            "built_at_utc": safe_str(legacy.get("built_at_utc")),
            "actual_column": safe_str(legacy.get("actual_column")),
            "mae": safe_str(legacy.get("mae")),
            "r2": safe_str(legacy.get("r2")),

            # Always include these, letting JSON override/augment
            "artifact_filename": filename,          # convenience, mirrors filename
            "artifact_sha256": file_sha,            # convenience, mirrors sha256

            # New fields from JSON
            "rmse": "",                             # filled below if present
            "rows": ""
        }

        # If JSON has fresher MAE/RMSE/rows, set/override
        if jsonm:
            if jsonm.get("MAE") is not None:
                row["mae"] = f"{float(jsonm['MAE']):.6g}"
            if jsonm.get("RMSE") is not None:
                row["rmse"] = f"{float(jsonm['RMSE']):.6g}"
            if jsonm.get("rows") is not None:
                row["rows"] = str(int(jsonm["rows"]))

        rows.append(row)

    return rows

# ---------- Write outputs ----------
def write_manifest_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Column order (stable)
    columns = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator","hyperparameters_json",
        "features_path","data_sha256","feature_count","built_at_utc",
        "actual_column","mae","r2","rmse","rows",
        "artifact_filename","artifact_sha256",
    ]

    # Ensure all keys exist
    for r in rows:
        for c in columns:
            r.setdefault(c, "")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_lock_json(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for r in rows:
        artifacts.append({
            "filename": r.get("filename",""),
            "path": r.get("path",""),
            "sha256": r.get("sha256",""),
            "target": r.get("target",""),
            "version_tag": r.get("version_tag",""),
        })
    with out_path.open("w") as f:
        json.dump({"artifacts": artifacts}, f, indent=2)

def main():
    rows = build_manifest_rows()
    write_manifest_csv(rows, MANIFEST_CSV)
    write_lock_json(rows, LOCK_JSON)
    print(f"[OK] Wrote {MANIFEST_CSV} and {LOCK_JSON} with {len(rows)} artifacts.")

if __name__ == "__main__":
    main()
