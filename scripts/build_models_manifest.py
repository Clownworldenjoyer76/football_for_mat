#!/usr/bin/env python3
"""
Build/refresh models manifest by scanning models/pregame/*.joblib and
merging with output/models/metrics_summary.csv and output/metrics_summary.json.

Outputs:
- models/manifest/models_manifest.csv  (rich table)
- models/manifest/models_manifest.lock.json  (lightweight list for Release uploads)
"""

import os
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/
MODELS_DIR = REPO_ROOT / "models" / "pregame"
MANIFEST_DIR = REPO_ROOT / "models" / "manifest"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = REPO_ROOT / "output" / "models" / "metrics_summary.csv"
METRICS_JSON = REPO_ROOT / "output" / "metrics_summary.json"

CSV_OUT = MANIFEST_DIR / "models_manifest.csv"
LOCK_OUT = MANIFEST_DIR / "models_manifest.lock.json"

def git_short_sha() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha[:7]
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "local"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_target_from_filename(fn: str) -> str:
    """
    From 'passing_yards_20250912_aaaaaaa.joblib' -> 'passing_yards'
    From legacy 'qb_passing_yards.joblib' -> 'qb_passing_yards' (kept as-is)
    """
    base = fn.replace(".joblib", "")
    parts = base.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-3]) > 0:
        # looks like <target>_<yyyymmdd>_<sha>
        return "_".join(parts[:-2])
    return base

def maybe_version_tag(fn: str) -> str:
    # Try to pull <YYYYMMDD_sha> if present
    base = fn.replace(".joblib", "")
    parts = base.split("_")
    if len(parts) >= 3 and parts[-2].isdigit():
        return "_".join(parts[-2:])
    return ""

def main():
    commit = git_short_sha()

    # ingest metrics (if present)
    metrics_df = pd.DataFrame()
    if METRICS_CSV.exists():
        metrics_df = pd.read_csv(METRICS_CSV)
    metrics_map: Dict[str, Dict[str, Any]] = {}
    if not metrics_df.empty:
        for _, r in metrics_df.iterrows():
            metrics_map[r["target"]] = {
                "actual_column": r.get("actual_column"),
                "mae": r.get("mae"),
                "r2": r.get("r2"),
                "rmse": r.get("rmse"),
                "rows": r.get("rows"),
                "version_tag": r.get("version_tag"),
                "artifact_path": r.get("artifact_path"),
                "artifact_sha256": r.get("artifact_sha256"),
                "built_at_utc": r.get("built_at_utc"),
                "git_commit": r.get("git_commit"),
                "random_seed": r.get("random_seed"),
            }

    rows: List[Dict[str, Any]] = []
    lock_items: List[Dict[str, Any]] = []

    for p in sorted(MODELS_DIR.glob("*.joblib")):
        stat = p.stat()
        filename = p.name
        target = parse_target_from_filename(filename)
        vtag = maybe_version_tag(filename)

        # compute sha256 (fast enough for our set)
        sha256 = sha256_file(p)

        # merge metrics if we have them
        m = metrics_map.get(target, {})
        # if the legacy 4, we also try to map to their true actuals
        # (already embedded in metrics_map if trained in this run)

        row = {
            "path": str(p).replace(str(REPO_ROOT) + os.sep, "").replace("\\", "/"),
            "filename": filename,
            "sha256": sha256,
            "filesize_bytes": stat.st_size,
            "modified_time_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            "target": target,
            "version_tag": vtag or m.get("version_tag", ""),
            "git_commit": m.get("git_commit", commit),
            "random_seed": m.get("random_seed", 42),
            "estimator": "RandomForestRegressor",
            "hyperparameters_json": '{"n_estimators":400,"max_depth":null,"n_jobs":-1,"random_state":42}',
            "features_path": "data/features/weekly_clean.csv.gz",
            "data_sha256": "",  # optional: fill if you hash the features file
            "feature_count": None,  # optional: fill if you record it
            "built_at_utc": m.get("built_at_utc", ""),
            "actual_column": m.get("actual_column", ""),
            "mae": m.get("mae", ""),
            "r2": m.get("r2", ""),
            "rmse": m.get("rmse", ""),
            "rows": m.get("rows", ""),
            "artifact_filename": filename,
            "artifact_sha256": sha256,
        }
        rows.append(row)

        lock_items.append({
            "filename": filename,
            "path": row["path"],
            "sha256": sha256,
            "target": target,
            "version_tag": row["version_tag"],
        })

    # Also append metrics rows that don’t have a corresponding artifact
    for t, m in metrics_map.items():
        has_artifact = any(r["target"] == t for r in rows)
        if has_artifact:
            continue
        rows.append({
            "path": "",
            "filename": "",
            "sha256": "",
            "filesize_bytes": "",
            "modified_time_utc": datetime.utcnow().isoformat() + "Z",
            "target": t,
            "version_tag": m.get("version_tag", ""),
            "git_commit": m.get("git_commit", commit),
            "random_seed": m.get("random_seed", 42),
            "estimator": "RandomForestRegressor",
            "hyperparameters_json": '{"n_estimators":400,"max_depth":null,"n_jobs":-1,"random_state":42}',
            "features_path": "data/features/weekly_clean.csv.gz",
            "data_sha256": "",
            "feature_count": None,
            "built_at_utc": m.get("built_at_utc", ""),
            "actual_column": m.get("actual_column", ""),
            "mae": m.get("mae", ""),
            "r2": m.get("r2", ""),
            "rmse": m.get("rmse", ""),
            "rows": m.get("rows", ""),
            "artifact_filename": "",
            "artifact_sha256": "",
        })

    # Write CSV
    cols = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator",
        "hyperparameters_json","features_path","data_sha256","feature_count",
        "built_at_utc","actual_column","mae","r2","rmse","rows",
        "artifact_filename","artifact_sha256"
    ]
    df = pd.DataFrame(rows, columns=cols)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    # Write lock json
    with open(LOCK_OUT, "w") as f:
        json.dump({"artifacts": lock_items}, f, indent=2)

    print("[INFO] Wrote manifest ->", CSV_OUT)
    print("[INFO] Wrote lock ->", LOCK_OUT)

if __name__ == "__main__":
    main()
