#!/usr/bin/env python3
"""
Build a models manifest that ALWAYS has a populated `target` column.

- Reads output/models/metrics_summary.csv (ground truth for `target`, metrics, version_tag, etc.)
- Scans models/**/*.joblib for current artifacts
- Joins by filename (or artifact_path endswith(filename))
- Writes:
    models/manifest/models_manifest.csv
    models/manifest/models_manifest.lock.json
"""

from __future__ import annotations
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import re
import sys
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_DIR = MODELS_DIR / "manifest"
MANIFEST_CSV = MANIFEST_DIR / "models_manifest.csv"
LOCK_JSON = MANIFEST_DIR / "models_manifest.lock.json"

# Primary source of truth for targets/metrics
METRICS_CSV_CANDIDATES = [
    REPO_ROOT / "output" / "models" / "metrics_summary.csv",
    # Fallbacks (keep in case layouts change)
    REPO_ROOT / "football_for_mat" / "output" / "models" / "metrics_summary.csv",
]

def sha256_file(p: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def find_metrics_csv() -> Optional[Path]:
    for cand in METRICS_CSV_CANDIDATES:
        if cand.exists():
            return cand
    return None

def infer_target_from_filename(name: str) -> Optional[str]:
    # Accept names like: passing_yards_20250912_cc4d012.joblib  -> passing_yards
    # or sacks.joblib -> sacks
    base = name
    if base.endswith(".joblib"):
        base = base[:-7]
    # split at last _YYYYMMDD_*
    m = re.match(r"^([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_(\d{8})_[0-9a-fA-F]+$", base)
    if m:
        return m.group(1)
    return base or None

def main() -> int:
    print("[build_models_manifest] starting...")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = find_metrics_csv()
    if metrics_path is None:
        print("WARN: metrics_summary.csv not found; targets will be inferred from filenames.", file=sys.stderr)
        metrics_df = pd.DataFrame()
    else:
        metrics_df = pd.read_csv(metrics_path)
        # Normalize filename key for joining
        metrics_df["__filename"] = metrics_df["artifact_path"].apply(lambda s: Path(str(s)).name if pd.notna(s) else "")
        metrics_df["__filename_lower"] = metrics_df["__filename"].str.lower()

    # Collect joblibs
    joblibs = [Path(p) for p in glob.glob(str(MODELS_DIR / "**" / "*.joblib"), recursive=True)]
    rows = []
    lock_entries = []

    git_sha = os.environ.get("GITHUB_SHA", "")
    git_short = git_sha[:7] if git_sha else ""

    for p in sorted(joblibs):
        rel = p.relative_to(REPO_ROOT)
        fname = p.name
        fname_lower = fname.lower()
        filesize = p.stat().st_size
        modified_utc = iso_utc(p.stat().st_mtime)
        file_sha = sha256_file(p)

        # Try join to metrics by filename (case-insensitive)
        metrics_row = None
        if not metrics_df.empty:
            m = metrics_df.loc[metrics_df["__filename_lower"] == fname_lower]
            if len(m) == 0:
                # Second chance: metrics artifact_path endswith our relative path
                m = metrics_df[metrics_df["artifact_path"].astype(str).str.endswith(fname, na=False)]
            if len(m) > 0:
                metrics_row = m.iloc[0].to_dict()

        # Populate fields, preferring metrics when present
        target = None
        version_tag = None
        actual_column = None
        mae = None
        r2 = None
        built_at_utc = None
        random_seed = None
        artifact_git = None

        if metrics_row:
            target = metrics_row.get("target") or target
            version_tag = metrics_row.get("version_tag") or version_tag
            actual_column = metrics_row.get("actual_column") or actual_column
            mae = metrics_row.get("mae")
            r2 = metrics_row.get("r2")
            built_at_utc = metrics_row.get("built_at_utc")
            random_seed = metrics_row.get("random_seed")
            artifact_git = metrics_row.get("git_commit") or git_short

        # Fallbacks if metrics missing
        if not target:
            target = infer_target_from_filename(fname)

        if not version_tag:
            # Try extract suffix like *_YYYYMMDD_<hash>
            m = re.match(r".*_(\d{8}_[0-9a-fA-F]+)\.joblib$", fname)
            if m:
                version_tag = m.group(1)
            else:
                version_tag = ""

        row = {
            "path": str(rel).replace("\\", "/"),
            "filename": fname,
            "sha256": file_sha,
            "filesize_bytes": filesize,
            "modified_time_utc": modified_utc,
            "target": target or "",                            # must be populated
            "version_tag": version_tag,
            "git_commit": artifact_git or git_short,
            "random_seed": random_seed if pd.notna(random_seed) else "",
            "estimator": "",                                   # optional (not available here)
            "hyperparameters_json": "tuple",                   # keep shape consistent with prior CSVs
            "features_path": str(metrics_row.get("features_path")) if metrics_row and pd.notna(metrics_row.get("features_path")) else "",
            "data_sha256": str(metrics_row.get("data_sha256")) if metrics_row and pd.notna(metrics_row.get("data_sha256")) else "",
            "feature_count": int(metrics_row.get("feature_count")) if metrics_row and pd.notna(metrics_row.get("feature_count")) else "",
            "built_at_utc": built_at_utc or modified_utc,
            # Nice-to-have columns, if consumers ever need them:
            "actual_column": actual_column or "",
            "mae": mae if (mae is not None and pd.notna(mae)) else "",
            "r2": r2 if (r2 is not None and pd.notna(r2)) else "",
        }
        rows.append(row)

        lock_entries.append({
            "filename": fname,
            "path": row["path"],
            "sha256": file_sha,
            "target": row["target"],
            "version_tag": version_tag,
        })

    df = pd.DataFrame(rows)

    # Hard guard: fail if we somehow created empty/blank targets
    if df.empty or df["target"].isna().any() or (df["target"].astype(str).str.strip() == "").any():
        print("ERROR: manifest has no populated 'target' values.", file=sys.stderr)
        # Show a quick debug sample
        print(df[["filename", "target"]].head().to_string(index=False))
        return 1

    # Write outputs
    df.sort_values(["target", "filename"], inplace=True)
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    with LOCK_JSON.open("w") as f:
        json.dump({"artifacts": lock_entries}, f, indent=2)

    print(f"[build_models_manifest] wrote {len(df)} rows -> {MANIFEST_CSV}")
    print(f"[build_models_manifest] wrote lock -> {LOCK_JSON}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
