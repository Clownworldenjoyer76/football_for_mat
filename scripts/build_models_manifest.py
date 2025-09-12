#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

# --- helpers -----------------------------------------------------------------

def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_of_gzip(path: Path, chunk: int = 1024 * 1024) -> str:
    # Works the same for .gz files but ensures we hash the bytes on disk (not decompressed)
    return sha256_of_file(path, chunk)

def file_mtime_utc_iso(path: Path) -> str:
    return datetime.utcfromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")

def is_ymd(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", s))

def infer_target_and_version(fname: str) -> Tuple[str, str]:
    """
    Accepts:
      - passing_yards_20250912_cc4d012.joblib      -> ('passing_yards', '20250912_cc4d012')
      - qb_sacks_taken_20250912_cc4d012.joblib     -> ('qb_sacks_taken', '20250912_cc4d012')
      - receptions.latest.joblib                   -> ('receptions.latest', '')
      - qb_passing_yards.joblib (legacy)           -> ('qb_passing_yards', '')
    Rule: if the last 2 underscore groups look like YYYYMMDD and a short git hash, strip them.
    """
    base = fname.replace(".joblib", "")
    parts = base.split("_")
    if len(parts) >= 3 and is_ymd(parts[-2]) and len(parts[-1]) in (7, 8):
        target = "_".join(parts[:-2])
        version_tag = f"{parts[-2]}_{parts[-1]}"
        return target, version_tag
    return base, ""  # legacy or "latest" style

def load_metrics_map(metrics_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Build a map keyed by joblib filename -> row dict from metrics_summary.csv.
    We key on the filename to be robust to absolute paths in the CSV.
    """
    m: Dict[str, Dict[str, str]] = {}
    if not metrics_path.exists():
        return m
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            apath = row.get("artifact_path", "") or ""
            fname = Path(apath).name if apath else ""
            if fname:
                m[fname] = row
    return m

def count_features(features_path: Path) -> int:
    """
    Count columns in the features file (supports .csv and .csv.gz).
    We don’t load the whole dataset; just read the header.
    """
    if not features_path.exists():
        return 0
    opener = gzip.open if features_path.suffix == ".gz" else open
    with opener(features_path, "rt", encoding="utf-8", newline="") as f:
        header = f.readline().strip()
        # handle empty files safely
        if not header:
            return 0
        reader = csv.reader([header])
        cols = next(reader, [])
        return len(cols)

def safe_json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build models manifest with rich metadata.")
    parser.add_argument("--models-dir", default="models/pregame", help="Directory with .joblib models")
    parser.add_argument("--features", default="data/features/weekly_clean.csv.gz", help="Path to features CSV(.gz)")
    parser.add_argument("--metrics", default="output/models/metrics_summary.csv", help="Path to metrics summary CSV")
    parser.add_argument("--out-csv", default="models/manifest/models_manifest.csv", help="Manifest CSV path")
    parser.add_argument("--out-lock", default="models/manifest/models_manifest.lock.json", help="Lock JSON path")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    features_path = Path(args.features)
    metrics_path = Path(args.metrics)
    out_csv = Path(args.out_csv)
    out_lock = Path(args.out_lock)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_map = load_metrics_map(metrics_path)
    feature_count = count_features(features_path)
    features_sha = sha256_of_gzip(features_path) if features_path.exists() else ""
    built_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    lock = {
        "repo_root": Path(".").resolve().name,
        "built_at_utc": built_at,
        "artifacts": {}
    }

    # Walk models dir for .joblib
    joblibs = sorted(models_dir.glob("*.joblib"))
    for p in joblibs:
        fname = p.name
        fsize = p.stat().st_size
        fsha = sha256_of_file(p)
        mtime_utc = file_mtime_utc_iso(p)

        target, version_tag = infer_target_and_version(fname)

        # Defaults; fill from metrics when available
        git_commit = ""
        random_seed = ""
        estimator_name = ""
        hyperparams_json = ""
        built_at_utc = ""
        actual_column = ""
        # Try to hydrate from metrics (preferred)
        mrow = metrics_map.get(fname)
        if mrow:
            target_from_metrics = mrow.get("target", "") or ""
            # "target" in metrics is the logical label (e.g., passing_yards), so prefer it.
            if target_from_metrics:
                target = target_from_metrics
            git_commit = (mrow.get("git_commit") or "").strip()
            random_seed = (mrow.get("random_seed") or "").strip()
            built_at_utc = (mrow.get("built_at_utc") or "").strip()
            actual_column = (mrow.get("actual_column") or "").strip()
            # version_tag in metrics is authoritative if present
            version_tag = (mrow.get("version_tag") or version_tag).strip()

        # Load model to capture estimator + params (best-effort)
        try:
            import joblib  # local import to avoid cost when unused
            model = joblib.load(p)
            est = getattr(model, "estimator", model)  # if you saved a wrapper, fall back to obj
            estimator_name = est.__class__.__name__
            # Try scikit-learn get_params, otherwise fallback to __dict__
            if hasattr(est, "get_params"):
                hyperparams_json = safe_json_dumps(est.get_params(deep=False))
            else:
                hyperparams_json = safe_json_dumps({k: v for k, v in est.__dict__.items() if not k.startswith("_")})
        except Exception as e:
            # Keep going; leave estimator/hyperparams empty if load fails
            estimator_name = estimator_name or ""
            hyperparams_json = hyperparams_json or ""

        # Build CSV row
        rows.append({
            "path": str(p.resolve()),
            "filename": fname,
            "sha256": fsha,
            "filesize_bytes": str(fsize),
            "modified_time_utc": mtime_utc,
            "target": target,
            "version_tag": version_tag,
            "git_commit": git_commit,
            "random_seed": random_seed,
            "estimator": estimator_name,
            "hyperparameters_json": hyperparams_json,
            "features_path": str(features_path.resolve()) if features_path.exists() else "",
            "data_sha256": features_sha,
            "feature_count": str(feature_count),
            "built_at_utc": built_at_utc or built_at,
        })

        # Lock entry
        rel_key = str(p.as_posix())
        lock["artifacts"][rel_key] = fsha

    # Write CSV
    fieldnames = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed",
        "estimator","hyperparameters_json",
        "features_path","data_sha256","feature_count","built_at_utc"
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write lock
    with open(out_lock, "w") as f:
        json.dump(lock, f, indent=2)

    # Basic stdout for the workflow logs
    print(f"[OK] Wrote manifest -> {out_csv} ({len(rows)} rows)")
    print(f"[OK] Wrote lock -> {out_lock}")

if __name__ == "__main__":
    sys.exit(main())
