#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import joblib

# --- Helpers -------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return ""

def get_version_tag() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d")
    short_commit = get_git_commit()[:7]
    return f"{ts}_{short_commit}" if short_commit else ts

def sizeof_file(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

def get_modified_time(path: Path) -> str:
    return datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z"

# --- Main ----------------------------------------------------------------

def build_manifest(models_dir: Path, features_file: Path, output_csv: Path, output_json: Path):
    rows = []
    version_tag = get_version_tag()
    git_commit = get_git_commit()
    built_at = datetime.utcnow().isoformat() + "Z"

    features_sha = sha256_of_file(features_file) if features_file.exists() else ""
    feature_count = 0
    if features_file.suffix.endswith("gz"):
        import pandas as pd
        try:
            df = pd.read_csv(features_file)
            feature_count = len(df.columns)
        except Exception:
            pass

    for path in models_dir.glob("*.joblib"):
        sha = sha256_of_file(path)
        row = {
            "path": str(path),
            "filename": path.name,
            "sha256": sha,
            "filesize_bytes": sizeof_file(path),
            "modified_time_utc": get_modified_time(path),
            "target": "",
            "version_tag": version_tag,
            "git_commit": git_commit,
            "random_seed": "",
            "estimator": "",
            "hyperparameters_json": "",
            "features_path": str(features_file) if features_file.exists() else "",
            "data_sha256": features_sha,
            "feature_count": feature_count,
            "built_at_utc": built_at,
        }

        # try to extract metadata from joblib
        try:
            model_obj = joblib.load(path)
            if hasattr(model_obj, "target"):
                row["target"] = model_obj.target
            if hasattr(model_obj, "random_state"):
                row["random_seed"] = getattr(model_obj, "random_state")
            row["estimator"] = type(model_obj).__name__
            if hasattr(model_obj, "get_params"):
                row["hyperparameters_json"] = json.dumps(model_obj.get_params())
        except Exception as e:
            print(f"[WARN] could not load {path}: {e}")

        rows.append(row)

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # write lock JSON
    lock = {
        "repo_root": str(Path(".").resolve().name),
        "built_at_utc": built_at,
        "artifacts": {r["path"]: r["sha256"] for r in rows},
    }
    with open(output_json, "w") as f:
        json.dump(lock, f, indent=2)

    print(f"[OK] wrote manifest -> {output_csv}")
    print(f"[OK] wrote lock -> {output_json}")

# --- Entrypoint ----------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models/pregame")
    ap.add_argument("--features-file", default="data/features/weekly_clean.csv.gz")
    ap.add_argument("--output-csv", default="models/manifest/models_manifest.csv")
    ap.add_argument("--output-json", default="models/manifest/models_manifest.lock.json")
    args = ap.parse_args()

    build_manifest(Path(args.models_dir), Path(args.features_file), Path(args.output_csv), Path(args.output_json))
