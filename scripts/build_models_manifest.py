#!/usr/bin/env python3
# /scripts/build_models_manifest.py
"""
Build a reproducible manifest for trained model artifacts.

Outputs:
  - models/manifest/models_manifest.csv
  - models/manifest/models_manifest.lock.json
"""

from __future__ import annotations
import csv, json, hashlib, os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models" / "pregame"
MANIFEST_DIR = REPO / "models" / "manifest"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = MANIFEST_DIR / "models_manifest.csv"
MANIFEST_LOCK = MANIFEST_DIR / "models_manifest.lock.json"

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

def load_sidecar(joblib_path: Path) -> dict:
    # infer meta path beside artifact: <target>_<tag>.meta.json
    meta = joblib_path.with_suffix(".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except Exception:
            return {}
    return {}

def build_manifest():
    rows = []
    lock = {}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        if p.name.endswith(".latest.joblib"):
            # Skip 'latest' pointers; manifest should track versioned artifacts
            continue
        stat = p.stat()
        sha = sha256_file(p)
        meta = load_sidecar(p)

        row = {
            "path": str(p),
            "filename": p.name,
            "sha256": sha,
            "filesize_bytes": stat.st_size,
            "modified_time_utc": datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # From sidecar (if available)
            "target": meta.get("target"),
            "version_tag": meta.get("version_tag"),
            "git_commit": meta.get("git_commit"),
            "random_seed": meta.get("random_seed"),
            "estimator": meta.get("estimator"),
            "hyperparameters_json": json.dumps(meta.get("hyperparameters")) if meta.get("hyperparameters") else None,
            "features_path": meta.get("features_path"),
            "data_sha256": meta.get("data_sha256"),
            "feature_count": len(meta.get("feature_columns", [])) if meta.get("feature_columns") else None,
            "built_at_utc": meta.get("built_at_utc"),
        }
        rows.append(row)
        lock[str(p.relative_to(REPO))] = sha

    # Write CSV
    headers = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator",
        "hyperparameters_json","features_path","data_sha256","feature_count","built_at_utc"
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write lock JSON
    with MANIFEST_LOCK.open("w", encoding="utf-8") as f:
        json.dump({
            "repo_root": REPO.name,
            "built_at_utc": utc_now_iso(),
            "artifacts": lock,
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote CSV: {MANIFEST_CSV} ({len(rows)} artifacts)")
    print(f"Wrote JSON: {MANIFEST_LOCK}")
    return rows, lock

if __name__ == "__main__":
    build_manifest()
