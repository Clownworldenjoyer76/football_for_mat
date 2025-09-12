#!/usr/bin/env python3
"""
Build models manifest from output/models/metrics_summary.csv only.

- Does NOT scan the repo for .joblib files
- Writes:
    models/manifest/models_manifest.csv
    models/manifest/models_manifest.lock.json
- Fails the job if no valid rows (so your validation step stays meaningful)
"""

from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "output/models/metrics_summary.csv"
OUT_DIR = ROOT / "models/manifest"
MANIFEST_CSV = OUT_DIR / "models_manifest.csv"
LOCK_JSON = OUT_DIR / "models_manifest.lock.json"

# Columns we’ll emit to the CSV (keep stable order)
CSV_COLUMNS = [
    "path", "filename", "sha256", "filesize_bytes", "modified_time_utc",
    "target", "version_tag", "git_commit", "random_seed",
    "estimator", "hyperparameters_json", "features_path", "data_sha256",
    "feature_count", "built_at_utc", "actual_column", "mae", "r2"
]

def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: metrics file missing: {path}", file=sys.stderr)
        sys.exit(1)
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize keys just in case
            r = {k.strip(): v for k, v in r.items()}
            # must have a target
            if not r.get("target"):
                continue
            rows.append(r)
    return rows

def coalesce(*vals):
    for v in vals:
        if v is None: 
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return ""

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(METRICS)

    if not metrics:
        print("ERROR: manifest has no populated 'target' values.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate by (target, version_tag) keeping latest built_at_utc if present
    def parse_dt(s: str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    dedup = {}
    for r in metrics:
        key = (r.get("target",""), r.get("version_tag",""))
        chosen = dedup.get(key)
        if not chosen or parse_dt(r.get("built_at_utc","")) > parse_dt(chosen.get("built_at_utc","")):
            dedup[key] = r
    rows = list(dedup.values())

    # Build CSV + lock entries
    csv_rows = []
    lock_rows = []

    for r in rows:
        artifact_path = r.get("artifact_path","")              # absolute in CI
        filename = Path(artifact_path).name if artifact_path else ""
        sha256 = r.get("artifact_sha256","")
        # We don’t know size/mtime for artifacts stored off-repo; leave blank
        csv_rows.append({
            "path": artifact_path,                             # keep full path (informational)
            "filename": filename,
            "sha256": sha256,
            "filesize_bytes": "",
            "modified_time_utc": "",
            "target": r.get("target",""),
            "version_tag": r.get("version_tag",""),
            "git_commit": r.get("git_commit",""),
            "random_seed": r.get("random_seed",""),
            "estimator": "",                                   # unknown in metrics_summary
            "hyperparameters_json": "",                        # unknown in metrics_summary
            "features_path": "",                               # unknown in metrics_summary
            "data_sha256": "",                                 # unknown in metrics_summary
            "feature_count": "",                               # unknown in metrics_summary
            "built_at_utc": r.get("built_at_utc",""),
            "actual_column": r.get("actual_column",""),
            "mae": r.get("mae",""),
            "r2": r.get("r2",""),
        })
        lock_rows.append({
            "filename": filename,
            "artifact_path": artifact_path,
            "sha256": sha256,
            "target": r.get("target",""),
            "version_tag": r.get("version_tag",""),
            "built_at_utc": r.get("built_at_utc",""),
            "git_commit": r.get("git_commit",""),
            # helpful hint to consumers that these are workflow artifacts, not repo files
            "source": "metrics_summary",
        })

    # Write CSV
    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write lock JSON
    with LOCK_JSON.open("w") as f:
        json.dump({"artifacts": lock_rows}, f, indent=2)

    print(f"[OK] Wrote manifest -> {MANIFEST_CSV}")
    print(f"[OK] Wrote lock     -> {LOCK_JSON}")
    # explicit success exit
    sys.exit(0)

if __name__ == "__main__":
    main()
