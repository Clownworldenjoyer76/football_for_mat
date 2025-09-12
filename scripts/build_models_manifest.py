#!/usr/bin/env python3
import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
MODELS_DIR = REPO / "models" / "pregame"
METRICS = REPO / "output" / "models" / "metrics_summary.csv"
FEATURES_PATH = REPO / "data" / "features" / "weekly_clean.csv.gz"
FEATURE_COLS_TXT = REPO / "output" / "logs" / "features_columns.txt"
MANIFEST_OUT = REPO / "models" / "manifest" / "models_manifest.csv"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def read_feature_count() -> int:
    if not FEATURE_COLS_TXT.exists():
        return ""
    try:
        with FEATURE_COLS_TXT.open() as f:
            return sum(1 for _ in f if _.strip())
    except Exception:
        return ""

def main():
    if not METRICS.exists():
        print(f"ERROR: missing metrics file: {METRICS}", file=sys.stderr)
        sys.exit(1)

    # Index metrics by artifact filename and sha256
    by_fname = {}
    by_sha = {}

    with METRICS.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = os.path.basename(row["artifact_path"])
            by_fname[fname] = row
            by_sha[row["artifact_sha256"]] = row

    # Compute data hash (optional)
    data_sha = ""
    if FEATURES_PATH.exists():
        try:
            data_sha = sha256_file(FEATURES_PATH)
        except Exception:
            pass

    feature_count = read_feature_count()

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator",
        "hyperparameters_json","features_path","data_sha256","feature_count",
        "built_at_utc",
    ]

    rows = []

    # Prefer artifacts listed in metrics (authoritative)
    for row in by_fname.values():
        fname = os.path.basename(row["artifact_path"])
        local_path = MODELS_DIR / fname
        if not local_path.exists():
            # if training wrote somewhere else, try to map anyway but skip if missing
            continue

        try:
            sha = sha256_file(local_path)
        except Exception:
            sha = ""

        stat = local_path.stat()
        rows.append({
            "path": str(local_path.relative_to(REPO)).replace("\\","/"),
            "filename": fname,
            "sha256": sha or row.get("artifact_sha256",""),
            "filesize_bytes": stat.st_size,
            "modified_time_utc": iso_utc(stat.st_mtime),
            "target": row.get("target",""),
            "version_tag": row.get("version_tag",""),
            "git_commit": row.get("git_commit",""),
            "random_seed": row.get("random_seed",""),
            "estimator": "",  # not tracked in metrics; leave blank
            "hyperparameters_json": "",  # not tracked; leave blank
            "features_path": str(FEATURES_PATH.relative_to(REPO)).replace("\\","/") if FEATURES_PATH.exists() else "",
            "data_sha256": data_sha,
            "feature_count": feature_count,
            "built_at_utc": row.get("built_at_utc",""),
        })

    # Write manifest
    with MANIFEST_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # keep stable order (by target then filename)
        for r in sorted(rows, key=lambda r: (r.get("target",""), r["filename"])):
            w.writerow(r)

    # Sanity check: at least one populated target
    if not any(r["target"] for r in rows):
        print("ERROR: manifest has no populated 'target' values.", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Wrote manifest -> {MANIFEST_OUT} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
