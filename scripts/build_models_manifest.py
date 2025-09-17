#!/usr/bin/env python3
import csv, hashlib, json, os, sys, gzip, io
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
MODELS_DIR = Path("models/pregame")
MANIFEST_DIR = Path("models/manifest")
MANIFEST_CSV = MANIFEST_DIR / "models_manifest.csv"
LOCK_JSON = MANIFEST_DIR / "models_manifest.lock.json"

METRICS_CSV = Path("output/models/metrics_summary.csv")
METRICS_JSON = Path("output/metrics_summary.json")  # optional/legacy

# Legacy four targets → their actual stat column names in metrics JSONs
LEGACY_ACTUAL_MAP = {
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "wrte_receptions": "receptions",
}

# infer more targets from filenames like: passing_yards_20250912_ABC123.joblib
def infer_target_from_filename(name: str) -> str | None:
    stem = name.replace(".joblib", "")
    # common pattern: target_<tag>.joblib
    parts = stem.split("_")
    if len(parts) >= 2:
        # try longest leading slice that matches a known stat-y word
        # fallback to first part
        return parts[0]
    return stem

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def utc_iso(dt: float | datetime) -> str:
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()

def read_metrics_csv() -> dict:
    """Return dict keyed by target with columns: mae,r2,artifact_path,artifact_sha256,version_tag,actual_column,rmse,rows."""
    out = {}
    if not METRICS_CSV.exists():
        return out
    with open(METRICS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tgt = r.get("target", "").strip()
            if not tgt:
                continue
            out[tgt] = {
                "actual_column": r.get("actual_column") or "",
                "artifact_path": r.get("artifact_path") or "",
                "artifact_sha256": r.get("artifact_sha256") or "",
                "git_commit": r.get("git_commit") or "",
                "random_seed": r.get("random_seed") or "",
                "built_at_utc": r.get("built_at_utc") or "",
                "mae": r.get("mae") or "",
                "r2": r.get("r2") or "",
                "rmse": r.get("rmse") or "",  # some CSVs may not have it
                "rows": r.get("rows") or "",
                "version_tag": r.get("version_tag") or "",
            }
    return out

def read_metrics_json() -> dict:
    """Also support output/metrics_summary.json (your 4-legacy-metrics file)."""
    out = {}
    if not METRICS_JSON.exists():
        return out
    with open(METRICS_JSON, encoding="utf-8") as f:
        blob = json.load(f)
    # format:
    # {
    #   "qb_passing_yards": {"status":"ok","MAE":43.2669,"RMSE":62.2819,"rows":1315},
    #   ...
    # }
    for legacy_key, vals in blob.items():
        if not isinstance(vals, dict):
            continue
        actual_col = LEGACY_ACTUAL_MAP.get(legacy_key, legacy_key)
        out[legacy_key] = {
            "actual_column": actual_col,
            "artifact_path": "",
            "artifact_sha256": "",
            "git_commit": "",
            "random_seed": "",
            "built_at_utc": "",
            "mae": vals.get("MAE", ""),
            "r2": "",  # JSON didn’t include r2
            "rmse": vals.get("RMSE", ""),
            "rows": vals.get("rows", ""),
            "version_tag": "",
        }
    return out

def merge_metrics() -> dict:
    # CSV (rich) wins; JSON fills gaps and adds the 4 legacy targets if CSV lacks them
    csvm = read_metrics_csv()
    jsonm = read_metrics_json()
    merged = dict(jsonm)
    merged.update(csvm)
    return merged

def read_git_commit() -> str:
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return ""

def ensure_dirs():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

def list_local_joblibs() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.joblib"))

def version_from_name(name: str) -> str:
    # try to pull trailing _<tag> from filename
    base = name.replace(".joblib", "")
    if "_" in base:
        tag = base.split("_")[-1]
        return tag
    return ""

def main():
    ensure_dirs()
    git_commit = read_git_commit()
    metrics = merge_metrics()

    rows = []
    lock = {"artifacts": []}

    seen_targets: set[str] = set()

    # 1) real artifacts present locally
    for p in list_local_joblibs():
        sha = sha256_file(p)
        stat = p.stat()
        modified_iso = utc_iso(stat.st_mtime)

        target = infer_target_from_filename(p.name)
        # map the four legacy filenames back to their target keys if needed
        if p.name in LEGACY_ACTUAL_MAP:
            target = p.name.replace(".joblib", "")

        m = metrics.get(target, {})
        actual_column = m.get("actual_column", "")
        # If this is one of the 4 legacy targets and actual_column is empty, fill from map.
        if not actual_column and target in LEGACY_ACTUAL_MAP:
            actual_column = LEGACY_ACTUAL_MAP[target]

        row = {
            "path": str(p),
            "filename": p.name,
            "sha256": sha,
            "filesize_bytes": stat.st_size,
            "modified_time_utc": modified_iso,
            "target": target,
            "version_tag": m.get("version_tag", "") or version_from_name(p.name),
            "git_commit": m.get("git_commit", "") or git_commit,
            "random_seed": m.get("random_seed", ""),
            "estimator": "",  # not stored in artifact; leave blank unless you serialize it
            "hyperparameters_json": "",
            "features_path": "",
            "data_sha256": "",
            "feature_count": "",
            "built_at_utc": m.get("built_at_utc", "") or modified_iso,
            "actual_column": actual_column,
            "mae": m.get("mae", ""),
            "r2": m.get("r2", ""),
            "rmse": m.get("rmse", ""),
            "rows": m.get("rows", ""),
            "artifact_filename": p.name,
            "artifact_sha256": sha,
        }
        rows.append(row)
        seen_targets.add(target)

        lock["artifacts"].append({
            "filename": p.name,
            "path": str(p),
            "sha256": sha,
            "target": target,
            "version_tag": row["version_tag"],
        })

    # 2) metrics-only rows (no local artifact) → include so the manifest shows >4
    for tgt, m in metrics.items():
        if tgt in seen_targets:
            continue
        # synthesize a row with missing artifact info
        row = {
            "path": "",
            "filename": "",
            "sha256": "",
            "filesize_bytes": "",
            "modified_time_utc": utc_iso(datetime.now(timezone.utc)),
            "target": tgt,
            "version_tag": m.get("version_tag", ""),
            "git_commit": m.get("git_commit", "") or git_commit,
            "random_seed": m.get("random_seed", ""),
            "estimator": "",
            "hyperparameters_json": "",
            "features_path": "",
            "data_sha256": "",
            "feature_count": "",
            "built_at_utc": m.get("built_at_utc", ""),
            "actual_column": m.get("actual_column", ""),
            "mae": m.get("mae", ""),
            "r2": m.get("r2", ""),
            "rmse": m.get("rmse", ""),
            "rows": m.get("rows", ""),
            "artifact_filename": "",
            "artifact_sha256": "",
        }
        rows.append(row)

    # 3) write CSV
    fieldnames = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator",
        "hyperparameters_json","features_path","data_sha256","feature_count",
        "built_at_utc","actual_column","mae","r2","rmse","rows",
        "artifact_filename","artifact_sha256",
    ]
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 4) write lock JSON
    with open(LOCK_JSON, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)

    print(f"[OK] Wrote {MANIFEST_CSV} with {len(rows)} rows")
    print(f"[OK] Wrote {LOCK_JSON} with {len(lock['artifacts'])} artifacts")

if __name__ == "__main__":
    sys.exit(main())
