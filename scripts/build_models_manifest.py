#!/usr/bin/env python3
import hashlib, json, os, sys, csv, glob
from datetime import datetime, timezone

# ---- config
MODELS_DIR = "models/pregame"
OUT_DIR = "models/manifest"
CSV_PATH = os.path.join(OUT_DIR, "models_manifest.csv")
LOCK_PATH = os.path.join(OUT_DIR, "models_manifest.lock.json")
METRICS_JSON = "output/metrics_summary.json"  # source of MAE/RMSE/rows

os.makedirs(OUT_DIR, exist_ok=True)

def sha256_file(path, buf_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def parse_tags_from_filename(fname: str):
    """
    Expected patterns we’ve seen:
      passing_yards_YYYYMMDD_<git>.joblib  -> target, version_tag, git_commit
      qb_passing_yards.joblib              -> target only
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    target = parts[0]
    version_tag = ""
    git_commit = ""
    if len(parts) >= 3 and parts[-1] and len(parts[-1]) >= 6 and parts[-2].isdigit():
        # e.g. passing_yards_20250912_8eb60df
        git_commit = parts[-1]
        version_tag = f"{parts[-2]}_{parts[-1]}"
        target = "_".join(parts[:-2])  # everything before date+commit
    return target, version_tag, git_commit

def load_metrics():
    """
    metrics_summary.json structure:
    {
      "qb_passing_yards": {"status":"ok","MAE":..., "RMSE":..., "rows":...},
      ...
    }
    """
    if not os.path.exists(METRICS_JSON):
        return {}
    try:
        with open(METRICS_JSON, "r") as f:
            raw = json.load(f)
        # normalize keys to lower for leniency
        out = {}
        for k, v in raw.items():
            out[k] = {
                "mae": v.get("MAE"),
                "rmse": v.get("RMSE"),
                "rows": v.get("rows"),
            }
        return out
    except Exception as e:
        print(f"[WARN] Could not read {METRICS_JSON}: {e}", file=sys.stderr)
        return {}

def main():
    metrics = load_metrics()

    rows_csv = []
    lock_items = []

    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.joblib"))):
        fname = os.path.basename(path)
        sha = sha256_file(path)
        stat = os.stat(path)
        size = stat.st_size
        mtime_iso = iso_utc(stat.st_mtime)

        target, version_tag, git_commit = parse_tags_from_filename(fname)

        # attach metrics if available
        m = metrics.get(target, {})
        mae = m.get("mae")
        rmse = m.get("rmse")
        rows = m.get("rows")

        # CSV row (keeps existing columns + a couple handy extras)
        rows_csv.append({
            "path": path,
            "filename": fname,
            "sha256": sha,
            "filesize_bytes": size,
            "modified_time_utc": mtime_iso,
            "target": target,
            "version_tag": version_tag,
            "git_commit": git_commit,
            "random_seed": "",
            "estimator": "",
            "hyperparameters_json": "",
            "features_path": "",
            "data_sha256": "",
            "feature_count": "",
            "built_at_utc": mtime_iso,
            "actual_column": target if "_" in target else "",  # leave blank if generic
            "mae": mae if mae is not None else "",
            "r2": "",  # not in metrics_summary.json
            "rmse": rmse if rmse is not None else "",
            "rows": rows if rows is not None else "",
            "artifact_filename": fname,
            "artifact_sha256": sha,
        })

        # LOCK item (now enriched)
        lock_items.append({
            "filename": fname,
            "path": path,
            "sha256": sha,
            "filesize_bytes": size,
            "modified_time_utc": mtime_iso,
            "target": target,
            "version_tag": version_tag,
            "git_commit": git_commit,
            # metrics
            "mae": mae,
            "rmse": rmse,
            "rows": rows,
        })

    # write CSV
    fieldnames = [
        "path","filename","sha256","filesize_bytes","modified_time_utc",
        "target","version_tag","git_commit","random_seed","estimator",
        "hyperparameters_json","features_path","data_sha256","feature_count",
        "built_at_utc","actual_column","mae","r2","rmse","rows",
        "artifact_filename","artifact_sha256"
    ]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    # write lock.json (pretty + stable ordering)
    with open(LOCK_PATH, "w") as f:
        json.dump({"artifacts": lock_items}, f, indent=2, sort_keys=False)

    print(f"[OK] Wrote CSV -> {CSV_PATH}")
    print(f"[OK] Wrote LOCK -> {LOCK_PATH}")
    print(f"[INFO] Artifacts counted: {len(lock_items)}")

if __name__ == "__main__":
    main()
