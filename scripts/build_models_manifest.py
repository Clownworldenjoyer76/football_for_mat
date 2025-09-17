#!/usr/bin/env python3
"""
Build a CSV + lock.json manifest of trained model artifacts.

Outputs:
  - models/manifest/models_manifest.csv
  - models/manifest/models_manifest.lock.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------
# Config
# -------------------------

MODELS_GLOB = "models/pregame/**/*.joblib"
MANIFEST_DIR = Path("models/manifest")
MANIFEST_CSV = MANIFEST_DIR / "models_manifest.csv"
LOCK_JSON = MANIFEST_DIR / "models_manifest.lock.json"
METRICS_JSON = Path("output/metrics_summary.json")

# If you later add more model families, extend here.
ACTUAL_COLUMN_MAP = {
    "qb_passing_yards": "passing_yards",
    "rb_rushing_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "wrte_receptions": "receptions",
}

# Columns to write in the CSV (kept compatible with your repo history)
CSV_COLUMNS = [
    "path",
    "filename",
    "sha256",
    "filesize_bytes",
    "modified_time_utc",
    "target",
    "version_tag",
    "git_commit",
    "random_seed",
    "estimator",
    "hyperparameters_json",
    "features_path",
    "data_sha256",
    "feature_count",
    "built_at_utc",
    "actual_column",
    "mae",
    "r2",
    "rmse",
    "rows",
    "artifact_filename",
    "artifact_sha256",
]

# -------------------------
# Utilities
# -------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

_VERSION_RE = re.compile(r"^(.*?)(?:_(\d{8}_[0-9a-f]{7,})|_[0-9a-f]{7,})$")

def infer_target_from_filename(filename: str) -> str:
    """
    Return the full target name from a model filename.
    Handles legacy 4 and version-suffixed artifacts:
      qb_passing_yards.joblib -> qb_passing_yards
      passing_yards_20250912_8eb60df.joblib -> passing_yards
    """
    stem = Path(filename).stem
    # Known legacy names: keep verbatim
    legacy = {"qb_passing_yards", "rb_rushing_yards", "wr_rec_yards", "wrte_receptions"}
    if stem in legacy:
        return stem
    m = _VERSION_RE.match(stem)
    return m.group(1) if m else stem

def infer_version_tag_from_filename(filename: str) -> str:
    """
    Extract a version suffix like 20250912_8eb60df or just the trailing hash if present.
    """
    stem = Path(filename).stem
    m = _VERSION_RE.match(stem)
    if not m:
        return ""
    # Prefer the combined date+hash group if present; else drop to plain suffix we already stripped
    return m.group(2) or stem.split("_")[-1]

def read_metrics_map(metrics_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict keyed by target with metrics like {"MAE": ..., "RMSE": ..., "rows": ...}
    If file is missing or invalid, returns empty dict.
    """
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open() as f:
            data = json.load(f)
        # Normalize keys to strings; values expected as dicts
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out
    except Exception:
        return {}

def get_git_commit() -> str:
    """
    Best-effort git commit resolution:
      - GITHUB_SHA in Actions environment
      - else 'git rev-parse --short HEAD' if available
      - else empty string
    """
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if sha:
        return sha[:7]
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return res.stdout.strip()
    except Exception:
        return ""

# -------------------------
# Data structures
# -------------------------

@dataclass
class ManifestRow:
    path: str
    filename: str
    sha256: str
    filesize_bytes: int
    modified_time_utc: str
    target: str
    version_tag: str
    git_commit: str
    random_seed: str
    estimator: str
    hyperparameters_json: str
    features_path: str
    data_sha256: str
    feature_count: str
    built_at_utc: str
    actual_column: str
    mae: Optional[float]
    r2: Optional[float]
    rmse: Optional[float]
    rows: Optional[int]
    artifact_filename: str
    artifact_sha256: str

    @classmethod
    def from_path(
        cls,
        p: Path,
        metrics: Dict[str, Dict[str, float]],
        git_commit: str,
    ) -> "ManifestRow":
        stat = p.stat()
        file_sha = sha256_file(p)
        filename = p.name
        target = infer_target_from_filename(filename)
        version_tag = infer_version_tag_from_filename(filename)
        mt = metrics.get(target, {})

        # Pull metrics fields (may be missing)
        mae = _get_float(mt, "MAE")
        rmse = _get_float(mt, "RMSE")
        rows = _get_int(mt, "rows")

        # r2 not present in metrics_summary.json (leave blank)
        r2 = _get_float(mt, "R2")  # harmless if absent

        return cls(
            path=str(p),
            filename=filename,
            sha256=file_sha,
            filesize_bytes=stat.st_size,
            modified_time_utc=iso_utc(stat.st_mtime),
            target=target,
            version_tag=version_tag,
            git_commit=git_commit,
            random_seed="",
            estimator="",
            hyperparameters_json="",
            features_path="",
            data_sha256="",
            feature_count="",
            built_at_utc=iso_utc(stat.st_mtime),
            actual_column=ACTUAL_COLUMN_MAP.get(target, ""),
            mae=mae,
            r2=r2,
            rmse=rmse,
            rows=rows,
            artifact_filename=filename,
            artifact_sha256=file_sha,
        )

def _get_float(d: Dict, key: str) -> Optional[float]:
    try:
        v = d.get(key)
        return float(v) if v is not None else None
    except Exception:
        return None

def _get_int(d: Dict, key: str) -> Optional[int]:
    try:
        v = d.get(key)
        return int(v) if v is not None else None
    except Exception:
        return None

# -------------------------
# Main
# -------------------------

def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics_map(METRICS_JSON)
    git_commit = get_git_commit()

    model_paths = sorted(Path().glob(MODELS_GLOB))
    rows: List[ManifestRow] = [ManifestRow.from_path(p, metrics, git_commit) for p in model_paths]

    # Write CSV
    with MANIFEST_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            # Ensure we only write the defined columns, in order
            d = asdict(r)
            w.writerow({k: d.get(k, "") for k in CSV_COLUMNS})

    # Write lock.json
    lock_payload = {
        "artifacts": [
            {
                "filename": r.filename,
                "path": r.path,
                "sha256": r.sha256,
                "target": r.target,
                "version_tag": r.version_tag or "",
            }
            for r in rows
        ]
    }
    with LOCK_JSON.open("w") as f:
        json.dump(lock_payload, f, indent=2)

    print(f"[OK] Wrote manifest -> {MANIFEST_CSV}")
    print(f"[OK] Wrote lock -> {LOCK_JSON}")
    print(f"[INFO] Models found: {len(rows)}")
    # Optional: summarize per-target counts
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.target] = counts.get(r.target, 0) + 1
    if counts:
        print("[INFO] Counts by target:")
        for k in sorted(counts):
            print(f"  - {k}: {counts[k]}")

if __name__ == "__main__":
    main()
