#!/usr/bin/env python3
"""
Build a reproducible manifest for trained model artifacts.

Outputs:
  - football_for_mat-main/models/manifest/models_manifest.csv
  - football_for_mat-main/models/manifest/models_manifest.lock.json
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator, Dict, Any, List, Tuple
import subprocess

# --- paths (derived from this script location; no assumptions beyond repo layout) ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # football_for_mat-main/
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_DIR = MODELS_DIR / "manifest"
MANIFEST_CSV = MANIFEST_DIR / "models_manifest.csv"
MANIFEST_LOCK = MANIFEST_DIR / "models_manifest.lock.json"

# --- helpers ---
def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def iter_joblibs(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for p in root.rglob("*.joblib"):
        if p.is_file():
            yield p

def git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""

def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

# --- main ---
def build_manifest() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    commit = git_commit(REPO_ROOT)
    built_at = datetime.now(timezone.utc).isoformat()

    rows: List[Dict[str, Any]] = []
    lock: Dict[str, str] = {}

    for p in sorted(iter_joblibs(MODELS_DIR)):
        rel = p.relative_to(REPO_ROOT).as_posix()
        stat = p.stat()
        h = sha256_file(p)

        row = {
            "path": rel,
            "filename": p.name,
            "sha256": h,
            "filesize_bytes": stat.st_size,
            "modified_time_utc": utc_iso(stat.st_mtime),
            "git_commit": commit,
            "built_at_utc": built_at,
        }
        rows.append(row)
        lock[rel] = h

    # write CSV
    headers = [
        "path",
        "filename",
        "sha256",
        "filesize_bytes",
        "modified_time_utc",
        "git_commit",
        "built_at_utc",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write JSON lock
    with MANIFEST_LOCK.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_root": REPO_ROOT.name,
                "git_commit": commit,
                "built_at_utc": built_at,
                "artifacts": lock,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return rows, lock

if __name__ == "__main__":
    rows, lock = build_manifest()
    print(f"Wrote CSV: {MANIFEST_CSV} ({len(rows)} artifacts)")
    print(f"Wrote JSON: {MANIFEST_LOCK}")
