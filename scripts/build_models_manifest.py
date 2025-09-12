#!/usr/bin/env python3
"""
Builds a models manifest from files under models/pregame and (optionally)
enriches with metrics from output/models/metrics_summary.csv.

Outputs:
- models/manifest/models_manifest.csv
- models/manifest/models_manifest.json
- models/manifest/models_manifest.lock.json  (stable, sorted, deduped)
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root (…/football_for_mat)
MODELS_DIR = ROOT / "models" / "pregame"
OUT_DIR = ROOT / "models" / "manifest"
METRICS_CSV = ROOT / "output" / "models" / "metrics_summary.csv"

OUT_CSV = OUT_DIR / "models_manifest.csv"
OUT_JSON = OUT_DIR / "models_manifest.json"
OUT_LOCK = OUT_DIR / "models_manifest.lock.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def git_short_sha() -> str:
    # Prefer CI-provided env var; fallback to git command; else blank
    env = os.getenv("GITHUB_SHA")
    if env:
        return env[:7]
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        return out
    except Exception:
        return ""


FN_PATTERN = re.compile(r"^(?P<target>.+?)_(?P<version_tag>[0-9]{8}_[0-9a-f]{7,})\.joblib$")


def parse_name(filename: str):
    """
    Returns (target, version_tag). If pattern doesn't match, falls back to (stem, "").
    """
    m = FN_PATTERN.match(filename)
    if m:
        return m.group("target"), m.group("version_tag")
    # Fallback: use stem as 'target'-ish, blank version
    return Path(filename).stem, ""


def collect_model_files():
    """
    Yields dict rows with file metadata for every .joblib under models/pregame
    """
    rows = []
    if not MODELS_DIR.exists():
        return rows
    for p in sorted(MODELS_DIR.rglob("*.joblib")):
        rel = p.relative_to(ROOT).as_posix()
        target, version_tag = parse_name(p.name)

        try:
            stat = p.stat()
            filesize = stat.st_size
            mtime_iso = iso_utc(stat.st_mtime)
        except FileNotFoundError:
            # In case of a transient delete
            continue

        rows.append(
            {
                "path": rel,
                "filename": p.name,
                "sha256": sha256_file(p),
                "filesize_bytes": filesize,
                "modified_time_utc": mtime_iso,
                "target": target,
                "version_tag": version_tag,
            }
        )
    return rows


def read_metrics():
    """
    Returns metrics dataframe or empty df.
    Expected columns in metrics_summary.csv:
      target, actual_column, artifact_path, artifact_sha256, git_commit, random_seed,
      built_at_utc, mae, r2, version_tag
    """
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(METRICS_CSV)
    except Exception:
        return pd.DataFrame()

    # Normalize keys we’ll merge on
    # Use filename + sha256 + version_tag for robust join when absolute paths differ.
    df["artifact_filename"] = df["artifact_path"].fillna("").apply(lambda s: Path(s).name if s else "")
    for col in ("artifact_sha256", "version_tag", "target", "actual_column", "mae", "r2", "git_commit", "random_seed", "built_at_utc"):
        if col not in df.columns:
            df[col] = ""
    return df[
        [
            "artifact_filename",
            "artifact_sha256",
            "version_tag",
            "target",
            "actual_column",
            "mae",
            "r2",
            "git_commit",
            "random_seed",
            "built_at_utc",
        ]
    ]


def stable_sort_columns(df: pd.DataFrame, desired_order: list) -> pd.DataFrame:
    for col in desired_order:
        if col not in df.columns:
            df[col] = ""
    # Keep extras but put them after
    ordered = [c for c in desired_order if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def write_if_changed(path: Path, content: bytes) -> bool:
    """
    Write only if bytes differ. Returns True if file written/updated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as f:
            if f.read() == content:
                return False
    with path.open("wb") as f:
        f.write(content)
    return True


def main():
    rows = collect_model_files()
    if not rows:
        print("[WARN] No .joblib artifacts found under models/pregame")
    df = pd.DataFrame(rows)

    # Merge metrics if available
    mdf = read_metrics()
    if not df.empty and not mdf.empty:
        # Merge on filename + sha256 (most robust); if sha256 missing in metrics, fallback on filename + version_tag
        left = df.copy()
        left["artifact_filename"] = left["filename"]
        merged = left.merge(
            mdf,
            how="left",
            on="artifact_filename",
            suffixes=("", "_m"),
        )

        # Prefer exact sha256 match where available
        def pick(row, col):
            base = row.get(col, "")
            met = row.get(col + "_m", "")
            if col in ("target", "version_tag"):  # prefer file-derived for these
                return base or met
            # prefer metrics value when present
            return met or base

        merged["git_commit"] = merged.apply(lambda r: pick(r, "git_commit"), axis=1)
        merged["random_seed"] = merged.apply(lambda r: pick(r, "random_seed"), axis=1)
        merged["built_at_utc"] = merged.apply(lambda r: pick(r, "built_at_utc"), axis=1)
        merged["actual_column"] = merged.apply(lambda r: pick(r, "actual_column"), axis=1)
        merged["mae"] = merged.apply(lambda r: pick(r, "mae"), axis=1)
        merged["r2"] = merged.apply(lambda r: pick(r, "r2"), axis=1)

        # If version_tag is blank in file name but present in metrics, keep metrics
        merged["version_tag"] = merged.apply(
            lambda r: r["version_tag"] or r.get("version_tag_m", "") or "", axis=1
        )

        df = merged.drop(columns=[c for c in merged.columns if c.endswith("_m")])
    else:
        # fill metric columns with blanks to keep header consistent
        for c in ("git_commit", "random_seed", "built_at_utc", "actual_column", "mae", "r2"):
            df[c] = ""

    # If git_commit still blank, fill with current short sha (best effort)
    short_sha = git_short_sha()
    if short_sha:
        df["git_commit"] = df["git_commit"].replace("", short_sha)

    # Stable ordering
    df["__sort_key"] = (
        df["target"].astype(str).str.lower()
        + "|"
        + df["version_tag"].astype(str)
        + "|"
        + df["filename"].astype(str).str.lower()
    )
    df = df.sort_values("__sort_key").drop(columns="__sort_key")

    # Normalize column order (keeps extras at end)
    desired_cols = [
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
    ]
    df = stable_sort_columns(df, desired_cols)

    # Ensure the optional (unknown) fields are empty strings rather than NaN
    df = df.fillna("")

    # === Write CSV ===
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    wrote_csv = write_if_changed(OUT_CSV, csv_bytes)

    # === Write JSON mirror ===
    json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
    wrote_json = write_if_changed(OUT_JSON, json_bytes)

    # === Write LOCK (stable, minimal keys) ===
    artifacts = []
    for _, r in df.iterrows():
        artifacts.append(
            {
                "filename": r["filename"],
                "path": r["path"],
                "sha256": r["sha256"],
                "target": r["target"],
                "version_tag": r["version_tag"],
            }
        )
    # Dedupe, then stable sort
    seen = set()
    deduped = []
    for a in artifacts:
        key = (a["path"], a["sha256"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    deduped.sort(key=lambda a: (a["target"], a["version_tag"], a["filename"]))

    lock_payload = {"artifacts": deduped}
    lock_bytes = (json.dumps(lock_payload, indent=2) + "\n").encode("utf-8")
    wrote_lock = write_if_changed(OUT_LOCK, lock_bytes)

    # Summary
    wrote_any = wrote_csv or wrote_json or wrote_lock
    print(f"[INFO] Models found: {len(df)}")
    print(f"[INFO] Wrote CSV: {wrote_csv} -> {OUT_CSV.relative_to(ROOT)}")
    print(f"[INFO] Wrote JSON: {wrote_json} -> {OUT_JSON.relative_to(ROOT)}")
    print(f"[INFO] Wrote LOCK: {wrote_lock} -> {OUT_LOCK.relative_to(ROOT)}")
    if not wrote_any:
        print("[INFO] No manifest changes.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
