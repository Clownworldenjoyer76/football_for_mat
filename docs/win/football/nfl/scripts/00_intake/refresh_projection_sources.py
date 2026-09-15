#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


def nfl_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_parquet(tag: str, filename: str) -> pd.DataFrame:
    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"{tag}/{filename}"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "football_for_mat/1.0"},
    )

    temp_path: Path | None = None

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".parquet",
                delete=False,
            )
            temp_path = Path(handle.name)

            with handle:
                shutil.copyfileobj(response, handle)

        df = pd.read_parquet(temp_path)

        if df.empty:
            raise RuntimeError(f"Downloaded source is empty: {url}")

        return df

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_atomic(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".tmp")

    try:
        df.to_parquet(temp_target, index=False)
        os.replace(temp_target, target)
    finally:
        if temp_target.exists():
            temp_target.unlink()


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}"
        )


def validate_depth(df: pd.DataFrame) -> None:
    columns = set(df.columns)

    current_schema = (
        {"dt", "team", "pos_rank"}
        <= columns
        and bool({"pos_slot", "position", "pos_abb"} & columns)
    )

    legacy_schema = (
        {"week", "club_code", "depth_team"}
        <= columns
        and bool({"depth_position", "position"} & columns)
    )

    if not (current_schema or legacy_schema):
        raise ValueError(
            "depth charts: unsupported schema; "
            f"columns={sorted(columns)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    season = args.season
    root = nfl_root()

    snap_filename = f"snap_counts_{season}.parquet"
    depth_filename = f"depth_charts_{season}.parquet"

    snaps = download_parquet(
        "snap_counts",
        snap_filename,
    )
    require_columns(
        snaps,
        {"team", "week", "offense_pct", "defense_pct"},
        "snap counts",
    )

    depth = download_parquet(
        "depth_charts",
        depth_filename,
    )
    validate_depth(depth)

    snap_target = (
        root
        / "data/historic_data/snap_counts"
        / snap_filename
    )
    depth_target = (
        root
        / "data/historic_data/depth_charts"
        / depth_filename
    )

    write_atomic(snaps, snap_target)
    write_atomic(depth, depth_target)

    print(
        f"WROTE {snap_target} | rows={len(snaps)}"
    )
    print(
        f"WROTE {depth_target} | rows={len(depth)}"
    )
    print(
        "Participation is intentionally optional for "
        "current-season projection."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
