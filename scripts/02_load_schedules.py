#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script path: /mnt/data/football_for_mat-main/scripts/02_load_schedules.py

Purpose:
  Ingest all raw schedule CSVs from data/raw/schedules/, standardize columns and types,
  convert to canonical UTC fields when local datetime is given, and write a single
  staged parquet at data/processed/schedules/_staging/schedules_staged.parquet.

Inputs:
  - data/raw/schedules/*.csv
    Accepts any file containing at least these columns (case-insensitive, flexible names):
      season, season_type, week, home_team, away_team
    Optional columns:
      game_datetime_local (ISO-like), game_date_utc, game_time_utc, venue, venue_city, venue_state,
      source, source_event_id
  - mappings/timezones.csv (optional; if present, used to resolve venue -> IANA tz for UTC conversion)

Outputs:
  - data/processed/schedules/_staging/schedules_staged.parquet
  - data/processed/schedules/_staging/schedules_staged.csv  (mirror for convenience)

Notes:
  - No network calls.
  - No assumptions beyond provided mappings. If UTC cannot be resolved, UTC fields remain empty.
"""
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dateutil import tz, parser as dtparser

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "schedules"
STAGING_DIR = ROOT / "data" / "processed" / "schedules" / "_staging"
MAPPINGS_DIR = ROOT / "mappings"
TZ_MAP_PATH = MAPPINGS_DIR / "timezones.csv"

STAGED_PARQUET = STAGING_DIR / "schedules_staged.parquet"
STAGED_CSV = STAGING_DIR / "schedules_staged.csv"

# Flexible header normalization map
CANON_MAP: Dict[str, str] = {
    "season": "season",
    "year": "season",
    "season_type": "season_type",
    "seasontype": "season_type",
    "week": "week",
    "wk": "week",
    "home_team": "home_team",
    "home": "home_team",
    "home_name": "home_team",
    "away_team": "away_team",
    "away": "away_team",
    "away_name": "away_team",
    "game_datetime_local": "game_datetime_local",
    "local_datetime": "game_datetime_local",
    "local_kickoff": "game_datetime_local",
    "kickoff_local": "game_datetime_local",
    "game_date_utc": "game_date_utc",
    "game_time_utc": "game_time_utc",
    "venue": "venue",
    "stadium": "venue",
    "venue_city": "venue_city",
    "city": "venue_city",
    "venue_state": "venue_state",
    "state": "venue_state",
    "source": "source",
    "source_event_id": "source_event_id",
    "event_id": "source_event_id",
}

REQUIRED_MIN = ["season", "season_type", "week", "home_team", "away_team"]


def load_timezone_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    tz_map = {}
    for _, r in df.iterrows():
        venue = str(r.get("venue", "")).strip().lower()
        zone = str(r.get("iana_tz", "")).strip()
        if venue and zone:
            tz_map[venue] = zone
    return tz_map


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in CANON_MAP:
            rename[c] = CANON_MAP[key]
    return df.rename(columns=rename)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["season", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "season_type" in df.columns:
        df["season_type"] = df["season_type"].astype(str).str.strip().str.lower()
    for col in [
        "home_team", "away_team", "venue", "venue_city", "venue_state",
        "source", "source_event_id", "game_datetime_local",
        "game_date_utc", "game_time_utc"
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def split_utc(dt_utc: pd.Timestamp) -> (str, str):
    if pd.isna(dt_utc):
        return "", ""
    return dt_utc.strftime("%Y-%m-%d"), dt_utc.strftime("%H:%M:%S")


def to_utc_from_local(local_str: str, venue: str, tz_map: Dict[str, str]) -> (str, str):
    if not local_str:
        return "", ""
    zone = tz_map.get(str(venue).strip().lower(), "")
    if not zone:
        return "", ""
    try:
        dt_local = dtparser.parse(local_str)
        if not dt_local.tzinfo:
            dt_local = dt_local.replace(tzinfo=tz.gettz(zone))
        else:
            dt_local = dt_local.astimezone(tz.gettz(zone))
        dt_utc = dt_local.astimezone(tz.UTC)
        return split_utc(pd.Timestamp(dt_utc))
    except Exception:
        return "", ""


def standardize(df: pd.DataFrame, tz_map: Dict[str, str]) -> pd.DataFrame:
    df = normalize_headers(df)
    df = coerce_types(df)
    for col in REQUIRED_MIN:
        if col not in df.columns:
            df[col] = pd.NA
    if "game_date_utc" not in df.columns:
        df["game_date_utc"] = ""
    if "game_time_utc" not in df.columns:
        df["game_time_utc"] = ""
    if "game_datetime_local" in df.columns:
        needs_utc = (df["game_date_utc"].astype(str).eq("") |
                     df["game_time_utc"].astype(str).eq(""))
        if needs_utc.any():
            dates, times = [], []
            for _, r in df.loc[needs_utc].iterrows():
                d, t = to_utc_from_local(
                    local_str=str(r.get("game_datetime_local", "")),
                    venue=str(r.get("venue", "")),
                    tz_map=tz_map
                )
                dates.append(d)
                times.append(t)
            df.loc[needs_utc, "game_date_utc"] = dates
            df.loc[needs_utc, "game_time_utc"] = times
    base_cols = [
        "season", "season_type", "week", "game_date_utc", "game_time_utc",
        "home_team", "away_team", "venue", "venue_city", "venue_state",
        "source", "source_event_id", "game_datetime_local"
    ]
    ordered = [c for c in base_cols if c in df.columns] + \
              [c for c in df.columns if c not in base_cols]
    return df[ordered]


def read_all_raw() -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
            df["__source_file"] = str(p.relative_to(ROOT))
            frames.append(df)
        except Exception:
            continue
    return frames


def main() -> int:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    tz_map = load_timezone_map(TZ_MAP_PATH)
    frames = read_all_raw()
    if not frames:
        empty_cols = [
            "season","season_type","week","game_date_utc","game_time_utc",
            "home_team","away_team","venue","venue_city","venue_state",
            "source","source_event_id","game_datetime_local","__source_file"
        ]
        pd.DataFrame(columns=empty_cols).to_parquet(STAGED_PARQUET, index=False)
        pd.DataFrame(columns=empty_cols).to_csv(STAGED_CSV, index=False)
        print(f"Wrote empty staged outputs: {STAGED_PARQUET} ; {STAGED_CSV}")
        return 0
    normed = [standardize(df, tz_map) for df in frames]
    out = pd.concat(normed, ignore_index=True)
    mask_req = (
        out["season"].notna() &
        out["season_type"].astype(str).ne("") &
        out["week"].notna() &
        out["home_team"].astype(str).ne("") &
        out["away_team"].astype(str).ne("")
    )
    out = out.loc[mask_req].copy()
    out.to_parquet(STAGED_PARQUET, index=False)
    out.to_csv(STAGED_CSV, index=False)
    print(f"Wrote staged outputs:\\n  {STAGED_PARQUET}\\n  {STAGED_CSV}\\n  rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
