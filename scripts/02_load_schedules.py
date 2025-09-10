#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full path: /football_for_mat-main/scripts/02_load_schedules.py

Purpose:
  Load raw schedule CSVs, derive required canonical fields, and write the staged file.

Inputs (must exist):
  - /football_for_mat-main/data/raw/schedules/schedules_2024.csv
  - /football_for_mat-main/data/raw/schedules/schedules_2025.csv
  - /football_for_mat-main/mappings/timezones.csv         # columns: team_abbr,timezone

Assumed raw columns (from your repo files):
  - game_id,season,game_type,week,game_date,weekday,gametime,away_team,home_team
  - Optional: venue,venue_city,venue_state

Derivations (no network calls):
  - season_type: "regular" if week in [1..18], "postseason" if week in [19..22], else "preseason"
  - game_date_utc, game_time_utc: convert local date+time using home team timezone from mappings/timezones.csv
  - source: basename of input file (e.g., schedules_2024.csv)
  - source_event_id: SHA1("season|season_type|week|game_date_utc|home_team|away_team")
  - game_status: "scheduled" (deterministic constant)

Output:
  - /football_for_mat-main/data/processed/schedules/_staging/schedules_staged.csv
  - /football_for_mat-main/data/processed/schedules/_staging/schedules_staged.parquet
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import pandas as pd
from dateutil import parser
import pytz

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "schedules"
RAW_2024 = RAW_DIR / "schedules_2024.csv"
RAW_2025 = RAW_DIR / "schedules_2025.csv"

TZ_MAP_CSV = ROOT / "mappings" / "timezones.csv"   # columns: team_abbr,timezone (IANA like America/New_York)

STAGING_DIR = ROOT / "data" / "processed" / "schedules" / "_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = STAGING_DIR / "schedules_staged.csv"
OUT_PARQUET = STAGING_DIR / "schedules_staged.parquet"


def _season_type_from_week(week_val: str) -> str:
    try:
        w = int(float(str(week_val)))
    except Exception:
        return "regular"
    if 1 <= w <= 18:
        return "regular"
    if 19 <= w <= 22:
        return "postseason"
    return "preseason"


def _parse_local_to_utc(date_str: str, time_str: str, tz_name: str) -> Tuple[str, str]:
    """
    Returns (YYYY-MM-DD, HH:MM:SS) in UTC.
    If parsing fails, returns ("","").
    """
    try:
        # Fallbacks
        date_str = (date_str or "").strip()
        time_str = (time_str or "").strip()
        if not date_str or not time_str:
            return ("", "")

        # Parse date and time components
        d = parser.parse(date_str).date()
        t = parser.parse(time_str).time()

        # Localize using tz and convert to UTC
        tz = pytz.timezone(tz_name)
        local_dt = tz.localize(pd.Timestamp.combine(d, t).to_pydatetime(), is_dst=None)
        utc_dt = local_dt.astimezone(pytz.UTC)

        return (utc_dt.strftime("%Y-%m-%d"), utc_dt.strftime("%H:%M:%S"))
    except Exception:
        return ("", "")


def _load_tz_map(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}
    abbr_col = cols.get("team_abbr")
    tz_col = cols.get("timezone")
    if not abbr_col or not tz_col:
        return {}
    return {str(r[abbr_col]).strip().upper(): str(r[tz_col]).strip() for _, r in df.iterrows() if str(r[abbr_col]).strip()}


def _prepare(df: pd.DataFrame, source_name: str, tz_map: dict) -> pd.DataFrame:
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    req = ["season", "week", "game_date", "gametime", "home_team", "away_team"]
    for r in req:
        if r not in cols:
            df[r] = ""

    # Derive season_type
    season_type = df[cols.get("week", "week")].apply(_season_type_from_week)

    # Derive UTC fields using home team tz
    home_col = cols.get("home_team", "home_team")
    utc_dates, utc_times = [], []
    for _, row in df.iterrows():
        home_abbr = str(row[home_col]).strip().upper()
        tz = tz_map.get(home_abbr, "UTC")
        d_utc, t_utc = _parse_local_to_utc(str(row[cols.get("game_date","game_date")]),
                                           str(row[cols.get("gametime","gametime")]), tz)
        utc_dates.append(d_utc)
        utc_times.append(t_utc)

    # Optional venue fields
    venue = df[cols.get("venue")] if "venue" in cols else ""
    venue_city = df[cols.get("venue_city")] if "venue_city" in cols else ""
    venue_state = df[cols.get("venue_state")] if "venue_state" in cols else ""

    # Build staged frame
    staged = pd.DataFrame({
        "season": df[cols.get("season","season")],
        "season_type": season_type,
        "week": df[cols.get("week","week")],
        "game_date_utc": utc_dates,
        "game_time_utc": utc_times,
        "home_team": df[home_col],
        "away_team": df[cols.get("away_team","away_team")],
        "venue": venue,
        "venue_city": venue_city,
        "venue_state": venue_state,
        "source": source_name,
        "source_event_id": "",     # filled below
        "game_datetime_local": df[cols.get("gametime","gametime")],
        "__source_file": source_name,
    })

    # Deterministic source_event_id
    def _mk_id(r) -> str:
        raw = f"{r.get('season','')}|{r.get('season_type','')}|{r.get('week','')}|{r.get('game_date_utc','')}|{r.get('home_team','')}|{r.get('away_team','')}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    staged["source_event_id"] = staged.apply(_mk_id, axis=1)

    # Add game_status = "scheduled"
    staged["game_status"] = "scheduled"

    # Final typing/cleanup
    staged = staged.fillna("")
    return staged


def main() -> int:
    # Load tz mapping
    tz_map = _load_tz_map(TZ_MAP_CSV)

    frames = []
    for path in [RAW_2024, RAW_2025]:
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            frames.append(_prepare(df, path.name, tz_map))

    if not frames:
        # Write empty outputs if nothing present
        pd.DataFrame(columns=[
            "season","season_type","week","game_date_utc","game_time_utc",
            "home_team","away_team","venue","venue_city","venue_state",
            "source","source_event_id","game_datetime_local","game_status","__source_file"
        ]).to_csv(OUT_CSV, index=False)
        try:
            pd.DataFrame().to_parquet(OUT_PARQUET, index=False)
        except Exception:
            pass
        print("No raw schedules found. Wrote empty staged outputs.")
        return 0

    staged = pd.concat(frames, ignore_index=True)

    # Enforce string types and stable sort
    staged = staged.astype(str)
    staged["__wk"] = pd.to_numeric(staged["week"], errors="coerce")
    staged.sort_values(by=["season", "season_type", "__wk", "home_team", "away_team"], inplace=True, kind="mergesort")
    staged.drop(columns="__wk", inplace=True)

    # Write
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged.to_csv(OUT_CSV, index=False)
    try:
        staged.to_parquet(OUT_PARQUET, index=False)
    except Exception:
        pass

    print(f"Wrote staged schedules:\n  {OUT_CSV}\n  {OUT_PARQUET if OUT_PARQUET.exists() else '(parquet skipped)'}\n  rows={len(staged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
