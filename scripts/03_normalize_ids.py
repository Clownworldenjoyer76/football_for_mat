#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full script path: /mnt/data/football_for_mat-main/scripts/03_normalize_ids.py

Purpose:
  Normalize staged schedules by resolving team abbreviations and IDs, standardizing
  venues, and generating canonical identifiers.

Inputs:
  - data/processed/schedules/_staging/schedules_staged.csv
  - mappings/team_aliases.csv (columns: alias,team_abbr,team_full)
  - mappings/team_ids.csv (columns: team_abbr,team_full,gsis_id,pfr_id,espn_id,sportradar_id)
  - mappings/venue_aliases.csv (columns: alias,venue,venue_city,venue_state)

Outputs:
  - data/processed/schedules/schedules_normalized.csv
  - data/processed/schedules/schedules_normalized.parquet

Behavior (no assumptions):
  - Only uses explicit mappings to resolve team abbreviations and IDs.
  - If a team cannot be resolved via mappings, leaves abbr/IDs empty.
  - Only computes canonical_game_id and canonical_hash when required inputs exist.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]

STAGED_CSV = ROOT / "data" / "processed" / "schedules" / "_staging" / "schedules_staged.csv"

MAPPINGS_DIR = ROOT / "mappings"
TEAM_ALIASES_CSV = MAPPINGS_DIR / "team_aliases.csv"       # alias,team_abbr,team_full
TEAM_IDS_CSV = MAPPINGS_DIR / "team_ids.csv"               # team_abbr,team_full,gsis_id,pfr_id,espn_id,sportradar_id
VENUE_ALIASES_CSV = MAPPINGS_DIR / "venue_aliases.csv"     # alias,venue,venue_city,venue_state

OUT_DIR = ROOT / "data" / "processed" / "schedules"
OUT_CSV = OUT_DIR / "schedules_normalized.csv"
OUT_PARQUET = OUT_DIR / "schedules_normalized.parquet"


def load_csv_if_exists(path: Path, **read_csv_kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **read_csv_kwargs)


def build_alias_map(df_alias: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Returns lowercased alias -> (team_abbr, team_full)
    """
    alias_map: Dict[str, Tuple[str, str]] = {}
    if df_alias.empty:
        return alias_map
    cols = {c.lower(): c for c in df_alias.columns}
    alias_col = cols.get("alias")
    abbr_col = cols.get("team_abbr")
    full_col = cols.get("team_full")
    if not alias_col or not abbr_col:
        return alias_map
    for _, r in df_alias.iterrows():
        alias = str(r.get(alias_col, "")).strip()
        abbr = str(r.get(abbr_col, "")).strip()
        full = str(r.get(full_col, "")).strip() if full_col else ""
        if alias and abbr:
            alias_map[alias.lower()] = (abbr, full)
    return alias_map


def build_team_ids_map(df_ids: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns to lowercase canonical names and drop duplicates on team_abbr.
    """
    if df_ids.empty:
        return pd.DataFrame(columns=["team_abbr","team_full","gsis_id","pfr_id","espn_id","sportradar_id"])
    df = df_ids.copy()
    df.columns = [c.lower() for c in df.columns]
    keep_cols = ["team_abbr","team_full","gsis_id","pfr_id","espn_id","sportradar_id"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[keep_cols].copy()
    df["team_abbr"] = df["team_abbr"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["team_abbr"])
    return df


def resolve_team(value: str,
                 alias_map: Dict[str, Tuple[str, str]],
                 team_ids_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Resolve a provided team string to (team_abbr, team_full) using explicit data only.
    Rules:
      - If value matches a known team_abbr in team_ids_df, return that abbr + full.
      - Else if value matches an alias (case-insensitive), return mapped abbr + full.
      - Else return ("","") (no assumptions).
    """
    s = (value or "").strip()
    if not s:
        return ("", "")
    # Direct abbreviation match?
    if not team_ids_df.empty:
        hit = team_ids_df.loc[team_ids_df["team_abbr"].str.casefold() == s.casefold()]
        if not hit.empty:
            row = hit.iloc[0]
            return (row["team_abbr"], row.get("team_full", ""))
    # Alias match?
    t = alias_map.get(s.lower())
    if t:
        abbr, full = t
        # If team_full missing in alias map, try lookup from team_ids_df
        if not full and not team_ids_df.empty:
            hit = team_ids_df.loc[team_ids_df["team_abbr"].str.casefold() == abbr.casefold()]
            if not hit.empty:
                full = hit.iloc[0].get("team_full", "")
        return (abbr, full)
    # No resolution
    return ("", "")


def apply_venue_aliases(df: pd.DataFrame, df_venue_alias: pd.DataFrame) -> pd.DataFrame:
    if df_venue_alias.empty or df.empty:
        return df
    va = df_venue_alias.copy()
    va.columns = [c.lower() for c in va.columns]
    # Expected columns: alias, venue, venue_city, venue_state
    for c in ["alias","venue","venue_city","venue_state"]:
        if c not in va.columns:
            va[c] = ""
    va = va[["alias","venue","venue_city","venue_state"]].copy()
    va["alias"] = va["alias"].astype(str).str.strip().str.lower()
    # Merge on venue alias
    df = df.copy()
    df["__venue_key"] = df.get("venue", "").astype(str).str.strip().str.lower()
    df = df.merge(va.rename(columns={"alias":"__venue_key"}),
                  on="__venue_key", how="left", suffixes=("","__map"))
    # Fill standardized venue fields if mapped
    for col in ["venue","venue_city","venue_state"]:
        mapped_col = f"{col}__map"
        if mapped_col in df.columns:
            df[col] = df[mapped_col].where(df[mapped_col].astype(str).ne(""), df.get(col, ""))
            df.drop(columns=[mapped_col], inplace=True, errors="ignore")
    df.drop(columns=["__venue_key"], inplace=True, errors="ignore")
    return df


def compute_canonical_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - canonical_game_id = "{season}-{season_type}-{week:02d}-{away_abbr}@{home_abbr}"
      - canonical_hash = SHA1("{season}|{season_type}|{week}|{game_date_utc}|{home_abbr}|{away_abbr}")
    Only when all required components exist; otherwise leave empty.
    """
    df = df.copy()

    def make_game_id(row) -> str:
        season = row.get("season")
        season_type = str(row.get("season_type", "")).strip().lower()
        week = row.get("week")
        home_abbr = str(row.get("home_abbr", "")).strip()
        away_abbr = str(row.get("away_abbr", "")).strip()
        if pd.isna(season) or pd.isna(week) or not season_type or not home_abbr or not away_abbr:
            return ""
        try:
            week_int = int(week)
        except Exception:
            return ""
        return f"{int(season)}-{season_type}-{week_int:02d}-{away_abbr}@{home_abbr}"

    def make_hash(row) -> str:
        season = row.get("season")
        season_type = str(row.get("season_type", "")).strip().lower()
        week = row.get("week")
        game_date_utc = str(row.get("game_date_utc", "")).strip()
        home_abbr = str(row.get("home_abbr", "")).strip()
        away_abbr = str(row.get("away_abbr", "")).strip()
        if pd.isna(season) or pd.isna(week) or not season_type or not game_date_utc or not home_abbr or not away_abbr:
            return ""
        raw = f"{int(season)}|{season_type}|{int(week)}|{game_date_utc}|{home_abbr}|{away_abbr}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    df["canonical_game_id"] = df.apply(make_game_id, axis=1)
    df["canonical_hash"] = df.apply(make_hash, axis=1)
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load staged
    if not STAGED_CSV.exists():
        # No input; write empty outputs
        empty_cols = [
            "season","season_type","week","game_date_utc","game_time_utc",
            "home_team","away_team","venue","venue_city","venue_state",
            "source","source_event_id","game_datetime_local",
            "home_abbr","away_abbr","home_team_full","away_team_full",
            "home_gsis_id","away_gsis_id","home_pfr_id","away_pfr_id",
            "home_espn_id","away_espn_id","home_sportradar_id","away_sportradar_id",
            "canonical_game_id","canonical_hash"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_CSV, index=False)
        try:
            pd.DataFrame(columns=empty_cols).to_parquet(OUT_PARQUET, index=False)
        except Exception:
            pass
        print(f"Input missing: {STAGED_CSV}. Wrote empty outputs.")
        return 0

    df = pd.read_csv(STAGED_CSV)

    # Load mappings
    df_alias = load_csv_if_exists(TEAM_ALIASES_CSV, dtype=str).fillna("")
    df_ids = load_csv_if_exists(TEAM_IDS_CSV, dtype=str).fillna("")
    df_venue_alias = load_csv_if_exists(VENUE_ALIASES_CSV, dtype=str).fillna("")

    alias_map = build_alias_map(df_alias)
    team_ids = build_team_ids_map(df_ids)

    # Resolve teams -> abbr/full
    df = df.copy()
    df["home_abbr"] = ""
    df["away_abbr"] = ""
    df["home_team_full"] = ""
    df["away_team_full"] = ""

    for idx, row in df.iterrows():
        h_abbr, h_full = resolve_team(str(row.get("home_team","")), alias_map, team_ids)
        a_abbr, a_full = resolve_team(str(row.get("away_team","")), alias_map, team_ids)
        df.at[idx, "home_abbr"] = h_abbr
        df.at[idx, "away_abbr"] = a_abbr
        df.at[idx, "home_team_full"] = h_full
        df.at[idx, "away_team_full"] = a_full

    # Join IDs per side (only where abbr is present)
    if not team_ids.empty:
        # Home IDs
        df = df.merge(
            team_ids.add_prefix("home_"),
            how="left",
            left_on="home_abbr", right_on="home_team_abbr"
        )
        # Away IDs
        df = df.merge(
            team_ids.add_prefix("away_"),
            how="left",
            left_on="away_abbr", right_on="away_team_abbr"
        )
        # Clean helper join keys
        for c in ["home_team_abbr","away_team_abbr"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    # Apply venue aliases
    df = apply_venue_aliases(df, df_venue_alias)

    # Compute canonical ids
    df = compute_canonical_ids(df)

    # Column ordering
    ordered_cols = [
        "season","season_type","week","game_date_utc","game_time_utc",
        "home_team","away_team","home_abbr","away_abbr","home_team_full","away_team_full",
        "venue","venue_city","venue_state",
        "source","source_event_id","game_datetime_local",
        "home_gsis_id","away_gsis_id","home_pfr_id","away_pfr_id",
        "home_espn_id","away_espn_id","home_sportradar_id","away_sportradar_id",
        "canonical_game_id","canonical_hash"
    ]
    # Ensure all columns exist
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered_cols].copy()

    # Write outputs
    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception:
        pass

    print(f"Wrote normalized schedules:\n  {OUT_CSV}\n  {OUT_PARQUET if OUT_PARQUET.exists() else '(parquet skipped)'}\n  rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
