#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pull NFL schedule from nfl_data_py.

Creates:
  docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv

Output columns:
  season
  season_type
  week
  game_id
  game_date
  game_time
  away_team
  home_team
  neutral_site
  stadium
  roof
  surface
  home_timezone
  away_timezone
  game_timezone
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:
    sys.exit(f"ERROR: nfl_data_py import failed: {e}")


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

NFL_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = NFL_DIR / "config"
MAPPING_DIR = CONFIG_DIR / "mapping"

SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
TEAM_MAP_FILE = MAPPING_DIR / "team_map_nfl.csv"
STADIUM_MAP_FILE = MAPPING_DIR / "stadium_map_nfl.csv"

OUTPUT_DIR = NFL_DIR / "00_intake" / "schedule"
ERROR_DIR = NFL_DIR / "errors" / "00_intake"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_schedule.txt"


OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
]


REQUIRED_SOURCE_COLS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
]


# ─────────────────────────────────────────────
# LOGGING / FAIL
# ─────────────────────────────────────────────

def write_log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def fail(message: str) -> None:
    write_log(f"ERROR: {message}")
    sys.exit(f"ERROR: {message}")


# ─────────────────────────────────────────────
# BASIC HELPERS
# ─────────────────────────────────────────────

def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def norm_key(value: Any) -> str:
    return clean(value).lower()


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def bool_to_int(value: Any) -> str:
    text = clean(value).lower()

    if text in {"true", "t", "yes", "y", "1", "neutral"}:
        return "1"

    if text in {"false", "f", "no", "n", "0", "home"}:
        return "0"

    return ""


def normalize_season_type(value: Any) -> str:
    text = clean(value).lower().replace("_", " ").replace("-", " ")

    if text in {"reg", "regular", "regular season"}:
        return "REG"

    if text in {"post", "playoff", "playoffs", "postseason"}:
        return "POST"

    if text in {"pre", "preseason"}:
        return "PRE"

    return clean(value).upper()


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing settings file: {path}")

    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            fail(f"settings.yaml must contain key/value settings: {path}")
        return data
    except ImportError:
        data: dict[str, Any] = {}

        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')

                if not key:
                    continue

                data[key] = value

        return data
    except Exception as e:
        fail(f"Could not read settings.yaml: {e}")


def load_target_settings() -> tuple[int, str]:
    settings = load_simple_yaml(SETTINGS_FILE)

    raw_season = settings.get("season", "")
    if clean(raw_season) == "":
        fail("settings.yaml missing required value: season")

    try:
        season = int(raw_season)
    except Exception:
        fail(f"settings.yaml season must be an integer. Found: {raw_season}")

    season_type = normalize_season_type(settings.get("season_type", ""))

    return season, season_type


# ─────────────────────────────────────────────
# OPTIONAL MAP LOADERS
# ─────────────────────────────────────────────

def load_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        write_log(f"WARNING: Optional mapping file missing: {path}")
        return []

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({clean(k): clean(v) for k, v in row.items()})

    return rows


def build_team_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}

    for row in rows:
        source_name = clean(row.get("source_name"))
        canonical_team = clean(row.get("canonical_team"))
        team_abbr = clean(row.get("team_abbr"))
        team_id = clean(row.get("team_id"))

        payload = {
            "source_name": source_name,
            "canonical_team": canonical_team,
            "team_abbr": team_abbr,
            "team_id": team_id,
        }

        for key_value in [source_name, canonical_team, team_abbr, team_id]:
            key = norm_key(key_value)
            if key:
                lookup[key] = payload

    return lookup


def build_stadium_indexes(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    team_index: dict[str, dict[str, str]] = {}
    stadium_index: dict[str, dict[str, str]] = {}

    for row in rows:
        team = clean(row.get("team"))
        stadium = clean(row.get("stadium"))

        if team:
            team_index[norm_key(team)] = row

        if stadium:
            stadium_index[norm_key(stadium)] = row

    return team_index, stadium_index


def resolve_team_payload(
    team_value: Any,
    team_lookup: dict[str, dict[str, str]],
) -> dict[str, str]:
    key = norm_key(team_value)

    if key in team_lookup:
        return team_lookup[key]

    return {
        "source_name": clean(team_value),
        "canonical_team": clean(team_value),
        "team_abbr": clean(team_value),
        "team_id": "",
    }


def lookup_stadium_row_for_team(
    team_value: Any,
    team_lookup: dict[str, dict[str, str]],
    stadium_team_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    team_payload = resolve_team_payload(team_value, team_lookup)

    keys = [
        team_payload.get("canonical_team", ""),
        team_payload.get("team_abbr", ""),
        team_payload.get("source_name", ""),
        clean(team_value),
    ]

    for key_value in keys:
        key = norm_key(key_value)
        if key and key in stadium_team_index:
            return stadium_team_index[key]

    return {}


def get_timezone_for_team(
    team_value: Any,
    team_lookup: dict[str, dict[str, str]],
    stadium_team_index: dict[str, dict[str, str]],
) -> str:
    row = lookup_stadium_row_for_team(team_value, team_lookup, stadium_team_index)
    return clean(row.get("timezone"))


def get_game_timezone(
    stadium: Any,
    neutral_site: Any,
    home_timezone: str,
    stadium_name_index: dict[str, dict[str, str]],
) -> str:
    stadium_key = norm_key(stadium)

    if stadium_key and stadium_key in stadium_name_index:
        return clean(stadium_name_index[stadium_key].get("timezone"))

    if clean(neutral_site) == "1":
        return ""

    return home_timezone


# ─────────────────────────────────────────────
# SOURCE FETCH / NORMALIZATION
# ─────────────────────────────────────────────

def fetch_schedule(season: int) -> pd.DataFrame:
    try:
        df = nfl.import_schedules([season])
    except Exception as e:
        fail(f"nfl_data_py import_schedules failed for season {season}: {e}")

    if df is None or df.empty:
        fail(f"No schedule rows returned for season {season}")

    if "gameday" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"gameday": "game_date"})

    if "gametime" in df.columns and "game_time" not in df.columns:
        df = df.rename(columns={"gametime": "game_time"})

    missing = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        fail(f"Schedule source missing required columns: {missing}")

    return df


def apply_season_type_filter(df: pd.DataFrame, requested_season_type: str) -> pd.DataFrame:
    season_type_col = first_existing_col(df, ["season_type", "game_type"])

    if season_type_col is None:
        df = df.copy()
        df["season_type"] = requested_season_type
        return df

    df = df.copy()
    df["season_type"] = df[season_type_col].apply(normalize_season_type)

    if requested_season_type:
        df = df[df["season_type"] == requested_season_type].copy()

    if df.empty:
        fail(f"No schedule rows remain after season_type filter: {requested_season_type}")

    return df


def build_output_df(
    source_df: pd.DataFrame,
    team_lookup: dict[str, dict[str, str]],
    stadium_team_index: dict[str, dict[str, str]],
    stadium_name_index: dict[str, dict[str, str]],
) -> pd.DataFrame:
    stadium_col = first_existing_col(source_df, ["stadium", "stadium_name"])
    roof_col = first_existing_col(source_df, ["roof"])
    surface_col = first_existing_col(source_df, ["surface"])
    neutral_col = first_existing_col(source_df, ["neutral_site", "location"])

    rows: list[dict[str, str]] = []

    for _, row in source_df.iterrows():
        home_team = clean(row.get("home_team"))
        away_team = clean(row.get("away_team"))

        neutral_site = ""
        if neutral_col:
            neutral_site = bool_to_int(row.get(neutral_col))

        home_stadium_row = lookup_stadium_row_for_team(
            home_team,
            team_lookup,
            stadium_team_index,
        )

        source_stadium = clean(row.get(stadium_col)) if stadium_col else ""
        stadium = source_stadium or clean(home_stadium_row.get("stadium"))

        source_roof = clean(row.get(roof_col)) if roof_col else ""
        source_surface = clean(row.get(surface_col)) if surface_col else ""

        stadium_row = stadium_name_index.get(norm_key(stadium), {}) if stadium else {}

        roof = (
            source_roof
            or clean(stadium_row.get("roof_type"))
            or clean(home_stadium_row.get("roof_type"))
            or clean(stadium_row.get("roof"))
            or clean(home_stadium_row.get("roof"))
        )

        surface = (
            source_surface
            or clean(stadium_row.get("surface"))
            or clean(home_stadium_row.get("surface"))
        )

        home_timezone = get_timezone_for_team(
            home_team,
            team_lookup,
            stadium_team_index,
        )

        away_timezone = get_timezone_for_team(
            away_team,
            team_lookup,
            stadium_team_index,
        )

        game_timezone = get_game_timezone(
            stadium=stadium,
            neutral_site=neutral_site,
            home_timezone=home_timezone,
            stadium_name_index=stadium_name_index,
        )

        rows.append(
            {
                "season": clean(row.get("season")),
                "season_type": normalize_season_type(row.get("season_type")),
                "week": clean(row.get("week")),
                "game_id": clean(row.get("game_id")),
                "game_date": clean(row.get("game_date")),
                "game_time": clean(row.get("game_time")),
                "away_team": away_team,
                "home_team": home_team,
                "neutral_site": neutral_site,
                "stadium": stadium,
                "roof": roof,
                "surface": surface,
                "home_timezone": home_timezone,
                "away_timezone": away_timezone,
                "game_timezone": game_timezone,
            }
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    out = out.sort_values(
        ["season", "season_type", "week", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    return out


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        fail(f"Output missing required columns: {missing}")

    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    if extra:
        fail(f"Output contains unexpected columns: {extra}")

    if df.empty:
        fail("Output schedule is empty")

    if df["game_id"].astype(str).str.strip().eq("").any():
        fail("Output contains blank game_id")

    if df["season"].astype(str).str.strip().eq("").any():
        fail("Output contains blank season")

    if df["week"].astype(str).str.strip().eq("").any():
        fail("Output contains blank week")

    if df["game_date"].astype(str).str.strip().eq("").any():
        fail("Output contains blank game_date")

    if df["home_team"].astype(str).str.strip().eq("").any():
        fail("Output contains blank home_team")

    if df["away_team"].astype(str).str.strip().eq("").any():
        fail("Output contains blank away_team")

    if df.duplicated(subset=["game_id"]).any():
        dups = df[df.duplicated(subset=["game_id"], keep=False)]
        fail(
            "Duplicate game_id values detected:\n"
            + dups[["season", "season_type", "week", "game_id"]]
            .head(25)
            .to_string(index=False)
        )

    try:
        pd.to_datetime(df["game_date"], errors="raise")
    except Exception as e:
        fail(f"game_date values are not parseable: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    reset_log()

    try:
        season, requested_season_type = load_target_settings()

        write_log(f"season={season}")
        write_log(f"requested_season_type={requested_season_type}")

        team_map_rows = load_csv_if_exists(TEAM_MAP_FILE)
        stadium_map_rows = load_csv_if_exists(STADIUM_MAP_FILE)

        team_lookup = build_team_lookup(team_map_rows)
        stadium_team_index, stadium_name_index = build_stadium_indexes(stadium_map_rows)

        source_df = fetch_schedule(season)
        source_df = apply_season_type_filter(source_df, requested_season_type)

        out_df = build_output_df(
            source_df=source_df,
            team_lookup=team_lookup,
            stadium_team_index=stadium_team_index,
            stadium_name_index=stadium_name_index,
        )

        validate_output(out_df)

        out_file = OUTPUT_DIR / f"{season}_schedule.csv"
        out_df.to_csv(out_file, index=False)

        write_log(f"rows_written={len(out_df)}")
        write_log(f"output={out_file}")

        print(f"Wrote {len(out_df)} rows to {out_file}")

    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        write_log(tb)
        sys.exit(f"ERROR: pull_schedule.py failed. See log: {LOG_FILE}")


if __name__ == "__main__":
    main()
