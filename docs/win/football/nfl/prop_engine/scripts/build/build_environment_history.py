#!/usr/bin/env python3
"""
Build historical NFL game-environment features.

READS:
    docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv
    docs/win/football/nfl/data/master/team_master.csv

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/environment.parquet

POLICY:
    - Regular-season 2012-2025 games only.
    - No sportsbook/odds source is read.
    - Only schedule/game environment columns are loaded from the historical
      games file; embedded betting columns in that file are not loaded.
    - Historical temperature/wind map directly to the same concepts available
      in current-week weather: temperature and wind_speed.
    - Travel calculations mirror the repository's current build_travel.py:
      away-team home coordinates -> home-team home coordinates, regardless of
      neutral site, with haversine miles and DST-aware timezone differences.
    - Neutral-site and international flags describe the actual scheduled game.
    - Historical relocation aliases are used only for travel lookup, not to
      rewrite historical SD/OAK/STL team labels.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import json
import math
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "divisional_game_flag",
    "neutral_site_flag",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "home_rest_days",
    "away_rest_days",
    "miles_traveled_away",
    "time_zones_crossed_away",
    "east_to_west_flag",
    "west_to_east_flag",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]

GAME_READ_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "location",
    "away_rest",
    "home_rest",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium_id",
    "stadium",
]

TEAM_MASTER_COLUMNS = [
    "canonical_team",
    "team_abbr",
    "latitude",
    "longitude",
    "timezone",
    "venue_country",
]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

EARTH_RADIUS_MILES = 3958.8

INTERNATIONAL_STADIUMS = {
    "allianz arena",
    "arena corinthians",
    "azteca stadium",
    "corinthians arena",
    "croke park",
    "deutsche bank park",
    "estadio azteca",
    "neo quimica arena",
    "olympic stadium",
    "rogers centre",
    "santiago bernabeu stadium",
    "tottenham hotspur stadium",
    "tottenham stadium",
    "twickenham stadium",
    "wembley stadium",
}

# The historical games source currently carries domestic stadium names for
# these 2025 international games. Preserve the source stadium fields but
# correct international_flag so historical/current semantics remain aligned.
# Key: (season, week, normalized home_team, normalized away_team)
INTERNATIONAL_GAME_KEYS = {
    (2025, 1, "LAC", "KC"),
    (2025, 4, "PIT", "MIN"),
    (2025, 5, "CLE", "MIN"),
    (2025, 6, "NYJ", "DEN"),
    (2025, 7, "JAX", "LAR"),
    (2025, 10, "IND", "ATL"),
    (2025, 11, "MIA", "WSH"),
}


def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def normalized_output_team(value: Any) -> str:
    """
    Normalize only the repository-wide aliases used by common.py.

    SD/OAK/STL are intentionally retained in historical output labels.
    """
    return common.normalize_team(value)


def travel_team(value: Any) -> str:
    """
    Resolve a historical franchise to the current team-master lookup key.

    This lookup-only normalization mirrors earlier Prop Engine franchise
    continuity policy without rewriting historical output labels.
    """
    team = normalized_output_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def stadium_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text.casefold(),
    )

    return " ".join(text.split())


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def parse_binary(series: pd.Series, label: str) -> pd.Series:
    values = numeric(series)

    invalid = values.notna() & ~values.isin([0.0, 1.0])

    if invalid.any():
        sample = (
            series.loc[invalid]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{label}: expected binary 0/1 values; sample={sample}"
        )

    if values.isna().any():
        raise ValueError(f"{label}: contains missing values.")

    return values.astype("int8")


def parse_location_flag(value: Any) -> int:
    text = clean(value).casefold()

    if text == "neutral":
        return 1

    if text == "home":
        return 0

    raise ValueError(
        f"Unsupported historical game location value: {value!r}"
    )


def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
    )

    return EARTH_RADIUS_MILES * (
        2.0 * math.asin(math.sqrt(a))
    )


def utc_offset_hours(
    timezone_name: str,
    gameday: str,
) -> float:
    dt = datetime.strptime(
        gameday,
        "%Y-%m-%d",
    ).replace(
        tzinfo=ZoneInfo(timezone_name)
    )

    offset = dt.utcoffset()

    if offset is None:
        raise ValueError(
            f"No UTC offset for timezone={timezone_name!r}, "
            f"gameday={gameday!r}"
        )

    return offset.total_seconds() / 3600.0


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = common.repo_root() / path
    return path


def load_team_lookup(path: Path) -> dict[str, dict[str, Any]]:
    source = pd.read_csv(
        path,
        usecols=TEAM_MASTER_COLUMNS,
        low_memory=False,
    )

    common.require_columns(
        source,
        TEAM_MASTER_COLUMNS,
        str(path),
    )

    lookup: dict[str, dict[str, Any]] = {}

    for index, row in source.iterrows():
        abbr = travel_team(row["team_abbr"])

        if not abbr:
            continue

        lat = pd.to_numeric(
            pd.Series([row["latitude"]]),
            errors="coerce",
        ).iloc[0]
        lon = pd.to_numeric(
            pd.Series([row["longitude"]]),
            errors="coerce",
        ).iloc[0]
        timezone_name = clean(row["timezone"])

        if (
            pd.isna(lat)
            or pd.isna(lon)
            or not timezone_name
        ):
            raise ValueError(
                f"{path} row {index + 2}: incomplete travel data "
                f"for team_abbr={abbr!r}"
            )

        record = {
            "latitude": float(lat),
            "longitude": float(lon),
            "timezone": timezone_name,
            "venue_country": clean(row["venue_country"]).upper(),
            "canonical_team": clean(row["canonical_team"]),
        }

        existing = lookup.get(abbr)

        if existing is None:
            lookup[abbr] = record
        elif existing != record:
            raise ValueError(
                f"{path}: conflicting travel records for team_abbr={abbr}"
            )

    if not lookup:
        raise RuntimeError(f"{path}: no usable team-master records.")

    return lookup


def international_flag(
    *,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    stadium: str,
) -> int:
    key = (
        season,
        week,
        home_team,
        away_team,
    )

    if key in INTERNATIONAL_GAME_KEYS:
        return 1

    return int(
        stadium_key(stadium)
        in INTERNATIONAL_STADIUMS
    )


def build_environment(
    games_path: Path,
    team_master_path: Path,
    *,
    historical_start: int,
    historical_end: int,
) -> pd.DataFrame:
    games = pd.read_csv(
        games_path,
        usecols=GAME_READ_COLUMNS,
        low_memory=False,
    )

    common.require_columns(
        games,
        GAME_READ_COLUMNS,
        str(games_path),
    )

    games["season"] = pd.to_numeric(
        games["season"],
        errors="raise",
    ).astype(int)

    games["week"] = pd.to_numeric(
        games["week"],
        errors="raise",
    ).astype(int)

    games = games.loc[
        games["season"].between(
            historical_start,
            historical_end,
        )
        & games["game_type"]
        .astype(str)
        .str.upper()
        .eq("REG")
    ].copy()

    if games.empty:
        raise RuntimeError(
            f"{games_path}: no modeled regular-season games."
        )

    # Preserve the historical source game_id exactly so this table joins
    # directly to the historical player-game universe. Validate format without
    # replacing the source ID with common.parse_game_id's normalized rendering.
    games["game_id"] = games["game_id"].map(clean)

    if games["game_id"].eq("").any():
        raise ValueError(f"{games_path}: blank modeled game_id.")

    for game_id in games["game_id"]:
        common.parse_game_id(game_id)

    games["home_team"] = games["home_team"].map(normalized_output_team)
    games["away_team"] = games["away_team"].map(normalized_output_team)

    if games["home_team"].eq("").any() or games["away_team"].eq("").any():
        raise ValueError(f"{games_path}: blank team after normalization.")

    games["divisional_game_flag"] = parse_binary(
        games["div_game"],
        "div_game",
    )

    games["neutral_site_flag"] = (
        games["location"]
        .map(parse_location_flag)
        .astype("int8")
    )

    games["temperature"] = numeric(games["temp"])
    games["wind"] = numeric(games["wind"])
    games["home_rest_days"] = numeric(games["home_rest"])
    games["away_rest_days"] = numeric(games["away_rest"])

    if games["home_rest_days"].isna().any():
        raise ValueError("home_rest contains missing/non-numeric modeled rows.")

    if games["away_rest_days"].isna().any():
        raise ValueError("away_rest contains missing/non-numeric modeled rows.")

    games["weather_missing_flag"] = (
        games["temperature"].isna()
        | games["wind"].isna()
    ).astype("int8")

    team_lookup = load_team_lookup(team_master_path)

    miles: list[float] = []
    zones: list[float] = []
    east_to_west: list[int] = []
    west_to_east: list[int] = []
    international: list[int] = []
    travel_missing: list[int] = []

    for row in games.itertuples(index=False):
        away_key = travel_team(row.away_team)
        home_key = travel_team(row.home_team)

        away = team_lookup.get(away_key)
        home = team_lookup.get(home_key)

        if away is None or home is None:
            miles.append(float("nan"))
            zones.append(float("nan"))
            east_to_west.append(0)
            west_to_east.append(0)
            travel_missing.append(1)
        else:
            try:
                distance = round(
                    haversine_miles(
                        away["latitude"],
                        away["longitude"],
                        home["latitude"],
                        home["longitude"],
                    ),
                    1,
                )

                away_offset = utc_offset_hours(
                    away["timezone"],
                    clean(row.gameday),
                )
                home_offset = utc_offset_hours(
                    home["timezone"],
                    clean(row.gameday),
                )

                zone_count = abs(
                    home_offset - away_offset
                )

                away_lon = float(away["longitude"])
                home_lon = float(home["longitude"])

                if home_lon < away_lon:
                    e2w = 1
                    w2e = 0
                elif home_lon > away_lon:
                    e2w = 0
                    w2e = 1
                else:
                    e2w = 0
                    w2e = 0

                miles.append(float(distance))
                zones.append(float(zone_count))
                east_to_west.append(e2w)
                west_to_east.append(w2e)
                travel_missing.append(0)

            except Exception:
                miles.append(float("nan"))
                zones.append(float("nan"))
                east_to_west.append(0)
                west_to_east.append(0)
                travel_missing.append(1)

        international.append(
            international_flag(
                season=int(row.season),
                week=int(row.week),
                home_team=clean(row.home_team),
                away_team=clean(row.away_team),
                stadium=clean(row.stadium),
            )
        )

    games["miles_traveled_away"] = pd.Series(
        miles,
        index=games.index,
        dtype="float64",
    )
    games["time_zones_crossed_away"] = pd.Series(
        zones,
        index=games.index,
        dtype="float64",
    )
    games["east_to_west_flag"] = pd.Series(
        east_to_west,
        index=games.index,
        dtype="int8",
    )
    games["west_to_east_flag"] = pd.Series(
        west_to_east,
        index=games.index,
        dtype="int8",
    )
    games["international_flag"] = pd.Series(
        international,
        index=games.index,
        dtype="int8",
    )
    games["travel_missing_flag"] = pd.Series(
        travel_missing,
        index=games.index,
        dtype="int8",
    )

    output = pd.DataFrame(
        {
            "season": games["season"].astype(int),
            "week": games["week"].astype(int),
            "game_id": games["game_id"],
            "gameday": games["gameday"].map(clean),
            "home_team": games["home_team"],
            "away_team": games["away_team"],
            "divisional_game_flag": games["divisional_game_flag"],
            "neutral_site_flag": games["neutral_site_flag"],
            "stadium": games["stadium"].map(clean),
            "stadium_id": games["stadium_id"].map(clean),
            "roof": games["roof"].map(clean),
            "surface": games["surface"].map(
                lambda value: clean(value) or None
            ),
            "temperature": games["temperature"],
            "wind": games["wind"],
            "home_rest_days": games["home_rest_days"],
            "away_rest_days": games["away_rest_days"],
            "miles_traveled_away": games["miles_traveled_away"],
            "time_zones_crossed_away": games["time_zones_crossed_away"],
            "east_to_west_flag": games["east_to_west_flag"],
            "west_to_east_flag": games["west_to_east_flag"],
            "international_flag": games["international_flag"],
            "weather_missing_flag": games["weather_missing_flag"],
            "travel_missing_flag": games["travel_missing_flag"],
        }
    )

    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Issue 14 output header/order mismatch.")

    common.ensure_unique(
        output,
        ["season", "week", "game_id"],
        "Issue 14 environment grain",
    )

    for column in [
        "divisional_game_flag",
        "neutral_site_flag",
        "east_to_west_flag",
        "west_to_east_flag",
        "international_flag",
        "weather_missing_flag",
        "travel_missing_flag",
    ]:
        if not output[column].isin([0, 1]).all():
            raise ValueError(f"{column}: non-binary value found.")

    if (
        output["east_to_west_flag"]
        + output["west_to_east_flag"]
        > 1
    ).any():
        raise ValueError(
            "Travel direction flags are simultaneously active."
        )

    if (
        output.loc[
            output["travel_missing_flag"].eq(0),
            [
                "miles_traveled_away",
                "time_zones_crossed_away",
            ],
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Travel fields missing while travel_missing_flag=0."
        )

    numeric_columns = [
        "temperature",
        "wind",
        "home_rest_days",
        "away_rest_days",
        "miles_traveled_away",
        "time_zones_crossed_away",
    ]

    matrix = (
        output[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64")
    )

    if np.isinf(matrix).any():
        raise ValueError("Issue 14 output contains infinity.")

    return output


CONFIG = common.load_config()


def main() -> None:
    paths = CONFIG["paths"]
    seasons = CONFIG["seasons"]

    games_path = resolve_repo_path(
        paths["historical_games"]
    )
    team_master_path = resolve_repo_path(
        paths["team_master"]
    )

    output_path = paths.get(
        "environment_history",
        "docs/win/football/nfl/prop_engine/data/historical/features/environment.parquet",
    )

    environment = build_environment(
        games_path,
        team_master_path,
        historical_start=int(seasons["historical_start"]),
        historical_end=int(seasons["historical_end"]),
    )

    common.write_parquet_atomic(
        environment,
        output_path,
    )

    payload = {
        "status": "passed",
        "output": str(output_path),
        "rows": int(len(environment)),
        "games": int(environment["game_id"].nunique()),
        "columns": int(len(environment.columns)),
        "weather_nonmissing": int(
            environment["weather_missing_flag"].eq(0).sum()
        ),
        "weather_missing": int(
            environment["weather_missing_flag"].eq(1).sum()
        ),
        "travel_nonmissing": int(
            environment["travel_missing_flag"].eq(0).sum()
        ),
        "travel_missing": int(
            environment["travel_missing_flag"].eq(1).sum()
        ),
        "neutral_site_games": int(
            environment["neutral_site_flag"].sum()
        ),
        "international_games": int(
            environment["international_flag"].sum()
        ),
        "travel_policy": (
            "mirror current build_travel.py: away franchise home coordinates "
            "to home franchise home coordinates; neutral/international status "
            "is represented separately"
        ),
        "weather_policy": (
            "historical temp/wind only; compatible with current-week "
            "temperature/wind_speed"
        ),
        "odds_directory_read": False,
        "embedded_game_betting_columns_loaded": False,
    }

    print(
        json.dumps(
            {
                "script": Path(__file__).name,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
