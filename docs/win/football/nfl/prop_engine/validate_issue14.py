from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import math
import re
import unicodedata

import numpy as np
import pandas as pd

NFL = Path("docs/win/football/nfl")

GAMES = NFL / "data/historic_data/games/games_2010_2025.csv"
TEAMS = NFL / "data/master/team_master.csv"
OUT = NFL / "prop_engine/data/historical/features/environment.parquet"

EXPECTED_COLUMNS = [
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

TEAM_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

TRAVEL_ALIASES = {
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

INTERNATIONAL_GAME_KEYS = {
    (2025, 1, "LAC", "KC"),
    (2025, 4, "PIT", "MIN"),
    (2025, 5, "CLE", "MIN"),
    (2025, 6, "NYJ", "DEN"),
    (2025, 7, "JAX", "LAR"),
    (2025, 10, "IND", "ATL"),
    (2025, 11, "MIA", "WSH"),
}


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""

    return text


def output_team(value: Any) -> str:
    key = clean(value).upper()
    return TEAM_ALIASES.get(key, key)


def travel_team(value: Any) -> str:
    key = output_team(value)
    return TRAVEL_ALIASES.get(key, key)


def stadium_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text.casefold(),
    )

    return " ".join(text.split())


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return EARTH_RADIUS_MILES * (
        2 * math.asin(math.sqrt(a))
    )


def utc_offset_hours(timezone_name: str, gameday: str) -> float:
    dt = datetime.strptime(
        gameday,
        "%Y-%m-%d",
    ).replace(
        tzinfo=ZoneInfo(timezone_name)
    )

    offset = dt.utcoffset()

    if offset is None:
        fail(
            f"no UTC offset for timezone={timezone_name!r}, "
            f"gameday={gameday!r}"
        )

    return offset.total_seconds() / 3600.0


def assert_same_numeric(actual, expected, label, rtol=1e-10, atol=1e-10):
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = np.asarray(expected, dtype="float64")

    ok = (
        (np.isnan(a) & np.isnan(e))
        |
        (
            np.isfinite(a)
            & np.isfinite(e)
            & np.isclose(a, e, rtol=rtol, atol=atol)
        )
    )

    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        sample = [
            {
                "row": int(i),
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in bad[:10]
        ]
        fail(f"{label}: {len(bad):,} mismatches; sample={sample}")


print("=" * 110)
print("ISSUE 14 ACCEPTANCE VALIDATION")
print("=" * 110)

out = pd.read_parquet(OUT)

if list(out.columns) != EXPECTED_COLUMNS:
    missing = [c for c in EXPECTED_COLUMNS if c not in out.columns]
    extra = [c for c in out.columns if c not in EXPECTED_COLUMNS]
    fail(
        "exact header/order mismatch; "
        f"missing={missing} extra={extra}"
    )

if len(out) != 3663:
    fail(f"expected 3,663 rows, got {len(out):,}")

if out.duplicated(["season", "week", "game_id"]).any():
    fail("duplicate season/week/game_id rows")

print("PASS: Exact 23-column contract, 3,663 rows, and canonical game grain.")

# ---------------------------------------------------------------------
# Reconstruct source rows using only non-betting columns.
# ---------------------------------------------------------------------

game_cols = [
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

games = pd.read_csv(
    GAMES,
    usecols=game_cols,
    low_memory=False,
)

games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
games["week"] = pd.to_numeric(games["week"], errors="raise").astype(int)

games = games.loc[
    games["season"].between(2012, 2025)
    & games["game_type"].astype(str).str.upper().eq("REG")
].copy()

games["home_team"] = games["home_team"].map(output_team)
games["away_team"] = games["away_team"].map(output_team)

games = games.sort_values(
    ["season", "week", "game_id"],
    kind="mergesort",
).reset_index(drop=True)

out = out.sort_values(
    ["season", "week", "game_id"],
    kind="mergesort",
).reset_index(drop=True)

for c in ["season", "week", "game_id", "gameday", "home_team", "away_team"]:
    if not out[c].astype(str).eq(games[c].astype(str)).all():
        fail(f"{c}: output differs from historical games source")

print("PASS: Output game identity/date/team fields exactly match regular-season schedule source.")

# ---------------------------------------------------------------------
# Direct field reconciliation.
# ---------------------------------------------------------------------

div_flag = pd.to_numeric(games["div_game"], errors="raise").astype(int)
neutral_flag = (
    games["location"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .map({"home": 0, "neutral": 1})
)

if neutral_flag.isna().any():
    fail("unsupported location value in historical games")

assert_same_numeric(
    out["divisional_game_flag"],
    div_flag,
    "divisional_game_flag",
    rtol=0,
    atol=0,
)

assert_same_numeric(
    out["neutral_site_flag"],
    neutral_flag,
    "neutral_site_flag",
    rtol=0,
    atol=0,
)

for c_src, c_out in [
    ("stadium", "stadium"),
    ("stadium_id", "stadium_id"),
    ("roof", "roof"),
]:
    left = games[c_src].fillna("").astype(str)
    right = out[c_out].fillna("").astype(str)

    if not right.eq(left).all():
        fail(f"{c_out}: output differs from source")

surface_expected = games["surface"].map(
    lambda value: clean(value)
)

surface_actual = out["surface"].map(
    lambda value: clean(value)
)

if not surface_actual.eq(surface_expected).all():
    fail("surface: normalized output differs from source")

assert_same_numeric(
    out["temperature"],
    pd.to_numeric(games["temp"], errors="coerce"),
    "temperature",
)

assert_same_numeric(
    out["wind"],
    pd.to_numeric(games["wind"], errors="coerce"),
    "wind",
)

assert_same_numeric(
    out["home_rest_days"],
    pd.to_numeric(games["home_rest"], errors="raise"),
    "home_rest_days",
)

assert_same_numeric(
    out["away_rest_days"],
    pd.to_numeric(games["away_rest"], errors="raise"),
    "away_rest_days",
)

weather_missing = (
    pd.to_numeric(games["temp"], errors="coerce").isna()
    | pd.to_numeric(games["wind"], errors="coerce").isna()
).astype(int)

assert_same_numeric(
    out["weather_missing_flag"],
    weather_missing,
    "weather_missing_flag",
    rtol=0,
    atol=0,
)

print("PASS: Divisional, neutral, stadium, roof, surface, weather, rest, and weather-missing fields are exact.")

# ---------------------------------------------------------------------
# Independent team-master travel reconstruction.
# ---------------------------------------------------------------------

team_cols = [
    "team_abbr",
    "latitude",
    "longitude",
    "timezone",
    "venue_country",
]

tm = pd.read_csv(
    TEAMS,
    usecols=team_cols,
    low_memory=False,
)

lookup = {}

for _, row in tm.iterrows():
    abbr = travel_team(row["team_abbr"])

    if not abbr:
        continue

    record = {
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "timezone": clean(row["timezone"]),
        "venue_country": clean(row["venue_country"]).upper(),
    }

    if abbr not in lookup:
        lookup[abbr] = record
    elif lookup[abbr] != record:
        fail(f"conflicting team-master travel records for {abbr}")

exp_miles = []
exp_zones = []
exp_e2w = []
exp_w2e = []
exp_missing = []

for row in games.itertuples(index=False):
    away = lookup.get(travel_team(row.away_team))
    home = lookup.get(travel_team(row.home_team))

    if away is None or home is None:
        exp_miles.append(np.nan)
        exp_zones.append(np.nan)
        exp_e2w.append(0)
        exp_w2e.append(0)
        exp_missing.append(1)
        continue

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

    zones = abs(home_offset - away_offset)

    away_lon = away["longitude"]
    home_lon = home["longitude"]

    if home_lon < away_lon:
        e2w, w2e = 1, 0
    elif home_lon > away_lon:
        e2w, w2e = 0, 1
    else:
        e2w, w2e = 0, 0

    exp_miles.append(distance)
    exp_zones.append(zones)
    exp_e2w.append(e2w)
    exp_w2e.append(w2e)
    exp_missing.append(0)

assert_same_numeric(
    out["miles_traveled_away"],
    exp_miles,
    "miles_traveled_away",
)

assert_same_numeric(
    out["time_zones_crossed_away"],
    exp_zones,
    "time_zones_crossed_away",
)

assert_same_numeric(
    out["east_to_west_flag"],
    exp_e2w,
    "east_to_west_flag",
    rtol=0,
    atol=0,
)

assert_same_numeric(
    out["west_to_east_flag"],
    exp_w2e,
    "west_to_east_flag",
    rtol=0,
    atol=0,
)

assert_same_numeric(
    out["travel_missing_flag"],
    exp_missing,
    "travel_missing_flag",
    rtol=0,
    atol=0,
)

print("PASS: Away travel miles, timezone crossings, direction, and travel-missing flags exactly match repository travel convention.")

# ---------------------------------------------------------------------
# International flag.
# ---------------------------------------------------------------------

exp_international = []

for row in games.itertuples(index=False):
    key = (
        int(row.season),
        int(row.week),
        clean(row.home_team),
        clean(row.away_team),
    )

    flag = int(
        key in INTERNATIONAL_GAME_KEYS
        or stadium_key(row.stadium) in INTERNATIONAL_STADIUMS
    )

    exp_international.append(flag)

assert_same_numeric(
    out["international_flag"],
    exp_international,
    "international_flag",
    rtol=0,
    atol=0,
)

print("PASS: International-game flags exactly match historical venue/game-key convention.")

# ---------------------------------------------------------------------
# Sanity.
# ---------------------------------------------------------------------

binary_cols = [
    "divisional_game_flag",
    "neutral_site_flag",
    "east_to_west_flag",
    "west_to_east_flag",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]

for c in binary_cols:
    if not out[c].isin([0, 1]).all():
        fail(f"{c}: contains non-binary values")

if (
    out["east_to_west_flag"]
    + out["west_to_east_flag"]
    > 1
).any():
    fail("east_to_west_flag and west_to_east_flag both active")

num_cols = [
    "temperature",
    "wind",
    "home_rest_days",
    "away_rest_days",
    "miles_traveled_away",
    "time_zones_crossed_away",
]

matrix = (
    out[num_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype="float64")
)

if np.isinf(matrix).any():
    fail("output contains infinity")

print("PASS: Binary and finite-value sanity checks.")

print("=" * 110)
print("ISSUE 14 ACCEPTANCE: PASS")
print("=" * 110)
