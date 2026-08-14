#!/usr/bin/env python3
"""
Step 15 NFL selection engine.

READS:
  docs/win/football/nfl/config/settings.yaml
  docs/win/football/nfl/config/markets.yaml
  docs/win/football/nfl/01_merge/week_{week}_NFL_enriched.csv
  docs/win/football/nfl/00_intake/schedule/weekly/
      week_{week}_NFL_weekly_schedule.csv
  docs/win/football/nfl/data/weather/
      week_{week}_NFL_weekly_weather.csv  (optional unless config requires it)

WRITES:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

The combined file must already contain the 10 projection/probability fields.
For each market, this script evaluates both possible sides using current
American odds, the model probability, implied probability, edge, expected
value, and Kelly fraction. It then keeps the best candidate that passes the
configured filters.

Global thresholds are defined in settings.yaml -> selection_defaults.
A market may override any threshold key in markets.yaml.

This script does not grade results.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_SETTINGS_PATH = NFL_ROOT / "config/settings.yaml"
DEFAULT_MARKETS_PATH = NFL_ROOT / "config/markets.yaml"

PREDICTION_COLUMNS = [
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
    "home_win_probability",
    "away_win_probability",
    "home_cover_probability",
    "away_cover_probability",
    "over_probability",
    "under_probability",
]

SELECTION_COLUMNS = [
    "ml_selected",
    "ml_selection",
    "ml_selection_reason",
    "ml_odds_american",
    "ml_model_probability",
    "ml_implied_probability",
    "ml_edge",
    "ml_ev",
    "ml_full_kelly",
    "ml_kelly",
    "spread_selected",
    "spread_selection",
    "spread_selection_reason",
    "spread_line",
    "spread_odds_american",
    "spread_model_probability",
    "spread_implied_probability",
    "spread_edge",
    "spread_ev",
    "spread_full_kelly",
    "spread_kelly",
    "total_selected",
    "total_selection",
    "total_selection_reason",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "total_implied_probability",
    "total_edge",
    "total_ev",
    "total_full_kelly",
    "total_kelly",
]

THRESHOLD_KEYS = [
    "min_ev",
    "min_edge",
    "min_kelly",
    "max_kelly",
    "min_odds_american",
    "max_odds_american",
    "min_model_prob",
    "max_model_prob",
]

SEASON_TYPE_ALIASES = {
    "reg": "reg",
    "regular": "reg",
    "regularseason": "reg",
    "pre": "pre",
    "preseason": "pre",
    "post": "post",
    "postseason": "post",
    "playoff": "post",
    "playoffs": "post",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def parse_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def parse_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    text = clean(value).casefold()
    if text in {"true", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "no", "n", "0", "off"}:
        return False
    fail(f"{key} must be true/false; found {value!r}")


def normalize_game_id(value: Any) -> str:
    return re.sub(r"\.0$", "", clean(value))


def normalize_season_type(value: Any) -> str:
    text = re.sub(r"[\s_-]+", "", clean(value).casefold())
    return SEASON_TYPE_ALIASES.get(text, text)


def normalize_bookmaker(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def read_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail(f"{label} must contain a YAML mapping: {path}")
    return data


def read_csv(
    path: Path,
    label: str,
    *,
    optional: bool = False,
) -> pd.DataFrame | None:
    if not path.is_file():
        if optional:
            return None
        fail(f"Missing {label}: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    if df.empty and not optional:
        fail(f"{label} contains no data rows: {path}")
    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def validate_unique_game_ids(df: pd.DataFrame, label: str) -> None:
    ids = df["game_id"].map(normalize_game_id)
    if ids.eq("").any():
        fail(f"{label} contains blank game_id values")
    if ids.duplicated().any():
        examples = ids[ids.duplicated(False)].head(10).tolist()
        fail(f"{label} contains duplicate game_id values: {examples}")
    df["game_id"] = ids


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        fail("American odds cannot be 0")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def calculate_metrics(
    model_probability: float,
    odds_american: float,
) -> dict[str, float]:
    if not 0.0 <= model_probability <= 1.0:
        fail(f"Model probability outside [0,1]: {model_probability}")

    decimal_odds = american_to_decimal(odds_american)
    implied_probability = 1.0 / decimal_odds
    edge = model_probability - implied_probability

    net_win = decimal_odds - 1.0
    loss_probability = 1.0 - model_probability
    ev = model_probability * net_win - loss_probability

    raw_kelly = (
        net_win * model_probability - loss_probability
    ) / net_win
    full_kelly = max(0.0, raw_kelly)

    return {
        "implied_probability": implied_probability,
        "edge": edge,
        "ev": ev,
        "full_kelly": full_kelly,
    }


def resolve_thresholds(
    settings: dict[str, Any],
    market_name: str,
    market_config: dict[str, Any],
) -> dict[str, float]:
    defaults = settings.get("selection_defaults")
    if not isinstance(defaults, dict):
        fail("settings.yaml must contain selection_defaults")

    resolved: dict[str, float] = {}
    for key in THRESHOLD_KEYS:
        value = market_config.get(key, defaults.get(key))
        parsed = parse_float(value)
        if parsed is None:
            fail(
                f"Missing/non-numeric {key!r} for {market_name}; "
                "define it in settings.yaml selection_defaults or markets.yaml"
            )
        resolved[key] = parsed

    if not -1.0 <= resolved["min_edge"] <= 1.0:
        fail(f"{market_name}.min_edge must be in [-1,1]")
    if resolved["min_kelly"] < 0 or resolved["max_kelly"] < 0:
        fail(f"{market_name} Kelly values cannot be negative")
    if resolved["min_kelly"] > resolved["max_kelly"]:
        fail(f"{market_name}.min_kelly cannot exceed max_kelly")
    if not 0 <= resolved["min_model_prob"] <= resolved["max_model_prob"] <= 1:
        fail(f"{market_name} model-probability limits are invalid")
    if resolved["min_odds_american"] > resolved["max_odds_american"]:
        fail(f"{market_name} American-odds limits are invalid")

    return resolved


def numeric_probability(row: pd.Series, column: str) -> float:
    value = parse_float(row[column])
    if value is None or not 0.0 <= value <= 1.0:
        fail(
            f"game_id={row['game_id']}: {column} must be a finite probability "
            f"in [0,1]; found {row[column]!r}"
        )
    return value


def odds_value(row: pd.Series, column: str) -> float | None:
    value = parse_float(row.get(column, ""))
    if value is None or value == 0:
        return None
    return value


def make_candidate(
    selection: str,
    model_probability: float,
    odds_american: float,
    *,
    line: float | None = None,
    is_favorite: bool = False,
    is_underdog: bool = False,
) -> dict[str, Any]:
    return {
        "selection": selection,
        "line": line,
        "odds_american": odds_american,
        "model_probability": model_probability,
        "is_favorite": is_favorite,
        "is_underdog": is_underdog,
        **calculate_metrics(model_probability, odds_american),
    }


def thresholds_pass(
    candidate: dict[str, Any],
    thresholds: dict[str, float],
) -> bool:
    return (
        thresholds["min_model_prob"]
        <= float(candidate["model_probability"])
        <= thresholds["max_model_prob"]
        and thresholds["min_odds_american"]
        <= float(candidate["odds_american"])
        <= thresholds["max_odds_american"]
        and float(candidate["edge"]) >= thresholds["min_edge"]
        and float(candidate["ev"]) >= thresholds["min_ev"]
        and float(candidate["full_kelly"]) >= thresholds["min_kelly"]
    )


def restrictions_pass(
    market_name: str,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    home_only = parse_bool(
        config.get("home_only", False),
        key=f"{market_name}.home_only",
    )
    away_only = parse_bool(
        config.get("away_only", False),
        key=f"{market_name}.away_only",
    )
    favorite_only = parse_bool(
        config.get("favorite_only", False),
        key=f"{market_name}.favorite_only",
    )
    underdog_only = parse_bool(
        config.get("underdog_only", False),
        key=f"{market_name}.underdog_only",
    )

    if home_only and away_only:
        fail(f"{market_name}: home_only and away_only cannot both be true")
    if favorite_only and underdog_only:
        fail(
            f"{market_name}: favorite_only and underdog_only "
            "cannot both be true"
        )

    side = str(candidate["selection"])
    if home_only and side != "HOME":
        return False
    if away_only and side != "AWAY":
        return False
    if favorite_only and not bool(candidate["is_favorite"]):
        return False
    if underdog_only and not bool(candidate["is_underdog"]):
        return False
    return True


def choose_best(
    candidates: list[dict[str, Any]],
    market_name: str,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any] | None:
    qualifying = [
        candidate
        for candidate in candidates
        if thresholds_pass(candidate, thresholds)
        and restrictions_pass(market_name, candidate, config)
    ]
    if not qualifying:
        return None

    qualifying.sort(
        key=lambda candidate: (
            float(candidate["ev"]),
            float(candidate["edge"]),
            float(candidate["model_probability"]),
            str(candidate["selection"]),
        ),
        reverse=True,
    )

    selected = qualifying[0].copy()
    selected["kelly"] = min(
        float(selected["full_kelly"]),
        thresholds["max_kelly"],
    )
    return selected


def market_enabled(config: dict[str, Any], name: str) -> bool:
    return parse_bool(config.get("enabled", True), key=f"{name}.enabled")


def empty_market(prefix: str, reason: str) -> dict[str, Any]:
    return {
        f"{prefix}_selected": 0,
        f"{prefix}_selection": "",
        f"{prefix}_selection_reason": reason,
        f"{prefix}_odds_american": np.nan,
        f"{prefix}_model_probability": np.nan,
        f"{prefix}_implied_probability": np.nan,
        f"{prefix}_edge": np.nan,
        f"{prefix}_ev": np.nan,
        f"{prefix}_full_kelly": np.nan,
        f"{prefix}_kelly": np.nan,
    }


def selected_market(
    prefix: str,
    candidate: dict[str, Any],
    *,
    include_line: bool,
) -> dict[str, Any]:
    output = {
        f"{prefix}_selected": 1,
        f"{prefix}_selection": candidate["selection"],
        f"{prefix}_selection_reason": "SELECTED",
        f"{prefix}_odds_american": candidate["odds_american"],
        f"{prefix}_model_probability": candidate["model_probability"],
        f"{prefix}_implied_probability": candidate["implied_probability"],
        f"{prefix}_edge": candidate["edge"],
        f"{prefix}_ev": candidate["ev"],
        f"{prefix}_full_kelly": candidate["full_kelly"],
        f"{prefix}_kelly": candidate["kelly"],
    }
    if include_line:
        output[f"{prefix}_line"] = candidate["line"]
    return output


def evaluate_moneyline(
    row: pd.Series,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not market_enabled(config, "moneyline"):
        return empty_market("ml", "MARKET_DISABLED")

    home_odds = odds_value(row, "sched_home_moneyline_american")
    away_odds = odds_value(row, "sched_away_moneyline_american")
    if home_odds is None or away_odds is None:
        return empty_market("ml", "CURRENT_LINE_MISSING")

    home_metrics = calculate_metrics(
        numeric_probability(row, "home_win_probability"),
        home_odds,
    )
    away_metrics = calculate_metrics(
        numeric_probability(row, "away_win_probability"),
        away_odds,
    )

    home_implied = home_metrics["implied_probability"]
    away_implied = away_metrics["implied_probability"]
    favorite_tie = math.isclose(
        home_implied,
        away_implied,
        rel_tol=0,
        abs_tol=1e-12,
    )

    candidates = [
        {
            "selection": "HOME",
            "line": None,
            "odds_american": home_odds,
            "model_probability": numeric_probability(
                row, "home_win_probability"
            ),
            "is_favorite": (not favorite_tie and home_implied > away_implied),
            "is_underdog": (not favorite_tie and home_implied < away_implied),
            **home_metrics,
        },
        {
            "selection": "AWAY",
            "line": None,
            "odds_american": away_odds,
            "model_probability": numeric_probability(
                row, "away_win_probability"
            ),
            "is_favorite": (not favorite_tie and away_implied > home_implied),
            "is_underdog": (not favorite_tie and away_implied < home_implied),
            **away_metrics,
        },
    ]

    chosen = choose_best(
        candidates,
        "moneyline",
        config,
        thresholds,
    )
    if chosen is None:
        return empty_market("ml", "NO_CANDIDATE_PASSED")
    return selected_market("ml", chosen, include_line=False)


def evaluate_spread(
    row: pd.Series,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not market_enabled(config, "spread"):
        result = empty_market("spread", "MARKET_DISABLED")
        result["spread_line"] = np.nan
        return result

    home_line = parse_float(row.get("sched_home_spread", ""))
    away_line = parse_float(row.get("sched_away_spread", ""))
    home_odds = odds_value(row, "sched_home_spread_american")
    away_odds = odds_value(row, "sched_away_spread_american")

    if any(
        value is None
        for value in [home_line, away_line, home_odds, away_odds]
    ):
        result = empty_market("spread", "CURRENT_LINE_MISSING")
        result["spread_line"] = np.nan
        return result

    max_spread_abs = parse_float(config.get("max_spread_abs", 100.0))
    if max_spread_abs is None or max_spread_abs < 0:
        fail("spread.max_spread_abs must be a non-negative number")

    candidates = [
        make_candidate(
            "HOME",
            numeric_probability(row, "home_cover_probability"),
            home_odds,
            line=home_line,
            is_favorite=home_line < 0,
            is_underdog=home_line > 0,
        ),
        make_candidate(
            "AWAY",
            numeric_probability(row, "away_cover_probability"),
            away_odds,
            line=away_line,
            is_favorite=away_line < 0,
            is_underdog=away_line > 0,
        ),
    ]

    candidates = [
        candidate
        for candidate in candidates
        if abs(float(candidate["line"])) <= max_spread_abs
    ]

    chosen = choose_best(
        candidates,
        "spread",
        config,
        thresholds,
    )
    if chosen is None:
        result = empty_market("spread", "NO_CANDIDATE_PASSED")
        result["spread_line"] = np.nan
        return result
    return selected_market("spread", chosen, include_line=True)


def roof_is_dome(row: pd.Series) -> bool:
    dome_flag = parse_int(row.get("wx_dome_flag", ""))
    if dome_flag is not None:
        return dome_flag == 1

    roof = clean(row.get("sched_roof", "")).casefold()
    return roof in {
        "dome",
        "indoor",
        "indoors",
        "closed",
        "retractable_closed",
        "retractable-closed",
    }


def roof_is_open_air(row: pd.Series) -> bool:
    open_flag = parse_int(row.get("wx_open_air_flag", ""))
    if open_flag is not None:
        return open_flag == 1

    roof = clean(row.get("sched_roof", "")).casefold()
    return roof in {
        "open_air",
        "open-air",
        "outdoor",
        "outdoors",
        "open",
    }


def weather_available(row: pd.Series) -> bool:
    return any(
        clean(row.get(column, ""))
        for column in [
            "wx_temperature",
            "wx_wind_speed",
            "wx_wind_gust",
            "wx_precip_probability",
            "wx_rain_flag",
            "wx_snow_flag",
        ]
    )


def total_environment_passes(
    row: pd.Series,
    config: dict[str, Any],
    game_filters: dict[str, Any],
) -> bool:
    dome_only = parse_bool(
        config.get("dome_only", False),
        key="total.dome_only",
    )
    open_air_only = parse_bool(
        config.get("open_air_only", False),
        key="total.open_air_only",
    )
    if dome_only and open_air_only:
        fail("total: dome_only and open_air_only cannot both be true")

    is_dome = roof_is_dome(row)
    is_open_air = roof_is_open_air(row)

    if dome_only and not is_dome:
        return False
    if open_air_only and not is_open_air:
        return False

    # Weather restrictions are ignored for a confirmed dome.
    if is_dome:
        return True

    allow_missing = parse_bool(
        config.get(
            "allow_missing_weather",
            game_filters.get("allow_weather_missing", True),
        ),
        key="total.allow_missing_weather",
    )

    if not weather_available(row):
        return allow_missing

    max_wind = parse_float(config.get("max_wind_speed", 999))
    max_gust = parse_float(config.get("max_gust_speed", 999))
    if max_wind is None or max_gust is None:
        fail("total max_wind_speed/max_gust_speed must be numeric")

    wind = parse_float(row.get("wx_wind_speed", ""))
    gust = parse_float(row.get("wx_wind_gust", ""))

    if wind is None and not allow_missing:
        return False
    if gust is None and not allow_missing:
        return False
    if wind is not None and wind > max_wind:
        return False
    if gust is not None and gust > max_gust:
        return False

    allow_precipitation = parse_bool(
        config.get("allow_precipitation", True),
        key="total.allow_precipitation",
    )
    if not allow_precipitation:
        rain = parse_int(row.get("wx_rain_flag", "")) or 0
        snow = parse_int(row.get("wx_snow_flag", "")) or 0
        if rain == 1 or snow == 1:
            return False

    return True


def evaluate_total(
    row: pd.Series,
    config: dict[str, Any],
    thresholds: dict[str, float],
    game_filters: dict[str, Any],
) -> dict[str, Any]:
    if not market_enabled(config, "total"):
        result = empty_market("total", "MARKET_DISABLED")
        result["total_line"] = np.nan
        return result

    total_line = parse_float(row.get("sched_total", ""))
    over_odds = odds_value(row, "sched_over_american")
    under_odds = odds_value(row, "sched_under_american")

    if total_line is None or over_odds is None or under_odds is None:
        result = empty_market("total", "CURRENT_LINE_MISSING")
        result["total_line"] = np.nan
        return result

    min_total = parse_float(config.get("min_total", 0.0))
    max_total = parse_float(config.get("max_total", 100.0))
    if min_total is None or max_total is None or min_total > max_total:
        fail("total min_total/max_total configuration is invalid")

    if not min_total <= total_line <= max_total:
        result = empty_market("total", "TOTAL_LINE_OUTSIDE_RANGE")
        result["total_line"] = total_line
        return result

    if not total_environment_passes(row, config, game_filters):
        result = empty_market("total", "TOTAL_ENVIRONMENT_FILTER")
        result["total_line"] = total_line
        return result

    candidates = [
        make_candidate(
            "OVER",
            numeric_probability(row, "over_probability"),
            over_odds,
            line=total_line,
        ),
        make_candidate(
            "UNDER",
            numeric_probability(row, "under_probability"),
            under_odds,
            line=total_line,
        ),
    ]

    chosen = choose_best(
        candidates,
        "total",
        config,
        thresholds,
    )
    if chosen is None:
        result = empty_market("total", "NO_CANDIDATE_PASSED")
        result["total_line"] = total_line
        return result
    return selected_market("total", chosen, include_line=True)


def validate_probability_pairs(df: pd.DataFrame) -> None:
    pairs = [
        (
            "home_win_probability",
            "away_win_probability",
            "moneyline",
        ),
        (
            "home_cover_probability",
            "away_cover_probability",
            "spread",
        ),
        (
            "over_probability",
            "under_probability",
            "total",
        ),
    ]

    for first, second, label in pairs:
        a = pd.to_numeric(df[first], errors="coerce")
        b = pd.to_numeric(df[second], errors="coerce")
        if a.isna().any() or b.isna().any():
            fail(f"{label} probability columns contain blank/non-numeric values")
        if ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
            fail(f"{label} probability outside [0,1]")
        if not np.allclose(
            a.to_numpy(dtype=float) + b.to_numpy(dtype=float),
            1.0,
            rtol=0,
            atol=1e-9,
        ):
            fail(
                f"{label} complementary probabilities do not sum to 1"
            )


def validate_settings(
    settings: dict[str, Any],
    season_override: int | None,
    week_override: int | None,
) -> tuple[int, int, str, str]:
    season = (
        season_override
        if season_override is not None
        else parse_int(settings.get("season"))
    )
    week = (
        week_override
        if week_override is not None
        else parse_int(settings.get("week"))
    )

    if season is None or season < 1900:
        fail(f"Invalid season: {settings.get('season')!r}")
    if week is None or week <= 0:
        fail(f"Invalid week: {settings.get('week')!r}")

    season_type = normalize_season_type(
        settings.get("season_type", "reg")
    )
    if season_type not in {"reg", "pre", "post"}:
        fail(
            f"Unsupported season_type: "
            f"{settings.get('season_type')!r}"
        )

    sportsbook = clean(settings.get("sportsbook"))
    if not sportsbook:
        fail("settings.yaml sportsbook is required")

    odds_format = clean(
        settings.get("odds_format", "american")
    ).casefold()
    if odds_format != "american":
        fail("selections.py requires odds_format: american")

    return season, week, season_type, sportsbook


def validate_combined(
    df: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    label: str,
) -> None:
    require_columns(
        df,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "away_team",
            "home_team",
            *PREDICTION_COLUMNS,
        ],
        label,
    )
    validate_unique_game_ids(df, label)

    seasons = {parse_int(value) for value in df["season"]}
    weeks = {parse_int(value) for value in df["week"]}
    types = {
        normalize_season_type(value)
        for value in df["season_type"]
    }

    if seasons != {season}:
        fail(
            f"{label}: expected only season={season}; "
            f"found {seasons}"
        )
    if weeks != {week}:
        fail(
            f"{label}: expected only week={week}; "
            f"found {weeks}"
        )
    if types != {season_type}:
        fail(
            f"{label}: expected season_type={season_type!r}; "
            f"found {types}"
        )

    validate_probability_pairs(df)


def merge_schedule(
    combined: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    sportsbook: str,
) -> pd.DataFrame:
    require_columns(
        schedule,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "neutral_site",
            "roof",
            "bookmaker",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
            "odds_available",
        ],
        "weekly schedule",
    )
    validate_unique_game_ids(schedule, "weekly schedule")

    season_values = pd.to_numeric(
        schedule["season"], errors="coerce"
    )
    week_values = pd.to_numeric(
        schedule["week"], errors="coerce"
    )
    type_values = schedule["season_type"].map(
        normalize_season_type
    )

    schedule = schedule.loc[
        (season_values == season)
        & (week_values == week)
        & (type_values == season_type)
    ].copy()

    if schedule.empty:
        fail(
            f"Weekly schedule has no rows for "
            f"season={season}, week={week}, season_type={season_type}"
        )

    configured_book = normalize_bookmaker(sportsbook)
    odds_available = pd.to_numeric(
        schedule["odds_available"], errors="coerce"
    ).fillna(0)
    available_rows = odds_available.eq(1)
    bad_book = (
        schedule["bookmaker"].map(normalize_bookmaker)
        .ne(configured_book)
        & available_rows
    )
    if bad_book.any():
        examples = schedule.loc[
            bad_book, ["game_id", "bookmaker"]
        ].head(10).to_dict("records")
        fail(
            f"Weekly schedule bookmaker does not match "
            f"settings sportsbook {sportsbook!r}: {examples}"
        )

    base_ids = set(combined["game_id"])
    schedule_ids = set(schedule["game_id"])
    missing = sorted(base_ids - schedule_ids)
    if missing:
        fail(
            f"Weekly schedule missing {len(missing)} projected games; "
            f"examples={missing[:10]}"
        )

    columns = [
        "game_id",
        "neutral_site",
        "roof",
        "bookmaker",
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
        "odds_available",
    ]
    source = schedule[columns].copy()
    source = source.rename(
        columns={
            column: f"sched_{column}"
            for column in columns
            if column != "game_id"
        }
    )

    return combined.merge(
        source,
        on="game_id",
        how="left",
        validate="one_to_one",
    )


def merge_weather(
    working: pd.DataFrame,
    weather: pd.DataFrame | None,
) -> pd.DataFrame:
    if weather is None or weather.empty:
        return working

    require_columns(weather, ["game_id"], "weekly weather")
    validate_unique_game_ids(weather, "weekly weather")

    source = weather.rename(
        columns={
            column: f"wx_{column}"
            for column in weather.columns
            if column != "game_id"
        }
    )
    return working.merge(
        source,
        on="game_id",
        how="left",
        validate="one_to_one",
    )


def global_eligibility(
    row: pd.Series,
    settings: dict[str, Any],
) -> tuple[bool, str]:
    game_filters = settings.get("game_filters", {})
    if not isinstance(game_filters, dict):
        fail("settings.yaml game_filters must be a mapping")

    allow_playoffs = parse_bool(
        game_filters.get("allow_playoffs", False),
        key="game_filters.allow_playoffs",
    )
    if normalize_season_type(row["season_type"]) == "post":
        if not allow_playoffs:
            return False, "PLAYOFFS_DISABLED"

    allow_neutral = parse_bool(
        game_filters.get("allow_neutral_site", True),
        key="game_filters.allow_neutral_site",
    )
    if (parse_int(row.get("sched_neutral_site", "")) or 0) == 1:
        if not allow_neutral:
            return False, "NEUTRAL_SITE_DISABLED"

    allow_dome = parse_bool(
        game_filters.get("allow_dome_games", True),
        key="game_filters.allow_dome_games",
    )
    if roof_is_dome(row) and not allow_dome:
        return False, "DOME_GAMES_DISABLED"

    return True, ""


def build_output(
    original: pd.DataFrame,
    working: pd.DataFrame,
    settings: dict[str, Any],
    market_document: dict[str, Any],
) -> pd.DataFrame:
    markets = market_document.get("markets")
    if not isinstance(markets, dict):
        fail("markets.yaml must contain a markets mapping")

    missing = [
        name
        for name in ["moneyline", "spread", "total"]
        if not isinstance(markets.get(name), dict)
    ]
    if missing:
        fail(f"markets.yaml missing market sections: {missing}")

    ml_config = markets["moneyline"]
    spread_config = markets["spread"]
    total_config = markets["total"]

    ml_thresholds = resolve_thresholds(
        settings, "moneyline", ml_config
    )
    spread_thresholds = resolve_thresholds(
        settings, "spread", spread_config
    )
    total_thresholds = resolve_thresholds(
        settings, "total", total_config
    )

    game_filters = settings.get("game_filters", {})
    selection_rows: list[dict[str, Any]] = []

    for _, row in working.iterrows():
        eligible, reason = global_eligibility(row, settings)

        if not eligible:
            ml = empty_market("ml", reason)
            spread = empty_market("spread", reason)
            spread["spread_line"] = np.nan
            total = empty_market("total", reason)
            total["total_line"] = np.nan
        elif (parse_int(row.get("sched_odds_available", "")) or 0) != 1:
            ml = empty_market(
                "ml", "CURRENT_ODDS_UNAVAILABLE"
            )
            spread = empty_market(
                "spread", "CURRENT_ODDS_UNAVAILABLE"
            )
            spread["spread_line"] = np.nan
            total = empty_market(
                "total", "CURRENT_ODDS_UNAVAILABLE"
            )
            total["total_line"] = np.nan
        else:
            ml = evaluate_moneyline(
                row, ml_config, ml_thresholds
            )
            spread = evaluate_spread(
                row, spread_config, spread_thresholds
            )
            total = evaluate_total(
                row,
                total_config,
                total_thresholds,
                game_filters,
            )

        selection_rows.append(
            {"game_id": row["game_id"], **ml, **spread, **total}
        )

    selection_frame = pd.DataFrame(
        selection_rows,
        columns=["game_id", *SELECTION_COLUMNS],
    )
    if len(selection_frame) != len(original):
        fail("Internal selection row-count mismatch")

    validate_unique_game_ids(selection_frame, "selection results")
    original_ids = set(original["game_id"])
    selection_ids = set(selection_frame["game_id"])
    if selection_ids != original_ids:
        missing_ids = sorted(original_ids - selection_ids)
        extra_ids = sorted(selection_ids - original_ids)
        fail(
            "Selection game_id mismatch: "
            f"missing={missing_ids[:10]} extra={extra_ids[:10]}"
        )

    # Reorder selection results by game_id to exactly match the projected
    # input. Never attach selections by row position because pandas merges
    # are not a safe ordering contract.
    selection_frame = original[["game_id"]].merge(
        selection_frame,
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    output = original.copy()
    for column in SELECTION_COLUMNS:
        output[column] = selection_frame[column].to_numpy()

    return output


def write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=DEFAULT_MARKETS_PATH,
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = read_yaml(
        args.settings.resolve(),
        "settings config",
    )
    market_document = read_yaml(
        args.markets.resolve(),
        "markets config",
    )

    season, week, season_type, sportsbook = validate_settings(
        settings,
        args.season,
        args.week,
    )

    input_path = (
        args.input.resolve()
        if args.input is not None
        else NFL_ROOT
        / "01_merge"
        / f"week_{week}_NFL_enriched.csv"
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else NFL_ROOT
        / "02_select"
        / f"week_{week}_NFL_selected.csv"
    )
    if output_path == input_path:
        fail(
            "Selection output path must differ from the input path; "
            "selections.py will not overwrite a file it reads."
        )

    combined = read_csv(
        input_path,
        "projected combined enriched file",
    )
    assert combined is not None

    # The selection step is intentionally re-runnable as odds/config change.
    # Remove only prior columns owned by this script, then rebuild them.
    prior_selection_columns = [
        column
        for column in SELECTION_COLUMNS
        if column in combined.columns
    ]
    if prior_selection_columns:
        combined = combined.drop(
            columns=prior_selection_columns
        )

    validate_combined(
        combined,
        season,
        week,
        season_type,
        str(input_path),
    )

    schedule_path = (
        NFL_ROOT
        / "00_intake/schedule/weekly"
        / f"week_{week}_NFL_weekly_schedule.csv"
    )
    schedule = read_csv(schedule_path, "weekly schedule")
    assert schedule is not None

    working = merge_schedule(
        combined.copy(),
        schedule,
        season,
        week,
        season_type,
        sportsbook,
    )

    weather_path = (
        NFL_ROOT
        / "data/weather"
        / f"week_{week}_NFL_weekly_weather.csv"
    )
    weather = read_csv(
        weather_path,
        "weekly weather",
        optional=True,
    )
    working = merge_weather(working, weather)

    output = build_output(
        combined,
        working,
        settings,
        market_document,
    )

    if list(output.columns) != list(combined.columns) + SELECTION_COLUMNS:
        fail("Final selection column order/integrity check failed")
    if output["game_id"].tolist() != combined["game_id"].tolist():
        fail("game_id order changed during selection processing")
    if output["away_team"].tolist() != combined["away_team"].tolist():
        fail("away_team changed during selection processing")
    if output["home_team"].tolist() != combined["home_team"].tolist():
        fail("home_team changed during selection processing")

    write_atomic_csv(output, output_path)

    print(
        f"Step 15 selections complete: season={season} "
        f"week={week} games={len(output)}"
    )
    for column, label in [
        ("ml_selected", "moneyline"),
        ("spread_selected", "spread"),
        ("total_selected", "total"),
    ]:
        count = int(
            pd.to_numeric(
                output[column], errors="coerce"
            ).fillna(0).sum()
        )
        print(f"{label}_selections={count}")
    print(f"Updated: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
