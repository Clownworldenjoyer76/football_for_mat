#!/usr/bin/env python3
"""
Standalone Week 1 NFL model projection.


Confirmed Week 1 sources used:
  docs/win/football/nfl/00_intake/predictions/enriched/combined/week_1_NFL_enriched.csv
  docs/win/football/nfl/00_intake/schedule/weekly/week_1_NFL_weekly_schedule.csv
  docs/win/football/nfl/00_intake/schedule/{SEASON}_schedule.csv
  docs/win/football/nfl/00_intake/predictions/drat/clean/{SEASON}_week_1_drat.csv
  docs/win/football/nfl/00_intake/predictions/final/{SEASON}_{season_type}_1_clean_predictions.csv
  docs/win/football/nfl/00_intake/team_stats/{SEASON-1}_team_stats.csv
  docs/win/football/nfl/00_intake/qb/{SEASON-1}_qb_stats.csv
  docs/win/football/nfl/00_intake/injuries/{SEASON}_injuries.csv
  docs/win/football/nfl/data/master/depth_charts/{TEAM}/{TEAM}_depth.csv
  docs/win/football/nfl/data/historic_data/players/players.parquet
  docs/win/football/nfl/data/master/league_master.csv
  docs/win/football/nfl/data/weather/week_1_NFL_weekly_weather.csv
  docs/win/football/nfl/data/travel/{SEASON}_week_1_travel.csv
  docs/win/football/nfl/models/step11_feature_schema.json
  docs/win/football/nfl/models/step11_margin_model.cbm
  docs/win/football/nfl/models/step11_total_points_model.cbm
  docs/win/football/nfl/models/step14_probability_calibration.json

Only confirmed file columns and confirmed Week 1 logic are used. Model features
whose construction is not confirmed here are represented as missing values
rather than guessed.

WRITES ONLY:
  docs/win/football/nfl/01_merge/week_1_NFL_enriched.csv
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================================
# CHANGE THIS EACH SEASON
# ============================================================================
SEASON = 2026
# ============================================================================

WEEK = 1
MISSING_CAT = "__MISSING__"

TEAM_METRICS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

QB_METRICS = [
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

INJURY_FEATURES = [
    "inj_out_count",
    "inj_doubtful_count",
    "inj_questionable_count",
    "inj_starter_out_count",
    "inj_top2_depth_out_count",
    "inj_qb1_out",
    "inj_ol_starter_out_count",
    "inj_skill_starter_out_count",
    "inj_front7_starter_out_count",
    "inj_secondary_starter_out_count",
    "inj_offense_unavailable_snap_share",
    "inj_defense_unavailable_snap_share",
    "depth_starter_changes",
]


EXPECTED_FEATURE_COUNT = 260


def repo_root() -> Path:
    """
    Expected file location:
      <repo>/docs/win/football/nfl/scripts/01_merge/projection_week1.py
    """
    here = Path(__file__).resolve()
    try:
        return here.parents[6]
    except IndexError as exc:
        raise RuntimeError(
            f"Cannot resolve repository root from script path: {here}"
        ) from exc


def nfl_root() -> Path:
    root = repo_root() / "docs/win/football/nfl"
    if not root.exists():
        raise FileNotFoundError(f"NFL root does not exist: {root}")
    return root


def clean(value: object) -> str:
    if value is None:
        return ""

    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""

    return text


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(text.split())


def normalize_team(value: object) -> str:
    return clean(value).upper()


def normalize_position(value: object) -> str:
    return clean(value).upper().replace(" ", "")


def parse_float(value: object) -> float | None:
    text = clean(value)

    if not text:
        return None

    text = text.replace(",", "").replace("%", "")

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def parse_int(value: object) -> int | None:
    number = parse_float(value)

    if number is None:
        return None

    rounded = round(number)

    if abs(number - rounded) > 1e-9:
        return None

    return int(rounded)


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def read_csv(path: Path, *, allow_empty_rows: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(df.columns) != len(set(df.columns)):
        raise ValueError(f"{path}: duplicate column names")

    if not allow_empty_rows and df.empty:
        raise ValueError(f"{path}: no data rows")

    if "game_id" in df.columns:
        df["game_id"] = normalize_game_id(df["game_id"])

    return df


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read parquet file {path}: {exc}") from exc


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def require_unique_game_id(df: pd.DataFrame, label: str) -> None:
    require_columns(df, ["game_id"], label)

    duplicates = (
        df.loc[
            df["game_id"].duplicated(keep=False),
            "game_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if duplicates:
        raise ValueError(
            f"{label}: duplicate game_id values: {duplicates[:10]}"
        )


def merge_by_game_id(
    base: pd.DataFrame,
    source: pd.DataFrame,
    source_columns: list[str],
    label: str,
) -> pd.DataFrame:
    require_unique_game_id(base, "base")
    require_unique_game_id(source, label)
    require_columns(source, ["game_id", *source_columns], label)

    source_ids = set(source["game_id"].dropna())
    missing_ids = [
        game_id
        for game_id in base["game_id"]
        if game_id not in source_ids
    ]

    if missing_ids:
        raise ValueError(
            f"{label}: missing game_id rows required by Week 1 base: "
            f"{missing_ids[:10]}"
        )

    overlapping = [
        column
        for column in source_columns
        if column in base.columns
    ]

    if overlapping:
        base = base.drop(columns=overlapping)

    return base.merge(
        source[["game_id", *source_columns]].copy(),
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )


def load_schema(root: Path) -> dict:
    path = root / "models/step11_feature_schema.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    required_keys = [
        "feature_order",
        "numeric_features",
        "categorical_features",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in schema
    ]

    if missing_keys:
        raise ValueError(f"{path}: missing schema keys: {missing_keys}")

    feature_order = list(schema["feature_order"])
    numeric_features = set(schema["numeric_features"])
    categorical_features = set(schema["categorical_features"])

    if len(feature_order) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURE_COUNT} features; "
            f"found {len(feature_order)}"
        )

    if len(feature_order) != len(set(feature_order)):
        raise ValueError(f"{path}: duplicate feature names in feature_order")

    if numeric_features & categorical_features:
        raise ValueError(
            f"{path}: a feature is classified as both numeric and categorical"
        )

    if numeric_features | categorical_features != set(feature_order):
        raise ValueError(
            f"{path}: numeric_features and categorical_features do not "
            "exactly cover feature_order"
        )

    return schema


def validate_week1_base(
    base: pd.DataFrame,
    season: int,
    label: str,
) -> None:
    require_columns(
        base,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ],
        label,
    )

    require_unique_game_id(base, label)

    seasons = {
        parse_int(value)
        for value in base["season"]
    }

    weeks = {
        parse_int(value)
        for value in base["week"]
    }

    if seasons != {season}:
        raise ValueError(
            f"{label}: expected only season={season}; found {seasons}"
        )

    if weeks != {WEEK}:
        raise ValueError(
            f"{label}: expected only week={WEEK}; found {weeks}"
        )


def add_market_features(
    base: pd.DataFrame,
    weekly_schedule: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    require_columns(
        weekly_schedule,
        [
            "game_id",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
        ],
        label,
    )

    source = weekly_schedule[
        [
            "game_id",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
        ]
    ].copy()

    source = source.rename(
        columns={
            "home_moneyline_american": "home_moneyline",
            "away_moneyline_american": "away_moneyline",
            "home_spread": "spread_line",
            "home_spread_american": "home_spread_odds",
            "away_spread_american": "away_spread_odds",
            "total": "total_line",
            "over_american": "over_odds",
            "under_american": "under_odds",
        }
    )

    base = merge_by_game_id(
        base,
        source,
        [
            "home_moneyline",
            "away_moneyline",
            "spread_line",
            "away_spread",
            "home_spread_odds",
            "away_spread_odds",
            "total_line",
            "over_odds",
            "under_odds",
        ],
        label,
    )

    return base


def add_drat_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    path = (
        root
        / "00_intake/predictions/drat/clean"
        / f"{season}_week_1_drat.csv"
    )

    drat = read_csv(path)

    require_columns(
        drat,
        [
            "game_id",
            "away_prob",
            "home_prob",
            "moneyline_away",
            "moneyline_home",
            "spread_away",
            "spread_home",
        ],
        str(path),
    )

    source = drat[
        [
            "game_id",
            "away_prob",
            "home_prob",
            "moneyline_away",
            "moneyline_home",
            "spread_away",
            "spread_home",
        ]
    ].rename(
        columns={
            "away_prob": "drat_away_prob",
            "home_prob": "drat_home_prob",
            "moneyline_away": "drat_away_moneyline",
            "moneyline_home": "drat_home_moneyline",
            "spread_away": "drat_away_spread",
            "spread_home": "drat_home_spread",
        }
    )

    return merge_by_game_id(
        base,
        source,
        [
            "drat_away_prob",
            "drat_home_prob",
            "drat_away_moneyline",
            "drat_home_moneyline",
            "drat_away_spread",
            "drat_home_spread",
        ],
        str(path),
    )


def add_epred_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    season_types = [
        clean(value)
        for value in base["season_type"].dropna().unique()
        if clean(value)
    ]

    if len(set(season_types)) != 1:
        raise ValueError(
            f"Week 1 base must contain exactly one season_type; "
            f"found {season_types}"
        )

    season_type = season_types[0].casefold()

    path = (
        root
        / "00_intake/predictions/final"
        / f"{season}_{season_type}_1_clean_predictions.csv"
    )

    epred = read_csv(path)

    require_columns(
        epred,
        [
            "game_id",
            "matchupQuality",
            "home_prob",
            "away_prob",
            "tie_prob",
            "away_projected_pts",
            "home_projected_pts",
            "total_projected_pts",
            "home_PtDiff",
            "away_PtDiff",
            "home_rating",
            "away_rating",
        ],
        str(path),
    )

    source = epred[
        [
            "game_id",
            "matchupQuality",
            "home_prob",
            "away_prob",
            "tie_prob",
            "away_projected_pts",
            "home_projected_pts",
            "total_projected_pts",
            "home_PtDiff",
            "away_PtDiff",
            "home_rating",
            "away_rating",
        ]
    ].rename(
        columns={
            "matchupQuality": "epred_matchupQuality",
            "home_prob": "epred_home_prob",
            "away_prob": "epred_away_prob",
            "tie_prob": "epred_tie_prob",
            "away_projected_pts": "epred_away_projected_pts",
            "home_projected_pts": "epred_home_projected_pts",
            "total_projected_pts": "epred_total_projected_pts",
            "home_PtDiff": "epred_home_PtDiff",
            "away_PtDiff": "epred_away_PtDiff",
            "home_rating": "epred_home_rating",
            "away_rating": "epred_away_rating",
        }
    )

    return merge_by_game_id(
        base,
        source,
        [
            "epred_matchupQuality",
            "epred_home_prob",
            "epred_away_prob",
            "epred_tie_prob",
            "epred_away_projected_pts",
            "epred_home_projected_pts",
            "epred_total_projected_pts",
            "epred_home_PtDiff",
            "epred_away_PtDiff",
            "epred_home_rating",
            "epred_away_rating",
        ],
        str(path),
    )


def add_schedule_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    path = root / "00_intake/schedule" / f"{season}_schedule.csv"
    schedule = read_csv(path)

    require_columns(
        schedule,
        [
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
        ],
        str(path),
    )

    season_number = pd.to_numeric(schedule["season"], errors="coerce")
    week_number = pd.to_numeric(schedule["week"], errors="coerce")

    week1 = schedule[
        (season_number == season)
        & (week_number == WEEK)
    ].copy()

    require_unique_game_id(week1, f"{path} season={season} week=1")

    source = week1[
        [
            "game_id",
            "season_type",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "neutral_site",
            "stadium",
            "roof",
            "surface",
        ]
    ].rename(
        columns={
            "season_type": "sched_season_type",
            "game_date": "sched_game_date",
            "game_time": "sched_game_time",
            "away_team": "sched_away_team",
            "home_team": "sched_home_team",
            "neutral_site": "sched_neutral_site",
            "stadium": "sched_stadium",
            "roof": "sched_roof",
            "surface": "sched_surface",
        }
    )

    base = merge_by_game_id(
        base,
        source,
        [
            "sched_season_type",
            "sched_game_date",
            "sched_game_time",
            "sched_away_team",
            "sched_home_team",
            "sched_neutral_site",
            "sched_stadium",
            "sched_roof",
            "sched_surface",
        ],
        str(path),
    )

    base["game_type"] = base["sched_season_type"]
    base["week"] = "1"
    base["gametime"] = base["sched_game_time"]
    base["away_team"] = base["sched_away_team"]
    base["home_team"] = base["sched_home_team"]
    base["roof"] = base["sched_roof"]
    base["surface"] = base["sched_surface"]
    base["stadium"] = base["sched_stadium"]

    parsed_dates = pd.to_datetime(
        base["sched_game_date"],
        errors="coerce",
    )

    base["weekday"] = parsed_dates.dt.day_name()

    return base


def add_division_feature(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    path = root / "data/master/league_master.csv"
    league = read_csv(path)

    require_columns(
        league,
        [
            "team_abbr",
            "division",
            "season",
        ],
        str(path),
    )

    season_number = pd.to_numeric(
        league["season"],
        errors="coerce",
    )

    rows = league[season_number == season].copy()

    division_by_team = {
        normalize_team(row["team_abbr"]): clean(row["division"])
        for _, row in rows.iterrows()
        if normalize_team(row["team_abbr"])
    }

    div_game: list[float] = []

    for _, row in base.iterrows():
        home = normalize_team(row["home_team"])
        away = normalize_team(row["away_team"])

        home_division = division_by_team.get(home, "")
        away_division = division_by_team.get(away, "")

        if not home_division or not away_division:
            div_game.append(np.nan)
        else:
            div_game.append(
                1.0
                if home_division == away_division
                else 0.0
            )

    base["div_game"] = div_game

    return base


def add_weather_features(
    base: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    path = root / "data/weather/week_1_NFL_weekly_weather.csv"
    weather = read_csv(path)

    require_columns(
        weather,
        [
            "game_id",
            "temperature",
            "wind_speed",
        ],
        str(path),
    )

    source = weather[
        [
            "game_id",
            "temperature",
            "wind_speed",
        ]
    ].rename(
        columns={
            "temperature": "temp",
            "wind_speed": "wind",
        }
    )

    return merge_by_game_id(
        base,
        source,
        [
            "temp",
            "wind",
        ],
        str(path),
    )


def add_travel_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    path = root / "data/travel" / f"{season}_week_1_travel.csv"
    travel = read_csv(path)

    travel_columns = [
        "miles_traveled",
        "time_zones_crossed",
        "east_to_west",
        "west_to_east",
        "international_flag",
        "neutral_site_flag",
    ]

    require_columns(
        travel,
        ["game_id", *travel_columns],
        str(path),
    )

    return merge_by_game_id(
        base,
        travel[["game_id", *travel_columns]].copy(),
        travel_columns,
        str(path),
    )


def load_prior_team_stats(
    root: Path,
    season: int,
) -> dict[str, dict[str, float | None]]:
    prior_season = season - 1
    path = (
        root
        / "00_intake/team_stats"
        / f"{prior_season}_team_stats.csv"
    )

    stats = read_csv(path)

    require_columns(
        stats,
        [
            "season",
            "week",
            "team",
            *TEAM_METRICS,
        ],
        str(path),
    )

    stats["_season_num"] = pd.to_numeric(
        stats["season"],
        errors="coerce",
    )

    stats["_week_num"] = pd.to_numeric(
        stats["week"],
        errors="coerce",
    )

    stats = stats[
        stats["_season_num"] == prior_season
    ].copy()

    result: dict[str, dict[str, float | None]] = {}

    for team, group in stats.groupby(
        stats["team"].map(normalize_team),
        sort=False,
    ):
        valid = group[group["_week_num"].notna()].copy()

        if valid.empty:
            continue

        latest_week = valid["_week_num"].max()
        latest_rows = valid[
            valid["_week_num"] == latest_week
        ]

        if len(latest_rows) != 1:
            raise ValueError(
                f"{path}: duplicate final-week rows for team={team}, "
                f"week={int(latest_week)}"
            )

        row = latest_rows.iloc[0]

        result[team] = {
            metric: parse_float(row[metric])
            for metric in TEAM_METRICS
        }

    return result


def add_prior_team_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    prior_stats = load_prior_team_stats(
        root,
        season,
    )

    for metric in TEAM_METRICS:
        home_values: list[float | None] = []
        away_values: list[float | None] = []
        diff_values: list[float | None] = []

        for _, row in base.iterrows():
            home = normalize_team(row["home_team"])
            away = normalize_team(row["away_team"])

            home_row = prior_stats.get(home)
            away_row = prior_stats.get(away)

            if home_row is None or away_row is None:
                raise ValueError(
                    "Prior-season team performance fallback could not be "
                    f"matched for game_id={row['game_id']} "
                    f"home={home} away={away}"
                )

            home_value = home_row[metric]
            away_value = away_row[metric]

            home_values.append(home_value)
            away_values.append(away_value)

            if home_value is None or away_value is None:
                diff_values.append(None)
            else:
                diff_values.append(home_value - away_value)

        base[f"home_{metric}"] = home_values
        base[f"away_{metric}"] = away_values
        base[f"{metric}_diff"] = diff_values

    return base


@dataclass(frozen=True)
class PlayerRecord:
    gsis_id: str
    espn_id: str
    name: str


class PlayerCrosswalk:
    def __init__(self, players: pd.DataFrame) -> None:
        require_columns(
            players,
            ["gsis_id"],
            "players.parquet",
        )

        espn_col = "espn_id" if "espn_id" in players.columns else None

        name_col = None
        for candidate in [
            "display_name",
            "full_name",
            "player_name",
        ]:
            if candidate in players.columns:
                name_col = candidate
                break

        self.by_espn: dict[str, PlayerRecord] = {}
        self.by_gsis: dict[str, PlayerRecord] = {}
        name_candidates: dict[str, list[PlayerRecord]] = {}

        for _, row in players.iterrows():
            gsis_id = clean(row["gsis_id"])

            if not gsis_id:
                continue

            espn_id = (
                clean(row[espn_col])
                if espn_col
                else ""
            )

            name = (
                clean(row[name_col])
                if name_col
                else ""
            )

            record = PlayerRecord(
                gsis_id=gsis_id,
                espn_id=espn_id,
                name=name,
            )

            self.by_gsis[gsis_id] = record

            if espn_id:
                self.by_espn[espn_id] = record

            name_key = normalize_name(name)

            if name_key:
                name_candidates.setdefault(
                    name_key,
                    [],
                ).append(record)

        self.by_unique_name = {
            key: rows[0]
            for key, rows in name_candidates.items()
            if len({row.gsis_id for row in rows}) == 1
        }

    def to_gsis(
        self,
        player_id: object,
        player_name: object,
    ) -> str:
        raw_id = clean(player_id)

        if raw_id in self.by_gsis:
            return raw_id

        if raw_id in self.by_espn:
            return self.by_espn[raw_id].gsis_id

        name_key = normalize_name(player_name)

        if name_key in self.by_unique_name:
            return self.by_unique_name[name_key].gsis_id

        return ""


def load_current_qb1(
    root: Path,
    players: PlayerCrosswalk,
) -> dict[str, str]:
    path = root / "config/mapping/qb_map_nfl.csv"
    qb_map = read_csv(path)

    require_columns(
        qb_map,
        [
            "player_id",
            "qb_name",
            "team_abbr",
            "starter_flag",
            "position_abb",
        ],
        str(path),
    )

    starters: dict[str, str] = {}

    for _, row in qb_map.iterrows():
        if parse_int(row["starter_flag"]) != 1:
            continue

        if normalize_position(row["position_abb"]) != "QB":
            continue

        team = normalize_team(row["team_abbr"])

        if not team:
            continue

        gsis_id = players.to_gsis(
            row["player_id"],
            row["qb_name"],
        )

        if team in starters and starters[team] != gsis_id:
            raise ValueError(
                f"{path}: multiple QB1 rows for team={team}"
            )

        starters[team] = gsis_id

    if not starters:
        raise RuntimeError(f"{path}: no QB1 rows found")

    return starters


def load_prior_qb_stats(
    root: Path,
    season: int,
    qb1_by_team: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    prior_season = season - 1
    path = (
        root
        / "00_intake/qb"
        / f"{prior_season}_qb_stats.csv"
    )

    qb_stats = read_csv(path)

    require_columns(
        qb_stats,
        [
            "season",
            "week",
            "player_id",
            "dropbacks",
            *QB_METRICS,
        ],
        str(path),
    )

    qb_stats["_season_num"] = pd.to_numeric(
        qb_stats["season"],
        errors="coerce",
    )

    qb_stats["_week_num"] = pd.to_numeric(
        qb_stats["week"],
        errors="coerce",
    )

    qb_stats["_dropbacks_num"] = pd.to_numeric(
        qb_stats["dropbacks"],
        errors="coerce",
    ).fillna(-1.0)

    qb_stats["_player_id_clean"] = (
        qb_stats["player_id"]
        .astype("string")
        .str.strip()
    )

    qb_stats = qb_stats[
        qb_stats["_season_num"] == prior_season
    ].copy()

    result: dict[str, dict[str, float | None]] = {}

    for team, gsis_id in qb1_by_team.items():
        if not gsis_id:
            result[team] = {
                metric: None
                for metric in QB_METRICS
            }
            continue

        rows = qb_stats[
            (qb_stats["_player_id_clean"] == gsis_id)
            & qb_stats["_week_num"].notna()
        ].copy()

        if rows.empty:
            result[team] = {
                metric: None
                for metric in QB_METRICS
            }
            continue

        latest_week = rows["_week_num"].max()

        rows = rows[
            rows["_week_num"] == latest_week
        ].sort_values(
            "_dropbacks_num",
            kind="stable",
        )

        row = rows.iloc[-1]

        result[team] = {
            metric: parse_float(row[metric])
            for metric in QB_METRICS
        }

    return result


def add_prior_qb_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    players_path = (
        root
        / "data/historic_data/players/players.parquet"
    )

    players = PlayerCrosswalk(
        read_parquet(players_path)
    )

    qb1_by_team = load_current_qb1(
        root,
        players,
    )

    prior_stats = load_prior_qb_stats(
        root,
        season,
        qb1_by_team,
    )

    for metric in QB_METRICS:
        home_values: list[float | None] = []
        away_values: list[float | None] = []
        diff_values: list[float | None] = []

        for _, row in base.iterrows():
            home = normalize_team(row["home_team"])
            away = normalize_team(row["away_team"])

            home_value = prior_stats.get(
                home,
                {},
            ).get(metric)

            away_value = prior_stats.get(
                away,
                {},
            ).get(metric)

            home_values.append(home_value)
            away_values.append(away_value)

            if home_value is None or away_value is None:
                diff_values.append(None)
            else:
                diff_values.append(home_value - away_value)

        base[f"home_qb_{metric}"] = home_values
        base[f"away_qb_{metric}"] = away_values
        base[f"qb_{metric}_diff"] = diff_values

    return base


@dataclass(frozen=True)
class DepthRecord:
    player_id: str
    name: str
    position: str
    rank: int
    starter: bool


def load_current_depth(
    root: Path,
) -> dict[str, list[DepthRecord]]:
    depth_root = root / "data/master/depth_charts"

    if not depth_root.exists():
        raise FileNotFoundError(
            f"Missing input directory: {depth_root}"
        )

    output: dict[str, list[DepthRecord]] = {}

    paths = sorted(
        depth_root.glob("*/*_depth.csv")
    )

    if not paths:
        raise RuntimeError(
            f"No depth-chart CSV files found under {depth_root}"
        )

    for path in paths:
        depth = read_csv(path)

        require_columns(
            depth,
            [
                "player_id",
                "name",
                "team",
                "position_abb",
                "depth_chart_rank",
                "starter_flag",
            ],
            str(path),
        )

        teams = {
            normalize_team(value)
            for value in depth["team"]
            if normalize_team(value)
        }

        if len(teams) != 1:
            raise ValueError(
                f"{path}: expected exactly one team; found {sorted(teams)}"
            )

        team = next(iter(teams))

        rows: list[DepthRecord] = []

        for _, row in depth.iterrows():
            rank = parse_int(row["depth_chart_rank"])

            if rank is None:
                continue

            rows.append(
                DepthRecord(
                    player_id=clean(row["player_id"]),
                    name=clean(row["name"]),
                    position=normalize_position(row["position_abb"]),
                    rank=rank,
                    starter=(parse_int(row["starter_flag"]) == 1),
                )
            )

        output[team] = rows

    return output


def index_depth(
    records: list[DepthRecord],
) -> tuple[
    dict[str, DepthRecord],
    dict[str, DepthRecord],
]:
    by_id: dict[str, DepthRecord] = {}
    by_name: dict[str, DepthRecord] = {}

    for record in records:
        if record.player_id:
            previous = by_id.get(record.player_id)

            if previous is None or record.rank < previous.rank:
                by_id[record.player_id] = record

        name_key = normalize_name(record.name)

        if name_key:
            previous = by_name.get(name_key)

            if previous is None or record.rank < previous.rank:
                by_name[name_key] = record

    return by_id, by_name


def load_current_injuries(
    root: Path,
    season: int,
) -> dict[str, list[dict[str, str]]]:
    path = root / "00_intake/injuries" / f"{season}_injuries.csv"
    injuries = read_csv(
        path,
        allow_empty_rows=True,
    )

    require_columns(
        injuries,
        [
            "season",
            "team",
            "player_id",
            "player_name",
            "position",
            "game_status",
            "report_date",
        ],
        str(path),
    )

    output: dict[str, list[dict[str, str]]] = {}

    for _, row in injuries.iterrows():
        row_season = parse_int(row["season"])

        if row_season is not None and row_season != season:
            continue

        team = normalize_team(row["team"])

        if not team:
            continue

        status = clean(row["game_status"]).casefold()

        output.setdefault(
            team,
            [],
        ).append(
            {
                "player_id": clean(row["player_id"]),
                "player_name": clean(row["player_name"]),
                "position": normalize_position(row["position"]),
                "status": status,
                "report_date": clean(row["report_date"]),
            }
        )

    return output


def compute_confirmed_injury_counts(
    team: str,
    current_depth: dict[str, list[DepthRecord]],
    current_injuries: dict[str, list[dict[str, str]]],
) -> dict[str, float]:
    values = {
        feature: 0.0
        for feature in INJURY_FEATURES
    }

    values["depth_starter_changes"] = np.nan

    depth_records = current_depth.get(
        team,
        [],
    )

    by_id, by_name = index_depth(
        depth_records
    )

    for injury in current_injuries.get(
        team,
        [],
    ):
        status = injury["status"]

        if status == "out":
            values["inj_out_count"] += 1.0
        elif status == "doubtful":
            values["inj_doubtful_count"] += 1.0
        elif status == "questionable":
            values["inj_questionable_count"] += 1.0
        else:
            continue

        if status != "out":
            continue

        depth_record = None

        raw_id = injury["player_id"]

        if raw_id:
            depth_record = by_id.get(raw_id)

        if depth_record is None:
            name_key = normalize_name(
                injury["player_name"]
            )

            if name_key:
                depth_record = by_name.get(
                    name_key
                )

        if depth_record is None:
            continue

        is_starter = (
            depth_record.starter
            or depth_record.rank == 1
        )

        is_top2 = depth_record.rank <= 2

        position = (
            injury["position"]
            or depth_record.position
        )

        if is_starter:
            values[
                "inj_starter_out_count"
            ] += 1.0

            if position == "QB":
                values[
                    "inj_qb1_out"
                ] = 1.0

        if is_top2:
            values[
                "inj_top2_depth_out_count"
            ] += 1.0

    values[
        "inj_ol_starter_out_count"
    ] = np.nan

    values[
        "inj_skill_starter_out_count"
    ] = np.nan

    values[
        "inj_front7_starter_out_count"
    ] = np.nan

    values[
        "inj_secondary_starter_out_count"
    ] = np.nan

    values[
        "inj_offense_unavailable_snap_share"
    ] = np.nan

    values[
        "inj_defense_unavailable_snap_share"
    ] = np.nan

    return values


def add_injury_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    current_depth = load_current_depth(
        root
    )

    current_injuries = load_current_injuries(
        root,
        season,
    )

    home_by_feature: dict[str, list[float]] = {
        feature: []
        for feature in INJURY_FEATURES
    }

    away_by_feature: dict[str, list[float]] = {
        feature: []
        for feature in INJURY_FEATURES
    }

    for _, row in base.iterrows():
        home = normalize_team(
            row["home_team"]
        )

        away = normalize_team(
            row["away_team"]
        )

        home_values = compute_confirmed_injury_counts(
            home,
            current_depth,
            current_injuries,
        )

        away_values = compute_confirmed_injury_counts(
            away,
            current_depth,
            current_injuries,
        )

        for feature in INJURY_FEATURES:
            home_by_feature[feature].append(
                home_values[feature]
            )

            away_by_feature[feature].append(
                away_values[feature]
            )

    for feature in INJURY_FEATURES:
        home_column = f"home_{feature}"
        away_column = f"away_{feature}"
        diff_column = f"{feature}_diff"

        base[home_column] = home_by_feature[
            feature
        ]

        base[away_column] = away_by_feature[
            feature
        ]

        home_numeric = pd.to_numeric(
            base[home_column],
            errors="coerce",
        )

        away_numeric = pd.to_numeric(
            base[away_column],
            errors="coerce",
        )

        base[diff_column] = (
            home_numeric
            - away_numeric
        )

    return base


def add_confirmed_week1_features(
    base: pd.DataFrame,
    root: Path,
    season: int,
) -> pd.DataFrame:
    weekly_schedule_path = (
        root
        / "00_intake/schedule/weekly"
        / "week_1_NFL_weekly_schedule.csv"
    )

    weekly_schedule = read_csv(
        weekly_schedule_path
    )

    base = add_market_features(
        base,
        weekly_schedule,
        str(weekly_schedule_path),
    )

    base = add_drat_features(
        base,
        root,
        season,
    )

    base = add_epred_features(
        base,
        root,
        season,
    )

    base = add_schedule_features(
        base,
        root,
        season,
    )

    base = add_division_feature(
        base,
        root,
        season,
    )

    base = add_weather_features(
        base,
        root,
    )

    base = add_travel_features(
        base,
        root,
        season,
    )

    base = add_prior_team_features(
        base,
        root,
        season,
    )

    base = add_prior_qb_features(
        base,
        root,
        season,
    )

    base = add_injury_features(
        base,
        root,
        season,
    )

    return base


def prepare_model_features(
    base: pd.DataFrame,
    schema: dict,
) -> tuple[pd.DataFrame, list[str]]:
    feature_order = list(
        schema["feature_order"]
    )

    numeric_features = set(
        schema["numeric_features"]
    )

    categorical_features = set(
        schema["categorical_features"]
    )

    unsupported: list[str] = []

    for feature in feature_order:
        if feature in base.columns:
            continue

        unsupported.append(feature)

        if feature in numeric_features:
            base[feature] = np.nan
        elif feature in categorical_features:
            base[feature] = MISSING_CAT
        else:
            raise RuntimeError(
                f"Schema feature is not classified: {feature}"
            )

    features = base[
        feature_order
    ].copy()

    for feature in feature_order:
        if feature in numeric_features:
            features[feature] = pd.to_numeric(
                features[feature],
                errors="coerce",
            )
        else:
            features[feature] = (
                features[feature]
                .map(clean)
                .replace("", MISSING_CAT)
                .astype(str)
            )

    if list(features.columns) != feature_order:
        raise RuntimeError(
            "Prepared feature order does not exactly match "
            "step11_feature_schema.json"
        )

    return features, unsupported


def sigmoid(
    value: np.ndarray | float,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=float,
    )

    array = np.clip(
        array,
        -700.0,
        700.0,
    )

    return 1.0 / (
        1.0
        + np.exp(-array)
    )


def load_calibrations(root: Path) -> dict:
    path = (
        root
        / "models/step14_probability_calibration.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    calibrations = raw.get(
        "calibrations",
        raw,
    )

    for key in [
        "moneyline",
        "spread",
        "total",
    ]:
        if key not in calibrations:
            raise ValueError(
                f"{path}: missing calibration section {key!r}"
            )

        if (
            "intercept" not in calibrations[key]
            or "slope" not in calibrations[key]
        ):
            raise ValueError(
                f"{path}: calibration section {key!r} must "
                "contain intercept and slope"
            )

    return calibrations


def apply_models(
    root: Path,
    original: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    margin_model_path = (
        root
        / "models/step11_margin_model.cbm"
    )

    total_model_path = (
        root
        / "models/step11_total_points_model.cbm"
    )

    for path in [
        margin_model_path,
        total_model_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    margin_model = CatBoostRegressor()
    total_model = CatBoostRegressor()

    margin_model.load_model(
        str(margin_model_path)
    )

    total_model.load_model(
        str(total_model_path)
    )

    expected_names = list(
        features.columns
    )

    if list(
        margin_model.feature_names_
    ) != expected_names:
        raise RuntimeError(
            "Margin model feature names/order do not match "
            "step11_feature_schema.json"
        )

    if list(
        total_model.feature_names_
    ) != expected_names:
        raise RuntimeError(
            "Total model feature names/order do not match "
            "step11_feature_schema.json"
        )

    predicted_margin = np.asarray(
        margin_model.predict(
            features
        ),
        dtype=float,
    )

    predicted_total = np.asarray(
        total_model.predict(
            features
        ),
        dtype=float,
    )

    if len(predicted_margin) != len(original):
        raise RuntimeError(
            "Margin prediction row count does not match Week 1 rows"
        )

    if len(predicted_total) != len(original):
        raise RuntimeError(
            "Total prediction row count does not match Week 1 rows"
        )

    if not np.isfinite(
        predicted_margin
    ).all():
        raise RuntimeError(
            "Margin model produced non-finite predictions"
        )

    if not np.isfinite(
        predicted_total
    ).all():
        raise RuntimeError(
            "Total model produced non-finite predictions"
        )

    spread_line = pd.to_numeric(
        features["spread_line"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    total_line = pd.to_numeric(
        features["total_line"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        spread_line
    ).all():
        bad_rows = np.where(
            ~np.isfinite(
                spread_line
            )
        )[0].tolist()

        raise ValueError(
            "spread_line is missing/non-numeric for Week 1 rows: "
            f"{bad_rows[:10]}"
        )

    if not np.isfinite(
        total_line
    ).all():
        bad_rows = np.where(
            ~np.isfinite(
                total_line
            )
        )[0].tolist()

        raise ValueError(
            "total_line is missing/non-numeric for Week 1 rows: "
            f"{bad_rows[:10]}"
        )

    calibrations = load_calibrations(
        root
    )

    moneyline = calibrations[
        "moneyline"
    ]

    spread = calibrations[
        "spread"
    ]

    total = calibrations[
        "total"
    ]

    home_win_probability = sigmoid(
        float(
            moneyline["intercept"]
        )
        + float(
            moneyline["slope"]
        )
        * predicted_margin
    )

    home_cover_probability = sigmoid(
        float(
            spread["intercept"]
        )
        + float(
            spread["slope"]
        )
        * (
            predicted_margin
            - spread_line
        )
    )

    over_probability = sigmoid(
        float(
            total["intercept"]
        )
        + float(
            total["slope"]
        )
        * (
            predicted_total
            - total_line
        )
    )

    predicted_home_score = (
        predicted_total
        + predicted_margin
    ) / 2.0

    predicted_away_score = (
        predicted_total
        - predicted_margin
    ) / 2.0

    output = original.copy()

    output["predicted_margin"] = predicted_margin
    output["predicted_total"] = predicted_total
    output["predicted_home_score"] = predicted_home_score
    output["predicted_away_score"] = predicted_away_score
    output["home_win_probability"] = home_win_probability
    output["away_win_probability"] = 1.0 - home_win_probability
    output["home_cover_probability"] = home_cover_probability
    output["away_cover_probability"] = 1.0 - home_cover_probability
    output["over_probability"] = over_probability
    output["under_probability"] = 1.0 - over_probability

    return output


def main() -> None:
    root = nfl_root()

    combined_path = (
        root
        / "00_intake/predictions/enriched/combined"
        / "week_1_NFL_enriched.csv"
    )

    base = read_csv(
        combined_path
    )

    validate_week1_base(
        base,
        SEASON,
        str(combined_path),
    )

    original = base.copy()

    schema = load_schema(
        root
    )

    enriched = add_confirmed_week1_features(
        base.copy(),
        root,
        SEASON,
    )

    features, unsupported = prepare_model_features(
        enriched,
        schema,
    )

    if unsupported:
        print(
            "UNCONFIRMED_MODEL_FEATURES_FILLED_AS_MISSING="
            + ",".join(unsupported)
        )

    projected = apply_models(
        root,
        original,
        features,
    )

    output_dir = (
        root
        / "01_merge"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "week_1_NFL_enriched.csv"
    )

    projected.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(projected)} | "
        f"columns={len(projected.columns)} | "
        f"unconfirmed_features={len(unsupported)}"
    )


if __name__ == "__main__":
    main()
