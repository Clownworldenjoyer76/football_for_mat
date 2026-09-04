#!/usr/bin/env python3
"""
Build immutable weekly team and opponent opportunity tables.

READS:
    docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
    docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv
    docs/win/football/nfl/data/historic_data/player_stats/stats_player_week_{season}.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/opportunity/team_week_opportunity.parquet
    docs/win/football/nfl/prop_engine/data/historical/opportunity/opponent_week_opportunity.parquet

POLICY:
    - 2012-2020: use verified weekly player-stat totals for core opportunity
      fields because local PBP does not exist; PBP-only fields remain null.
    - 2021-2025: build offense from posteam and defense/opponent context from
      defteam using local PBP.
    - Same-week realized tables are immutable raw measurements.
    - All lagging occurs downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


TEAM_COLUMNS = [
    "season",
    "week",
    "team",
    "games",
    "offensive_plays",
    "drives",
    "dropbacks",
    "pass_attempts",
    "rush_attempts",
    "targets",
    "pass_rate",
    "rush_rate",
    "plays_per_drive",
    "points_per_drive",
    "red_zone_drives",
    "red_zone_plays",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "sacks_allowed",
    "qb_hits_allowed",
    "field_goal_attempts",
    "field_goals_made",
    "extra_point_attempts",
    "extra_points_made",
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPPONENT_COLUMNS = [
    "season",
    "week",
    "team",
    "defensive_plays",
    "opponent_drives",
    "opponent_dropbacks",
    "opponent_pass_attempts",
    "opponent_rush_attempts",
    "opponent_targets",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "passing_tds_allowed",
    "rushing_tds_allowed",
    "sacks",
    "qb_hits",
    "red_zone_plays_allowed",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "field_goal_attempts_allowed",
    "extra_point_attempts_allowed",
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

TEAM_GRAIN = ["season", "week", "team"]
GAME_TEAM_GRAIN = ["season", "week", "game_id", "team"]

TEAM_ADVANCED_COLUMNS = [
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "points_per_drive",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPPONENT_ADVANCED_COLUMNS = [
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

TEAM_PBP_ONLY_COLUMNS = [
    "drives",
    "plays_per_drive",
    "points_per_drive",
    "red_zone_drives",
    "red_zone_plays",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPPONENT_PBP_ONLY_COLUMNS = [
    "opponent_drives",
    "red_zone_plays_allowed",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

NONNEGATIVE_TEAM_COLUMNS = [
    "games",
    "offensive_plays",
    "drives",
    "dropbacks",
    "pass_attempts",
    "rush_attempts",
    "targets",
    "red_zone_drives",
    "red_zone_plays",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "passing_tds",
    "rushing_tds",
    "sacks_allowed",
    "qb_hits_allowed",
    "field_goal_attempts",
    "field_goals_made",
    "extra_point_attempts",
    "extra_points_made",
]

NONNEGATIVE_OPPONENT_COLUMNS = [
    "defensive_plays",
    "opponent_drives",
    "opponent_dropbacks",
    "opponent_pass_attempts",
    "opponent_rush_attempts",
    "opponent_targets",
    "passing_tds_allowed",
    "rushing_tds_allowed",
    "sacks",
    "qb_hits",
    "red_zone_plays_allowed",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "field_goal_attempts_allowed",
    "extra_point_attempts_allowed",
]

RATE_COLUMNS = [
    "pass_rate",
    "rush_rate",
    "off_success_rate",
    "red_zone_td_rate",
    "third_down_conversion_rate",
    "def_success_rate",
    "red_zone_td_rate_allowed",
]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

GAME_ID_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{1,2})_"
    r"(?P<away>[A-Za-z0-9]+)_(?P<home>[A-Za-z0-9]+)$"
)


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


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def opponent_from_game_id(
    game_id: Any,
    team: Any,
) -> str:
    text = clean(game_id)
    match = GAME_ID_RE.fullmatch(text)

    if not match:
        raise ValueError(
            f"Unsupported nflverse game_id for opponent mapping: {game_id!r}"
        )

    away = canonical_team(match.group("away"))
    home = canonical_team(match.group("home"))
    club = canonical_team(team)

    if club == away:
        return home

    if club == home:
        return away

    raise ValueError(
        f"Team {team!r} does not belong to game_id {game_id!r} "
        f"after franchise normalization."
    )


def numeric_series(
    series: pd.Series,
    *,
    label: str,
    fill_zero: bool = False,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid = (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & converted.isna()
    )

    if invalid.any():
        examples = (
            series.loc[invalid]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{label}: non-numeric values found. "
            f"Examples={examples}"
        )

    converted = converted.astype(float)

    if fill_zero:
        converted = converted.fillna(0.0)

    return converted


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype(float)

    den = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype(float)

    valid = (
        num.notna()
        & den.notna()
        & den.ne(0.0)
    )

    result = pd.Series(
        float("nan"),
        index=num.index,
        dtype="float64",
    )

    result.loc[valid] = (
        num.loc[valid]
        / den.loc[valid]
    )

    return result


def normalize_week_team_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = df.copy()

    output["season"] = pd.to_numeric(
        output["season"],
        errors="raise",
    ).astype(int)

    output["week"] = pd.to_numeric(
        output["week"],
        errors="raise",
    ).astype(int)

    output["team"] = output["team"].map(
        canonical_team
    )

    return output


def build_stats_game_tables(
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "attempts",
        "carries",
        "targets",
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "sacks_suffered",
        "def_sacks",
        "def_qb_hits",
        "fg_att",
        "fg_made",
        "pat_att",
        "pat_made",
    ]

    common.require_columns(
        source,
        required,
        str(path),
    )

    working = source.loc[
        pd.to_numeric(
            source["season"],
            errors="coerce",
        ).eq(season)
        &
        source["season_type"]
        .astype(str)
        .str.upper()
        .eq("REG")
    ].copy()

    if working.empty:
        raise RuntimeError(
            f"{path}: no regular-season rows for {season}."
        )

    working["season"] = season

    working["week"] = pd.to_numeric(
        working["week"],
        errors="raise",
    ).astype(int)

    working["game_id"] = (
        working["game_id"]
        .map(clean)
    )

    working["team"] = (
        working["team"]
        .map(canonical_team)
    )

    if working["game_id"].eq("").any():
        raise ValueError(
            f"{path}: blank regular-season game_id."
        )

    if working["team"].eq("").any():
        raise ValueError(
            f"{path}: blank regular-season team."
        )

    numeric_columns = [
        "attempts",
        "carries",
        "targets",
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "sacks_suffered",
        "def_sacks",
        "def_qb_hits",
        "fg_att",
        "fg_made",
        "pat_att",
        "pat_made",
    ]

    for column in numeric_columns:
        working[column] = numeric_series(
            working[column],
            label=f"{path}:{column}",
            fill_zero=True,
        )

    offense = (
        working.groupby(
            GAME_TEAM_GRAIN,
            as_index=False,
            dropna=False,
        )
        .agg(
            pass_attempts=("attempts", "sum"),
            rush_attempts=("carries", "sum"),
            targets=("targets", "sum"),
            passing_yards=("passing_yards", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            passing_tds=("passing_tds", "sum"),
            rushing_tds=("rushing_tds", "sum"),
            sacks_allowed=("sacks_suffered", "sum"),
            field_goal_attempts=("fg_att", "sum"),
            field_goals_made=("fg_made", "sum"),
            extra_point_attempts=("pat_att", "sum"),
            extra_points_made=("pat_made", "sum"),
        )
    )

    defense = (
        working.groupby(
            GAME_TEAM_GRAIN,
            as_index=False,
            dropna=False,
        )
        .agg(
            sacks=("def_sacks", "sum"),
            qb_hits=("def_qb_hits", "sum"),
        )
    )

    offense["opponent"] = [
        opponent_from_game_id(
            game_id,
            team,
        )
        for game_id, team in zip(
            offense["game_id"],
            offense["team"],
        )
    ]

    defense["opponent"] = [
        opponent_from_game_id(
            game_id,
            team,
        )
        for game_id, team in zip(
            defense["game_id"],
            defense["team"],
        )
    ]

    common.ensure_unique(
        offense,
        GAME_TEAM_GRAIN,
        f"{path} offense game-team grain",
    )

    common.ensure_unique(
        defense,
        GAME_TEAM_GRAIN,
        f"{path} defense game-team grain",
    )

    return offense, defense


def build_pre_pbp_tables(
    offense_games: pd.DataFrame,
    defense_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offense_games = offense_games.copy()
    defense_games = defense_games.copy()

    offense_games["dropbacks"] = (
        offense_games["pass_attempts"]
        + offense_games["sacks_allowed"]
    )

    offense_games["offensive_plays"] = (
        offense_games["pass_attempts"]
        + offense_games["rush_attempts"]
        + offense_games["sacks_allowed"]
    )

    defense_allowed = (
        defense_games[
            [
                "season",
                "week",
                "game_id",
                "team",
                "opponent",
                "qb_hits",
            ]
        ]
        .rename(
            columns={
                "team": "_defense_team",
                "opponent": "team",
                "qb_hits": "qb_hits_allowed",
            }
        )
    )

    offense_games = offense_games.merge(
        defense_allowed[
            [
                "season",
                "week",
                "game_id",
                "team",
                "qb_hits_allowed",
            ]
        ],
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    offense_games["qb_hits_allowed"] = (
        offense_games["qb_hits_allowed"]
        .fillna(0.0)
    )

    team = (
        offense_games.groupby(
            TEAM_GRAIN,
            as_index=False,
            dropna=False,
        )
        .agg(
            games=("game_id", "nunique"),
            offensive_plays=("offensive_plays", "sum"),
            dropbacks=("dropbacks", "sum"),
            pass_attempts=("pass_attempts", "sum"),
            rush_attempts=("rush_attempts", "sum"),
            targets=("targets", "sum"),
            passing_yards=("passing_yards", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            passing_tds=("passing_tds", "sum"),
            rushing_tds=("rushing_tds", "sum"),
            sacks_allowed=("sacks_allowed", "sum"),
            qb_hits_allowed=("qb_hits_allowed", "sum"),
            field_goal_attempts=("field_goal_attempts", "sum"),
            field_goals_made=("field_goals_made", "sum"),
            extra_point_attempts=("extra_point_attempts", "sum"),
            extra_points_made=("extra_points_made", "sum"),
        )
    )

    team["pass_rate"] = safe_divide(
        team["pass_attempts"],
        team["offensive_plays"],
    )

    team["rush_rate"] = safe_divide(
        team["rush_attempts"],
        team["offensive_plays"],
    )

    for column in TEAM_PBP_ONLY_COLUMNS:
        team[column] = float("nan")

    allowed = offense_games[
        [
            "season",
            "week",
            "game_id",
            "opponent",
            "offensive_plays",
            "dropbacks",
            "pass_attempts",
            "rush_attempts",
            "targets",
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
            "field_goal_attempts",
            "extra_point_attempts",
        ]
    ].rename(
        columns={
            "opponent": "team",
            "offensive_plays": "defensive_plays",
            "dropbacks": "opponent_dropbacks",
            "pass_attempts": "opponent_pass_attempts",
            "rush_attempts": "opponent_rush_attempts",
            "targets": "opponent_targets",
            "passing_yards": "passing_yards_allowed",
            "rushing_yards": "rushing_yards_allowed",
            "passing_tds": "passing_tds_allowed",
            "rushing_tds": "rushing_tds_allowed",
            "field_goal_attempts": "field_goal_attempts_allowed",
            "extra_point_attempts": "extra_point_attempts_allowed",
        }
    )

    defense_week = (
        defense_games.groupby(
            TEAM_GRAIN,
            as_index=False,
            dropna=False,
        )
        .agg(
            sacks=("sacks", "sum"),
            qb_hits=("qb_hits", "sum"),
        )
    )

    opponent = (
        allowed.groupby(
            TEAM_GRAIN,
            as_index=False,
            dropna=False,
        )
        .agg(
            defensive_plays=("defensive_plays", "sum"),
            opponent_dropbacks=("opponent_dropbacks", "sum"),
            opponent_pass_attempts=("opponent_pass_attempts", "sum"),
            opponent_rush_attempts=("opponent_rush_attempts", "sum"),
            opponent_targets=("opponent_targets", "sum"),
            passing_yards_allowed=("passing_yards_allowed", "sum"),
            rushing_yards_allowed=("rushing_yards_allowed", "sum"),
            passing_tds_allowed=("passing_tds_allowed", "sum"),
            rushing_tds_allowed=("rushing_tds_allowed", "sum"),
            field_goal_attempts_allowed=("field_goal_attempts_allowed", "sum"),
            extra_point_attempts_allowed=("extra_point_attempts_allowed", "sum"),
        )
        .merge(
            defense_week,
            on=TEAM_GRAIN,
            how="left",
            validate="one_to_one",
        )
    )

    opponent["sacks"] = opponent["sacks"].fillna(0.0)
    opponent["qb_hits"] = opponent["qb_hits"].fillna(0.0)

    for column in OPPONENT_PBP_ONLY_COLUMNS:
        opponent[column] = float("nan")

    return (
        team[TEAM_COLUMNS].copy(),
        opponent[OPPONENT_COLUMNS].copy(),
    )


def get_drive_column(
    pbp: pd.DataFrame,
) -> str:
    if "fixed_drive" in pbp.columns:
        return "fixed_drive"

    if "drive" in pbp.columns:
        return "drive"

    raise ValueError(
        "PBP requires fixed_drive or drive."
    )


def build_pbp_tables(
    pbp: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "season",
        "season_type",
        "week",
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "play_type",
        "yardline_100",
        "pass_attempt",
        "qb_dropback",
        "rush_attempt",
        "receiver_player_id",
        "passing_yards",
        "rushing_yards",
        "pass_touchdown",
        "rush_touchdown",
        "sack",
        "qb_hit",
        "field_goal_attempt",
        "field_goal_result",
        "extra_point_attempt",
        "extra_point_result",
    ]

    common.require_columns(
        pbp,
        required,
        str(path),
    )

    drive_col = get_drive_column(pbp)

    working = pbp.loc[
        pd.to_numeric(
            pbp["season"],
            errors="coerce",
        ).eq(season)
        &
        pbp["season_type"]
        .astype(str)
        .str.upper()
        .eq("REG")
    ].copy()

    if working.empty:
        raise RuntimeError(
            f"{path}: no regular-season PBP rows for {season}."
        )

    working["season"] = season

    working["week"] = pd.to_numeric(
        working["week"],
        errors="raise",
    ).astype(int)

    working["game_id"] = working["game_id"].map(clean)
    working["posteam"] = working["posteam"].map(canonical_team)
    working["defteam"] = working["defteam"].map(canonical_team)
    working["receiver_player_id"] = working["receiver_player_id"].map(clean)

    numeric_columns = [
        "yardline_100",
        "pass_attempt",
        "qb_dropback",
        "rush_attempt",
        "passing_yards",
        "rushing_yards",
        "pass_touchdown",
        "rush_touchdown",
        "sack",
        "qb_hit",
        "field_goal_attempt",
        "extra_point_attempt",
    ]

    for column in numeric_columns:
        working[column] = numeric_series(
            working[column],
            label=f"{path}:{column}",
            fill_zero=column != "yardline_100",
        )

    eligible = (
        working["posteam"].ne("")
        & working["defteam"].ne("")
        & working["game_id"].ne("")
    )

    scrimmage = (
        eligible
        & working["play_type"].isin(
            ["pass", "run"]
        )
    )

    rz = (
        eligible
        & working["yardline_100"].notna()
        & working["yardline_100"].le(20.0)
    )

    goal_line = (
        eligible
        & working["yardline_100"].notna()
        & working["yardline_100"].le(5.0)
    )

    target = (
        eligible
        & working["pass_attempt"].eq(1.0)
        & working["receiver_player_id"].ne("")
    )

    working["_target"] = target.astype(float)
    working["_field_goal_made"] = (
        working["field_goal_attempt"].eq(1.0)
        & working["field_goal_result"]
        .astype(str)
        .str.casefold()
        .eq("made")
    ).astype(float)

    working["_extra_point_made"] = (
        working["extra_point_attempt"].eq(1.0)
        & working["extra_point_result"]
        .astype(str)
        .str.casefold()
        .eq("good")
    ).astype(float)

    working["_scrimmage"] = scrimmage.astype(float)
    working["_rz_scrimmage"] = (
        scrimmage
        & rz
    ).astype(float)

    working["_rz_pass_attempt"] = (
        rz
        & working["pass_attempt"].eq(1.0)
    ).astype(float)

    working["_rz_rush_attempt"] = (
        rz
        & working["rush_attempt"].eq(1.0)
    ).astype(float)

    working["_goal_line_rush_attempt"] = (
        goal_line
        & working["rush_attempt"].eq(1.0)
    ).astype(float)

    offense_rows = working.loc[
        eligible
    ].copy()

    team = (
        offense_rows.groupby(
            ["season", "week", "posteam"],
            as_index=False,
            dropna=False,
        )
        .agg(
            games=("game_id", "nunique"),
            offensive_plays=("_scrimmage", "sum"),
            dropbacks=("qb_dropback", "sum"),
            pass_attempts=("pass_attempt", "sum"),
            rush_attempts=("rush_attempt", "sum"),
            targets=("_target", "sum"),
            red_zone_plays=("_rz_scrimmage", "sum"),
            red_zone_pass_attempts=("_rz_pass_attempt", "sum"),
            red_zone_rush_attempts=("_rz_rush_attempt", "sum"),
            goal_line_rush_attempts=("_goal_line_rush_attempt", "sum"),
            passing_yards=("passing_yards", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            passing_tds=("pass_touchdown", "sum"),
            rushing_tds=("rush_touchdown", "sum"),
            sacks_allowed=("sack", "sum"),
            qb_hits_allowed=("qb_hit", "sum"),
            field_goal_attempts=("field_goal_attempt", "sum"),
            field_goals_made=("_field_goal_made", "sum"),
            extra_point_attempts=("extra_point_attempt", "sum"),
            extra_points_made=("_extra_point_made", "sum"),
        )
        .rename(
            columns={
                "posteam": "team",
            }
        )
    )

    valid_drive_rows = offense_rows.loc[
        offense_rows[drive_col].notna()
    ].copy()

    drive_counts = (
        valid_drive_rows[
            [
                "season",
                "week",
                "game_id",
                "posteam",
                drive_col,
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["season", "week", "posteam"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "posteam": "team",
                "size": "drives",
            }
        )
    )

    red_zone_drive_counts = (
        valid_drive_rows.loc[
            valid_drive_rows["yardline_100"].notna()
            & valid_drive_rows["yardline_100"].le(20.0),
            [
                "season",
                "week",
                "game_id",
                "posteam",
                drive_col,
            ],
        ]
        .drop_duplicates()
        .groupby(
            ["season", "week", "posteam"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "posteam": "team",
                "size": "red_zone_drives",
            }
        )
    )

    team = (
        team.merge(
            drive_counts,
            on=TEAM_GRAIN,
            how="left",
            validate="one_to_one",
        )
        .merge(
            red_zone_drive_counts,
            on=TEAM_GRAIN,
            how="left",
            validate="one_to_one",
        )
    )

    team["drives"] = team["drives"].fillna(0.0)
    team["red_zone_drives"] = team["red_zone_drives"].fillna(0.0)

    team["pass_rate"] = safe_divide(
        team["pass_attempts"],
        team["offensive_plays"],
    )

    team["rush_rate"] = safe_divide(
        team["rush_attempts"],
        team["offensive_plays"],
    )

    team["plays_per_drive"] = safe_divide(
        team["offensive_plays"],
        team["drives"],
    )

    defense = (
        offense_rows.groupby(
            ["season", "week", "defteam"],
            as_index=False,
            dropna=False,
        )
        .agg(
            defensive_plays=("_scrimmage", "sum"),
            opponent_dropbacks=("qb_dropback", "sum"),
            opponent_pass_attempts=("pass_attempt", "sum"),
            opponent_rush_attempts=("rush_attempt", "sum"),
            opponent_targets=("_target", "sum"),
            passing_yards_allowed=("passing_yards", "sum"),
            rushing_yards_allowed=("rushing_yards", "sum"),
            passing_tds_allowed=("pass_touchdown", "sum"),
            rushing_tds_allowed=("rush_touchdown", "sum"),
            sacks=("sack", "sum"),
            qb_hits=("qb_hit", "sum"),
            red_zone_plays_allowed=("_rz_scrimmage", "sum"),
            red_zone_pass_attempts_allowed=("_rz_pass_attempt", "sum"),
            red_zone_rush_attempts_allowed=("_rz_rush_attempt", "sum"),
            goal_line_rush_attempts_allowed=("_goal_line_rush_attempt", "sum"),
            field_goal_attempts_allowed=("field_goal_attempt", "sum"),
            extra_point_attempts_allowed=("extra_point_attempt", "sum"),
        )
        .rename(
            columns={
                "defteam": "team",
            }
        )
    )

    opponent_drive_counts = (
        valid_drive_rows[
            [
                "season",
                "week",
                "game_id",
                "defteam",
                drive_col,
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["season", "week", "defteam"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "defteam": "team",
                "size": "opponent_drives",
            }
        )
    )

    defense = defense.merge(
        opponent_drive_counts,
        on=TEAM_GRAIN,
        how="left",
        validate="one_to_one",
    )

    defense["opponent_drives"] = (
        defense["opponent_drives"]
        .fillna(0.0)
    )

    return team, defense


def attach_team_stats(
    team: pd.DataFrame,
    opponent: pd.DataFrame,
    source: pd.DataFrame,
    *,
    season: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "season",
        "week",
        "team",
        *TEAM_ADVANCED_COLUMNS,
        *OPPONENT_ADVANCED_COLUMNS,
    ]

    common.require_columns(
        source,
        required,
        str(path),
    )

    metrics = normalize_week_team_frame(
        source[required].copy()
    )

    metrics = metrics.loc[
        metrics["season"].eq(season)
    ].copy()

    common.ensure_unique(
        metrics,
        TEAM_GRAIN,
        f"{path} season/week/team grain",
    )

    team = team.merge(
        metrics[
            [
                *TEAM_GRAIN,
                *TEAM_ADVANCED_COLUMNS,
            ]
        ],
        on=TEAM_GRAIN,
        how="left",
        validate="one_to_one",
    )

    opponent = opponent.merge(
        metrics[
            [
                *TEAM_GRAIN,
                *OPPONENT_ADVANCED_COLUMNS,
            ]
        ],
        on=TEAM_GRAIN,
        how="left",
        validate="one_to_one",
    )

    return team, opponent


def validate_output(
    team: pd.DataFrame,
    opponent: pd.DataFrame,
    *,
    config: dict,
    rich_feature_start: int,
) -> None:
    if list(team.columns) != TEAM_COLUMNS:
        raise ValueError(
            "Team opportunity headers/order do not match contract. "
            f"Got={list(team.columns)}"
        )

    if list(opponent.columns) != OPPONENT_COLUMNS:
        raise ValueError(
            "Opponent opportunity headers/order do not match contract. "
            f"Got={list(opponent.columns)}"
        )

    common.ensure_unique(
        team,
        TEAM_GRAIN,
        "team weekly opportunity grain",
    )

    common.ensure_unique(
        opponent,
        TEAM_GRAIN,
        "opponent weekly opportunity grain",
    )

    team_keys = set(
        map(
            tuple,
            team[TEAM_GRAIN].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    opponent_keys = set(
        map(
            tuple,
            opponent[TEAM_GRAIN].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    if team_keys != opponent_keys:
        raise ValueError(
            "Team and opponent weekly key sets differ."
        )

    common.reject_forbidden_feature_columns(
        team.columns,
        config,
    )

    common.reject_forbidden_feature_columns(
        opponent.columns,
        config,
    )

    for column in NONNEGATIVE_TEAM_COLUMNS:
        values = pd.to_numeric(
            team[column],
            errors="coerce",
        )

        negative = (
            values.notna()
            & values.lt(0.0)
        )

        if negative.any():
            raise ValueError(
                f"Negative team opportunity value in {column!r}."
            )

    for column in NONNEGATIVE_OPPONENT_COLUMNS:
        values = pd.to_numeric(
            opponent[column],
            errors="coerce",
        )

        negative = (
            values.notna()
            & values.lt(0.0)
        )

        if negative.any():
            raise ValueError(
                f"Negative opponent opportunity value in {column!r}."
            )

    combined_rate_frames = [
        team[
            [
                "pass_rate",
                "rush_rate",
                "off_success_rate",
                "red_zone_td_rate",
                "third_down_conversion_rate",
            ]
        ],
        opponent[
            [
                "def_success_rate",
                "red_zone_td_rate_allowed",
            ]
        ],
    ]

    for frame in combined_rate_frames:
        for column in frame.columns:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            invalid = (
                values.notna()
                & ~values.between(0.0, 1.0)
            )

            if invalid.any():
                raise ValueError(
                    f"Rate {column!r} outside [0,1]."
                )

    for frame, columns in [
        (team, TEAM_COLUMNS),
        (opponent, OPPONENT_COLUMNS),
    ]:
        for column in columns:
            if column in {
                "team",
            }:
                continue

            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            infinite = (
                values.notna()
                & np.isinf(values)
            )

            if infinite.any():
                raise ValueError(
                    f"Infinite values in {column!r}."
                )

    pre_rich_team = team["season"].lt(
        rich_feature_start
    )

    pre_rich_opponent = opponent["season"].lt(
        rich_feature_start
    )

    for column in TEAM_PBP_ONLY_COLUMNS:
        if team.loc[
            pre_rich_team,
            column,
        ].notna().any():
            raise ValueError(
                f"{column} must remain null before "
                f"rich_feature_start={rich_feature_start}."
            )

    for column in OPPONENT_PBP_ONLY_COLUMNS:
        if opponent.loc[
            pre_rich_opponent,
            column,
        ].notna().any():
            raise ValueError(
                f"{column} must remain null before "
                f"rich_feature_start={rich_feature_start}."
            )


def run() -> dict[str, Any]:
    config = common.load_config()
    repo = common.repo_root()

    start_season = int(
        config["seasons"]["historical_start"]
    )

    end_season = int(
        config["seasons"]["historical_end"]
    )

    rich_feature_start = int(
        config["seasons"]["rich_feature_start"]
    )

    team_output_path = (
        repo
        / config["paths"]["team_opportunity"]
    )

    opponent_output_path = (
        repo
        / config["paths"]["opponent_opportunity"]
    )

    team_frames = []
    opponent_frames = []
    diagnostics = {}

    for season in range(
        start_season,
        end_season + 1,
    ):
        stats_path = (
            repo
            / config["paths"][
                "historical_player_stats_pattern"
            ].format(
                season=season
            )
        )

        stats_source = (
            common.read_parquet_required(
                stats_path
            )
        )

        offense_games, defense_games = (
            build_stats_game_tables(
                stats_source,
                season=season,
                path=stats_path,
            )
        )

        if season < rich_feature_start:
            team, opponent = build_pre_pbp_tables(
                offense_games,
                defense_games,
            )

            diagnostics[
                str(season)
            ] = {
                "source": "player_stats_core",
                "pbp_available": False,
                "team_rows": int(
                    len(team)
                ),
            }

        else:
            pbp_path = (
                repo
                / config["paths"]["pbp_pattern"].format(
                    season=season
                )
            )

            team_stats_path = (
                repo
                / config["paths"][
                    "team_stats_pattern"
                ].format(
                    season=season
                )
            )

            if not pbp_path.is_file():
                raise FileNotFoundError(
                    f"Required rich-season PBP missing: {pbp_path}"
                )

            if not team_stats_path.is_file():
                raise FileNotFoundError(
                    f"Required rich-season team stats missing: {team_stats_path}"
                )

            pbp_source = common.read_csv_required(
                pbp_path
            )

            team_stats_source = (
                common.read_csv_required(
                    team_stats_path
                )
            )

            team, opponent = build_pbp_tables(
                pbp_source,
                season=season,
                path=pbp_path,
            )

            team, opponent = attach_team_stats(
                team,
                opponent,
                team_stats_source,
                season=season,
                path=team_stats_path,
            )

            team = team[TEAM_COLUMNS].copy()
            opponent = opponent[
                OPPONENT_COLUMNS
            ].copy()

            diagnostics[
                str(season)
            ] = {
                "source": "pbp_posteam_defteam",
                "pbp_available": True,
                "team_rows": int(
                    len(team)
                ),
            }

        common.ensure_unique(
            team,
            TEAM_GRAIN,
            f"team opportunity {season}",
        )

        common.ensure_unique(
            opponent,
            TEAM_GRAIN,
            f"opponent opportunity {season}",
        )

        team_frames.append(
            team
        )

        opponent_frames.append(
            opponent
        )

    team_output = pd.concat(
        team_frames,
        ignore_index=True,
    )[TEAM_COLUMNS].copy()

    opponent_output = pd.concat(
        opponent_frames,
        ignore_index=True,
    )[OPPONENT_COLUMNS].copy()

    validate_output(
        team_output,
        opponent_output,
        config=config,
        rich_feature_start=rich_feature_start,
    )

    common.write_parquet_atomic(
        team_output,
        team_output_path,
    )

    common.write_parquet_atomic(
        opponent_output,
        opponent_output_path,
    )

    payload = {
        "status": "passed",
        "historical_start": start_season,
        "historical_end": end_season,
        "rich_feature_start": rich_feature_start,
        "team_rows": int(
            len(team_output)
        ),
        "opponent_rows": int(
            len(opponent_output)
        ),
        "team_weeks": int(
            team_output[
                ["season", "week"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "pre_pbp_policy": (
            "2012-2020 core team/opponent opportunity derives from "
            "verified player-stat totals; drive/red-zone/EPA/success "
            "fields remain null because local PBP is unavailable"
        ),
        "rich_policy": (
            "2021-2025 offense built from posteam and opponent/defense "
            "context built from defteam"
        ),
        "offensive_play_policy": (
            "2021-2025 count play_type in {pass, run}; "
            "2012-2020 pass_attempts + rush_attempts + sacks_allowed"
        ),
        "pass_rate_policy": (
            "pass_attempts / offensive_plays; zero denominator -> null"
        ),
        "rush_rate_policy": (
            "rush_attempts / offensive_plays; zero denominator -> null"
        ),
        "red_zone_definition": (
            "yardline_100 <= 20"
        ),
        "goal_line_definition": (
            "yardline_100 <= 5"
        ),
        "same_week_policy": (
            "immutable raw realized weekly tables; all lagging downstream"
        ),
        "season_diagnostics": diagnostics,
        "team_output": str(
            team_output_path.relative_to(
                repo
            )
        ),
        "opponent_output": str(
            opponent_output_path.relative_to(
                repo
            )
        ),
    }

    common.log_run(
        "build_team_opportunity.py",
        payload,
    )

    return payload


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
