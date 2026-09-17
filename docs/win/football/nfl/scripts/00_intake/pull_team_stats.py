#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_team_stats.py
#
# Builds weekly NFL team-strength stats from the local nflverse PBP intake file.
#
# Input:
#   docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
#
# Output:
#   docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv
#
# Standard report:
#   docs/win/football/nfl/errors/00_intake/pull_team_stats.json
#
# Legacy log:
#   docs/win/football/nfl/errors/00_intake/pull_team_stats.txt

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


PBP_DIR = NFL_ROOT / "00_intake" / "pbp"
OUTPUT_DIR = NFL_ROOT / "00_intake" / "team_stats"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
REPORT_ROOT = NFL_ROOT / "errors"
LEGACY_LOG_FILE = REPORT_ROOT / "00_intake" / "pull_team_stats.txt"


OUTPUT_COLUMNS = [
    "season",
    "week",
    "team",
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

RATE_COLUMNS = [
    "off_success_rate",
    "def_success_rate",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "third_down_conversion_rate",
]

PBP_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "yards_gained",
    "down",
    "yardline_100",
    "posteam_score",
    "posteam_score_post",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "third_down_converted",
    "third_down_failed",
]

SCRIMMAGE_PLAY_TYPES = {"pass", "run"}


class TeamStatsError(RuntimeError):
    pass


class RunLog:
    def __init__(self, reporter: PipelineReporter) -> None:
        self.reporter = reporter
        self.lines: list[str] = []

    @staticmethod
    def _stamp() -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def info(self, message: str) -> None:
        line = f"[{self._stamp()}] {str(message).rstrip()}"
        self.lines.append(line)
        print(line)

    def warning(self, message: str, **details: Any) -> None:
        text = str(message).rstrip()
        line = f"[{self._stamp()}] WARNING: {text}"
        self.lines.append(line)
        self.reporter.warning(text, **details)
        print(line, file=sys.stderr)

    def error_text(self, message: str) -> None:
        line = f"[{self._stamp()}] ERROR: {str(message).rstrip()}"
        self.lines.append(line)
        print(line, file=sys.stderr)

    def write_legacy(self) -> None:
        LEGACY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{LEGACY_LOG_FILE.name}.",
            suffix=".tmp",
            dir=LEGACY_LOG_FILE.parent,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(self.lines).rstrip() + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, LEGACY_LOG_FILE)
        finally:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NFL weekly team stats from local nflverse PBP."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def read_pbp(pbp_path: Path) -> pd.DataFrame:
    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP input file not found: {pbp_path}")

    if pbp_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(
            pbp_path,
            compression="gzip",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_existing_output(
    output_path: Path,
    log: RunLog,
) -> tuple[pd.DataFrame | None, bool]:
    if not output_path.exists():
        return None, True

    try:
        return pd.read_csv(output_path, low_memory=False), True
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), True
    except Exception as exc:
        log.warning(
            "Existing team-stat output could not be read for empty-input safety",
            path=str(output_path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None, False


def count_completed_games(
    season: int,
    log: RunLog,
) -> tuple[int, bool, int]:
    paths = sorted(RESULTS_DIR.glob(f"{season}_*.csv"))
    completed: set[str] = set()
    inspection_uncertain = False

    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            inspection_uncertain = True
            log.warning(
                "Could not inspect final-results file while evaluating empty team-stat safety",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue

        if "status" not in frame.columns:
            inspection_uncertain = True
            log.warning(
                "Final-results file lacks status column while evaluating empty team-stat safety",
                path=str(path),
            )
            continue

        statuses = (
            frame["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        done = (
            statuses.str.contains("final", regex=False)
            | statuses.str.contains("completed", regex=False)
        )

        if not done.any():
            continue

        if "game_id" in frame.columns:
            ids = (
                frame.loc[done, "game_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            completed.update(value for value in ids if value)
        else:
            completed.update(
                f"{path.name}:{index}"
                for index in frame.index[done].tolist()
            )

    return len(completed), inspection_uncertain, len(paths)


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def validate_pbp_input(pbp: pd.DataFrame, season: int) -> str:
    if pbp.empty:
        raise TeamStatsError("Nonempty PBP validation received an empty dataframe")

    missing = [
        column
        for column in PBP_REQUIRED_COLUMNS
        if column not in pbp.columns
    ]
    if missing:
        raise TeamStatsError(
            "PBP input is missing required team-stat columns: "
            + ",".join(missing)
        )

    drive_column = get_drive_column(pbp)

    season_values = pd.to_numeric(pbp["season"], errors="coerce")
    invalid_season = season_values.isna() | season_values.ne(season)
    if invalid_season.any():
        examples = (
            pbp.loc[
                invalid_season,
                ["season", "week", "game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            f"PBP input contains rows outside requested season {season}: "
            f"{examples}"
        )

    week_values = pd.to_numeric(pbp["week"], errors="coerce")
    invalid_week = (
        week_values.isna()
        | week_values.le(0)
        | week_values.mod(1).ne(0)
    )
    if invalid_week.any():
        examples = (
            pbp.loc[
                invalid_week,
                ["season", "week", "game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            f"PBP input contains invalid week values: {examples}"
        )

    blank_game_id = _blank_mask(pbp["game_id"])
    blank_play_id = _blank_mask(pbp["play_id"])
    if blank_game_id.any() or blank_play_id.any():
        examples = (
            pbp.loc[
                blank_game_id | blank_play_id,
                ["game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            "PBP input contains blank game_id/play_id values: "
            f"{examples}"
        )

    duplicate_mask = pbp.duplicated(
        subset=["game_id", "play_id"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            pbp.loc[
                duplicate_mask,
                ["game_id", "play_id"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            "PBP input contains duplicate game_id/play_id rows: "
            f"{examples}"
        )

    return drive_column


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "season",
        "week",
        "epa",
        "success",
        "yards_gained",
        "down",
        "yardline_100",
        "posteam_score",
        "posteam_score_post",
        "touchdown",
        "pass_touchdown",
        "rush_touchdown",
        "third_down_converted",
        "third_down_failed",
        "first_down",
    ]

    df = df.copy()

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    context: str,
) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise TeamStatsError(
            f"Missing required columns for {context}: {missing}"
        )


def get_drive_column(df: pd.DataFrame) -> str:
    if "fixed_drive" in df.columns:
        return "fixed_drive"
    if "drive" in df.columns:
        return "drive"

    raise TeamStatsError(
        "Missing required drive column: fixed_drive or drive"
    )


def build_valid_scrimmage_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        pbp,
        [
            "season",
            "week",
            "posteam",
            "defteam",
            "play_type",
            "epa",
            "success",
            "yards_gained",
            "down",
        ],
        "scrimmage-play team stats",
    )

    mask = (
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["epa"].notna()
        & pbp["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    )

    return pbp.loc[mask].copy()


def build_offense_stats(valid_plays: pd.DataFrame) -> pd.DataFrame:
    off = (
        valid_plays.groupby(
            ["season", "week", "posteam"],
            dropna=False,
        )
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    early_downs = valid_plays[
        valid_plays["down"].isin([1, 2])
    ].copy()

    if early_downs.empty:
        early = pd.DataFrame(
            columns=["season", "week", "team", "early_down_epa"]
        )
    else:
        early = (
            early_downs.groupby(
                ["season", "week", "posteam"],
                dropna=False,
            )
            .agg(early_down_epa=("epa", "mean"))
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

    return off.merge(
        early,
        on=["season", "week", "team"],
        how="outer",
    )


def build_defense_stats(
    valid_plays: pd.DataFrame,
) -> pd.DataFrame:
    return (
        valid_plays.groupby(
            ["season", "week", "defteam"],
            dropna=False,
        )
        .agg(
            def_epa_per_play=("epa", "mean"),
            def_success_rate=("success", "mean"),
            yards_per_play_allowed=("yards_gained", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )


def build_third_down_stats(
    pbp: pd.DataFrame,
    valid_plays: pd.DataFrame,
) -> pd.DataFrame:
    if {
        "third_down_converted",
        "third_down_failed",
    }.issubset(pbp.columns):
        require_columns(
            pbp,
            ["season", "week", "posteam"],
            "third-down conversion rate",
        )

        third = pbp[
            pbp["season"].notna()
            & pbp["week"].notna()
            & pbp["posteam"].notna()
            & (
                (pbp["third_down_converted"] == 1)
                | (pbp["third_down_failed"] == 1)
            )
        ].copy()

        if third.empty:
            return pd.DataFrame(
                columns=[
                    "season",
                    "week",
                    "team",
                    "third_down_conversion_rate",
                ]
            )

        third["third_down_conversion_flag"] = np.where(
            third["third_down_converted"] == 1,
            1.0,
            0.0,
        )

    elif {"down", "first_down"}.issubset(valid_plays.columns):
        third = valid_plays[
            valid_plays["posteam"].notna()
            & valid_plays["down"].eq(3)
        ].copy()

        if third.empty:
            return pd.DataFrame(
                columns=[
                    "season",
                    "week",
                    "team",
                    "third_down_conversion_rate",
                ]
            )

        third["third_down_conversion_flag"] = np.where(
            third["first_down"] == 1,
            1.0,
            0.0,
        )

    else:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "third_down_conversion_rate",
            ]
        )

    return (
        third.groupby(
            ["season", "week", "posteam"],
            dropna=False,
        )
        .agg(
            third_down_conversion_rate=(
                "third_down_conversion_flag",
                "mean",
            )
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


def build_drive_points_stats(
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drive_col = get_drive_column(pbp)

    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            drive_col,
            "posteam",
            "defteam",
            "posteam_score",
            "posteam_score_post",
        ],
        "points per drive",
    )

    sort_columns = [
        "season",
        "week",
        "game_id",
        drive_col,
    ]
    if "play_id" in pbp.columns:
        sort_columns.append("play_id")

    drives = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp[drive_col].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if drives.empty:
        empty_off = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "points_per_drive",
            ]
        )
        empty_def = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "points_per_drive_allowed",
            ]
        )
        return empty_off, empty_def

    drives = drives.sort_values(sort_columns)

    drive_keys = [
        "season",
        "week",
        "game_id",
        drive_col,
        "posteam",
        "defteam",
    ]

    drive_scores = (
        drives.groupby(drive_keys, dropna=False)
        .agg(
            drive_start_score=("posteam_score", "first"),
            drive_end_score=("posteam_score_post", "last"),
        )
        .reset_index()
    )

    drive_scores["drive_points"] = (
        drive_scores["drive_end_score"]
        - drive_scores["drive_start_score"]
    )

    drive_scores.loc[
        drive_scores["drive_points"] < 0,
        "drive_points",
    ] = 0

    off_points = (
        drive_scores.groupby(
            ["season", "week", "posteam"],
            dropna=False,
        )
        .agg(points_per_drive=("drive_points", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_points = (
        drive_scores.groupby(
            ["season", "week", "defteam"],
            dropna=False,
        )
        .agg(
            points_per_drive_allowed=("drive_points", "mean")
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    return off_points, def_points


def add_offensive_touchdown_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    if {"touchdown", "td_team"}.issubset(df.columns):
        df["offensive_touchdown_flag"] = np.where(
            (df["touchdown"] == 1)
            & (df["td_team"] == df["posteam"]),
            1.0,
            0.0,
        )
        return df

    touchdown_parts = []

    if "pass_touchdown" in df.columns:
        touchdown_parts.append(
            df["pass_touchdown"].fillna(0).eq(1)
        )

    if "rush_touchdown" in df.columns:
        touchdown_parts.append(
            df["rush_touchdown"].fillna(0).eq(1)
        )

    if touchdown_parts:
        flag = touchdown_parts[0]
        for part in touchdown_parts[1:]:
            flag = flag | part

        df["offensive_touchdown_flag"] = np.where(
            flag,
            1.0,
            0.0,
        )
        return df

    if "touchdown" in df.columns:
        df["offensive_touchdown_flag"] = np.where(
            df["touchdown"] == 1,
            1.0,
            0.0,
        )
        return df

    df["offensive_touchdown_flag"] = np.nan
    return df


def build_red_zone_stats(
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drive_col = get_drive_column(pbp)

    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            drive_col,
            "posteam",
            "defteam",
            "yardline_100",
        ],
        "red-zone touchdown rate",
    )

    df = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp[drive_col].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if df.empty:
        empty_off = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "red_zone_td_rate",
            ]
        )
        empty_def = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "red_zone_td_rate_allowed",
            ]
        )
        return empty_off, empty_def

    df = add_offensive_touchdown_flag(df)

    drive_keys = [
        "season",
        "week",
        "game_id",
        drive_col,
        "posteam",
        "defteam",
    ]

    red_zone_trips = (
        df[
            df["yardline_100"].between(
                0,
                20,
                inclusive="both",
            )
        ][drive_keys]
        .drop_duplicates()
    )

    if red_zone_trips.empty:
        empty_off = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "red_zone_td_rate",
            ]
        )
        empty_def = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "red_zone_td_rate_allowed",
            ]
        )
        return empty_off, empty_def

    td_by_drive = (
        df.groupby(drive_keys, dropna=False)
        .agg(
            red_zone_drive_td=(
                "offensive_touchdown_flag",
                "max",
            )
        )
        .reset_index()
    )

    trips = red_zone_trips.merge(
        td_by_drive,
        on=drive_keys,
        how="left",
    )
    trips["red_zone_drive_td"] = (
        trips["red_zone_drive_td"].fillna(0)
    )

    off_rz = (
        trips.groupby(
            ["season", "week", "posteam"],
            dropna=False,
        )
        .agg(
            red_zone_td_rate=(
                "red_zone_drive_td",
                "mean",
            )
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_rz = (
        trips.groupby(
            ["season", "week", "defteam"],
            dropna=False,
        )
        .agg(
            red_zone_td_rate_allowed=(
                "red_zone_drive_td",
                "mean",
            )
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    return off_rz, def_rz


def merge_stat_frames(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    result: pd.DataFrame | None = None

    for frame in frames:
        if frame is None or frame.empty:
            continue

        if result is None:
            result = frame.copy()
        else:
            result = result.merge(
                frame,
                on=["season", "week", "team"],
                how="outer",
            )

    if result is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[OUTPUT_COLUMNS]
    result = result.sort_values(
        ["season", "week", "team"]
    ).reset_index(drop=True)

    return result


def build_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pbp = coerce_numeric_columns(pbp)

    valid_plays = build_valid_scrimmage_plays(pbp)

    if valid_plays.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    offense_stats = build_offense_stats(valid_plays)
    defense_stats = build_defense_stats(valid_plays)
    third_down_stats = build_third_down_stats(
        pbp,
        valid_plays,
    )
    off_points, def_points = build_drive_points_stats(pbp)
    off_rz, def_rz = build_red_zone_stats(pbp)

    return merge_stat_frames(
        [
            offense_stats,
            defense_stats,
            off_points,
            def_points,
            off_rz,
            def_rz,
            third_down_stats,
        ]
    )


def _validate_numeric_metric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    source = frame[column]
    numeric = pd.to_numeric(source, errors="coerce")
    nonblank = ~_blank_mask(source)
    invalid_text = nonblank & numeric.isna()

    if invalid_text.any():
        examples = source.loc[invalid_text].head(5).tolist()
        raise TeamStatsError(
            f"Team-stat output column {column} contains nonnumeric values: "
            f"{examples}"
        )

    nonfinite = numeric.notna() & ~np.isfinite(numeric)
    if nonfinite.any():
        examples = source.loc[nonfinite].head(5).tolist()
        raise TeamStatsError(
            f"Team-stat output column {column} contains nonfinite values: "
            f"{examples}"
        )

    return numeric


def validate_team_stats_candidate(
    frame: pd.DataFrame,
    season: int,
    *,
    allow_empty: bool,
) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise TeamStatsError(
            "Team-stat output columns do not exactly match OUTPUT_COLUMNS"
        )

    if frame.empty:
        if allow_empty:
            return
        raise TeamStatsError(
            "Nonempty PBP input produced zero team-stat rows"
        )

    season_values = pd.to_numeric(
        frame["season"],
        errors="coerce",
    )
    if (
        season_values.isna().any()
        or season_values.ne(season).any()
    ):
        examples = (
            frame.loc[
                season_values.isna()
                | season_values.ne(season),
                ["season", "week", "team"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            f"Team-stat output contains rows outside requested season "
            f"{season}: {examples}"
        )

    week_values = pd.to_numeric(
        frame["week"],
        errors="coerce",
    )
    invalid_week = (
        week_values.isna()
        | week_values.le(0)
        | week_values.mod(1).ne(0)
    )
    if invalid_week.any():
        examples = (
            frame.loc[
                invalid_week,
                ["season", "week", "team"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            f"Team-stat output contains invalid week values: {examples}"
        )

    blank_team = _blank_mask(frame["team"])
    if blank_team.any():
        examples = (
            frame.loc[
                blank_team,
                ["season", "week", "team"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            f"Team-stat output contains blank team keys: {examples}"
        )

    duplicate_mask = frame.duplicated(
        subset=["season", "week", "team"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            frame.loc[
                duplicate_mask,
                ["season", "week", "team"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise TeamStatsError(
            "Team-stat output contains duplicate season/week/team keys: "
            f"{examples}"
        )

    for column in OUTPUT_COLUMNS[3:]:
        numeric = _validate_numeric_metric(
            frame,
            column,
        )

        if column in RATE_COLUMNS:
            out_of_range = (
                numeric.notna()
                & ((numeric < 0) | (numeric > 1))
            )
            if out_of_range.any():
                examples = (
                    frame.loc[
                        out_of_range,
                        ["season", "week", "team", column],
                    ]
                    .head(5)
                    .to_dict("records")
                )
                raise TeamStatsError(
                    f"Team-stat rate column {column} is outside [0, 1]: "
                    f"{examples}"
                )


def write_candidate_file(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        frame.to_csv(
            temporary_path,
            index=False,
        )

        with temporary_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def verify_candidate_file(
    temporary_path: Path,
    expected: pd.DataFrame,
    season: int,
    *,
    allow_empty: bool,
) -> None:
    try:
        reloaded = pd.read_csv(
            temporary_path,
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise TeamStatsError(
            "Staged team-stat CSV contains no header"
        ) from exc

    if len(reloaded) != len(expected):
        raise TeamStatsError(
            "Staged team-stat row count changed during CSV round trip: "
            f"expected={len(expected)} actual={len(reloaded)}"
        )

    if list(reloaded.columns) != list(expected.columns):
        raise TeamStatsError(
            "Staged team-stat columns changed during CSV round trip"
        )

    validate_team_stats_candidate(
        reloaded,
        season,
        allow_empty=allow_empty,
    )


def publish_candidate(
    temporary_path: Path,
    output_path: Path,
) -> None:
    try:
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    season = int(args.season)
    pbp_path = PBP_DIR / f"{season}_pbp.csv.gz"
    output_path = OUTPUT_DIR / f"{season}_team_stats.csv"

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=season,
            extra_context={
                "component": "weekly team statistics",
            },
        ) as reporter:
            log = RunLog(reporter)

            pbp_rows = 0
            pbp_columns = 0
            valid_scrimmage_rows = 0
            output_rows = 0
            output_columns = len(OUTPUT_COLUMNS)
            output_weeks: list[int] = []
            output_teams: list[str] = []
            drive_column: str | None = None

            existing_output_rows: int | None = None
            existing_output_readable = True
            completed_games = 0
            result_files_checked = 0
            completed_game_inspection_uncertain = False
            empty_input_allowed = False

            staged_roundtrip_verified = False
            publication_completed = False
            temporary_path: Path | None = None

            reporter.add_input(pbp_path)
            if output_path.exists():
                reporter.add_input(output_path)

            try:
                log.info("=" * 80)
                log.info(
                    f"pull_team_stats.py started | season={season}"
                )
                log.info(f"input={pbp_path}")
                log.info(f"output={output_path}")
                log.info(f"log={LEGACY_LOG_FILE}")

                existing_output, existing_output_readable = (
                    read_existing_output(
                        output_path,
                        log,
                    )
                )
                if existing_output is not None:
                    existing_output_rows = int(
                        len(existing_output)
                    )

                pbp = read_pbp(pbp_path)
                pbp_rows = int(len(pbp))
                pbp_columns = int(len(pbp.columns))

                if pbp.empty:
                    (
                        completed_games,
                        completed_game_inspection_uncertain,
                        result_files_checked,
                    ) = count_completed_games(
                        season,
                        log,
                    )

                    existing_has_real_rows = (
                        existing_output is not None
                        and not existing_output.empty
                    )
                    unreadable_existing_may_contain_data = (
                        output_path.exists()
                        and not existing_output_readable
                        and output_path.stat().st_size > 0
                    )

                    if (
                        existing_has_real_rows
                        or unreadable_existing_may_contain_data
                        or completed_games > 0
                        or completed_game_inspection_uncertain
                    ):
                        raise TeamStatsError(
                            "PBP input is empty after team-stat data may "
                            "already be required or available; existing "
                            "canonical team-stat output was preserved"
                        )

                    empty_input_allowed = True
                    candidate = pd.DataFrame(
                        columns=OUTPUT_COLUMNS
                    )
                    log.warning(
                        "PBP input is empty before completed games and before "
                        "any readable nonempty canonical team-stat output; "
                        "publishing empty pre-game team-stat snapshot",
                        season=season,
                    )
                else:
                    drive_column = validate_pbp_input(
                        pbp,
                        season,
                    )

                    numeric_pbp = coerce_numeric_columns(pbp)
                    valid_scrimmage = (
                        build_valid_scrimmage_plays(
                            numeric_pbp
                        )
                    )
                    valid_scrimmage_rows = int(
                        len(valid_scrimmage)
                    )

                    if valid_scrimmage.empty:
                        raise TeamStatsError(
                            "Nonempty PBP input produced zero valid "
                            "pass/run scrimmage plays"
                        )

                    candidate = build_team_stats(
                        numeric_pbp
                    )

                    if candidate.empty:
                        raise TeamStatsError(
                            "Nonempty PBP input produced zero "
                            "team-stat rows"
                        )

                validate_team_stats_candidate(
                    candidate,
                    season,
                    allow_empty=empty_input_allowed,
                )

                output_rows = int(len(candidate))
                output_columns = int(
                    len(candidate.columns)
                )

                if not candidate.empty:
                    output_weeks = sorted(
                        {
                            int(value)
                            for value in pd.to_numeric(
                                candidate["week"],
                                errors="raise",
                            ).tolist()
                        }
                    )
                    output_teams = sorted(
                        {
                            str(value).strip()
                            for value in candidate[
                                "team"
                            ].tolist()
                            if str(value).strip()
                        }
                    )

                temporary_path = write_candidate_file(
                    candidate,
                    output_path,
                )

                verify_candidate_file(
                    temporary_path,
                    candidate,
                    season,
                    allow_empty=empty_input_allowed,
                )
                staged_roundtrip_verified = True

                publish_candidate(
                    temporary_path,
                    output_path,
                )
                temporary_path = None
                publication_completed = True
                reporter.add_output(output_path)

                log.info(f"pbp_rows={pbp_rows}")
                log.info(
                    f"valid_scrimmage_rows={valid_scrimmage_rows}"
                )
                log.info(f"output_rows={output_rows}")
                log.info(
                    f"output_columns={output_columns}"
                )
                log.info(
                    f"publication_completed={str(publication_completed).lower()}"
                )
                log.info("status=success")
                log.info("=" * 80)

            except Exception as exc:
                log.error_text(
                    f"{type(exc).__name__}: {exc}"
                )
                for line in (
                    traceback.format_exc()
                    .rstrip()
                    .splitlines()
                ):
                    log.error_text(line)
                raise
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(
                        missing_ok=True
                    )

                reporter.set_rows(
                    rows_in=pbp_rows,
                    rows_out=(
                        output_rows
                        if publication_completed
                        else 0
                    ),
                )
                reporter.update_details(
                    {
                        "pbp_rows": pbp_rows,
                        "pbp_columns": pbp_columns,
                        "valid_scrimmage_rows": valid_scrimmage_rows,
                        "drive_column_used": drive_column,
                        "output_rows": output_rows,
                        "output_columns": output_columns,
                        "output_weeks": output_weeks,
                        "output_week_count": len(output_weeks),
                        "output_teams": output_teams,
                        "output_team_count": len(output_teams),
                        "existing_output_rows": existing_output_rows,
                        "existing_output_readable": existing_output_readable,
                        "completed_games_detected_for_empty_check": completed_games,
                        "result_files_checked_for_empty_safety": result_files_checked,
                        "completed_game_inspection_uncertain": (
                            completed_game_inspection_uncertain
                        ),
                        "empty_input_allowed": empty_input_allowed,
                        "staged_roundtrip_verified": staged_roundtrip_verified,
                        "publication_completed": publication_completed,
                    }
                )

                log.write_legacy()
                reporter.add_output(
                    LEGACY_LOG_FILE
                )

        return 0

    except Exception as exc:
        print(
            "nfl pull_team_stats failed",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
