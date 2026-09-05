#!/usr/bin/env python3
"""
Build strictly lagged historical team and opponent form features.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/data/historical/opportunity/team_week_opportunity.parquet
    docs/win/football/nfl/prop_engine/data/historical/opportunity/opponent_week_opportunity.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/team_form.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/opponent_form.parquet

POLICY:
    - All form values use only earlier realized games for the same canonical franchise.
    - Rolling windows use the last N observed prior metric values.
    - EWM history continues across seasons for the same canonical franchise.
    - season_to_date resets each season and uses only observed prior values.
    - Missing historical source values are not converted to zero and do not update state.
    - No same-week realized value may enter that week's feature row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


TEAM_METRICS = [
    "offensive_plays",
    "drives",
    "dropbacks",
    "pass_attempts",
    "rush_attempts",
    "pass_rate",
    "rush_rate",
    "points_per_drive",
    "red_zone_drives",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "field_goal_attempts",
    "extra_point_attempts",
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPPONENT_METRICS = [
    "defensive_plays",
    "opponent_dropbacks",
    "opponent_pass_attempts",
    "opponent_rush_attempts",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "passing_tds_allowed",
    "rushing_tds_allowed",
    "sacks",
    "qb_hits",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

SUFFIXES = [
    "lag1",
    "roll3_mean",
    "roll5_mean",
    "roll8_mean",
    "ewm3",
    "ewm5",
    "season_to_date",
]

GRAIN = ["season", "week", "team"]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def feature_columns(metrics: list[str]) -> list[str]:
    return [
        f"{metric}_{suffix}"
        for metric in metrics
        for suffix in SUFFIXES
    ]


def output_columns(metrics: list[str]) -> list[str]:
    return GRAIN + feature_columns(metrics)


def numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def update_ewm(previous: float | None, value: float, span: int) -> float:
    alpha = 2.0 / (span + 1.0)

    if previous is None:
        return float(value)

    return float(
        alpha * value
        + (1.0 - alpha) * previous
    )


def normalize_source(
    source: pd.DataFrame,
    metrics: list[str],
    label: str,
) -> pd.DataFrame:
    common.require_columns(
        source,
        GRAIN + metrics,
        label,
    )

    result = source[GRAIN + metrics].copy()

    result["season"] = pd.to_numeric(
        result["season"],
        errors="raise",
    ).astype(int)

    result["week"] = pd.to_numeric(
        result["week"],
        errors="raise",
    ).astype(int)

    result["team"] = result["team"].map(canonical_team)

    if result["team"].eq("").any():
        raise ValueError(f"{label}: blank canonical team.")

    for metric in metrics:
        result[metric] = common.safe_numeric(result[metric]).astype("float64")

    common.ensure_unique(
        result,
        GRAIN,
        f"{label} canonical grain",
    )

    return (
        result.sort_values(
            ["team", "season", "week"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_form(
    source: pd.DataFrame,
    metrics: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    source = normalize_source(
        source,
        metrics,
        label,
    )

    result = source[GRAIN].copy()

    by_team = source.groupby(
        "team",
        sort=False,
    ).indices

    for metric_index, metric in enumerate(metrics, start=1):
        print(
            f"ISSUE13 {label} metric "
            f"{metric_index:02d}/{len(metrics):02d}: {metric}"
        )

        lag1 = np.full(len(source), np.nan, dtype="float64")
        roll3 = np.full(len(source), np.nan, dtype="float64")
        roll5 = np.full(len(source), np.nan, dtype="float64")
        roll8 = np.full(len(source), np.nan, dtype="float64")
        ewm3 = np.full(len(source), np.nan, dtype="float64")
        ewm5 = np.full(len(source), np.nan, dtype="float64")
        std_mean = np.full(len(source), np.nan, dtype="float64")

        values = source[metric].to_numpy(dtype="float64")
        seasons = source["season"].to_numpy(dtype="int64")

        for _, positions in by_team.items():
            history: list[float] = []
            season_history: list[float] = []
            current_season: int | None = None
            state_ewm3: float | None = None
            state_ewm5: float | None = None

            for pos in positions:
                season = int(seasons[pos])

                if current_season != season:
                    current_season = season
                    season_history = []

                if history:
                    lag1[pos] = history[-1]
                    roll3[pos] = mean_or_nan(history[-3:])
                    roll5[pos] = mean_or_nan(history[-5:])
                    roll8[pos] = mean_or_nan(history[-8:])

                if state_ewm3 is not None:
                    ewm3[pos] = state_ewm3

                if state_ewm5 is not None:
                    ewm5[pos] = state_ewm5

                if season_history:
                    std_mean[pos] = mean_or_nan(season_history)

                current = numeric_value(values[pos])

                if current is None:
                    continue

                history.append(current)
                season_history.append(current)
                state_ewm3 = update_ewm(
                    state_ewm3,
                    current,
                    3,
                )
                state_ewm5 = update_ewm(
                    state_ewm5,
                    current,
                    5,
                )

        result[f"{metric}_lag1"] = lag1
        result[f"{metric}_roll3_mean"] = roll3
        result[f"{metric}_roll5_mean"] = roll5
        result[f"{metric}_roll8_mean"] = roll8
        result[f"{metric}_ewm3"] = ewm3
        result[f"{metric}_ewm5"] = ewm5
        result[f"{metric}_season_to_date"] = std_mean

    expected = output_columns(metrics)

    if list(result.columns) != expected:
        raise RuntimeError(
            f"{label}: output column order mismatch."
        )

    common.ensure_unique(
        result,
        GRAIN,
        f"{label} form grain",
    )

    if len(result) != len(source):
        raise RuntimeError(
            f"{label}: output row count changed."
        )

    feature_values = (
        result[feature_columns(metrics)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64")
    )

    if np.isinf(feature_values).any():
        raise ValueError(
            f"{label}: output contains infinity."
        )

    common.reject_forbidden_feature_columns(
        result.columns,
        CONFIG,
    )

    return result


def validate_config(config: dict) -> None:
    windows = (
        config.get("rolling_windows", {})
        .get("games", [])
    )
    spans = (
        config.get("ewm", {})
        .get("spans", [])
    )
    adjust = (
        config.get("ewm", {})
        .get("adjust")
    )

    for required_window in (3, 5, 8):
        if required_window not in windows:
            raise ValueError(
                "Issue 13 requires rolling window "
                f"{required_window}; config has {windows}."
            )

    for required_span in (3, 5):
        if required_span not in spans:
            raise ValueError(
                "Issue 13 requires EWM span "
                f"{required_span}; config has {spans}."
            )

    if adjust is not False:
        raise ValueError(
            "Issue 13 requires ewm.adjust: false."
        )


def resolve_path(
    config: dict,
    key: str,
    fallback: str,
) -> str:
    value = config.get("paths", {}).get(key)

    if value:
        return str(value)

    return fallback


CONFIG = common.load_config()


def main() -> None:
    validate_config(CONFIG)

    paths = CONFIG["paths"]

    team_source_path = paths["team_opportunity"]
    opponent_source_path = paths["opponent_opportunity"]

    team_output_path = resolve_path(
        CONFIG,
        "team_form",
        "docs/win/football/nfl/prop_engine/data/historical/features/team_form.parquet",
    )
    opponent_output_path = resolve_path(
        CONFIG,
        "opponent_form",
        "docs/win/football/nfl/prop_engine/data/historical/features/opponent_form.parquet",
    )

    team_source = common.read_parquet_required(
        team_source_path,
        GRAIN + TEAM_METRICS,
    )
    opponent_source = common.read_parquet_required(
        opponent_source_path,
        GRAIN + OPPONENT_METRICS,
    )

    team_keys = set(
        map(
            tuple,
            team_source[GRAIN].to_numpy(),
        )
    )
    opponent_keys = set(
        map(
            tuple,
            opponent_source[GRAIN].to_numpy(),
        )
    )

    if team_keys != opponent_keys:
        raise ValueError(
            "Issue 13 source team/opponent key sets differ."
        )

    team_form = build_form(
        team_source,
        TEAM_METRICS,
        label="team",
    )

    opponent_form = build_form(
        opponent_source,
        OPPONENT_METRICS,
        label="opponent",
    )

    common.write_parquet_atomic(
        team_form,
        team_output_path,
    )
    common.write_parquet_atomic(
        opponent_form,
        opponent_output_path,
    )

    payload = {
        "status": "passed",
        "team_output": team_output_path,
        "opponent_output": opponent_output_path,
        "rows_each": int(len(team_form)),
        "team_base_metrics": int(len(TEAM_METRICS)),
        "opponent_base_metrics": int(len(OPPONENT_METRICS)),
        "suffixes": list(SUFFIXES),
        "team_feature_columns": int(len(feature_columns(TEAM_METRICS))),
        "opponent_feature_columns": int(len(feature_columns(OPPONENT_METRICS))),
        "team_total_columns": int(len(team_form.columns)),
        "opponent_total_columns": int(len(opponent_form.columns)),
        "rolling_policy": "last N observed strictly prior team games",
        "ewm_policy": "strictly prior observed games; adjust=false; franchise history continues across seasons",
        "season_to_date_policy": "mean of observed prior values in current season only",
        "missing_policy": "missing source values remain missing and do not update rolling state",
        "same_week_realized_forbidden": True,
        "franchise_aliases": HISTORICAL_FRANCHISE_ALIASES,
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
