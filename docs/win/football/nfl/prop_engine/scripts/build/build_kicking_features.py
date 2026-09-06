#!/usr/bin/env python3
"""
Build historical kicking-specific player-game features.

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/kicking_features.parquet

GRAIN:
    K/PK player rows at season + week + game_id + player_id

POLICY:
    - All rolling/player history is already leakage-safe.
    - Career/season FG and PAT make percentages are prior makes divided by
      prior attempts; zero prior attempts -> null.
    - Team offense comes from already-lagged team_form.
    - Opponent defense comes from already-lagged opponent_form.
    - Weather comes from historical environment.
    - primary_kicker_flag uses only pregame roster/depth/injury state and
      strictly prior kicking usage. It never uses the target game's result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import sys

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
    "player_id",
    "team",
    "fg_attempts_lag1",
    "fg_attempts_roll3",
    "fg_attempts_roll5",
    "fg_make_pct_career_prior",
    "fg_make_pct_season_prior",
    "pat_attempts_roll3",
    "pat_make_pct_career_prior",
    "team_drives_roll3",
    "team_points_per_drive_roll3",
    "team_red_zone_td_rate_roll3",
    "opponent_points_per_drive_allowed_roll3",
    "opponent_red_zone_td_rate_allowed_roll3",
    "temperature",
    "wind",
    "roof",
    "surface",
    "primary_kicker_flag",
]

GRAIN = [
    "season",
    "week",
    "game_id",
    "player_id",
]

TEAM_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

PLAYER_FORM_COLUMNS = [
    "field_goal_attempts_lag1",
    "field_goal_attempts_roll3_mean",
    "field_goal_attempts_roll5_mean",
    "field_goal_attempts_season_to_date",
    "field_goal_attempts_career_prior",
    "field_goals_made_season_to_date",
    "field_goals_made_career_prior",
    "extra_point_attempts_roll3_mean",
    "extra_point_attempts_career_prior",
    "extra_points_made_career_prior",
]

TEAM_FORM_MAP = {
    "team_drives_roll3": "drives_roll3_mean",
    "team_points_per_drive_roll3": "points_per_drive_roll3_mean",
    "team_red_zone_td_rate_roll3": "red_zone_td_rate_roll3_mean",
}

OPPONENT_FORM_MAP = {
    "opponent_points_per_drive_allowed_roll3":
        "points_per_drive_allowed_roll3_mean",
    "opponent_red_zone_td_rate_allowed_roll3":
        "red_zone_td_rate_allowed_roll3_mean",
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


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return TEAM_ALIASES.get(team, team)


def numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = numeric(numerator)
    den = numeric(denominator)

    result = pd.Series(
        np.nan,
        index=num.index,
        dtype="float64",
    )

    valid = (
        num.notna()
        & den.notna()
        & den.gt(0.0)
    )

    result.loc[valid] = (
        num.loc[valid]
        / den.loc[valid]
    )

    return result


def depth_role_score(value: Any) -> int:
    """
    Pregame preference for field-goal/PAT responsibility.

    Explicit FG/K/PK depth roles beat kickoff-only, punter, or holder roles.
    Blank/unknown roles remain usable as fallbacks.
    """
    text = clean(value).upper()
    compact = re.sub(r"\s+", "", text)

    if not compact:
        return 1

    tail = compact.split("|")[-1]

    if tail in {"FG", "K", "PK"}:
        return 5

    if "FG" in tail:
        return 5

    if tail in {"K/KO", "KICKER"}:
        return 4

    if tail in {"KO", "KOS"}:
        return 2

    if tail in {"P", "H"}:
        return 0

    return 1


def build_primary_kicker_flag(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Select exactly one primary kicker per team-game using pregame information.

    Priority:
      1. not listed Out
      2. currently rostered
      3. current depth presence
      4. field-goal/PAT-oriented depth role
      5. depth starter
      6. better depth rank
      7. greater recent/prior kicking usage
      8. deterministic player_id tie-break
    """
    work = frame.copy()

    work["_not_out"] = (
        numeric(work["injury_out_flag"])
        .fillna(0)
        .eq(0)
        .astype(int)
    )

    work["_roster"] = (
        numeric(work["roster_flag"])
        .fillna(0)
        .astype(int)
    )

    work["_depth_present"] = (
        numeric(work["depth_present_flag"])
        .fillna(0)
        .astype(int)
    )

    work["_role_score"] = (
        work["depth_slot"]
        .map(depth_role_score)
        .astype(int)
    )

    work["_starter"] = (
        numeric(work["depth_starter_flag"])
        .fillna(0)
        .astype(int)
    )

    rank = numeric(work["depth_rank"])
    work["_rank_score"] = (
        -rank.fillna(999.0)
    )

    usage_cols = [
        "field_goal_attempts_roll3_mean",
        "extra_point_attempts_roll3_mean",
        "field_goal_attempts_lag1",
        "field_goal_attempts_career_prior",
        "extra_point_attempts_career_prior",
    ]

    for column in usage_cols:
        work[f"_usage_{column}"] = (
            numeric(work[column])
            .fillna(-1.0)
        )

    group_cols = [
        "season",
        "week",
        "game_id",
        "team",
    ]

    sort_cols = [
        *group_cols,
        "_not_out",
        "_roster",
        "_depth_present",
        "_role_score",
        "_starter",
        "_rank_score",
        *[
            f"_usage_{column}"
            for column in usage_cols
        ],
        "player_id",
    ]

    ascending = (
        [True] * len(group_cols)
        + [False] * (
            6 + len(usage_cols)
        )
        + [True]
    )

    ranked = work.sort_values(
        sort_cols,
        ascending=ascending,
        kind="mergesort",
    )

    chosen = (
        ranked.groupby(
            group_cols,
            sort=False,
            as_index=False,
        )
        .head(1)
        .index
    )

    flag = pd.Series(
        0,
        index=frame.index,
        dtype="int8",
    )

    flag.loc[chosen] = 1

    return flag


def main() -> None:
    config = common.load_config()
    paths = config["paths"]

    universe = common.read_parquet_required(
        paths["historical_universe"],
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
            "roster_flag",
            "depth_present_flag",
            "depth_rank",
            "depth_slot",
            "depth_starter_flag",
        ],
    )

    player_form = common.read_parquet_required(
        paths["player_form"],
        [
            *GRAIN,
            *PLAYER_FORM_COLUMNS,
        ],
    )

    role_history = common.read_parquet_required(
        paths["role_history"],
        [
            *GRAIN,
            "injury_out_flag",
        ],
    )

    team_form = common.read_parquet_required(
        paths["team_form"],
        [
            "season",
            "week",
            "team",
            *TEAM_FORM_MAP.values(),
        ],
    )

    opponent_form = common.read_parquet_required(
        paths["opponent_form"],
        [
            "season",
            "week",
            "team",
            *OPPONENT_FORM_MAP.values(),
        ],
    )

    environment = common.read_parquet_required(
        paths["environment_history"],
        [
            "season",
            "week",
            "game_id",
            "temperature",
            "wind",
            "roof",
            "surface",
        ],
    )

    common.ensure_unique(
        universe,
        GRAIN,
        "historical universe",
    )

    common.ensure_unique(
        player_form,
        GRAIN,
        "player form",
    )

    common.ensure_unique(
        role_history,
        GRAIN,
        "role history",
    )

    common.ensure_unique(
        team_form,
        ["season", "week", "team"],
        "team form",
    )

    common.ensure_unique(
        opponent_form,
        ["season", "week", "team"],
        "opponent form",
    )

    common.ensure_unique(
        environment,
        ["season", "week", "game_id"],
        "environment",
    )

    position = (
        universe["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    kickers = universe.loc[
        position.isin(["K", "PK"]),
        [
            *GRAIN,
            "team",
            "opponent",
            "roster_flag",
            "depth_present_flag",
            "depth_rank",
            "depth_slot",
            "depth_starter_flag",
        ],
    ].copy()

    if kickers.empty:
        raise RuntimeError(
            "No historical K/PK player rows found."
        )

    pf = player_form[
        [
            *GRAIN,
            *PLAYER_FORM_COLUMNS,
        ]
    ].copy()

    kickers = kickers.merge(
        pf,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    injury = role_history[
        [
            *GRAIN,
            "injury_out_flag",
        ]
    ].copy()

    kickers = kickers.merge(
        injury,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    # Team offensive form.
    tf = team_form[
        [
            "season",
            "week",
            "team",
            *TEAM_FORM_MAP.values(),
        ]
    ].copy()

    tf["team"] = tf["team"].map(
        canonical_team
    )

    tf = tf.rename(
        columns={
            "team": "_team_key",
            **{
                source: target
                for target, source
                in TEAM_FORM_MAP.items()
            },
        }
    )

    kickers["_team_key"] = (
        kickers["team"]
        .map(canonical_team)
    )

    kickers = kickers.merge(
        tf,
        on=[
            "season",
            "week",
            "_team_key",
        ],
        how="left",
        validate="many_to_one",
    )

    # Opponent defensive form.
    of = opponent_form[
        [
            "season",
            "week",
            "team",
            *OPPONENT_FORM_MAP.values(),
        ]
    ].copy()

    of["team"] = of["team"].map(
        canonical_team
    )

    of = of.rename(
        columns={
            "team": "_opponent_key",
            **{
                source: target
                for target, source
                in OPPONENT_FORM_MAP.items()
            },
        }
    )

    kickers["_opponent_key"] = (
        kickers["opponent"]
        .map(canonical_team)
    )

    kickers = kickers.merge(
        of,
        on=[
            "season",
            "week",
            "_opponent_key",
        ],
        how="left",
        validate="many_to_one",
    )

    env = environment[
        [
            "season",
            "week",
            "game_id",
            "temperature",
            "wind",
            "roof",
            "surface",
        ]
    ].copy()

    kickers = kickers.merge(
        env,
        on=[
            "season",
            "week",
            "game_id",
        ],
        how="left",
        validate="many_to_one",
    )

    kickers["fg_attempts_lag1"] = (
        numeric(
            kickers[
                "field_goal_attempts_lag1"
            ]
        )
    )

    kickers["fg_attempts_roll3"] = (
        numeric(
            kickers[
                "field_goal_attempts_roll3_mean"
            ]
        )
    )

    kickers["fg_attempts_roll5"] = (
        numeric(
            kickers[
                "field_goal_attempts_roll5_mean"
            ]
        )
    )

    kickers["fg_make_pct_career_prior"] = (
        safe_ratio(
            kickers[
                "field_goals_made_career_prior"
            ],
            kickers[
                "field_goal_attempts_career_prior"
            ],
        )
    )

    kickers["fg_make_pct_season_prior"] = (
        safe_ratio(
            kickers[
                "field_goals_made_season_to_date"
            ],
            kickers[
                "field_goal_attempts_season_to_date"
            ],
        )
    )

    kickers["pat_attempts_roll3"] = (
        numeric(
            kickers[
                "extra_point_attempts_roll3_mean"
            ]
        )
    )

    kickers["pat_make_pct_career_prior"] = (
        safe_ratio(
            kickers[
                "extra_points_made_career_prior"
            ],
            kickers[
                "extra_point_attempts_career_prior"
            ],
        )
    )

    kickers["primary_kicker_flag"] = (
        build_primary_kicker_flag(
            kickers
        )
    )

    output = kickers[
        OUTPUT_COLUMNS
    ].copy()

    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(
            "Issue 16 output header/order mismatch."
        )

    common.ensure_unique(
        output,
        GRAIN,
        "Issue 16 kicking feature grain",
    )

    if not output[
        "primary_kicker_flag"
    ].isin([0, 1]).all():
        raise ValueError(
            "primary_kicker_flag contains non-binary values."
        )

    primary_counts = (
        output.groupby(
            [
                "season",
                "week",
                "game_id",
                "team",
            ]
        )["primary_kicker_flag"]
        .sum()
    )

    if not primary_counts.eq(1).all():
        bad = primary_counts[
            ~primary_counts.eq(1)
        ].head(20)

        raise ValueError(
            "Expected exactly one primary kicker per "
            f"kicker-present team-game: {bad.to_dict()}"
        )

    for column in [
        "fg_make_pct_career_prior",
        "fg_make_pct_season_prior",
        "pat_make_pct_career_prior",
    ]:
        values = numeric(output[column])

        bad = (
            values.notna()
            & (
                values.lt(0.0)
                | values.gt(1.0)
            )
        )

        if bad.any():
            raise ValueError(
                f"{column}: outside [0,1]."
            )

    numeric_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column
        not in {
            "game_id",
            "player_id",
            "team",
            "roof",
            "surface",
        }
    ]

    matrix = (
        output[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype="float64")
    )

    if np.isinf(matrix).any():
        raise ValueError(
            "Issue 16 output contains infinity."
        )

    output_path = paths.get(
        "kicking_features",
        "docs/win/football/nfl/prop_engine/data/historical/features/kicking_features.parquet",
    )

    common.write_parquet_atomic(
        output,
        output_path,
    )

    payload = {
        "status": "passed",
        "output": str(output_path),
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "team_games": int(
            output[
                [
                    "season",
                    "week",
                    "game_id",
                    "team",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "primary_kickers": int(
            output[
                "primary_kicker_flag"
            ].sum()
        ),
        "primary_kicker_policy": (
            "pregame roster/depth/injury state first; "
            "field-goal role then strictly prior kicking usage"
        ),
        "same_game_result_used": False,
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
