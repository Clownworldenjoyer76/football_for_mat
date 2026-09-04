#!/usr/bin/env python3
"""
Build weekly opponent position-allowed statistics.

READS:
    docs/win/football/nfl/prop_engine/data/historical/opportunity/player_week_opportunity.parquet
    docs/win/football/nfl/prop_engine/data/historical/opportunity/team_week_opportunity.parquet
    docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/opportunity/position_allowed_week.parquet

RATE CONTRACT:
    QB:
        numerator = passing_yards_allowed
        sample = pass attempts faced

    RB / WR / TE:
        numerator = receiving_yards_allowed + rushing_yards_allowed
        sample = targets_allowed + carries_allowed

    league_rate is the same-season/week weighted league positional rate.

    shrunk_rate =
        (raw_numerator + prior_sample_size * league_rate)
        / (raw_rate_sample_size + prior_sample_size)

TACKLES GENERATED:
    Available only for PBP-covered seasons.
    Count unique defensive tackle credits on:
      - sacks, attributed to the passer
      - completed passes, attributed to the receiver
      - rush attempts, attributed to the rusher

    Tackle-credit IDs:
      solo_tackle_1_player_id
      solo_tackle_2_player_id
      assist_tackle_1_player_id
      assist_tackle_2_player_id
      assist_tackle_3_player_id
      assist_tackle_4_player_id

LEAKAGE:
    This is an immutable same-week realized table. Lag before model use.
    Never use raw early-season unshrunk rates as production features.
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


OUTPUT_COLUMNS = [
    "season",
    "week",
    "defense_team",
    "offense_position_group",
    "players_faced",
    "targets_allowed",
    "receptions_allowed",
    "receiving_yards_allowed",
    "receiving_tds_allowed",
    "carries_allowed",
    "rushing_yards_allowed",
    "rushing_tds_allowed",
    "passing_yards_allowed",
    "passing_tds_allowed",
    "tackles_generated",
    "raw_rate_sample_size",
    "league_rate",
    "shrunk_rate",
]

SUPPORTED_GROUPS = ["QB", "RB", "WR", "TE"]

GRAIN = [
    "season",
    "week",
    "defense_team",
    "offense_position_group",
]

TEAM_WEEK_GRAIN = [
    "season",
    "week",
    "team",
]

COUNT_COLUMNS = [
    "players_faced",
    "targets_allowed",
    "receptions_allowed",
    "receiving_tds_allowed",
    "carries_allowed",
    "rushing_tds_allowed",
    "passing_tds_allowed",
    "tackles_generated",
    "raw_rate_sample_size",
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

TACKLE_ID_COLUMNS = [
    "solo_tackle_1_player_id",
    "solo_tackle_2_player_id",
    "assist_tackle_1_player_id",
    "assist_tackle_2_player_id",
    "assist_tackle_3_player_id",
    "assist_tackle_4_player_id",
]


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


def defense_from_game_id(
    game_id: Any,
    offense_team: Any,
) -> str:
    text = clean(game_id)
    match = GAME_ID_RE.fullmatch(text)

    if not match:
        raise ValueError(
            f"Unsupported nflverse game_id: {game_id!r}"
        )

    away = canonical_team(match.group("away"))
    home = canonical_team(match.group("home"))
    offense = canonical_team(offense_team)

    if offense == away:
        return home

    if offense == home:
        return away

    raise ValueError(
        f"Offense team {offense_team!r} does not belong to "
        f"game_id {game_id!r} after team normalization."
    )


def numeric_series(
    series: pd.Series,
    *,
    label: str,
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

    return converted.astype(float)


def get_position_allowed_config(
    config: dict,
) -> tuple[list[str], float]:
    positions = config.get("positions")

    if not isinstance(positions, dict):
        raise ValueError(
            "Config section 'positions' must be a mapping."
        )

    block = positions.get("position_allowed")

    if not isinstance(block, dict):
        raise ValueError(
            "Config missing positions.position_allowed mapping."
        )

    groups = block.get("supported_groups")
    prior = block.get("prior_sample_size")

    if groups != SUPPORTED_GROUPS:
        raise ValueError(
            "positions.position_allowed.supported_groups must equal "
            f"{SUPPORTED_GROUPS}; got {groups!r}"
        )

    try:
        prior_value = float(prior)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "positions.position_allowed.prior_sample_size "
            "must be numeric."
        ) from exc

    if not math.isfinite(prior_value) or prior_value <= 0:
        raise ValueError(
            "positions.position_allowed.prior_sample_size "
            "must be finite and > 0."
        )

    return list(groups), prior_value


def build_dense_grid(
    team_week: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    common.require_columns(
        team_week,
        TEAM_WEEK_GRAIN,
        "team weekly opportunity",
    )

    base = team_week[TEAM_WEEK_GRAIN].copy()

    base["season"] = pd.to_numeric(
        base["season"],
        errors="raise",
    ).astype(int)

    base["week"] = pd.to_numeric(
        base["week"],
        errors="raise",
    ).astype(int)

    base["defense_team"] = (
        base["team"]
        .map(canonical_team)
    )

    base = (
        base[
            [
                "season",
                "week",
                "defense_team",
            ]
        ]
        .drop_duplicates()
    )

    group_frame = pd.DataFrame(
        {
            "offense_position_group": groups,
        }
    )

    base["_join"] = 1
    group_frame["_join"] = 1

    grid = (
        base.merge(
            group_frame,
            on="_join",
            how="inner",
        )
        .drop(
            columns="_join"
        )
    )

    common.ensure_unique(
        grid,
        GRAIN,
        "position-allowed dense grid",
    )

    return grid


def build_allowed_totals(
    player_opportunity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "season",
        "week",
        "game_id",
        "player_id",
        "team",
        "position_group",
        "pass_attempts",
        "passing_yards",
        "passing_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ]

    common.require_columns(
        player_opportunity,
        required,
        "player weekly opportunity",
    )

    working = player_opportunity.loc[
        player_opportunity[
            "position_group"
        ].isin(
            SUPPORTED_GROUPS
        )
    ].copy()

    if working.empty:
        raise RuntimeError(
            "No supported offensive position groups found."
        )

    working["season"] = pd.to_numeric(
        working["season"],
        errors="raise",
    ).astype(int)

    working["week"] = pd.to_numeric(
        working["week"],
        errors="raise",
    ).astype(int)

    working["player_id"] = (
        working["player_id"]
        .map(common.normalize_player_id)
    )

    working["team"] = (
        working["team"]
        .map(canonical_team)
    )

    working["position_group"] = (
        working["position_group"]
        .astype(str)
        .str.upper()
    )

    numeric_columns = [
        "pass_attempts",
        "passing_yards",
        "passing_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ]

    for column in numeric_columns:
        working[column] = (
            numeric_series(
                working[column],
                label=f"player opportunity:{column}",
            )
            .fillna(0.0)
        )

    working["defense_team"] = [
        defense_from_game_id(
            game_id,
            offense_team,
        )
        for game_id, offense_team in zip(
            working["game_id"],
            working["team"],
        )
    ]

    activity = (
        working["pass_attempts"].ne(0.0)
        | working["carries"].ne(0.0)
        | working["targets"].ne(0.0)
        | working["receptions"].ne(0.0)
        | working["passing_yards"].ne(0.0)
        | working["rushing_yards"].ne(0.0)
        | working["receiving_yards"].ne(0.0)
        | working["passing_tds"].ne(0.0)
        | working["rushing_tds"].ne(0.0)
        | working["receiving_tds"].ne(0.0)
    )

    working["_player_faced_id"] = (
        working["player_id"]
        .where(
            activity,
            "",
        )
    )

    totals = (
        working.groupby(
            [
                "season",
                "week",
                "defense_team",
                "position_group",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            players_faced=(
                "_player_faced_id",
                lambda s: int(
                    s[s.ne("")].nunique()
                ),
            ),
            targets_allowed=(
                "targets",
                "sum",
            ),
            receptions_allowed=(
                "receptions",
                "sum",
            ),
            receiving_yards_allowed=(
                "receiving_yards",
                "sum",
            ),
            receiving_tds_allowed=(
                "receiving_tds",
                "sum",
            ),
            carries_allowed=(
                "carries",
                "sum",
            ),
            rushing_yards_allowed=(
                "rushing_yards",
                "sum",
            ),
            rushing_tds_allowed=(
                "rushing_tds",
                "sum",
            ),
            passing_yards_allowed=(
                "passing_yards",
                "sum",
            ),
            passing_tds_allowed=(
                "passing_tds",
                "sum",
            ),
            _pass_attempts_faced=(
                "pass_attempts",
                "sum",
            ),
        )
        .rename(
            columns={
                "position_group": "offense_position_group",
            }
        )
    )

    position_map = (
        working[
            [
                "season",
                "week",
                "game_id",
                "player_id",
                "position_group",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "position_group": "offense_position_group",
            }
        )
    )

    common.ensure_unique(
        position_map,
        [
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        "supported player position map",
    )

    return totals, position_map


def tackle_credit_count(
    row: pd.Series,
) -> int:
    defenders = {
        clean(row[column])
        for column in TACKLE_ID_COLUMNS
        if clean(row[column])
    }

    return len(defenders)


def build_tackles_generated(
    *,
    repo: Path,
    config: dict,
    position_map: pd.DataFrame,
    rich_start: int,
    end_season: int,
) -> pd.DataFrame:
    records = []

    usecols = [
        "season",
        "season_type",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "rush_attempt",
        "complete_pass",
        "sack",
        "rusher_player_id",
        "receiver_player_id",
        "passer_player_id",
        *TACKLE_ID_COLUMNS,
    ]

    for season in range(
        rich_start,
        end_season + 1,
    ):
        path = (
            repo
            / config["paths"]["pbp_pattern"].format(
                season=season
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Rich-season PBP missing: {path}"
            )

        pbp = pd.read_csv(
            path,
            usecols=usecols,
            low_memory=False,
        )

        pbp = pbp.loc[
            pbp["season_type"]
            .astype(str)
            .str.upper()
            .eq("REG")
        ].copy()

        pbp["season"] = season

        pbp["week"] = pd.to_numeric(
            pbp["week"],
            errors="raise",
        ).astype(int)

        pbp["game_id"] = (
            pbp["game_id"]
            .map(clean)
        )

        pbp["posteam"] = (
            pbp["posteam"]
            .map(canonical_team)
        )

        pbp["defteam"] = (
            pbp["defteam"]
            .map(canonical_team)
        )

        for column in [
            "rush_attempt",
            "complete_pass",
            "sack",
        ]:
            pbp[column] = (
                pd.to_numeric(
                    pbp[column],
                    errors="coerce",
                )
                .fillna(0.0)
            )

        for column in [
            "rusher_player_id",
            "receiver_player_id",
            "passer_player_id",
            *TACKLE_ID_COLUMNS,
        ]:
            pbp[column] = (
                pbp[column]
                .map(common.normalize_player_id)
            )

        pbp["_offense_player_id"] = ""

        sack_mask = (
            pbp["sack"].eq(1.0)
            & pbp["passer_player_id"].ne("")
        )

        completion_mask = (
            ~sack_mask
            & pbp["complete_pass"].eq(1.0)
            & pbp["receiver_player_id"].ne("")
        )

        rush_mask = (
            ~sack_mask
            & ~completion_mask
            & pbp["rush_attempt"].eq(1.0)
            & pbp["rusher_player_id"].ne("")
        )

        pbp.loc[
            sack_mask,
            "_offense_player_id",
        ] = pbp.loc[
            sack_mask,
            "passer_player_id",
        ]

        pbp.loc[
            completion_mask,
            "_offense_player_id",
        ] = pbp.loc[
            completion_mask,
            "receiver_player_id",
        ]

        pbp.loc[
            rush_mask,
            "_offense_player_id",
        ] = pbp.loc[
            rush_mask,
            "rusher_player_id",
        ]

        pbp["_tackle_credits"] = (
            pbp.apply(
                tackle_credit_count,
                axis=1,
            )
        )

        pbp = pbp.loc[
            pbp["_offense_player_id"].ne("")
            & pbp["_tackle_credits"].gt(0)
            & pbp["defteam"].ne("")
        ].copy()

        if pbp.empty:
            continue

        season_map = position_map.loc[
            position_map["season"].eq(
                season
            )
        ].copy()

        pbp = pbp.merge(
            season_map,
            left_on=[
                "season",
                "week",
                "game_id",
                "_offense_player_id",
            ],
            right_on=[
                "season",
                "week",
                "game_id",
                "player_id",
            ],
            how="left",
            validate="many_to_one",
        )

        pbp = pbp.loc[
            pbp["offense_position_group"].isin(
                SUPPORTED_GROUPS
            )
        ].copy()

        if pbp.empty:
            continue

        grouped = (
            pbp.groupby(
                [
                    "season",
                    "week",
                    "defteam",
                    "offense_position_group",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                tackles_generated=(
                    "_tackle_credits",
                    "sum",
                )
            )
            .rename(
                columns={
                    "defteam": "defense_team",
                }
            )
        )

        records.append(
            grouped
        )

    if not records:
        return pd.DataFrame(
            columns=[
                *GRAIN,
                "tackles_generated",
            ]
        )

    output = pd.concat(
        records,
        ignore_index=True,
    )

    common.ensure_unique(
        output,
        GRAIN,
        "tackles-generated grain",
    )

    return output


def add_rate_contract(
    frame: pd.DataFrame,
    *,
    prior_sample_size: float,
) -> pd.DataFrame:
    output = frame.copy()

    qb = (
        output[
            "offense_position_group"
        ].eq("QB")
    )

    output["_raw_numerator"] = (
        output["receiving_yards_allowed"]
        + output["rushing_yards_allowed"]
    )

    output["raw_rate_sample_size"] = (
        output["targets_allowed"]
        + output["carries_allowed"]
    )

    output.loc[
        qb,
        "_raw_numerator",
    ] = output.loc[
        qb,
        "passing_yards_allowed",
    ]

    output.loc[
        qb,
        "raw_rate_sample_size",
    ] = output.loc[
        qb,
        "_pass_attempts_faced",
    ]

    league = (
        output.groupby(
            [
                "season",
                "week",
                "offense_position_group",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            _league_numerator=(
                "_raw_numerator",
                "sum",
            ),
            _league_sample=(
                "raw_rate_sample_size",
                "sum",
            ),
        )
    )

    league["league_rate"] = np.where(
        league["_league_sample"].ne(0.0),
        (
            league["_league_numerator"]
            / league["_league_sample"]
        ),
        np.nan,
    )

    output = output.merge(
        league[
            [
                "season",
                "week",
                "offense_position_group",
                "league_rate",
            ]
        ],
        on=[
            "season",
            "week",
            "offense_position_group",
        ],
        how="left",
        validate="many_to_one",
    )

    denominator = (
        output["raw_rate_sample_size"]
        + prior_sample_size
    )

    output["shrunk_rate"] = np.where(
        output["league_rate"].notna()
        & denominator.ne(0.0),
        (
            output["_raw_numerator"]
            + (
                prior_sample_size
                * output["league_rate"]
            )
        )
        / denominator,
        np.nan,
    )

    return output


def validate_output(
    output: pd.DataFrame,
    *,
    config: dict,
    rich_start: int,
    prior_sample_size: float,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Position-allowed headers/order do not match contract. "
            f"Got={list(output.columns)}"
        )

    common.ensure_unique(
        output,
        GRAIN,
        "position-allowed weekly grain",
    )

    common.reject_forbidden_feature_columns(
        output.columns,
        config,
    )

    groups = sorted(
        output[
            "offense_position_group"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if groups != sorted(
        SUPPORTED_GROUPS
    ):
        raise ValueError(
            f"Unsupported/missing position groups: {groups}"
        )

    for column in COUNT_COLUMNS:
        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        negative = (
            values.notna()
            & values.lt(0.0)
        )

        if negative.any():
            raise ValueError(
                f"Negative value found in {column!r}."
            )

    for column in OUTPUT_COLUMNS:
        if column in {
            "defense_team",
            "offense_position_group",
        }:
            continue

        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        if (
            values.notna()
            & np.isinf(values)
        ).any():
            raise ValueError(
                f"Infinite values found in {column!r}."
            )

    if output.loc[
        output["season"].lt(
            rich_start
        ),
        "tackles_generated",
    ].notna().any():
        raise ValueError(
            "tackles_generated must remain null "
            f"before {rich_start}."
        )

    rich = output.loc[
        output["season"].ge(
            rich_start
        )
    ]

    if rich[
        "tackles_generated"
    ].isna().any():
        raise ValueError(
            "PBP-covered rows must have tackles_generated "
            "populated, including legitimate zeroes."
        )

    if prior_sample_size <= 0:
        raise ValueError(
            "Prior sample size must be > 0."
        )


def run() -> dict[str, Any]:
    config = common.load_config()
    repo = common.repo_root()

    groups, prior_sample_size = (
        get_position_allowed_config(
            config
        )
    )

    start_season = int(
        config["seasons"][
            "historical_start"
        ]
    )

    rich_start = int(
        config["seasons"][
            "rich_feature_start"
        ]
    )

    end_season = int(
        config["seasons"][
            "historical_end"
        ]
    )

    player_path = (
        repo
        / config["paths"][
            "player_opportunity"
        ]
    )

    team_path = (
        repo
        / config["paths"][
            "team_opportunity"
        ]
    )

    output_relative = (
        config["paths"]
        .get(
            "position_allowed",
            (
                "docs/win/football/nfl/prop_engine/"
                "data/historical/opportunity/"
                "position_allowed_week.parquet"
            ),
        )
    )

    output_path = (
        repo
        / output_relative
    )

    player = common.read_parquet_required(
        player_path
    )

    team_week = common.read_parquet_required(
        team_path
    )

    grid = build_dense_grid(
        team_week,
        groups,
    )

    totals, position_map = (
        build_allowed_totals(
            player
        )
    )

    output = grid.merge(
        totals,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    zero_fill = [
        "players_faced",
        "targets_allowed",
        "receptions_allowed",
        "receiving_yards_allowed",
        "receiving_tds_allowed",
        "carries_allowed",
        "rushing_yards_allowed",
        "rushing_tds_allowed",
        "passing_yards_allowed",
        "passing_tds_allowed",
        "_pass_attempts_faced",
    ]

    for column in zero_fill:
        output[column] = (
            pd.to_numeric(
                output[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    tackles = build_tackles_generated(
        repo=repo,
        config=config,
        position_map=position_map,
        rich_start=rich_start,
        end_season=end_season,
    )

    output = output.merge(
        tackles,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    rich_mask = (
        output["season"].ge(
            rich_start
        )
    )

    output.loc[
        rich_mask
        & output[
            "tackles_generated"
        ].isna(),
        "tackles_generated",
    ] = 0.0

    output = add_rate_contract(
        output,
        prior_sample_size=prior_sample_size,
    )

    output = output.loc[
        output["season"].between(
            start_season,
            end_season,
        )
    ].copy()

    output = output[
        OUTPUT_COLUMNS
    ].copy()

    validate_output(
        output,
        config=config,
        rich_start=rich_start,
        prior_sample_size=prior_sample_size,
    )

    common.write_parquet_atomic(
        output,
        output_path,
    )

    payload = {
        "status": "passed",
        "rows": int(
            len(output)
        ),
        "historical_start": start_season,
        "historical_end": end_season,
        "rich_feature_start": rich_start,
        "supported_groups": groups,
        "prior_sample_size": prior_sample_size,
        "rate_contract": {
            "QB": (
                "passing_yards_allowed / pass_attempts_faced"
            ),
            "RB_WR_TE": (
                "(receiving_yards_allowed + rushing_yards_allowed) / "
                "(targets_allowed + carries_allowed)"
            ),
        },
        "league_rate": (
            "same-season/week weighted league positional rate"
        ),
        "shrinkage": (
            "(raw_numerator + prior_sample_size * league_rate) / "
            "(raw_rate_sample_size + prior_sample_size)"
        ),
        "tackles_generated": (
            "2021-2025 unique defensive tackle credits on sacks, "
            "completed receptions, and rush attempts; null before PBP coverage"
        ),
        "same_week_policy": (
            "immutable realized table; lag before model use"
        ),
        "production_policy": (
            "do not use raw early-season unshrunk rates as production features"
        ),
        "output": str(
            output_path.relative_to(
                repo
            )
        ),
    }

    common.log_run(
        "build_position_allowed.py",
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
