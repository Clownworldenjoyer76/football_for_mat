#!/usr/bin/env python3
"""
Build historical Prop Engine targets at player-game grain.

READS:
    docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet
    docs/win/football/nfl/data/historic_data/player_stats/stats_player_week_{season}.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet
    docs/win/football/nfl/prop_engine/logs/build_targets.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import sys
import tempfile

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
    "position",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "field_goals_made",
    "extra_points_made",
    "kicking_points",
    "solo_tackles",
    "assisted_tackles",
    "tackles",
    "sacks",
    "target_source_present",
]


DIRECT_TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]


TARGET_VALUE_COLUMNS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "field_goals_made",
    "extra_points_made",
    "kicking_points",
    "solo_tackles",
    "assisted_tackles",
    "tackles",
    "sacks",
]


SIGNED_YARDAGE_TARGETS = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
}


NONNEGATIVE_TARGET_COLUMNS = [
    column
    for column in TARGET_VALUE_COLUMNS
    if column not in SIGNED_YARDAGE_TARGETS
]


GAME_ID_ALIASES = [
    "game_id",
    "nflverse_game_id",
]

PLAYER_ID_ALIASES = [
    "player_id",
    "gsis_id",
]

TEAM_ALIASES = [
    "team",
    "recent_team",
    "club_code",
]

POSITION_ALIASES = [
    "position",
    "position_group",
]

FIELD_GOALS_MADE_ALIASES = [
    "field_goals_made",
    "fg_made",
    "field_goal_made",
]

EXTRA_POINTS_MADE_ALIASES = [
    "extra_points_made",
    "pat_made",
    "xp_made",
]

SOLO_TACKLE_ALIASES = [
    "solo_tackles",
    "def_tackles_solo",
    "tackles_solo",
]

# IMPORTANT:
# def_tackles_with_assist is intentionally NOT an alias here.
# nflfastR defines it as a tackle where somebody else assisted
# the primary tackler. The configured target requires the
# assisting defender's credited tackle assist, represented by
# def_tackle_assists.
ASSISTED_TACKLE_ALIASES = [
    "assisted_tackles",
    "def_tackle_assists",
    "tackle_assists",
]

SACK_ALIASES = [
    "sacks",
    "def_sacks",
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


def choose_column(
    df: pd.DataFrame,
    aliases: list[str],
    *,
    label: str,
    required: bool = True,
) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"{label}: none of the required aliases exist: "
            f"{aliases}"
        )

    return None


def normalize_position(value: Any) -> str:
    return clean(value).upper()


# Issue 7 source/team validation only.
#
# Historical player-stat files use modern franchise codes for
# several relocated/renamed teams while the historical universe
# intentionally retains game-era abbreviations such as SD, OAK,
# and STL. These aliases are equivalent only for validating that
# a game_id + player_id source match belongs to the same franchise.
#
# Do not move these aliases into common.normalize_team(); Issue 6
# has already been validated with its existing normalization rules.
HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "LAC": "LAC",
    "OAK": "LV",
    "LV": "LV",
    "STL": "LAR",
    "LA": "LAR",
    "LAR": "LAR",
    "WAS": "WSH",
    "WSH": "WSH",
    "JAC": "JAX",
    "JAX": "JAX",
}


def normalize_franchise_team(value: Any) -> str:
    team = common.normalize_team(
        value
    )

    return HISTORICAL_FRANCHISE_ALIASES.get(
        team,
        team,
    )


def normalize_game_id(value: Any) -> str:
    return clean(value)


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
            f"{label}: non-numeric target values found. "
            f"Examples={examples}"
        )

    return converted.astype(float)


def configured_direct_columns(
    config: dict,
) -> dict[str, str]:
    targets = config["targets"]

    selected: dict[str, str] = {}

    for target in DIRECT_TARGETS:
        definition = targets.get(target)

        if not isinstance(definition, dict):
            raise ValueError(
                f"Config target {target!r} is missing."
            )

        source_column = clean(
            definition.get("source_column")
        )

        if not source_column:
            raise ValueError(
                f"Config target {target!r} does not "
                "define source_column."
            )

        selected[target] = source_column

    sacks_definition = targets.get("sacks")

    if not isinstance(sacks_definition, dict):
        raise ValueError(
            "Config target 'sacks' is missing."
        )

    configured_sacks = clean(
        sacks_definition.get("source_column")
    )

    if not configured_sacks:
        raise ValueError(
            "Config target 'sacks' does not define "
            "source_column."
        )

    if configured_sacks != "sacks":
        raise ValueError(
            "Unexpected configured sacks source. "
            f"Expected logical source 'sacks', got "
            f"{configured_sacks!r}."
        )

    tackle_definition = clean(
        targets.get("tackles", {}).get("definition")
    ).replace(" ", "")

    if tackle_definition != (
        "solo_tackles+assisted_tackles"
    ):
        raise ValueError(
            "Configured tackle definition must be exactly "
            "'solo_tackles + assisted_tackles'."
        )

    kicking_formula = clean(
        targets.get("kicking_points", {}).get("formula")
    ).replace(" ", "")

    if kicking_formula != (
        "3*field_goals_made+extra_points_made"
    ):
        raise ValueError(
            "Configured kicking formula must be exactly "
            "'3 * field_goals_made + extra_points_made'."
        )

    return selected


def verify_source_columns(
    source: pd.DataFrame,
    *,
    config: dict,
    path: Path,
) -> dict[str, str]:
    label = str(path)

    direct = configured_direct_columns(
        config
    )

    selected: dict[str, str] = {
        "season": choose_column(
            source,
            ["season"],
            label=label,
        ),
        "week": choose_column(
            source,
            ["week"],
            label=label,
        ),
        "season_type": choose_column(
            source,
            ["season_type"],
            label=label,
        ),
        "player_id": choose_column(
            source,
            PLAYER_ID_ALIASES,
            label=label,
        ),
        "team": choose_column(
            source,
            TEAM_ALIASES,
            label=label,
        ),
        "position": choose_column(
            source,
            POSITION_ALIASES,
            label=label,
        ),
        "player_name": choose_column(
            source,
            [
                "player_name",
                "player_display_name",
            ],
            label=label,
            required=False,
        ),
        "field_goals_made": choose_column(
            source,
            FIELD_GOALS_MADE_ALIASES,
            label=label,
        ),
        "extra_points_made": choose_column(
            source,
            EXTRA_POINTS_MADE_ALIASES,
            label=label,
        ),
        "solo_tackles": choose_column(
            source,
            SOLO_TACKLE_ALIASES,
            label=label,
        ),
        "assisted_tackles": choose_column(
            source,
            ASSISTED_TACKLE_ALIASES,
            label=label,
        ),
        "sacks": choose_column(
            source,
            SACK_ALIASES,
            label=label,
        ),
    }

    game_id = choose_column(
        source,
        GAME_ID_ALIASES,
        label=label,
        required=False,
    )

    if game_id is not None:
        selected["game_id"] = game_id

    for target, source_column in direct.items():
        if source_column not in source.columns:
            raise ValueError(
                f"{label}: configured target source "
                f"{target} -> {source_column!r} "
                "cannot be verified."
            )

        selected[target] = source_column

    # Exact tackle semantics are required.
    if (
        selected["assisted_tackles"]
        == "def_tackles_with_assist"
    ):
        raise ValueError(
            f"{label}: def_tackles_with_assist cannot "
            "be used as assisted_tackles."
        )

    return selected


def filter_unidentifiable_source_rows(
    source: pd.DataFrame,
    *,
    season: int,
    selected: dict[str, str],
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Remove source rows that lack the canonical player ID.

    These rows cannot be joined to player-game targets without
    inventing an identity. Nonzero rows are retained in the
    build log as source-quality diagnostics.

    Explicit Team rows are aggregate/non-player statistics.
    Blank-name rows are unattributed source statistics.
    Named rows that still lack canonical identity are unresolved
    source anomalies and are never guessed.
    """
    label = str(path)

    season_col = selected["season"]
    week_col = selected["week"]
    type_col = selected["season_type"]
    player_col = selected["player_id"]
    team_col = selected["team"]

    game_col = selected.get(
        "game_id"
    )

    name_col = selected.get(
        "player_name"
    )

    regular = (
        pd.to_numeric(
            source[season_col],
            errors="coerce",
        ).eq(season)
        &
        source[type_col]
        .astype(str)
        .str.upper()
        .eq("REG")
    )

    canonical_ids = (
        source[player_col]
        .map(common.normalize_player_id)
    )

    blank_id = canonical_ids.eq("")

    blank_regular = (
        regular
        & blank_id
    )

    logical_sources = [
        *DIRECT_TARGETS,
        "field_goals_made",
        "extra_points_made",
        "solo_tackles",
        "assisted_tackles",
        "sacks",
    ]

    numeric_sources = pd.DataFrame(
        index=source.index
    )

    for logical in logical_sources:
        column = selected[logical]

        numeric_sources[logical] = (
            pd.to_numeric(
                source[column],
                errors="coerce",
            ).fillna(0.0)
        )

    nonzero = (
        numeric_sources
        .ne(0.0)
        .any(axis=1)
    )

    nonzero_blank = (
        blank_regular
        & nonzero
    )

    anomalies = []

    for index in source.index[
        nonzero_blank
    ]:
        row = source.loc[index]

        player_name = (
            clean(row[name_col])
            if name_col
            else ""
        )

        if (
            player_name.casefold()
            == "team"
        ):
            classification = (
                "team_aggregate"
            )
        elif not player_name:
            classification = (
                "unnamed_unattributed"
            )
        else:
            classification = (
                "named_unresolved"
            )

        nonzero_values = {}

        for logical in logical_sources:
            value = float(
                numeric_sources.at[
                    index,
                    logical,
                ]
            )

            if value != 0.0:
                nonzero_values[
                    logical
                ] = value

        anomaly = {
            "season": season,
            "week": int(
                pd.to_numeric(
                    pd.Series(
                        [row[week_col]]
                    ),
                    errors="raise",
                ).iloc[0]
            ),
            "game_id": (
                clean(row[game_col])
                if game_col
                else ""
            ),
            "team": common.normalize_team(
                row[team_col]
            ),
            "player_name": player_name,
            "classification": (
                classification
            ),
            "nonzero_targets": (
                nonzero_values
            ),
        }

        anomalies.append(
            anomaly
        )

    named_unresolved = [
        record
        for record in anomalies
        if record["classification"]
        == "named_unresolved"
    ]

    diagnostics = {
        "blank_player_id_rows": int(
            blank_regular.sum()
        ),
        "nonzero_blank_player_id_rows": int(
            nonzero_blank.sum()
        ),
        "team_aggregate_nonzero_rows": int(
            sum(
                record["classification"]
                == "team_aggregate"
                for record in anomalies
            )
        ),
        "unnamed_unattributed_nonzero_rows": int(
            sum(
                record["classification"]
                == "unnamed_unattributed"
                for record in anomalies
            )
        ),
        "named_unresolved_nonzero_rows": int(
            len(
                named_unresolved
            )
        ),
        "nonzero_excluded_rows": (
            anomalies
        ),
    }

    # Every blank canonical-ID row is unusable at player grain.
    # Do not invent an ID, do not allocate Team aggregates, and
    # do not use a name-only guess.
    filtered = source.loc[
        ~blank_id
    ].copy()

    if filtered.empty:
        raise RuntimeError(
            f"{label}: no identifiable "
            "player rows remain."
        )

    return (
        filtered,
        diagnostics,
    )


def prepare_source(
    source: pd.DataFrame,
    *,
    season: int,
    selected: dict[str, str],
    path: Path,
) -> tuple[
    pd.DataFrame,
    str,
]:
    label = str(path)

    season_col = selected["season"]
    week_col = selected["week"]
    type_col = selected["season_type"]
    player_col = selected["player_id"]
    team_col = selected["team"]
    position_col = selected["position"]

    working = source.copy()

    working[season_col] = pd.to_numeric(
        working[season_col],
        errors="coerce",
    )

    working[week_col] = pd.to_numeric(
        working[week_col],
        errors="coerce",
    )

    working = working[
        working[season_col].eq(season)
        & working[type_col]
        .astype(str)
        .str.upper()
        .eq("REG")
    ].copy()

    if working.empty:
        raise RuntimeError(
            f"{label}: no regular-season rows "
            f"for {season}."
        )

    working["_player_id"] = (
        working[player_col]
        .map(common.normalize_player_id)
    )

    working["_team"] = (
        working[team_col]
        .map(common.normalize_team)
    )

    working["_position"] = (
        working[position_col]
        .map(normalize_position)
    )

    if working["_player_id"].eq("").any():
        count = int(
            working["_player_id"].eq("").sum()
        )

        raise ValueError(
            f"{label}: {count} regular-season rows "
            "have blank canonical player_id."
        )

    if working["_team"].eq("").any():
        count = int(
            working["_team"].eq("").sum()
        )

        raise ValueError(
            f"{label}: {count} regular-season rows "
            "have blank team."
        )

    for target in DIRECT_TARGETS:
        working[f"_src_{target}"] = (
            numeric_series(
                working[selected[target]],
                label=(
                    f"{label}:{selected[target]}"
                ),
            )
        )

    for logical in [
        "field_goals_made",
        "extra_points_made",
        "solo_tackles",
        "assisted_tackles",
        "sacks",
    ]:
        working[f"_src_{logical}"] = (
            numeric_series(
                working[selected[logical]],
                label=(
                    f"{label}:{selected[logical]}"
                ),
            )
        )

    if "game_id" in selected:
        game_col = selected["game_id"]

        working["_game_id"] = (
            working[game_col]
            .map(normalize_game_id)
        )

        if working["_game_id"].eq("").any():
            count = int(
                working["_game_id"].eq("").sum()
            )

            raise ValueError(
                f"{label}: game_id column exists "
                f"but {count} regular-season rows are blank."
            )

        common.ensure_unique(
            working,
            [
                "_game_id",
                "_player_id",
            ],
            (
                f"{label} source grain "
                "game_id + player_id"
            ),
        )

        join_mode = "game_id_player_id"

        keep = [
            "_game_id",
            "_player_id",
            "_team",
            "_position",
        ]

    else:
        common.ensure_unique(
            working,
            [
                season_col,
                week_col,
                "_team",
                "_player_id",
            ],
            (
                f"{label} source grain "
                "season + week + team + player_id"
            ),
        )

        join_mode = (
            "season_week_team_player_id"
        )

        working["_season"] = (
            working[season_col]
            .astype(int)
        )

        working["_week"] = (
            working[week_col]
            .astype(int)
        )

        keep = [
            "_season",
            "_week",
            "_team",
            "_player_id",
            "_position",
        ]

    keep.extend(
        f"_src_{column}"
        for column in [
            *DIRECT_TARGETS,
            "field_goals_made",
            "extra_points_made",
            "solo_tackles",
            "assisted_tackles",
            "sacks",
        ]
    )

    return (
        working[keep].copy(),
        join_mode,
    )


def build_season_targets(
    universe: pd.DataFrame,
    source: pd.DataFrame,
    *,
    season: int,
    join_mode: str,
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    base = universe[
        universe["season"].eq(season)
    ][
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
            "position",
            "played_game_flag",
        ]
    ].copy()

    if base.empty:
        raise RuntimeError(
            f"Universe has no rows for {season}."
        )

    base["player_id"] = (
        base["player_id"]
        .map(common.normalize_player_id)
    )

    base["team"] = (
        base["team"]
        .map(common.normalize_team)
    )

    base["game_id"] = (
        base["game_id"]
        .map(normalize_game_id)
    )

    base["_merge_order"] = range(
        len(base)
    )

    if join_mode == "game_id_player_id":
        merged = base.merge(
            source,
            how="left",
            left_on=[
                "game_id",
                "player_id",
            ],
            right_on=[
                "_game_id",
                "_player_id",
            ],
            validate="one_to_one",
            indicator=True,
        )

    elif join_mode == (
        "season_week_team_player_id"
    ):
        merged = base.merge(
            source,
            how="left",
            left_on=[
                "season",
                "week",
                "team",
                "player_id",
            ],
            right_on=[
                "_season",
                "_week",
                "_team",
                "_player_id",
            ],
            validate="one_to_one",
            indicator=True,
        )

    else:
        raise ValueError(
            f"Unknown join mode: {join_mode}"
        )

    merged = merged.sort_values(
        "_merge_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    source_present = (
        merged["_merge"]
        .eq("both")
    )

    # When joining on game_id + player_id, team is a
    # validation field rather than part of the join.
    if join_mode == "game_id_player_id":
        universe_franchise = (
            merged["team"]
            .map(
                normalize_franchise_team
            )
        )

        source_franchise = (
            merged["_team"]
            .map(
                normalize_franchise_team
            )
        )

        mismatched_team = (
            source_present
            & source_franchise.ne("")
            & source_franchise.ne(
                universe_franchise
            )
        )

        if mismatched_team.any():
            sample = merged.loc[
                mismatched_team,
                [
                    "season",
                    "week",
                    "game_id",
                    "player_id",
                    "team",
                    "_team",
                ],
            ].copy()

            sample[
                "universe_franchise"
            ] = universe_franchise.loc[
                sample.index
            ]

            sample[
                "source_franchise"
            ] = source_franchise.loc[
                sample.index
            ]

            sample = sample.head(10)

            raise ValueError(
                f"{path}: source team disagrees with "
                "universe after game_id + player_id join "
                "and historical franchise normalization. "
                f"Sample={sample.to_dict(orient='records')}"
            )

    universe_confirmed = (
        pd.to_numeric(
            merged["played_game_flag"],
            errors="raise",
        )
        .astype(int)
        .eq(1)
    )

    source_columns = [
        f"_src_{column}"
        for column in [
            *DIRECT_TARGETS,
            "field_goals_made",
            "extra_points_made",
            "solo_tackles",
            "assisted_tackles",
            "sacks",
        ]
    ]

    # A verified nonzero target event is itself definitive
    # evidence that the player participated in the target game.
    #
    # This is target-construction evidence only. It must never
    # become a pregame feature.
    #
    # Source presence alone is NOT enough: historical player-stat
    # files can contain all-zero rows for players who did not play.
    source_event_confirmed = (
        source_present
        & merged[source_columns]
        .fillna(0.0)
        .ne(0.0)
        .any(axis=1)
    )

    confirmed = (
        universe_confirmed
        | source_event_confirmed
    )

    output = merged[
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
        ]
    ].copy()

    source_position = (
        merged["_position"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    universe_position = (
        merged["position"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    output["position"] = (
        universe_position.where(
            universe_position.ne(""),
            source_position,
        )
    )

    for logical in [
        *DIRECT_TARGETS,
        "field_goals_made",
        "extra_points_made",
        "solo_tackles",
        "assisted_tackles",
        "sacks",
    ]:
        values = merged[
            f"_src_{logical}"
        ].astype(float)

        # Missing source event values become zero only
        # when actual participation is confirmed.
        values = values.where(
            values.notna(),
            0.0,
        )

        # Nonparticipants never receive zero targets.
        values = values.where(
            confirmed,
            float("nan"),
        )

        output[logical] = values

    output["kicking_points"] = (
        3.0
        * output["field_goals_made"]
        + output["extra_points_made"]
    )

    output["tackles"] = (
        output["solo_tackles"]
        + output["assisted_tackles"]
    )

    output["target_source_present"] = (
        source_present.astype("int64")
    )

    zero_filled_participants = (
        confirmed
        & ~source_present
    )

    diagnostics = {
        "universe_rows": int(
            len(base)
        ),
        "confirmed_participant_rows": int(
            confirmed.sum()
        ),
        "universe_confirmed_participant_rows": int(
            universe_confirmed.sum()
        ),
        "source_event_confirmed_participant_rows": int(
            (
                source_event_confirmed
                & ~universe_confirmed
            ).sum()
        ),
        "source_present_rows": int(
            source_present.sum()
        ),
        "zero_filled_participant_rows": int(
            zero_filled_participants.sum()
        ),
        "nonparticipant_rows": int(
            (~confirmed).sum()
        ),
        "nonparticipant_source_rows": int(
            (
                ~confirmed
                & source_present
            ).sum()
        ),
    }

    return (
        output,
        diagnostics,
    )


def validate_output(
    output: pd.DataFrame,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Target output headers/order do not match "
            f"contract. Got={list(output.columns)}"
        )

    common.ensure_unique(
        output,
        [
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        "historical target grain",
    )

    # Passing, rushing, and receiving yardage are signed NFL
    # outcomes. Preserve legitimate negative values exactly.
    #
    # Count/component targets remain nonnegative by contract.
    for column in NONNEGATIVE_TARGET_COLUMNS:
        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        negative = (
            values.notna()
            & values.lt(0)
        )

        if negative.any():
            sample = output.loc[
                negative,
                [
                    "season",
                    "week",
                    "game_id",
                    "player_id",
                    "team",
                    column,
                ],
            ].head(10)

            raise ValueError(
                f"Negative values found in non-yardage target "
                f"{column!r}. "
                f"Count={int(negative.sum())}. "
                f"Sample={sample.to_dict(orient='records')}"
            )

    source_values = set(
        output[
            "target_source_present"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if not source_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "target_source_present contains values "
            f"outside 0/1: {source_values}"
        )


def build_log_path() -> Path:
    return (
        common.prop_root()
        / "logs"
        / "build_targets.json"
    )


def write_build_log(
    payload: dict[str, Any],
) -> None:
    destination = (
        build_log_path()
        .resolve()
    )

    prop_root = (
        common.prop_root()
        .resolve()
    )

    try:
        destination.relative_to(
            prop_root
        )
    except ValueError as exc:
        raise ValueError(
            "build_targets.json must remain "
            "inside Prop Engine."
        ) from exc

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "script": "build_targets.py",
        "payload": payload,
    }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=(
            f".{destination.name}."
        ),
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )

    temp_path = Path(
        handle.name
    )

    try:
        with handle:
            json.dump(
                record,
                handle,
                sort_keys=True,
                indent=2,
                default=str,
            )

            handle.write(
                "\n"
            )

        os.replace(
            temp_path,
            destination,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise


def run() -> dict:
    config = common.load_config()
    repo = common.repo_root()

    universe_path = (
        repo
        / config["paths"][
            "historical_universe"
        ]
    )

    output_path = (
        repo
        / config["paths"][
            "historical_targets"
        ]
    )

    universe = (
        common.read_parquet_required(
            universe_path
        )
    )

    common.require_columns(
        universe,
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "team",
            "position",
            "played_game_flag",
        ],
        str(universe_path),
    )

    common.ensure_unique(
        universe,
        [
            "season",
            "week",
            "game_id",
            "player_id",
        ],
        "historical universe grain",
    )

    start_season = int(
        config["seasons"][
            "historical_start"
        ]
    )

    end_season = int(
        config["seasons"][
            "historical_end"
        ]
    )

    selected_by_season = {}
    join_modes = {}
    diagnostics_by_season = {}
    excluded_source_rows_by_season = {}
    frames = []

    for season in range(
        start_season,
        end_season + 1,
    ):
        source_path = (
            repo
            / config["paths"][
                "historical_player_stats_pattern"
            ].format(
                season=season
            )
        )

        source = (
            common.read_parquet_required(
                source_path
            )
        )

        selected = (
            verify_source_columns(
                source,
                config=config,
                path=source_path,
            )
        )

        (
            source,
            excluded_source_diagnostics,
        ) = filter_unidentifiable_source_rows(
            source,
            season=season,
            selected=selected,
            path=source_path,
        )

        prepared, join_mode = (
            prepare_source(
                source,
                season=season,
                selected=selected,
                path=source_path,
            )
        )

        season_output, diagnostics = (
            build_season_targets(
                universe,
                prepared,
                season=season,
                join_mode=join_mode,
                path=source_path,
            )
        )

        selected_by_season[
            str(season)
        ] = selected

        join_modes[
            str(season)
        ] = join_mode

        diagnostics_by_season[
            str(season)
        ] = diagnostics

        excluded_source_rows_by_season[
            str(season)
        ] = (
            excluded_source_diagnostics
        )

        frames.append(
            season_output
        )

    output = pd.concat(
        frames,
        ignore_index=True,
    )

    output = output[
        OUTPUT_COLUMNS
    ].copy()

    validate_output(
        output
    )

    if len(output) != len(universe):
        raise ValueError(
            "Target table row count does not equal "
            "historical universe row count. "
            f"targets={len(output)} "
            f"universe={len(universe)}"
        )

    common.write_parquet_atomic(
        output,
        output_path,
    )

    payload = {
        "status": "passed",
        "historical_start": start_season,
        "historical_end": end_season,
        "rows": int(len(output)),
        "unique_players": int(
            output["player_id"].nunique()
        ),
        "games": int(
            output["game_id"].nunique()
        ),
        "source_present_rows": int(
            output[
                "target_source_present"
            ].sum()
        ),
        "selected_source_columns": (
            selected_by_season
        ),
        "join_modes": join_modes,
        "season_diagnostics": (
            diagnostics_by_season
        ),
        "excluded_unidentifiable_source_rows": (
            excluded_source_rows_by_season
        ),
        "excluded_blank_player_id_rows": int(
            sum(
                item[
                    "blank_player_id_rows"
                ]
                for item in
                excluded_source_rows_by_season.values()
            )
        ),
        "excluded_nonzero_blank_player_id_rows": int(
            sum(
                item[
                    "nonzero_blank_player_id_rows"
                ]
                for item in
                excluded_source_rows_by_season.values()
            )
        ),
        "named_unresolved_source_anomalies": [
            row
            for item in
            excluded_source_rows_by_season.values()
            for row in item[
                "nonzero_excluded_rows"
            ]
            if row[
                "classification"
            ]
            == "named_unresolved"
        ],
        "unidentifiable_source_policy": (
            "exclude from player-level joining; "
            "never invent canonical identity"
        ),
        "tackle_definition": (
            "solo_tackles + assisted_tackles"
        ),
        "tackle_source_semantics": {
            "solo_tackles": (
                "def_tackles_solo"
            ),
            "assisted_tackles": (
                "def_tackle_assists"
            ),
            "excluded_alias": (
                "def_tackles_with_assist"
            ),
        },
        "kicking_formula": (
            "3 * field_goals_made "
            "+ extra_points_made"
        ),
        "nonparticipant_target_policy": (
            "target values remain null"
        ),
        "participant_missing_event_policy": (
            "fill missing event values with zero"
        ),
        "signed_yardage_targets": sorted(
            SIGNED_YARDAGE_TARGETS
        ),
        "negative_value_policy": (
            "preserve signed passing/rushing/receiving yardage; "
            "reject negative non-yardage targets"
        ),
        "output": str(
            output_path.relative_to(
                repo
            )
        ),
    }

    return payload


def main() -> int:
    try:
        payload = run()

        write_build_log(
            payload
        )

        common.log_run(
            "build_targets.py",
            payload,
        )

        return 0

    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "error_type": (
                type(exc).__name__
            ),
            "error": str(exc),
        }

        try:
            write_build_log(
                failure_payload
            )
        finally:
            common.log_run(
                "build_targets.py",
                failure_payload,
            )

        raise


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
