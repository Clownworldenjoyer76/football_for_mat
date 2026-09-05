#!/usr/bin/env python3
"""
Build historical player role and availability features.

READS:
    docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet
    docs/win/football/nfl/prop_engine/data/historical/opportunity/player_week_opportunity.parquet
    docs/win/football/nfl/data/historic_data/injuries/injuries_{season}.parquet
    docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/player_role_history.parquet

LEAKAGE CONTRACT:
    - Depth values come from the already leakage-protected historical universe.
    - Injury reports are matched to the target pregame week using the repository's
      historical injury-week convention.
    - Snap and participation rolling/change features consume only games with
      kickoff timestamps strictly earlier than the target kickoff.
    - Same-game realized snap/participation is never eligible.
    - Team-history counts consume only prior player-game history.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any
import math
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
    "position",
    "depth_rank_pregame",
    "depth_starter_flag_pregame",
    "injury_status_pregame",
    "injury_out_flag",
    "injury_doubtful_flag",
    "injury_questionable_flag",
    "prior_offense_snap_pct",
    "prior_defense_snap_pct",
    "snap_pct_roll3",
    "snap_pct_roll5",
    "snap_pct_ewm3",
    "snap_pct_ewm5",
    "prior_offense_participation",
    "prior_defense_participation",
    "participation_roll3",
    "participation_roll5",
    "depth_rank_change",
    "snap_share_change",
    "participation_change",
    "team_change_flag",
    "games_with_current_team_before_game",
    "starter_promotion_flag",
    "starter_demotion_flag",
    "teammate_out_count_position",
    "teammate_unavailable_snap_share_position",
    "role_history_games",
    "role_missing_flag",
]

GRAIN = [
    "season",
    "week",
    "game_id",
    "player_id",
]

FLAG_COLUMNS = [
    "depth_starter_flag_pregame",
    "injury_out_flag",
    "injury_doubtful_flag",
    "injury_questionable_flag",
    "team_change_flag",
    "starter_promotion_flag",
    "starter_demotion_flag",
    "role_missing_flag",
]

NONNEGATIVE_COLUMNS = [
    "prior_offense_snap_pct",
    "prior_defense_snap_pct",
    "snap_pct_roll3",
    "snap_pct_roll5",
    "snap_pct_ewm3",
    "snap_pct_ewm5",
    "prior_offense_participation",
    "prior_defense_participation",
    "participation_roll3",
    "participation_roll5",
    "games_with_current_team_before_game",
    "teammate_out_count_position",
    "teammate_unavailable_snap_share_position",
    "role_history_games",
]

ROLE_GROUP_ALIASES = {
    "HB": "RB",
    "FB": "RB",
    "H-BACK": "RB",
    "OT": "OL",
    "T": "OL",
    "LT": "OL",
    "RT": "OL",
    "G": "OL",
    "OG": "OL",
    "LG": "OL",
    "RG": "OL",
    "C": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "EDGE": "DL",
    "LDE": "DL",
    "RDE": "DL",
    "LDT": "DL",
    "RDT": "DL",
    "ILB": "LB",
    "OLB": "LB",
    "MLB": "LB",
    "WLB": "LB",
    "SLB": "LB",
    "CB": "DB",
    "LCB": "DB",
    "RCB": "DB",
    "NB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "K": "SPEC",
    "PK": "SPEC",
    "P": "SPEC",
    "LS": "SPEC",
    "KR": "SPEC",
    "PR": "SPEC",
}

HISTORICAL_FRANCHISE_ALIASES = {
    # Do not relocate historical franchises globally. These aliases only
    # harmonize repository spelling variants that refer to the same club.
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

TEAM_HISTORY_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
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


def normalize_team(value: Any) -> str:
    team = clean(value).upper()

    if not team:
        return ""

    return HISTORICAL_FRANCHISE_ALIASES.get(
        team,
        team,
    )


def normalize_team_history_identity(
    value: Any,
) -> str:
    team = normalize_team(
        value
    )

    return TEAM_HISTORY_ALIASES.get(
        team,
        team,
    )


def normalize_position(value: Any) -> str:
    raw = (
        clean(value)
        .upper()
        .replace(" ", "")
    )

    aliases = {
        "QUARTERBACK": "QB",
        "RUNNINGBACK": "RB",
        "WIDERECEIVER": "WR",
        "TIGHTEND": "TE",
        "OFFENSIVELINE": "OL",
        "OFFENSIVETACKLE": "OT",
        "OFFENSIVEGUARD": "G",
        "DEFENSIVELINE": "DL",
        "DEFENSIVEEND": "DE",
        "DEFENSIVETACKLE": "DT",
        "LINEBACKER": "LB",
        "DEFENSIVEBACK": "DB",
        "CORNERBACK": "CB",
        "SAFETY": "S",
        "PLACEKICKER": "K",
    }

    return aliases.get(
        raw,
        raw,
    )


def role_group(
    position: Any,
    position_group: Any = "",
) -> str:
    position = normalize_position(
        position
    )

    if position in {
        "QB",
        "RB",
        "WR",
        "TE",
        "OL",
        "DL",
        "LB",
        "DB",
        "SPEC",
    }:
        return position

    if position in ROLE_GROUP_ALIASES:
        return ROLE_GROUP_ALIASES[
            position
        ]

    group = clean(
        position_group
    ).upper()

    if group in {
        "QB",
        "RB",
        "WR",
        "TE",
        "OL",
        "DL",
        "LB",
        "DB",
        "SPEC",
    }:
        return group

    return (
        group
        or position
        or "UNKNOWN"
    )


def normalize_injury_status(
    value: Any,
) -> str:
    status = (
        clean(value)
        .casefold()
        .replace("-", " ")
        .replace("_", " ")
    )

    status = " ".join(
        status.split()
    )

    if (
        status in {"o", "out"}
        or status.startswith("out ")
        or status == "ir"
    ):
        return "out"

    if (
        status in {"d", "doubtful"}
        or "doubt" in status
    ):
        return "doubtful"

    if (
        status in {"q", "questionable"}
        or "question" in status
    ):
        return "questionable"

    if (
        status in {"p", "probable"}
        or "probable" in status
    ):
        return "probable"

    return ""


def numeric(
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
        sample = (
            series.loc[invalid]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{label}: non-numeric values. "
            f"Sample={sample}"
        )

    return converted.astype(
        float
    )


def max_available(
    left: Any,
    right: Any,
) -> float:
    values = []

    for value in (
        left,
        right,
    ):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            values.append(
                number
            )

    if not values:
        return float("nan")

    return max(
        values
    )


def mean_last(
    values: list[float],
    window: int,
) -> float:
    if not values:
        return float("nan")

    sample = values[
        -window:
    ]

    if not sample:
        return float("nan")

    return float(
        np.mean(sample)
    )


def update_ewm(
    previous: float | None,
    value: float,
    span: int,
) -> float:
    alpha = 2.0 / (
        float(span) + 1.0
    )

    if previous is None:
        return float(value)

    return (
        alpha * float(value)
        + (1.0 - alpha) * previous
    )


def build_name_map(
    crosswalk: pd.DataFrame,
) -> dict[str, str]:
    common.require_columns(
        crosswalk,
        [
            "gsis_id",
            "display_name",
            "normalized_name",
        ],
        "player crosswalk",
    )

    candidates: dict[
        str,
        set[str],
    ] = {}

    for row in crosswalk.itertuples(
        index=False
    ):
        gsis = (
            common.normalize_player_id(
                getattr(
                    row,
                    "gsis_id",
                )
            )
        )

        if not gsis:
            continue

        normalized = clean(
            getattr(
                row,
                "normalized_name",
            )
        )

        if not normalized:
            normalized = (
                common.normalize_name(
                    getattr(
                        row,
                        "display_name",
                    )
                )
            )

        if not normalized:
            continue

        candidates.setdefault(
            normalized,
            set(),
        ).add(
            gsis
        )

    return {
        name: next(
            iter(ids)
        )
        for name, ids in candidates.items()
        if len(ids) == 1
    }


def parse_modified(
    value: Any,
) -> pd.Timestamp | None:
    text = clean(
        value
    )

    if not text:
        return None

    timestamp = pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        timestamp
    ):
        return None

    return timestamp


def load_injury_maps(
    *,
    repo: Path,
    config: dict,
    start_season: int,
    end_season: int,
    unique_name_map: dict[str, str],
) -> tuple[
    dict[tuple[int, int, str, str], str],
    dict[int, list[int]],
    dict[str, int],
]:
    status_map: dict[
        tuple[int, int, str, str],
        str,
    ] = {}

    rank_map: dict[
        tuple[int, int, str, str],
        tuple[
            int,
            int,
            int,
            str,
        ],
    ] = {}

    source_weeks: dict[
        int,
        set[int],
    ] = {}

    diagnostics = {
        "rows": 0,
        "resolved_ids": 0,
        "unresolved_ids": 0,
        "known_status_rows": 0,
    }

    for season in range(
        start_season,
        end_season + 1,
    ):
        path = (
            repo
            / config["paths"][
                "historical_injuries_pattern"
            ].format(
                season=season
            )
        )

        df = (
            common.read_parquet_required(
                path
            )
        )

        common.require_columns(
            df,
            [
                "team",
                "week",
                "gsis_id",
                "full_name",
                "position",
                "report_status",
            ],
            str(path),
        )

        source_weeks.setdefault(
            season,
            set(),
        )

        modified_column = (
            "date_modified"
            if "date_modified" in df.columns
            else None
        )

        for source_index, row in df.iterrows():
            diagnostics[
                "rows"
            ] += 1

            week_value = pd.to_numeric(
                pd.Series(
                    [row["week"]]
                ),
                errors="coerce",
            ).iloc[0]

            if (
                pd.isna(
                    week_value
                )
                or float(
                    week_value
                ).is_integer()
                is False
            ):
                continue

            week = int(
                week_value
            )

            source_weeks[
                season
            ].add(
                week
            )

            status = (
                normalize_injury_status(
                    row[
                        "report_status"
                    ]
                )
            )

            if not status:
                continue

            diagnostics[
                "known_status_rows"
            ] += 1

            team = normalize_team(
                row["team"]
            )

            if not team:
                continue

            player_id = (
                common.normalize_player_id(
                    row["gsis_id"]
                )
            )

            if not player_id:
                name_key = (
                    common.normalize_name(
                        row["full_name"]
                    )
                )

                player_id = (
                    unique_name_map.get(
                        name_key,
                        "",
                    )
                )

            if not player_id:
                diagnostics[
                    "unresolved_ids"
                ] += 1
                continue

            diagnostics[
                "resolved_ids"
            ] += 1

            modified = (
                parse_modified(
                    row[
                        modified_column
                    ]
                )
                if modified_column
                else None
            )

            timestamp_rank = (
                int(
                    modified.value
                )
                if modified is not None
                else -1
            )

            dated_rank = (
                1
                if modified is not None
                else 0
            )

            key = (
                season,
                week,
                team,
                player_id,
            )

            candidate_rank = (
                dated_rank,
                timestamp_rank,
                int(
                    source_index
                ),
                status,
            )

            existing = rank_map.get(
                key
            )

            if (
                existing is None
                or candidate_rank[:3]
                > existing[:3]
            ):
                rank_map[
                    key
                ] = candidate_rank

                status_map[
                    key
                ] = status

    normalized_weeks = {
        season: sorted(
            weeks
        )
        for season, weeks in source_weeks.items()
    }

    return (
        status_map,
        normalized_weeks,
        diagnostics,
    )


def resolve_injury_source_week(
    *,
    season: int,
    target_week: int,
    source_weeks: dict[int, list[int]],
) -> int | None:
    weeks = source_weeks.get(
        season,
        [],
    )

    if not weeks:
        return None

    if target_week in weeks:
        return target_week

    index = (
        bisect_left(
            weeks,
            target_week,
        )
        - 1
    )

    if index < 0:
        return None

    return weeks[
        index
    ]


def prepare_universe(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "season",
        "week",
        "game_id",
        "kickoff_timestamp",
        "player_id",
        "team",
        "position",
        "position_group",
        "depth_present_flag",
        "depth_rank",
        "depth_starter_flag",
        "prior_offense_snap_pct",
        "prior_defense_snap_pct",
        "prior_offense_participation",
        "prior_defense_participation",
    ]

    common.require_columns(
        universe,
        required,
        "historical player-game universe",
    )

    base = universe[
        required
    ].copy()

    base["season"] = pd.to_numeric(
        base["season"],
        errors="raise",
    ).astype(int)

    base["week"] = pd.to_numeric(
        base["week"],
        errors="raise",
    ).astype(int)

    base["player_id"] = (
        base["player_id"]
        .map(
            common.normalize_player_id
        )
    )

    base["team"] = (
        base["team"]
        .map(
            normalize_team
        )
    )

    base["position"] = (
        base["position"]
        .map(
            normalize_position
        )
    )

    base["_kickoff"] = pd.to_datetime(
        base[
            "kickoff_timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    if base[
        "_kickoff"
    ].isna().any():
        sample = (
            base.loc[
                base[
                    "_kickoff"
                ].isna(),
                GRAIN
                + [
                    "kickoff_timestamp"
                ],
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            "Universe contains invalid kickoff timestamps. "
            f"Sample={sample}"
        )

    if base[
        "player_id"
    ].eq("").any():
        raise ValueError(
            "Universe contains blank canonical player_id."
        )

    common.ensure_unique(
        base,
        GRAIN,
        "role-history universe base",
    )

    for column in [
        "prior_offense_snap_pct",
        "prior_defense_snap_pct",
        "prior_offense_participation",
        "prior_defense_participation",
    ]:
        base[column] = numeric(
            base[column],
            label=f"universe:{column}",
        )

    base[
        "depth_rank_pregame"
    ] = numeric(
        base["depth_rank"],
        label="universe:depth_rank",
    )

    starter = pd.to_numeric(
        base[
            "depth_starter_flag"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    if (
        ~starter.isin(
            [0, 1]
        )
    ).any():
        raise ValueError(
            "Universe depth_starter_flag must be binary."
        )

    base[
        "depth_starter_flag_pregame"
    ] = starter.astype(
        "int64"
    )

    base["_role_group"] = [
        role_group(
            position,
            position_group,
        )
        for position, position_group
        in zip(
            base["position"],
            base[
                "position_group"
            ],
        )
    ]

    return base


def prepare_realized_history(
    player_opportunity: pd.DataFrame,
    game_kickoffs: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "season",
        "week",
        "game_id",
        "player_id",
        "team",
        "offense_snap_pct",
        "defense_snap_pct",
        "offense_participation",
        "defense_participation",
    ]

    common.require_columns(
        player_opportunity,
        required,
        "player weekly opportunity",
    )

    history = (
        player_opportunity[
            required
        ].copy()
    )

    history["season"] = pd.to_numeric(
        history["season"],
        errors="raise",
    ).astype(int)

    history["week"] = pd.to_numeric(
        history["week"],
        errors="raise",
    ).astype(int)

    history["player_id"] = (
        history["player_id"]
        .map(
            common.normalize_player_id
        )
    )

    history["team"] = (
        history["team"]
        .map(
            normalize_team
        )
    )

    for column in [
        "offense_snap_pct",
        "defense_snap_pct",
        "offense_participation",
        "defense_participation",
    ]:
        history[column] = numeric(
            history[column],
            label=(
                "player opportunity:"
                + column
            ),
        )

    history = history.merge(
        game_kickoffs,
        on=[
            "season",
            "week",
            "game_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if history[
        "_kickoff"
    ].isna().any():
        sample = (
            history.loc[
                history[
                    "_kickoff"
                ].isna(),
                GRAIN,
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            "Player opportunity rows missing universe kickoff mapping. "
            f"Sample={sample}"
        )

    history["_snap_share"] = [
        max_available(
            off,
            defense,
        )
        for off, defense in zip(
            history[
                "offense_snap_pct"
            ],
            history[
                "defense_snap_pct"
            ],
        )
    ]

    history[
        "_participation_share"
    ] = [
        max_available(
            off,
            defense,
        )
        for off, defense in zip(
            history[
                "offense_participation"
            ],
            history[
                "defense_participation"
            ],
        )
    ]

    history["_role_observed"] = (
        history[
            [
                "_snap_share",
                "_participation_share",
            ]
        ]
        .notna()
        .any(
            axis=1
        )
    )

    common.ensure_unique(
        history,
        GRAIN,
        "realized role-history source",
    )

    return history


def add_strict_prior_history(
    base: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    result = base.copy()

    feature_columns = [
        "snap_pct_roll3",
        "snap_pct_roll5",
        "snap_pct_ewm3",
        "snap_pct_ewm5",
        "participation_roll3",
        "participation_roll5",
        "snap_share_change",
        "participation_change",
        "team_change_flag",
        "games_with_current_team_before_game",
        "role_history_games",
    ]

    for column in feature_columns:
        result[column] = float(
            "nan"
        )

    result[
        "team_change_flag"
    ] = 0

    result[
        "games_with_current_team_before_game"
    ] = 0

    result[
        "role_history_games"
    ] = 0

    history_groups = {
        player_id: group.sort_values(
            [
                "_kickoff",
                "game_id",
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        )
        for player_id, group
        in history.groupby(
            "player_id",
            sort=False,
        )
    }

    for player_id, target_group in (
        result.groupby(
            "player_id",
            sort=False,
        )
    ):
        target_indexes = (
            target_group.sort_values(
                [
                    "_kickoff",
                    "game_id",
                ],
                kind="mergesort",
            ).index.tolist()
        )

        source = history_groups.get(
            player_id
        )

        if source is None or source.empty:
            continue

        source_rows = source.to_dict(
            orient="records"
        )

        pointer = 0
        snap_values: list[
            float
        ] = []
        participation_values: list[
            float
        ] = []

        snap_ewm3: float | None = None
        snap_ewm5: float | None = None

        previous_team = ""
        team_counts: Counter[
            str
        ] = Counter()

        role_history_games = 0

        for target_index in target_indexes:
            target_kickoff = result.at[
                target_index,
                "_kickoff",
            ]

            while (
                pointer
                < len(
                    source_rows
                )
                and source_rows[
                    pointer
                ][
                    "_kickoff"
                ]
                < target_kickoff
            ):
                source_row = (
                    source_rows[
                        pointer
                    ]
                )

                source_team = (
                    normalize_team_history_identity(
                        source_row[
                            "team"
                        ]
                    )
                )

                if source_team:
                    team_counts[
                        source_team
                    ] += 1

                    previous_team = (
                        source_team
                    )

                if bool(
                    source_row[
                        "_role_observed"
                    ]
                ):
                    role_history_games += 1

                snap_value = source_row[
                    "_snap_share"
                ]

                if (
                    snap_value is not None
                    and not pd.isna(
                        snap_value
                    )
                ):
                    snap_value = float(
                        snap_value
                    )

                    snap_values.append(
                        snap_value
                    )

                    snap_ewm3 = (
                        update_ewm(
                            snap_ewm3,
                            snap_value,
                            3,
                        )
                    )

                    snap_ewm5 = (
                        update_ewm(
                            snap_ewm5,
                            snap_value,
                            5,
                        )
                    )

                participation_value = (
                    source_row[
                        "_participation_share"
                    ]
                )

                if (
                    participation_value
                    is not None
                    and not pd.isna(
                        participation_value
                    )
                ):
                    participation_values.append(
                        float(
                            participation_value
                        )
                    )

                pointer += 1

            result.at[
                target_index,
                "snap_pct_roll3",
            ] = mean_last(
                snap_values,
                3,
            )

            result.at[
                target_index,
                "snap_pct_roll5",
            ] = mean_last(
                snap_values,
                5,
            )

            result.at[
                target_index,
                "snap_pct_ewm3",
            ] = (
                float(
                    snap_ewm3
                )
                if snap_ewm3
                is not None
                else float("nan")
            )

            result.at[
                target_index,
                "snap_pct_ewm5",
            ] = (
                float(
                    snap_ewm5
                )
                if snap_ewm5
                is not None
                else float("nan")
            )

            result.at[
                target_index,
                "participation_roll3",
            ] = mean_last(
                participation_values,
                3,
            )

            result.at[
                target_index,
                "participation_roll5",
            ] = mean_last(
                participation_values,
                5,
            )

            if len(
                snap_values
            ) >= 2:
                result.at[
                    target_index,
                    "snap_share_change",
                ] = (
                    snap_values[-1]
                    - snap_values[-2]
                )

            if len(
                participation_values
            ) >= 2:
                result.at[
                    target_index,
                    "participation_change",
                ] = (
                    participation_values[-1]
                    - participation_values[-2]
                )

            current_team = (
                normalize_team_history_identity(
                    result.at[
                        target_index,
                        "team",
                    ]
                )
            )

            result.at[
                target_index,
                "team_change_flag",
            ] = int(
                bool(
                    previous_team
                )
                and current_team
                != previous_team
            )

            result.at[
                target_index,
                "games_with_current_team_before_game",
            ] = int(
                team_counts.get(
                    current_team,
                    0,
                )
            )

            result.at[
                target_index,
                "role_history_games",
            ] = int(
                role_history_games
            )

    result[
        "team_change_flag"
    ] = (
        result[
            "team_change_flag"
        ]
        .astype(
            "int64"
        )
    )

    result[
        "games_with_current_team_before_game"
    ] = (
        result[
            "games_with_current_team_before_game"
        ]
        .astype(
            "int64"
        )
    )

    result[
        "role_history_games"
    ] = (
        result[
            "role_history_games"
        ]
        .astype(
            "int64"
        )
    )

    return result


def add_depth_changes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result[
        "depth_rank_change"
    ] = float(
        "nan"
    )

    result[
        "starter_promotion_flag"
    ] = 0

    result[
        "starter_demotion_flag"
    ] = 0

    for _, group in result.groupby(
        "player_id",
        sort=False,
    ):
        ordered = group.sort_values(
            [
                "_kickoff",
                "game_id",
            ],
            kind="mergesort",
        )

        previous_rank: float | None = None
        previous_starter: int | None = None
        previous_kickoff: pd.Timestamp | None = None

        for index in ordered.index:
            kickoff = result.at[
                index,
                "_kickoff",
            ]

            current_rank_value = (
                result.at[
                    index,
                    "depth_rank_pregame",
                ]
            )

            current_rank = (
                None
                if pd.isna(
                    current_rank_value
                )
                else float(
                    current_rank_value
                )
            )

            current_starter = int(
                result.at[
                    index,
                    "depth_starter_flag_pregame",
                ]
            )

            if (
                previous_kickoff
                is not None
                and previous_kickoff
                < kickoff
                and previous_rank
                is not None
                and current_rank
                is not None
            ):
                result.at[
                    index,
                    "depth_rank_change",
                ] = (
                    current_rank
                    - previous_rank
                )

                if previous_starter is not None:
                    result.at[
                        index,
                        "starter_promotion_flag",
                    ] = int(
                        previous_starter == 0
                        and current_starter == 1
                    )

                    result.at[
                        index,
                        "starter_demotion_flag",
                    ] = int(
                        previous_starter == 1
                        and current_starter == 0
                    )

            if (
                previous_kickoff is None
                or previous_kickoff
                < kickoff
            ):
                previous_rank = (
                    current_rank
                )

                previous_starter = (
                    current_starter
                )

                previous_kickoff = (
                    kickoff
                )

    result[
        "starter_promotion_flag"
    ] = (
        result[
            "starter_promotion_flag"
        ].astype(
            "int64"
        )
    )

    result[
        "starter_demotion_flag"
    ] = (
        result[
            "starter_demotion_flag"
        ].astype(
            "int64"
        )
    )

    return result


def add_injury_features(
    frame: pd.DataFrame,
    *,
    status_map: dict[
        tuple[int, int, str, str],
        str,
    ],
    source_weeks: dict[
        int,
        list[int],
    ],
) -> pd.DataFrame:
    result = frame.copy()

    source_week_cache: dict[
        tuple[int, int],
        int | None,
    ] = {}

    statuses = []

    for season, week, team, player_id in zip(
        result["season"],
        result["week"],
        result["team"],
        result["player_id"],
    ):
        cache_key = (
            int(season),
            int(week),
        )

        if cache_key not in source_week_cache:
            source_week_cache[
                cache_key
            ] = (
                resolve_injury_source_week(
                    season=int(
                        season
                    ),
                    target_week=int(
                        week
                    ),
                    source_weeks=source_weeks,
                )
            )

        source_week = (
            source_week_cache[
                cache_key
            ]
        )

        if source_week is None:
            statuses.append(
                ""
            )
            continue

        statuses.append(
            status_map.get(
                (
                    int(
                        season
                    ),
                    int(
                        source_week
                    ),
                    normalize_team(
                        team
                    ),
                    common.normalize_player_id(
                        player_id
                    ),
                ),
                "",
            )
        )

    result[
        "injury_status_pregame"
    ] = statuses

    result[
        "injury_out_flag"
    ] = (
        result[
            "injury_status_pregame"
        ]
        .eq(
            "out"
        )
        .astype(
            "int64"
        )
    )

    result[
        "injury_doubtful_flag"
    ] = (
        result[
            "injury_status_pregame"
        ]
        .eq(
            "doubtful"
        )
        .astype(
            "int64"
        )
    )

    result[
        "injury_questionable_flag"
    ] = (
        result[
            "injury_status_pregame"
        ]
        .eq(
            "questionable"
        )
        .astype(
            "int64"
        )
    )

    return result


def add_teammate_availability(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["_prior_snap_share"] = [
        max_available(
            off,
            defense,
        )
        for off, defense in zip(
            result[
                "prior_offense_snap_pct"
            ],
            result[
                "prior_defense_snap_pct"
            ],
        )
    ]

    result[
        "_out_snap_share"
    ] = np.where(
        result[
            "injury_out_flag"
        ].eq(1),
        result[
            "_prior_snap_share"
        ].fillna(0.0),
        0.0,
    )

    group_keys = [
        "season",
        "week",
        "game_id",
        "team",
        "_role_group",
    ]

    group_values = (
        result.groupby(
            group_keys,
            as_index=False,
            dropna=False,
        )
        .agg(
            _position_out_count=(
                "injury_out_flag",
                "sum",
            ),
            _position_out_snap_share=(
                "_out_snap_share",
                "sum",
            ),
        )
    )

    result = result.merge(
        group_values,
        on=group_keys,
        how="left",
        validate="many_to_one",
    )

    result[
        "teammate_out_count_position"
    ] = (
        result[
            "_position_out_count"
        ]
        - result[
            "injury_out_flag"
        ]
    ).clip(
        lower=0
    ).astype(
        "int64"
    )

    self_out_share = np.where(
        result[
            "injury_out_flag"
        ].eq(1),
        result[
            "_prior_snap_share"
        ].fillna(0.0),
        0.0,
    )

    result[
        "teammate_unavailable_snap_share_position"
    ] = (
        result[
            "_position_out_snap_share"
        ]
        - self_out_share
    ).clip(
        lower=0.0
    )

    return result


def add_role_missing_flag(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    missing = (
        result[
            "depth_rank_pregame"
        ].isna()
        & result[
            "prior_offense_snap_pct"
        ].isna()
        & result[
            "prior_defense_snap_pct"
        ].isna()
        & result[
            "prior_offense_participation"
        ].isna()
        & result[
            "prior_defense_participation"
        ].isna()
    )

    result[
        "role_missing_flag"
    ] = missing.astype(
        "int64"
    )

    return result


def validate_output(
    output: pd.DataFrame,
    *,
    universe_rows: int,
    config: dict,
) -> None:
    if list(
        output.columns
    ) != OUTPUT_COLUMNS:
        raise ValueError(
            "Role-history headers/order do not match contract. "
            f"Got={list(output.columns)}"
        )

    if len(
        output
    ) != universe_rows:
        raise ValueError(
            "Role-history output row count must equal universe. "
            f"output={len(output)} universe={universe_rows}"
        )

    common.ensure_unique(
        output,
        GRAIN,
        "player role-history grain",
    )

    common.reject_forbidden_feature_columns(
        output.columns,
        config,
    )

    for column in FLAG_COLUMNS:
        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        if (
            values.isna()
            | ~values.isin(
                [0, 1]
            )
        ).any():
            raise ValueError(
                f"{column} must be non-null binary."
            )

    for column in NONNEGATIVE_COLUMNS:
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
                f"Negative role value found in {column!r}."
            )

    for column in [
        "prior_offense_snap_pct",
        "prior_defense_snap_pct",
        "snap_pct_roll3",
        "snap_pct_roll5",
        "snap_pct_ewm3",
        "snap_pct_ewm5",
        "prior_offense_participation",
        "prior_defense_participation",
        "participation_roll3",
        "participation_roll5",
    ]:
        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        invalid = (
            values.notna()
            & ~values.between(
                0.0,
                1.0,
            )
        )

        if invalid.any():
            sample = (
                output.loc[
                    invalid,
                    GRAIN
                    + [
                        column
                    ],
                ]
                .head(10)
                .to_dict(
                    orient="records"
                )
            )

            raise ValueError(
                f"{column} outside [0,1]. "
                f"Sample={sample}"
            )

    for column in OUTPUT_COLUMNS:
        if column in {
            "game_id",
            "player_id",
            "team",
            "position",
            "injury_status_pregame",
        }:
            continue

        values = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        if (
            values.notna()
            & np.isinf(
                values
            )
        ).any():
            raise ValueError(
                f"Infinite values found in {column!r}."
            )


def run() -> dict[str, Any]:
    config = common.load_config()
    repo = common.repo_root()

    start_season = int(
        config[
            "seasons"
        ][
            "historical_start"
        ]
    )

    end_season = int(
        config[
            "seasons"
        ][
            "historical_end"
        ]
    )

    universe_path = (
        repo
        / config[
            "paths"
        ][
            "historical_universe"
        ]
    )

    player_opportunity_path = (
        repo
        / config[
            "paths"
        ][
            "player_opportunity"
        ]
    )

    crosswalk_path = (
        repo
        / config[
            "paths"
        ][
            "identity_crosswalk"
        ]
    )

    output_relative = (
        config[
            "paths"
        ].get(
            "role_history",
            (
                "docs/win/football/nfl/prop_engine/"
                "data/historical/features/"
                "player_role_history.parquet"
            ),
        )
    )

    output_path = (
        repo
        / output_relative
    )

    universe_raw = (
        common.read_parquet_required(
            universe_path
        )
    )

    player_opportunity = (
        common.read_parquet_required(
            player_opportunity_path
        )
    )

    crosswalk = (
        common.read_parquet_required(
            crosswalk_path
        )
    )

    base = prepare_universe(
        universe_raw
    )

    base = base.loc[
        base[
            "season"
        ].between(
            start_season,
            end_season,
        )
    ].copy()

    game_kickoffs = (
        base[
            [
                "season",
                "week",
                "game_id",
                "_kickoff",
            ]
        ]
        .drop_duplicates()
    )

    common.ensure_unique(
        game_kickoffs,
        [
            "season",
            "week",
            "game_id",
        ],
        "game kickoff mapping",
    )

    history = prepare_realized_history(
        player_opportunity,
        game_kickoffs,
    )

    unique_name_map = build_name_map(
        crosswalk
    )

    (
        injury_status_map,
        injury_source_weeks,
        injury_diagnostics,
    ) = load_injury_maps(
        repo=repo,
        config=config,
        start_season=start_season,
        end_season=end_season,
        unique_name_map=unique_name_map,
    )

    base = add_strict_prior_history(
        base,
        history,
    )

    base = add_depth_changes(
        base
    )

    base = add_injury_features(
        base,
        status_map=injury_status_map,
        source_weeks=injury_source_weeks,
    )

    base = add_teammate_availability(
        base
    )

    base = add_role_missing_flag(
        base
    )

    output = base.rename(
        columns={}
    )

    output = output[
        OUTPUT_COLUMNS
    ].copy()

    validate_output(
        output,
        universe_rows=int(
            len(
                base
            )
        ),
        config=config,
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
        "players": int(
            output[
                "player_id"
            ].nunique()
        ),
        "games": int(
            output[
                "game_id"
            ].nunique()
        ),
        "historical_start": start_season,
        "historical_end": end_season,
        "depth_policy": (
            "pregame depth rank/starter copied from leakage-protected "
            "historical universe"
        ),
        "injury_policy": (
            "target week report; if an entire source week is absent, "
            "latest strictly earlier injury source week; latest report row "
            "per player by date_modified when available"
        ),
        "unavailable_policy": (
            "teammate unavailable means confirmed report_status=out only"
        ),
        "snap_policy": (
            "rolling/EWM/change consume only realized games with kickoff "
            "strictly before target kickoff; primary snap share=max(offense, defense)"
        ),
        "participation_policy": (
            "rolling/change consume only realized games with kickoff "
            "strictly before target kickoff; primary participation=max(offense, defense)"
        ),
        "team_change_policy": (
            "current team compared with latest strictly prior realized player-game team"
        ),
        "role_history_games_policy": (
            "count strictly prior player-opportunity games having observed snap "
            "or participation data"
        ),
        "teammate_position_policy": (
            "same normalized positional family; out teammate snap share uses "
            "that teammate's pregame prior offense/defense snap maximum"
        ),
        "role_missing_policy": (
            "1 only when current pregame depth rank and all prior snap/"
            "participation values are missing"
        ),
        "same_game_realized_forbidden": True,
        "injury_diagnostics": injury_diagnostics,
        "output": str(
            output_path.relative_to(
                repo
            )
        ),
    }

    common.log_run(
        "build_role_history.py",
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
