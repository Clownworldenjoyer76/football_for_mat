#!/usr/bin/env python3
"""
Step 7: fill Week 1 team and QB lagged statistics using the final
available rows from the previous season.

For each Week 1 game:

TEAM VALUES:
  Use the final available prior-season row for the home/away team from:
    docs/win/football/nfl/00_intake/team_stats/{season-1}_team_stats.csv

QB VALUES:
  Identify the historical starting QBs using home_qb_id / away_qb_id from:
    docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv

  Then use each QB's final available prior-season row from:
    docs/win/football/nfl/00_intake/qb/{season-1}_qb_stats.csv

READS/WRITES IN PLACE:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

2021 EXCEPTION:
  The historical team/QB intake directories begin with 2021.
  There are no 2020_team_stats.csv or 2020_qb_stats.csv source files.
  Therefore 2021 Week 1 remains unchanged rather than causing the
  workflow to fail.

Only Week 1 Step 5 / Step 6 feature columns are populated.
Rows from Week 2 onward are not modified.

No raw source files are edited.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_DIR = NFL_ROOT / "training"
TEAM_STATS_DIR = NFL_ROOT / "00_intake/team_stats"
QB_STATS_DIR = NFL_ROOT / "00_intake/qb"
GAMES_PATH = NFL_ROOT / "data/historic_data/games/games_2010_2025.csv"

TRAINING_SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

TRAINING_PATHS = {
    season: TRAINING_DIR / f"historical_core_{season}.csv"
    for season in TRAINING_SEASONS
}

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

TEAM_SOURCE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "team",
    *TEAM_METRICS,
]

QB_SOURCE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "team",
    "player_id",
    "qb_name",
    "dropbacks",
    *QB_METRICS,
]

GAMES_REQUIRED_COLUMNS = [
    "game_id",
    "home_qb_id",
    "away_qb_id",
]

TRAINING_REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
]

HOME_TEAM_COLUMNS = [
    f"home_{metric}"
    for metric in TEAM_METRICS
]

AWAY_TEAM_COLUMNS = [
    f"away_{metric}"
    for metric in TEAM_METRICS
]

TEAM_DIFF_COLUMNS = [
    f"{metric}_diff"
    for metric in TEAM_METRICS
]

STEP5_COLUMNS = [
    *HOME_TEAM_COLUMNS,
    *AWAY_TEAM_COLUMNS,
    *TEAM_DIFF_COLUMNS,
]

HOME_QB_COLUMNS = [
    f"home_qb_{metric}"
    for metric in QB_METRICS
]

AWAY_QB_COLUMNS = [
    f"away_qb_{metric}"
    for metric in QB_METRICS
]

QB_DIFF_COLUMNS = [
    f"qb_{metric}_diff"
    for metric in QB_METRICS
]

STEP6_COLUMNS = [
    *HOME_QB_COLUMNS,
    *AWAY_QB_COLUMNS,
    *QB_DIFF_COLUMNS,
]

STEP7_COLUMNS = [
    *STEP5_COLUMNS,
    *STEP6_COLUMNS,
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}"
        )


def normalize_integer_key(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    bad = (
        numeric.isna()
        & series.notna()
        & series.astype(str).str.strip().ne("")
    )

    if bad.any():
        values = (
            series.loc[bad]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: invalid numeric values: "
            + ", ".join(values)
        )

    non_integer = (
        numeric.notna()
        & ((numeric % 1).abs() > 1e-9)
    )

    if non_integer.any():
        values = (
            series.loc[non_integer]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: non-integer values: "
            + ", ".join(values)
        )

    return numeric.astype("Int64")


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_player_id(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "<na>",
        "nan",
        "none",
    }:
        return ""

    return text


def normalize_game_id(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def numeric_metric(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    bad = (
        converted.isna()
        & series.notna()
        & series.astype(str).str.strip().ne("")
    )

    if bad.any():
        values = (
            series.loc[bad]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: non-numeric values: "
            + ", ".join(values)
        )

    return converted


def clean_numeric_value(
    value: object,
) -> float | None:
    if pd.isna(value):
        return None

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def load_games_qb_index() -> dict[
    str,
    tuple[str, str],
]:
    games = read_csv(
        GAMES_PATH
    )

    require_columns(
        games,
        GAMES_REQUIRED_COLUMNS,
        "historical games",
    )

    games = games[
        GAMES_REQUIRED_COLUMNS
    ].copy()

    games["_game_key"] = (
        games["game_id"]
        .map(normalize_game_id)
    )

    if games["_game_key"].eq("").any():
        raise ValueError(
            "Historical games file contains blank "
            "game_id values."
        )

    duplicate_game_ids = (
        games["_game_key"]
        .duplicated(keep=False)
    )

    if duplicate_game_ids.any():
        values = (
            games.loc[
                duplicate_game_ids,
                "_game_key",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Historical games file contains duplicate "
            "game_id values: "
            + ", ".join(values)
        )

    qb_index: dict[
        str,
        tuple[str, str],
    ] = {}

    for _, row in games.iterrows():
        game_id = str(
            row["_game_key"]
        )

        qb_index[game_id] = (
            normalize_player_id(
                row["home_qb_id"]
            ),
            normalize_player_id(
                row["away_qb_id"]
            ),
        )

    return qb_index


def load_final_team_rows(
    prior_season: int,
) -> dict[
    str,
    dict[str, float | None],
]:
    path = (
        TEAM_STATS_DIR
        / f"{prior_season}_team_stats.csv"
    )

    team_stats = read_csv(
        path
    )

    require_columns(
        team_stats,
        TEAM_SOURCE_REQUIRED_COLUMNS,
        f"team stats {prior_season}",
    )

    team_stats = team_stats[
        TEAM_SOURCE_REQUIRED_COLUMNS
    ].copy()

    team_stats["_season_key"] = (
        normalize_integer_key(
            team_stats["season"],
            f"{path}: season",
        )
    )

    team_stats["_week_key"] = (
        normalize_integer_key(
            team_stats["week"],
            f"{path}: week",
        )
    )

    team_stats["_team_key"] = (
        team_stats["team"]
        .map(normalize_team)
    )

    valid_seasons = sorted(
        int(value)
        for value in (
            team_stats["_season_key"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if valid_seasons != [
        prior_season
    ]:
        raise ValueError(
            f"{path}: expected only season "
            f"{prior_season}, found "
            f"{valid_seasons}"
        )

    for metric in TEAM_METRICS:
        team_stats[metric] = (
            numeric_metric(
                team_stats[metric],
                f"{path}: {metric}",
            )
        )

    team_stats = team_stats[
        team_stats["_week_key"].notna()
        & team_stats["_team_key"].ne("")
    ].copy()

    duplicate_keys = (
        team_stats.duplicated(
            subset=[
                "_season_key",
                "_week_key",
                "_team_key",
            ],
            keep=False,
        )
    )

    if duplicate_keys.any():
        duplicates = (
            team_stats.loc[
                duplicate_keys,
                [
                    "_season_key",
                    "_week_key",
                    "_team_key",
                ],
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            f"{path}: duplicate "
            f"season/week/team rows: "
            f"{duplicates}"
        )

    team_stats = (
        team_stats
        .sort_values(
            by=[
                "_team_key",
                "_week_key",
            ],
            kind="stable",
        )
        .groupby(
            "_team_key",
            sort=False,
            as_index=False,
        )
        .tail(1)
    )

    final_rows: dict[
        str,
        dict[str, float | None],
    ] = {}

    for _, row in team_stats.iterrows():
        team = str(
            row["_team_key"]
        )

        values: dict[
            str,
            float | None,
        ] = {}

        for metric in TEAM_METRICS:
            values[metric] = (
                clean_numeric_value(
                    row[metric]
                )
            )

        final_rows[team] = values

    return final_rows


def load_final_qb_rows(
    prior_season: int,
) -> dict[
    str,
    dict[str, float | None],
]:
    path = (
        QB_STATS_DIR
        / f"{prior_season}_qb_stats.csv"
    )

    qb_stats = read_csv(
        path
    )

    require_columns(
        qb_stats,
        QB_SOURCE_REQUIRED_COLUMNS,
        f"QB stats {prior_season}",
    )

    qb_stats = qb_stats[
        QB_SOURCE_REQUIRED_COLUMNS
    ].copy()

    qb_stats["_season_key"] = (
        normalize_integer_key(
            qb_stats["season"],
            f"{path}: season",
        )
    )

    qb_stats["_week_key"] = (
        normalize_integer_key(
            qb_stats["week"],
            f"{path}: week",
        )
    )

    qb_stats["_player_key"] = (
        qb_stats["player_id"]
        .map(normalize_player_id)
    )

    valid_seasons = sorted(
        int(value)
        for value in (
            qb_stats["_season_key"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if valid_seasons != [
        prior_season
    ]:
        raise ValueError(
            f"{path}: expected only season "
            f"{prior_season}, found "
            f"{valid_seasons}"
        )

    qb_stats["dropbacks"] = (
        numeric_metric(
            qb_stats["dropbacks"],
            f"{path}: dropbacks",
        )
    )

    for metric in QB_METRICS:
        qb_stats[metric] = (
            numeric_metric(
                qb_stats[metric],
                f"{path}: {metric}",
            )
        )

    qb_stats = qb_stats[
        qb_stats["_week_key"].notna()
        & qb_stats["_player_key"].ne("")
    ].copy()

    # Match Step 6's deterministic handling of multiple
    # player rows in the same season/week: retain the row
    # with the most dropbacks.
    qb_stats["_dropbacks_sort"] = (
        qb_stats["dropbacks"]
        .fillna(-1)
    )

    qb_stats = qb_stats.sort_values(
        by=[
            "_player_key",
            "_week_key",
            "_dropbacks_sort",
        ],
        kind="stable",
    )

    qb_stats = qb_stats.drop_duplicates(
        subset=[
            "_season_key",
            "_player_key",
            "_week_key",
        ],
        keep="last",
    )

    # Retain each player's final available row from the
    # prior season.
    qb_stats = (
        qb_stats
        .sort_values(
            by=[
                "_player_key",
                "_week_key",
            ],
            kind="stable",
        )
        .groupby(
            "_player_key",
            sort=False,
            as_index=False,
        )
        .tail(1)
    )

    final_rows: dict[
        str,
        dict[str, float | None],
    ] = {}

    for _, row in qb_stats.iterrows():
        player_id = str(
            row["_player_key"]
        )

        values: dict[
            str,
            float | None,
        ] = {}

        for metric in QB_METRICS:
            values[metric] = (
                clean_numeric_value(
                    row[metric]
                )
            )

        final_rows[player_id] = values

    return final_rows


def assign_value(
    training: pd.DataFrame,
    index: object,
    column: str,
    value: float | None,
) -> None:
    """
    Training CSVs are intentionally read with dtype=str.

    Pandas may represent those columns with a strict string dtype,
    which rejects direct assignment of Python floats. Write finite
    numeric values back as strings so the CSV schema stays unchanged.
    """
    if value is None:
        training.at[
            index,
            column,
        ] = pd.NA
    else:
        training.at[
            index,
            column,
        ] = str(value)


def subtract_values(
    home_value: float | None,
    away_value: float | None,
) -> float | None:
    if (
        home_value is None
        or away_value is None
    ):
        return None

    return (
        home_value
        - away_value
    )


def process_season(
    season: int,
    games_qb_index: dict[
        str,
        tuple[str, str],
    ],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    bool,
]:
    training_path = (
        TRAINING_PATHS[
            season
        ]
    )

    training = read_csv(
        training_path
    )

    require_columns(
        training,
        [
            *TRAINING_REQUIRED_COLUMNS,
            *STEP7_COLUMNS,
        ],
        f"historical training table {season}",
    )

    original_columns = (
        training.columns.tolist()
    )

    original_row_count = len(
        training
    )

    season_keys = (
        normalize_integer_key(
            training["season"],
            f"{training_path}: season",
        )
    )

    week_keys = (
        normalize_integer_key(
            training["week"],
            f"{training_path}: week",
        )
    )

    seasons_in_file = sorted(
        int(value)
        for value in (
            season_keys
            .dropna()
            .unique()
            .tolist()
        )
    )

    if seasons_in_file != [
        season
    ]:
        raise ValueError(
            f"{training_path}: expected only "
            f"season {season}, found "
            f"{seasons_in_file}"
        )

    week_one_mask = (
        week_keys == 1
    )

    week_one_indexes = (
        training.index[
            week_one_mask
        ].tolist()
    )

    if not week_one_indexes:
        raise RuntimeError(
            f"{training_path}: no Week 1 "
            f"rows found."
        )

    prior_season = (
        season - 1
    )

    team_home_matches = 0
    team_away_matches = 0
    team_both_matches = 0

    qb_home_matches = 0
    qb_away_matches = 0
    qb_both_matches = 0

    missing_game_ids = 0

    # The available historical team/QB intake starts in 2021.
    # Therefore the 2021 training file has no 2020 source from
    # which to populate its Week 1 lagged values.
    #
    # Leave 2021 completely unchanged and do not rewrite the file.
    if season == 2021:
        print(
            "Season 2021: Week 1 prior-season fallback "
            "skipped because 2020 team/QB source files "
            "are unavailable."
        )

        stats = {
            "rows": original_row_count,
            "week_one_rows": len(
                week_one_indexes
            ),
            "team_home_matches": 0,
            "team_away_matches": 0,
            "team_both_matches": 0,
            "qb_home_matches": 0,
            "qb_away_matches": 0,
            "qb_both_matches": 0,
            "missing_game_ids": 0,
        }

        return (
            training,
            stats,
            False,
        )

    non_week_one_before = (
        training.loc[
            ~week_one_mask,
            STEP7_COLUMNS,
        ]
        .copy()
    )

    final_team_rows = (
        load_final_team_rows(
            prior_season
        )
    )

    final_qb_rows = (
        load_final_qb_rows(
            prior_season
        )
    )

    for index in week_one_indexes:
        row = training.loc[
            index
        ]

        home_team = (
            normalize_team(
                row["home_team"]
            )
        )

        away_team = (
            normalize_team(
                row["away_team"]
            )
        )

        home_team_values = (
            final_team_rows.get(
                home_team
            )
        )

        away_team_values = (
            final_team_rows.get(
                away_team
            )
        )

        if (
            home_team_values
            is not None
        ):
            team_home_matches += 1

            for metric in TEAM_METRICS:
                assign_value(
                    training,
                    index,
                    f"home_{metric}",
                    home_team_values[
                        metric
                    ],
                )
        else:
            for metric in TEAM_METRICS:
                assign_value(
                    training,
                    index,
                    f"home_{metric}",
                    None,
                )

        if (
            away_team_values
            is not None
        ):
            team_away_matches += 1

            for metric in TEAM_METRICS:
                assign_value(
                    training,
                    index,
                    f"away_{metric}",
                    away_team_values[
                        metric
                    ],
                )
        else:
            for metric in TEAM_METRICS:
                assign_value(
                    training,
                    index,
                    f"away_{metric}",
                    None,
                )

        if (
            home_team_values
            is not None
            and away_team_values
            is not None
        ):
            team_both_matches += 1

        for metric in TEAM_METRICS:
            home_value = (
                None
                if home_team_values
                is None
                else home_team_values[
                    metric
                ]
            )

            away_value = (
                None
                if away_team_values
                is None
                else away_team_values[
                    metric
                ]
            )

            assign_value(
                training,
                index,
                f"{metric}_diff",
                subtract_values(
                    home_value,
                    away_value,
                ),
            )

        game_id = (
            normalize_game_id(
                row["game_id"]
            )
        )

        qb_ids = (
            games_qb_index.get(
                game_id
            )
        )

        if qb_ids is None:
            missing_game_ids += 1
            home_qb_id = ""
            away_qb_id = ""
        else:
            (
                home_qb_id,
                away_qb_id,
            ) = qb_ids

        home_qb_values = (
            final_qb_rows.get(
                home_qb_id
            )
            if home_qb_id
            else None
        )

        away_qb_values = (
            final_qb_rows.get(
                away_qb_id
            )
            if away_qb_id
            else None
        )

        if (
            home_qb_values
            is not None
        ):
            qb_home_matches += 1

            for metric in QB_METRICS:
                assign_value(
                    training,
                    index,
                    f"home_qb_{metric}",
                    home_qb_values[
                        metric
                    ],
                )
        else:
            for metric in QB_METRICS:
                assign_value(
                    training,
                    index,
                    f"home_qb_{metric}",
                    None,
                )

        if (
            away_qb_values
            is not None
        ):
            qb_away_matches += 1

            for metric in QB_METRICS:
                assign_value(
                    training,
                    index,
                    f"away_qb_{metric}",
                    away_qb_values[
                        metric
                    ],
                )
        else:
            for metric in QB_METRICS:
                assign_value(
                    training,
                    index,
                    f"away_qb_{metric}",
                    None,
                )

        if (
            home_qb_values
            is not None
            and away_qb_values
            is not None
        ):
            qb_both_matches += 1

        for metric in QB_METRICS:
            home_value = (
                None
                if home_qb_values
                is None
                else home_qb_values[
                    metric
                ]
            )

            away_value = (
                None
                if away_qb_values
                is None
                else away_qb_values[
                    metric
                ]
            )

            assign_value(
                training,
                index,
                f"qb_{metric}_diff",
                subtract_values(
                    home_value,
                    away_value,
                ),
            )

    if len(training) != original_row_count:
        raise RuntimeError(
            f"{season}: row count changed "
            f"during Step 7: "
            f"before={original_row_count} "
            f"after={len(training)}"
        )

    if (
        training.columns.tolist()
        != original_columns
    ):
        raise RuntimeError(
            f"{season}: column order changed "
            f"during Step 7."
        )

    non_week_one_after = (
        training.loc[
            ~week_one_mask,
            STEP7_COLUMNS,
        ]
    )

    if not non_week_one_after.equals(
        non_week_one_before
    ):
        raise RuntimeError(
            f"{season}: Step 7 modified "
            f"Week 2+ feature values."
        )

    stats = {
        "rows": original_row_count,
        "week_one_rows": len(
            week_one_indexes
        ),
        "team_home_matches": (
            team_home_matches
        ),
        "team_away_matches": (
            team_away_matches
        ),
        "team_both_matches": (
            team_both_matches
        ),
        "qb_home_matches": (
            qb_home_matches
        ),
        "qb_away_matches": (
            qb_away_matches
        ),
        "qb_both_matches": (
            qb_both_matches
        ),
        "missing_game_ids": (
            missing_game_ids
        ),
    }

    return (
        training,
        stats,
        True,
    )


def write_outputs(
    outputs: dict[
        int,
        pd.DataFrame,
    ],
) -> None:
    temp_paths: dict[
        int,
        Path,
    ] = {}

    try:
        for season, training in (
            outputs.items()
        ):
            output_path = (
                TRAINING_PATHS[
                    season
                ]
            )

            temp_path = (
                output_path
                .with_suffix(
                    ".step7.tmp.csv"
                )
            )

            temp_paths[
                season
            ] = temp_path

            training.to_csv(
                temp_path,
                index=False,
                encoding="utf-8",
            )

        for season in sorted(
            temp_paths
        ):
            temp_paths[
                season
            ].replace(
                TRAINING_PATHS[
                    season
                ]
            )

    except Exception:
        for temp_path in (
            temp_paths.values()
        ):
            if temp_path.exists():
                temp_path.unlink()

        raise


def main() -> int:
    games_qb_index = (
        load_games_qb_index()
    )

    outputs: dict[
        int,
        pd.DataFrame,
    ] = {}

    results: dict[
        int,
        dict[str, int],
    ] = {}

    for season in TRAINING_SEASONS:
        (
            training,
            stats,
            should_write,
        ) = process_season(
            season,
            games_qb_index,
        )

        if should_write:
            outputs[
                season
            ] = training

        results[
            season
        ] = stats

    write_outputs(
        outputs
    )

    total_week_one_rows = 0
    total_team_home = 0
    total_team_away = 0
    total_team_both = 0
    total_qb_home = 0
    total_qb_away = 0
    total_qb_both = 0
    total_missing_games = 0

    for season in TRAINING_SEASONS:
        stats = results[
            season
        ]

        total_week_one_rows += (
            stats["week_one_rows"]
        )

        total_team_home += (
            stats[
                "team_home_matches"
            ]
        )

        total_team_away += (
            stats[
                "team_away_matches"
            ]
        )

        total_team_both += (
            stats[
                "team_both_matches"
            ]
        )

        total_qb_home += (
            stats[
                "qb_home_matches"
            ]
        )

        total_qb_away += (
            stats[
                "qb_away_matches"
            ]
        )

        total_qb_both += (
            stats[
                "qb_both_matches"
            ]
        )

        total_missing_games += (
            stats[
                "missing_game_ids"
            ]
        )

        print(
            f"Season {season}"
        )

        print(
            f"Week 1 rows: "
            f"{stats['week_one_rows']}"
        )

        print(
            f"Home team prior-season matches: "
            f"{stats['team_home_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Away team prior-season matches: "
            f"{stats['team_away_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Both-team prior-season matches: "
            f"{stats['team_both_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Home QB prior-season matches: "
            f"{stats['qb_home_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Away QB prior-season matches: "
            f"{stats['qb_away_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Both-QB prior-season matches: "
            f"{stats['qb_both_matches']}/"
            f"{stats['week_one_rows']}"
        )

        print(
            f"Week 1 game IDs missing from "
            f"historical games: "
            f"{stats['missing_game_ids']}"
        )

        if season == 2021:
            print(
                "Wrote: no "
                "(2021 left unchanged)"
            )
        else:
            print(
                f"Wrote: "
                f"{TRAINING_PATHS[season]}"
            )

        print()

    print(
        "Step 7 complete."
    )

    print(
        f"Total Week 1 rows: "
        f"{total_week_one_rows}"
    )

    print(
        f"Total home team matches: "
        f"{total_team_home}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total away team matches: "
        f"{total_team_away}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total both-team matches: "
        f"{total_team_both}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total home QB matches: "
        f"{total_qb_home}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total away QB matches: "
        f"{total_qb_away}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total both-QB matches: "
        f"{total_qb_both}/"
        f"{total_week_one_rows}"
    )

    print(
        f"Total Week 1 game IDs missing "
        f"from historical games: "
        f"{total_missing_games}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
