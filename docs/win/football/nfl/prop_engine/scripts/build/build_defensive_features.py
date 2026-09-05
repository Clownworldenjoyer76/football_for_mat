#!/usr/bin/env python3
"""
Build defensive-specific historical player-game features.

READS:
    historical player-game universe
    player_form
    player_role_history
    team_form
    opponent_week_opportunity

WRITES:
    defensive_features.parquet

POLICY:
    - Output preserves the full canonical player-game universe grain.
    - All player history columns are copied from already leakage-safe player_form.
    - Opponent offensive context comes from the opponent's lagged team_form row.
    - team_def_sack_rate_roll3 is the mean of strictly prior per-game
      defense sack rates (sacks / opponent_dropbacks) over the last 3
      observed team games.
    - No pass_rush_share is created.
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


OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "position",
    "def_snap_pct_lag1",
    "def_snap_pct_roll3",
    "def_participation_lag1",
    "def_participation_roll3",
    "tackles_lag1",
    "tackles_roll3",
    "tackles_roll5",
    "tackle_rate_roll3",
    "tackle_rate_roll5",
    "sacks_lag1",
    "sacks_roll3",
    "sacks_roll5",
    "sack_rate_roll5",
    "qb_hits_roll3",
    "qb_hits_roll5",
    "opponent_plays_roll3",
    "opponent_dropbacks_roll3",
    "opponent_rush_rate_roll3",
    "opponent_pass_rate_roll3",
    "team_def_sack_rate_roll3",
    "starter_flag",
    "front7_flag",
    "secondary_flag",
]

GRAIN = ["season", "week", "game_id", "player_id"]

PLAYER_FORM_MAP = {
    "def_snap_pct_lag1": "defense_snap_pct_lag1",
    "def_snap_pct_roll3": "defense_snap_pct_roll3_mean",
    "def_participation_lag1": "defense_participation_lag1",
    "def_participation_roll3": "defense_participation_roll3_mean",
    "tackles_lag1": "tackles_lag1",
    "tackles_roll3": "tackles_roll3_mean",
    "tackles_roll5": "tackles_roll5_mean",
    "tackle_rate_roll3": "tackle_rate_per_def_play_roll3_mean",
    "tackle_rate_roll5": "tackle_rate_per_def_play_roll5_mean",
    "sacks_lag1": "sacks_lag1",
    "sacks_roll3": "sacks_roll3_mean",
    "sacks_roll5": "sacks_roll5_mean",
    "sack_rate_roll5": "sack_rate_per_def_play_roll5_mean",
    "qb_hits_roll3": "qb_hits_roll3_mean",
    "qb_hits_roll5": "qb_hits_roll5_mean",
}

OPPONENT_TEAM_FORM_MAP = {
    "opponent_plays_roll3": "offensive_plays_roll3_mean",
    "opponent_dropbacks_roll3": "dropbacks_roll3_mean",
    "opponent_rush_rate_roll3": "rush_rate_roll3_mean",
    "opponent_pass_rate_roll3": "pass_rate_roll3_mean",
}

FRONT7_POSITIONS = {
    "DL",
    "DE",
    "LDE",
    "RDE",
    "DT",
    "LDT",
    "RDT",
    "NT",
    "EDGE",
    "LB",
    "ILB",
    "OLB",
    "MLB",
    "WLB",
    "SLB",
}

SECONDARY_POSITIONS = {
    "DB",
    "CB",
    "LCB",
    "RCB",
    "NB",
    "S",
    "SAF",
    "FS",
    "SS",
}

HISTORICAL_FRANCHISE_ALIASES = {
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


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def normalize_position(value: Any) -> str:
    return clean(value).upper().replace(" ", "")


def safe_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype("float64")

    den = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype("float64")

    result = pd.Series(
        np.nan,
        index=num.index,
        dtype="float64",
    )

    valid = (
        num.notna()
        & den.notna()
        & den.ne(0.0)
    )

    result.loc[valid] = (
        num.loc[valid]
        / den.loc[valid]
    )

    return result


def build_team_def_sack_rate_roll3(
    opponent_raw: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "season",
        "week",
        "team",
        "sacks",
        "opponent_dropbacks",
    ]

    common.require_columns(
        opponent_raw,
        required,
        "opponent opportunity",
    )

    source = opponent_raw[required].copy()

    source["season"] = pd.to_numeric(
        source["season"],
        errors="raise",
    ).astype(int)

    source["week"] = pd.to_numeric(
        source["week"],
        errors="raise",
    ).astype(int)

    source["team"] = source["team"].map(
        canonical_team
    )

    common.ensure_unique(
        source,
        ["season", "week", "team"],
        "opponent opportunity team grain",
    )

    source["game_sack_rate"] = safe_rate(
        source["sacks"],
        source["opponent_dropbacks"],
    )

    source = (
        source.sort_values(
            ["team", "season", "week"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    output = source[
        ["season", "week", "team"]
    ].copy()

    values = source[
        "game_sack_rate"
    ].to_numpy(dtype="float64")

    roll3 = np.full(
        len(source),
        np.nan,
        dtype="float64",
    )

    for _, indices in source.groupby(
        "team",
        sort=False,
    ).indices.items():
        history: list[float] = []

        for pos in indices:
            if history:
                roll3[pos] = float(
                    np.mean(history[-3:])
                )

            current = values[pos]

            if np.isfinite(current):
                history.append(float(current))

    output["team_def_sack_rate_roll3"] = roll3

    common.ensure_unique(
        output,
        ["season", "week", "team"],
        "team defensive sack-rate form",
    )

    return output


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
            "depth_starter_flag",
        ],
    )

    player_form = common.read_parquet_required(
        paths["player_form"],
        [
            *GRAIN,
            *PLAYER_FORM_MAP.values(),
        ],
    )

    role_history = common.read_parquet_required(
        paths["role_history"],
        [
            *GRAIN,
            "depth_starter_flag_pregame",
        ],
    )

    team_form = common.read_parquet_required(
        paths["team_form"],
        [
            "season",
            "week",
            "team",
            *OPPONENT_TEAM_FORM_MAP.values(),
        ],
    )

    opponent_raw = common.read_parquet_required(
        paths["opponent_opportunity"],
        [
            "season",
            "week",
            "team",
            "sacks",
            "opponent_dropbacks",
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

    base = universe[
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
        ]
    ].copy()

    # Player-level strictly prior history.
    player_cols = [
        *GRAIN,
        *PLAYER_FORM_MAP.values(),
    ]

    pf = player_form[player_cols].rename(
        columns={
            source: target
            for target, source in PLAYER_FORM_MAP.items()
        }
    )

    base = base.merge(
        pf,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    # Pregame starter state.
    starter = role_history[
        [
            *GRAIN,
            "depth_starter_flag_pregame",
        ]
    ].rename(
        columns={
            "depth_starter_flag_pregame": "starter_flag",
        }
    )

    base = base.merge(
        starter,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )

    # Matchup offense: defender's opponent -> opponent team_form.
    tf = team_form[
        [
            "season",
            "week",
            "team",
            *OPPONENT_TEAM_FORM_MAP.values(),
        ]
    ].copy()

    tf["team"] = tf["team"].map(canonical_team)

    tf = tf.rename(
        columns={
            "team": "_matchup_team",
            **{
                source: target
                for target, source
                in OPPONENT_TEAM_FORM_MAP.items()
            },
        }
    )

    base["_matchup_team"] = base[
        "opponent"
    ].map(canonical_team)

    base = base.merge(
        tf,
        on=[
            "season",
            "week",
            "_matchup_team",
        ],
        how="left",
        validate="many_to_one",
    )

    # Team defense form.
    team_sack_rate = (
        build_team_def_sack_rate_roll3(
            opponent_raw
        )
    )

    team_sack_rate["team"] = (
        team_sack_rate["team"]
        .map(canonical_team)
    )

    team_sack_rate = team_sack_rate.rename(
        columns={
            "team": "_defense_team",
        }
    )

    base["_defense_team"] = base[
        "team"
    ].map(canonical_team)

    base = base.merge(
        team_sack_rate,
        on=[
            "season",
            "week",
            "_defense_team",
        ],
        how="left",
        validate="many_to_one",
    )

    position = base[
        "position"
    ].map(normalize_position)

    base["front7_flag"] = (
        position.isin(FRONT7_POSITIONS)
        .astype("int8")
    )

    base["secondary_flag"] = (
        position.isin(SECONDARY_POSITIONS)
        .astype("int8")
    )

    base["starter_flag"] = (
        pd.to_numeric(
            base["starter_flag"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )

    output = base[OUTPUT_COLUMNS].copy()

    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(
            "Issue 15 output header/order mismatch."
        )

    if len(output) != len(universe):
        raise RuntimeError(
            "Issue 15 output row count differs from universe."
        )

    common.ensure_unique(
        output,
        GRAIN,
        "Issue 15 defensive feature grain",
    )

    if (
        output["front7_flag"]
        + output["secondary_flag"]
        > 1
    ).any():
        raise ValueError(
            "front7_flag and secondary_flag overlap."
        )

    for column in [
        "starter_flag",
        "front7_flag",
        "secondary_flag",
    ]:
        if not output[column].isin([0, 1]).all():
            raise ValueError(
                f"{column}: non-binary value."
            )

    numeric_cols = [
        c
        for c in OUTPUT_COLUMNS
        if c
        not in {
            "game_id",
            "player_id",
            "position",
        }
    ]

    matrix = (
        output[numeric_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype="float64")
    )

    if np.isinf(matrix).any():
        raise ValueError(
            "Issue 15 output contains infinity."
        )

    if any(
        "pass_rush_share" in c.casefold()
        for c in output.columns
    ):
        raise ValueError(
            "Forbidden pass_rush_share created."
        )

    output_path = paths.get(
        "defensive_features",
        "docs/win/football/nfl/prop_engine/data/historical/features/defensive_features.parquet",
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
        "front7_rows": int(
            output["front7_flag"].sum()
        ),
        "secondary_rows": int(
            output["secondary_flag"].sum()
        ),
        "starter_rows": int(
            output["starter_flag"].sum()
        ),
        "team_def_sack_rate_policy": (
            "mean of last 3 observed strictly prior per-game "
            "sacks/opponent_dropbacks rates"
        ),
        "opponent_context_policy": (
            "defender opponent joined to already-lagged team_form"
        ),
        "pass_rush_share_created": False,
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
