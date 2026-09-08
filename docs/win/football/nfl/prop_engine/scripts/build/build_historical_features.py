#!/usr/bin/env python3
"""
Assemble the canonical historical Prop Engine feature table.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    docs/win/football/nfl/prop_engine/data/historical/universe/player_game_universe.parquet
    docs/win/football/nfl/prop_engine/data/historical/targets/player_game_targets.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/player_role_history.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/player_form.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/team_form.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/opponent_form.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/environment.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/defensive_features.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/kicking_features.parquet
    docs/win/football/nfl/prop_engine/data/historical/opportunity/position_allowed_week.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/historical/features/player_game_features.parquet
    docs/win/football/nfl/prop_engine/data/historical/features/feature_manifest.json

POLICY:
    - Canonical grain is season + week + game_id + player_id.
    - Target-game outcomes are stored only in target_* columns and never enter
      the model feature manifest.
    - Weekly player/team/opponent form inputs are already strictly pregame.
    - Position-allowed realized weekly values are shifted one observed defense
      game before joining to the target row.
    - No sportsbook, market, score-result, same-game snap, same-game
      participation, or played_game_flag field is exposed as a model feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


GRAIN = ["season", "week", "game_id", "player_id"]

LEADING_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "gameday",
    "kickoff_timestamp",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "position_group",
    "home_flag",
]

REQUIRED_TARGET_ORDER = list(common.load_config()["targets"].keys())

TARGET_COLUMNS = [f"target_{name}" for name in REQUIRED_TARGET_ORDER]

REQUIRED_MATCHUP_COLUMNS = [
    "matchup_expected_team_plays",
    "matchup_expected_team_dropbacks",
    "matchup_expected_team_rush_attempts",
    "matchup_expected_opponent_plays",
    "matchup_expected_opponent_dropbacks",
    "matchup_player_target_share_x_opp_targets",
    "matchup_player_carry_share_x_opp_rushes",
    "matchup_player_tackle_rate_x_opp_plays",
    "matchup_player_sack_rate_x_opp_plays",
    "matchup_off_epa_vs_def_epa",
    "matchup_pass_rate_vs_opponent",
    "matchup_rush_rate_vs_opponent",
]

AUDIT_COLUMNS = [
    "audit_feature_asof",
    "audit_max_player_source_game",
    "audit_max_team_source_week",
    "audit_depth_snapshot_at",
    "audit_injury_snapshot_at",
    "audit_market_feature_count",
    "audit_row_valid",
]

ROLE_KEYS = set(GRAIN + ["team", "position"])
PLAYER_FORM_KEYS = set(
    GRAIN + ["team", "position", "position_group"]
)
TEAM_FORM_KEYS = {"season", "week", "team"}
ENVIRONMENT_KEYS = {
    "season",
    "week",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
}

PLAYER_HISTORY_COLUMNS = [
    "no_nfl_history_flag",
    "new_team_flag",
    "history_games",
]

DEFENSIVE_ROLE_MAP = {
    "starter_flag": "role_defensive_starter_flag",
    "front7_flag": "role_front7_flag",
    "secondary_flag": "role_secondary_flag",
}

KICKING_ROLE_MAP = {
    "primary_kicker_flag": "role_primary_kicker_flag",
}

KICKING_REDUNDANT_ENVIRONMENT = {
    "temperature",
    "wind",
    "roof",
    "surface",
}

POSITION_ALLOWED_KEYS = [
    "season",
    "week",
    "defense_team",
    "offense_position_group",
]

HISTORICAL_FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

TEAM_FORM_SAFE_SUFFIXES = (
    "_lag1",
    "_roll3_mean",
    "_roll5_mean",
    "_roll8_mean",
    "_ewm3",
    "_ewm5",
    "_season_to_date",
)

FINAL_SCORE_FORBIDDEN_NAMES = {
    "score",
    "home_score",
    "away_score",
    "final_score",
    "score_differential",
    "point_differential",
    "margin",
    "result",
    "win_flag",
    "loss_flag",
}

ENVIRONMENT_DIRECT_COLUMNS = [
    "divisional_game_flag",
    "neutral_site_flag",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]

ENVIRONMENT_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "divisional_game_flag",
    "neutral_site_flag",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "home_rest_days",
    "away_rest_days",
    "miles_traveled_away",
    "time_zones_crossed_away",
    "east_to_west_flag",
    "west_to_east_flag",
    "international_flag",
    "weather_missing_flag",
    "travel_missing_flag",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def canonical_franchise(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_FRANCHISE_ALIASES.get(team, team)


def normalize_position_group(value: Any) -> str:
    return clean_text(value).upper()


def require_config_path(config: dict, key: str) -> str:
    value = config.get("paths", {}).get(key)
    if not value:
        raise ValueError(
            f"Issue 17 requires config paths.{key}."
        )
    return str(value)


def stable_schema_hash(df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_mean_pair(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce").astype("float64")
    b = pd.to_numeric(right, errors="coerce").astype("float64")
    return pd.concat([a, b], axis=1).mean(axis=1, skipna=True)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & den.ne(0.0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    return result


def safe_product(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce").astype("float64")
    b = pd.to_numeric(right, errors="coerce").astype("float64")
    result = a * b
    result.loc[a.isna() | b.isna()] = np.nan
    return result.astype("float64")


def validate_exact_full_grain(
    universe_keys: pd.DataFrame,
    source: pd.DataFrame,
    label: str,
) -> None:
    common.require_columns(source, GRAIN, label)
    common.ensure_unique(source, GRAIN, f"{label} grain")
    left = (
        universe_keys[GRAIN]
        .sort_values(GRAIN, kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        source[GRAIN]
        .sort_values(GRAIN, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(left) != len(right) or not left.equals(right):
        raise ValueError(
            f"{label}: grain does not exactly match historical universe."
        )


def validate_sparse_grain_subset(
    universe_keys: pd.DataFrame,
    source: pd.DataFrame,
    label: str,
) -> None:
    common.require_columns(source, GRAIN, label)
    common.ensure_unique(source, GRAIN, f"{label} grain")
    probe = source[GRAIN].merge(
        universe_keys[GRAIN],
        on=GRAIN,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    bad = probe["_merge"].ne("both")
    if bad.any():
        sample = probe.loc[bad, GRAIN].head(10).to_dict("records")
        raise ValueError(
            f"{label}: contains rows outside universe; sample={sample}"
        )


def prefix_columns(
    source: pd.DataFrame,
    source_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    return source[source_columns].rename(
        columns={column: f"{prefix}{column}" for column in source_columns}
    )


def build_position_allowed_lag(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    common.require_columns(
        source,
        POSITION_ALLOWED_KEYS + [
            "targets_allowed",
            "carries_allowed",
            "league_rate",
            "shrunk_rate",
        ],
        "position allowed",
    )
    common.ensure_unique(
        source,
        POSITION_ALLOWED_KEYS,
        "position allowed grain",
    )

    data = source.copy()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["week"] = pd.to_numeric(data["week"], errors="raise").astype(int)
    data["_join_defense"] = data["defense_team"].map(canonical_franchise)
    data["_join_position_group"] = data[
        "offense_position_group"
    ].map(normalize_position_group)

    value_columns = [
        column
        for column in data.columns
        if column not in set(POSITION_ALLOWED_KEYS)
        and column not in {"_join_defense", "_join_position_group"}
    ]

    for column in value_columns:
        data[column] = common.safe_numeric(data[column])

    data = data.sort_values(
        [
            "_join_defense",
            "_join_position_group",
            "season",
            "week",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = data.groupby(
        ["_join_defense", "_join_position_group"],
        sort=False,
        dropna=False,
    )

    output = data[
        [
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ]
    ].copy()

    matchup_columns: list[str] = []
    for column in value_columns:
        output_name = f"matchup_position_allowed_{column}_lag1"
        output[output_name] = grouped[column].shift(1)
        matchup_columns.append(output_name)

    common.ensure_unique(
        output,
        [
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ],
        "lagged position allowed join grain",
    )
    return output, matchup_columns


def build_player_audit(universe: pd.DataFrame) -> pd.DataFrame:
    common.require_columns(
        universe,
        GRAIN + ["kickoff_timestamp", "played_game_flag"],
        "historical universe player audit",
    )

    audit = universe[
        GRAIN + ["kickoff_timestamp", "played_game_flag"]
    ].copy()

    audit["_kickoff_sort"] = pd.to_datetime(
        audit["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )
    played = pd.to_numeric(
        audit["played_game_flag"],
        errors="coerce",
    ).fillna(0).eq(1)

    audit = audit.sort_values(
        ["player_id", "_kickoff_sort", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    played = pd.to_numeric(
        audit["played_game_flag"],
        errors="coerce",
    ).fillna(0).eq(1)

    audit["_played_source_game"] = (
        audit["game_id"].astype("string").where(played)
    )
    audit["audit_max_player_source_game"] = (
        audit.groupby("player_id", sort=False)["_played_source_game"]
        .transform(lambda series: series.ffill().shift(1))
    )

    return audit[
        GRAIN + ["audit_max_player_source_game"]
    ]


def build_team_source_audit(
    team_form: pd.DataFrame,
) -> pd.DataFrame:
    common.require_columns(
        team_form,
        ["season", "week", "team"],
        "team form audit",
    )

    keys = team_form[["season", "week", "team"]].copy()
    keys["season"] = pd.to_numeric(keys["season"], errors="raise").astype(int)
    keys["week"] = pd.to_numeric(keys["week"], errors="raise").astype(int)
    keys["_join_team"] = keys["team"].map(canonical_franchise)

    common.ensure_unique(
        keys,
        ["season", "week", "_join_team"],
        "team form canonical audit grain",
    )

    keys = keys.sort_values(
        ["_join_team", "season", "week"],
        kind="mergesort",
    ).reset_index(drop=True)

    source_label = (
        keys["season"].astype(str)
        + "-W"
        + keys["week"].astype(str).str.zfill(2)
    )

    keys["audit_max_team_source_week"] = (
        source_label.groupby(keys["_join_team"], sort=False).shift(1)
    )

    return keys[
        [
            "season",
            "week",
            "_join_team",
            "audit_max_team_source_week",
        ]
    ]


def write_json_atomic(
    payload: dict,
    relative_path: str,
) -> None:
    destination = (common.repo_root() / relative_path).resolve()
    root = common.prop_root().resolve()

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest write outside Prop Engine is forbidden: {destination}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)

    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
            )
            handle.write("\n")
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def validate_form_column_names(
    columns: list[str],
    label: str,
) -> None:
    bad = [
        column
        for column in columns
        if not column.endswith(TEAM_FORM_SAFE_SUFFIXES)
    ]
    if bad:
        raise ValueError(
            f"{label}: non-lagged form columns detected: {bad[:20]}"
        )


def validate_no_same_game_usage_features(
    feature_columns: list[str],
) -> None:
    usage_tokens = ("snap", "participation")
    safe_tokens = (
        "prior",
        "lag",
        "roll",
        "ewm",
        "change",
        "unavailable",
        "missing",
        "season_to_date",
        "career_prior",
    )

    bad: list[str] = []
    for column in feature_columns:
        lowered = column.casefold()
        if not any(token in lowered for token in usage_tokens):
            continue
        if not any(token in lowered for token in safe_tokens):
            bad.append(column)

    if bad:
        raise ValueError(
            "Potential same-game snap/participation features detected: "
            + ", ".join(bad[:20])
        )


def validate_no_final_score_features(
    feature_columns: list[str],
) -> None:
    bad = [
        column
        for column in feature_columns
        if column.casefold() in FINAL_SCORE_FORBIDDEN_NAMES
        or "final_score" in column.casefold()
    ]
    if bad:
        raise ValueError(
            f"Final-score/result features are forbidden: {bad}"
        )


def classify_feature_columns(
    frame: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []

    for column in candidate_columns:
        dtype = frame[column].dtype

        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric.append(column)
        elif (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            raise ValueError(
                f"Unclassified model-feature dtype: {column}={dtype}"
            )

    return numeric, categorical


def main() -> int:
    config = common.load_config()

    required_config_targets = list(config.get("targets", {}).keys())
    missing_targets = [
        name
        for name in REQUIRED_TARGET_ORDER
        if name not in required_config_targets
    ]
    if missing_targets:
        raise ValueError(
            "Issue 17 missing configured target(s): "
            + ", ".join(missing_targets)
        )

    paths = {
        "universe": require_config_path(config, "historical_universe"),
        "targets": require_config_path(config, "historical_targets"),
        "role": require_config_path(config, "role_history"),
        "player": require_config_path(config, "player_form"),
        "team": require_config_path(config, "team_form"),
        "opponent": require_config_path(config, "opponent_form"),
        "environment": require_config_path(config, "environment_history"),
        "defensive": require_config_path(config, "defensive_features"),
        "kicking": require_config_path(config, "kicking_features"),
        "position_allowed": require_config_path(config, "position_allowed"),
        "output": require_config_path(config, "historical_features"),
    }

    manifest_path = (
        "docs/win/football/nfl/prop_engine/data/historical/"
        "features/feature_manifest.json"
    )

    universe = common.read_parquet_required(
        paths["universe"],
        LEADING_COLUMNS + ["played_game_flag"],
    )
    common.ensure_unique(universe, GRAIN, "historical universe")
    universe_keys = universe[GRAIN].copy()

    universe_row_count = len(universe)
    out = universe[LEADING_COLUMNS].copy()
    out["_join_team"] = out["team"].map(canonical_franchise)
    out["_join_opponent"] = out["opponent"].map(canonical_franchise)
    out["_join_position_group"] = out["position_group"].map(
        normalize_position_group
    )

    if out["_join_team"].eq("").any() or out["_join_opponent"].eq("").any():
        raise ValueError("Universe contains blank canonical team/opponent.")

    role_columns: list[str] = []
    player_columns: list[str] = []
    team_columns: list[str] = []
    opponent_columns: list[str] = []
    matchup_columns: list[str] = []
    environment_columns: list[str] = []
    history_columns: list[str] = []

    # Role history: full-universe one-to-one join.
    role = common.read_parquet_required(paths["role"], GRAIN)
    validate_exact_full_grain(universe_keys, role, "role history")

    role_source_columns = [
        column
        for column in role.columns
        if column not in ROLE_KEYS
    ]
    role_renamed = {
        column: f"role_{column}"
        for column in role_source_columns
    }
    role_columns.extend(role_renamed.values())

    out = out.merge(
        role[GRAIN + role_source_columns].rename(columns=role_renamed),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del role
    gc.collect()

    # Player form: full-universe one-to-one join. History flags get their
    # own family; all rolling/form values get player_*.
    player = common.read_parquet_required(paths["player"], GRAIN)
    validate_exact_full_grain(universe_keys, player, "player form")

    for required_history in PLAYER_HISTORY_COLUMNS:
        if required_history not in player.columns:
            raise ValueError(
                f"player form missing required history field {required_history}"
            )

    player_source_columns = [
        column
        for column in player.columns
        if column not in PLAYER_FORM_KEYS
        and column not in PLAYER_HISTORY_COLUMNS
    ]
    player_renamed = {
        column: f"player_{column}"
        for column in player_source_columns
    }
    player_columns.extend(player_renamed.values())

    history_renamed = {
        column: f"history_{column}"
        for column in PLAYER_HISTORY_COLUMNS
    }
    history_columns.extend(history_renamed.values())

    out = out.merge(
        player[
            GRAIN + player_source_columns + PLAYER_HISTORY_COLUMNS
        ].rename(columns={**player_renamed, **history_renamed}),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del player
    gc.collect()

    # Team form: join player's offense by current game-specific team.
    team = common.read_parquet_required(
        paths["team"],
        ["season", "week", "team"],
    )
    common.ensure_unique(team, ["season", "week", "team"], "team form")
    team_feature_source = [
        column
        for column in team.columns
        if column not in TEAM_FORM_KEYS
    ]
    validate_form_column_names(team_feature_source, "team form")

    team["_join_team"] = team["team"].map(canonical_franchise)
    common.ensure_unique(
        team,
        ["season", "week", "_join_team"],
        "team form canonical join grain",
    )

    team_renamed = {
        column: f"team_{column}"
        for column in team_feature_source
    }
    team_columns.extend(team_renamed.values())

    out = out.merge(
        team[
            ["season", "week", "_join_team"] + team_feature_source
        ].rename(columns=team_renamed),
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Keep a minimal copy for opponent-offense matchup expectations and audit.
    team_internal = team[
        [
            "season",
            "week",
            "_join_team",
            "offensive_plays_roll3_mean",
            "dropbacks_roll3_mean",
        ]
    ].rename(
        columns={
            "_join_team": "_join_opponent",
            "offensive_plays_roll3_mean": "_opp_offensive_plays_roll3",
            "dropbacks_roll3_mean": "_opp_offense_dropbacks_roll3",
        }
    )

    team_audit = build_team_source_audit(team)

    out = out.merge(
        team_internal,
        on=["season", "week", "_join_opponent"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    out = out.merge(
        team_audit,
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    del team, team_internal, team_audit
    gc.collect()

    # Opponent form: opponent defensive context for offensive players.
    opponent = common.read_parquet_required(
        paths["opponent"],
        ["season", "week", "team"],
    )
    common.ensure_unique(
        opponent,
        ["season", "week", "team"],
        "opponent form",
    )
    opponent_feature_source = [
        column
        for column in opponent.columns
        if column not in TEAM_FORM_KEYS
    ]
    validate_form_column_names(opponent_feature_source, "opponent form")

    opponent["_join_defense"] = opponent["team"].map(
        canonical_franchise
    )
    common.ensure_unique(
        opponent,
        ["season", "week", "_join_defense"],
        "opponent form canonical join grain",
    )

    opponent_renamed = {
        column: f"opponent_{column}"
        for column in opponent_feature_source
    }
    opponent_columns.extend(opponent_renamed.values())

    opponent_for_join = opponent[
        ["season", "week", "_join_defense"] + opponent_feature_source
    ].rename(columns={"_join_defense": "_join_opponent", **opponent_renamed})

    out = out.merge(
        opponent_for_join,
        on=["season", "week", "_join_opponent"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Minimal player's-team defense copy for opponent-volume expectations.
    team_defense_internal = opponent[
        [
            "season",
            "week",
            "_join_defense",
            "defensive_plays_roll3_mean",
            "opponent_dropbacks_roll3_mean",
        ]
    ].rename(
        columns={
            "_join_defense": "_join_team",
            "defensive_plays_roll3_mean": "_team_defensive_plays_roll3",
            "opponent_dropbacks_roll3_mean": "_team_defense_dropbacks_roll3",
        }
    )

    out = out.merge(
        team_defense_internal,
        on=["season", "week", "_join_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    del opponent, opponent_for_join, team_defense_internal
    gc.collect()

    # Environment: one row per game; expose player/team-relative rest/travel.
    environment = common.read_parquet_required(
        paths["environment"],
        ENVIRONMENT_REQUIRED_COLUMNS,
    )
    common.ensure_unique(
        environment,
        ["season", "week", "game_id"],
        "environment",
    )

    env = environment.copy()
    env["_home_join"] = env["home_team"].map(canonical_franchise)
    env["_away_join"] = env["away_team"].map(canonical_franchise)

    direct_rename = {
        column: f"environment_{column}"
        for column in ENVIRONMENT_DIRECT_COLUMNS
    }
    direct_names = list(direct_rename.values())

    env = env.rename(columns=direct_rename)

    out = out.merge(
        env[
            [
                "season",
                "week",
                "game_id",
                "_home_join",
                "_away_join",
                "home_rest_days",
                "away_rest_days",
                "miles_traveled_away",
                "time_zones_crossed_away",
                "east_to_west_flag",
                "west_to_east_flag",
            ]
            + direct_names
        ],
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    home_side = out["_join_team"].eq(out["_home_join"])
    away_side = out["_join_team"].eq(out["_away_join"])
    if (~(home_side | away_side)).any():
        sample = out.loc[
            ~(home_side | away_side),
            GRAIN + ["team", "opponent"],
        ].head(10).to_dict("records")
        raise ValueError(
            f"Environment team/game mismatch; sample={sample}"
        )

    out["environment_team_rest_days"] = np.where(
        home_side,
        out["home_rest_days"],
        out["away_rest_days"],
    )
    out["environment_opponent_rest_days"] = np.where(
        home_side,
        out["away_rest_days"],
        out["home_rest_days"],
    )
    out["environment_team_miles_traveled"] = np.where(
        away_side,
        out["miles_traveled_away"],
        0.0,
    )
    out["environment_opponent_miles_traveled"] = np.where(
        home_side,
        out["miles_traveled_away"],
        0.0,
    )
    out["environment_team_time_zones_crossed"] = np.where(
        away_side,
        out["time_zones_crossed_away"],
        0.0,
    )
    out["environment_opponent_time_zones_crossed"] = np.where(
        home_side,
        out["time_zones_crossed_away"],
        0.0,
    )
    out["environment_team_east_to_west_flag"] = np.where(
        away_side,
        out["east_to_west_flag"],
        0,
    )
    out["environment_opponent_east_to_west_flag"] = np.where(
        home_side,
        out["east_to_west_flag"],
        0,
    )
    out["environment_team_west_to_east_flag"] = np.where(
        away_side,
        out["west_to_east_flag"],
        0,
    )
    out["environment_opponent_west_to_east_flag"] = np.where(
        home_side,
        out["west_to_east_flag"],
        0,
    )

    environment_columns.extend(
        direct_names
        + [
            "environment_team_rest_days",
            "environment_opponent_rest_days",
            "environment_team_miles_traveled",
            "environment_opponent_miles_traveled",
            "environment_team_time_zones_crossed",
            "environment_opponent_time_zones_crossed",
            "environment_team_east_to_west_flag",
            "environment_opponent_east_to_west_flag",
            "environment_team_west_to_east_flag",
            "environment_opponent_west_to_east_flag",
        ]
    )

    out = out.drop(
        columns=[
            "_home_join",
            "_away_join",
            "home_rest_days",
            "away_rest_days",
            "miles_traveled_away",
            "time_zones_crossed_away",
            "east_to_west_flag",
            "west_to_east_flag",
        ]
    )
    del environment, env
    gc.collect()

    # Defensive-specific full-universe features.
    defensive = common.read_parquet_required(paths["defensive"], GRAIN)
    validate_exact_full_grain(universe_keys, defensive, "defensive features")

    defensive_source = [
        column
        for column in defensive.columns
        if column not in set(GRAIN + ["position"])
    ]

    defensive_rename: dict[str, str] = {}
    for column in defensive_source:
        if column in DEFENSIVE_ROLE_MAP:
            output_name = DEFENSIVE_ROLE_MAP[column]
            role_columns.append(output_name)
        else:
            output_name = f"player_defensive_{column}"
            player_columns.append(output_name)
        defensive_rename[column] = output_name

    out = out.merge(
        defensive[GRAIN + defensive_source].rename(
            columns=defensive_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del defensive
    gc.collect()

    # Sparse kicking-specific features; non-kickers remain null.
    kicking = common.read_parquet_required(paths["kicking"], GRAIN)
    validate_sparse_grain_subset(universe_keys, kicking, "kicking features")

    kicking_source = [
        column
        for column in kicking.columns
        if column not in set(GRAIN + ["team"])
        and column not in KICKING_REDUNDANT_ENVIRONMENT
    ]

    kicking_rename: dict[str, str] = {}
    for column in kicking_source:
        if column in KICKING_ROLE_MAP:
            output_name = KICKING_ROLE_MAP[column]
            role_columns.append(output_name)
        else:
            output_name = f"player_kicking_{column}"
            player_columns.append(output_name)
        kicking_rename[column] = output_name

    out = out.merge(
        kicking[GRAIN + kicking_source].rename(
            columns=kicking_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del kicking
    gc.collect()

    # Strictly lag position-allowed values before joining.
    position_allowed = common.read_parquet_required(
        paths["position_allowed"],
        POSITION_ALLOWED_KEYS,
    )
    position_lag, position_matchup_columns = build_position_allowed_lag(
        position_allowed
    )
    matchup_columns.extend(REQUIRED_MATCHUP_COLUMNS)
    matchup_columns.extend(position_matchup_columns)

    out = out.merge(
        position_lag,
        left_on=[
            "season",
            "week",
            "_join_opponent",
            "_join_position_group",
        ],
        right_on=[
            "season",
            "week",
            "_join_defense",
            "_join_position_group",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if "_join_defense" in out.columns:
        out = out.drop(columns=["_join_defense"])
    del position_allowed, position_lag
    gc.collect()

    # Matchup expectations. All inputs are already strictly lagged or shifted.
    out["matchup_expected_team_plays"] = safe_mean_pair(
        out["team_offensive_plays_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )
    out["matchup_expected_team_dropbacks"] = safe_mean_pair(
        out["team_dropbacks_roll3_mean"],
        out["opponent_opponent_dropbacks_roll3_mean"],
    )
    out["matchup_expected_team_rush_attempts"] = safe_mean_pair(
        out["team_rush_attempts_roll3_mean"],
        out["opponent_opponent_rush_attempts_roll3_mean"],
    )
    out["matchup_expected_opponent_plays"] = safe_mean_pair(
        out["_opp_offensive_plays_roll3"],
        out["_team_defensive_plays_roll3"],
    )
    out["matchup_expected_opponent_dropbacks"] = safe_mean_pair(
        out["_opp_offense_dropbacks_roll3"],
        out["_team_defense_dropbacks_roll3"],
    )

    out["matchup_player_target_share_x_opp_targets"] = safe_product(
        out["player_target_share_roll3_mean"],
        out["matchup_position_allowed_targets_allowed_lag1"],
    )
    out["matchup_player_carry_share_x_opp_rushes"] = safe_product(
        out["player_carry_share_roll3_mean"],
        out["matchup_position_allowed_carries_allowed_lag1"],
    )
    out["matchup_player_tackle_rate_x_opp_plays"] = safe_product(
        out["player_tackle_rate_per_def_play_roll3_mean"],
        out["matchup_expected_opponent_plays"],
    )
    out["matchup_player_sack_rate_x_opp_plays"] = safe_product(
        out["player_sack_rate_per_def_play_roll5_mean"],
        out["matchup_expected_opponent_plays"],
    )
    out["matchup_off_epa_vs_def_epa"] = (
        pd.to_numeric(
            out["team_off_epa_per_play_roll3_mean"],
            errors="coerce",
        )
        - pd.to_numeric(
            out["opponent_def_epa_per_play_roll3_mean"],
            errors="coerce",
        )
    )

    opponent_pass_rate = safe_divide(
        out["opponent_opponent_pass_attempts_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )
    opponent_rush_rate = safe_divide(
        out["opponent_opponent_rush_attempts_roll3_mean"],
        out["opponent_defensive_plays_roll3_mean"],
    )

    out["matchup_pass_rate_vs_opponent"] = (
        pd.to_numeric(
            out["team_pass_rate_roll3_mean"],
            errors="coerce",
        )
        - opponent_pass_rate
    )
    out["matchup_rush_rate_vs_opponent"] = (
        pd.to_numeric(
            out["team_rush_rate_roll3_mean"],
            errors="coerce",
        )
        - opponent_rush_rate
    )

    out = out.drop(
        columns=[
            "_opp_offensive_plays_roll3",
            "_opp_offense_dropbacks_roll3",
            "_team_defensive_plays_roll3",
            "_team_defense_dropbacks_roll3",
        ]
    )

    # Targets: canonical output target names derive from configured target keys.
    targets = common.read_parquet_required(
        paths["targets"],
        GRAIN + REQUIRED_TARGET_ORDER,
    )
    validate_exact_full_grain(universe_keys, targets, "historical targets")

    target_rename = {
        name: f"target_{name}"
        for name in REQUIRED_TARGET_ORDER
    }
    out = out.merge(
        targets[GRAIN + REQUIRED_TARGET_ORDER].rename(
            columns=target_rename
        ),
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del targets
    gc.collect()

    # Audit metadata. played_game_flag is used only to identify prior realized
    # player-game provenance and is never exposed as a feature.
    player_audit = build_player_audit(universe)
    out = out.merge(
        player_audit,
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    del player_audit, universe, universe_keys
    gc.collect()

    out["audit_feature_asof"] = out["kickoff_timestamp"]
    out["audit_depth_snapshot_at"] = pd.NaT
    out["audit_injury_snapshot_at"] = pd.NaT
    out["audit_market_feature_count"] = np.int8(0)
    out["audit_row_valid"] = np.int8(1)

    # No join-helper columns may survive.
    helper_columns = [
        column
        for column in out.columns
        if column.startswith("_join_")
    ]
    if helper_columns:
        out = out.drop(columns=helper_columns)

    # Deterministic family order.
    family_lists = [
        role_columns,
        player_columns,
        team_columns,
        opponent_columns,
        matchup_columns,
        environment_columns,
        history_columns,
        TARGET_COLUMNS,
        AUDIT_COLUMNS,
    ]

    for family in family_lists:
        duplicates = [
            value
            for value in family
            if family.count(value) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate output columns inside family: {sorted(set(duplicates))}"
            )

    ordered_columns = (
        LEADING_COLUMNS
        + role_columns
        + player_columns
        + team_columns
        + opponent_columns
        + matchup_columns
        + environment_columns
        + history_columns
        + TARGET_COLUMNS
        + AUDIT_COLUMNS
    )

    if len(ordered_columns) != len(set(ordered_columns)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for column in ordered_columns:
            if column in seen:
                duplicates.append(column)
            seen.add(column)
        raise ValueError(
            f"Duplicate final schema columns: {sorted(set(duplicates))}"
        )

    missing_output = [
        column
        for column in ordered_columns
        if column not in out.columns
    ]
    if missing_output:
        raise ValueError(
            f"Missing assembled output columns: {missing_output[:30]}"
        )

    extra_output = [
        column
        for column in out.columns
        if column not in ordered_columns
    ]
    if extra_output:
        raise ValueError(
            f"Unexpected assembled columns: {extra_output[:30]}"
        )

    out = out[ordered_columns]

    common.ensure_unique(out, GRAIN, "historical feature table")
    if len(out) != universe_row_count:
        raise ValueError(
            f"Historical feature row count changed: "
            f"{universe_row_count:,} -> {len(out):,}"
        )

    if list(out.columns[: len(LEADING_COLUMNS)]) != LEADING_COLUMNS:
        raise ValueError("Leading header order mismatch.")

    if [column for column in out.columns if column.startswith("target_")] != TARGET_COLUMNS:
        raise ValueError("Target header order mismatch.")

    if [column for column in out.columns if column.startswith("audit_")] != AUDIT_COLUMNS:
        raise ValueError("Audit header order mismatch.")

    # Explicit model-feature candidates: identity/time IDs, target_* and
    # audit_* are not features. Team/opponent/position context is allowed.
    candidate_features = (
        ["home_flag", "team", "opponent", "position", "position_group"]
        + role_columns
        + player_columns
        + team_columns
        + opponent_columns
        + matchup_columns
        + environment_columns
        + history_columns
    )

    if any(column.startswith("target_") for column in candidate_features):
        raise ValueError("Target leakage into candidate features.")

    if any(column.startswith("audit_") for column in candidate_features):
        raise ValueError("Audit columns entered candidate features.")

    if "played_game_flag" in candidate_features or "played_game_flag" in out.columns:
        raise ValueError("played_game_flag must not enter assembled schema.")

    common.reject_forbidden_feature_columns(candidate_features, config)
    validate_no_same_game_usage_features(candidate_features)
    validate_no_final_score_features(candidate_features)

    numeric_features, categorical_features = classify_feature_columns(
        out,
        candidate_features,
    )

    if set(numeric_features) & set(categorical_features):
        raise ValueError("Feature manifest numeric/categorical overlap.")

    if set(numeric_features + categorical_features) != set(candidate_features):
        raise ValueError("Feature manifest does not cover candidate features exactly.")

    numeric_block = out.select_dtypes(include=[np.number])
    if np.isinf(numeric_block.to_numpy(dtype="float64", copy=False)).any():
        raise ValueError("Historical feature table contains infinity.")

    schema_hash = stable_schema_hash(out)

    manifest = {
        "schema_version": 1,
        "canonical_grain": GRAIN,
        "leading_columns": LEADING_COLUMNS,
        "family_order": [
            "role",
            "player",
            "team",
            "opponent",
            "matchup",
            "environment",
            "history",
            "target",
            "audit",
        ],
        "column_families": {
            "role": role_columns,
            "player": player_columns,
            "team": team_columns,
            "opponent": opponent_columns,
            "matchup": matchup_columns,
            "environment": environment_columns,
            "history": history_columns,
            "target": TARGET_COLUMNS,
            "audit": AUDIT_COLUMNS,
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": numeric_features + categorical_features,
        "excluded_from_features": {
            "identity_and_time": [
                "season",
                "season_type",
                "week",
                "game_id",
                "gameday",
                "kickoff_timestamp",
                "player_id",
                "player_name",
            ],
            "targets": TARGET_COLUMNS,
            "audit": AUDIT_COLUMNS,
            "outcome_metadata": ["played_game_flag"],
        },
        "target_columns": TARGET_COLUMNS,
        "required_matchup_columns": REQUIRED_MATCHUP_COLUMNS,
        "matchup_formulas": {
            "matchup_expected_team_plays": "mean(team offensive_plays roll3, opponent defensive_plays roll3)",
            "matchup_expected_team_dropbacks": "mean(team dropbacks roll3, opponent opponent_dropbacks roll3)",
            "matchup_expected_team_rush_attempts": "mean(team rush_attempts roll3, opponent opponent_rush_attempts roll3)",
            "matchup_expected_opponent_plays": "mean(opponent offensive_plays roll3, team defensive_plays roll3)",
            "matchup_expected_opponent_dropbacks": "mean(opponent dropbacks roll3, team opponent_dropbacks roll3)",
            "matchup_player_target_share_x_opp_targets": "player target_share roll3 * opponent-position targets_allowed lag1",
            "matchup_player_carry_share_x_opp_rushes": "player carry_share roll3 * opponent-position carries_allowed lag1",
            "matchup_player_tackle_rate_x_opp_plays": "player tackle_rate roll3 * expected opponent plays",
            "matchup_player_sack_rate_x_opp_plays": "player sack_rate roll5 * expected opponent plays",
            "matchup_off_epa_vs_def_epa": "team off_epa_per_play roll3 - opponent def_epa_per_play roll3",
            "matchup_pass_rate_vs_opponent": "team pass_rate roll3 - opponent prior pass attempts / opponent prior defensive plays",
            "matchup_rush_rate_vs_opponent": "team rush_rate roll3 - opponent prior rush attempts / opponent prior defensive plays",
        },
        "position_allowed_policy": (
            "Every realized position_allowed_week metric is shifted one "
            "observed defense/position game before target-row use."
        ),
        "audit_policy": {
            "audit_feature_asof": "target kickoff timestamp; all feature sources must be strictly earlier",
            "audit_max_player_source_game": "latest strictly prior played modeled-universe game for player; 2010-2011 prehistory may not have a modeled game_id",
            "audit_max_team_source_week": "latest prior canonical-franchise team-form source week",
            "audit_depth_snapshot_at": "null because upstream role-history output does not retain the source snapshot timestamp; upstream builder validated pregame depth",
            "audit_injury_snapshot_at": "null because upstream role-history output does not retain the source snapshot timestamp; upstream builder validated pregame injury status",
        },
        "source_paths": {
            key: value
            for key, value in paths.items()
            if key != "output"
        },
        "output_path": paths["output"],
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "feature_count": int(len(candidate_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "schema_hash": schema_hash,
        "market_features_used": False,
        "target_columns_in_feature_manifest": False,
    }

    common.write_parquet_atomic(out, paths["output"])
    write_json_atomic(manifest, manifest_path)

    payload = {
        "status": "passed",
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "features": int(len(candidate_features)),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "schema_hash": schema_hash,
        "market_feature_count": 0,
        "target_columns_in_manifest": False,
        "position_allowed_lagged": True,
        "output": paths["output"],
        "manifest": manifest_path,
    }
    common.log_run("build_historical_features.py", payload)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
