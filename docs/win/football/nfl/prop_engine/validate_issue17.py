#!/usr/bin/env python3
"""
Independent acceptance validator for NFL Prop Engine Issue 17.

Validates the assembled historical player-game feature table and manifest
against the upstream source tables without importing build_historical_features.py.

READS:
    prop_engine.yaml
    player_game_universe.parquet
    player_game_targets.parquet
    player_role_history.parquet
    player_form.parquet
    team_form.parquet
    opponent_form.parquet
    environment.parquet
    defensive_features.parquet
    kicking_features.parquet
    position_allowed_week.parquet
    player_game_features.parquet
    feature_manifest.json

WRITES:
    Nothing.

ACCEPTANCE:
    - exact canonical grain and leading headers
    - deterministic family/schema order
    - source reconciliation
    - exact target reconciliation
    - strict lag of position-allowed data
    - independent matchup formula reconstruction
    - environment team-relative reconstruction
    - explicit manifest coverage/types
    - no target/audit/outcome leakage into manifest
    - no forbidden market/final-score/same-game snap-participation features
    - audit provenance reconstruction
    - no infinities
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


GRAIN = ["season", "week", "game_id", "player_id"]

LEADING = [
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

TARGET_NAMES = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
]
TARGETS = [f"target_{x}" for x in TARGET_NAMES]

AUDIT = [
    "audit_feature_asof",
    "audit_max_player_source_game",
    "audit_max_team_source_week",
    "audit_depth_snapshot_at",
    "audit_injury_snapshot_at",
    "audit_market_feature_count",
    "audit_row_valid",
]

REQUIRED_MATCHUPS = [
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

FAMILY_ORDER = [
    "role",
    "player",
    "team",
    "opponent",
    "matchup",
    "environment",
    "history",
    "target",
    "audit",
]

FRANCHISE_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

POSITION_KEYS = [
    "season",
    "week",
    "defense_team",
    "offense_position_group",
]

ENV_DIRECT = [
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

SAFE_FORM_SUFFIXES = (
    "_lag1",
    "_roll3_mean",
    "_roll5_mean",
    "_roll8_mean",
    "_ewm3",
    "_ewm5",
    "_season_to_date",
)

FINAL_SCORE_NAMES = {
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


def fail(message: str) -> None:
    raise AssertionError(message)


def canon_team(value: Any) -> str:
    team = common.normalize_team(value)
    return FRANCHISE_ALIASES.get(team, team)


def pos_group(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().upper()


def full_path(relative: str) -> Path:
    return (common.repo_root() / relative).resolve()


def read_json(relative: str) -> dict:
    path = full_path(relative)
    if not path.is_file():
        fail(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_grain(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GRAIN, kind="mergesort").reset_index(drop=True)


def assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    a = sort_grain(left[GRAIN].copy())
    b = sort_grain(right[GRAIN].copy())
    if len(a) != len(b) or not a.equals(b):
        fail(f"{label}: canonical grain mismatch")


def assert_numeric_equal(
    actual: pd.Series,
    expected: pd.Series,
    label: str,
    *,
    atol: float = 1e-10,
) -> None:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype="float64")

    if len(a) != len(e):
        fail(f"{label}: length mismatch")

    equal = np.isclose(a, e, rtol=1e-10, atol=atol, equal_nan=True)
    if not bool(np.all(equal)):
        idx = np.flatnonzero(~equal)[:10]
        sample = [
            {
                "row": int(i),
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in idx
        ]
        fail(f"{label}: numeric mismatch; sample={sample}")


def assert_text_equal(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a = actual.astype("string").fillna("<NA>")
    e = expected.astype("string").fillna("<NA>")
    bad = a.ne(e)
    if bad.any():
        idx = np.flatnonzero(bad.to_numpy())[:10]
        sample = [
            {
                "row": int(i),
                "actual": str(a.iloc[i]),
                "expected": str(e.iloc[i]),
            }
            for i in idx
        ]
        fail(f"{label}: text mismatch; sample={sample}")


def assert_datetime_equal(actual: pd.Series, expected: pd.Series, label: str) -> None:
    a = pd.to_datetime(actual, errors="coerce", utc=True)
    e = pd.to_datetime(expected, errors="coerce", utc=True)
    bad = ~(a.eq(e) | (a.isna() & e.isna()))
    if bad.any():
        fail(f"{label}: datetime mismatch; count={int(bad.sum())}")


def safe_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce").astype("float64")
    bb = pd.to_numeric(b, errors="coerce").astype("float64")
    return pd.concat([aa, bb], axis=1).mean(axis=1, skipna=True)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce").astype("float64")
    bb = pd.to_numeric(b, errors="coerce").astype("float64")
    out = pd.Series(np.nan, index=aa.index, dtype="float64")
    valid = aa.notna() & bb.notna() & bb.ne(0.0)
    out.loc[valid] = aa.loc[valid] / bb.loc[valid]
    return out


def deterministic_sample(columns: list[str], n: int) -> list[str]:
    if len(columns) <= n:
        return columns
    positions = np.linspace(0, len(columns) - 1, num=n, dtype=int)
    return [columns[int(i)] for i in positions]


def parquet_schema(relative: str) -> tuple[list[str], pa.Schema, int]:
    pf = pq.ParquetFile(full_path(relative))
    schema = pf.schema_arrow
    return schema.names, schema, int(pf.metadata.num_rows)


def check_manifest_type_lists(
    manifest: dict,
    arrow_schema: pa.Schema,
) -> None:
    numeric = manifest.get("numeric_features")
    categorical = manifest.get("categorical_features")
    features = manifest.get("feature_columns")

    if not isinstance(numeric, list) or not isinstance(categorical, list):
        fail("Manifest numeric/categorical feature lists are not explicit lists")
    if not isinstance(features, list):
        fail("Manifest feature_columns is not an explicit list")

    if len(features) != len(set(features)):
        fail("Manifest feature_columns contains duplicates")
    if set(numeric) & set(categorical):
        fail("Manifest numeric/categorical lists overlap")
    if set(numeric + categorical) != set(features):
        fail("Manifest numeric/categorical lists do not cover feature_columns exactly")

    schema_by_name = {field.name: field.type for field in arrow_schema}

    for column in numeric:
        if column not in schema_by_name:
            fail(f"Manifest numeric feature missing from parquet: {column}")
        typ = schema_by_name[column]
        if not (
            pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_boolean(typ)
            or pa.types.is_decimal(typ)
        ):
            fail(f"Manifest numeric feature has nonnumeric parquet type: {column}={typ}")

    for column in categorical:
        if column not in schema_by_name:
            fail(f"Manifest categorical feature missing from parquet: {column}")
        typ = schema_by_name[column]
        if not (
            pa.types.is_string(typ)
            or pa.types.is_large_string(typ)
            or pa.types.is_dictionary(typ)
        ):
            fail(f"Manifest categorical feature has noncategorical parquet type: {column}={typ}")


def compare_source_columns(
    source_rel: str,
    output_rel: str,
    source_columns: list[str],
    rename: dict[str, str],
    label: str,
) -> None:
    needed_source = GRAIN + source_columns
    needed_output = GRAIN + [rename[c] for c in source_columns]

    src = pd.read_parquet(full_path(source_rel), columns=needed_source)
    out = pd.read_parquet(full_path(output_rel), columns=needed_output)

    assert_same_keys(src, out, label)

    src = sort_grain(src)
    out = sort_grain(out)

    for source_col in source_columns:
        output_col = rename[source_col]
        if pd.api.types.is_numeric_dtype(src[source_col].dtype):
            assert_numeric_equal(out[output_col], src[source_col], f"{label}:{source_col}")
        elif pd.api.types.is_datetime64_any_dtype(src[source_col].dtype):
            assert_datetime_equal(out[output_col], src[source_col], f"{label}:{source_col}")
        else:
            assert_text_equal(out[output_col], src[source_col], f"{label}:{source_col}")


def main() -> int:
    config = common.load_config()
    paths = config["paths"]

    required_path_keys = [
        "historical_universe",
        "historical_targets",
        "role_history",
        "player_form",
        "team_form",
        "opponent_form",
        "environment_history",
        "defensive_features",
        "kicking_features",
        "position_allowed",
        "historical_features",
    ]
    missing = [k for k in required_path_keys if not paths.get(k)]
    if missing:
        fail(f"Config missing Issue 17 path keys: {missing}")

    output_rel = paths["historical_features"]
    manifest_rel = (
        "docs/win/football/nfl/prop_engine/data/historical/"
        "features/feature_manifest.json"
    )

    output_names, arrow_schema, output_rows = parquet_schema(output_rel)
    universe_names, _, universe_rows = parquet_schema(paths["historical_universe"])

    print("CHECK 01: row count and canonical grain")
    if output_rows != universe_rows:
        fail(f"Row count mismatch: output={output_rows:,}, universe={universe_rows:,}")

    output_keys = pd.read_parquet(full_path(output_rel), columns=GRAIN)
    if output_keys["player_id"].astype("string").str.strip().eq("").any():
        fail("Blank player_id in assembled table")
    common.ensure_unique(output_keys, GRAIN, "Issue 17 output grain")

    universe_keys = pd.read_parquet(full_path(paths["historical_universe"]), columns=GRAIN)
    assert_same_keys(output_keys, universe_keys, "output vs universe")
    del output_keys, universe_keys

    print("CHECK 02: exact leading headers and deterministic family order")
    if output_names[: len(LEADING)] != LEADING:
        fail(f"Leading headers mismatch: {output_names[:len(LEADING)]}")

    manifest = read_json(manifest_rel)
    if manifest.get("family_order") != FAMILY_ORDER:
        fail(f"Manifest family_order mismatch: {manifest.get('family_order')}")

    families = manifest.get("column_families")
    if not isinstance(families, dict):
        fail("Manifest column_families missing")

    expected_schema = list(LEADING)
    for family in FAMILY_ORDER:
        values = families.get(family)
        if not isinstance(values, list):
            fail(f"Manifest family {family} is not an explicit list")
        expected_schema.extend(values)

    if expected_schema != output_names:
        fail("Parquet schema does not exactly equal leading headers + manifest family order")

    if families["target"] != TARGETS:
        fail("Target family/order mismatch")
    if families["audit"] != AUDIT:
        fail("Audit family/order mismatch")
    if any(not c.startswith("role_") for c in families["role"]):
        fail("Non-role_* column in role family")
    if any(not c.startswith("player_") for c in families["player"]):
        fail("Non-player_* column in player family")
    if any(not c.startswith("team_") for c in families["team"]):
        fail("Non-team_* column in team family")
    if any(not c.startswith("opponent_") for c in families["opponent"]):
        fail("Non-opponent_* column in opponent family")
    if any(not c.startswith("matchup_") for c in families["matchup"]):
        fail("Non-matchup_* column in matchup family")
    if any(not c.startswith("environment_") for c in families["environment"]):
        fail("Non-environment_* column in environment family")
    if any(not c.startswith("history_") for c in families["history"]):
        fail("Non-history_* column in history family")

    print("CHECK 03: manifest coverage, explicit types, and leakage exclusions")
    check_manifest_type_lists(manifest, arrow_schema)

    features = manifest["feature_columns"]
    numeric = manifest["numeric_features"]
    categorical = manifest["categorical_features"]

    if any(c.startswith("target_") for c in features):
        fail("target_* appears in feature manifest")
    if any(c.startswith("audit_") for c in features):
        fail("audit_* appears in feature manifest")
    if "played_game_flag" in features or "played_game_flag" in output_names:
        fail("played_game_flag entered assembled schema/manifest")

    common.reject_forbidden_feature_columns(features, config)

    bad_score = [
        c for c in features
        if c.casefold() in FINAL_SCORE_NAMES
        or "final_score" in c.casefold()
    ]
    if bad_score:
        fail(f"Final-score/result feature(s) found: {bad_score[:20]}")

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
    unsafe_usage = [
        c for c in features
        if any(t in c.casefold() for t in usage_tokens)
        and not any(t in c.casefold() for t in safe_tokens)
    ]
    if unsafe_usage:
        fail(f"Potential same-game snap/participation feature(s): {unsafe_usage[:20]}")

    if set(numeric) | set(categorical) != set(features):
        fail("Manifest feature classification coverage mismatch")

    print("CHECK 04: leading-column source reconciliation")
    universe = pd.read_parquet(full_path(paths["historical_universe"]), columns=LEADING)
    assembled = pd.read_parquet(full_path(output_rel), columns=LEADING)
    universe = sort_grain(universe)
    assembled = sort_grain(assembled)
    for col in LEADING:
        if pd.api.types.is_numeric_dtype(universe[col].dtype):
            assert_numeric_equal(assembled[col], universe[col], f"leading:{col}")
        elif pd.api.types.is_datetime64_any_dtype(universe[col].dtype):
            assert_datetime_equal(assembled[col], universe[col], f"leading:{col}")
        else:
            assert_text_equal(assembled[col], universe[col], f"leading:{col}")
    del assembled

    print("CHECK 05: exact nine-target reconciliation")
    src_targets = pd.read_parquet(
        full_path(paths["historical_targets"]),
        columns=GRAIN + TARGET_NAMES,
    )
    out_targets = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + TARGETS,
    )
    assert_same_keys(src_targets, out_targets, "targets")
    src_targets = sort_grain(src_targets)
    out_targets = sort_grain(out_targets)
    for name in TARGET_NAMES:
        assert_numeric_equal(
            out_targets[f"target_{name}"],
            src_targets[name],
            f"target:{name}",
        )
    del src_targets, out_targets

    print("CHECK 06: role-history reconciliation")
    role_source = pd.read_parquet(full_path(paths["role_history"]))
    role_cols = [
        c for c in role_source.columns
        if c not in set(GRAIN + ["team", "position"])
    ]
    expected_role_names = [f"role_{c}" for c in role_cols]
    # Defensive/kicking role additions are validated separately.
    if families["role"][: len(expected_role_names)] != expected_role_names:
        fail("Role-history family prefix/order does not match source")
    compare_source_columns(
        paths["role_history"],
        output_rel,
        role_cols,
        {c: f"role_{c}" for c in role_cols},
        "role history",
    )
    del role_source

    print("CHECK 07: player-form/history join reconciliation")
    player_meta_names, _, _ = parquet_schema(paths["player_form"])
    history_source = ["no_nfl_history_flag", "new_team_flag", "history_games"]
    player_source_cols = [
        c for c in player_meta_names
        if c not in set(GRAIN + ["team", "position", "position_group"] + history_source)
    ]
    sample_player = deterministic_sample(player_source_cols, 36)
    # Ensure critical matchup inputs are always checked.
    for critical in [
        "target_share_roll3_mean",
        "carry_share_roll3_mean",
        "tackle_rate_per_def_play_roll3_mean",
        "sack_rate_per_def_play_roll5_mean",
    ]:
        if critical not in sample_player:
            sample_player.append(critical)

    compare_source_columns(
        paths["player_form"],
        output_rel,
        sample_player,
        {c: f"player_{c}" for c in sample_player},
        "player form sampled join",
    )
    compare_source_columns(
        paths["player_form"],
        output_rel,
        history_source,
        {c: f"history_{c}" for c in history_source},
        "player history flags",
    )

    expected_player_form_names = [f"player_{c}" for c in player_source_cols]
    if families["player"][: len(expected_player_form_names)] != expected_player_form_names:
        fail("Player-form family prefix/order does not match source")

    print("CHECK 08: team/opponent form naming, lag-only policy, and sampled reconciliation")
    team_names, _, _ = parquet_schema(paths["team_form"])
    team_source_cols = [c for c in team_names if c not in {"season", "week", "team"}]
    opponent_names, _, _ = parquet_schema(paths["opponent_form"])
    opponent_source_cols = [c for c in opponent_names if c not in {"season", "week", "team"}]

    bad_team_form = [c for c in team_source_cols if not c.endswith(SAFE_FORM_SUFFIXES)]
    bad_opp_form = [c for c in opponent_source_cols if not c.endswith(SAFE_FORM_SUFFIXES)]
    if bad_team_form or bad_opp_form:
        fail(
            "Non-lagged team/opponent form source columns detected: "
            f"team={bad_team_form[:10]}, opponent={bad_opp_form[:10]}"
        )

    expected_team_names = [f"team_{c}" for c in team_source_cols]
    expected_opp_names = [f"opponent_{c}" for c in opponent_source_cols]
    if families["team"] != expected_team_names:
        fail("team_* family does not exactly match team_form source")
    if families["opponent"] != expected_opp_names:
        fail("opponent_* family does not exactly match opponent_form source")

    base_join = universe[GRAIN + ["team", "opponent"]].copy()
    base_join["_team"] = base_join["team"].map(canon_team)
    base_join["_opp"] = base_join["opponent"].map(canon_team)

    sample_team = deterministic_sample(team_source_cols, 20)
    sample_opp = deterministic_sample(opponent_source_cols, 20)
    for critical in [
        "offensive_plays_roll3_mean",
        "dropbacks_roll3_mean",
        "rush_attempts_roll3_mean",
        "pass_rate_roll3_mean",
        "rush_rate_roll3_mean",
        "off_epa_per_play_roll3_mean",
    ]:
        if critical not in sample_team:
            sample_team.append(critical)
    for critical in [
        "defensive_plays_roll3_mean",
        "opponent_dropbacks_roll3_mean",
        "opponent_pass_attempts_roll3_mean",
        "opponent_rush_attempts_roll3_mean",
        "def_epa_per_play_roll3_mean",
    ]:
        if critical not in sample_opp:
            sample_opp.append(critical)

    team_src = pd.read_parquet(
        full_path(paths["team_form"]),
        columns=["season", "week", "team"] + sample_team,
    )
    team_src["_team"] = team_src["team"].map(canon_team)
    team_src = team_src.drop(columns=["team"])
    expected_team = base_join[GRAIN + ["_team"]].merge(
        team_src,
        on=["season", "week", "_team"],
        how="left",
        validate="many_to_one",
    )
    actual_team = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + [f"team_{c}" for c in sample_team],
    )
    expected_team = sort_grain(expected_team)
    actual_team = sort_grain(actual_team)
    for c in sample_team:
        assert_numeric_equal(actual_team[f"team_{c}"], expected_team[c], f"team join:{c}")

    opp_src = pd.read_parquet(
        full_path(paths["opponent_form"]),
        columns=["season", "week", "team"] + sample_opp,
    )
    opp_src["_opp"] = opp_src["team"].map(canon_team)
    opp_src = opp_src.drop(columns=["team"])
    expected_opp = base_join[GRAIN + ["_opp"]].merge(
        opp_src,
        on=["season", "week", "_opp"],
        how="left",
        validate="many_to_one",
    )
    actual_opp = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + [f"opponent_{c}" for c in sample_opp],
    )
    expected_opp = sort_grain(expected_opp)
    actual_opp = sort_grain(actual_opp)
    for c in sample_opp:
        assert_numeric_equal(actual_opp[f"opponent_{c}"], expected_opp[c], f"opponent join:{c}")

    print("CHECK 09: environment reconstruction")
    env = pd.read_parquet(full_path(paths["environment_history"]))
    env["_home"] = env["home_team"].map(canon_team)
    env["_away"] = env["away_team"].map(canon_team)

    env_join_cols = [
        "season", "week", "game_id", "_home", "_away",
        "home_rest_days", "away_rest_days",
        "miles_traveled_away", "time_zones_crossed_away",
        "east_to_west_flag", "west_to_east_flag",
    ] + ENV_DIRECT

    exp_env = universe[GRAIN + ["team"]].copy()
    exp_env["_team"] = exp_env["team"].map(canon_team)
    exp_env = exp_env.merge(
        env[env_join_cols],
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )
    home_side = exp_env["_team"].eq(exp_env["_home"])
    away_side = exp_env["_team"].eq(exp_env["_away"])
    if (~(home_side | away_side)).any():
        fail("Environment reconstruction found player team outside game")

    for c in ENV_DIRECT:
        exp_env[f"environment_{c}"] = exp_env[c]

    exp_env["environment_team_rest_days"] = np.where(
        home_side, exp_env["home_rest_days"], exp_env["away_rest_days"]
    )
    exp_env["environment_opponent_rest_days"] = np.where(
        home_side, exp_env["away_rest_days"], exp_env["home_rest_days"]
    )
    exp_env["environment_team_miles_traveled"] = np.where(
        away_side, exp_env["miles_traveled_away"], 0.0
    )
    exp_env["environment_opponent_miles_traveled"] = np.where(
        home_side, exp_env["miles_traveled_away"], 0.0
    )
    exp_env["environment_team_time_zones_crossed"] = np.where(
        away_side, exp_env["time_zones_crossed_away"], 0.0
    )
    exp_env["environment_opponent_time_zones_crossed"] = np.where(
        home_side, exp_env["time_zones_crossed_away"], 0.0
    )
    exp_env["environment_team_east_to_west_flag"] = np.where(
        away_side, exp_env["east_to_west_flag"], 0
    )
    exp_env["environment_opponent_east_to_west_flag"] = np.where(
        home_side, exp_env["east_to_west_flag"], 0
    )
    exp_env["environment_team_west_to_east_flag"] = np.where(
        away_side, exp_env["west_to_east_flag"], 0
    )
    exp_env["environment_opponent_west_to_east_flag"] = np.where(
        home_side, exp_env["west_to_east_flag"], 0
    )

    env_output_cols = list(families["environment"])
    actual_env = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + env_output_cols,
    )
    exp_env = sort_grain(exp_env)
    actual_env = sort_grain(actual_env)

    for c in env_output_cols:
        if c not in exp_env.columns:
            fail(f"Unexpected environment output column: {c}")
        if pd.api.types.is_numeric_dtype(exp_env[c].dtype):
            assert_numeric_equal(actual_env[c], exp_env[c], f"environment:{c}")
        else:
            assert_text_equal(actual_env[c], exp_env[c], f"environment:{c}")

    print("CHECK 10: defensive and sparse kicking reconciliation")
    defensive_names, _, _ = parquet_schema(paths["defensive_features"])
    defensive_source_cols = [
        c for c in defensive_names
        if c not in set(GRAIN + ["position"])
    ]
    defensive_rename = {}
    for c in defensive_source_cols:
        if c == "starter_flag":
            defensive_rename[c] = "role_defensive_starter_flag"
        elif c == "front7_flag":
            defensive_rename[c] = "role_front7_flag"
        elif c == "secondary_flag":
            defensive_rename[c] = "role_secondary_flag"
        else:
            defensive_rename[c] = f"player_defensive_{c}"

    compare_source_columns(
        paths["defensive_features"],
        output_rel,
        defensive_source_cols,
        defensive_rename,
        "defensive features",
    )

    kicking_names, _, _ = parquet_schema(paths["kicking_features"])
    kicking_source_cols = [
        c for c in kicking_names
        if c not in set(GRAIN + ["team", "temperature", "wind", "roof", "surface"])
    ]
    kicking_rename = {
        c: (
            "role_primary_kicker_flag"
            if c == "primary_kicker_flag"
            else f"player_kicking_{c}"
        )
        for c in kicking_source_cols
    }

    # Sparse source: compare only kicker rows exactly.
    kick_src = pd.read_parquet(
        full_path(paths["kicking_features"]),
        columns=GRAIN + kicking_source_cols,
    )
    kick_out = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + list(kicking_rename.values()),
    )
    kick_merged = kick_src.merge(
        kick_out,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if len(kick_merged) != len(kick_src):
        fail("Kicking sparse join row count changed")
    for c in kicking_source_cols:
        a = kick_merged[kicking_rename[c]]
        e = kick_merged[c]
        if pd.api.types.is_numeric_dtype(e.dtype):
            assert_numeric_equal(a, e, f"kicking:{c}")
        else:
            assert_text_equal(a, e, f"kicking:{c}")

    print("CHECK 11: strictly prior position-allowed reconstruction")
    pos = pd.read_parquet(full_path(paths["position_allowed"]))
    common.ensure_unique(pos, POSITION_KEYS, "validator position allowed")
    pos["_def"] = pos["defense_team"].map(canon_team)
    pos["_pg"] = pos["offense_position_group"].map(pos_group)

    value_cols = [
        c for c in pos.columns
        if c not in set(POSITION_KEYS + ["_def", "_pg"])
    ]
    for c in value_cols:
        pos[c] = pd.to_numeric(pos[c], errors="coerce")

    pos = pos.sort_values(
        ["_def", "_pg", "season", "week"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = pos.groupby(["_def", "_pg"], sort=False, dropna=False)
    lag = pos[["season", "week", "_def", "_pg"]].copy()
    lag_cols = []
    for c in value_cols:
        outc = f"matchup_position_allowed_{c}_lag1"
        lag[outc] = grouped[c].shift(1)
        lag_cols.append(outc)

    expected_pos = universe[GRAIN + ["opponent", "position_group"]].copy()
    expected_pos["_def"] = expected_pos["opponent"].map(canon_team)
    expected_pos["_pg"] = expected_pos["position_group"].map(pos_group)
    expected_pos = expected_pos.merge(
        lag,
        on=["season", "week", "_def", "_pg"],
        how="left",
        validate="many_to_one",
    )
    actual_pos = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + lag_cols,
    )
    expected_pos = sort_grain(expected_pos)
    actual_pos = sort_grain(actual_pos)
    for c in lag_cols:
        assert_numeric_equal(actual_pos[c], expected_pos[c], f"position allowed lag:{c}")

    # Explicitly prove target-week realized shrunk_rate is not the value used.
    current = universe[GRAIN + ["opponent", "position_group"]].copy()
    current["_def"] = current["opponent"].map(canon_team)
    current["_pg"] = current["position_group"].map(pos_group)
    current_rate = pos[
        ["season", "week", "_def", "_pg", "shrunk_rate"]
    ].rename(columns={"shrunk_rate": "_current_shrunk"})
    current = current.merge(
        current_rate,
        on=["season", "week", "_def", "_pg"],
        how="left",
        validate="many_to_one",
    )
    lag_actual = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + ["matchup_position_allowed_shrunk_rate_lag1"],
    )
    current = sort_grain(current)
    lag_actual = sort_grain(lag_actual)
    curv = pd.to_numeric(current["_current_shrunk"], errors="coerce")
    lagv = pd.to_numeric(
        lag_actual["matchup_position_allowed_shrunk_rate_lag1"],
        errors="coerce",
    )
    distinguishable = curv.notna() & lagv.notna() & ~np.isclose(
        curv.to_numpy(dtype="float64"),
        lagv.to_numpy(dtype="float64"),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    if int(distinguishable.sum()) == 0:
        fail("Could not establish position-allowed lag differs from same-week values")

    print("CHECK 12: independent reconstruction of all required matchup formulas")
    match_cols = [
        "team_offensive_plays_roll3_mean",
        "opponent_defensive_plays_roll3_mean",
        "team_dropbacks_roll3_mean",
        "opponent_opponent_dropbacks_roll3_mean",
        "team_rush_attempts_roll3_mean",
        "opponent_opponent_rush_attempts_roll3_mean",
        "player_target_share_roll3_mean",
        "player_carry_share_roll3_mean",
        "player_tackle_rate_per_def_play_roll3_mean",
        "player_sack_rate_per_def_play_roll5_mean",
        "team_off_epa_per_play_roll3_mean",
        "opponent_def_epa_per_play_roll3_mean",
        "opponent_opponent_pass_attempts_roll3_mean",
        "team_pass_rate_roll3_mean",
        "team_rush_rate_roll3_mean",
        "matchup_position_allowed_targets_allowed_lag1",
        "matchup_position_allowed_carries_allowed_lag1",
    ]
    formula_frame = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + ["team", "opponent"] + match_cols + REQUIRED_MATCHUPS,
    )

    # Opponent offense and player's team defense are reconstructed independently
    # from source tables rather than using hidden builder helper columns.
    team_critical = pd.read_parquet(
        full_path(paths["team_form"]),
        columns=[
            "season", "week", "team",
            "offensive_plays_roll3_mean",
            "dropbacks_roll3_mean",
        ],
    )
    team_critical["_join"] = team_critical["team"].map(canon_team)

    opp_def_critical = pd.read_parquet(
        full_path(paths["opponent_form"]),
        columns=[
            "season", "week", "team",
            "defensive_plays_roll3_mean",
            "opponent_dropbacks_roll3_mean",
        ],
    )
    opp_def_critical["_join"] = opp_def_critical["team"].map(canon_team)

    calc = formula_frame.copy()
    calc["_team"] = calc["team"].map(canon_team)
    calc["_opp"] = calc["opponent"].map(canon_team)

    opp_off = team_critical[
        ["season", "week", "_join", "offensive_plays_roll3_mean", "dropbacks_roll3_mean"]
    ].rename(
        columns={
            "_join": "_opp",
            "offensive_plays_roll3_mean": "_opp_off_plays",
            "dropbacks_roll3_mean": "_opp_off_dropbacks",
        }
    )
    team_def = opp_def_critical[
        ["season", "week", "_join", "defensive_plays_roll3_mean", "opponent_dropbacks_roll3_mean"]
    ].rename(
        columns={
            "_join": "_team",
            "defensive_plays_roll3_mean": "_team_def_plays",
            "opponent_dropbacks_roll3_mean": "_team_def_dropbacks",
        }
    )

    calc = calc.merge(
        opp_off,
        on=["season", "week", "_opp"],
        how="left",
        validate="many_to_one",
    )
    calc = calc.merge(
        team_def,
        on=["season", "week", "_team"],
        how="left",
        validate="many_to_one",
    )

    expected = {}
    expected["matchup_expected_team_plays"] = safe_mean(
        calc["team_offensive_plays_roll3_mean"],
        calc["opponent_defensive_plays_roll3_mean"],
    )
    expected["matchup_expected_team_dropbacks"] = safe_mean(
        calc["team_dropbacks_roll3_mean"],
        calc["opponent_opponent_dropbacks_roll3_mean"],
    )
    expected["matchup_expected_team_rush_attempts"] = safe_mean(
        calc["team_rush_attempts_roll3_mean"],
        calc["opponent_opponent_rush_attempts_roll3_mean"],
    )
    expected["matchup_expected_opponent_plays"] = safe_mean(
        calc["_opp_off_plays"],
        calc["_team_def_plays"],
    )
    expected["matchup_expected_opponent_dropbacks"] = safe_mean(
        calc["_opp_off_dropbacks"],
        calc["_team_def_dropbacks"],
    )
    expected["matchup_player_target_share_x_opp_targets"] = (
        pd.to_numeric(calc["player_target_share_roll3_mean"], errors="coerce")
        * pd.to_numeric(
            calc["matchup_position_allowed_targets_allowed_lag1"],
            errors="coerce",
        )
    )
    expected["matchup_player_carry_share_x_opp_rushes"] = (
        pd.to_numeric(calc["player_carry_share_roll3_mean"], errors="coerce")
        * pd.to_numeric(
            calc["matchup_position_allowed_carries_allowed_lag1"],
            errors="coerce",
        )
    )
    expected["matchup_player_tackle_rate_x_opp_plays"] = (
        pd.to_numeric(
            calc["player_tackle_rate_per_def_play_roll3_mean"],
            errors="coerce",
        )
        * expected["matchup_expected_opponent_plays"]
    )
    expected["matchup_player_sack_rate_x_opp_plays"] = (
        pd.to_numeric(
            calc["player_sack_rate_per_def_play_roll5_mean"],
            errors="coerce",
        )
        * expected["matchup_expected_opponent_plays"]
    )
    expected["matchup_off_epa_vs_def_epa"] = (
        pd.to_numeric(calc["team_off_epa_per_play_roll3_mean"], errors="coerce")
        - pd.to_numeric(calc["opponent_def_epa_per_play_roll3_mean"], errors="coerce")
    )
    expected_pass_rate = safe_divide(
        calc["opponent_opponent_pass_attempts_roll3_mean"],
        calc["opponent_defensive_plays_roll3_mean"],
    )
    expected_rush_rate = safe_divide(
        calc["opponent_opponent_rush_attempts_roll3_mean"],
        calc["opponent_defensive_plays_roll3_mean"],
    )
    expected["matchup_pass_rate_vs_opponent"] = (
        pd.to_numeric(calc["team_pass_rate_roll3_mean"], errors="coerce")
        - expected_pass_rate
    )
    expected["matchup_rush_rate_vs_opponent"] = (
        pd.to_numeric(calc["team_rush_rate_roll3_mean"], errors="coerce")
        - expected_rush_rate
    )

    for c in REQUIRED_MATCHUPS:
        assert_numeric_equal(calc[c], expected[c], f"matchup formula:{c}")

    print("CHECK 13: audit fields and strict-prior provenance")
    audit_frame = pd.read_parquet(
        full_path(output_rel),
        columns=GRAIN + [
            "kickoff_timestamp",
            "audit_feature_asof",
            "audit_max_player_source_game",
            "audit_max_team_source_week",
            "audit_depth_snapshot_at",
            "audit_injury_snapshot_at",
            "audit_market_feature_count",
            "audit_row_valid",
        ],
    )
    assert_datetime_equal(
        audit_frame["audit_feature_asof"],
        audit_frame["kickoff_timestamp"],
        "audit_feature_asof",
    )
    if audit_frame["audit_depth_snapshot_at"].notna().any():
        fail("audit_depth_snapshot_at should be null because upstream timestamp is unavailable")
    if audit_frame["audit_injury_snapshot_at"].notna().any():
        fail("audit_injury_snapshot_at should be null because upstream timestamp is unavailable")
    if not pd.to_numeric(
        audit_frame["audit_market_feature_count"], errors="coerce"
    ).eq(0).all():
        fail("audit_market_feature_count is not zero")
    if not pd.to_numeric(
        audit_frame["audit_row_valid"], errors="coerce"
    ).eq(1).all():
        fail("audit_row_valid is not one")

    # Player source-game provenance: latest strictly prior played universe game.
    player_audit = pd.read_parquet(
        full_path(paths["historical_universe"]),
        columns=GRAIN + ["kickoff_timestamp", "played_game_flag"],
    )
    player_audit["_kick"] = pd.to_datetime(
        player_audit["kickoff_timestamp"], errors="raise", utc=True
    )
    player_audit = player_audit.sort_values(
        ["player_id", "_kick", "game_id"], kind="mergesort"
    ).reset_index(drop=True)
    played = pd.to_numeric(
        player_audit["played_game_flag"], errors="coerce"
    ).fillna(0).eq(1)
    source_game = player_audit["game_id"].astype("string").where(played)
    player_audit["_expected_player_source"] = (
        source_game.groupby(player_audit["player_id"], sort=False)
        .transform(lambda s: s.ffill().shift(1))
    )
    exp_player_audit = player_audit[GRAIN + ["_expected_player_source"]]
    actual_player_audit = audit_frame[GRAIN + ["audit_max_player_source_game"]]
    exp_player_audit = sort_grain(exp_player_audit)
    actual_player_audit = sort_grain(actual_player_audit)
    assert_text_equal(
        actual_player_audit["audit_max_player_source_game"],
        exp_player_audit["_expected_player_source"],
        "audit_max_player_source_game",
    )

    # Team source week provenance: previous canonical franchise team_form row.
    team_audit_src = pd.read_parquet(
        full_path(paths["team_form"]),
        columns=["season", "week", "team"],
    )
    team_audit_src["_team"] = team_audit_src["team"].map(canon_team)
    team_audit_src = team_audit_src.sort_values(
        ["_team", "season", "week"], kind="mergesort"
    ).reset_index(drop=True)
    label = (
        team_audit_src["season"].astype(str)
        + "-W"
        + team_audit_src["week"].astype(str).str.zfill(2)
    )
    team_audit_src["_expected_team_source"] = (
        label.groupby(team_audit_src["_team"], sort=False).shift(1)
    )
    exp_team_audit = universe[GRAIN + ["team"]].copy()
    exp_team_audit["_team"] = exp_team_audit["team"].map(canon_team)
    exp_team_audit = exp_team_audit.merge(
        team_audit_src[
            ["season", "week", "_team", "_expected_team_source"]
        ],
        on=["season", "week", "_team"],
        how="left",
        validate="many_to_one",
    )
    actual_team_audit = audit_frame[GRAIN + ["audit_max_team_source_week"]]
    exp_team_audit = sort_grain(exp_team_audit)
    actual_team_audit = sort_grain(actual_team_audit)
    assert_text_equal(
        actual_team_audit["audit_max_team_source_week"],
        exp_team_audit["_expected_team_source"],
        "audit_max_team_source_week",
    )

    print("CHECK 14: infinities and required manifest metadata")
    # Scan all numeric columns in manageable batches.
    numeric_output_cols = [
        field.name
        for field in arrow_schema
        if (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_boolean(field.type)
            or pa.types.is_decimal(field.type)
        )
    ]
    batch_size = 80
    for start in range(0, len(numeric_output_cols), batch_size):
        batch = numeric_output_cols[start : start + batch_size]
        frame = pd.read_parquet(full_path(output_rel), columns=batch)
        for c in batch:
            values = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype="float64")
            if np.isinf(values).any():
                fail(f"Infinity detected in {c}")

    if manifest.get("market_features_used") is not False:
        fail("Manifest market_features_used must be false")
    if manifest.get("target_columns_in_feature_manifest") is not False:
        fail("Manifest target_columns_in_feature_manifest must be false")
    if manifest.get("row_count") != output_rows:
        fail("Manifest row_count does not match parquet")
    if manifest.get("column_count") != len(output_names):
        fail("Manifest column_count does not match parquet")
    if manifest.get("feature_count") != len(features):
        fail("Manifest feature_count does not match feature_columns")
    if manifest.get("numeric_feature_count") != len(numeric):
        fail("Manifest numeric_feature_count mismatch")
    if manifest.get("categorical_feature_count") != len(categorical):
        fail("Manifest categorical_feature_count mismatch")
    if manifest.get("required_matchup_columns") != REQUIRED_MATCHUPS:
        fail("Manifest required_matchup_columns mismatch")

    print(
        json.dumps(
            {
                "status": "passed",
                "issue": 17,
                "rows": output_rows,
                "columns": len(output_names),
                "features": len(features),
                "numeric_features": len(numeric),
                "categorical_features": len(categorical),
                "market_feature_count": 0,
                "target_columns_in_manifest": False,
                "position_allowed_lag_verified": True,
                "required_matchup_formulas_verified": len(REQUIRED_MATCHUPS),
                "targets_verified": len(TARGETS),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print("ISSUE 17 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "issue": 17,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        print("ISSUE 17 ACCEPTANCE: FAIL", file=sys.stderr)
        raise

