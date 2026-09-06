#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 37."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

GRAIN = ["season", "week", "game_id", "player_id"]
TARGETS = [
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
OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "passing_yards",
    "passing_yards_low",
    "passing_yards_high",
    "passing_tds",
    "passing_tds_prob_1plus",
    "rushing_yards",
    "rushing_yards_low",
    "rushing_yards_high",
    "rushing_tds",
    "rushing_tds_prob_1plus",
    "receiving_yards",
    "receiving_yards_low",
    "receiving_yards_high",
    "receiving_tds",
    "receiving_tds_prob_1plus",
    "kicking_points",
    "kicking_points_low",
    "kicking_points_high",
    "tackles",
    "tackles_low",
    "tackles_high",
    "sacks",
    "sacks_prob_1plus",
    "injury_game_status",
    "role_status",
    "generated_at",
]
TEXT_COLUMNS = [
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "injury_game_status",
    "role_status",
    "generated_at",
]
NUMERIC_OUTPUT_COLUMNS = [
    column for column in OUTPUT_COLUMNS if column not in set(TEXT_COLUMNS)
]
TARGET_MAP: dict[str, dict[str, str]] = {
    "passing_yards": {"projection": "passing_yards", "low": "passing_yards_low", "high": "passing_yards_high"},
    "passing_tds": {"projection": "passing_tds", "probability_1_plus": "passing_tds_prob_1plus"},
    "rushing_yards": {"projection": "rushing_yards", "low": "rushing_yards_low", "high": "rushing_yards_high"},
    "rushing_tds": {"projection": "rushing_tds", "probability_1_plus": "rushing_tds_prob_1plus"},
    "receiving_yards": {"projection": "receiving_yards", "low": "receiving_yards_low", "high": "receiving_yards_high"},
    "receiving_tds": {"projection": "receiving_tds", "probability_1_plus": "receiving_tds_prob_1plus"},
    "kicking_points": {"projection": "kicking_points", "low": "kicking_points_low", "high": "kicking_points_high"},
    "tackles": {"projection": "tackles", "low": "tackles_low", "high": "tackles_high"},
    "sacks": {"projection": "sacks", "probability_1_plus": "sacks_prob_1plus"},
}
META = ["player_name", "team", "opponent", "position", "injury_game_status", "role_status", "generated_at"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, required=True)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def normalize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    for column in ["game_id", "player_id"]:
        out[column] = out[column].fillna("").astype(str).str.strip()
        if out[column].eq("").any():
            raise AssertionError(f"{label}: blank {column}")
    return out


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def independent_expected(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby(GRAIN, sort=False, dropna=False)
    for column in META:
        counts = grouped[column].nunique(dropna=False)
        if counts.gt(1).any():
            raise AssertionError(f"Source metadata not invariant across targets: {column}")

    target_counts = grouped["target"].count()
    if not target_counts.eq(len(TARGETS)).all():
        raise AssertionError("Issue 36 long source does not contain exactly nine targets per player")
    target_sets = grouped["target"].agg(lambda x: set(map(str, x)))
    if not target_sets.map(lambda x: x == set(TARGETS)).all():
        raise AssertionError("Issue 36 long source target coverage differs by player")

    expected = long[[*GRAIN, *META]].drop_duplicates(GRAIN, keep="first").copy()
    if expected.duplicated(GRAIN).any():
        raise AssertionError("Expected base grain is not unique")

    # Independently attach each target's required values instead of importing the builder.
    for target in TARGETS:
        src = long.loc[long["target"].astype(str).eq(target)].copy()
        if src.duplicated(GRAIN).any():
            raise AssertionError(f"Duplicate long source row for {target}")
        part = src[GRAIN].copy()
        for in_col, out_col in TARGET_MAP[target].items():
            part[out_col] = num(src[in_col]).to_numpy()
        expected = expected.merge(part, on=GRAIN, how="left", validate="one_to_one", sort=False)

    return expected[OUTPUT_COLUMNS]


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)
    prop = common.prop_root()

    builder = prop / "scripts" / "report" / "build_wide_output.py"
    long_path = prop / "output" / str(season) / f"week_{week}_player_projections.csv"
    wide_path = prop / "output" / str(season) / f"week_{week}_player_projections_wide.csv"
    issue36_log_path = prop / "logs" / f"week_projections_{season}_week_{week}.json"
    log_path = prop / "logs" / f"wide_output_{season}_week_{week}.json"

    print("CHECK 01: required builder/source/output/log, Issue 36 sequence gate, and exact headers")
    for path in [builder, long_path, wide_path, issue36_log_path, log_path]:
        if not path.is_file():
            raise AssertionError(f"Missing Issue 37 artifact/input: {path}")
    issue36 = read_json(issue36_log_path)
    log = read_json(log_path)
    if issue36.get("status") != "passed":
        raise AssertionError("Issue 36 sequence gate is not passed")
    if issue36.get("market_features_used") is not False:
        raise AssertionError("Issue 36 market policy violation")
    if log.get("status") != "passed":
        raise AssertionError("Issue 37 log status is not passed")
    if log.get("market_features_used") is not False or log.get("market_exclusion_passed") is not True:
        raise AssertionError("Issue 37 market policy/log failure")

    long = pd.read_csv(long_path, dtype={"game_id": "string", "player_id": "string"}, keep_default_na=False)
    wide = pd.read_csv(wide_path, dtype={"game_id": "string", "player_id": "string"}, keep_default_na=False)
    long = normalize(long, "Issue 36 long output")
    wide = normalize(wide, "Issue 37 wide output")
    if list(wide.columns) != OUTPUT_COLUMNS:
        raise AssertionError(f"Issue 37 exact header/order mismatch: {list(wide.columns)}")
    if set(long["season"]) != {season} or set(long["week"]) != {week}:
        raise AssertionError("Long source season/week mismatch")
    if set(wide["season"]) != {season} or set(wide["week"]) != {week}:
        raise AssertionError("Wide output season/week mismatch")

    print("CHECK 02: one-row-per-player grain and complete nine-target source coverage")
    if wide.duplicated(GRAIN).any():
        raise AssertionError("Wide output duplicates canonical player-game grain")
    expected_players = long[GRAIN].drop_duplicates()
    if len(wide) != len(expected_players):
        raise AssertionError(f"Wide row count {len(wide)} != source players {len(expected_players)}")
    expected_keys = set(map(tuple, expected_players.to_numpy()))
    actual_keys = set(map(tuple, wide[GRAIN].to_numpy()))
    if actual_keys != expected_keys:
        raise AssertionError("Wide output player-game key set differs from Issue 36 audit source")
    if set(long["target"].astype(str).unique()) != set(TARGETS):
        raise AssertionError("Issue 36 long source target set mismatch")

    print("CHECK 03: independently reconstruct every target projection/interval/probability column")
    expected = independent_expected(long)
    expected = expected.set_index(GRAIN).sort_index()
    actual = wide.set_index(GRAIN).sort_index()
    actual = actual.reindex(expected.index)

    for column in META:
        av = actual[column].fillna("").astype(str)
        ev = expected[column].fillna("").astype(str)
        if not av.equals(ev):
            raise AssertionError(f"Issue 37 metadata mismatch: {column}")

    max_error = 0.0
    numeric_compare = [column for column in OUTPUT_COLUMNS if column not in {*GRAIN, *META}]
    for column in numeric_compare:
        av = num(actual[column]).to_numpy(dtype="float64")
        ev = num(expected[column]).to_numpy(dtype="float64")
        if not np.array_equal(np.isnan(av), np.isnan(ev)):
            raise AssertionError(f"Issue 37 null mask mismatch: {column}")
        finite = np.isfinite(av) & np.isfinite(ev)
        if finite.any():
            error = float(np.max(np.abs(av[finite] - ev[finite])))
            max_error = max(max_error, error)
            if error > 1e-12:
                raise AssertionError(f"Issue 37 value mismatch {column}: max_abs_error={error}")

    print("CHECK 04: probabilities, metadata preservation, and no added reporting math")
    probability_columns = [
        "passing_tds_prob_1plus",
        "rushing_tds_prob_1plus",
        "receiving_tds_prob_1plus",
        "sacks_prob_1plus",
    ]
    for column in probability_columns:
        values = num(wide[column])
        if values.isna().any() or values.lt(0).any() or values.gt(1).any():
            raise AssertionError(f"Invalid Issue 37 probability column: {column}")
    if log.get("additional_rounding_applied") is not False:
        raise AssertionError("Issue 37 must not add rounding after Issue 36")
    policy = log.get("policy", {})
    for key in [
        "issue36_audit_output_is_authoritative",
        "full_audit_player_population_preserved",
        "one_row_per_player_game",
        "no_model_scoring",
        "no_recalibration",
        "no_additional_rounding",
        "market_exclusion_preflight",
    ]:
        if policy.get(key) is not True:
            raise AssertionError(f"Issue 37 log policy missing/false: {key}")

    print("CHECK 05: final counts and market policy")
    if int(log.get("players", -1)) != len(wide):
        raise AssertionError("Issue 37 log player count mismatch")
    if int(log.get("columns", -1)) != len(OUTPUT_COLUMNS):
        raise AssertionError("Issue 37 log column count mismatch")
    if int(log.get("source_long_rows", -1)) != len(long):
        raise AssertionError("Issue 37 log source row count mismatch")
    if int(log.get("targets_reshaped", -1)) != len(TARGETS):
        raise AssertionError("Issue 37 log target count mismatch")

    print(f"season={season}")
    print(f"week={week}")
    print(f"players={len(wide)}")
    print(f"games={wide['game_id'].nunique()}")
    print(f"teams={wide['team'].nunique()}")
    print(f"columns={len(wide.columns)}")
    print(f"source_long_rows={len(long)}")
    print(f"targets_reshaped={len(TARGETS)}")
    print(f"max_wide_value_abs_error={max_error:.12g}")
    print("probabilities_in_unit_interval=true")
    print("additional_rounding_applied=false")
    print("market_features_used=false")
    print("ISSUE 37 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
