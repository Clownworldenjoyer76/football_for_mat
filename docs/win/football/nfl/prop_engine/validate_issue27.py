#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 27."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = THIS_DIR / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common

GRAIN = ["season", "week", "game_id", "player_id"]
TRAINER = THIS_DIR / "scripts" / "validate" / "validate_historical_data.py"
REPORT = THIS_DIR / "evaluation" / "historical_validation.json"
MANIFEST = THIS_DIR / "data" / "historical" / "features" / "feature_manifest.json"
ELIGIBILITY = THIS_DIR / "config" / "target_eligibility.yaml"

REQUIRED_CHECKS = [
    "unique_feature_table_grain",
    "nonblank_player_ids",
    "team_belongs_to_game",
    "opponent_is_other_game_team",
    "targets_valid_and_nonnegative",
    "no_forbidden_feature",
    "no_sportsbook_market_path_in_manifest",
    "no_same_game_snap_feature",
    "no_same_game_participation_feature",
    "no_same_game_target_share",
    "no_same_game_team_performance",
    "no_same_game_opponent_performance",
    "no_final_score_feature",
    "no_target_column_in_feature_list",
    "depth_snapshot_precedes_kickoff",
    "injury_snapshot_precedes_kickoff_when_timestamps_exist",
    "no_join_multiplication",
    "minimum_training_sample_per_target",
    "failure_returns_nonzero_exit_code",
]

HISTORICAL_TEAM_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}
SAFE = (
    "prior", "lag", "roll", "ewm", "change", "unavailable", "missing",
    "season_to_date", "career_prior",
)
SAFE_SUFFIXES = (
    "_lag1", "_roll3_mean", "_roll5_mean", "_roll8_mean",
    "_ewm3", "_ewm5", "_season_to_date",
)
BAD_PATH_TOKENS = ("sportsbook", "market", "odds", "prop_line", "betting", "wager", "drat", "epred")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    assert isinstance(value, dict), f"Expected JSON object: {path}"
    return value


def canon_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_TEAM_ALIASES.get(team, team)


def load_games(config: dict[str, Any]) -> pd.DataFrame:
    path = common.repo_root() / config["paths"]["historical_games"]
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    type_col = "game_type" if "game_type" in header.columns else "season_type"
    cols = ["season", "week", "game_id", "home_team", "away_team", type_col]
    assert all(column in header.columns for column in cols), "Historical games schema mismatch"
    games = pd.read_csv(path, usecols=cols, encoding="utf-8-sig", low_memory=False)
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games.loc[
        games[type_col].astype("string").str.upper().eq("REG")
        & games["season"].between(
            int(config["seasons"]["historical_start"]),
            int(config["seasons"]["historical_end"]),
            inclusive="both",
        )
    ].copy()
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    games["game_id"] = games["game_id"].astype("string").str.strip()
    games["home"] = games["home_team"].map(canon_team)
    games["away"] = games["away_team"].map(canon_team)
    assert not games.duplicated(["season", "week", "game_id"]).any(), "Duplicate game grain"
    return games[["season", "week", "game_id", "home", "away"]]


def matchup_safe(column: str, manifest: dict[str, Any]) -> bool:
    lowered = column.casefold()
    if any(token in lowered for token in SAFE):
        return True
    formula = str(manifest.get("matchup_formulas", {}).get(column, "")).casefold()
    return bool(formula) and any(token in formula for token in ("prior", "lag", "roll", "ewm", "season_to_date", "career_prior"))


def main() -> int:
    print("CHECK 01: required historical validation script and output")
    assert TRAINER.is_file(), f"Missing {TRAINER}"
    assert REPORT.is_file(), f"Missing {REPORT}"
    assert MANIFEST.is_file(), f"Missing {MANIFEST}"
    assert ELIGIBILITY.is_file(), f"Missing {ELIGIBILITY}"

    report = read_json(REPORT)
    manifest = read_json(MANIFEST)
    config = common.load_config()
    eligibility = yaml.safe_load(ELIGIBILITY.read_text(encoding="utf-8-sig"))
    assert isinstance(eligibility, dict)

    assert report.get("status") == "passed", f"Historical gate status={report.get('status')}"
    report_checks = report.get("checks", [])
    assert isinstance(report_checks, list)
    names = [item.get("name") for item in report_checks]
    assert names == REQUIRED_CHECKS, f"Historical gate check list mismatch: {names}"
    assert all(item.get("passed") is True for item in report_checks), "Historical gate contains failed check"
    assert report.get("checks_failed") == 0
    assert report.get("checks_passed") == len(REQUIRED_CHECKS)

    feature_path = common.repo_root() / config["paths"]["historical_features"]
    universe_path = common.repo_root() / config["paths"]["historical_universe"]
    schema = list(pq.ParquetFile(feature_path).schema.names)
    targets = list(config["targets"].keys())
    target_cols = [f"target_{target}" for target in targets]
    cols = GRAIN + [
        "team", "opponent", "position", "kickoff_timestamp",
        "audit_depth_snapshot_at", "audit_injury_snapshot_at",
        "audit_market_feature_count",
    ] + target_cols
    assert all(column in schema for column in cols), "Missing required historical columns"
    df = pd.read_parquet(feature_path, columns=cols)

    print("CHECK 02: independent grain, identity, game-team, and opponent checks")
    assert not df.duplicated(GRAIN).any(), "Historical feature grain is not unique"
    ids = df["player_id"].astype("string").str.strip().str.casefold()
    assert not (df["player_id"].isna() | ids.isin(["", "nan", "none", "null", "<na>"])).any(), "Blank player_id"

    games = load_games(config)
    probe = df[["season", "week", "game_id", "team", "opponent"]].copy()
    probe["season"] = pd.to_numeric(probe["season"], errors="raise").astype(int)
    probe["week"] = pd.to_numeric(probe["week"], errors="raise").astype(int)
    probe["game_id"] = probe["game_id"].astype("string").str.strip()
    probe["team_c"] = probe["team"].map(canon_team)
    probe["opp_c"] = probe["opponent"].map(canon_team)
    probe = probe.merge(games, on=["season", "week", "game_id"], how="left", validate="many_to_one")
    assert probe["home"].notna().all() and probe["away"].notna().all(), "Feature game missing from games source"
    is_home = probe["team_c"].eq(probe["home"])
    is_away = probe["team_c"].eq(probe["away"])
    assert (is_home | is_away).all(), "Team does not belong to game"
    expected_opp = pd.Series(np.where(is_home, probe["away"], probe["home"]), index=probe.index)
    assert probe["opp_c"].eq(expected_opp).all(), "Opponent is not the other game team"
    assert ~probe["team_c"].eq(probe["opp_c"]).any(), "Team equals opponent"

    print("CHECK 03: independent target validity and minimum samples")
    training_end = int(config["seasons"]["historical_end"]) - 1
    minimum = int(config.get("validation", {}).get("minimum_training_rows_per_target", 100))
    assert minimum >= 1
    position = df["position"].astype("string").str.strip().str.upper()
    season = pd.to_numeric(df["season"], errors="raise")
    sample_counts: dict[str, int] = {}
    for target, spec in config["targets"].items():
        col = f"target_{target}"
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        assert not (raw.notna() & numeric.isna()).any(), f"Nonnumeric target values: {target}"
        assert not (numeric.notna() & ~np.isfinite(numeric)).any(), f"Nonfinite target values: {target}"
        if spec.get("type") != "continuous_signed":
            assert not (numeric.notna() & numeric.lt(0)).any(), f"Negative nonnegative target: {target}"
        eligible_positions = {
            str(value).strip().upper()
            for value in eligibility.get(target, {}).get("eligible_positions", [])
        }
        mask = season.le(training_end) & numeric.notna()
        if eligible_positions:
            mask &= position.isin(eligible_positions)
        sample_counts[target] = int(mask.sum())
        assert sample_counts[target] >= minimum, (
            f"Insufficient training rows for {target}: {sample_counts[target]} < {minimum}"
        )

    print("CHECK 04: independent manifest leakage and market-path checks")
    features = list(manifest.get("feature_columns", []))
    assert features, "Empty feature manifest"
    common.reject_forbidden_feature_columns(features, config)
    assert manifest.get("market_features_used") is False
    assert pd.to_numeric(df["audit_market_feature_count"], errors="coerce").fillna(-1).eq(0).all()

    source_paths = manifest.get("source_paths", {})
    assert isinstance(source_paths, dict)
    path_values = [str(value) for value in source_paths.values()] + [str(manifest.get("output_path", ""))]
    assert not [
        value for value in path_values
        if any(token in value.casefold().replace("\\", "/") for token in BAD_PATH_TOKENS)
    ], "Sportsbook/market path detected in feature manifest"

    bad_snap = [c for c in features if "snap" in c.casefold() and not any(t in c.casefold() for t in SAFE)]
    bad_part = [c for c in features if "participation" in c.casefold() and not any(t in c.casefold() for t in SAFE)]
    assert not bad_snap, f"Same-game snap feature(s): {bad_snap}"
    assert not bad_part, f"Same-game participation feature(s): {bad_part}"

    bad_share = []
    for column in features:
        if "target_share" not in column.casefold():
            continue
        if any(token in column.casefold() for token in SAFE):
            continue
        if column.startswith("matchup_") and matchup_safe(column, manifest):
            continue
        bad_share.append(column)
    assert not bad_share, f"Same-game target share feature(s): {bad_share}"

    families = manifest.get("column_families", {})
    assert isinstance(families, dict)
    bad_team = [c for c in families.get("team", []) if not c.endswith(SAFE_SUFFIXES)]
    bad_opp = [c for c in families.get("opponent", []) if not c.endswith(SAFE_SUFFIXES)]
    bad_matchup = [c for c in families.get("matchup", []) if not matchup_safe(c, manifest)]
    assert not bad_team, f"Same-game team performance feature(s): {bad_team}"
    assert not bad_opp, f"Same-game opponent performance feature(s): {bad_opp}"
    assert not bad_matchup, f"Unsafe matchup feature(s): {bad_matchup}"

    bad_final = [
        c for c in features
        if c.casefold() in {"score", "home_score", "away_score", "final_score", "score_differential", "point_differential", "margin", "result", "win_flag", "loss_flag"}
        or "final_score" in c.casefold()
        or "score_differential" in c.casefold()
        or "point_differential" in c.casefold()
    ]
    assert not bad_final, f"Final score feature(s): {bad_final}"
    target_manifest_cols = set(manifest.get("target_columns", []))
    leaks = target_manifest_cols.intersection(features) | {c for c in features if c.startswith("target_")}
    assert not leaks, f"Target column in feature list: {sorted(leaks)}"
    assert manifest.get("target_columns_in_feature_manifest") is False

    print("CHECK 05: independent snapshot and join-multiplication checks")
    kickoff = pd.to_datetime(df["kickoff_timestamp"], errors="coerce", utc=True)
    assert kickoff.notna().all(), "Invalid kickoff timestamp"
    depth = pd.to_datetime(df["audit_depth_snapshot_at"], errors="coerce", utc=True)
    injury = pd.to_datetime(df["audit_injury_snapshot_at"], errors="coerce", utc=True)
    if depth.notna().any():
        assert (depth.dropna() < kickoff.loc[depth.notna()]).all(), "Depth snapshot not strictly pre-kickoff"
    else:
        policy = str(manifest.get("audit_policy", {}).get("audit_depth_snapshot_at", "")).casefold()
        assert "validated pregame depth" in policy, "Missing upstream pregame depth audit contract"
    if injury.notna().any():
        assert (injury.dropna() < kickoff.loc[injury.notna()]).all(), "Injury snapshot not strictly pre-kickoff"

    universe = pd.read_parquet(universe_path, columns=GRAIN)
    assert not universe.duplicated(GRAIN).any(), "Universe grain duplicate"
    assert len(universe) == len(df) == int(manifest.get("row_count", -1)), "Row multiplication detected"
    left = df[GRAIN].copy()
    right = universe[GRAIN].copy()
    for block in (left, right):
        block["season"] = pd.to_numeric(block["season"], errors="raise").astype(int)
        block["week"] = pd.to_numeric(block["week"], errors="raise").astype(int)
        block["game_id"] = block["game_id"].astype("string").str.strip()
        block["player_id"] = block["player_id"].astype("string").str.strip()
    parity = left.merge(right, on=GRAIN, how="outer", indicator=True, validate="one_to_one")
    assert parity["_merge"].eq("both").all(), "Feature/universe grain mismatch"

    print("CHECK 06: failure path returns nonzero exit code")
    result = subprocess.run(
        [sys.executable, str(TRAINER), "--self-test-failure-exit"],
        cwd=str(common.repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode != 0, "Historical validation self-test failure returned zero"
    assert result.returncode == 1, f"Unexpected failure exit code: {result.returncode}"

    print("CHECK 07: summarize independently validated historical gate")
    print(f"rows={len(df)}")
    print(f"features={len(features)}")
    print(f"targets={len(targets)}")
    print(f"checks={len(REQUIRED_CHECKS)}")
    print(f"training_end={training_end}")
    print(f"minimum_training_rows={minimum}")
    for target in targets:
        print(f"  {target}: training_rows={sample_counts[target]}")
    print("market_features_used=false")
    print("sportsbook_market_manifest_paths=false")
    print("same_game_leakage=false")
    print("join_multiplication=false")
    print("failure_exit_code=1")
    print("ISSUE 27 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
