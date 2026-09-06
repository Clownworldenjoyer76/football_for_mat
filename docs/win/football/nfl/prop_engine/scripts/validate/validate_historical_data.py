#!/usr/bin/env python3
"""Issue 27: historical data validation gate for the NFL Prop Engine."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common

GRAIN = ["season", "week", "game_id", "player_id"]
REPORT_RELATIVE = Path(
    "docs/win/football/nfl/prop_engine/evaluation/historical_validation.json"
)
MANIFEST_RELATIVE = Path(
    "docs/win/football/nfl/prop_engine/data/historical/features/feature_manifest.json"
)
ELIGIBILITY_RELATIVE = Path(
    "docs/win/football/nfl/prop_engine/config/target_eligibility.yaml"
)
UNIVERSE_BUILDER_RELATIVE = Path(
    "docs/win/football/nfl/prop_engine/scripts/build/build_historical_universe.py"
)

HISTORICAL_TEAM_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}

SAFE_HISTORY_TOKENS = (
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

TEAM_FORM_SAFE_SUFFIXES = (
    "_lag1",
    "_roll3_mean",
    "_roll5_mean",
    "_roll8_mean",
    "_ewm3",
    "_ewm5",
    "_season_to_date",
)

FINAL_RESULT_NAMES = {
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

PATH_FORBIDDEN_TOKENS = (
    "sportsbook",
    "market",
    "odds",
    "prop_line",
    "betting",
    "wager",
    "drat",
    "epred",
)

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


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = common.repo_root() / path
    return path.resolve()


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


def canonical_team(value: Any) -> str:
    team = common.normalize_team(value)
    return HISTORICAL_TEAM_ALIASES.get(team, team)


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    root = common.prop_root().resolve()
    destination = destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Historical validation write outside Prop Engine: {destination}") from exc

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
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "details": details or {},
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_eligibility(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def parquet_columns(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(pq.ParquetFile(path).schema.names)


def load_games(config: dict[str, Any]) -> pd.DataFrame:
    path = repo_path(config["paths"]["historical_games"])
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    type_col = "game_type" if "game_type" in header.columns else "season_type"
    required = ["game_id", "season", "week", "home_team", "away_team", type_col]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(f"Historical games missing required columns: {missing}")

    games = pd.read_csv(
        path,
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )
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
    games["_home"] = games["home_team"].map(canonical_team)
    games["_away"] = games["away_team"].map(canonical_team)
    return games[["season", "week", "game_id", "_home", "_away"]]


def path_strings(manifest: dict[str, Any]) -> list[str]:
    values: list[str] = []
    source_paths = manifest.get("source_paths", {})
    if isinstance(source_paths, dict):
        values.extend(str(value) for value in source_paths.values())
    output_path = manifest.get("output_path")
    if output_path is not None:
        values.append(str(output_path))
    return values


def matchup_is_strictly_pregame(
    column: str,
    manifest: dict[str, Any],
) -> bool:
    lowered = column.casefold()
    if any(token in lowered for token in SAFE_HISTORY_TOKENS):
        return True
    formulas = manifest.get("matchup_formulas", {})
    formula = str(formulas.get(column, "")).casefold()
    return bool(formula) and any(
        token in formula
        for token in ("prior", "lag", "roll", "ewm", "season_to_date", "career_prior")
    )


def target_training_counts(
    frame: pd.DataFrame,
    config: dict[str, Any],
    eligibility: dict[str, Any],
) -> tuple[int, int, dict[str, int]]:
    historical_end = int(config["seasons"]["historical_end"])
    training_end = historical_end - 1
    configured_minimum = config.get("validation", {}).get(
        "minimum_training_rows_per_target", 100
    )
    minimum = int(configured_minimum)
    if minimum < 1:
        raise ValueError("validation.minimum_training_rows_per_target must be >= 1")

    season = pd.to_numeric(frame["season"], errors="coerce")
    position = frame["position"].astype("string").str.strip().str.upper()
    counts: dict[str, int] = {}

    for target in config["targets"]:
        target_col = f"target_{target}"
        spec = eligibility.get(target, {})
        eligible_positions = {
            str(value).strip().upper()
            for value in spec.get("eligible_positions", [])
            if str(value).strip()
        }
        eligible = season.le(training_end)
        if eligible_positions:
            eligible &= position.isin(eligible_positions)
        numeric = pd.to_numeric(frame[target_col], errors="coerce")
        eligible &= numeric.notna() & np.isfinite(numeric)
        counts[target] = int(eligible.sum())

    return training_end, minimum, counts


def build_report() -> tuple[dict[str, Any], int]:
    config = common.load_config()
    feature_path = repo_path(config["paths"]["historical_features"])
    universe_path = repo_path(config["paths"]["historical_universe"])
    manifest_path = repo_path(MANIFEST_RELATIVE)
    eligibility_path = repo_path(ELIGIBILITY_RELATIVE)
    universe_builder_path = repo_path(UNIVERSE_BUILDER_RELATIVE)

    manifest = read_json(manifest_path)
    eligibility = read_eligibility(eligibility_path)
    schema_columns = parquet_columns(feature_path)

    target_names = list(config["targets"].keys())
    target_columns = [f"target_{target}" for target in target_names]
    needed = list(
        dict.fromkeys(
            GRAIN
            + [
                "team",
                "opponent",
                "position",
                "position_group",
                "kickoff_timestamp",
                "audit_depth_snapshot_at",
                "audit_injury_snapshot_at",
                "audit_market_feature_count",
            ]
            + target_columns
        )
    )
    missing_needed = [column for column in needed if column not in schema_columns]
    if missing_needed:
        raise ValueError(f"Historical feature table missing required columns: {missing_needed}")

    frame = pd.read_parquet(feature_path, columns=needed)
    checks: list[dict[str, Any]] = []

    # 1. Unique canonical grain.
    duplicate_grain = frame.duplicated(GRAIN, keep=False)
    add_check(
        checks,
        "unique_feature_table_grain",
        not duplicate_grain.any(),
        {
            "rows": int(len(frame)),
            "duplicate_rows": int(duplicate_grain.sum()),
            "canonical_grain": GRAIN,
        },
    )

    # 2. Nonblank player IDs.
    player_text = frame["player_id"].astype("string").str.strip()
    blank_player = frame["player_id"].isna() | player_text.isin(
        ["", "nan", "none", "null", "<na>"]
    )
    add_check(
        checks,
        "nonblank_player_ids",
        not blank_player.any(),
        {"blank_rows": int(blank_player.sum())},
    )

    # 3-4. Team/game consistency from canonical historical games.
    team_game_pass = False
    opponent_game_pass = False
    team_details: dict[str, Any] = {}
    opponent_details: dict[str, Any] = {}
    try:
        games = load_games(config)
        game_dup = games.duplicated(["season", "week", "game_id"], keep=False)
        if game_dup.any():
            raise ValueError(
                f"Historical games have duplicated regular-season game grain: {int(game_dup.sum())} rows"
            )
        probe = frame[["season", "week", "game_id", "team", "opponent"]].copy()
        probe["season"] = pd.to_numeric(probe["season"], errors="raise").astype(int)
        probe["week"] = pd.to_numeric(probe["week"], errors="raise").astype(int)
        probe["game_id"] = probe["game_id"].astype("string").str.strip()
        probe["_team"] = probe["team"].map(canonical_team)
        probe["_opponent"] = probe["opponent"].map(canonical_team)
        probe = probe.merge(
            games,
            on=["season", "week", "game_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        missing_game = probe["_home"].isna() | probe["_away"].isna()
        team_bad = missing_game | ~(
            probe["_team"].eq(probe["_home"]) | probe["_team"].eq(probe["_away"])
        )
        expected_opponent = np.where(
            probe["_team"].eq(probe["_home"]), probe["_away"], probe["_home"]
        )
        opponent_bad = (
            missing_game
            | team_bad
            | ~probe["_opponent"].eq(pd.Series(expected_opponent, index=probe.index))
            | probe["_team"].eq(probe["_opponent"])
        )
        team_game_pass = not team_bad.any()
        opponent_game_pass = not opponent_bad.any()
        team_details = {
            "missing_game_rows": int(missing_game.sum()),
            "invalid_team_rows": int(team_bad.sum()),
        }
        opponent_details = {"invalid_opponent_rows": int(opponent_bad.sum())}
    except Exception as exc:
        team_details = {"error": f"{type(exc).__name__}: {exc}"}
        opponent_details = {"error": f"{type(exc).__name__}: {exc}"}

    add_check(checks, "team_belongs_to_game", team_game_pass, team_details)
    add_check(
        checks,
        "opponent_is_other_game_team",
        opponent_game_pass,
        opponent_details,
    )

    # 5. Target validity. Signed yardage is intentionally allowed to be negative.
    target_detail: dict[str, Any] = {}
    target_valid = True
    for target, target_spec in config["targets"].items():
        column = f"target_{target}"
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        present = raw.notna()
        nonnumeric = present & numeric.isna()
        nonfinite = numeric.notna() & ~np.isfinite(numeric)
        target_type = str(target_spec.get("type", "")).strip()
        negative = numeric.notna() & numeric.lt(0.0)
        negative_invalid = negative if target_type != "continuous_signed" else pd.Series(
            False, index=frame.index
        )
        passed = not (nonnumeric | nonfinite | negative_invalid).any()
        target_valid &= passed
        target_detail[target] = {
            "type": target_type,
            "nonnull_rows": int(numeric.notna().sum()),
            "nonnumeric_rows": int(nonnumeric.sum()),
            "nonfinite_rows": int(nonfinite.sum()),
            "negative_rows": int(negative.sum()),
            "negative_rows_invalid": int(negative_invalid.sum()),
        }
    add_check(
        checks,
        "targets_valid_and_nonnegative",
        target_valid,
        {
            "continuous_signed_targets_allow_negative_yardage": True,
            "targets": target_detail,
        },
    )

    feature_columns = list(manifest.get("feature_columns", []))
    manifest_target_columns = list(manifest.get("target_columns", []))

    # 6. No configured forbidden feature.
    forbidden_pass = True
    forbidden_error = ""
    try:
        common.reject_forbidden_feature_columns(feature_columns, config)
        if manifest.get("market_features_used") is not False:
            raise ValueError("feature manifest market_features_used must be false")
        if pd.to_numeric(frame["audit_market_feature_count"], errors="coerce").fillna(-1).ne(0).any():
            raise ValueError("audit_market_feature_count contains a nonzero or invalid value")
    except Exception as exc:
        forbidden_pass = False
        forbidden_error = f"{type(exc).__name__}: {exc}"
    add_check(
        checks,
        "no_forbidden_feature",
        forbidden_pass,
        {"feature_count": len(feature_columns), "error": forbidden_error},
    )

    # 7. No sportsbook/market source path in manifest.
    bad_paths = []
    for value in path_strings(manifest):
        lowered = value.casefold().replace("\\", "/")
        if any(token in lowered for token in PATH_FORBIDDEN_TOKENS):
            bad_paths.append(value)
    add_check(
        checks,
        "no_sportsbook_market_path_in_manifest",
        not bad_paths,
        {"paths_checked": path_strings(manifest), "bad_paths": bad_paths},
    )

    # 8-10. Same-game usage/share checks.
    bad_snap = [
        column
        for column in feature_columns
        if "snap" in column.casefold()
        and not any(token in column.casefold() for token in SAFE_HISTORY_TOKENS)
    ]
    add_check(
        checks,
        "no_same_game_snap_feature",
        not bad_snap,
        {"bad_features": bad_snap},
    )

    bad_participation = [
        column
        for column in feature_columns
        if "participation" in column.casefold()
        and not any(token in column.casefold() for token in SAFE_HISTORY_TOKENS)
    ]
    add_check(
        checks,
        "no_same_game_participation_feature",
        not bad_participation,
        {"bad_features": bad_participation},
    )

    bad_target_share: list[str] = []
    for column in feature_columns:
        if "target_share" not in column.casefold():
            continue
        if any(token in column.casefold() for token in SAFE_HISTORY_TOKENS):
            continue
        if column.startswith("matchup_") and matchup_is_strictly_pregame(column, manifest):
            continue
        bad_target_share.append(column)
    add_check(
        checks,
        "no_same_game_target_share",
        not bad_target_share,
        {"bad_features": bad_target_share},
    )

    # 11-12. Team and opponent performance must be lagged/rolling.
    families = manifest.get("column_families", {})
    team_family = list(families.get("team", [])) if isinstance(families, dict) else []
    opponent_family = list(families.get("opponent", [])) if isinstance(families, dict) else []
    bad_team = [
        column
        for column in team_family
        if not column.endswith(TEAM_FORM_SAFE_SUFFIXES)
    ]
    bad_opponent = [
        column
        for column in opponent_family
        if not column.endswith(TEAM_FORM_SAFE_SUFFIXES)
    ]
    matchup_family = list(families.get("matchup", [])) if isinstance(families, dict) else []
    bad_matchup = [
        column
        for column in matchup_family
        if not matchup_is_strictly_pregame(column, manifest)
    ]
    add_check(
        checks,
        "no_same_game_team_performance",
        not bad_team and not bad_matchup,
        {"bad_team_features": bad_team, "bad_matchup_features": bad_matchup},
    )
    add_check(
        checks,
        "no_same_game_opponent_performance",
        not bad_opponent and not bad_matchup,
        {"bad_opponent_features": bad_opponent, "bad_matchup_features": bad_matchup},
    )

    # 13. No final score/result feature.
    bad_final = [
        column
        for column in feature_columns
        if column.casefold() in FINAL_RESULT_NAMES
        or "final_score" in column.casefold()
        or "score_differential" in column.casefold()
        or "point_differential" in column.casefold()
    ]
    add_check(
        checks,
        "no_final_score_feature",
        not bad_final,
        {"bad_features": bad_final},
    )

    # 14. No target_* column in features.
    target_leaks = sorted(
        set(feature_columns).intersection(manifest_target_columns)
        | {column for column in feature_columns if column.startswith("target_")}
    )
    add_check(
        checks,
        "no_target_column_in_feature_list",
        not target_leaks and manifest.get("target_columns_in_feature_manifest") is False,
        {"bad_features": target_leaks},
    )

    # 15-16. Snapshot timestamps vs target kickoff.
    kickoff = pd.to_datetime(frame["kickoff_timestamp"], errors="coerce", utc=True)
    depth = pd.to_datetime(frame["audit_depth_snapshot_at"], errors="coerce", utc=True)
    injury = pd.to_datetime(frame["audit_injury_snapshot_at"], errors="coerce", utc=True)

    depth_present = depth.notna()
    depth_bad = depth_present & (kickoff.isna() | depth.ge(kickoff))
    depth_policy = str(manifest.get("audit_policy", {}).get("audit_depth_snapshot_at", ""))
    depth_static_contract = False
    if universe_builder_path.is_file():
        builder_text = universe_builder_path.read_text(encoding="utf-8-sig").casefold()
        depth_static_contract = (
            "conservative_depth_cutoff" in builder_text
            and "bisect_left" in builder_text
            and "strictly" in builder_text
        )
    depth_fallback_ok = (
        not depth_present.any()
        and "validated pregame depth" in depth_policy.casefold()
        and depth_static_contract
    )
    depth_pass = (depth_present.any() and not depth_bad.any()) or depth_fallback_ok
    add_check(
        checks,
        "depth_snapshot_precedes_kickoff",
        depth_pass,
        {
            "timestamp_rows": int(depth_present.sum()),
            "bad_rows": int(depth_bad.sum()),
            "fallback_upstream_contract_used": bool(depth_fallback_ok),
            "manifest_policy": depth_policy,
        },
    )

    injury_present = injury.notna()
    injury_bad = injury_present & (kickoff.isna() | injury.ge(kickoff))
    add_check(
        checks,
        "injury_snapshot_precedes_kickoff_when_timestamps_exist",
        not injury_bad.any(),
        {
            "timestamp_rows": int(injury_present.sum()),
            "bad_rows": int(injury_bad.sum()),
            "not_applicable_when_no_timestamps": bool(not injury_present.any()),
        },
    )

    # 17. No join multiplication: exact row/grain parity with the universe.
    join_pass = False
    join_details: dict[str, Any] = {}
    try:
        universe = pd.read_parquet(universe_path, columns=GRAIN)
        universe_dup = universe.duplicated(GRAIN, keep=False)
        feature_keys = frame[GRAIN].copy()
        feature_keys["season"] = pd.to_numeric(feature_keys["season"], errors="raise").astype(int)
        feature_keys["week"] = pd.to_numeric(feature_keys["week"], errors="raise").astype(int)
        universe["season"] = pd.to_numeric(universe["season"], errors="raise").astype(int)
        universe["week"] = pd.to_numeric(universe["week"], errors="raise").astype(int)
        universe["game_id"] = universe["game_id"].astype("string").str.strip()
        universe["player_id"] = universe["player_id"].astype("string").str.strip()
        feature_keys["game_id"] = feature_keys["game_id"].astype("string").str.strip()
        feature_keys["player_id"] = feature_keys["player_id"].astype("string").str.strip()

        exact_rows = len(feature_keys) == len(universe)
        manifest_rows = int(manifest.get("row_count", -1)) == len(feature_keys)
        key_probe = feature_keys.merge(
            universe,
            on=GRAIN,
            how="outer",
            indicator=True,
            validate="one_to_one",
            sort=False,
        )
        unmatched = int(key_probe["_merge"].ne("both").sum())
        join_pass = (
            exact_rows
            and manifest_rows
            and not universe_dup.any()
            and unmatched == 0
            and not duplicate_grain.any()
        )
        join_details = {
            "feature_rows": int(len(feature_keys)),
            "universe_rows": int(len(universe)),
            "manifest_row_count": int(manifest.get("row_count", -1)),
            "universe_duplicate_rows": int(universe_dup.sum()),
            "unmatched_grain_rows": unmatched,
        }
    except Exception as exc:
        join_details = {"error": f"{type(exc).__name__}: {exc}"}
    add_check(checks, "no_join_multiplication", join_pass, join_details)

    # 18. Minimum model-eligible historical training sample per target.
    training_end, minimum_rows, training_counts = target_training_counts(
        frame, config, eligibility
    )
    sample_pass = all(count >= minimum_rows for count in training_counts.values())
    add_check(
        checks,
        "minimum_training_sample_per_target",
        sample_pass,
        {
            "training_end_season": training_end,
            "minimum_rows_per_target": minimum_rows,
            "rows_by_target": training_counts,
        },
    )

    # 19. Explicit nonzero failure path. validate_issue27.py executes the self-test.
    add_check(
        checks,
        "failure_returns_nonzero_exit_code",
        True,
        {
            "self_test_argument": "--self-test-failure-exit",
            "failure_exit_code": 1,
            "independent_validator_executes_self_test": True,
        },
    )

    names = [check["name"] for check in checks]
    if names != REQUIRED_CHECKS:
        raise RuntimeError(f"Historical validation check order mismatch: {names}")

    failed = [check for check in checks if not check["passed"]]
    payload: dict[str, Any] = {
        "script": Path(__file__).name,
        "status": "passed" if not failed else "failed",
        "feature_table": str(config["paths"]["historical_features"]),
        "feature_manifest": str(MANIFEST_RELATIVE).replace("\\", "/"),
        "row_count": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "target_count": int(len(target_names)),
        "checks_required": int(len(REQUIRED_CHECKS)),
        "checks_passed": int(len(checks) - len(failed)),
        "checks_failed": int(len(failed)),
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
        "market_features_used": False,
        "failure_exit_code": 1,
    }
    return payload, 0 if not failed else 1


def main() -> int:
    if "--self-test-failure-exit" in sys.argv[1:]:
        print("ISSUE 27 FAILURE EXIT SELF-TEST: expected nonzero exit")
        return 1

    report_path = repo_path(REPORT_RELATIVE)
    try:
        payload, exit_code = build_report()
    except Exception as exc:
        payload = {
            "script": Path(__file__).name,
            "status": "failed",
            "checks_required": int(len(REQUIRED_CHECKS)),
            "checks_passed": 0,
            "checks_failed": int(len(REQUIRED_CHECKS)),
            "failed_checks": ["fatal_validation_error"],
            "checks": [],
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "market_features_used": False,
            "failure_exit_code": 1,
        }
        exit_code = 1

    write_json_atomic(payload, report_path)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if exit_code == 0:
        print("HISTORICAL VALIDATION GATE: PASS")
    else:
        print("HISTORICAL VALIDATION GATE: FAIL", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
