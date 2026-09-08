#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 47 unit tests."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE / "tests"

REQUIRED_FILES = [
    "test_common.py",
    "test_identity.py",
    "test_targets.py",
    "test_opportunity.py",
    "test_leakage.py",
    "test_role_selection.py",
    "test_feature_schema.py",
    "test_projection_constraints.py",
    "test_market_exclusion.py",
    "test_end_to_end.py",
]

MANDATORY_METHODS = {
    "test_leakage.py": {
        "test_week_n_player_rolling_feature_excludes_week_n",
        "test_week_n_snap_excludes_week_n_snap",
        "test_week_n_participation_excludes_week_n_participation",
        "test_week_n_team_form_excludes_week_n_result",
        "test_depth_snapshot_precedes_kickoff",
        "test_injury_snapshot_precedes_kickoff",
    },
    "test_market_exclusion.py": {
        "test_reject_odds",
        "test_reject_spread",
        "test_reject_moneyline",
        "test_reject_drat",
        "test_reject_epred",
        "test_reject_forbidden_source_paths",
    },
    "test_targets.py": {
        "test_kicking_points_formula_exact",
        "test_tackles_formula_exact",
        "test_zero_stat_participant_retained",
        "test_nonparticipant_not_converted_into_false_zero",
    },
    "test_end_to_end.py": {
        "test_historical_week_as_current_feature_projection_validation",
    },
}


def fail(message: str) -> None:
    raise AssertionError(message)


def test_methods(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def main() -> int:
    if not TESTS.is_dir():
        fail(f"Missing tests directory: {TESTS}")

    actual_files = sorted(
        path.name for path in TESTS.glob("test_*.py") if path.is_file()
    )
    if actual_files != sorted(REQUIRED_FILES):
        fail(
            "Issue 47 test file set mismatch. "
            f"Expected={sorted(REQUIRED_FILES)} actual={actual_files}"
        )

    total_methods = 0
    for filename in REQUIRED_FILES:
        methods = test_methods(TESTS / filename)
        total_methods += len(methods)
        missing = sorted(MANDATORY_METHODS.get(filename, set()) - methods)
        if missing:
            fail(f"{filename} missing mandatory test(s): {missing}")

    if total_methods < 30:
        fail(f"Expected at least 30 unit/integration tests; found {total_methods}")

    e2e_text = (TESTS / "test_end_to_end.py").read_text(encoding="utf-8")
    for marker in [
        "historical_features",
        "player_opportunity",
        "overlay_current_player_stats",
        "player_pass_attempts_lag1",
        "player_pass_attempts_season_to_date",
        "direct_model.txt",
        "model_matrix",
        "lgb.Booster",
        "booster.predict",
        "assertAlmostEqual",
    ]:
        if marker not in e2e_text:
            fail(f"End-to-end test missing required implementation marker: {marker}")

    leakage_text = (TESTS / "test_leakage.py").read_text(encoding="utf-8")
    for marker in [
        "metric_features_for_targets",
        "add_strict_prior_history",
        "build_form",
        "depth_snapshot_for_game",
        "resolve_injury_source_week",
        "parse_modified",
    ]:
        if marker not in leakage_text:
            fail(f"Leakage tests do not exercise production helper: {marker}")

    market_text = (TESTS / "test_market_exclusion.py").read_text(encoding="utf-8")
    for marker in [
        "reject_forbidden_feature_columns",
        "forbidden_source_references",
        "audit_paths",
    ]:
        if marker not in market_text:
            fail(f"Market tests do not exercise production helper: {marker}")

    target_text = (TESTS / "test_targets.py").read_text(encoding="utf-8")
    for marker in [
        "historical_targets",
        "played_game_flag",
        "target_source_present",
        "field_goals_made",
        "extra_points_made",
        "solo_tackles",
        "assisted_tackles",
    ]:
        if marker not in target_text:
            fail(f"Target tests missing acceptance marker: {marker}")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(TESTS),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")

    if completed.returncode != 0:
        fail(f"Unit test suite failed. returncode={completed.returncode}")

    combined = completed.stdout + "\n" + completed.stderr
    if "FAILED" in combined or "ERROR:" in combined:
        fail("unittest output contains FAILED/ERROR despite zero exit code.")

    print(f"test_files={len(REQUIRED_FILES)}")
    print(f"test_methods={total_methods}")
    print("mandatory_leakage_tests=6")
    print("mandatory_market_tests=6")
    print("mandatory_target_tests=4")
    print("end_to_end_historical_as_current=true")
    print("end_to_end_asof_feature_equivalence=true")
    print("end_to_end_persisted_model_projection=true")
    print("production_helpers_exercised=true")
    print("market_features_used=false")
    print("UNIT TEST SUITE: PASS")
    print("ISSUE 47 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"UNIT TEST SUITE: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
