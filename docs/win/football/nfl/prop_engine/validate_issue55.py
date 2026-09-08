#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPECTED_FEATURE_CONFIGS = {
    "kicking_points.json",
    "passing_tds.json",
    "passing_yards.json",
    "receiving_tds.json",
    "receiving_yards.json",
    "rushing_tds.json",
    "rushing_yards.json",
    "sacks.json",
    "tackles.json",
}

REQUIRED_CONFIG = [
    "config/prop_engine.yaml",
    "config/target_eligibility.yaml",
    "config/fallback_rules.yaml",
    "config/acceptance_thresholds.yaml",
]

REQUIRED_BUILD = [
    "scripts/build/build_player_identity.py",
    "scripts/build/refresh_nflverse_player_data.py",
    "scripts/build/build_historical_universe.py",
    "scripts/build/build_targets.py",
    "scripts/build/build_player_opportunity.py",
    "scripts/build/build_team_opportunity.py",
    "scripts/build/build_position_allowed.py",
    "scripts/build/build_role_history.py",
    "scripts/build/build_player_form.py",
    "scripts/build/build_team_form.py",
    "scripts/build/build_environment_history.py",
    "scripts/build/build_defensive_features.py",
    "scripts/build/build_kicking_features.py",
    "scripts/build/build_historical_features.py",
]

REQUIRED_VALIDATORS = [
    "scripts/validate/audit_market_exclusion.py",
    "scripts/validate/validate_source_quality.py",
    "scripts/validate/validate_historical_data.py",
    "scripts/validate/validate_week.py",
]

REQUIRED_TRAIN = [
    "scripts/train/build_backtest_folds.py",
    "scripts/train/train_baselines.py",
    "scripts/train/train_opportunity_models.py",
    "scripts/train/train_efficiency_models.py",
    "scripts/train/train_direct_models.py",
    "scripts/train/select_model_architecture.py",
    "scripts/train/calibrate_uncertainty.py",
    "scripts/report/build_model_report.py",
]

REQUIRED_CURRENT = [
    "scripts/project/build_week1_priors.py",
    "scripts/project/build_current_universe.py",
    "scripts/project/select_roles.py",
    "scripts/project/build_current_features.py",
    "scripts/project/project_components.py",
    "scripts/project/allocate_team_opportunity.py",
    "scripts/project/project_direct.py",
    "scripts/project/project_week.py",
    "scripts/report/build_wide_output.py",
]

HISTORICAL_PIPELINE = [
    "build/build_player_identity.py",
    "build/build_historical_universe.py",
    "build/build_targets.py",
    "build/build_player_opportunity.py",
    "build/build_team_opportunity.py",
    "build/build_position_allowed.py",
    "build/build_role_history.py",
    "build/build_player_form.py",
    "build/build_team_form.py",
    "build/build_environment_history.py",
    "build/build_defensive_features.py",
    "build/build_kicking_features.py",
    "build/build_historical_features.py",
    "validate/audit_market_exclusion.py",
    "validate/validate_historical_data.py",
]

TRAINING_PIPELINE = [
    "validate/audit_market_exclusion.py",
    "validate/validate_historical_data.py",
    "train/build_backtest_folds.py",
    "train/train_baselines.py",
    "train/train_opportunity_models.py",
    "train/train_efficiency_models.py",
    "train/train_direct_models.py",
    "train/select_model_architecture.py",
    "train/calibrate_uncertainty.py",
    "report/build_model_report.py",
]

WEEKLY_PIPELINE = [
    "build/refresh_nflverse_player_data.py",
    "build/build_player_identity.py",
    "validate/audit_market_exclusion.py",
    "validate/validate_source_quality.py",
    "project/build_current_universe.py",
    "project/select_roles.py",
    "project/build_week1_priors.py",
    "project/build_current_features.py",
    "project/project_components.py",
    "project/allocate_team_opportunity.py",
    "project/project_direct.py",
    "project/project_week.py",
    "report/build_wide_output.py",
    "validate/validate_week.py",
]

REQUIRED_REQUIREMENTS = {
    "pandas",
    "numpy",
    "pyyaml",
    "pyarrow",
    "lightgbm",
    "nflreadpy",
    "pytest",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def require_files(paths: list[str], label: str) -> None:
    missing = [p for p in paths if not (HERE / p).is_file()]
    if missing:
        fail(f"{label} missing: {missing}")


def extract_pipeline(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        target_names: list[str] = []
        value = None

        if isinstance(node, ast.Assign):
            target_names = [
                t.id for t in node.targets if isinstance(t, ast.Name)
            ]
            value = node.value
        else:
            if isinstance(node.target, ast.Name):
                target_names = [node.target.id]
            value = node.value

        if "PIPELINE" not in target_names or value is None:
            continue

        literal = ast.literal_eval(value)
        if not isinstance(literal, (tuple, list)):
            fail(f"PIPELINE in {path} is not tuple/list")
        return [str(x) for x in literal]

    fail(f"PIPELINE not found in {path}")


def validate_sequence_document() -> None:
    path = HERE / "IMPLEMENTATION_SEQUENCE.md"
    if not path.is_file():
        fail("IMPLEMENTATION_SEQUENCE.md missing")

    text = path.read_text(encoding="utf-8-sig")
    matches = re.findall(r"(?m)^- \*\*(55\.\d+)\*\* ", text)
    expected = [f"55.{i}" for i in range(1, 58)]

    if matches != expected:
        fail(
            "Implementation sequence IDs are not exactly 55.1 through 55.57 "
            f"in order. found={matches}"
        )

    required_phrases = [
        "Stages 55.51–55.57 are execution and acceptance gates",
        "--allow-unapproved-models",
        "must run without the unapproved-model override",
        "no unavailable rows may be fabricated",
        "`train/select_model_architecture.py`",
        "`report/build_model_report.py`",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            fail(f"IMPLEMENTATION_SEQUENCE.md missing contract phrase: {phrase}")

    forbidden_phrases = [
        "`train/select_architecture.py`",
        "`train/build_model_report.py`",
    ]
    for phrase in forbidden_phrases:
        if phrase in text:
            fail(f"IMPLEMENTATION_SEQUENCE.md contains wrong path: {phrase}")


def validate_requirements() -> None:
    path = HERE / "requirements.txt"
    if not path.is_file():
        fail("requirements.txt missing")

    packages: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[\]; ]", line, maxsplit=1)[0].strip().casefold()
        if name:
            packages.add(name)

    missing = sorted(REQUIRED_REQUIREMENTS - packages)
    if missing:
        fail(f"requirements.txt missing direct dependency/dependencies: {missing}")


def main() -> int:
    print("CHECK 01: bootstrap/config/common requirements")
    require_files(REQUIRED_CONFIG + ["scripts/common.py"], "bootstrap")

    print("CHECK 02: exact nine target feature configs")
    feature_root = HERE / "config" / "features"
    if not feature_root.is_dir():
        fail("config/features directory missing")
    actual_features = {p.name for p in feature_root.glob("*.json")}
    if actual_features != EXPECTED_FEATURE_CONFIGS:
        fail(
            "Feature config set mismatch. "
            f"expected={sorted(EXPECTED_FEATURE_CONFIGS)} "
            f"actual={sorted(actual_features)}"
        )

    print("CHECK 03: historical build implementation")
    require_files(REQUIRED_BUILD + REQUIRED_VALIDATORS[:3], "historical implementation")
    require_files(["scripts/run_historical_build.py"], "historical runner")
    actual = extract_pipeline(HERE / "scripts/run_historical_build.py")
    if actual != HISTORICAL_PIPELINE:
        fail(f"Historical pipeline order mismatch: {actual}")

    print("CHECK 04: training implementation")
    require_files(REQUIRED_TRAIN + ["scripts/run_training.py"], "training implementation")
    actual = extract_pipeline(HERE / "scripts/run_training.py")
    if actual != TRAINING_PIPELINE:
        fail(f"Training pipeline order mismatch: {actual}")

    print("CHECK 05: current-week implementation")
    require_files(
        REQUIRED_CURRENT + REQUIRED_VALIDATORS[3:] + ["scripts/run_weekly.py"],
        "weekly implementation",
    )
    actual = extract_pipeline(HERE / "scripts/run_weekly.py")
    if actual != WEEKLY_PIPELINE:
        fail(f"Weekly pipeline order mismatch: {actual}")

    print("CHECK 06: production registry")
    if not (HERE / "models" / "production_registry.json").is_file():
        fail("models/production_registry.json missing")

    print("CHECK 07: test suite exists")
    tests = sorted((HERE / "tests").glob("test_*.py"))
    if not tests:
        fail("No tests/test_*.py files found")

    print("CHECK 08: requirements and execution documentation")
    validate_requirements()
    validate_sequence_document()

    print("sequence_steps=57")
    print("feature_configs=9")
    print(f"historical_runner_stages={len(HISTORICAL_PIPELINE)}")
    print(f"training_runner_stages={len(TRAINING_PIPELINE)}")
    print(f"weekly_runner_stages={len(WEEKLY_PIPELINE)}")
    print(f"test_files={len(tests)}")
    print("requirements_file=true")
    print("execution_sequence_document=true")
    print("heavy_execution_gates_rerun=false")
    print("ISSUE 55 EXECUTION SEQUENCE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ISSUE 55 EXECUTION SEQUENCE VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
