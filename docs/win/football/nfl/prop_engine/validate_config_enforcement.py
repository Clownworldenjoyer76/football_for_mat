#!/usr/bin/env python3
"""Independent P0 validator for shared Prop Engine configuration enforcement."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

EXPECTED_TARGETS = [
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

SIGNED_YARDAGE = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
}

EXPECTED_SEASONS = {
    "historical_start": 2012,
    "participation_feature_start": 2016,
    "rich_feature_start": 2021,
    "historical_end": 2025,
    "current": 2026,
}

EXPECTED_TRAINING = {
    "model_selection_train_end_season": 2023,
    "development_validation_season": 2024,
    "final_train_end_season": 2024,
    "untouched_test_season": 2025,
}

PRODUCTION_DIRS = [
    HERE / "scripts/build",
    HERE / "scripts/train",
    HERE / "scripts/project",
    HERE / "scripts/validate",
    HERE / "scripts/report",
]

ROOT_PRODUCTION_FILES = [
    HERE / "scripts/run_historical_build.py",
    HERE / "scripts/run_training.py",
    HERE / "scripts/run_weekly.py",
]

SPLIT_TRAINERS = [
    HERE / "scripts/train/train_opportunity_models.py",
    HERE / "scripts/train/train_efficiency_models.py",
    HERE / "scripts/train/train_direct_models.py",
]

MANIFEST_ENFORCERS = {
    HERE / "scripts/train/train_opportunity_models.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/train/train_efficiency_models.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/train/train_direct_models.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/train/select_model_architecture.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/project/build_current_features.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/project/project_components.py":
        "common.reject_forbidden_feature_columns",
    HERE / "scripts/project/project_direct.py":
        "common.reject_forbidden_feature_columns",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Required JSON missing: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(value, dict):
        fail(f"Expected JSON object: {path}")
    return value


def source(path: Path) -> str:
    if not path.is_file():
        fail(f"Required source missing: {path}")
    text = path.read_text(encoding="utf-8-sig")
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        fail(f"Python syntax failure in {path}: {exc}")
    return text


def production_scripts() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT_PRODUCTION_FILES:
        if not path.is_file():
            fail(f"Production entrypoint missing: {path}")
        paths.append(path)
    for directory in PRODUCTION_DIRS:
        if not directory.is_dir():
            fail(f"Production directory missing: {directory}")
        paths.extend(sorted(directory.glob("*.py")))
    return paths


def top_level_assignment(
    tree: ast.Module,
    name: str,
) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == name
                for target in node.targets
            ):
                return node
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                return node
    return None


def selected_features(
    payload: dict[str, Any],
) -> list[str]:
    for key in (
        "selected_features",
        "feature_columns",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]

    numeric = payload.get("numeric_features")
    categorical = payload.get(
        "categorical_features"
    )
    if (
        isinstance(numeric, list)
        or isinstance(categorical, list)
    ):
        return [
            *[
                str(item)
                for item in (numeric or [])
            ],
            *[
                str(item)
                for item in (categorical or [])
            ],
        ]

    return []


def manifest_paths() -> list[Path]:
    paths: set[Path] = set()

    paths.update(
        (HERE / "config/features").glob(
            "*.json"
        )
    )
    paths.update(
        (HERE / "models/components").glob(
            "*/feature_manifest.json"
        )
    )
    paths.update(
        (HERE / "models/efficiency").glob(
            "*/feature_manifest.json"
        )
    )

    for target in EXPECTED_TARGETS:
        path = (
            HERE
            / "models"
            / target
            / "feature_manifest.json"
        )
        if path.is_file():
            paths.add(path)

    canonical = (
        HERE
        / "data/historical/features/"
          "feature_manifest.json"
    )
    if canonical.is_file():
        paths.add(canonical)

    paths.update(
        (HERE / "data/current/features").glob(
            "*_feature_manifest.json"
        )
    )

    return sorted(
        paths,
        key=lambda path: str(path).casefold(),
    )


def check_config() -> dict[str, Any]:
    config = common.load_config()

    if (
        config.get("system", {}).get(
            "market_data_allowed"
        )
        is not False
    ):
        fail(
            "system.market_data_allowed "
            "must be false."
        )

    seasons = config.get("seasons")
    if not isinstance(seasons, dict):
        fail(
            "config.seasons must be a mapping."
        )

    for key, expected in (
        EXPECTED_SEASONS.items()
    ):
        if seasons.get(key) != expected:
            fail(
                f"seasons.{key}="
                f"{seasons.get(key)!r}; "
                f"expected {expected!r}"
            )

    training = config.get("training")
    if not isinstance(training, dict):
        fail(
            "config.training must be a mapping."
        )

    for key, expected in (
        EXPECTED_TRAINING.items()
    ):
        if training.get(key) != expected:
            fail(
                f"training.{key}="
                f"{training.get(key)!r}; "
                f"expected {expected!r}"
            )

    if (
        training["untouched_test_season"]
        != seasons["historical_end"]
    ):
        fail(
            "training.untouched_test_season "
            "must equal seasons.historical_end."
        )

    if (
        training["final_train_end_season"]
        >= training["untouched_test_season"]
    ):
        fail(
            "Final training must end before "
            "untouched test."
        )

    if (
        training[
            "development_validation_season"
        ]
        != training["final_train_end_season"]
    ):
        fail(
            "Development validation season "
            "must equal final train end season."
        )

    if (
        training[
            "model_selection_train_end_season"
        ]
        >= training[
            "development_validation_season"
        ]
    ):
        fail(
            "Model-selection training must "
            "end before development validation."
        )

    if not (
        seasons["historical_start"]
        < seasons[
            "participation_feature_start"
        ]
        <= seasons["rich_feature_start"]
        <= seasons["historical_end"]
        < seasons["current"]
    ):
        fail(
            "Configured season availability "
            "boundaries are nonchronological."
        )

    targets = config.get("targets")
    if (
        not isinstance(targets, dict)
        or list(targets)
        != EXPECTED_TARGETS
    ):
        actual = (
            list(targets)
            if isinstance(targets, dict)
            else targets
        )
        fail(
            "Configured targets differ from "
            "accepted nine-target order: "
            f"{actual!r}"
        )

    for target in SIGNED_YARDAGE:
        if (
            targets[target].get("type")
            != "continuous_signed"
        ):
            fail(
                f"{target} must remain "
                "continuous_signed."
            )

    if (
        targets["tackles"].get(
            "definition"
        )
        != "solo_tackles + assisted_tackles"
    ):
        fail(
            "Configured tackle definition "
            "changed."
        )

    if (
        targets["kicking_points"].get(
            "formula"
        )
        != (
            "3 * field_goals_made "
            "+ extra_points_made"
        )
    ):
        fail(
            "Configured kicking-points "
            "formula changed."
        )

    return config


def check_all_downstream_load_config() -> int:
    paths = production_scripts()
    violations: list[str] = []

    for path in paths:
        text = source(path)
        relative = str(
            path.relative_to(HERE)
        ).replace("\\", "/")

        if "import common" not in text:
            violations.append(
                f"{relative}: no import common"
            )
        elif "common.load_config(" not in text:
            violations.append(
                f"{relative}: "
                "no common.load_config()"
            )

    if violations:
        fail(
            "Downstream config-load violations: "
            + "; ".join(violations)
        )

    return len(paths)


def check_no_literal_target_universe() -> int:
    checked = 0
    candidates = [
        *production_scripts(),
        HERE / "scripts/run_training.py",
        HERE / "scripts/run_weekly.py",
    ]

    for path in candidates:
        if not path.is_file():
            continue

        text = source(path)
        tree = ast.parse(text)
        node = top_level_assignment(
            tree,
            "TARGETS",
        )
        if node is None:
            continue

        checked += 1

        try:
            value = ast.literal_eval(node.value)
        except Exception:
            segment = (
                ast.get_source_segment(
                    text,
                    node,
                )
                or ""
            )
            if "targets" not in segment.lower():
                fail(
                    f"{path.relative_to(HERE)}: "
                    "TARGETS is dynamic but "
                    "not config-derived."
                )
            continue

        if (
            isinstance(value, (list, tuple))
            and list(value)
            == EXPECTED_TARGETS
        ):
            fail(
                f"{path.relative_to(HERE)} "
                "retains an independent literal "
                "nine-target TARGETS list."
            )

    return checked


def check_split_trainer_sources() -> None:
    key_for = {
        "MODEL_SELECTION_TRAIN_END":
            "model_selection_train_end_season",
        "DEVELOPMENT_VALIDATION_SEASON":
            "development_validation_season",
        "FINAL_TRAIN_END":
            "final_train_end_season",
        "UNTOUCHED_TEST_SEASON":
            "untouched_test_season",
    }

    for path in SPLIT_TRAINERS:
        text = source(path)

        if (
            '_TRAINING_CONTRACT = '
            '_CONFIG_CONTRACT["training"]'
            not in text
        ):
            fail(
                f"{path.name}: no "
                "config.training contract alias."
            )

        tree = ast.parse(text)

        for variable, key in key_for.items():
            node = top_level_assignment(
                tree,
                variable,
            )
            if node is None:
                fail(
                    f"{path.name}: "
                    f"missing {variable}."
                )

            try:
                ast.literal_eval(node.value)
            except Exception:
                segment = (
                    ast.get_source_segment(
                        text,
                        node,
                    )
                    or ""
                )
                if (
                    "_TRAINING_CONTRACT"
                    not in segment
                    or key not in segment
                ):
                    fail(
                        f"{path.name}: "
                        f"{variable} is not "
                        f"derived from "
                        f"config.training.{key}."
                    )
            else:
                fail(
                    f"{path.name}: "
                    f"{variable} remains "
                    "an independent literal."
                )


def check_other_split_consumers() -> None:
    baselines = source(
        HERE
        / "scripts/train/train_baselines.py"
    )

    for marker in [
        "UNTOUCHED_TEST_SEASON",
        "FINAL_TRAIN_END",
        "DEVELOPMENT_VALIDATION_SEASON",
        "MODEL_SELECTION_TRAIN_END",
    ]:
        if marker not in baselines:
            fail(
                "train_baselines.py missing "
                f"config-derived {marker}."
            )

    for marker in [
        'validation_start_season"]) != 2025',
        'train_end_season"]) != 2024',
        'validation_start_season"].eq(2024)',
        "rows_2025",
        '"untouched_test_season": 2025',
    ]:
        if marker in baselines:
            fail(
                "train_baselines.py retains "
                f"hard-coded split source: {marker}"
            )

    folds = source(
        HERE
        / "scripts/train/"
          "build_backtest_folds.py"
    )
    if (
        "_CONFIG_ENFORCED_BACKTEST_POLICY"
        not in folds
    ):
        fail(
            "build_backtest_folds.py does "
            "not enforce config.training "
            "split boundaries."
        )

    selection = source(
        HERE
        / "scripts/train/"
          "select_model_architecture.py"
    )
    if (
        "_CONFIG_ENFORCED_SELECTION_POLICY"
        not in selection
    ):
        fail(
            "select_model_architecture.py "
            "does not enforce config.training "
            "split boundaries."
        )

    calibration = source(
        HERE
        / "scripts/train/"
          "calibrate_uncertainty.py"
    )
    if (
        "_CONFIG_ENFORCED_CALIBRATION_SPLIT"
        not in calibration
    ):
        fail(
            "calibrate_uncertainty.py does "
            "not enforce config-driven "
            "selected-model split contracts."
        )

    report = source(
        HERE
        / "scripts/report/"
          "build_model_report.py"
    )
    if (
        "_CONFIG_ENFORCED_REPORTING_SPLIT"
        not in report
    ):
        fail(
            "build_model_report.py does "
            "not enforce config-driven "
            "validation/test season."
        )


def check_participation_and_pbp(
    config: dict[str, Any],
) -> tuple[int, int]:
    player_opp_path = (
        HERE
        / "scripts/build/"
          "build_player_opportunity.py"
    )
    player_opp = source(player_opp_path)

    if (
        'config["seasons"]'
        '["participation_feature_start"]'
        not in player_opp
    ):
        fail(
            "build_player_opportunity.py "
            "does not read "
            "participation_feature_start."
        )

    if (
        'output["season"].lt(2016)'
        in player_opp
    ):
        fail(
            "build_player_opportunity.py "
            "still hard-codes participation "
            "cutoff 2016."
        )

    if (
        "season >= participation_feature_start"
        not in player_opp
    ):
        fail(
            "build_player_opportunity.py does "
            "not gate participation consumption "
            "at the configured start season."
        )

    for marker in [
        'config["seasons"]["rich_feature_start"]',
        'config["paths"]["pbp_pattern"]',
        "if season >= rich_feature_start:",
        "if not pbp_path.is_file():",
        "required_pbp_columns = [",
    ]:
        if marker not in player_opp:
            fail(
                "build_player_opportunity.py "
                "missing rich-PBP enforcement "
                f"marker: {marker}"
            )

    universe = source(
        HERE
        / "scripts/build/"
          "build_historical_universe.py"
    )

    if (
        'config["seasons"]'
        '["participation_feature_start"]'
        not in universe
    ):
        fail(
            "build_historical_universe.py "
            "does not read "
            "participation_feature_start."
        )

    if re.search(
        r"\bseason\s*<\s*2016\b",
        universe,
    ):
        fail(
            "build_historical_universe.py "
            "still hard-codes participation "
            "cutoff 2016."
        )

    required_header = {
        "season_type",
        "week",
        "game_id",
        "posteam",
        "yardline_100",
        "pass_attempt",
        "qb_dropback",
        "rush_attempt",
        "qb_kneel",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
    }

    repo = common.repo_root()
    rich_start = int(
        config["seasons"][
            "rich_feature_start"
        ]
    )
    historical_end = int(
        config["seasons"]["historical_end"]
    )
    pbp_checked = 0

    for season in range(
        rich_start,
        historical_end + 1,
    ):
        path = (
            repo
            / str(
                config["paths"]["pbp_pattern"]
            ).format(season=season)
        )

        if not path.is_file():
            fail(
                "Configured rich-feature "
                f"PBP file missing: {path}"
            )

        header = set(
            pd.read_csv(
                path,
                nrows=0,
            ).columns
        )
        missing = sorted(
            required_header - header
        )
        if missing:
            fail(
                f"{path}: rich-feature PBP "
                "header validation failed; "
                f"missing={missing}"
            )
        pbp_checked += 1

    return (
        int(
            config["seasons"][
                "participation_feature_start"
            ]
        ),
        pbp_checked,
    )


def check_target_builder() -> None:
    text = source(
        HERE
        / "scripts/build/build_targets.py"
    )

    for marker in [
        '_CONFIG_TARGETS = '
        '_CONFIG_CONTRACT["targets"]',
        'spec.get("source_column")',
        'spec.get("type") '
        '== "continuous_signed"',
        "SOLO_TACKLE_ALIASES",
        "ASSISTED_TACKLE_ALIASES",
        "def_tackles_with_assist",
        "solo_tackles+assisted_tackles",
        "3*field_goals_made"
        "+extra_points_made",
    ]:
        if marker not in text:
            fail(
                "build_targets.py missing "
                "config/target contract marker: "
                f"{marker}"
            )

    tree = ast.parse(text)

    for name in (
        "DIRECT_TARGETS",
        "SIGNED_YARDAGE_TARGETS",
    ):
        node = top_level_assignment(
            tree,
            name,
        )
        if node is None:
            fail(
                "build_targets.py missing "
                f"{name}."
            )

        try:
            ast.literal_eval(node.value)
        except Exception:
            segment = (
                ast.get_source_segment(
                    text,
                    node,
                )
                or ""
            )
            if "_CONFIG_TARGETS" not in segment:
                fail(
                    "build_targets.py "
                    f"{name} is dynamic but "
                    "not config-derived."
                )
        else:
            fail(
                "build_targets.py "
                f"{name} remains an "
                "independent literal source."
            )


def check_forbidden_enforcement(
    config: dict[str, Any],
) -> tuple[int, int]:
    forbidden = [
        str(item).strip()
        for item in config[
            "forbidden_features"
        ]
        if str(item).strip()
    ]

    if not forbidden:
        fail(
            "Configured forbidden_features "
            "is empty."
        )

    rejected = 0
    for token in forbidden:
        try:
            common.reject_forbidden_feature_columns(
                [
                    f"safe_prefix_"
                    f"{token}_safe_suffix"
                ],
                config,
            )
        except ValueError:
            rejected += 1
        else:
            fail(
                "Configured forbidden token "
                f"was not rejected: {token}"
            )

    for path, marker in (
        MANIFEST_ENFORCERS.items()
    ):
        text = source(path)
        if marker not in text:
            fail(
                f"{path.relative_to(HERE)} "
                "does not enforce forbidden "
                "feature manifests."
            )

    manifests = manifest_paths()
    if not manifests:
        fail(
            "No production/config feature "
            "manifests found."
        )

    checked = 0

    for path in manifests:
        payload = load_json(path)
        features = selected_features(
            payload
        )

        if features:
            common.reject_forbidden_feature_columns(
                features,
                config,
            )
            checked += 1

        if (
            "market_features_used" in payload
            and payload[
                "market_features_used"
            ]
            is not False
        ):
            fail(
                f"{path}: market_features_used "
                "is not false."
            )

    if checked == 0:
        fail(
            "No feature-bearing manifest "
            "was validated."
        )

    return rejected, checked


def main() -> int:
    config = check_config()

    downstream_count = (
        check_all_downstream_load_config()
    )
    target_assignments_checked = (
        check_no_literal_target_universe()
    )

    check_split_trainer_sources()
    check_other_split_consumers()
    check_target_builder()

    (
        participation_start,
        pbp_checked,
    ) = check_participation_and_pbp(
        config
    )

    (
        forbidden_tokens_rejected,
        manifests_checked,
    ) = check_forbidden_enforcement(
        config
    )

    print(
        "config_source="
        "docs/win/football/nfl/"
        "prop_engine/config/prop_engine.yaml"
    )
    print(
        f"configured_targets="
        f"{len(config['targets'])}"
    )
    print(
        "all_modeled_targets_"
        "derive_from_yaml=true"
    )
    print("historical_start=2012")
    print(
        "participation_feature_start="
        f"{participation_start}"
    )
    print("rich_feature_start=2021")
    print("historical_end=2025")
    print("current_season=2026")
    print(
        "model_selection_train_end_season="
        "2023"
    )
    print(
        "development_validation_season="
        "2024"
    )
    print(
        "final_train_end_season=2024"
    )
    print(
        "untouched_test_season=2025"
    )
    print(
        "downstream_production_scripts_"
        f"loading_config={downstream_count}"
    )
    print(
        "target_assignments_checked="
        f"{target_assignments_checked}"
    )
    print(
        "hardcoded_training_split_sources=0"
    )
    print(
        "signed_historical_yardage_"
        "preserved=true"
    )
    print(
        f"rich_pbp_seasons_validated="
        f"{pbp_checked}"
    )
    print(
        "configured_forbidden_tokens_"
        f"rejected={forbidden_tokens_rejected}"
    )
    print(
        f"feature_manifests_checked="
        f"{manifests_checked}"
    )
    print(
        "training_manifest_rejection_"
        "enforced=true"
    )
    print(
        "projection_manifest_rejection_"
        "enforced=true"
    )
    print(
        "CONFIG ENFORCEMENT VALIDATION: PASS"
    )
    print(
        "P0 CONFIG SOURCE-OF-TRUTH "
        "ACCEPTANCE: PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "CONFIG ENFORCEMENT VALIDATION: "
            f"FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
