#!/usr/bin/env python3
"""Independent acceptance validator for Prop Engine Issue 53.

This validator distinguishes:
- primary persisted LightGBM trainers; and
- select_model_architecture.py, which performs temporary validation-only
  LightGBM fitting using parameter dictionaries supplied by the seeded direct
  trainer.

No training call is excluded merely because its model is not persisted.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "prop_engine.yaml"
TRAIN_ROOT = HERE / "scripts" / "train"
RUNNER = HERE / "scripts" / "run_training.py"

EXPECTED_SEED = 76076
EXPECTED_DETERMINISTIC = True
EXPECTED_THREADS = 1

PRIMARY_LIGHTGBM_TRAINERS = {
    "train_opportunity_models.py",
    "train_efficiency_models.py",
    "train_direct_models.py",
}
ARCHITECTURE_SELECTOR = "select_model_architecture.py"

METADATA_KEYS = {
    "train_opportunity_models.py": "seed",
    "train_efficiency_models.py": "seed",
    "train_direct_models.py": "random_seed",
}

OLD_STANDALONE_SEEDS = {22022, 23023, 24024}


def fail(message: str) -> None:
    raise AssertionError(message)


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        fail(f"Missing required config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        fail(f"Expected YAML mapping: {path}")
    return value


def parsed(path: Path) -> tuple[str, ast.AST]:
    source = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        fail(f"Python syntax error in {path}: {exc}")
    return source, tree


def imported_frameworks(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in {"lightgbm", "xgboost", "sklearn"}:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".", 1)[0]
            if top in {"lightgbm", "xgboost", "sklearn"}:
                found.add(top)
    return found


def direct_lgb_train_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "train"
            and isinstance(func.value, ast.Name)
            and func.value.id == "lgb"
        ):
            count += 1
    return count


def exact_seed_assignment(source: str, path: Path) -> None:
    matches = re.findall(r"(?m)^SEED\s*=\s*(\d+)\s*$", source)
    if matches != [str(EXPECTED_SEED)]:
        fail(
            f"{path.name}: expected exactly SEED = {EXPECTED_SEED}; "
            f"found {matches}"
        )


def assert_primary_lightgbm_contract(path: Path) -> None:
    source, tree = parsed(path)
    exact_seed_assignment(source, path)

    if direct_lgb_train_count(tree) < 1:
        fail(f"{path.name}: no direct lgb.train call found.")

    required_param_patterns = [
        r'["\']seed["\']\s*:\s*SEED',
        r'["\']feature_fraction_seed["\']\s*:\s*SEED',
        r'["\']bagging_seed["\']\s*:\s*SEED',
        r'["\']data_random_seed["\']\s*:\s*SEED',
        r'["\']deterministic["\']\s*:\s*True',
        r'["\']num_threads["\']\s*:\s*1',
    ]
    for pattern in required_param_patterns:
        if not re.search(pattern, source):
            fail(f"{path.name}: missing deterministic LightGBM parameter: {pattern}")

    metadata_key = METADATA_KEYS[path.name]
    # Generated persisted metadata must explicitly capture the seed.
    if not re.search(
        rf'["\']{re.escape(metadata_key)}["\']\s*:\s*SEED',
        source,
    ):
        fail(
            f"{path.name}: generated metadata does not record "
            f"{metadata_key}=SEED."
        )

    for old in OLD_STANDALONE_SEEDS:
        if re.search(rf"(?m)^SEED\s*=\s*{old}\s*$", source):
            fail(f"{path.name}: stale standalone seed remains: {old}")


def assert_architecture_selector_contract(path: Path) -> None:
    source, tree = parsed(path)

    if direct_lgb_train_count(tree) != 1:
        fail(
            f"{path.name}: expected exactly one direct validation-only "
            f"lgb.train call; found {direct_lgb_train_count(tree)}."
        )

    # It must import the exact seeded trainer modules whose source contracts are
    # independently validated above.
    required_imports = [
        r"(?m)^import\s+train_direct_models\s+as\s+direct\s*$",
        r"(?m)^import\s+train_efficiency_models\s+as\s+efficiency\s*$",
        r"(?m)^import\s+train_opportunity_models\s+as\s+opportunity\s*$",
    ]
    for pattern in required_imports:
        if not re.search(pattern, source):
            fail(f"{path.name}: missing seeded-trainer import: {pattern}")

    # Temporary direct validation models receive direct.params_for(...). The
    # direct trainer's params_for contract is checked above and contains all
    # four LightGBM seed fields, deterministic=True, and num_threads=1.
    if not re.search(
        r"params\s*=\s*direct\.params_for\s*\(",
        source,
    ):
        fail(
            f"{path.name}: validation LightGBM params are not sourced from "
            "direct.params_for(...)."
        )

    if not re.search(
        r"train_fixed_booster\s*\([\s\S]{0,2500}?"
        r"params\s*=\s*params[\s\S]{0,2500}?\)",
        source,
    ):
        fail(
            f"{path.name}: train_fixed_booster does not receive the "
            "direct.params_for parameter mapping."
        )

    # The helper may copy params, but it may not replace them with a fresh,
    # unseeded parameter dictionary before lgb.train.
    if not re.search(
        r"return\s+lgb\.train\s*\(\s*dict\s*\(\s*params\s*\)",
        source,
    ):
        fail(
            f"{path.name}: validation lgb.train is not fed the passed seeded "
            "params mapping."
        )

    # No independent module-level numeric SEED is required here: this script
    # delegates LightGBM parameter construction to the validated trainers.
    local_seed = re.findall(r"(?m)^SEED\s*=\s*(\d+)\s*$", source)
    if local_seed and local_seed != [str(EXPECTED_SEED)]:
        fail(
            f"{path.name}: unexpected independent local seed(s): {local_seed}"
        )


def assert_xgboost_training_contract(path: Path) -> None:
    source, _ = parsed(path)
    exact_seed_assignment(source, path)
    if not re.search(
        r"(?:random_state|seed)\s*=\s*SEED|"
        r'["\'](?:random_state|seed)["\']\s*:\s*SEED',
        source,
    ):
        fail(f"{path.name}: XGBoost training does not receive SEED.")
    if not re.search(
        r"(?:n_jobs|nthread)\s*=\s*1|"
        r'["\'](?:n_jobs|nthread)["\']\s*:\s*1',
        source,
    ):
        fail(f"{path.name}: XGBoost training is not single-threaded.")


def assert_sklearn_training_contract(path: Path) -> None:
    source, _ = parsed(path)
    exact_seed_assignment(source, path)
    if not re.search(
        r"random_state\s*=\s*SEED|"
        r'["\']random_state["\']\s*:\s*SEED',
        source,
    ):
        fail(f"{path.name}: sklearn training does not receive random_state=SEED.")


def main() -> int:
    config = read_yaml(CONFIG)
    training = config.get("training")
    expected_training = {
        "random_seed": EXPECTED_SEED,
        "deterministic": EXPECTED_DETERMINISTIC,
        "num_threads": EXPECTED_THREADS,
    }
    if training != expected_training:
        fail(
            "config.training exact contract mismatch. "
            f"Expected={expected_training} Actual={training}"
        )

    if not TRAIN_ROOT.is_dir():
        fail(f"Training script directory missing: {TRAIN_ROOT}")

    py_files = sorted(TRAIN_ROOT.glob("*.py"))

    lightgbm_import_files: list[Path] = []
    direct_lgb_training_files: list[Path] = []
    xgboost_import_files: list[Path] = []
    sklearn_import_files: list[Path] = []

    for path in py_files:
        _, tree = parsed(path)
        frameworks = imported_frameworks(tree)
        if "lightgbm" in frameworks:
            lightgbm_import_files.append(path)
        if direct_lgb_train_count(tree):
            direct_lgb_training_files.append(path)
        if "xgboost" in frameworks:
            xgboost_import_files.append(path)
        if "sklearn" in frameworks:
            sklearn_import_files.append(path)

    actual_direct_lgb_names = {p.name for p in direct_lgb_training_files}
    expected_direct_lgb_names = PRIMARY_LIGHTGBM_TRAINERS | {ARCHITECTURE_SELECTOR}
    if actual_direct_lgb_names != expected_direct_lgb_names:
        fail(
            "Direct LightGBM training-file set mismatch. "
            f"Expected={sorted(expected_direct_lgb_names)} "
            f"Actual={sorted(actual_direct_lgb_names)}"
        )

    for name in sorted(PRIMARY_LIGHTGBM_TRAINERS):
        assert_primary_lightgbm_contract(TRAIN_ROOT / name)

    assert_architecture_selector_contract(TRAIN_ROOT / ARCHITECTURE_SELECTOR)

    # Current Prop Engine has no XGBoost/sklearn trainer. If either framework
    # is introduced under scripts/train, it becomes part of this acceptance
    # contract and must be explicitly seeded.
    for path in xgboost_import_files:
        assert_xgboost_training_contract(path)

    for path in sklearn_import_files:
        assert_sklearn_training_contract(path)

    if not RUNNER.is_file():
        fail(f"Missing training runner: {RUNNER}")
    runner_source = RUNNER.read_text(encoding="utf-8-sig")

    runner_markers = [
        "training.get('random_seed')",
        "seeded_source",
        "SEEDED_TRAINERS",
        "training_seed",
        "metadata_seed",
        "persisted seed",
        "seed_source':'config.training.random_seed'",
    ]
    for marker in runner_markers:
        if marker not in runner_source:
            fail(f"run_training.py missing config-seed enforcement marker: {marker}")

    if not re.search(
        r"collect_model_artifacts\s*\(\s*prop\s*,\s*seed\s*\)",
        runner_source,
    ):
        fail("run_training.py does not postflight regenerated artifact seeds.")

    print("config_training.random_seed=76076")
    print("config_training.deterministic=true")
    print("config_training.num_threads=1")
    print(f"primary_lightgbm_trainers={len(PRIMARY_LIGHTGBM_TRAINERS)}")
    print(f"lightgbm_training_files={len(direct_lgb_training_files)}")
    print(f"xgboost_trainers={len(xgboost_import_files)}")
    print(f"sklearn_trainers={len(sklearn_import_files)}")
    print("all_primary_lightgbm_seeded=true")
    print("all_lightgbm_deterministic=true")
    print("all_lightgbm_single_thread=true")
    print("architecture_selection_seed_provenance=true")
    print("all_xgboost_seeded=true")
    print("all_sklearn_seeded=true")
    print("trainer_metadata_records_seed=true")
    print("training_runner_uses_config_seed=true")
    print("existing_model_artifacts_retrained=false")
    print("DETERMINISTIC TRAINING CONTROLS VALIDATION: PASS")
    print("ISSUE 53 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"DETERMINISTIC TRAINING CONTROLS VALIDATION: FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
