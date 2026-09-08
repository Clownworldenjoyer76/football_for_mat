#!/usr/bin/env python3
"""Apply the P0 Prop Engine shared-configuration enforcement patch.

The patch is idempotent and writes only under
docs/win/football/nfl/prop_engine/.
"""

from __future__ import annotations

import ast
import os
import re
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "prop_engine.yaml"

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

DIRECT_SOURCE_TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]

SIGNED_YARDAGE_TARGETS = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
}

TRAINER_SPLIT_FILES = [
    HERE / "scripts/train/train_opportunity_models.py",
    HERE / "scripts/train/train_efficiency_models.py",
    HERE / "scripts/train/train_direct_models.py",
]

TARGET_ENUMERATION_FILES = [
    HERE / "scripts/run_training.py",
    HERE / "scripts/run_weekly.py",
    HERE / "scripts/train/train_baselines.py",
    HERE / "scripts/train/train_direct_models.py",
    HERE / "scripts/project/project_direct.py",
    HERE / "scripts/project/project_week.py",
    HERE / "scripts/report/build_wide_output.py",
]

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

TRAINING_VALUES = {
    "model_selection_train_end_season": 2023,
    "development_validation_season": 2024,
    "final_train_end_season": 2024,
    "untouched_test_season": 2025,
}

# These three keys were accidentally introduced by the first failed P0
# package. They belong to the later deterministic-controls issue, not this
# config-source-of-truth gate. Remove only the exact values injected by P0.
ACCIDENTAL_P0_TRAINING_VALUES = {
    "random_seed": 76076,
    "deterministic": True,
    "num_threads": 1,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def atomic_write(path: Path, text: str) -> None:
    root = HERE.resolve()
    destination = path.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing write outside Prop Engine: {destination}"
        ) from exc

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".config-enforcement.tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path.read_text(encoding="utf-8-sig")


def top_level_section(
    lines: list[str],
    name: str,
) -> tuple[int, int]:
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(
            rf"{re.escape(name)}:\s*(?:\{{\}})?\s*",
            line,
        ):
            start = i
            break

    if start is None:
        fail(f"Missing top-level YAML section: {name}")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if (
            line
            and not line[0].isspace()
            and re.match(
                r"^[A-Za-z_][A-Za-z0-9_]*:",
                line,
            )
        ):
            end = i
            break
    return start, end


def patch_config(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    data = yaml.safe_load(normalized)
    if not isinstance(data, dict):
        fail("prop_engine.yaml must be a mapping.")

    seasons = data.get("seasons")
    if not isinstance(seasons, dict):
        fail("config.seasons must be a mapping.")

    required_existing = {
        "historical_start": 2012,
        "rich_feature_start": 2021,
        "historical_end": 2025,
        "current": 2026,
    }
    for key, expected in required_existing.items():
        if int(seasons.get(key, -1)) != expected:
            fail(
                f"Refusing unexpected seasons.{key}="
                f"{seasons.get(key)!r}; expected {expected}."
            )

    targets = data.get("targets")
    if (
        not isinstance(targets, dict)
        or list(targets) != EXPECTED_TARGETS
    ):
        actual = (
            list(targets)
            if isinstance(targets, dict)
            else targets
        )
        fail(
            "Configured target set/order differs from the "
            f"accepted nine-target contract. Actual={actual!r}"
        )

    for target in SIGNED_YARDAGE_TARGETS:
        if (
            str(targets[target].get("type"))
            != "continuous_signed"
        ):
            fail(
                f"{target} must remain continuous_signed."
            )

    lines = normalized.splitlines()

    # Add config.seasons.participation_feature_start.
    s0, s1 = top_level_section(lines, "seasons")
    found = None
    for i in range(s0 + 1, s1):
        if re.match(
            r"^\s{2}participation_feature_start\s*:",
            lines[i],
        ):
            found = i
            break

    if found is None:
        insert_at = s0 + 1
        for i in range(s0 + 1, s1):
            if re.match(
                r"^\s{2}historical_start\s*:",
                lines[i],
            ):
                insert_at = i + 1
                break
        lines.insert(
            insert_at,
            "  participation_feature_start: 2016",
        )
    else:
        lines[found] = (
            "  participation_feature_start: 2016"
        )

    # Replace training block while preserving any future unknown keys.
    t0, t1 = top_level_section(lines, "training")
    block = "\n".join(lines[t0:t1])
    parsed = yaml.safe_load(block)
    existing = (
        parsed.get("training")
        if isinstance(parsed, dict)
        else {}
    )
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        fail("config.training must be a mapping.")

    merged = dict(existing)
    for key, accidental_value in ACCIDENTAL_P0_TRAINING_VALUES.items():
        if merged.get(key) == accidental_value:
            merged.pop(key, None)
    merged.update(TRAINING_VALUES)

    ordered_keys = [
        "model_selection_train_end_season",
        "development_validation_season",
        "final_train_end_season",
        "untouched_test_season",
    ]

    replacement = ["training:"]
    for key in ordered_keys:
        value = merged[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        replacement.append(
            f"  {key}: {rendered}"
        )

    for key, value in merged.items():
        if key in ordered_keys:
            continue
        dumped = yaml.safe_dump(
            {key: value},
            default_flow_style=False,
            sort_keys=False,
        ).strip().splitlines()
        replacement.extend(
            "  " + line
            for line in dumped
        )

    lines[t0:t1] = replacement + [""]

    updated = "\n".join(lines).rstrip() + "\n"
    check = yaml.safe_load(updated)

    if (
        check["seasons"][
            "participation_feature_start"
        ]
        != 2016
    ):
        fail(
            "Failed to persist "
            "participation_feature_start."
        )

    for key, expected in TRAINING_VALUES.items():
        if check["training"].get(key) != expected:
            fail(
                f"Failed to persist training.{key}."
            )

    return updated


def ensure_common_config_load(text: str) -> str:
    if "common.load_config(" in text:
        return text

    match = re.search(
        r"(?m)^import common\s*$",
        text,
    )
    if not match:
        fail(
            "Downstream production script does not "
            "import common; cannot add config load safely."
        )

    return (
        text[:match.end()]
        + "\n_CONFIG_CONTRACT = common.load_config()\n"
        + text[match.end():]
    )


def ensure_config_contract_alias(text: str) -> str:
    if (
        "_CONFIG_CONTRACT = common.load_config()"
        in text
    ):
        return text

    match = re.search(
        r"(?m)^import common\s*$",
        text,
    )
    if not match:
        fail(
            "Required source does not import common."
        )

    return (
        text[:match.end()]
        + "\n\n_CONFIG_CONTRACT = common.load_config()"
        + text[match.end():]
    )


def ensure_training_contract_alias(text: str) -> str:
    text = ensure_config_contract_alias(text)

    marker = (
        '_TRAINING_CONTRACT = '
        '_CONFIG_CONTRACT["training"]'
    )
    if marker in text:
        return text

    config_marker = (
        "_CONFIG_CONTRACT = common.load_config()"
    )
    pos = text.find(config_marker)
    if pos < 0:
        fail(
            "Could not locate config contract marker."
        )
    end = pos + len(config_marker)

    return (
        text[:end]
        + '\n_TRAINING_CONTRACT = '
          '_CONFIG_CONTRACT["training"]'
        + text[end:]
    )


def replace_top_level_assignment(
    text: str,
    variable: str,
    replacement: str,
    *,
    expected_literal=None,
) -> tuple[str, bool]:
    tree = ast.parse(text)
    candidates = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign):
            names = (
                [node.target.id]
                if isinstance(
                    node.target,
                    ast.Name,
                )
                else []
            )
        else:
            continue

        if variable in names:
            candidates.append(node)

    if len(candidates) != 1:
        fail(
            f"Expected exactly one top-level "
            f"assignment for {variable}; "
            f"found {len(candidates)}."
        )

    node = candidates[0]

    if expected_literal is not None:
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
            if replacement in segment:
                return text, False
            fail(
                f"{variable} is neither the "
                "accepted prior literal nor "
                "the new dynamic contract."
            )

        if value != expected_literal:
            fail(
                f"Unexpected prior value for "
                f"{variable}: {value!r}"
            )

    lines = text.splitlines(keepends=True)
    start = int(node.lineno) - 1
    end = int(
        node.end_lineno
        or node.lineno
    )
    newline = (
        "\n"
        if lines[start].endswith(
            ("\n", "\r\n")
        )
        else ""
    )
    lines[start:end] = [
        replacement + newline
    ]

    updated = "".join(lines)
    ast.parse(updated)
    return updated, True


def patch_split_trainer(path: Path) -> str:
    text = read(path)
    text = ensure_training_contract_alias(text)

    replacements = {
        "MODEL_SELECTION_TRAIN_END":
            'MODEL_SELECTION_TRAIN_END = int('
            '_TRAINING_CONTRACT['
            '"model_selection_train_end_season"])',
        "DEVELOPMENT_VALIDATION_SEASON":
            'DEVELOPMENT_VALIDATION_SEASON = int('
            '_TRAINING_CONTRACT['
            '"development_validation_season"])',
        "FINAL_TRAIN_END":
            'FINAL_TRAIN_END = int('
            '_TRAINING_CONTRACT['
            '"final_train_end_season"])',
        "UNTOUCHED_TEST_SEASON":
            'UNTOUCHED_TEST_SEASON = int('
            '_TRAINING_CONTRACT['
            '"untouched_test_season"])',
    }

    accepted_old = {
        "MODEL_SELECTION_TRAIN_END": 2023,
        "DEVELOPMENT_VALIDATION_SEASON": 2024,
        "FINAL_TRAIN_END": 2024,
        "UNTOUCHED_TEST_SEASON": 2025,
    }

    for variable, replacement in replacements.items():
        text, _ = replace_top_level_assignment(
            text,
            variable,
            replacement,
            expected_literal=accepted_old[
                variable
            ],
        )

    return text


def patch_configured_targets(
    path: Path,
) -> str:
    text = read(path)
    text = ensure_config_contract_alias(text)
    tree = ast.parse(text)

    node = None
    for candidate in tree.body:
        if not isinstance(
            candidate,
            ast.Assign,
        ):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "TARGETS"
            for target in candidate.targets
        ):
            node = candidate
            break

    if node is None:
        return text

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
        if (
            "targets" in segment.lower()
            and (
                "_CONFIG_CONTRACT"
                in segment
                or "config" in segment
            )
        ):
            return text
        fail(
            f"{path.name}: TARGETS is neither "
            "the accepted literal nor config-derived."
        )

    if list(value) != EXPECTED_TARGETS:
        fail(
            f"{path.name}: unexpected literal "
            f"TARGETS={value!r}"
        )

    text, _ = replace_top_level_assignment(
        text,
        "TARGETS",
        'TARGETS = list('
        '_CONFIG_CONTRACT["targets"].keys())',
        expected_literal=value,
    )
    return text


def patch_build_targets(path: Path) -> str:
    text = read(path)
    text = ensure_config_contract_alias(text)

    marker = (
        "_CONFIG_CONTRACT = common.load_config()"
    )
    target_marker = (
        '_CONFIG_TARGETS = '
        '_CONFIG_CONTRACT["targets"]'
    )
    if target_marker not in text:
        pos = (
            text.find(marker)
            + len(marker)
        )
        text = (
            text[:pos]
            + "\n"
            + target_marker
            + text[pos:]
        )

    direct_replacement = (
        "DIRECT_TARGETS = [\n"
        "    target\n"
        "    for target, spec in "
        "_CONFIG_TARGETS.items()\n"
        "    if isinstance(spec, dict)\n"
        '    and spec.get("source_column")\n'
        '    and target != "sacks"\n'
        "]"
    )
    text, _ = replace_top_level_assignment(
        text,
        "DIRECT_TARGETS",
        direct_replacement,
        expected_literal=DIRECT_SOURCE_TARGETS,
    )

    signed_replacement = (
        "SIGNED_YARDAGE_TARGETS = {\n"
        "    target\n"
        "    for target, spec in "
        "_CONFIG_TARGETS.items()\n"
        "    if isinstance(spec, dict)\n"
        '    and spec.get("type") '
        '== "continuous_signed"\n'
        "}"
    )
    text, _ = replace_top_level_assignment(
        text,
        "SIGNED_YARDAGE_TARGETS",
        signed_replacement,
        expected_literal=SIGNED_YARDAGE_TARGETS,
    )

    return text


def patch_baselines(path: Path) -> str:
    text = patch_configured_targets(path)
    text = ensure_training_contract_alias(text)

    marker = (
        '_TRAINING_CONTRACT = '
        '_CONFIG_CONTRACT["training"]'
    )
    if (
        "UNTOUCHED_TEST_SEASON = int("
        "_TRAINING_CONTRACT"
        not in text
    ):
        constants = (
            '\nMODEL_SELECTION_TRAIN_END = int('
            '_TRAINING_CONTRACT['
            '"model_selection_train_end_season"])'
            '\nDEVELOPMENT_VALIDATION_SEASON = int('
            '_TRAINING_CONTRACT['
            '"development_validation_season"])'
            '\nFINAL_TRAIN_END = int('
            '_TRAINING_CONTRACT['
            '"final_train_end_season"])'
            '\nUNTOUCHED_TEST_SEASON = int('
            '_TRAINING_CONTRACT['
            '"untouched_test_season"])'
        )
        pos = (
            text.find(marker)
            + len(marker)
        )
        text = (
            text[:pos]
            + constants
            + text[pos:]
        )

    replacements = {
        'if int(test["validation_start_season"]) '
        "!= 2025:":
            'if int(test["validation_start_season"]) '
            "!= UNTOUCHED_TEST_SEASON:",
        'if int(test["train_end_season"]) '
        "!= 2024:":
            'if int(test["train_end_season"]) '
            "!= FINAL_TRAIN_END:",
        "dev_2024 = folds.loc[":
            "development_fold = folds.loc[",
        'folds["validation_start_season"]'
        ".eq(2024)":
            'folds["validation_start_season"]'
            ".eq(DEVELOPMENT_VALIDATION_SEASON)",
        "len(dev_2024)":
            "len(development_fold)",
        "dev_2024.iloc[0]":
            "development_fold.iloc[0]",
        "!= 2023:":
            "!= MODEL_SELECTION_TRAIN_END:",
        "rows_2025":
            "rows_test",
        'output["season"].eq(2025)':
            'output["season"].eq('
            "UNTOUCHED_TEST_SEASON)",
        '"untouched_test_season": 2025':
            '"untouched_test_season": '
            "UNTOUCHED_TEST_SEASON",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    ast.parse(text)
    return text


def patch_backtest_folds(path: Path) -> str:
    text = read(path)

    if (
        "_CONFIG_ENFORCED_BACKTEST_POLICY"
        in text
    ):
        return text

    marker = (
        '    feature_path = '
        'config["paths"]["historical_features"]'
    )
    if marker not in text:
        fail(
            "build_backtest_folds.py "
            "anchor changed."
        )

    block = r'''    # _CONFIG_ENFORCED_BACKTEST_POLICY
    training = config["training"]
    configured_policy = {
        "model_selection_train_end_season": int(
            training["model_selection_train_end_season"]
        ),
        "development_validation_season": int(
            training["development_validation_season"]
        ),
        "final_train_end_season": int(
            training["final_train_end_season"]
        ),
        "untouched_test_season": int(
            training["untouched_test_season"]
        ),
    }
    derived_policy = {
        "model_selection_train_end_season": historical_end - 2,
        "development_validation_season": historical_end - 1,
        "final_train_end_season": historical_end - 1,
        "untouched_test_season": historical_end,
    }
    if configured_policy != derived_policy:
        raise ValueError(
            "Configured training split boundaries disagree with the "
            "chronological annual-fold contract. "
            f"configured={configured_policy}, derived={derived_policy}"
        )

'''
    text = text.replace(
        marker,
        block + marker,
        1,
    )
    ast.parse(text)
    return text


def patch_select_architecture(
    path: Path,
) -> str:
    text = read(path)

    if (
        "_CONFIG_ENFORCED_SELECTION_POLICY"
        in text
    ):
        return text

    anchor = (
        "    policy = resolve_folds(folds)\n"
        "    assert_trainer_cutoffs(policy)"
    )
    if anchor not in text:
        fail(
            "select_model_architecture.py "
            "policy anchor changed."
        )

    block = '''    policy = resolve_folds(folds)
    assert_trainer_cutoffs(policy)

    # _CONFIG_ENFORCED_SELECTION_POLICY
    training = config["training"]
    configured_policy = {
        "selection_train_end_season": int(
            training["model_selection_train_end_season"]
        ),
        "validation_season": int(
            training["development_validation_season"]
        ),
        "final_train_end_season": int(
            training["final_train_end_season"]
        ),
        "test_season": int(
            training["untouched_test_season"]
        ),
    }
    observed_policy = {
        key: int(policy[key])
        for key in configured_policy
    }
    if observed_policy != configured_policy:
        raise ValueError(
            "Backtest-fold policy differs from config.training. "
            f"observed={observed_policy}, configured={configured_policy}"
        )'''

    text = text.replace(
        anchor,
        block,
        1,
    )
    ast.parse(text)
    return text


def patch_calibrate_uncertainty(
    path: Path,
) -> str:
    text = read(path)

    if (
        "_CONFIG_ENFORCED_CALIBRATION_SPLIT"
        in text
    ):
        return text

    anchor = (
        "    contracts = "
        "load_selected_contracts(targets)"
    )
    if anchor not in text:
        fail(
            "calibrate_uncertainty.py "
            "selected-contract anchor changed."
        )

    block = anchor + r'''
    # _CONFIG_ENFORCED_CALIBRATION_SPLIT
    training = config["training"]
    expected_validation = int(
        training["development_validation_season"]
    )
    expected_test = int(
        training["untouched_test_season"]
    )
    expected_train_end = int(
        training["model_selection_train_end_season"]
    )
    for target, contract in contracts.items():
        observed = (
            int(contract.validation_season),
            int(contract.test_season),
            int(contract.model_selection_train_end_season),
        )
        expected = (
            expected_validation,
            expected_test,
            expected_train_end,
        )
        if observed != expected:
            raise ValueError(
                f"{target}: selected-model split contract "
                f"{observed} != config.training {expected}"
            )'''

    text = text.replace(
        anchor,
        block,
        1,
    )
    ast.parse(text)
    return text


def patch_model_report(
    path: Path,
) -> str:
    text = read(path)

    if (
        "_CONFIG_ENFORCED_REPORTING_SPLIT"
        in text
    ):
        return text

    anchor = (
        "    selected, architectures = "
        "selected_rows(config, audit, args.split)"
    )
    if anchor not in text:
        fail(
            "build_model_report.py "
            "selected_rows anchor changed."
        )

    block = anchor + r'''
    # _CONFIG_ENFORCED_REPORTING_SPLIT
    training = config["training"]
    split_season = {
        "validation": int(
            training["development_validation_season"]
        ),
        "test": int(
            training["untouched_test_season"]
        ),
    }.get(str(args.split))
    if split_season is None:
        raise ValueError(
            f"Unsupported reporting split: {args.split!r}"
        )
    observed_seasons = set(
        pd.to_numeric(
            selected["season"],
            errors="raise",
        ).astype(int)
    )
    if observed_seasons != {split_season}:
        raise ValueError(
            f"{args.split} report rows must use configured "
            f"season {split_season}; "
            f"observed={sorted(observed_seasons)}"
        )'''

    text = text.replace(
        anchor,
        block,
        1,
    )
    ast.parse(text)
    return text


def patch_participation_opportunity(
    path: Path,
) -> str:
    text = read(path)

    text = text.replace(
        '    pre_participation = '
        'output["season"].lt(2016)',
        '    participation_feature_start = int(\n'
        '        config["seasons"]'
        '["participation_feature_start"]\n'
        '    )\n'
        '    pre_participation = '
        'output["season"].lt('
        'participation_feature_start)',
    )

    # Add run-level diagnostic variable once.
    run_anchor = (
        '    rich_feature_start = '
        'int(config["seasons"]'
        '["rich_feature_start"])\n'
    )
    run_extra = (
        run_anchor
        + '    participation_feature_start = int(\n'
          '        config["seasons"]'
          '["participation_feature_start"]\n'
          '    )\n'
    )
    if (
        run_extra not in text
        and run_anchor in text
    ):
        text = text.replace(
            run_anchor,
            run_extra,
            1,
        )

    text = text.replace(
        "        if part_path.is_file():\n"
        "            part_source = common.read_parquet_required(part_path)",
        "        if (\n"
        "            season >= participation_feature_start\n"
        "            and part_path.is_file()\n"
        "        ):\n"
        "            part_source = common.read_parquet_required(part_path)",
        1,
    )

    text = text.replace(
        '"participation": "2016-2025",',
        '"participation": '
        'f"{participation_feature_start}-'
        '{end_season}",',
    )

    ast.parse(text)
    return text


def patch_participation_universe(
    path: Path,
) -> str:
    text = read(path)

    text = text.replace(
        "        if season < 2016:\n"
        "            continue",
        '        if season < int('
        'config["seasons"]'
        '["participation_feature_start"]'
        '):\n'
        "            continue",
    )

    text = re.sub(
        r'\("participation_"\s*'
        r'"available_start"\s*\)'
        r':\s*2016,',
        '("participation_"\n'
        '                "available_start"\n'
        '            ): int(\n'
        '                config["seasons"]'
        '["participation_feature_start"]\n'
        '            ),',
        text,
        count=1,
    )

    ast.parse(text)
    return text


def patch_manifest_rejection(
    path: Path,
) -> str:
    text = read(path)

    marker = (
        "common.reject_forbidden_feature_columns("
        "feature_names, common.load_config())"
    )

    if marker in text:
        return text

    if path.name == "project_components.py":
        anchor = (
            "    feature_names = "
            "numeric_features + categorical"
        )
    elif path.name == "project_direct.py":
        anchor = (
            "    feature_names = "
            "[*numeric_features, "
            "*categorical_features]"
        )
    else:
        fail(
            f"Unsupported manifest patch target: "
            f"{path.name}"
        )

    if anchor not in text:
        fail(
            f"{path.name}: manifest anchor changed."
        )

    text = text.replace(
        anchor,
        anchor + "\n    " + marker,
        1,
    )
    ast.parse(text)
    return text


def patch_file(
    path: Path,
    new_text: str,
    changed: list[str],
) -> None:
    old = read(path).replace(
        "\r\n",
        "\n",
    )
    normalized = new_text.replace(
        "\r\n",
        "\n",
    )

    if old == normalized:
        return

    if path.suffix == ".py":
        ast.parse(
            normalized,
            filename=str(path),
        )

    atomic_write(
        path,
        normalized,
    )
    changed.append(
        str(
            path.relative_to(HERE)
        ).replace(
            "\\",
            "/",
        )
    )


def main() -> int:
    changed: list[str] = []

    patch_file(
        CONFIG,
        patch_config(read(CONFIG)),
        changed,
    )

    # Find every literal all-nine TARGETS assignment.
    target_paths = set(
        TARGET_ENUMERATION_FILES
    )
    for directory in PRODUCTION_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.py"):
            tree = ast.parse(
                read(path),
                filename=str(path),
            )
            for node in tree.body:
                if not isinstance(
                    node,
                    ast.Assign,
                ):
                    continue
                if not any(
                    isinstance(target, ast.Name)
                    and target.id == "TARGETS"
                    for target in node.targets
                ):
                    continue
                try:
                    value = ast.literal_eval(
                        node.value
                    )
                except Exception:
                    continue
                if (
                    isinstance(
                        value,
                        (list, tuple),
                    )
                    and list(value)
                    == EXPECTED_TARGETS
                ):
                    target_paths.add(path)

    for path in sorted(target_paths):
        if (
            path.name
            == "train_baselines.py"
        ):
            updated = patch_baselines(path)
        else:
            updated = patch_configured_targets(
                path
            )
        patch_file(
            path,
            updated,
            changed,
        )

    for path in TRAINER_SPLIT_FILES:
        patch_file(
            path,
            patch_split_trainer(path),
            changed,
        )

    targets_path = (
        HERE
        / "scripts/build/build_targets.py"
    )
    patch_file(
        targets_path,
        patch_build_targets(
            targets_path
        ),
        changed,
    )

    folds_path = (
        HERE
        / "scripts/train/build_backtest_folds.py"
    )
    patch_file(
        folds_path,
        patch_backtest_folds(
            folds_path
        ),
        changed,
    )

    selection_path = (
        HERE
        / "scripts/train/select_model_architecture.py"
    )
    patch_file(
        selection_path,
        patch_select_architecture(
            selection_path
        ),
        changed,
    )

    calibration_path = (
        HERE
        / "scripts/train/calibrate_uncertainty.py"
    )
    patch_file(
        calibration_path,
        patch_calibrate_uncertainty(
            calibration_path
        ),
        changed,
    )

    report_path = (
        HERE
        / "scripts/report/build_model_report.py"
    )
    patch_file(
        report_path,
        patch_model_report(
            report_path
        ),
        changed,
    )

    player_opp = (
        HERE
        / "scripts/build/build_player_opportunity.py"
    )
    patch_file(
        player_opp,
        patch_participation_opportunity(
            player_opp
        ),
        changed,
    )

    historical_universe = (
        HERE
        / "scripts/build/build_historical_universe.py"
    )
    patch_file(
        historical_universe,
        patch_participation_universe(
            historical_universe
        ),
        changed,
    )

    for relative in [
        "scripts/project/project_components.py",
        "scripts/project/project_direct.py",
    ]:
        path = HERE / relative
        patch_file(
            path,
            patch_manifest_rejection(path),
            changed,
        )

    # Literal acceptance: every downstream production script
    # directly loads the shared YAML through common.load_config().
    all_production_files = [
        *ROOT_PRODUCTION_FILES,
        *[
            path
            for directory in PRODUCTION_DIRS
            for path in sorted(directory.glob("*.py"))
        ],
    ]
    for path in all_production_files:
        current = read(path)
        if "common.load_config(" in current:
            continue
        patch_file(
            path,
            ensure_common_config_load(
                current
            ),
            changed,
        )

    checked = 0
    for path in all_production_files:
        ast.parse(
            read(path),
            filename=str(path),
        )
        checked += 1

    print(
        "config.seasons."
        "participation_feature_start=2016"
    )
    print(
        "config.training."
        "model_selection_train_end_season=2023"
    )
    print(
        "config.training."
        "development_validation_season=2024"
    )
    print(
        "config.training."
        "final_train_end_season=2024"
    )
    print(
        "config.training."
        "untouched_test_season=2025"
    )
    print(
        "downstream_python_files_"
        f"syntax_checked={checked}"
    )
    print(
        f"changed_files={len(changed)}"
    )
    for item in changed:
        print(f"changed={item}")
    print(
        "CONFIG ENFORCEMENT PATCH: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
