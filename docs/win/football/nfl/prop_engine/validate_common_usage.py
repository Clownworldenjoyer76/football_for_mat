#!/usr/bin/env python3
"""Independent P0 validator for downstream use of scripts/common.py."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS),
    )

import common

AUDIT_DIRS = [
    HERE / "scripts/build",
    HERE / "scripts/train",
    HERE / "scripts/project",
    HERE / "scripts/validate",
]

EXPECTED_TEAM_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
}

SHARED_FUNCTIONS = [
    "repo_root",
    "nfl_root",
    "prop_root",
    "load_config",
    "require_columns",
    "ensure_unique",
    "reject_forbidden_feature_columns",
    "normalize_team",
    "normalize_player_id",
    "season_week_sort",
    "write_csv_atomic",
    "write_parquet_atomic",
]

MANIFEST_ENFORCERS = [
    HERE
    / "scripts/build/"
      "build_historical_features.py",
    HERE
    / "scripts/train/"
      "train_opportunity_models.py",
    HERE
    / "scripts/train/"
      "train_efficiency_models.py",
    HERE
    / "scripts/train/"
      "train_direct_models.py",
    HERE
    / "scripts/train/"
      "select_model_architecture.py",
    HERE
    / "scripts/project/"
      "build_current_features.py",
    HERE
    / "scripts/project/"
      "project_components.py",
    HERE
    / "scripts/project/"
      "project_direct.py",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def source(path: Path) -> str:
    if not path.is_file():
        fail(
            f"Required source missing: {path}"
        )

    text = path.read_text(
        encoding="utf-8-sig"
    )

    try:
        ast.parse(
            text,
            filename=str(path),
        )
    except SyntaxError as exc:
        fail(
            f"Python syntax failure in "
            f"{path}: {exc}"
        )

    return text


def scripts() -> list[Path]:
    paths: list[Path] = []

    for directory in AUDIT_DIRS:
        if not directory.is_dir():
            fail(
                f"Missing audit directory: "
                f"{directory}"
            )

        paths.extend(
            sorted(
                directory.glob("*.py")
            )
        )

    return paths


def top_level_functions(
    text: str,
) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(text)
    return {
        node.name: node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }


def call_name(
    node: ast.Call,
) -> str:
    func = node.func

    if isinstance(func, ast.Name):
        return func.id

    if (
        isinstance(func, ast.Attribute)
        and isinstance(
            func.value,
            ast.Name,
        )
    ):
        return (
            f"{func.value.id}."
            f"{func.attr}"
        )

    return ""


def has_common_call(
    text: str,
    function: str,
) -> bool:
    tree = ast.parse(text)

    expected = f"common.{function}"

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and call_name(node)
            == expected
        ):
            return True

    return False


def direct_dataframe_writes(
    text: str,
) -> list[tuple[int, str]]:
    tree = ast.parse(text)
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func
        if not isinstance(
            func,
            ast.Attribute,
        ):
            continue

        if func.attr in {
            "to_csv",
            "to_parquet",
        }:
            found.append(
                (
                    int(node.lineno),
                    func.attr,
                )
            )

    return found


def is_thin_common_wrapper(
    text: str,
    node: ast.FunctionDef,
    shared_name: str,
) -> bool:
    # Accept an optional docstring plus one return/expr call
    # to the matching common helper.
    body = list(node.body)

    if (
        body
        and isinstance(
            body[0],
            ast.Expr,
        )
        and isinstance(
            body[0].value,
            ast.Constant,
        )
        and isinstance(
            body[0].value.value,
            str,
        )
    ):
        body = body[1:]

    if len(body) != 1:
        return False

    statement = body[0]

    if isinstance(
        statement,
        ast.Return,
    ):
        call = statement.value
    elif isinstance(
        statement,
        ast.Expr,
    ):
        call = statement.value
    else:
        return False

    return (
        isinstance(call, ast.Call)
        and call_name(call)
        == f"common.{shared_name}"
    )


def check_common_module() -> None:
    text = source(
        HERE / "scripts/common.py"
    )
    funcs = top_level_functions(text)

    missing = [
        name
        for name in SHARED_FUNCTIONS
        if name not in funcs
    ]
    if missing:
        fail(
            "common.py missing shared "
            f"function(s): {missing}"
        )

    tree = ast.parse(text)
    alias_node = None

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "_TEAM_ALIASES"
            for target in node.targets
        ):
            alias_node = node
            break

    if alias_node is None:
        fail(
            "common.py missing _TEAM_ALIASES."
        )

    aliases = ast.literal_eval(
        alias_node.value
    )

    for key, expected in (
        EXPECTED_TEAM_ALIASES.items()
    ):
        if aliases.get(key) != expected:
            fail(
                f"common.normalize_team alias "
                f"{key}->{expected} missing."
            )

    for writer in (
        "write_csv_atomic",
        "write_parquet_atomic",
    ):
        node = funcs[writer]
        segment = (
            ast.get_source_segment(
                text,
                node,
            )
            or ""
        )
        if "season_week_sort(df)" not in segment:
            fail(
                f"common.{writer} does not "
                "delegate deterministic ordering "
                "through season_week_sort()."
            )


def check_imports_and_config(
    paths: list[Path],
) -> None:
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

        if not has_common_call(
            text,
            "load_config",
        ):
            violations.append(
                f"{relative}: "
                "no common.load_config()"
            )

    if violations:
        fail(
            "Shared import/config violations: "
            + "; ".join(violations)
        )


def check_no_shared_reimplementation(
    paths: list[Path],
) -> tuple[int, int]:
    wrapper_count = 0
    forbidden_defs: list[str] = []

    shared_names = {
        "repo_root",
        "nfl_root",
        "prop_root",
        "require_columns",
        "ensure_unique",
        "reject_forbidden_feature_columns",
        "normalize_team",
        "normalize_player_id",
        "write_csv_atomic",
        "write_parquet_atomic",
    }

    for path in paths:
        text = source(path)
        funcs = top_level_functions(text)

        for name in sorted(
            shared_names.intersection(funcs)
        ):
            node = funcs[name]

            if is_thin_common_wrapper(
                text,
                node,
                name,
            ):
                wrapper_count += 1
                continue

            forbidden_defs.append(
                f"{path.relative_to(HERE)}:"
                f"{name}"
            )

    if forbidden_defs:
        fail(
            "Independent shared-function "
            "reimplementations remain: "
            + "; ".join(forbidden_defs)
        )

    # There should be no downstream direct DataFrame file writes.
    direct_writes = []
    for path in paths:
        for line, method in (
            direct_dataframe_writes(
                source(path)
            )
        ):
            direct_writes.append(
                f"{path.relative_to(HERE)}:"
                f"{line}:{method}"
            )

    if direct_writes:
        fail(
            "Direct CSV/parquet writes bypass "
            "common atomic writers: "
            + "; ".join(direct_writes)
        )

    return wrapper_count, len(direct_writes)


def check_path_resolution(
    paths: list[Path],
) -> None:
    violations: list[str] = []

    forbidden_function_names = {
        "find_repo_root",
        "resolve_repo_root",
        "find_nfl_root",
        "resolve_nfl_root",
        "find_prop_root",
        "resolve_prop_root",
    }

    for path in paths:
        text = source(path)
        tree = ast.parse(text)
        relative = str(
            path.relative_to(HERE)
        ).replace("\\", "/")

        for node in tree.body:
            if (
                isinstance(
                    node,
                    ast.FunctionDef,
                )
                and node.name
                in forbidden_function_names
            ):
                violations.append(
                    f"{relative}:{node.name}"
                )

            if not isinstance(
                node,
                (ast.Assign, ast.AnnAssign),
            ):
                continue

            targets = []
            if isinstance(node, ast.Assign):
                targets = [
                    target.id
                    for target in node.targets
                    if isinstance(
                        target,
                        ast.Name,
                    )
                ]
            elif isinstance(
                node.target,
                ast.Name,
            ):
                targets = [
                    node.target.id
                ]

            root_names = {
                "REPO_ROOT",
                "NFL_ROOT",
                "PROP_ROOT",
                "REPOSITORY_ROOT",
                "PROP_ENGINE_ROOT",
            }

            for name in (
                root_names.intersection(
                    targets
                )
            ):
                segment = (
                    ast.get_source_segment(
                        text,
                        node,
                    )
                    or ""
                )
                if "common." not in segment:
                    violations.append(
                        f"{relative}:{name}"
                    )

        # SCRIPT_DIR / SCRIPTS_ROOT bootstrap is allowed.
        # Multi-parent walking for repository/NFL/Prop roots is not.
        for match in re.finditer(
            r"Path\(__file__\)"
            r"\.resolve\(\)"
            r"\.parents\[[0-9]+\]",
            text,
        ):
            line = (
                text.count(
                    "\n",
                    0,
                    match.start(),
                )
                + 1
            )
            violations.append(
                f"{relative}:{line}:"
                "parents-root-resolution"
            )

    if violations:
        fail(
            "Independent root resolution "
            "remains: "
            + "; ".join(violations)
        )


def check_validation_centralization(
    paths: list[Path],
) -> tuple[int, int]:
    column_helper_users = 0
    uniqueness_helper_users = 0

    for path in paths:
        text = source(path)

        # Required-column validation may be direct or inherited
        # from common.read_*_required().
        if any(
            has_common_call(
                text,
                name,
            )
            for name in (
                "require_columns",
                "read_csv_required",
                "read_parquet_required",
            )
        ):
            column_helper_users += 1

        if has_common_call(
            text,
            "ensure_unique",
        ):
            uniqueness_helper_users += 1

    if column_helper_users == 0:
        fail(
            "No downstream script uses "
            "shared required-column validation."
        )

    if uniqueness_helper_users == 0:
        fail(
            "No downstream script uses "
            "common.ensure_unique()."
        )

    return (
        column_helper_users,
        uniqueness_helper_users,
    )


def check_normalization_centralization(
    paths: list[Path],
) -> tuple[int, int]:
    team_users = 0
    player_id_users = 0

    for path in paths:
        text = source(path)

        if has_common_call(
            text,
            "normalize_team",
        ):
            team_users += 1

        if has_common_call(
            text,
            "normalize_player_id",
        ):
            player_id_users += 1

    if team_users == 0:
        fail(
            "No downstream script uses "
            "common.normalize_team()."
        )

    if player_id_users == 0:
        fail(
            "No downstream script uses "
            "common.normalize_player_id()."
        )

    return team_users, player_id_users


def feature_list(
    payload: dict[str, Any],
) -> list[str]:
    for key in (
        "selected_features",
        "feature_columns",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [
                str(item)
                for item in value
            ]

    numeric = payload.get(
        "numeric_features"
    )
    categorical = payload.get(
        "categorical_features"
    )

    if (
        isinstance(numeric, list)
        or isinstance(
            categorical,
            list,
        )
    ):
        return [
            *[
                str(item)
                for item
                in (numeric or [])
            ],
            *[
                str(item)
                for item
                in (categorical or [])
            ],
        ]

    return []


def manifest_paths(
    targets: list[str],
) -> list[Path]:
    found: set[Path] = set()

    found.update(
        (HERE / "config/features").glob(
            "*.json"
        )
    )

    found.update(
        (HERE / "models/components").glob(
            "*/feature_manifest.json"
        )
    )

    found.update(
        (HERE / "models/efficiency").glob(
            "*/feature_manifest.json"
        )
    )

    for target in targets:
        path = (
            HERE
            / "models"
            / target
            / "feature_manifest.json"
        )
        if path.is_file():
            found.add(path)

    canonical = (
        HERE
        / "data/historical/features/"
          "feature_manifest.json"
    )
    if canonical.is_file():
        found.add(canonical)

    found.update(
        (
            HERE
            / "data/current/features"
        ).glob(
            "*_feature_manifest.json"
        )
    )

    return sorted(
        found,
        key=lambda path: str(path).casefold(),
    )


def check_forbidden_feature_contract(
    config: dict[str, Any],
) -> tuple[int, int]:
    for path in MANIFEST_ENFORCERS:
        text = source(path)
        if not has_common_call(
            text,
            "reject_forbidden_feature_columns",
        ):
            fail(
                f"{path.relative_to(HERE)} "
                "does not route feature-manifest "
                "validation through common."
            )

    targets = list(
        config["targets"].keys()
    )
    manifests = manifest_paths(
        targets
    )

    checked = 0
    for path in manifests:
        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            fail(
                f"Expected JSON object: {path}"
            )

        features = feature_list(
            payload
        )
        if not features:
            continue

        common.reject_forbidden_feature_columns(
            features,
            config,
        )
        checked += 1

    if checked == 0:
        fail(
            "No feature-bearing manifest "
            "was checked."
        )

    return len(MANIFEST_ENFORCERS), checked


def check_atomic_writer_usage(
    paths: list[Path],
) -> tuple[int, int]:
    csv_users = 0
    parquet_users = 0

    for path in paths:
        text = source(path)

        if has_common_call(
            text,
            "write_csv_atomic",
        ):
            csv_users += 1

        if has_common_call(
            text,
            "write_parquet_atomic",
        ):
            parquet_users += 1

    if csv_users == 0:
        fail(
            "No downstream script uses "
            "common.write_csv_atomic()."
        )

    if parquet_users == 0:
        fail(
            "No downstream script uses "
            "common.write_parquet_atomic()."
        )

    return csv_users, parquet_users


def main() -> int:
    check_common_module()

    paths = scripts()
    check_imports_and_config(
        paths
    )

    (
        wrapper_count,
        direct_write_count,
    ) = check_no_shared_reimplementation(
        paths
    )

    check_path_resolution(
        paths
    )

    (
        column_helper_users,
        uniqueness_helper_users,
    ) = check_validation_centralization(
        paths
    )

    (
        team_users,
        player_id_users,
    ) = check_normalization_centralization(
        paths
    )

    (
        csv_writer_users,
        parquet_writer_users,
    ) = check_atomic_writer_usage(
        paths
    )

    config = common.load_config()

    (
        manifest_enforcers,
        manifests_checked,
    ) = check_forbidden_feature_contract(
        config
    )

    print(
        "shared_utility="
        "scripts/common.py"
    )
    print(
        f"downstream_scripts_audited="
        f"{len(paths)}"
    )
    print(
        "all_downstream_import_common=true"
    )
    print(
        "all_downstream_load_config_via_common=true"
    )
    print(
        "independent_root_resolution=0"
    )
    print(
        "independent_required_column_"
        "implementations=0"
    )
    print(
        "independent_uniqueness_"
        "implementations=0"
    )
    print(
        "independent_forbidden_feature_"
        "implementations=0"
    )
    print(
        f"thin_common_wrappers="
        f"{wrapper_count}"
    )
    print(
        f"required_column_helper_users="
        f"{column_helper_users}"
    )
    print(
        f"uniqueness_helper_users="
        f"{uniqueness_helper_users}"
    )
    print(
        f"team_normalization_users="
        f"{team_users}"
    )
    print(
        f"player_id_normalization_users="
        f"{player_id_users}"
    )
    print(
        f"common_csv_writer_users="
        f"{csv_writer_users}"
    )
    print(
        f"common_parquet_writer_users="
        f"{parquet_writer_users}"
    )
    print(
        f"direct_dataframe_csv_parquet_"
        f"writes={direct_write_count}"
    )
    print(
        "deterministic_output_sorting_"
        "via_common=true"
    )
    print(
        f"manifest_enforcement_entrypoints="
        f"{manifest_enforcers}"
    )
    print(
        f"feature_manifests_checked="
        f"{manifests_checked}"
    )
    print(
        "all_model_feature_manifests_"
        "reject_forbidden_features=true"
    )
    print(
        "COMMON USAGE VALIDATION: PASS"
    )
    print(
        "P0 COMMON.PY ENFORCEMENT "
        "ACCEPTANCE: PASS"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "COMMON USAGE VALIDATION: "
            f"FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
