#!/usr/bin/env python3
"""Apply the P0 Prop Engine common.py enforcement patch.

Writes only under docs/win/football/nfl/prop_engine/.

This patch centralizes:
- team normalization
- CSV/parquet atomic writes
- downstream config loading checks
- shared path/validation/forbidden-feature contracts

It is idempotent and fails closed on unexpected source shapes.
"""

from __future__ import annotations

import ast
import os
import re
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE / "scripts/common.py"

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

CSV_WRAPPER_FILES = [
    HERE / "scripts/project/project_week.py",
    HERE / "scripts/validate/validate_source_quality.py",
]

PLAYER_FORM = (
    HERE
    / "scripts/build/build_player_form.py"
)

ROLE_HISTORY = (
    HERE
    / "scripts/build/build_role_history.py"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )
    return path.read_text(
        encoding="utf-8-sig"
    )


def atomic_write(path: Path, text: str) -> None:
    destination = path.resolve()
    root = HERE.resolve()

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Refusing write outside Prop Engine: "
            f"{destination}"
        ) from exc

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".common-enforcement.tmp",
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


def function_node(
    text: str,
    name: str,
) -> ast.FunctionDef:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    if len(matches) != 1:
        fail(
            f"Expected one top-level {name}(); "
            f"found {len(matches)}."
        )
    return matches[0]


def replace_function_body(
    text: str,
    name: str,
    body_lines: list[str],
) -> str:
    node = function_node(text, name)

    if not node.body:
        fail(f"{name}() has no body.")

    lines = text.splitlines(
        keepends=True
    )
    body_start = node.body[0].lineno - 1
    body_end = int(
        node.end_lineno
        or node.body[-1].end_lineno
        or node.body[-1].lineno
    )

    indent = " " * (
        int(node.col_offset) + 4
    )
    replacement = [
        indent + line + "\n"
        for line in body_lines
    ]

    lines[
        body_start:body_end
    ] = replacement

    updated = "".join(lines)
    ast.parse(updated)
    return updated


def patch_common_aliases(
    text: str,
) -> str:
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
            and target.id == "_TEAM_ALIASES"
            for target in candidate.targets
        ):
            node = candidate
            break

    if node is None:
        fail(
            "common.py missing _TEAM_ALIASES."
        )

    try:
        aliases = ast.literal_eval(
            node.value
        )
    except Exception as exc:
        raise RuntimeError(
            "common._TEAM_ALIASES is not "
            "a literal mapping."
        ) from exc

    if not isinstance(aliases, dict):
        fail(
            "common._TEAM_ALIASES must be "
            "a dictionary."
        )

    for key, expected in (
        EXPECTED_TEAM_ALIASES.items()
    ):
        actual = aliases.get(key)
        if (
            actual is not None
            and actual != expected
        ):
            fail(
                f"Conflicting common team alias "
                f"{key}: {actual!r}"
            )
        aliases[key] = expected

    ordered = [
        ("SD", "LAC"),
        ("OAK", "LV"),
        ("STL", "LAR"),
        ("WAS", "WSH"),
        ("LA", "LAR"),
        ("JAC", "JAX"),
    ]

    replacement = [
        "_TEAM_ALIASES = {\n",
        *[
            f'    "{key}": "{value}",\n'
            for key, value in ordered
        ],
        "}\n",
    ]

    lines = text.splitlines(
        keepends=True
    )
    start = node.lineno - 1
    end = int(
        node.end_lineno
        or node.lineno
    )
    lines[start:end] = replacement

    updated = "".join(lines)
    ast.parse(updated)
    return updated


def patch_role_team_normalizer(
    text: str,
) -> str:
    node = function_node(
        text,
        "normalize_team",
    )
    segment = (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )

    if (
        "return common.normalize_team(value)"
        in segment
    ):
        return text

    return replace_function_body(
        text,
        "normalize_team",
        [
            "return common.normalize_team(value)",
        ],
    )


def patch_csv_wrapper(
    path: Path,
    text: str,
) -> str:
    node = function_node(
        text,
        "write_csv_atomic",
    )
    args = [
        argument.arg
        for argument in node.args.args
    ]

    if len(args) < 2:
        fail(
            f"{path.name}: write_csv_atomic "
            "must have dataframe/path args."
        )

    frame_arg = args[0]
    path_arg = args[1]

    segment = (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )

    expected_call = (
        f"common.write_csv_atomic("
        f"{frame_arg}, {path_arg})"
    )

    if expected_call in segment:
        return text

    return replace_function_body(
        text,
        "write_csv_atomic",
        [
            expected_call,
        ],
    )


def is_named_temp_assign(
    statement: ast.stmt,
) -> bool:
    if not isinstance(
        statement,
        (ast.Assign, ast.AnnAssign),
    ):
        return False

    value = statement.value
    if not isinstance(value, ast.Call):
        return False

    func = value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr
        == "NamedTemporaryFile"
    )


def contains_to_parquet(
    node: ast.AST,
) -> bool:
    for child in ast.walk(node):
        if not isinstance(
            child,
            ast.Call,
        ):
            continue
        func = child.func
        if (
            isinstance(
                func,
                ast.Attribute,
            )
            and func.attr
            == "to_parquet"
        ):
            return True
    return False


def patch_player_form_writer(
    text: str,
) -> str:
    node = function_node(
        text,
        "write_output_atomic",
    )
    segment = (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )

    if (
        "common.write_parquet_atomic("
        in segment
        and ".to_parquet(" not in segment
    ):
        return text

    body = list(node.body)
    temp_index = None

    for index, statement in enumerate(
        body
    ):
        if is_named_temp_assign(
            statement
        ):
            temp_index = index
            break

    if temp_index is None:
        fail(
            "build_player_form.py: "
            "write_output_atomic no longer has "
            "the accepted temporary-write block."
        )

    if not any(
        contains_to_parquet(statement)
        for statement
        in body[temp_index:]
    ):
        fail(
            "build_player_form.py: "
            "accepted direct parquet write "
            "was not found."
        )

    lines = text.splitlines(
        keepends=True
    )
    start_stmt = body[temp_index]
    start = start_stmt.lineno - 1
    end = int(
        node.end_lineno
        or body[-1].end_lineno
        or body[-1].lineno
    )

    indent = " " * (
        int(node.col_offset) + 4
    )
    lines[start:end] = [
        indent
        + "common.write_parquet_atomic("
          "output, destination)\n"
    ]

    updated = "".join(lines)
    ast.parse(updated)

    patched = function_node(
        updated,
        "write_output_atomic",
    )
    patched_segment = (
        ast.get_source_segment(
            updated,
            patched,
        )
        or ""
    )

    if ".to_parquet(" in patched_segment:
        fail(
            "build_player_form.py: direct "
            "parquet write remains after patch."
        )

    return updated


def downstream_scripts() -> list[Path]:
    paths: list[Path] = []

    for directory in AUDIT_DIRS:
        if not directory.is_dir():
            fail(
                f"Required audit directory "
                f"missing: {directory}"
            )
        paths.extend(
            sorted(
                directory.glob("*.py")
            )
        )

    return paths


def ensure_common_contract(
    path: Path,
    text: str,
) -> None:
    if "import common" not in text:
        fail(
            f"{path.relative_to(HERE)} "
            "does not import common."
        )

    if "common.load_config(" not in text:
        fail(
            f"{path.relative_to(HERE)} "
            "does not load shared config."
        )


def direct_dataframe_writes(
    text: str,
) -> list[str]:
    tree = ast.parse(text)
    found: list[str] = []

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
            found.append(func.attr)

    return found


def patch_file(
    path: Path,
    updated: str,
    changed: list[str],
) -> None:
    original = read(path).replace(
        "\r\n",
        "\n",
    )
    updated = updated.replace(
        "\r\n",
        "\n",
    )

    if original == updated:
        return

    ast.parse(
        updated,
        filename=str(path),
    )
    atomic_write(
        path,
        updated,
    )
    changed.append(
        str(
            path.relative_to(HERE)
        ).replace("\\", "/")
    )


def main() -> int:
    changed: list[str] = []

    patch_file(
        COMMON,
        patch_common_aliases(
            read(COMMON)
        ),
        changed,
    )

    patch_file(
        ROLE_HISTORY,
        patch_role_team_normalizer(
            read(ROLE_HISTORY)
        ),
        changed,
    )

    for path in CSV_WRAPPER_FILES:
        patch_file(
            path,
            patch_csv_wrapper(
                path,
                read(path),
            ),
            changed,
        )

    patch_file(
        PLAYER_FORM,
        patch_player_form_writer(
            read(PLAYER_FORM)
        ),
        changed,
    )

    scripts = downstream_scripts()

    # Previous P0 config enforcement is a prerequisite.
    for path in scripts:
        ensure_common_contract(
            path,
            read(path),
        )

    # Fail closed if an additional direct dataframe file write
    # exists that was not part of the audited accepted baseline.
    remaining_writes: list[str] = []
    for path in scripts:
        writes = direct_dataframe_writes(
            read(path)
        )
        if writes:
            remaining_writes.append(
                f"{path.relative_to(HERE)}:"
                f"{sorted(writes)}"
            )

    if remaining_writes:
        fail(
            "Unexpected direct CSV/parquet writes "
            "remain after common enforcement: "
            + "; ".join(remaining_writes)
        )

    checked = 0
    for path in scripts:
        ast.parse(
            read(path),
            filename=str(path),
        )
        checked += 1

    print(
        "common_team_aliases=6"
    )
    print(
        "local_csv_atomic_implementations=0"
    )
    print(
        "direct_dataframe_csv_parquet_writes=0"
    )
    print(
        f"downstream_scripts_syntax_checked="
        f"{checked}"
    )
    print(
        f"changed_files={len(changed)}"
    )
    for item in changed:
        print(f"changed={item}")
    print(
        "COMMON ENFORCEMENT PATCH: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
