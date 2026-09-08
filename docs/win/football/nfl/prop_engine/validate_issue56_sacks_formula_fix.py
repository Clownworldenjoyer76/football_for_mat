#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import re
import sys


PROP_ROOT = Path(__file__).resolve().parent

SELECT_PATH = PROP_ROOT / "scripts" / "train" / "select_model_architecture.py"
PROJECT_PATH = PROP_ROOT / "scripts" / "project" / "project_week.py"
ISSUE36_PATH = PROP_ROOT / "validate_issue36.py"
SELECTED_PATH = PROP_ROOT / "models" / "sacks" / "selected_model.json"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def anchored_block(
    text: str,
    *,
    anchor: str,
    max_lines: int,
    label: str,
) -> str:
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if anchor in line]
    if len(hits) != 1:
        fail(f"{label}: expected one anchor {anchor!r}, found {len(hits)}")
    start = hits[0]
    return "\n".join(lines[start : start + max_lines])


def assert_corrected_block(block: str, label: str) -> None:
    if "sack_rate_per_defensive_play" not in block:
        fail(f"{label}: sack rate dependency missing")
    if "opponent_offensive_plays" not in block:
        fail(f"{label}: opponent_offensive_plays missing")
    if "opponent_dropbacks" in block:
        fail(f"{label}: stale opponent_dropbacks remains")


def validate_source() -> None:
    select = read(SELECT_PATH)
    project = read(PROJECT_PATH)
    issue36 = read(ISSUE36_PATH)

    for path, text in [
        (SELECT_PATH, select),
        (PROJECT_PATH, project),
        (ISSUE36_PATH, issue36),
    ]:
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            fail(f"{path.relative_to(PROP_ROOT)}: syntax error: {exc}")

    dep = re.search(
        r'"sacks"\s*:\s*\[(.*?)\]',
        select,
        flags=re.DOTALL,
    )
    if dep is None:
        fail("select_model_architecture.py: sacks component dependency block missing")
    assert_corrected_block(
        dep.group(0),
        "select_model_architecture.py dependency",
    )

    assert_corrected_block(
        anchored_block(
            select,
            anchor='elif target == "sacks":',
            max_lines=10,
            label="select_model_architecture.py sacks formula",
        ),
        "select_model_architecture.py sacks formula",
    )

    assert_corrected_block(
        anchored_block(
            project,
            anchor='points["sacks"] =',
            max_lines=7,
            label="project_week.py sacks formula",
        ),
        "project_week.py sacks formula",
    )

    assert_corrected_block(
        anchored_block(
            issue36,
            anchor='p["sacks"] =',
            max_lines=3,
            label="validate_issue36.py sacks formula",
        ),
        "validate_issue36.py sacks formula",
    )

    offenders: list[str] = []
    for path in PROP_ROOT.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "opponent_dropbacks" not in line:
                continue
            lo = max(0, i - 4)
            hi = min(len(lines), i + 5)
            window = "\n".join(lines[lo:hi])
            if (
                "sack_rate_per_defensive_play" in window
                and (
                    '["sacks"]' in window
                    or '"sacks": [' in window
                    or "target == \"sacks\"" in window
                    or "target == 'sacks'" in window
                )
            ):
                offenders.append(
                    f"{path.relative_to(PROP_ROOT)}:{i + 1}"
                )
    if offenders:
        fail(
            "remaining dimensionally mismatched sacks references: "
            + ", ".join(offenders)
        )

    print("PASS: sacks source dependency/formulas use opponent_offensive_plays")


def validate_generated() -> None:
    if not SELECTED_PATH.is_file():
        fail("models/sacks/selected_model.json missing after model selection")
    try:
        payload = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"invalid selected_model.json: {exc}")

    deps = payload.get("component_dependencies")
    if not isinstance(deps, list):
        fail("selected_model.json component_dependencies is not a list")
    if "opponent_offensive_plays" not in deps:
        fail("selected_model.json missing opponent_offensive_plays")
    if "opponent_dropbacks" in deps:
        fail("selected_model.json still contains opponent_dropbacks")
    if "sack_rate_per_defensive_play" not in deps:
        fail("selected_model.json missing sack_rate_per_defensive_play")

    if payload.get("test_used_for_selection") is not False:
        fail("selected_model.json does not preserve test_used_for_selection=false")
    if payload.get("selection_frozen_before_test_reporting") is not True:
        fail(
            "selected_model.json does not preserve "
            "selection_frozen_before_test_reporting=true"
        )

    print("PASS: regenerated sacks selected_model.json has corrected dependency")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--post-selection",
        action="store_true",
        help="also require regenerated sacks selected_model.json metadata",
    )
    args = parser.parse_args()

    validate_source()
    if args.post_selection:
        validate_generated()
    else:
        if SELECTED_PATH.is_file():
            payload = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
            deps = payload.get("component_dependencies")
            if isinstance(deps, list) and "opponent_dropbacks" in deps:
                print(
                    "NOTE: models/sacks/selected_model.json is stale until "
                    "select_model_architecture.py is rerun."
                )

    print("ISSUE 56 SACKS FORMULA VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
