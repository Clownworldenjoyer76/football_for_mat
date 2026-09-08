#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import ast
import os
import re
import tempfile


PROP_ROOT = Path(__file__).resolve().parent

SELECT_PATH = PROP_ROOT / "scripts" / "train" / "select_model_architecture.py"
PROJECT_PATH = PROP_ROOT / "scripts" / "project" / "project_week.py"
ISSUE36_PATH = PROP_ROOT / "validate_issue36.py"
OPTIONAL_EVAL_PATH = PROP_ROOT / "evaluate_production_approval.py"


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def _atomic_write(path: Path, text: str) -> None:
    ast.parse(text, filename=str(path))
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _patch_dependency_block(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r'("sacks"\s*:\s*\[\s*)"opponent_dropbacks"'
        r'(\s*,\s*"player_defensive_participation"\s*,'
        r'\s*"sack_rate_per_defensive_play"\s*,?\s*\])',
        flags=re.MULTILINE,
    )
    new_text, count = pattern.subn(
        r'\1"opponent_offensive_plays"\2',
        text,
        count=1,
    )
    if count == 0:
        corrected = re.compile(
            r'"sacks"\s*:\s*\[\s*"opponent_offensive_plays"'
            r'\s*,\s*"player_defensive_participation"\s*,'
            r'\s*"sack_rate_per_defensive_play"',
            flags=re.MULTILINE,
        )
        if not corrected.search(text):
            raise RuntimeError(
                "Could not locate the sacks component dependency block in "
                "select_model_architecture.py."
            )
    return new_text, count


def _patch_anchored_block(
    text: str,
    *,
    anchor: str,
    max_lines: int,
    label: str,
) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    anchor_indexes = [i for i, line in enumerate(lines) if anchor in line]
    if len(anchor_indexes) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one anchor {anchor!r}, "
            f"found {len(anchor_indexes)}."
        )

    start = anchor_indexes[0]
    end = min(len(lines), start + max_lines)
    block = "".join(lines[start:end])

    if "sack_rate_per_defensive_play" not in block:
        raise RuntimeError(
            f"{label}: anchor found but sack_rate_per_defensive_play "
            "was not in the local formula block."
        )

    if "opponent_dropbacks" in block:
        block2 = block.replace(
            "opponent_dropbacks",
            "opponent_offensive_plays",
            1,
        )
        lines[start:end] = [block2]
        return "".join(lines), 1

    if "opponent_offensive_plays" not in block:
        raise RuntimeError(
            f"{label}: neither old nor corrected opponent volume was found "
            "in the sacks formula block."
        )
    return text, 0


def _patch_optional_evaluator(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        if "opponent_dropbacks" not in line:
            continue
        lo = max(0, i - 4)
        hi = min(len(lines), i + 5)
        window = "".join(lines[lo:hi])
        if (
            "sack_rate_per_defensive_play" in window
            and (
                '["sacks"]' in window
                or "target == \"sacks\"" in window
                or "target == 'sacks'" in window
            )
        ):
            lines[i] = line.replace(
                "opponent_dropbacks",
                "opponent_offensive_plays",
            )
            changed += 1
    return "".join(lines), changed


def _write_if_changed(path: Path, before: str, after: str) -> None:
    if before == after:
        print(f"UNCHANGED: {path.relative_to(PROP_ROOT)}")
        return
    _atomic_write(path, after)
    print(f"PATCHED: {path.relative_to(PROP_ROOT)}")


def main() -> int:
    select_before = _read(SELECT_PATH)
    select_after, dep_count = _patch_dependency_block(select_before)
    select_after, formula_count = _patch_anchored_block(
        select_after,
        anchor='elif target == "sacks":',
        max_lines=10,
        label="select_model_architecture.py sacks formula",
    )
    _write_if_changed(SELECT_PATH, select_before, select_after)

    project_before = _read(PROJECT_PATH)
    project_after, project_count = _patch_anchored_block(
        project_before,
        anchor='points["sacks"] =',
        max_lines=7,
        label="project_week.py sacks formula",
    )
    _write_if_changed(PROJECT_PATH, project_before, project_after)

    issue36_before = _read(ISSUE36_PATH)
    issue36_after, issue36_count = _patch_anchored_block(
        issue36_before,
        anchor='p["sacks"] =',
        max_lines=3,
        label="validate_issue36.py sacks formula",
    )
    _write_if_changed(ISSUE36_PATH, issue36_before, issue36_after)

    eval_count = 0
    if OPTIONAL_EVAL_PATH.is_file():
        eval_before = _read(OPTIONAL_EVAL_PATH)
        eval_after, eval_count = _patch_optional_evaluator(eval_before)
        _write_if_changed(OPTIONAL_EVAL_PATH, eval_before, eval_after)

    print(
        "ISSUE 56 SACKS FORMULA FIX: PASS "
        f"(dependency={dep_count}, selection_formula={formula_count}, "
        f"projection_formula={project_count}, issue36_validator={issue36_count}, "
        f"optional_evaluator={eval_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
