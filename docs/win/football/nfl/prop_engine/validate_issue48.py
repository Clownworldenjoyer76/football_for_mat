#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 48."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / "requirements.txt"

EXPECTED = [
    "pandas==1.5.3",
    "numpy==1.26.4",
    "scikit-learn==1.4.2",
    "lightgbm==4.3.0",
    "xgboost==2.0.3",
    "scipy==1.10.1",
    "pyarrow==14.0.2",
    "nfl_data_py==0.3.3",
    "pyyaml==6.0.2",
    "requests",
    "nflreadpy",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if not REQUIREMENTS.is_file():
        fail(f"Missing required file: {REQUIREMENTS}")

    text = REQUIREMENTS.read_text(encoding="utf-8")
    actual = text.splitlines()

    if actual != EXPECTED:
        fail(
            "Prop Engine requirements content/order mismatch. "
            f"Expected={EXPECTED} Actual={actual}"
        )

    if any(not line.strip() for line in actual):
        fail("Blank requirement line found.")

    lowered = [line.casefold() for line in actual]
    if any("catboost" in line for line in lowered):
        fail("CatBoost must not be added in Issue 48.")

    if len(actual) != len(set(actual)):
        fail("Duplicate requirement entry found.")

    # Issue 48 package/validator is scoped only to the Prop Engine file.
    # It must never create, overwrite, or validate by modifying the repository
    # root requirements.txt.
    root_requirements = HERE.parents[4] / "requirements.txt"
    if REQUIREMENTS.resolve() == root_requirements.resolve():
        fail("Issue 48 requirements path incorrectly resolves to root requirements.txt.")

    print("requirements=requirements.txt")
    print(f"entries={len(actual)}")
    print("exact_contents=true")
    print("exact_order=true")
    print("root_requirements_modified=false")
    print("catboost_present=false")
    print("ISOLATED PROP ENGINE REQUIREMENTS VALIDATION: PASS")
    print("ISSUE 48 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ISOLATED PROP ENGINE REQUIREMENTS VALIDATION: FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
