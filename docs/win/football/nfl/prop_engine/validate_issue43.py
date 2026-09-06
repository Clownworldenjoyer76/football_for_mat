#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 43."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "models" / "production_registry.json"

TARGETS = [
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

EXPECTED_ENTRY_KEYS = ["production_approved", "version"]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if not REGISTRY.is_file():
        fail(f"Missing required Issue 43 registry: {REGISTRY}")

    try:
        with REGISTRY.open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"Registry is not valid JSON: {exc}")

    if not isinstance(data, dict):
        fail("production_registry.json must contain a JSON object.")

    actual_targets = list(data.keys())
    if actual_targets != TARGETS:
        fail(
            "Registry target keys/order mismatch. "
            f"Expected {TARGETS}, got {actual_targets}."
        )

    approved_count = 0
    versioned_count = 0

    for target in TARGETS:
        entry = data[target]
        if not isinstance(entry, dict):
            fail(f"{target} registry entry must be an object.")

        actual_keys = list(entry.keys())
        if actual_keys != EXPECTED_ENTRY_KEYS:
            fail(
                f"{target} entry keys/order mismatch. "
                f"Expected {EXPECTED_ENTRY_KEYS}, got {actual_keys}."
            )

        if entry["production_approved"] is not False:
            approved_count += 1
            fail(
                f"{target}.production_approved must be false in the initial "
                "Issue 43 production registry."
            )

        if entry["version"] is not None:
            versioned_count += 1
            fail(
                f"{target}.version must be null in the initial "
                "Issue 43 production registry."
            )

    expected = {
        target: {"production_approved": False, "version": None}
        for target in TARGETS
    }
    if data != expected:
        fail("Registry does not exactly match the required initial contents.")

    print("config=models/production_registry.json")
    print(f"targets={len(TARGETS)}")
    print(f"production_approved_true={approved_count}")
    print(f"versioned_targets={versioned_count}")
    print("all_production_approved=false")
    print("all_versions_null=true")
    print("PRODUCTION MODEL REGISTRY VALIDATION: PASS")
    print("ISSUE 43 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PRODUCTION MODEL REGISTRY VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
