#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 52."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "docs" / "MODEL_VERSIONING.md"
REGISTRY = HERE / "models" / "production_registry.json"

VERSION_FORMAT = "YYYYMMDD_HHMM_{target}_{architecture}"
EXAMPLE = "20260903_1642_receiving_yards_blend"

REQUIRED_METADATA = [
    "training_start",
    "training_end",
    "target",
    "feature_schema_hash",
    "git_commit_sha",
    "model_family",
    "validation_metrics",
    "final_test_metrics",
    "blend_weights",
    "uncertainty_calibration_version",
]

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


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if not DOC.is_file():
        fail(f"Missing required model versioning document: {DOC}")

    text = DOC.read_text(encoding="utf-8")

    if VERSION_FORMAT not in text:
        fail(f"Exact version format missing: {VERSION_FORMAT}")

    if EXAMPLE not in text:
        fail(f"Required version example missing: {EXAMPLE}")

    example_re = re.compile(
        r"^\d{8}_\d{4}_(passing_yards|passing_tds|rushing_yards|rushing_tds|"
        r"receiving_yards|receiving_tds|kicking_points|tackles|sacks)_"
        r"[a-z0-9_]+$"
    )
    if not example_re.fullmatch(EXAMPLE):
        fail("Required example does not conform to documented version grammar.")

    required_block = re.search(
        r"## Required Metadata(.*?)(?:\n## |\Z)",
        text,
        flags=re.DOTALL,
    )
    if not required_block:
        fail("Required Metadata section missing.")
    block = required_block.group(1)

    for key in REQUIRED_METADATA:
        if key not in block:
            fail(f"Required metadata key missing from Required Metadata section: {key}")

    for key in REQUIRED_METADATA:
        if f"### `{key}`" not in text:
            fail(f"Metadata key lacks dedicated documentation: {key}")

    for target in TARGETS:
        if target not in text:
            fail(f"Target missing from versioning document: {target}")

    required_rules = [
        "Version identifiers are immutable.",
        "must never be silently overwritten",
        "production_approved: true",
        "production_approved: false",
        "final test set must not be used for model selection",
        "Rollback means changing the production registry pointer",
        "A candidate must not be promoted when any required metadata field is missing.",
    ]
    for rule in required_rules:
        if rule.casefold() not in text.casefold():
            fail(f"Required versioning rule missing: {rule}")

    if "`blend_weights`" not in text or "sum to 1.0" not in text:
        fail("Blend-weight versioning contract is incomplete.")

    if "uncertainty_calibration_version" not in text:
        fail("Uncertainty calibration version contract missing.")
    if "models/calibration/" not in text:
        fail("Calibration artifact root not documented.")

    if not REGISTRY.is_file():
        fail(f"Production registry missing: {REGISTRY}")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    if list(registry.keys()) != TARGETS:
        fail(
            "Production registry target set/order mismatch. "
            f"Expected={TARGETS} Actual={list(registry.keys())}"
        )

    for target, entry in registry.items():
        if not isinstance(entry, dict):
            fail(f"{target}: registry entry is not an object.")
        if "production_approved" not in entry or "version" not in entry:
            fail(f"{target}: registry entry missing production_approved/version.")

    if "models/{target}/selected_model.json" not in text:
        fail("Selected-model promotion source not documented.")
    if "models/production_registry.json" not in text:
        fail("Production registry path not documented.")

    print("document=docs/MODEL_VERSIONING.md")
    print(f"version_format={VERSION_FORMAT}")
    print(f"example={EXAMPLE}")
    print(f"required_metadata_fields={len(REQUIRED_METADATA)}")
    print("all_required_metadata_documented=true")
    print("immutable_versions=true")
    print("silent_overwrite_forbidden=true")
    print("blend_weights_versioned=true")
    print("uncertainty_calibration_versioned=true")
    print("validation_vs_final_test_separated=true")
    print("production_registry_documented=true")
    print(f"registry_targets={len(registry)}")
    print("MODEL VERSIONING VALIDATION: PASS")
    print("ISSUE 52 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MODEL VERSIONING VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
