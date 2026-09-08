#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

AUDIT = HERE / "scripts" / "validate" / "audit_market_exclusion.py"

REQUIRED = [
    "docs/win/football/nfl/data/historic_data/odds/",
    "docs/win/football/nfl/scripts/00_intake/pull_odds.py",
    "docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_moneyline.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_spread.py",
    "docs/win/football/nfl/scripts/00_intake/enrich_totals.py",
    "docs/win/football/nfl/scripts/00_intake/pull_market_futures.py",
    "docs/win/football/nfl/data/historic_data/predictions/drat/",
    "docs/win/football/nfl/data/historic_data/predictions/epred/",
    "docs/win/football/nfl/training/",
    "docs/win/football/nfl/01_merge/",
    "docs/win/football/nfl/scripts/01_merge/",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    print("CHECK 01: exact Issue 54 forbidden-input contract")
    config = common.load_config()
    if config.get("forbidden_input_paths") != REQUIRED:
        fail("forbidden_input_paths does not exactly match Issue 54")
    if config.get("system", {}).get("market_data_allowed") is not False:
        fail("system.market_data_allowed must be false")

    print(f"CHECK 02: runtime guard rejects all {len(REQUIRED)} forbidden inputs")
    repo = common.repo_root()
    rejected = 0
    for reference in REQUIRED:
        try:
            common.reject_forbidden_input_path(repo / reference, config)
        except ValueError:
            rejected += 1
        else:
            fail(f"Forbidden input was not rejected: {reference}")

        if reference.endswith("/"):
            nested = repo / reference / "__issue54_probe__.parquet"
            try:
                common.reject_forbidden_input_path(nested, config)
            except ValueError:
                pass
            else:
                fail(f"Nested forbidden input was not rejected: {nested}")

    print("CHECK 03: configured operational inputs avoid forbidden paths")
    for key, value in config.get("paths", {}).items():
        try:
            common.reject_forbidden_input_path(repo / str(value), config)
        except ValueError as exc:
            fail(f"config.paths.{key} points at forbidden input: {exc}")

    print("CHECK 04: shared readers enforce runtime path rejection")
    source = (HERE / "scripts" / "common.py").read_text(encoding="utf-8-sig")
    for function_name in ("read_csv_required", "read_parquet_required"):
        start = source.find(f"def {function_name}(")
        if start < 0:
            fail(f"Missing shared reader: {function_name}")
        end = source.find("\ndef ", start + 4)
        if end < 0:
            end = len(source)
        if "reject_forbidden_input_path(resolved)" not in source[start:end]:
            fail(f"{function_name} lacks forbidden-input guard")

    print("CHECK 05: audit is config-driven")
    audit_source = AUDIT.read_text(encoding="utf-8-sig")
    for marker in (
        "def forbidden_source_references(config: dict | None = None)",
        "forbidden_source_references(config)",
        "def configured_path_forbidden_hits(",
        "FORBIDDEN_INPUT_CONTRACT_RELATIVE_PATH",
    ):
        if marker not in audit_source:
            fail(f"Audit missing marker: {marker}")

    print("CHECK 06: production market-exclusion audit passes")
    completed = subprocess.run(
        [sys.executable, str(AUDIT), "--preflight"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Market exclusion audit failed.\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    if "MARKET EXCLUSION AUDIT: PASS" not in completed.stdout:
        fail("Market exclusion audit did not print PASS")

    audit_json = HERE / "evaluation" / "market_exclusion_audit.json"
    payload = json.loads(audit_json.read_text(encoding="utf-8-sig"))
    if payload.get("passed") is not True:
        fail("market_exclusion_audit.json passed != true")
    if payload.get("forbidden_source_references"):
        fail("Forbidden direct source references remain")
    if payload.get("forbidden_feature_columns"):
        fail("Forbidden model feature columns remain")

    print("CHECK 07: training and weekly inference retain audit preflight")
    for relative in ("scripts/run_training.py", "scripts/run_weekly.py"):
        path = HERE / relative
        if not path.is_file():
            fail(f"Required runner missing: {relative}")
        text = path.read_text(encoding="utf-8-sig")
        if "validate/audit_market_exclusion.py" not in text:
            fail(f"{relative} does not retain market-exclusion preflight")

    print(f"forbidden_input_paths={len(REQUIRED)}")
    print(f"runtime_forbidden_paths_rejected={rejected}")
    print("configured_forbidden_inputs=0")
    print("forbidden_source_references=0")
    print("forbidden_model_feature_columns=0")
    print("training_preflight_enforced=true")
    print("inference_preflight_enforced=true")
    print("football_only_rebuild_contract=true")
    print("ISSUE 54 FORBIDDEN INPUT VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ISSUE 54 FORBIDDEN INPUT VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
