#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 51."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "docs" / "RUNBOOK.md"

REQUIRED_COMMANDS = [
    "python docs/win/football/nfl/prop_engine/scripts/run_historical_build.py --start-season 2021 --end-season 2025",
    "python docs/win/football/nfl/prop_engine/scripts/run_training.py",
    "python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1",
    "python docs/win/football/nfl/prop_engine/scripts/run_weekly.py --season 2026 --week 1 --skip-refresh",
]

TROUBLESHOOTING = {
    "Missing current player stats": {
        "script": "docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py",
        "inspect": "docs/win/football/nfl/prop_engine/data/current/source/stats_player_week_2026.parquet",
    },
    "Missing snap counts": {
        "script": "docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py",
        "inspect": "docs/win/football/nfl/prop_engine/data/current/source/snap_counts_2026.parquet",
    },
    "Missing participation": {
        "script": "docs/win/football/nfl/prop_engine/scripts/build/refresh_nflverse_player_data.py",
        "inspect": "docs/win/football/nfl/prop_engine/data/current/source/pbp_participation_2026.parquet",
    },
    "Unresolved player ID": {
        "script": "docs/win/football/nfl/prop_engine/scripts/build/build_player_identity.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json",
    },
    "Missing depth chart": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json",
    },
    "Missing injury file": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/build_current_universe.py",
        "inspect": "docs/win/football/nfl/prop_engine/evaluation/source_quality.csv",
    },
    "Missing weather": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json",
    },
    "Missing travel": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json",
    },
    "Feature schema mismatch": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/build_current_features.py",
        "inspect": "docs/win/football/nfl/prop_engine/data/current/features/2026_week_1_feature_manifest.json",
    },
    "Out player still projected": {
        "script": "docs/win/football/nfl/prop_engine/scripts/validate/validate_week.py",
        "inspect": "docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json",
    },
    "Multiple starting quarterbacks": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/select_roles.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json",
    },
    "Multiple kickers": {
        "script": "docs/win/football/nfl/prop_engine/scripts/project/select_roles.py",
        "inspect": "docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json",
    },
    "Market-feature rejection": {
        "script": "docs/win/football/nfl/prop_engine/scripts/validate/audit_market_exclusion.py",
        "inspect": "docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json",
    },
    "Historical leakage failure": {
        "script": "docs/win/football/nfl/prop_engine/scripts/validate/validate_historical_data.py",
        "inspect": "docs/win/football/nfl/prop_engine/evaluation/historical_validation.json",
    },
}


def fail(message: str) -> None:
    raise AssertionError(message)


def section(text: str, title: str) -> str:
    marker = f"### {title}"
    start = text.find(marker)
    if start < 0:
        fail(f"Missing troubleshooting section: {title}")
    next_start = text.find("\n### ", start + len(marker))
    if next_start < 0:
        next_start = text.find("\n## ", start + len(marker))
    if next_start < 0:
        next_start = len(text)
    return text[start:next_start]


def main() -> int:
    if not DOC.is_file():
        fail(f"Missing required runbook: {DOC}")

    text = DOC.read_text(encoding="utf-8")

    for command in REQUIRED_COMMANDS:
        if command not in text:
            fail(f"Required operations command missing exactly: {command}")

    headings = re.findall(r"^### (.+?)\s*$", text, flags=re.MULTILINE)
    troubleshooting_headings = [
        h for h in headings if h in TROUBLESHOOTING
    ]
    if troubleshooting_headings != list(TROUBLESHOOTING.keys()):
        fail(
            "Troubleshooting section set/order mismatch. "
            f"Expected={list(TROUBLESHOOTING.keys())} "
            f"Actual={troubleshooting_headings}"
        )

    repo_root = HERE.parents[4]

    for title, contract in TROUBLESHOOTING.items():
        body = section(text, title)
        if "**Rerun script**" not in body:
            fail(f"{title}: missing explicit Rerun script block.")
        if "**Inspect exact output/log**" not in body:
            fail(f"{title}: missing explicit Inspect exact output/log block.")
        if contract["script"] not in body:
            fail(f"{title}: required rerun script not named exactly: {contract['script']}")
        if contract["inspect"] not in body:
            fail(f"{title}: required inspection path not named exactly: {contract['inspect']}")

        script_path = repo_root / contract["script"]
        if not script_path.is_file():
            fail(f"{title}: referenced rerun script does not exist locally: {script_path}")

    required_operational_outputs = [
        "docs/win/football/nfl/prop_engine/logs/historical_build_{timestamp}.json",
        "docs/win/football/nfl/prop_engine/logs/training_{timestamp}.json",
        "docs/win/football/nfl/prop_engine/output/2026/week_1_run_manifest.json",
        "docs/win/football/nfl/prop_engine/output/2026/week_1_validation.json",
        "docs/win/football/nfl/prop_engine/evaluation/source_quality.csv",
        "docs/win/football/nfl/prop_engine/evaluation/market_exclusion_audit.json",
        "docs/win/football/nfl/prop_engine/evaluation/historical_validation.json",
        "docs/win/football/nfl/prop_engine/logs/current_universe_2026_week_1.json",
        "docs/win/football/nfl/prop_engine/logs/current_roles_2026_week_1.json",
        "docs/win/football/nfl/prop_engine/logs/current_features_2026_week_1.json",
    ]
    for item in required_operational_outputs:
        if item not in text:
            fail(f"Runbook missing required exact log/output reference: {item}")

    if "Do not bypass market-exclusion checks" not in text:
        fail("Runbook missing fail-closed market/leakage operational guidance.")
    if "Never weaken the leakage criterion." not in text:
        fail("Runbook missing historical leakage remediation rule.")
    if "do not choose one arbitrarily" not in text:
        fail("Runbook missing QB ambiguity hard-fail guidance.")
    if "do not use an unapproved-model override for a production run".casefold() not in text.casefold():
        fail("Runbook missing production model-approval guidance.")

    print("document=docs/RUNBOOK.md")
    print(f"required_commands={len(REQUIRED_COMMANDS)}")
    print(f"troubleshooting_sections={len(TROUBLESHOOTING)}")
    print("each_section_names_rerun_script=true")
    print("each_section_names_exact_log_or_output=true")
    print("referenced_scripts_exist=true")
    print("historical_rebuild_command_exact=true")
    print("training_command_exact=true")
    print("weekly_command_exact=true")
    print("weekly_skip_refresh_command_exact=true")
    print("market_troubleshooting_documented=true")
    print("leakage_troubleshooting_documented=true")
    print("OPERATIONS RUNBOOK VALIDATION: PASS")
    print("ISSUE 51 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OPERATIONS RUNBOOK VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
