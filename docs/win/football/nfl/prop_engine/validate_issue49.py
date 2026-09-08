#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 49."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE / "docs" / "DATA_CONTRACT.md"

REQUIRED_SECTIONS = [
    "1. Canonical Keys",
    "2. Existing Read-Only Repository Sources",
    "3. Prop Engine Generated Sources",
    "4. Player Identity Resolution",
    "5. Game Identity Resolution",
    "6. Target Definitions",
    "7. Role Definitions",
    "8. Time/As-Of Definitions",
    "9. Missing-Data Rules",
    "10. Market Data Exclusion",
    "11. Current-Week Source Freshness",
    "12. Known Unavailable Features",
]

UNAVAILABLE = [
    "verified routes run",
    "verified yards per route run",
    "verified pass-rush snaps",
    "verified individual pressure rate",
    "verified independent official last-minute inactive feed",
    "sportsbook/player-prop data",
]

REQUIRED_TARGETS = [
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

REQUIRED_MARKET_TOKENS = [
    "odds",
    "moneyline",
    "spread",
    "drat",
    "epred",
]

REQUIRED_CURRENT_SNAPSHOTS = [
    "weekly roster",
    "current ESPN roster",
    "depth charts",
    "injuries",
    "schedule",
    "weather",
    "travel",
]

REQUIRED_LAGGED = [
    "player stats",
    "snap counts",
    "participation",
    "PBP",
    "team stats",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if not DOC.is_file():
        fail(f"Missing required data contract: {DOC}")

    text = DOC.read_text(encoding="utf-8")
    lowered = text.casefold()

    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if headings != REQUIRED_SECTIONS:
        fail(
            "Required DATA_CONTRACT sections/order mismatch. "
            f"Expected={REQUIRED_SECTIONS} Actual={headings}"
        )

    if "season + week + game_id + player_id" not in text:
        fail("Canonical player-game grain is missing.")
    if "GSIS" not in text or "authoritative player identity" not in text:
        fail("GSIS authoritative identity contract is missing.")

    for alias in ["SD  -> LAC", "OAK -> LV", "STL -> LAR", "WAS -> WSH", "LA  -> LAR", "JAC -> JAX"]:
        if alias not in text:
            fail(f"Canonical team alias missing: {alias}")

    if "3 * field_goals_made + extra_points_made" not in text:
        fail("Exact kicking-points formula missing.")
    if "solo_tackles + assisted_tackles" not in text:
        fail("Exact tackles formula missing.")

    for target in REQUIRED_TARGETS:
        if target not in text:
            fail(f"Target definition missing: {target}")

    required_time_markers = [
        "source observation timestamp < target kickoff timestamp",
        "Week N player rolling features exclude Week N",
        "Week N snap features exclude Week N",
        "Week N participation features exclude Week N",
        "Week N team/opponent form excludes the Week N result",
        "Depth snapshots used for the target game must precede kickoff",
        "Injury snapshots used for the target game must precede kickoff",
        "source week `< N`",
    ]
    for marker in required_time_markers:
        if marker not in text:
            fail(f"Time/as-of rule missing: {marker}")

    if "Realized zero and missing are different states" not in text:
        fail("Zero-versus-missing target rule missing.")
    if "do not invent NFL production" not in text:
        fail("Rookie missing-data rule is missing.")
    if "retain player efficiency history; reset team share" not in text:
        fail("New-team/trade rule is missing.")

    for token in REQUIRED_MARKET_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lowered) is None:
            fail(f"Mandatory market exclusion token missing: {token}")
    if "market_data_allowed: false" not in text:
        fail("market_data_allowed=false contract missing.")
    if "must not use sportsbook/player-prop data" not in lowered:
        fail("Explicit sportsbook/player-prop exclusion missing.")

    section11 = text.split("## 11. Current-Week Source Freshness", 1)[1].split(
        "## 12. Known Unavailable Features", 1
    )[0]
    for source in REQUIRED_CURRENT_SNAPSHOTS:
        if source.casefold() not in section11.casefold():
            fail(f"Current-week snapshot freshness source missing: {source}")
    for source in REQUIRED_LAGGED:
        if source.casefold() not in section11.casefold():
            fail(f"Lagged freshness source missing: {source}")
    if "Week 1" not in section11 or "zero current-season rows are valid" not in section11:
        fail("Week 1 freshness exception missing.")

    section12 = text.split("## 12. Known Unavailable Features", 1)[1]
    for item in UNAVAILABLE:
        if f"- {item}" not in section12:
            fail(f"Required unavailable feature missing exactly: {item}")

    if "proxy must never be relabeled" not in section12.casefold():
        fail("Unavailable-feature proxy-labeling rule missing.")

    if "Existing Read-Only Repository Sources" not in text:
        fail("Read-only source section missing.")
    if "Prop Engine scripts must not" not in text:
        fail("Read-only source behavior not documented.")
    if "data/current/{season}_week_{week}_universe.parquet" not in text:
        fail("Generated current-universe path not documented.")
    if "output/{season}/week_{week}_run_manifest.json" not in text:
        fail("Weekly run manifest path not documented.")

    print("document=docs/DATA_CONTRACT.md")
    print(f"sections={len(REQUIRED_SECTIONS)}")
    print("section_order_exact=true")
    print("canonical_grain_documented=true")
    print("gsis_authoritative=true")
    print("team_aliases_documented=6")
    print("targets_documented=9")
    print("kicking_formula_exact=true")
    print("tackles_formula_exact=true")
    print("strict_asof_rules_documented=true")
    print("zero_vs_missing_documented=true")
    print("market_data_exclusion_documented=true")
    print("current_week_freshness_documented=true")
    print(f"known_unavailable_features={len(UNAVAILABLE)}")
    print("market_features_used=false")
    print("DATA CONTRACT VALIDATION: PASS")
    print("ISSUE 49 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DATA CONTRACT VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
