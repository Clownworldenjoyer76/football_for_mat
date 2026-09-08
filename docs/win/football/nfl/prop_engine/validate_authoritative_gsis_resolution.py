#!/usr/bin/env python3
"""Independent validation for authoritative GSIS source repair."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROP = Path(__file__).resolve().parent
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common

TARGETS = {
    "4599158": {"name": "Al-Jay Henderson"},
    "4691889": {"name": "Greg Desrosiers Jr."},
    "4699678": {"name": "Khalil Dinkins"},
}

PLAYERS_REL = Path(
    "docs/win/football/nfl/data/historic_data/players/players.parquet"
)
ROSTER_REL = Path(
    "docs/win/football/nfl/data/historic_data/weekly_rosters/"
    "roster_weekly_2026.parquet"
)
REFRESH_LOG = PROP / "logs/refresh_authoritative_identity_sources_2026.json"
BUILDER_LOG = PROP / "logs/build_player_identity.json"
APPLIER = PROP / "apply_gsis_identity_source_refresh.py"
BUILDER = PROP / "scripts/build/build_player_identity.py"


def fail(message: str) -> None:
    raise AssertionError(message)


def norm_id(value: Any) -> str:
    return common.normalize_player_id(value)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        fail(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()

    players_path = repo / PLAYERS_REL
    roster_path = repo / ROSTER_REL
    crosswalk_path = repo / config["paths"]["identity_crosswalk"]

    for path in (
        players_path,
        roster_path,
        crosswalk_path,
        REFRESH_LOG,
        BUILDER_LOG,
        APPLIER,
        BUILDER,
    ):
        if not path.is_file():
            fail(f"Required identity artifact missing: {path}")

    refresh = load_json(REFRESH_LOG)
    if refresh.get("manual_identity_override") is not False:
        fail("manual_identity_override must be false")
    if refresh.get("canonical_source_repair") is not True:
        fail("canonical_source_repair must be true")
    if str(refresh.get("status", "")).casefold() != "passed":
        fail(f"refresh status={refresh.get('status')!r}")
    if int(refresh.get("builder", {}).get("returncode", -1)) != 0:
        fail("refresh log records builder failure")

    mappings = refresh.get("mappings")
    if not isinstance(mappings, dict) or set(mappings) != set(TARGETS):
        fail("refresh log does not contain exactly the three target mappings")

    players = pd.read_parquet(players_path)
    roster = pd.read_parquet(roster_path)

    common.require_columns(players, ["gsis_id", "espn_id"], "players.parquet")
    common.require_columns(
        roster,
        ["gsis_id", "espn_id"],
        "roster_weekly_2026.parquet",
    )

    p_gsis = players["gsis_id"].map(norm_id)
    p_espn = players["espn_id"].map(norm_id)
    r_gsis = roster["gsis_id"].map(norm_id)
    r_espn = roster["espn_id"].map(norm_id)

    canonical_players = p_gsis[p_gsis.ne("")]
    if canonical_players.duplicated().any():
        fail("Repaired players.parquet has duplicate nonblank GSIS IDs")

    expected: dict[str, str] = {}

    for espn_id, target in TARGETS.items():
        mapping = mappings.get(espn_id, {})
        gsis_id = norm_id(mapping.get("gsis_id"))
        if not gsis_id:
            fail(f"{target['name']}: missing authoritative GSIS in refresh log")

        evidence = mapping.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            fail(f"{target['name']}: missing authoritative evidence")

        expected[espn_id] = gsis_id

        upstream_found = bool(
            (
                p_gsis.eq(gsis_id)
                & p_espn.eq(espn_id)
            ).any()
            or (
                r_gsis.eq(gsis_id)
                & r_espn.eq(espn_id)
            ).any()
        )
        if not upstream_found:
            fail(
                f"{target['name']}: repaired upstream sources do not contain "
                f"ESPN {espn_id} -> GSIS {gsis_id}"
            )

    crosswalk = pd.read_parquet(crosswalk_path)
    common.require_columns(
        crosswalk,
        [
            "player_id",
            "gsis_id",
            "espn_id",
            "current_espn_id",
            "resolution_status",
        ],
        "player_crosswalk.parquet",
    )

    canonical_crosswalk = crosswalk["gsis_id"].map(norm_id)
    if canonical_crosswalk[canonical_crosswalk.ne("")].duplicated().any():
        fail("Canonical GSIS records in crosswalk are not unique")

    for espn_id, gsis_id in expected.items():
        mask = (
            crosswalk["espn_id"].map(norm_id).eq(espn_id)
            | crosswalk["current_espn_id"].map(norm_id).eq(espn_id)
        )
        rows = crosswalk.loc[mask]
        if rows.empty:
            fail(f"Crosswalk has no row for ESPN {espn_id}")
        if set(rows["gsis_id"].map(norm_id)) != {gsis_id}:
            fail(
                f"Crosswalk ESPN {espn_id} does not resolve only to {gsis_id}"
            )
        if not rows["player_id"].map(norm_id).eq(gsis_id).all():
            fail(f"Crosswalk ESPN {espn_id} player_id is not canonical GSIS")
        if not (
            rows["resolution_status"]
            .astype(str)
            .str.casefold()
            .eq("resolved")
            .all()
        ):
            fail(f"Crosswalk ESPN {espn_id} is not resolved")

    build_log = load_json(BUILDER_LOG)
    counts = build_log.get("counts", {})
    if int(counts.get("critical_unresolved_records", -1)) != 0:
        fail(
            f"critical_unresolved_records="
            f"{counts.get('critical_unresolved_records')}; expected 0"
        )
    if str(build_log.get("status", "")).casefold() not in {
        "passed",
        "pass",
        "success",
    }:
        fail(f"build_player_identity status={build_log.get('status')!r}")

    # Target GSIS values may not be embedded in the repair implementation.
    source = APPLIER.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(APPLIER))
    hardcoded_gsis = [
        node.value.strip()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"(?:\d{2}-\d{7,}|[A-Z]{3}\d{6})", node.value.strip())
    ]
    if hardcoded_gsis:
        fail(
            "Hard-coded GSIS-like target override(s) found in repair source: "
            f"{sorted(set(hardcoded_gsis))}"
        )

    builder_source = BUILDER.read_text(encoding="utf-8-sig")
    for espn_id in TARGETS:
        if espn_id in builder_source:
            fail(f"Target ESPN {espn_id} appears directly in builder source")

    official_assets = refresh.get("official_assets", {})
    players_asset = official_assets.get("players", {})
    roster_asset = official_assets.get("weekly_roster_2026", {})
    if not str(players_asset.get("sha256", "")):
        fail("Refresh log missing official players release digest")
    if not str(roster_asset.get("sha256", "")):
        fail("Refresh log missing official roster release digest")

    print("authoritative_gsis_mappings=3")
    for espn_id, gsis_id in expected.items():
        print(
            f"espn_id={espn_id} "
            f"name={TARGETS[espn_id]['name']} "
            f"gsis_id={gsis_id}"
        )
    print(
        "official_players_sha256="
        f"{players_asset['sha256']}"
    )
    print(
        "official_roster_sha256="
        f"{roster_asset['sha256']}"
    )
    print("players_or_roster_upstream_aliases_resolved=true")
    print("manual_identity_override=false")
    print("critical_unresolved_records=0")
    print("build_player_identity_status=passed")
    print("canonical_gsis_unique=true")
    print("AUTHORITATIVE GSIS RESOLUTION VALIDATION: PASS")
    print("P0 GSIS IDENTITY SOURCE ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"AUTHORITATIVE GSIS RESOLUTION VALIDATION: FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
