#!/usr/bin/env python3
"""Independent acceptance validation for the repaired authoritative GSIS identities."""

from __future__ import annotations

import ast
import hashlib
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
    "4599158": {"name": "Al-Jay Henderson", "surname": "henderson", "position": "RB", "team": "NYJ"},
    "4691889": {"name": "Greg Desrosiers Jr.", "surname": "desrosiers", "position": "RB", "team": "LAC"},
    "4699678": {"name": "Khalil Dinkins", "surname": "dinkins", "position": "TE", "team": "SF"},
}
ROSTER_SHA256 = "41c188f55571ae756950d44e5c8e192f05692f4fc594106491f43228ed9a57b2"
PLAYERS_REL = Path("docs/win/football/nfl/data/historic_data/players/players.parquet")
ROSTER_REL = Path(
    "docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_2026.parquet"
)
REFRESH_LOG = PROP / "logs/refresh_authoritative_identity_sources_2026.json"
BUILDER_LOG = PROP / "logs/build_player_identity.json"
APPLIER = PROP / "apply_gsis_identity_source_refresh.py"
BUILDER = PROP / "scripts/build/build_player_identity.py"

NAME_ALIASES = ("full_name", "display_name", "player_name", "football_name")
POSITION_ALIASES = ("position", "depth_chart_position", "ngs_position")
TEAM_ALIASES = ("team", "team_abbr", "team_code", "club_code", "recent_team")
GSIS_ALIASES = ("gsis_id", "player_id")


def fail(message: str):
    raise AssertionError(message)


def norm_id(value: Any) -> str:
    return common.normalize_player_id(value)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def choose(df, aliases, label):
    for c in aliases:
        if c in df.columns:
            return c
    fail(f"{label}: no supported column in {list(aliases)}")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derive(roster):
    cols = {
        "name": choose(roster, NAME_ALIASES, "roster name"),
        "position": choose(roster, POSITION_ALIASES, "roster position"),
        "team": choose(roster, TEAM_ALIASES, "roster team"),
        "gsis": choose(roster, GSIS_ALIASES, "roster GSIS"),
    }
    work = roster.copy()
    work["_name"] = work[cols["name"]].map(common.normalize_name)
    work["_position"] = work[cols["position"]].map(lambda x: clean(x).upper())
    work["_team"] = work[cols["team"]].map(common.normalize_team)
    work["_gsis"] = work[cols["gsis"]].map(norm_id)

    out = {}
    for espn_id, target in TARGETS.items():
        rows = work.loc[
            work["_team"].eq(common.normalize_team(target["team"]))
            & work["_position"].eq(target["position"])
            & work["_name"].str.split().map(
                lambda tokens: target["surname"] in tokens if isinstance(tokens, list) else False
            )
            & work["_gsis"].ne("")
        ]
        ids = sorted(set(rows["_gsis"]))
        if len(ids) != 1:
            fail(
                f"{target['name']}: roster-derived GSIS count={len(ids)} ids={ids}"
            )
        out[espn_id] = ids[0]
    return out


def load_json(path):
    if not path.is_file():
        fail(f"Missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        fail(f"Expected object: {path}")
    return obj


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()
    roster_path = repo / ROSTER_REL
    players_path = repo / PLAYERS_REL

    if not roster_path.is_file() or not players_path.is_file():
        fail("Authorized identity source missing.")

    if sha256_file(roster_path) != ROSTER_SHA256:
        fail("2026 authorized roster is not the official nflverse release asset.")

    roster = pd.read_parquet(roster_path)
    players = pd.read_parquet(players_path)
    mappings = derive(roster)

    common.require_columns(players, ["gsis_id", "espn_id"], "players.parquet")
    gsis = players["gsis_id"].map(norm_id)
    espn = players["espn_id"].map(norm_id)

    nonblank = players.loc[gsis.ne("")].copy()
    if nonblank["gsis_id"].map(norm_id).duplicated().any():
        fail("Canonical GSIS records in repaired players.parquet are not unique.")

    for espn_id, expected_gsis in mappings.items():
        rows = players.loc[gsis.eq(expected_gsis)]
        if len(rows) != 1:
            fail(f"players.parquet GSIS {expected_gsis} row count={len(rows)}.")
        if norm_id(rows.iloc[0]["espn_id"]) != espn_id:
            fail(
                f"players.parquet GSIS {expected_gsis} ESPN alias "
                f"{norm_id(rows.iloc[0]['espn_id'])!r} != {espn_id!r}."
            )

    crosswalk_path = repo / config["paths"]["identity_crosswalk"]
    crosswalk = pd.read_parquet(crosswalk_path)
    common.require_columns(
        crosswalk,
        ["player_id", "gsis_id", "espn_id", "current_espn_id", "resolution_status"],
        "player_crosswalk.parquet",
    )
    canonical = crosswalk.loc[crosswalk["gsis_id"].map(norm_id).ne("")].copy()
    if canonical["gsis_id"].map(norm_id).duplicated().any():
        fail("Canonical GSIS records in player_crosswalk.parquet are not unique.")

    for espn_id, expected_gsis in mappings.items():
        mask = (
            crosswalk["espn_id"].map(norm_id).eq(espn_id)
            | crosswalk["current_espn_id"].map(norm_id).eq(espn_id)
        )
        rows = crosswalk.loc[mask]
        if rows.empty:
            fail(f"Crosswalk has no row for ESPN {espn_id}.")
        if set(rows["gsis_id"].map(norm_id)) != {expected_gsis}:
            fail(f"Crosswalk ESPN {espn_id} does not resolve only to {expected_gsis}.")
        if not rows["player_id"].map(norm_id).eq(expected_gsis).all():
            fail(f"Crosswalk ESPN {espn_id} player_id is not canonical GSIS.")
        if not rows["resolution_status"].astype(str).str.casefold().eq("resolved").all():
            fail(f"Crosswalk ESPN {espn_id} is not resolved.")

    build_log = load_json(BUILDER_LOG)
    counts = build_log.get("counts", {})
    if int(counts.get("critical_unresolved_records", -1)) != 0:
        fail(
            f"critical_unresolved_records={counts.get('critical_unresolved_records')}; "
            "expected 0."
        )
    if str(build_log.get("status", "")).casefold() != "passed":
        fail(f"build_player_identity.json status={build_log.get('status')!r}.")

    refresh = load_json(REFRESH_LOG)
    if refresh.get("manual_identity_override") is not False:
        fail("Refresh log does not certify manual_identity_override=false.")
    if refresh.get("canonical_source_repair") is not True:
        fail("Refresh log missing canonical_source_repair=true.")
    if int(refresh.get("builder", {}).get("returncode", -1)) != 0:
        fail("Refresh log records builder failure.")

    logged = refresh.get("mappings", {})
    for espn_id, expected_gsis in mappings.items():
        if norm_id(logged.get(espn_id, {}).get("gsis_id", "")) != expected_gsis:
            fail(f"Refresh log mapping mismatch for ESPN {espn_id}.")

    # No target GSIS ID may be embedded in the repair implementation.
    source = APPLIER.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(APPLIER))
    literals = [
        node.value.strip()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"\d{2}-\d{7,}", node.value.strip())
    ]
    if literals:
        fail(f"Hard-coded GSIS-like target override(s) found: {sorted(set(literals))}")

    builder_source = BUILDER.read_text(encoding="utf-8-sig")
    for espn_id in TARGETS:
        if espn_id in builder_source:
            fail(f"Target ESPN {espn_id} appears directly in builder source.")

    print("authorized_roster_release_hash=true")
    print("roster_identity_rule=team+position+surname->unique_gsis")
    print("authoritative_gsis_mappings=3")
    for espn_id, gsis_id in mappings.items():
        print(f"espn_id={espn_id} name={TARGETS[espn_id]['name']} gsis_id={gsis_id}")
    print("players_source_enriched=true")
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
