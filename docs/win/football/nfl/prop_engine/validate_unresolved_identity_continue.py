#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROP = Path(__file__).resolve().parent
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import common

IDENTITY = PROP / "scripts/build/build_player_identity.py"
UNIVERSE = PROP / "scripts/project/build_current_universe.py"
IDENTITY_LOG = PROP / "logs/build_player_identity.json"


def fail(message: str) -> None:
    raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Required JSON missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        fail(f"Expected JSON object: {path}")
    return value


def static_contract() -> None:
    identity = IDENTITY.read_text(encoding="utf-8-sig")
    universe = UNIVERSE.read_text(encoding="utf-8-sig")

    ast.parse(identity, filename=str(IDENTITY))
    ast.parse(universe, filename=str(UNIVERSE))

    if '"unresolved_identity_policy": "skip_and_continue"' not in identity:
        fail("Identity builder skip-and-continue policy missing.")
    if "Identity validation failed:" in identity:
        fail("Identity builder still has fatal unresolved-player exception.")
    if '"unresolved_identity_policy": "skip_and_continue"' not in universe:
        fail("Current-universe skip-and-continue policy missing.")
    if "Issue 29 unresolved starter identity failure." in universe:
        fail("Current-universe still has fatal unresolved-starter exception.")
    if '"unresolved_starter_identity_fails": False' not in universe:
        fail("Current-universe still declares unresolved starters fatal.")
    if '"unresolved_identity_skipped_and_logged": True' not in universe:
        fail("Current-universe skip-and-log rule missing.")


def validate_identity() -> int:
    config = common.load_config()
    repo = common.repo_root()
    log = load_json(IDENTITY_LOG)

    if log.get("status") != "passed":
        fail(f"build_player_identity status={log.get('status')!r}; expected passed.")
    if log.get("unresolved_identity_policy") != "skip_and_continue":
        fail("Identity log does not record skip_and_continue.")

    critical = log.get("critical_unresolved")
    if not isinstance(critical, list):
        fail("critical_unresolved is not a list.")

    count = int(log.get("counts", {}).get("critical_unresolved_records", -1))
    if count != len(critical):
        fail(
            f"critical_unresolved_records={count}, "
            f"critical_unresolved length={len(critical)}."
        )

    crosswalk_path = repo / config["paths"]["identity_crosswalk"]
    if not crosswalk_path.is_file():
        fail(f"Identity crosswalk missing: {crosswalk_path}")
    crosswalk = pd.read_parquet(crosswalk_path)

    common.require_columns(
        crosswalk,
        ["player_id", "gsis_id", "current_espn_id", "resolution_status"],
        "player_crosswalk.parquet",
    )

    canonical = crosswalk.loc[
        crosswalk["gsis_id"].map(common.normalize_player_id).ne("")
    ].copy()
    common.ensure_unique(canonical, ["gsis_id"], "canonical player crosswalk")

    unresolved = crosswalk.loc[
        crosswalk["resolution_status"].astype(str).str.casefold().eq("unresolved")
    ].copy()
    if not unresolved.empty:
        if unresolved["gsis_id"].map(common.normalize_player_id).ne("").any():
            fail("Unresolved crosswalk row contains a fabricated GSIS.")
        if unresolved["player_id"].map(common.normalize_player_id).ne("").any():
            fail("Unresolved crosswalk row contains a fabricated canonical player_id.")

    return count


def validate_current_week(season: int, week: int) -> tuple[int, int]:
    log_path = PROP / "logs" / f"current_universe_{season}_week_{week}.json"
    output_path = PROP / "data/current" / f"{season}_week_{week}_universe.parquet"

    log = load_json(log_path)
    if log.get("status") != "passed":
        fail(f"current_universe status={log.get('status')!r}; expected passed.")
    if log.get("unresolved_identity_policy") != "skip_and_continue":
        fail("Current-universe log does not record skip_and_continue.")

    rules = log.get("rules", {})
    if rules.get("unresolved_starter_identity_fails") is not False:
        fail("Unresolved starter still configured to fail the week.")
    if rules.get("unresolved_identity_skipped_and_logged") is not True:
        fail("Unresolved identity skip/log rule missing.")

    starters = log.get("critical_unresolved_starters")
    nonstarters = log.get("skipped_unresolved_nonstarters")
    if not isinstance(starters, list) or not isinstance(nonstarters, list):
        fail("Unresolved player log lists are malformed.")

    total = len(starters) + len(nonstarters)
    if int(log.get("critical_unresolved_count", -1)) != len(starters):
        fail("critical_unresolved_count mismatch.")
    if int(log.get("skipped_unresolved_count", -1)) != len(nonstarters):
        fail("skipped_unresolved_count mismatch.")
    if int(log.get("total_unresolved_skipped_count", -1)) != total:
        fail("total_unresolved_skipped_count mismatch.")

    if not output_path.is_file():
        fail(f"Current-week universe missing: {output_path}")
    output = pd.read_parquet(output_path)
    if output.empty:
        fail("Current-week universe is empty.")

    common.require_columns(
        output,
        ["season", "week", "game_id", "player_id", "espn_id"],
        "current-week universe",
    )
    common.ensure_unique(
        output,
        ["season", "week", "game_id", "player_id"],
        "current-week universe",
    )

    if output["player_id"].map(common.normalize_player_id).eq("").any():
        fail("Projection universe contains a blank canonical player_id.")

    skipped_espn = set()
    for item in starters + nonstarters:
        for value in item.get("espn_ids", []):
            value = common.normalize_player_id(value)
            if value:
                skipped_espn.add(value)

    projected_espn = set(output["espn_id"].map(common.normalize_player_id))
    overlap = sorted(skipped_espn & projected_espn)
    if overlap:
        fail(f"Unresolved ESPN identities leaked into projection universe: {overlap[:20]}")

    return total, len(output)


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)

    static_contract()
    identity_warnings = validate_identity()
    skipped, rows = validate_current_week(season, week)

    print("fatal_identity_exits=0")
    print("unresolved_identity_policy=skip_and_continue")
    print(f"identity_unresolved_warnings={identity_warnings}")
    print(f"current_week_unresolved_skipped={skipped}")
    print(f"current_week_projection_rows={rows}")
    print("unresolved_players_in_projection_output=0")
    print("fabricated_gsis_ids=0")
    print("canonical_gsis_unique=true")
    print("UNRESOLVED IDENTITY CONTINUE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"UNRESOLVED IDENTITY CONTINUE VALIDATION: FAIL - {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
