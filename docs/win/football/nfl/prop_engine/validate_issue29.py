#!/usr/bin/env python3
"""Independent acceptance validator for Issue 29 current-week universe."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "season", "week", "game_id", "game_date", "game_time", "kickoff_timestamp",
    "player_id", "espn_id", "player_name", "team", "opponent", "position",
    "position_group", "home_flag", "roster_status", "depth_rank",
    "depth_starter_flag", "depth_backup_flag", "depth_injury",
    "injury_game_status", "injury_report_date", "eligibility_status",
    "eligibility_reason", "role_status",
]

DEF_GROUPS = {"DL", "LB", "DB"}
KICKER_POSITIONS = {"K", "PK"}
NONPLAYING_TOKENS = (
    "injured reserve", "injury reserve", "reserve/injured", "reserve injured",
    "practice squad", "physically unable", "pup", "non-football injury",
    "non football injury", "suspension", "suspended", "commissioner exempt",
    "exempt", "retired", "waived", "released", "cut", "unsigned", "inactive",
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def is_nonplaying(value: Any) -> bool:
    text = clean(value).casefold()
    return bool(text) and any(token in text for token in NONPLAYING_TOKENS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def build_team_maps(team_master: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    common.require_columns(team_master, ["team_abbr"], "team master")
    aliases: dict[str, str] = {}
    abbrs: set[str] = set()
    columns = [column for column in ("canonical_team", "team", "alias", "nickname", "shortDisplayName", "team_abbr") if column in team_master.columns]
    for row in team_master.to_dict("records"):
        abbr = common.normalize_team(row.get("team_abbr"))
        if not abbr:
            continue
        abbrs.add(abbr)
        for column in columns:
            value = clean(row.get(column))
            if value:
                aliases.setdefault(value.casefold(), abbr)
    return aliases, abbrs


def resolve_team(value: Any, aliases: dict[str, str], abbrs: set[str]) -> str:
    text = clean(value)
    normalized = common.normalize_team(text)
    if normalized in abbrs:
        return normalized
    return aliases.get(text.casefold(), "")


def schedule_team_games(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    aliases: dict[str, str],
    abbrs: set[str],
) -> dict[str, tuple[str, str, int]]:
    common.require_columns(schedule, ["season", "season_type", "week", "game_id", "home_team", "away_team"], "schedule")
    season_values = pd.to_numeric(schedule["season"], errors="coerce")
    week_values = pd.to_numeric(schedule["week"], errors="coerce")
    frame = schedule[
        season_values.eq(season)
        & week_values.eq(week)
        & schedule["season_type"].astype(str).str.casefold().isin({"reg", "regular", "regular season"})
    ]
    if frame.empty:
        raise AssertionError("Independent validator found no schedule rows")
    out: dict[str, tuple[str, str, int]] = {}
    for row in frame.to_dict("records"):
        home = resolve_team(row["home_team"], aliases, abbrs)
        away = resolve_team(row["away_team"], aliases, abbrs)
        game_id = clean(row["game_id"])
        if not home or not away:
            raise AssertionError(f"Independent schedule team mapping failed for {game_id}")
        if home in out or away in out:
            raise AssertionError("A team appears in multiple target-week games")
        out[home] = (game_id, away, 1)
        out[away] = (game_id, home, 0)
    return out


def static_contract(builder_path: Path) -> dict[str, Any]:
    source = builder_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned_output = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OUTPUT_COLUMNS":
                    assigned_output = ast.literal_eval(node.value)
    if assigned_output != OUTPUT_COLUMNS:
        raise AssertionError("Builder OUTPUT_COLUMNS does not exactly match Issue 29 headers")
    markers = [
        "run_production_audit",
        "roster_weekly_{season}.parquet",
        "critical_unresolved",
        "skipped_unresolved",
        "defensive_no_current_depth_role_or_recent_participation",
        "questionable_flag",
        "verified_nonplaying_roster_or_injury_status",
        "native_current_roster_gsis",
        "materialize_required_current_roster_from_master",
        "roster_master_fallback",
    ]
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise AssertionError(f"Builder missing required policy marker(s): {missing}")
    return {"policy_markers": len(markers)}


def main() -> int:
    args = parse_args()
    config = common.load_config()
    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)
    repo = common.repo_root()
    prop = common.prop_root()

    builder = prop / "scripts" / "project" / "build_current_universe.py"
    output_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    log_path = prop / "logs" / f"current_universe_{season}_week_{week}.json"

    print("CHECK 01: required builder, output, log, and exact header contract")
    for path in (builder, output_path, log_path):
        if not path.is_file():
            raise AssertionError(f"Missing Issue 29 artifact: {path}")
    static = static_contract(builder)
    output = pd.read_parquet(output_path)
    if list(output.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Current universe output headers/order do not match Issue 29")
    if output.empty:
        raise AssertionError("Current universe is empty")

    print("CHECK 02: canonical grain, IDs, scheduled teams, opponent, and home flag")
    common.ensure_unique(output, ["season", "week", "game_id", "player_id"], "Issue 29 output")
    if pd.to_numeric(output["season"], errors="coerce").ne(season).any():
        raise AssertionError("Output contains a different season")
    if pd.to_numeric(output["week"], errors="coerce").ne(week).any():
        raise AssertionError("Output contains a different week")
    if output["player_id"].map(common.normalize_player_id).eq("").any():
        raise AssertionError("Blank canonical GSIS player_id exists")
    if output["player_name"].map(clean).eq("").any():
        raise AssertionError("Blank player_name exists")

    team_master = pd.read_csv(repo / config["paths"]["team_master"], low_memory=False)
    aliases, abbrs = build_team_maps(team_master)
    schedule_path = repo / str(config["paths"]["current_schedule"]).format(season=season, week=week)
    schedule = pd.read_csv(schedule_path, low_memory=False)
    team_games = schedule_team_games(schedule, season=season, week=week, aliases=aliases, abbrs=abbrs)
    bad_game_rows = []
    for row in output.to_dict("records"):
        team = clean(row["team"])
        expected = team_games.get(team)
        if expected is None:
            bad_game_rows.append({"reason": "nonscheduled_team", **{key: row[key] for key in ("game_id", "player_id", "team", "opponent")}})
            continue
        if (clean(row["game_id"]), clean(row["opponent"]), int(row["home_flag"])) != expected:
            bad_game_rows.append({"reason": "game_context_mismatch", **{key: row[key] for key in ("game_id", "player_id", "team", "opponent", "home_flag")}})
    if bad_game_rows:
        raise AssertionError(f"Current universe game context invalid: {bad_game_rows[:10]}")

    print("CHECK 03: eligibility rules for Out, nonplaying, Questionable, defense, and kickers")
    status_values = output["eligibility_status"].astype(str).str.casefold()
    if not status_values.isin({"eligible", "ineligible"}).all():
        raise AssertionError("eligibility_status contains unexpected values")
    out_rows = output["injury_game_status"].astype(str).str.casefold().eq("out")
    if output.loc[out_rows, "eligibility_status"].astype(str).str.casefold().ne("ineligible").any():
        raise AssertionError("Out player remained eligible")
    nonplaying = output.apply(
        lambda row: any(is_nonplaying(row[column]) for column in ("roster_status", "depth_injury", "injury_game_status")), axis=1
    )
    if output.loc[nonplaying, "eligibility_status"].astype(str).str.casefold().ne("ineligible").any():
        raise AssertionError("Verified nonplaying player remained eligible")
    questionable = (
        output["injury_game_status"].astype(str).str.casefold().eq("questionable")
        | output["depth_injury"].astype(str).str.casefold().str.contains("questionable", na=False)
    )
    # Questionable itself must not exclude a player. Independent hard blockers
    # may still do so. In particular, roster_master can carry a medical IR
    # status that is intentionally summarized into eligibility_reason rather
    # than exposed as another Issue 29 output column.
    allowed_questionable_blockers = {
        "out",
        "verified_nonplaying_roster_or_injury_status",
        "defensive_no_current_depth_role_or_recent_participation",
    }
    questionable_ineligible = output.loc[
        questionable & output["eligibility_status"].astype(str).str.casefold().eq("ineligible"),
        "eligibility_reason",
    ].astype(str).str.casefold()
    if not questionable_ineligible.isin(allowed_questionable_blockers).all():
        bad = output.loc[
            questionable
            & output["eligibility_status"].astype(str).str.casefold().eq("ineligible")
            & ~output["eligibility_reason"].astype(str).str.casefold().isin(allowed_questionable_blockers)
        ].head(10).to_dict("records")
        raise AssertionError(
            f"Questionable player was made ineligible without an independent hard blocker: {bad}"
        )
    questionable_eligible = questionable & output["eligibility_status"].astype(str).str.casefold().eq("eligible")
    if output.loc[questionable_eligible, "eligibility_reason"].astype(str).str.casefold().ne("questionable_flag").any():
        raise AssertionError("Questionable eligible rows are not explicitly flagged")

    eligible_def = output[output["position_group"].isin(DEF_GROUPS) & output["eligibility_status"].eq("eligible")]
    bad_def = eligible_def[eligible_def["depth_rank"].isna() & ~eligible_def["role_status"].eq("recent_defensive_participant")]
    if not bad_def.empty:
        raise AssertionError(f"Eligible defender lacks depth/recent participation: {bad_def.head(10).to_dict('records')}")
    special = output[output["position_group"].eq("SPEC")]
    if not special["position"].isin(KICKER_POSITIONS).all():
        raise AssertionError("Non-K/PK special-teams candidate entered current universe")

    print("CHECK 04: unresolved-starter failure and skipped-backup logging contract")
    with log_path.open("r", encoding="utf-8") as handle:
        log = json.load(handle)
    if log.get("status") != "passed":
        raise AssertionError("Issue 29 run log is not passed")
    if log.get("critical_unresolved_starters") != []:
        raise AssertionError("Passed output log contains unresolved starter identity")
    skipped = log.get("skipped_unresolved_nonstarters")
    if not isinstance(skipped, list):
        raise AssertionError("Skipped unresolved nonstarters are not logged")
    for item in skipped:
        if int(item.get("depth_starter_flag", 0)) != 0:
            raise AssertionError("A skipped unresolved record is actually a starter")
    if int(log.get("skipped_unresolved_count", -1)) != len(skipped):
        raise AssertionError("Skipped unresolved log count mismatch")

    print("CHECK 05: required reads and canonical identity evidence")
    required_reads = log.get("required_reads", {})
    expected_paths = {
        "schedule": Path(str(config["paths"]["current_schedule"]).format(season=season, week=week)).as_posix(),
        "roster_master": Path(config["paths"]["current_roster"]).as_posix(),
        "injuries": Path(str(config["paths"]["current_injuries"]).format(season=season, week=week)).as_posix(),
        "player_crosswalk": Path(config["paths"]["identity_crosswalk"]).as_posix(),
        "current_roster_weekly": (Path(config["paths"]["current_source_root"]) / f"roster_weekly_{season}.parquet").as_posix(),
    }
    for key, expected in expected_paths.items():
        if Path(str(required_reads.get(key, ""))).as_posix() != expected:
            raise AssertionError(f"Required read {key} mismatch: {required_reads.get(key)!r} != {expected!r}")
    depth_files = required_reads.get("depth_files")
    if not isinstance(depth_files, list) or len(depth_files) != len(team_games):
        raise AssertionError("Did not read exactly one current depth file per scheduled team")
    for path_text in depth_files:
        if not (repo / path_text).is_file():
            raise AssertionError(f"Logged depth file does not exist: {path_text}")

    crosswalk = pd.read_parquet(repo / config["paths"]["identity_crosswalk"])
    resolved_crosswalk_ids = set(
        crosswalk.loc[
            crosswalk["resolution_status"].astype(str).str.casefold().eq("resolved"), "player_id"
        ].map(common.normalize_player_id)
    )
    native_ids = set(map(common.normalize_player_id, log.get("native_current_gsis_ids_used", [])))
    unexplained = sorted(set(output["player_id"].map(common.normalize_player_id)) - resolved_crosswalk_ids - native_ids)
    if unexplained:
        raise AssertionError(f"Output GSIS IDs lack crosswalk/native-current identity evidence: {unexplained[:20]}")

    current_roster_path = repo / expected_paths["current_roster_weekly"]
    if not current_roster_path.is_file():
        raise AssertionError("Required current roster_weekly source is missing")
    source_info = log.get("current_roster_source", {})
    if source_info.get("mode") not in {"existing_current_source", "roster_master_fallback"}:
        raise AssertionError(f"Unexpected current roster source mode: {source_info!r}")
    if source_info.get("mode") == "roster_master_fallback":
        if source_info.get("gsis_ids_fabricated") is not False:
            raise AssertionError("Fallback current roster source fabricated GSIS IDs")
        if source_info.get("gsis_resolution_deferred_to_crosswalk") is not True:
            raise AssertionError("Fallback current roster did not defer GSIS resolution to crosswalk")
    if native_ids:
        current = pd.read_parquet(current_roster_path)
        native_column = next((column for column in ("gsis_id", "nflverse_player_id") if column in current.columns), None)
        if native_column is None:
            raise AssertionError("Builder reports native GSIS use but current roster has no native GSIS column")
        source_native = set(current[native_column].map(common.normalize_player_id))
        if not native_ids.issubset(source_native):
            raise AssertionError("Logged native GSIS IDs are not present in current roster source")

    print("CHECK 06: market exclusion preflight and output/log policy")
    if log.get("market_exclusion_passed") is not True or log.get("market_features_used") is not False:
        raise AssertionError("Issue 29 did not record successful market exclusion preflight")
    audit_path = prop / "evaluation" / "market_exclusion_audit.json"
    if not audit_path.is_file():
        raise AssertionError("Issue 28 market exclusion audit output is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise AssertionError("Current market exclusion audit is not passing")
    rules = log.get("rules", {})
    required_rule_flags = [
        "nonscheduled_teams_removed", "out_ineligible", "verified_nonplaying_ineligible",
        "questionable_remains_eligible_with_flag", "unresolved_starter_identity_fails",
        "noncritical_unresolved_backup_skipped_and_logged",
        "defensive_requires_depth_or_recent_participation", "kicker_requires_k_or_pk_role",
    ]
    if any(rules.get(key) is not True for key in required_rule_flags):
        raise AssertionError("Issue 29 log does not affirm every required rule")
    if rules.get("canonical_player_id") != "gsis_id" or rules.get("native_gsis_ids_fabricated") is not False:
        raise AssertionError("Canonical GSIS/no-fabrication contract missing")

    print("CHECK 07: summarize independently validated current universe")
    print(f"season={season}")
    print(f"week={week}")
    print(f"games={len(team_games)//2}")
    print(f"scheduled_teams={len(team_games)}")
    print(f"rows={len(output)}")
    print(f"eligible_rows={int(output['eligibility_status'].eq('eligible').sum())}")
    print(f"ineligible_rows={int(output['eligibility_status'].eq('ineligible').sum())}")
    print(f"skipped_unresolved_nonstarters={len(skipped)}")
    print(f"native_current_gsis_ids_used={len(native_ids)}")
    print(f"current_roster_source_mode={source_info.get('mode')}")
    print(f"policy_markers={static['policy_markers']}")
    print("market_features_used=false")
    print("ISSUE 29 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
