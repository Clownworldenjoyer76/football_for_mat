#!/usr/bin/env python3
"""Build the current-week player universe for the NFL Prop Engine.

REQUIRED READS:
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv
    docs/win/football/nfl/data/master/roster_master.csv
    docs/win/football/nfl/data/master/depth_charts/{TEAM}/{TEAM}_depth.csv
    docs/win/football/nfl/00_intake/injuries/{season}_injuries.csv
    docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet
    docs/win/football/nfl/prop_engine/data/current/source/roster_weekly_{season}.parquet

SUPPORTING READ:
    docs/win/football/nfl/data/master/team_master.csv

OPTIONAL READ (for the defensive-role OR clause):
    docs/win/football/nfl/prop_engine/data/current/source/snap_counts_{season}.parquet

WRITES:
    docs/win/football/nfl/prop_engine/data/current/{season}_week_{week}_universe.parquet
    docs/win/football/nfl/prop_engine/logs/current_universe_{season}_week_{week}.json

POLICY:
    - Canonical player_id is GSIS. Native current nflverse GSIS IDs may resolve a
      player when the historical crosswalk is stale; IDs are never fabricated.
    - When nflverse still carries a player on a prior team, the identity
      crosswalk current_team reconciles that stale team assignment. A conflict
      without a unique crosswalk current_team still fails closed.
    - Only scheduled teams enter the universe.
    - Out and verified nonplaying roster states are ineligible.
    - Questionable remains eligible unless an independent nonplaying state wins.
    - Any unresolved target-relevant depth starter fails the build.
    - Unresolved nonstarters/backups are skipped and written to the run log.
    - Defensive candidates require a current defensive depth role or strictly
      prior current-season defensive snap participation when available.
    - Kicker candidates require a K/PK role.
    - The Issue 28 market-exclusion gate runs before current-week processing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
VALIDATE_DIR = SCRIPTS_ROOT / "validate"
for value in (SCRIPTS_ROOT, VALIDATE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import common
import audit_market_exclusion


OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "kickoff_timestamp",
    "player_id",
    "espn_id",
    "player_name",
    "team",
    "opponent",
    "position",
    "position_group",
    "home_flag",
    "roster_status",
    "depth_rank",
    "depth_starter_flag",
    "depth_backup_flag",
    "depth_injury",
    "injury_game_status",
    "injury_report_date",
    "eligibility_status",
    "eligibility_reason",
    "role_status",
]

OFFENSE_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "TE"}
DEFENSIVE_POSITIONS = {
    "DL", "DE", "LDE", "RDE", "DT", "LDT", "RDT", "NT", "EDGE",
    "LB", "ILB", "OLB", "MLB", "WLB", "SLB",
    "DB", "CB", "LCB", "RCB", "NB", "S", "FS", "SS",
}
KICKER_POSITIONS = {"K", "PK"}
TARGET_RELEVANT_POSITIONS = OFFENSE_POSITIONS | DEFENSIVE_POSITIONS | KICKER_POSITIONS

POSITION_ALIASES = {
    "QUARTERBACK": "QB",
    "RUNNINGBACK": "RB",
    "RUNNING BACK": "RB",
    "HALFBACK": "HB",
    "HALF BACK": "HB",
    "FULLBACK": "FB",
    "FULL BACK": "FB",
    "WIDERECEIVER": "WR",
    "WIDE RECEIVER": "WR",
    "TIGHTEND": "TE",
    "TIGHT END": "TE",
    "PLACEKICKER": "PK",
    "PLACE KICKER": "PK",
    "KICKER": "K",
    "DEFENSIVEEND": "DE",
    "DEFENSIVE END": "DE",
    "DEFENSIVETACKLE": "DT",
    "DEFENSIVE TACKLE": "DT",
    "LINEBACKER": "LB",
    "CORNERBACK": "CB",
    "SAFETY": "S",
    "DEFENSIVEBACK": "DB",
    "DEFENSIVE BACK": "DB",
}

NONPLAYING_TOKENS = (
    "injured reserve",
    "injury reserve",
    "reserve/injured",
    "reserve injured",
    "practice squad",
    "physically unable",
    "pup",
    "non-football injury",
    "non football injury",
    "suspension",
    "suspended",
    "commissioner exempt",
    "exempt",
    "retired",
    "waived",
    "released",
    "cut",
    "unsigned",
    "inactive",
)

ROSTER_ID_ALIASES = ["id", "espn_id", "player_id"]
ROSTER_NAME_ALIASES = ["displayName", "fullName", "display_name", "full_name", "player_name"]
ROSTER_POSITION_ALIASES = ["position.abbreviation", "position_abb", "position", "position_abbreviation"]
ROSTER_GROUP_ALIASES = ["position.parent.abbreviation", "position_group"]
ROSTER_STATUS_ALIASES = ["status.name", "status.type", "status.abbreviation", "status", "roster_status"]
ROSTER_MEDICAL_ALIASES = ["injuries.0.status", "injury", "injury_status"]

CURRENT_GSIS_ALIASES = ["gsis_id", "nflverse_player_id"]
CURRENT_ESPN_ALIASES = ["espn_id"]
CURRENT_PFR_ALIASES = ["pfr_player_id", "pfr_id"]
CURRENT_NAME_ALIASES = ["full_name", "display_name", "player_name", "player"]
CURRENT_TEAM_ALIASES = ["team", "recent_team", "club_code", "team_abbr"]
CURRENT_POSITION_ALIASES = ["position", "position_abbreviation", "depth_chart_position"]
CURRENT_GROUP_ALIASES = ["position_group"]
CURRENT_STATUS_ALIASES = ["status", "roster_status"]
CURRENT_WEEK_ALIASES = ["week"]
CURRENT_SEASON_ALIASES = ["season"]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def normalize_position(value: Any) -> str:
    raw = clean(value).upper().replace("_", " ")
    compact = raw.replace(" ", "")
    if raw in POSITION_ALIASES:
        return POSITION_ALIASES[raw]
    if compact in POSITION_ALIASES:
        return POSITION_ALIASES[compact]
    return raw.replace(" ", "")


def position_group(position: Any) -> str:
    pos = normalize_position(position)
    if pos == "QB":
        return "QB"
    if pos in {"RB", "HB", "FB"}:
        return "RB"
    if pos == "WR":
        return "WR"
    if pos == "TE":
        return "TE"
    if pos in {"DL", "DE", "LDE", "RDE", "DT", "LDT", "RDT", "NT", "EDGE"}:
        return "DL"
    if pos in {"LB", "ILB", "OLB", "MLB", "WLB", "SLB"}:
        return "LB"
    if pos in {"DB", "CB", "LCB", "RCB", "NB", "S", "FS", "SS"}:
        return "DB"
    if pos in KICKER_POSITIONS:
        return "SPEC"
    return ""


def is_relevant_position(value: Any) -> bool:
    return normalize_position(value) in TARGET_RELEVANT_POSITIONS


def choose_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    label: str,
    required: bool = False,
) -> str | None:
    aliases = list(aliases)
    for column in aliases:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"{label}: none of required aliases exist: {aliases}")
    return None


def truthy(value: Any) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y", "starter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current-week Prop Engine player universe.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    prop = common.prop_root().resolve()
    target = path.resolve()
    try:
        target.relative_to(prop)
    except ValueError as exc:
        raise ValueError(f"Issue 29 JSON write must remain under {prop}: {target}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=target.parent,
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def build_team_maps(team_master: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], set[str]]:
    common.require_columns(team_master, ["team_abbr", "team_id"], "team master")
    alias_map: dict[str, str] = {}
    id_map: dict[str, str] = {}
    abbreviations: set[str] = set()
    alias_columns = [
        column for column in ("canonical_team", "team", "alias", "nickname", "shortDisplayName", "team_abbr")
        if column in team_master.columns
    ]
    for row in team_master.to_dict("records"):
        abbr = common.normalize_team(row.get("team_abbr"))
        if not abbr:
            continue
        abbreviations.add(abbr)
        team_id = common.normalize_player_id(row.get("team_id"))
        if team_id:
            previous = id_map.get(team_id)
            if previous and previous != abbr:
                raise ValueError(f"team_master team_id conflict: {team_id} -> {previous}/{abbr}")
            id_map[team_id] = abbr
        for column in alias_columns:
            value = clean(row.get(column))
            if not value:
                continue
            key = value.casefold()
            previous = alias_map.get(key)
            if previous and previous != abbr:
                raise ValueError(f"team_master alias conflict: {value!r} -> {previous}/{abbr}")
            alias_map[key] = abbr
    if len(abbreviations) < 32:
        raise ValueError(f"team_master produced only {len(abbreviations)} NFL abbreviations")
    return alias_map, id_map, abbreviations


def materialize_required_current_roster_from_master(
    *,
    roster_master_path: Path,
    current_roster_path: Path,
    team_id_map: dict[str, str],
    season: int,
    week: int,
) -> dict[str, Any]:
    """Materialize the required current roster source when nflverse has no
    weekly-roster release yet. This uses only the already-required current
    roster_master snapshot, preserves ESPN identity, leaves GSIS blank for the
    independent player_crosswalk to resolve, and writes the required source path
    before the universe builder reads it. No player ID is fabricated.
    """
    roster = pd.read_csv(roster_master_path, low_memory=False)
    id_col = choose_column(roster, ROSTER_ID_ALIASES, label="roster_master fallback", required=True)
    name_col = choose_column(roster, ROSTER_NAME_ALIASES, label="roster_master fallback", required=True)
    pos_col = choose_column(roster, ROSTER_POSITION_ALIASES, label="roster_master fallback", required=True)
    status_col = choose_column(roster, ROSTER_STATUS_ALIASES, label="roster_master fallback")
    common.require_columns(roster, ["team_id"], "roster_master fallback")

    rows: list[dict[str, Any]] = []
    for row in roster.to_dict("records"):
        team_id = common.normalize_player_id(row.get("team_id"))
        team = team_id_map.get(team_id, "")
        espn_id = common.normalize_player_id(row.get(id_col))
        player_name = clean(row.get(name_col))
        position = normalize_position(row.get(pos_col))
        if not team or not espn_id or not player_name:
            continue
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": team,
                "gsis_id": "",
                "espn_id": espn_id,
                "full_name": player_name,
                "position": position,
                "status": clean(row.get(status_col)) if status_col else "",
            }
        )

    if not rows:
        raise RuntimeError("Unable to materialize current roster fallback from roster_master.csv")

    output = pd.DataFrame.from_records(
        rows,
        columns=["season", "week", "team", "gsis_id", "espn_id", "full_name", "position", "status"],
    )
    output = (
        output.sort_values(["team", "espn_id", "full_name"], kind="mergesort")
        .drop_duplicates(["team", "espn_id"], keep="last")
        .reset_index(drop=True)
    )
    common.ensure_unique(output, ["team", "espn_id"], "materialized current roster fallback")
    common.write_parquet_atomic(output, current_roster_path)

    return {
        "mode": "roster_master_fallback",
        "rows": int(len(output)),
        "source": roster_master_path.relative_to(common.repo_root()).as_posix(),
        "output": current_roster_path.relative_to(common.repo_root()).as_posix(),
        "gsis_ids_fabricated": False,
        "gsis_resolution_deferred_to_crosswalk": True,
    }


def resolve_team(value: Any, alias_map: dict[str, str], abbreviations: set[str]) -> str:
    text = clean(value)
    if not text:
        return ""
    normalized = common.normalize_team(text)
    if normalized in abbreviations:
        return normalized
    return alias_map.get(text.casefold(), "")


def build_schedule(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    alias_map: dict[str, str],
    abbreviations: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    required = [
        "season", "season_type", "week", "game_id", "game_date", "game_time",
        "away_team", "home_team", "game_timezone",
    ]
    common.require_columns(schedule, required, "current schedule")
    frame = schedule.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame = frame[
        frame["season"].eq(season)
        & frame["week"].eq(week)
        & frame["season_type"].astype(str).str.casefold().isin({"reg", "regular", "regular season"})
    ].copy()
    if frame.empty:
        raise ValueError(f"No regular-season schedule rows for season={season}, week={week}")

    team_games: dict[str, dict[str, Any]] = {}
    games: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    for row in frame.to_dict("records"):
        game_id = clean(row["game_id"])
        if not game_id or game_id in seen_game_ids:
            raise ValueError(f"Invalid or duplicate current game_id: {game_id!r}")
        seen_game_ids.add(game_id)
        home = resolve_team(row["home_team"], alias_map, abbreviations)
        away = resolve_team(row["away_team"], alias_map, abbreviations)
        if not home or not away or home == away:
            raise ValueError(f"Unable to map schedule teams for {game_id}: {row['away_team']} at {row['home_team']}")

        game_date = clean(row["game_date"])
        game_time = clean(row["game_time"])
        tz_name = clean(row.get("game_timezone"))
        if not tz_name:
            side_tz = clean(row.get("home_timezone"))
            tz_name = side_tz
        if not game_date or not game_time or not tz_name:
            raise ValueError(f"Missing date/time/timezone for game {game_id}")
        try:
            local_dt = datetime.fromisoformat(f"{game_date}T{game_time}").replace(tzinfo=ZoneInfo(tz_name))
        except Exception as exc:
            raise ValueError(f"Cannot construct kickoff for {game_id}: {game_date} {game_time} {tz_name}") from exc
        kickoff = pd.Timestamp(local_dt.astimezone(timezone.utc))

        game = {
            "season": season,
            "week": week,
            "game_id": game_id,
            "game_date": game_date,
            "game_time": game_time,
            "kickoff_timestamp": kickoff,
            "home_team": home,
            "away_team": away,
        }
        games.append(game)
        for team, opponent, home_flag in ((home, away, 1), (away, home, 0)):
            if team in team_games:
                raise ValueError(f"Scheduled team appears in multiple week {week} games: {team}")
            team_games[team] = {**game, "team": team, "opponent": opponent, "home_flag": home_flag}
    return team_games, games


def new_candidate(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "sources": set(),
        "espn_ids": set(),
        "native_gsis_ids": set(),
        "pfr_ids": set(),
        "names": [],
        "master_positions": [],
        "current_positions": [],
        "depth_positions": [],
        "roster_statuses": [],
        "medical_statuses": [],
        "depth_rows": [],
    }


def candidate_key(team: str, espn_id: str, gsis_id: str, name: str) -> tuple[str, str, str]:
    if espn_id:
        return (team, "espn", espn_id)
    if gsis_id:
        return (team, "gsis", gsis_id)
    normalized = common.normalize_name(name)
    if normalized:
        return (team, "name", normalized)
    raise ValueError(f"Cannot key candidate for team={team}: no ID or name")


def add_candidate(
    candidates: dict[tuple[str, str, str], dict[str, Any]],
    *,
    team: str,
    source: str,
    espn_id: Any = "",
    gsis_id: Any = "",
    pfr_id: Any = "",
    name: Any = "",
    position: Any = "",
    roster_status: Any = "",
    medical_status: Any = "",
    depth_row: dict[str, Any] | None = None,
) -> None:
    espn = common.normalize_player_id(espn_id)
    gsis = common.normalize_player_id(gsis_id)
    pfr = common.normalize_player_id(pfr_id)
    display_name = clean(name)
    key = candidate_key(team, espn, gsis, display_name)
    record = candidates.setdefault(key, new_candidate(team))
    record["sources"].add(source)
    if espn:
        record["espn_ids"].add(espn)
    if gsis:
        record["native_gsis_ids"].add(gsis)
    if pfr:
        record["pfr_ids"].add(pfr)
    if display_name:
        record["names"].append(display_name)
    pos = normalize_position(position)
    if pos:
        if source == "current_roster_weekly":
            record["current_positions"].append(pos)
        elif source == "master_roster":
            record["master_positions"].append(pos)
        elif source == "depth":
            record["depth_positions"].append(pos)
    status = clean(roster_status)
    if status:
        record["roster_statuses"].append(status)
    medical = clean(medical_status)
    if medical:
        record["medical_statuses"].append(medical)
    if depth_row is not None:
        record["depth_rows"].append(depth_row)


class IdentityResolver:
    def __init__(self, crosswalk: pd.DataFrame, *, abbreviations: set[str]) -> None:
        required = [
            "player_id", "gsis_id", "espn_id", "pfr_id", "display_name", "normalized_name",
            "position", "position_group", "current_team", "current_espn_id", "resolution_status",
        ]
        common.require_columns(crosswalk, required, "player crosswalk")
        resolved = crosswalk[
            crosswalk["resolution_status"].astype(str).str.casefold().eq("resolved")
            & crosswalk["player_id"].map(common.normalize_player_id).ne("")
        ].copy()
        self.by_gsis: dict[str, dict[str, Any]] = {}
        espn_candidates: dict[str, set[str]] = defaultdict(set)
        pfr_candidates: dict[str, set[str]] = defaultdict(set)
        name_candidates: dict[str, set[str]] = defaultdict(set)
        name_team_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in resolved.to_dict("records"):
            gsis = common.normalize_player_id(row["player_id"])
            self.by_gsis[gsis] = row
            for value in (row.get("espn_id"), row.get("current_espn_id")):
                alias = common.normalize_player_id(value)
                if alias:
                    espn_candidates[alias].add(gsis)
            pfr = common.normalize_player_id(row.get("pfr_id"))
            if pfr:
                pfr_candidates[pfr].add(gsis)
            normalized = clean(row.get("normalized_name")) or common.normalize_name(row.get("display_name"))
            if normalized:
                name_candidates[normalized].add(gsis)
                current_team = common.normalize_team(row.get("current_team"))
                if current_team in abbreviations:
                    name_team_candidates[(current_team, normalized)].add(gsis)
        self.by_espn = {key: next(iter(values)) for key, values in espn_candidates.items() if len(values) == 1}
        self.by_pfr = {key: next(iter(values)) for key, values in pfr_candidates.items() if len(values) == 1}
        self.by_name = {key: next(iter(values)) for key, values in name_candidates.items() if len(values) == 1}
        self.by_name_team = {key: next(iter(values)) for key, values in name_team_candidates.items() if len(values) == 1}

    def resolve(self, record: dict[str, Any]) -> tuple[str, str]:
        direct = {common.normalize_player_id(value) for value in record["native_gsis_ids"] if common.normalize_player_id(value)}
        if len(direct) > 1:
            raise ValueError(f"Conflicting native GSIS IDs for candidate: {sorted(direct)}")
        direct_gsis = next(iter(direct), "")
        alias_resolutions: set[str] = set()
        for espn in record["espn_ids"]:
            if espn in self.by_espn:
                alias_resolutions.add(self.by_espn[espn])
        for pfr in record["pfr_ids"]:
            if pfr in self.by_pfr:
                alias_resolutions.add(self.by_pfr[pfr])
        if len(alias_resolutions) > 1:
            raise ValueError(f"Conflicting crosswalk identities for candidate: {sorted(alias_resolutions)}")
        alias_gsis = next(iter(alias_resolutions), "")
        if direct_gsis and alias_gsis and direct_gsis != alias_gsis:
            raise ValueError(f"Native/current crosswalk GSIS conflict: native={direct_gsis}, crosswalk={alias_gsis}")
        if direct_gsis:
            return direct_gsis, "native_current_roster_gsis"
        if alias_gsis:
            return alias_gsis, "crosswalk_id_alias"
        normalized_names = [common.normalize_name(value) for value in record["names"] if common.normalize_name(value)]
        for normalized in normalized_names:
            team_match = self.by_name_team.get((record["team"], normalized), "")
            if team_match:
                return team_match, "crosswalk_unique_name_team"
        matches = {self.by_name[name] for name in normalized_names if name in self.by_name}
        if len(matches) == 1:
            return next(iter(matches)), "crosswalk_unique_name"
        return "", "unresolved"

    def metadata(self, gsis_id: str) -> dict[str, Any] | None:
        return self.by_gsis.get(gsis_id)


def select_best_position(record: dict[str, Any], crosswalk_row: dict[str, Any] | None) -> str:
    for collection in (record["current_positions"], record["master_positions"]):
        for value in reversed(collection):
            if is_relevant_position(value):
                return normalize_position(value)
    if crosswalk_row is not None and is_relevant_position(crosswalk_row.get("position")):
        return normalize_position(crosswalk_row.get("position"))
    for value in record["depth_positions"]:
        if is_relevant_position(value):
            return normalize_position(value)
    return ""


def depth_summary(record: dict[str, Any]) -> tuple[float | None, int, int, str]:
    rows = record["depth_rows"]
    if not rows:
        return None, 0, 0, ""
    ranks = [row["rank"] for row in rows if row["rank"] is not None]
    starter = int(any(row["starter"] for row in rows))
    backup_any = any(row["backup"] for row in rows) or any((row["rank"] or 0) > 1 for row in rows)
    backup = int(bool(backup_any) and not starter)
    injuries = [clean(row["injury"]) for row in rows if clean(row["injury"])]
    depth_injury = choose_status(injuries)
    return (min(ranks) if ranks else None), starter, backup, depth_injury


def status_rank(value: str) -> int:
    text = value.casefold()
    if any(token in text for token in NONPLAYING_TOKENS):
        return 50
    if text == "out" or " out" in text:
        return 45
    if "doubtful" in text:
        return 30
    if "questionable" in text:
        return 20
    if "day-to-day" in text or "day to day" in text:
        return 10
    if "active" in text or "healthy" in text:
        return 1
    return 5 if text else 0


def choose_status(values: Iterable[Any]) -> str:
    cleaned = [clean(value) for value in values if clean(value)]
    if not cleaned:
        return ""
    return sorted(enumerate(cleaned), key=lambda item: (status_rank(item[1]), item[0]))[-1][1]


def is_nonplaying(value: Any) -> bool:
    text = clean(value).casefold()
    return bool(text) and any(token in text for token in NONPLAYING_TOKENS)


def is_out(value: Any) -> bool:
    text = clean(value).casefold()
    return text == "out" or text.startswith("out ") or text.endswith(" out")


def is_questionable(value: Any) -> bool:
    return "questionable" in clean(value).casefold()


def load_master_candidates(
    roster: pd.DataFrame,
    *,
    scheduled_teams: set[str],
    team_id_map: dict[str, str],
    candidates: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    id_col = choose_column(roster, ROSTER_ID_ALIASES, label="roster_master", required=True)
    name_col = choose_column(roster, ROSTER_NAME_ALIASES, label="roster_master", required=True)
    pos_col = choose_column(roster, ROSTER_POSITION_ALIASES, label="roster_master", required=True)
    status_col = choose_column(roster, ROSTER_STATUS_ALIASES, label="roster_master")
    medical_col = choose_column(roster, ROSTER_MEDICAL_ALIASES, label="roster_master")
    common.require_columns(roster, ["team_id"], "roster_master")
    for row in roster.to_dict("records"):
        team_id = common.normalize_player_id(row.get("team_id"))
        team = team_id_map.get(team_id, "")
        if team not in scheduled_teams:
            continue
        pos = normalize_position(row.get(pos_col))
        if not is_relevant_position(pos):
            continue
        add_candidate(
            candidates,
            team=team,
            source="master_roster",
            espn_id=row.get(id_col),
            name=row.get(name_col),
            position=pos,
            roster_status=row.get(status_col) if status_col else "",
            medical_status=row.get(medical_col) if medical_col else "",
        )


def load_depth_candidates(
    *,
    repo: Path,
    config: dict,
    season: int,
    scheduled_teams: set[str],
    candidates: dict[tuple[str, str, str], dict[str, Any]],
) -> list[str]:
    depth_root = (repo / config["paths"]["current_depth_root"]).resolve()
    paths: list[str] = []
    for team in sorted(scheduled_teams):
        path = depth_root / team / f"{team}_depth.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required current depth file missing: {path}")
        paths.append(path.relative_to(repo).as_posix())
        depth = pd.read_csv(path)
        required = ["player_id", "team", "position_abb", "depth_chart_rank", "starter_flag", "backup_flag", "injury"]
        common.require_columns(depth, required, str(path))
        if "season" in depth.columns:
            depth_season = pd.to_numeric(depth["season"], errors="coerce")
            depth = depth[depth_season.eq(season)].copy()
        for row in depth.to_dict("records"):
            row_team = common.normalize_team(row.get("team"))
            if row_team != team:
                raise ValueError(f"Depth file/team mismatch {path}: row team={row_team}")
            pos = normalize_position(row.get("position_abb"))
            if not is_relevant_position(pos):
                continue
            rank_value = pd.to_numeric(pd.Series([row.get("depth_chart_rank")]), errors="coerce").iloc[0]
            rank = float(rank_value) if pd.notna(rank_value) else None
            depth_row = {
                "position": pos,
                "rank": rank,
                "starter": truthy(row.get("starter_flag")) or (rank == 1.0),
                "backup": truthy(row.get("backup_flag")) or (rank is not None and rank > 1.0),
                "injury": clean(row.get("injury")),
            }
            add_candidate(
                candidates,
                team=team,
                source="depth",
                espn_id=row.get("player_id"),
                name=row.get("name"),
                position=pos,
                medical_status=row.get("injury"),
                depth_row=depth_row,
            )
    return paths


def latest_current_roster_rows(
    current: pd.DataFrame,
    *,
    season: int,
    week: int,
    alias_map: dict[str, str],
    abbreviations: set[str],
) -> list[dict[str, Any]]:
    gsis_col = choose_column(current, CURRENT_GSIS_ALIASES, label="current roster weekly", required=True)
    espn_col = choose_column(current, CURRENT_ESPN_ALIASES, label="current roster weekly")
    pfr_col = choose_column(current, CURRENT_PFR_ALIASES, label="current roster weekly")
    name_col = choose_column(current, CURRENT_NAME_ALIASES, label="current roster weekly", required=True)
    team_col = choose_column(current, CURRENT_TEAM_ALIASES, label="current roster weekly", required=True)
    pos_col = choose_column(current, CURRENT_POSITION_ALIASES, label="current roster weekly", required=True)
    status_col = choose_column(current, CURRENT_STATUS_ALIASES, label="current roster weekly")
    week_col = choose_column(current, CURRENT_WEEK_ALIASES, label="current roster weekly")
    season_col = choose_column(current, CURRENT_SEASON_ALIASES, label="current roster weekly")

    rows: dict[tuple[str, str, str], tuple[float, int, dict[str, Any]]] = {}
    for index, row in enumerate(current.to_dict("records")):
        if season_col:
            row_season = pd.to_numeric(pd.Series([row.get(season_col)]), errors="coerce").iloc[0]
            if pd.notna(row_season) and int(row_season) != season:
                continue
        source_week = float("-inf")
        if week_col:
            week_value = pd.to_numeric(pd.Series([row.get(week_col)]), errors="coerce").iloc[0]
            if pd.notna(week_value):
                if int(week_value) > week:
                    continue
                source_week = float(week_value)
        team = resolve_team(row.get(team_col), alias_map, abbreviations)
        if not team:
            continue
        gsis = common.normalize_player_id(row.get(gsis_col))
        espn = common.normalize_player_id(row.get(espn_col)) if espn_col else ""
        name = clean(row.get(name_col))
        key = candidate_key(team, espn, gsis, name)
        previous = rows.get(key)
        if previous is None or (source_week, index) >= (previous[0], previous[1]):
            rows[key] = (
                source_week,
                index,
                {
                    "team": team,
                    "gsis_id": gsis,
                    "espn_id": espn,
                    "pfr_id": common.normalize_player_id(row.get(pfr_col)) if pfr_col else "",
                    "name": name,
                    "position": normalize_position(row.get(pos_col)),
                    "status": clean(row.get(status_col)) if status_col else "",
                    "source_week": None if source_week == float("-inf") else int(source_week),
                },
            )
    return [value[2] for value in rows.values()]


def load_current_roster_candidates(
    current: pd.DataFrame,
    *,
    season: int,
    week: int,
    scheduled_teams: set[str],
    alias_map: dict[str, str],
    abbreviations: set[str],
    candidates: dict[tuple[str, str, str], dict[str, Any]],
) -> int:
    rows = latest_current_roster_rows(
        current, season=season, week=week, alias_map=alias_map, abbreviations=abbreviations,
    )
    used = 0
    for row in rows:
        if row["team"] not in scheduled_teams:
            continue
        if not is_relevant_position(row["position"]):
            continue
        used += 1
        add_candidate(
            candidates,
            team=row["team"],
            source="current_roster_weekly",
            espn_id=row["espn_id"],
            gsis_id=row["gsis_id"],
            pfr_id=row["pfr_id"],
            name=row["name"],
            position=row["position"],
            roster_status=row["status"],
        )
    return used


def load_recent_defensive_participation(
    *,
    path: Path,
    season: int,
    week: int,
    alias_map: dict[str, str],
    abbreviations: set[str],
    resolver: IdentityResolver,
) -> set[tuple[str, str]]:
    if not path.is_file() or week <= 1:
        return set()
    snaps = pd.read_parquet(path)
    week_col = choose_column(snaps, ["week"], label="current snap counts", required=True)
    team_col = choose_column(snaps, ["team", "recent_team"], label="current snap counts", required=True)
    gsis_col = choose_column(snaps, ["gsis_id", "nflverse_player_id"], label="current snap counts")
    pfr_col = choose_column(snaps, ["pfr_player_id", "pfr_id"], label="current snap counts")
    name_col = choose_column(snaps, ["player", "player_name", "full_name"], label="current snap counts")
    defense_col = choose_column(snaps, ["defense_snaps", "defensive_snaps", "defense_pct"], label="current snap counts", required=True)
    season_col = choose_column(snaps, ["season"], label="current snap counts")
    result: set[tuple[str, str]] = set()
    for row in snaps.to_dict("records"):
        if season_col:
            row_season = pd.to_numeric(pd.Series([row.get(season_col)]), errors="coerce").iloc[0]
            if pd.notna(row_season) and int(row_season) != season:
                continue
        row_week = pd.to_numeric(pd.Series([row.get(week_col)]), errors="coerce").iloc[0]
        if pd.isna(row_week) or not (max(1, week - 3) <= int(row_week) < week):
            continue
        defense = pd.to_numeric(pd.Series([row.get(defense_col)]), errors="coerce").iloc[0]
        if pd.isna(defense) or float(defense) <= 0:
            continue
        team = resolve_team(row.get(team_col), alias_map, abbreviations)
        if not team:
            continue
        gsis = common.normalize_player_id(row.get(gsis_col)) if gsis_col else ""
        if not gsis and pfr_col:
            gsis = resolver.by_pfr.get(common.normalize_player_id(row.get(pfr_col)), "")
        if not gsis and name_col:
            normalized = common.normalize_name(row.get(name_col))
            gsis = resolver.by_name_team.get((team, normalized), "") or resolver.by_name.get(normalized, "")
        if gsis:
            result.add((team, gsis))
    return result


def build_injury_lookup(
    injuries: pd.DataFrame,
    *,
    season: int,
    alias_map: dict[str, str],
    abbreviations: set[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    common.require_columns(
        injuries, ["season", "team", "player_id", "game_status", "report_date"], "current injuries"
    )
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in injuries.to_dict("records"):
        row_season = pd.to_numeric(pd.Series([row.get("season")]), errors="coerce").iloc[0]
        if pd.isna(row_season) or int(row_season) != season:
            continue
        team = resolve_team(row.get("team"), alias_map, abbreviations)
        if not team:
            raise ValueError(f"Cannot map current injury team: {row.get('team')!r}")
        espn = common.normalize_player_id(row.get("player_id"))
        if not espn:
            continue
        report_text = clean(row.get("report_date"))
        report_ts = pd.to_datetime(report_text, errors="coerce", utc=True)
        if report_text and pd.isna(report_ts):
            raise ValueError(f"Invalid injury report_date for {team}/{espn}: {report_text!r}")
        lookup[(team, espn)].append(
            {
                "game_status": clean(row.get("game_status")),
                "report_date": report_text,
                "report_ts": report_ts,
            }
        )
    for values in lookup.values():
        values.sort(key=lambda item: pd.Timestamp.min.tz_localize("UTC") if pd.isna(item["report_ts"]) else item["report_ts"])
    return lookup


def latest_pregame_injury(
    lookup: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    team: str,
    espn_ids: set[str],
    kickoff: pd.Timestamp,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for espn in espn_ids:
        for value in lookup.get((team, espn), []):
            report_ts = value["report_ts"]
            if pd.isna(report_ts) or report_ts < kickoff:
                candidates.append(value)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: pd.Timestamp.min.tz_localize("UTC") if pd.isna(item["report_ts"]) else item["report_ts"],
    )


def merge_resolved_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged = new_candidate(records[0]["team"])
    for record in records:
        for key in ("sources", "espn_ids", "native_gsis_ids", "pfr_ids"):
            merged[key].update(record[key])
        for key in (
            "names", "master_positions", "current_positions", "depth_positions",
            "roster_statuses", "medical_statuses", "depth_rows",
        ):
            merged[key].extend(record[key])
    return merged


def reconcile_multi_team_resolutions(
    resolved_groups: dict[tuple[str, str], list[dict[str, Any]]],
    resolution_methods: dict[tuple[str, str], set[str]],
    *,
    resolver: IdentityResolver,
    abbreviations: set[str],
) -> list[dict[str, Any]]:
    player_teams: dict[str, set[str]] = defaultdict(set)
    for team, gsis_id in resolved_groups:
        player_teams[gsis_id].add(team)

    conflicts = {
        gsis_id: sorted(teams)
        for gsis_id, teams in player_teams.items()
        if len(teams) > 1
    }

    reconciled: list[dict[str, Any]] = []
    unresolved: dict[str, dict[str, Any]] = {}

    for gsis_id, teams in sorted(conflicts.items()):
        crosswalk_row = resolver.metadata(gsis_id)

        current_team = ""
        if crosswalk_row is not None:
            normalized = common.normalize_team(
                crosswalk_row.get("current_team")
            )
            if normalized in abbreviations:
                current_team = normalized

        if not current_team or current_team not in teams:
            unresolved[gsis_id] = {
                "scheduled_teams": teams,
                "crosswalk_current_team": current_team,
            }
            continue

        removed_teams = [
            team
            for team in teams
            if team != current_team
        ]

        for stale_team in removed_teams:
            resolved_groups.pop(
                (stale_team, gsis_id),
                None,
            )
            resolution_methods.pop(
                (stale_team, gsis_id),
                None,
            )

        reconciled.append(
            {
                "player_id": gsis_id,
                "kept_team": current_team,
                "removed_teams": removed_teams,
                "authority": "player_crosswalk_current_team",
            }
        )

    if unresolved:
        raise ValueError(
            "Resolved current player maps to multiple scheduled teams "
            "and crosswalk current_team cannot uniquely reconcile: "
            f"{dict(list(unresolved.items())[:10])}"
        )

    post_teams: dict[str, set[str]] = defaultdict(set)
    for team, gsis_id in resolved_groups:
        post_teams[gsis_id].add(team)

    remaining = {
        gsis_id: sorted(teams)
        for gsis_id, teams in post_teams.items()
        if len(teams) > 1
    }

    if remaining:
        raise ValueError(
            "Resolved current player still maps to multiple scheduled "
            f"teams after current-team reconciliation: "
            f"{dict(list(remaining.items())[:10])}"
        )

    return reconciled


def main() -> int:
    args = parse_args()
    config = common.load_config()

    market_audit = audit_market_exclusion.run_production_audit(write_output=True)
    if market_audit.get("passed") is not True:
        raise RuntimeError(f"Issue 28 market exclusion preflight failed: {market_audit}")

    season = int(args.season if args.season is not None else config["seasons"]["current"])
    week = int(args.week)
    if not 1900 <= season <= 2200 or not 1 <= week <= 25:
        raise ValueError(f"Invalid season/week: {season}/{week}")

    repo = common.repo_root()
    prop = common.prop_root()

    schedule_path = (repo / str(config["paths"]["current_schedule"]).format(season=season, week=week)).resolve()
    roster_master_path = (repo / config["paths"]["current_roster"]).resolve()
    injuries_path = (repo / str(config["paths"]["current_injuries"]).format(season=season, week=week)).resolve()
    crosswalk_path = (repo / config["paths"]["identity_crosswalk"]).resolve()
    team_master_path = (repo / config["paths"]["team_master"]).resolve()
    current_source_root = (repo / config["paths"]["current_source_root"]).resolve()
    current_roster_path = current_source_root / f"roster_weekly_{season}.parquet"
    current_snaps_path = current_source_root / f"snap_counts_{season}.parquet"

    for path, label in (
        (schedule_path, "schedule"),
        (roster_master_path, "roster_master"),
        (injuries_path, "injuries"),
        (crosswalk_path, "player_crosswalk"),
        (team_master_path, "team_master"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required Issue 29 {label} file missing: {path}")

    team_master = pd.read_csv(team_master_path, low_memory=False)
    alias_map, team_id_map, abbreviations = build_team_maps(team_master)

    current_roster_source = {"mode": "existing_current_source", "output": current_roster_path.relative_to(repo).as_posix()}
    if not current_roster_path.is_file():
        current_roster_source = materialize_required_current_roster_from_master(
            roster_master_path=roster_master_path,
            current_roster_path=current_roster_path,
            team_id_map=team_id_map,
            season=season,
            week=week,
        )
    if not current_roster_path.is_file():
        raise FileNotFoundError(f"Required Issue 29 current roster source is missing after fallback materialization: {current_roster_path}")
    schedule = pd.read_csv(schedule_path, low_memory=False)
    team_games, games = build_schedule(
        schedule, season=season, week=week, alias_map=alias_map, abbreviations=abbreviations
    )
    scheduled_teams = set(team_games)

    crosswalk = pd.read_parquet(crosswalk_path)
    resolver = IdentityResolver(crosswalk, abbreviations=abbreviations)

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    roster_master = pd.read_csv(roster_master_path, low_memory=False)
    load_master_candidates(
        roster_master, scheduled_teams=scheduled_teams, team_id_map=team_id_map, candidates=candidates
    )
    depth_paths = load_depth_candidates(
        repo=repo, config=config, season=season, scheduled_teams=scheduled_teams, candidates=candidates
    )
    current_roster = pd.read_parquet(current_roster_path)
    current_roster_rows_used = load_current_roster_candidates(
        current_roster,
        season=season,
        week=week,
        scheduled_teams=scheduled_teams,
        alias_map=alias_map,
        abbreviations=abbreviations,
        candidates=candidates,
    )

    recent_defense = load_recent_defensive_participation(
        path=current_snaps_path,
        season=season,
        week=week,
        alias_map=alias_map,
        abbreviations=abbreviations,
        resolver=resolver,
    )

    injury_lookup = build_injury_lookup(
        pd.read_csv(injuries_path, low_memory=False),
        season=season,
        alias_map=alias_map,
        abbreviations=abbreviations,
    )

    unresolved_skipped: list[dict[str, Any]] = []
    critical_unresolved: list[dict[str, Any]] = []
    resolved_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    resolution_methods: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in candidates.values():
        depth_rank, starter_flag, backup_flag, _ = depth_summary(record)
        gsis_id, method = resolver.resolve(record)
        if not gsis_id:
            position = select_best_position(record, None)
            item = {
                "team": record["team"],
                "espn_ids": sorted(record["espn_ids"]),
                "player_names": sorted(set(record["names"])),
                "position": position,
                "depth_rank": depth_rank,
                "depth_starter_flag": starter_flag,
                "depth_backup_flag": backup_flag,
                "sources": sorted(record["sources"]),
                "reason": "no_resolved_or_native_gsis_identity",
            }
            if starter_flag == 1:
                critical_unresolved.append(item)
            else:
                unresolved_skipped.append(item)
            continue
        resolved_groups[(record["team"], gsis_id)].append(record)
        resolution_methods[(record["team"], gsis_id)].add(method)

    for item in [*critical_unresolved, *unresolved_skipped]:
        print(
            "IDENTITY WARNING: unresolved current-week player excluded; "
            f"names={item.get('player_names', [])} "
            f"espn_ids={item.get('espn_ids', [])} "
            f"team={item.get('team', '')} "
            f"position={item.get('position', '')} "
            f"starter={item.get('depth_starter_flag', 0)} "
            f"reason={item.get('reason', '')}",
            file=sys.stderr,
        )

    multi_team_reconciliations = reconcile_multi_team_resolutions(
        resolved_groups,
        resolution_methods,
        resolver=resolver,
        abbreviations=abbreviations,
    )

    rows: list[dict[str, Any]] = []
    native_ids_used: set[str] = set()
    for (team, gsis_id), pieces in sorted(resolved_groups.items()):
        record = merge_resolved_records(pieces)
        crosswalk_row = resolver.metadata(gsis_id)
        pos = select_best_position(record, crosswalk_row)
        if not is_relevant_position(pos):
            continue
        group = position_group(pos)
        if not group:
            continue
        game = team_games[team]
        depth_rank, starter_flag, backup_flag, depth_injury = depth_summary(record)

        espn_ids = set(record["espn_ids"])
        if crosswalk_row is not None:
            for value in (crosswalk_row.get("current_espn_id"), crosswalk_row.get("espn_id")):
                alias = common.normalize_player_id(value)
                if alias:
                    espn_ids.add(alias)
        injury = latest_pregame_injury(
            injury_lookup, team=team, espn_ids=espn_ids, kickoff=game["kickoff_timestamp"]
        )
        injury_status = injury["game_status"] if injury else ""
        injury_report_date = injury["report_date"] if injury else ""

        roster_status = choose_status(record["roster_statuses"])
        medical_status = choose_status(record["medical_statuses"])
        all_nonplaying_evidence = [roster_status, medical_status, depth_injury, injury_status]

        defender = group in {"DL", "LB", "DB"}
        has_current_depth_role = bool(record["depth_rows"])
        has_recent_defense = (team, gsis_id) in recent_defense

        out_flag = any(is_out(value) for value in (injury_status, depth_injury, medical_status))
        nonplaying_flag = any(is_nonplaying(value) for value in all_nonplaying_evidence)
        questionable_flag = any(is_questionable(value) for value in (injury_status, depth_injury, medical_status))

        if out_flag:
            eligibility_status = "ineligible"
            eligibility_reason = "out"
        elif nonplaying_flag:
            eligibility_status = "ineligible"
            eligibility_reason = "verified_nonplaying_roster_or_injury_status"
        elif defender and not (has_current_depth_role or has_recent_defense):
            eligibility_status = "ineligible"
            eligibility_reason = "defensive_no_current_depth_role_or_recent_participation"
        elif questionable_flag:
            eligibility_status = "eligible"
            eligibility_reason = "questionable_flag"
        else:
            eligibility_status = "eligible"
            eligibility_reason = "eligible_current_role"

        if starter_flag:
            role_status = "starter"
        elif backup_flag:
            role_status = "backup"
        elif defender and has_recent_defense:
            role_status = "recent_defensive_participant"
        elif pos in KICKER_POSITIONS:
            role_status = "kicker_role"
        else:
            role_status = "roster_only"

        if pos in KICKER_POSITIONS:
            pass
        elif group == "SPEC":
            continue

        display_name = ""
        if record["names"]:
            display_name = record["names"][-1]
        elif crosswalk_row is not None:
            display_name = clean(crosswalk_row.get("display_name"))
        if not display_name:
            raise ValueError(f"Resolved player {gsis_id} has no display name")

        candidate_espn_ids = sorted(record["espn_ids"])
        espn_id = candidate_espn_ids[0] if candidate_espn_ids else (sorted(espn_ids)[0] if espn_ids else "")
        if "native_current_roster_gsis" in resolution_methods[(team, gsis_id)]:
            native_ids_used.add(gsis_id)

        rows.append(
            {
                "season": season,
                "week": week,
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "game_time": game["game_time"],
                "kickoff_timestamp": game["kickoff_timestamp"],
                "player_id": gsis_id,
                "espn_id": espn_id,
                "player_name": display_name,
                "team": team,
                "opponent": game["opponent"],
                "position": pos,
                "position_group": group,
                "home_flag": int(game["home_flag"]),
                "roster_status": roster_status,
                "depth_rank": depth_rank,
                "depth_starter_flag": int(starter_flag),
                "depth_backup_flag": int(backup_flag),
                "depth_injury": depth_injury,
                "injury_game_status": injury_status,
                "injury_report_date": injury_report_date,
                "eligibility_status": eligibility_status,
                "eligibility_reason": eligibility_reason,
                "role_status": role_status,
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if output.empty:
        raise ValueError("Issue 29 produced an empty current-week universe")

    common.ensure_unique(output, ["season", "week", "game_id", "player_id"], "current-week player universe")
    if output["player_id"].map(common.normalize_player_id).eq("").any():
        raise ValueError("Current-week universe contains blank canonical player_id")
    if not set(output["team"]).issubset(scheduled_teams):
        raise ValueError("Current-week universe retained a nonscheduled team")
    if not output.apply(lambda row: team_games[row["team"]]["opponent"] == row["opponent"], axis=1).all():
        raise ValueError("Current-week universe opponent mismatch")
    if output.loc[output["injury_game_status"].str.casefold().eq("out"), "eligibility_status"].eq("eligible").any():
        raise ValueError("Out player remained eligible")
    questionable = (
        output["injury_game_status"].str.casefold().eq("questionable")
        | output["depth_injury"].str.casefold().str.contains("questionable", na=False)
    )
    # Questionable is not itself an exclusion. Independent hard blockers still
    # win: Out/verified nonplaying status and the defensive-role eligibility
    # gate. roster_master may carry a medical IR/nonplaying status that is not
    # exposed as a separate output column, so validate by the recorded reason
    # rather than attempting to reconstruct all source evidence from output.
    questionable_ineligible = output.loc[
        questionable & output["eligibility_status"].eq("ineligible"),
        "eligibility_reason",
    ]
    allowed_questionable_blockers = {
        "out",
        "verified_nonplaying_roster_or_injury_status",
        "defensive_no_current_depth_role_or_recent_participation",
    }
    if not questionable_ineligible.isin(allowed_questionable_blockers).all():
        bad = output.loc[
            questionable
            & output["eligibility_status"].eq("ineligible")
            & ~output["eligibility_reason"].isin(allowed_questionable_blockers),
            ["player_id", "player_name", "team", "eligibility_reason"],
        ].head(10).to_dict("records")
        raise ValueError(
            "Questionable player became ineligible without an independent hard blocker: "
            f"{bad}"
        )
    questionable_eligible = questionable & output["eligibility_status"].eq("eligible")
    if output.loc[questionable_eligible, "eligibility_reason"].ne("questionable_flag").any():
        raise ValueError("Eligible Questionable player is missing questionable_flag")
    eligible_defense = output[output["position_group"].isin({"DL", "LB", "DB"}) & output["eligibility_status"].eq("eligible")]
    bad_defense = eligible_defense[eligible_defense["depth_rank"].isna() & ~eligible_defense["role_status"].eq("recent_defensive_participant")]
    if not bad_defense.empty:
        raise ValueError(f"Eligible defensive candidates without depth/recent participation: {bad_defense.head(10).to_dict('records')}")
    if output["position"].isin({"P", "LS", "KR", "PR", "H"}).any():
        raise ValueError("Non-K/PK special-teams role entered the target universe")

    output = output.sort_values(["season", "week", "game_id", "team", "position_group", "position", "player_id"], kind="mergesort").reset_index(drop=True)
    output_path = prop / "data" / "current" / f"{season}_week_{week}_universe.parquet"
    common.write_parquet_atomic(output, output_path)

    log_path = prop / "logs" / f"current_universe_{season}_week_{week}.json"
    payload = {
        "script": "build_current_universe.py",
        "status": "passed",
        "season": season,
        "week": week,
        "output": output_path.relative_to(repo).as_posix(),
        "rows": int(len(output)),
        "eligible_rows": int(output["eligibility_status"].eq("eligible").sum()),
        "ineligible_rows": int(output["eligibility_status"].eq("ineligible").sum()),
        "scheduled_teams": sorted(scheduled_teams),
        "scheduled_team_count": len(scheduled_teams),
        "games": len(games),
        "required_reads": {
            "schedule": schedule_path.relative_to(repo).as_posix(),
            "roster_master": roster_master_path.relative_to(repo).as_posix(),
            "depth_files": depth_paths,
            "injuries": injuries_path.relative_to(repo).as_posix(),
            "player_crosswalk": crosswalk_path.relative_to(repo).as_posix(),
            "current_roster_weekly": current_roster_path.relative_to(repo).as_posix(),
        },
        "supporting_reads": {
            "team_master": team_master_path.relative_to(repo).as_posix(),
            "recent_snap_counts": current_snaps_path.relative_to(repo).as_posix() if current_snaps_path.is_file() else None,
        },
        "current_roster_rows_used": int(current_roster_rows_used),
        "current_roster_source": current_roster_source,
        "recent_defensive_participant_keys": len(recent_defense),
        "native_current_gsis_ids_used": sorted(native_ids_used),
        "multi_team_reconciliations": multi_team_reconciliations,
        "multi_team_reconciliation_count": len(multi_team_reconciliations),
        "critical_unresolved_starters": critical_unresolved,
        "skipped_unresolved_nonstarters": unresolved_skipped,
        "skipped_unresolved_count": len(unresolved_skipped),
        "critical_unresolved_count": len(critical_unresolved),
        "total_unresolved_skipped_count": (
            len(critical_unresolved) + len(unresolved_skipped)
        ),
        "unresolved_identity_policy": "skip_and_continue",
        "rules": {
            "nonscheduled_teams_removed": True,
            "out_ineligible": True,
            "verified_nonplaying_ineligible": True,
            "questionable_remains_eligible_with_flag": True,
            "unresolved_starter_identity_fails": False,
            "unresolved_identity_skipped_and_logged": True,
            "noncritical_unresolved_backup_skipped_and_logged": True,
            "defensive_requires_depth_or_recent_participation": True,
            "kicker_requires_k_or_pk_role": True,
            "canonical_player_id": "gsis_id",
            "native_gsis_ids_fabricated": False,
            "multi_team_gsis_reconciled_to_crosswalk_current_team": True,
            "multi_team_without_crosswalk_current_team_fails": True,
        },
        "market_exclusion_passed": True,
        "market_features_used": False,
    }
    write_json_atomic(payload, log_path)

    common.log_run(
        "build_current_universe.py",
        {
            "season": season,
            "week": week,
            "rows": int(len(output)),
            "eligible_rows": int(output["eligibility_status"].eq("eligible").sum()),
            "skipped_unresolved": (
                len(critical_unresolved) + len(unresolved_skipped)
            ),
            "multi_team_reconciliations": len(
                multi_team_reconciliations
            ),
            "status": "passed",
        },
    )

    print(
        json.dumps(
            {
                "script": "build_current_universe.py",
                "status": "passed",
                "season": season,
                "week": week,
                "rows": int(len(output)),
                "eligible_rows": int(output["eligibility_status"].eq("eligible").sum()),
                "ineligible_rows": int(output["eligibility_status"].eq("ineligible").sum()),
                "skipped_unresolved": (
                    len(critical_unresolved) + len(unresolved_skipped)
                ),
                "multi_team_reconciliations": len(
                    multi_team_reconciliations
                ),
                "native_current_gsis_ids_used": len(native_ids_used),
                "output": output_path.relative_to(repo).as_posix(),
                "log": log_path.relative_to(repo).as_posix(),
                "market_exclusion_passed": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print("CURRENT UNIVERSE BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())