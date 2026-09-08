#!/usr/bin/env python3
"""Repair three current ESPN aliases from authoritative 2026 NFL roster GSIS data.

The issue-provided ESPN IDs are aliases only. Canonical GSIS IDs are never
hard-coded; each is derived from the authorized NFL Shield-derived weekly roster
using team + position + surname uniqueness, then written into the authorized
players.parquet identity source before the existing identity builder is rerun.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROP = Path(__file__).resolve().parent
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import common

PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.parquet"
PLAYERS_RELEASE_SHA256 = "b38d690910364b9d7e0df46d7bb1dbbe44ec6b7142b3f7ce1cbcab142395a208"
ROSTER_RELEASE_SHA256 = "41c188f55571ae756950d44e5c8e192f05692f4fc594106491f43228ed9a57b2"

TARGETS = {
    "4599158": {"name": "Al-Jay Henderson", "surname": "henderson", "position": "RB", "team": "NYJ"},
    "4691889": {"name": "Greg Desrosiers Jr.", "surname": "desrosiers", "position": "RB", "team": "LAC"},
    "4699678": {"name": "Khalil Dinkins", "surname": "dinkins", "position": "TE", "team": "SF"},
}

PLAYERS_REL = Path("docs/win/football/nfl/data/historic_data/players/players.parquet")
ROSTER_REL = Path(
    "docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_2026.parquet"
)
REFRESH_LOG = PROP / "logs/refresh_authoritative_identity_sources_2026.json"
BUILDER = PROP / "scripts/build/build_player_identity.py"

NAME_ALIASES = ("full_name", "display_name", "player_name", "football_name")
POSITION_ALIASES = ("position", "depth_chart_position", "ngs_position")
TEAM_ALIASES = ("team", "team_abbr", "team_code", "club_code", "recent_team")
GSIS_ALIASES = ("gsis_id", "player_id")


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


def choose(df: pd.DataFrame, aliases, label: str) -> str:
    for column in aliases:
        if column in df.columns:
            return column
    raise ValueError(f"{label}: none of {list(aliases)} exist.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, expected_sha: str, directory: Path, label: str):
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{label}.", suffix=".download.tmp",
        dir=directory, delete=False
    )
    temp = Path(handle.name)
    handle.close()
    digest = hashlib.sha256()
    request = urllib.request.Request(
        url, headers={"User-Agent": "football-for-mat-prop-engine-identity-repair"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temp.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha:
            raise ValueError(f"{label}: SHA-256 {observed} != official {expected_sha}.")
        return temp
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def roster_identity_columns(roster: pd.DataFrame):
    return {
        "name": choose(roster, NAME_ALIASES, "2026 roster name"),
        "position": choose(roster, POSITION_ALIASES, "2026 roster position"),
        "team": choose(roster, TEAM_ALIASES, "2026 roster team"),
        "gsis": choose(roster, GSIS_ALIASES, "2026 roster GSIS"),
    }


def derive_roster_mappings(roster: pd.DataFrame):
    cols = roster_identity_columns(roster)
    work = roster.copy()
    work["_name_norm"] = work[cols["name"]].map(common.normalize_name)
    work["_position_norm"] = work[cols["position"]].map(lambda x: clean(x).upper())
    work["_team_norm"] = work[cols["team"]].map(common.normalize_team)
    work["_gsis_norm"] = work[cols["gsis"]].map(norm_id)

    resolved = {}
    for espn_id, target in TARGETS.items():
        surname = target["surname"]
        candidate = work.loc[
            work["_team_norm"].eq(common.normalize_team(target["team"]))
            & work["_position_norm"].eq(target["position"])
            & work["_name_norm"].str.split().map(
                lambda tokens: surname in tokens if isinstance(tokens, list) else False
            )
            & work["_gsis_norm"].ne("")
        ].copy()

        # Weekly roster repeats a player by week, so uniqueness is by GSIS identity.
        gsis_ids = sorted(set(candidate["_gsis_norm"]))
        if len(gsis_ids) != 1:
            sample_cols = [cols["name"], cols["position"], cols["team"], cols["gsis"]]
            sample = candidate[sample_cols].drop_duplicates().head(20).to_dict("records")
            raise ValueError(
                f"{target['name']}: expected exactly one authoritative roster GSIS "
                f"for team={target['team']} position={target['position']} "
                f"surname={surname}; got {gsis_ids}. sample={sample}"
            )

        gsis_id = gsis_ids[0]
        source_names = sorted(set(clean(x) for x in candidate[cols["name"]] if clean(x)))
        resolved[espn_id] = {
            **target,
            "gsis_id": gsis_id,
            "roster_names": source_names,
            "roster_rows": int(len(candidate)),
            "resolution_rule": "authorized_roster_team_position_surname_unique_gsis",
        }
    return resolved


def enrich_players(players: pd.DataFrame, mappings):
    common.require_columns(players, ["gsis_id", "espn_id"], "players.parquet")
    result = players.copy()
    gsis_norm = result["gsis_id"].map(norm_id)

    for espn_id, mapping in mappings.items():
        matches = result.index[gsis_norm.eq(mapping["gsis_id"])].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"{mapping['name']}: authoritative players.parquet must contain exactly "
                f"one row for derived GSIS {mapping['gsis_id']}; rows={len(matches)}."
            )

        idx = matches[0]
        existing = norm_id(result.at[idx, "espn_id"])
        if existing and existing != espn_id:
            raise ValueError(
                f"{mapping['name']}: players.parquet GSIS {mapping['gsis_id']} already "
                f"has conflicting ESPN alias {existing}; refusing overwrite."
            )
        result.at[idx, "espn_id"] = espn_id

    # Preserve canonical GSIS uniqueness in the source being repaired.
    nonblank = result.loc[result["gsis_id"].map(norm_id).ne("")].copy()
    if nonblank["gsis_id"].map(norm_id).duplicated().any():
        raise ValueError("Repaired players.parquet is not unique by nonblank gsis_id.")

    return result


def atomic_write_parquet(frame: pd.DataFrame, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False
    )
    temp = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def write_json(payload, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()
    players_dest = (repo / PLAYERS_REL).resolve()
    roster_dest = (repo / ROSTER_REL).resolve()

    if Path(config["paths"]["historical_players"]) != PLAYERS_REL:
        raise ValueError("Configured historical_players path differs from authorized path.")
    if Path(str(config["paths"]["historical_rosters_pattern"]).format(season=2026)) != ROSTER_REL:
        raise ValueError("Configured 2026 weekly roster path differs from authorized path.")

    players_temp = roster_temp = None
    try:
        players_temp = download_verified(
            PLAYERS_URL, PLAYERS_RELEASE_SHA256, players_dest.parent, "players"
        )
        roster_temp = download_verified(
            ROSTER_URL, ROSTER_RELEASE_SHA256, roster_dest.parent, "roster_weekly_2026"
        )

        players_release = pd.read_parquet(players_temp)
        roster_release = pd.read_parquet(roster_temp)

        mappings = derive_roster_mappings(roster_release)
        repaired_players = enrich_players(players_release, mappings)

        # Write only the two authorized repository identity sources.
        os.replace(roster_temp, roster_dest)
        roster_temp = None
        atomic_write_parquet(repaired_players, players_dest)
        players_temp.unlink()
        players_temp = None

        repaired_players_sha = sha256_file(players_dest)
        if sha256_file(roster_dest) != ROSTER_RELEASE_SHA256:
            raise RuntimeError("Authorized roster hash changed unexpectedly.")

        completed = subprocess.run(
            [sys.executable, str(BUILDER)],
            cwd=str(repo), capture_output=True, text=True, check=False
        )

        payload = {
            "script": "apply_gsis_identity_source_refresh.py",
            "status": "passed" if completed.returncode == 0 else "failed",
            "refreshed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "manual_identity_override": False,
            "canonical_source_repair": True,
            "target_count": 3,
            "mappings": mappings,
            "sources": {
                "players_release": {
                    "url": PLAYERS_URL,
                    "release_sha256": PLAYERS_RELEASE_SHA256,
                },
                "weekly_roster_2026": {
                    "url": ROSTER_URL,
                    "release_sha256": ROSTER_RELEASE_SHA256,
                    "repository_sha256": sha256_file(roster_dest),
                },
                "repaired_players": {
                    "repository_path": str(PLAYERS_REL).replace("\\", "/"),
                    "repository_sha256": repaired_players_sha,
                },
            },
            "builder": {
                "path": str(BUILDER.relative_to(repo)).replace("\\", "/"),
                "returncode": int(completed.returncode),
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-12000:],
            },
        }
        write_json(payload, REFRESH_LOG)

        for espn_id, mapping in mappings.items():
            print(
                f"espn_id={espn_id} name={mapping['name']} "
                f"authoritative_gsis={mapping['gsis_id']} "
                f"rule={mapping['resolution_rule']}"
            )
        print(f"repaired_players_sha256={repaired_players_sha}")
        print("manual_identity_override=false")

        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"build_player_identity.py failed with return code {completed.returncode}."
            )

        print("AUTHORITATIVE GSIS SOURCE REPAIR: PASS")
        return 0
    finally:
        for temp in (players_temp, roster_temp):
            if temp is not None and temp.exists():
                temp.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
