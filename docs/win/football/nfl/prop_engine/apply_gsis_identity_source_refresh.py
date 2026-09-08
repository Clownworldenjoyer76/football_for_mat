#!/usr/bin/env python3
"""Repair three ESPN aliases using live authoritative nflverse identity sources.

GSIS IDs are never hard-coded. For each target ESPN identity, the canonical
GSIS is derived from the current official nflverse players and/or 2026 weekly
roster release. The result is written only into the two authorized upstream
identity parquet sources before the existing player-identity builder is run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
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

RELEASE_API = (
    "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
)
TARGETS = {
    "4599158": {
        "name": "Al-Jay Henderson",
        "surname": "henderson",
        "position": "RB",
        "team": "NYJ",
    },
    "4691889": {
        "name": "Greg Desrosiers Jr.",
        "surname": "desrosiers",
        "position": "RB",
        "team": "LAC",
    },
    "4699678": {
        "name": "Khalil Dinkins",
        "surname": "dinkins",
        "position": "TE",
        "team": "SF",
    },
}

PLAYERS_REL = Path(
    "docs/win/football/nfl/data/historic_data/players/players.parquet"
)
ROSTER_REL = Path(
    "docs/win/football/nfl/data/historic_data/weekly_rosters/"
    "roster_weekly_2026.parquet"
)
REFRESH_LOG = PROP / "logs/refresh_authoritative_identity_sources_2026.json"
BUILDER = PROP / "scripts/build/build_player_identity.py"

NAME_ALIASES = (
    "full_name",
    "display_name",
    "player_name",
    "football_name",
)
POSITION_ALIASES = (
    "position",
    "depth_chart_position",
    "ngs_position",
)
TEAM_ALIASES = (
    "team",
    "team_abbr",
    "team_code",
    "club_code",
    "recent_team",
    "latest_team",
)
GSIS_ALIASES = ("gsis_id", "player_id")
ESPN_ALIASES = ("espn_id", "current_espn_id")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def norm_id(value: Any) -> str:
    return common.normalize_player_id(value)


def choose(df: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "football-for-mat-prop-engine-identity-repair",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def current_release_asset(tag: str, asset_name: str) -> dict[str, str]:
    release = request_json(RELEASE_API.format(tag=tag))
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"{tag}: release assets missing")

    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == asset_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{tag}: expected exactly one {asset_name}; found {len(matches)}"
        )

    asset = matches[0]
    digest = clean(asset.get("digest"))
    if not digest.startswith("sha256:"):
        raise ValueError(f"{tag}/{asset_name}: GitHub SHA-256 digest missing")

    return {
        "tag": tag,
        "name": asset_name,
        "url": clean(asset.get("browser_download_url")),
        "sha256": digest.split(":", 1)[1],
        "asset_id": str(asset.get("id")),
        "created_at": clean(asset.get("created_at")),
        "updated_at": clean(asset.get("updated_at")),
        "release_updated_at": clean(release.get("updated_at")),
    }


def download_verified(asset: dict[str, str], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{asset['name']}.",
        suffix=".download.tmp",
        dir=directory,
        delete=False,
    )
    temp = Path(handle.name)
    handle.close()

    digest = hashlib.sha256()
    request = urllib.request.Request(
        asset["url"],
        headers={"User-Agent": "football-for-mat-prop-engine-identity-repair"},
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
        if observed != asset["sha256"]:
            raise ValueError(
                f"{asset['name']}: downloaded SHA-256 {observed} "
                f"!= GitHub release digest {asset['sha256']}"
            )
        return temp
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def identity_columns(df: pd.DataFrame, label: str) -> dict[str, str | None]:
    cols = {
        "name": choose(df, NAME_ALIASES, label),
        "position": choose(df, POSITION_ALIASES, label),
        "team": choose(df, TEAM_ALIASES, label),
        "gsis": choose(df, GSIS_ALIASES, label),
        "espn": choose(df, ESPN_ALIASES, label),
    }
    if cols["gsis"] is None:
        raise ValueError(f"{label}: GSIS column missing")
    return cols


def normalized_identity(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    cols = identity_columns(df, label)
    work = df.copy()

    work["_name"] = (
        work[cols["name"]].map(common.normalize_name)
        if cols["name"]
        else ""
    )
    work["_position"] = (
        work[cols["position"]].map(lambda x: clean(x).upper())
        if cols["position"]
        else ""
    )
    work["_team"] = (
        work[cols["team"]].map(common.normalize_team)
        if cols["team"]
        else ""
    )
    work["_gsis"] = work[cols["gsis"]].map(norm_id)
    work["_espn"] = (
        work[cols["espn"]].map(norm_id)
        if cols["espn"]
        else ""
    )
    return work, cols


def unique_gsis(rows: pd.DataFrame) -> list[str]:
    return sorted(
        {
            norm_id(value)
            for value in rows["_gsis"]
            if norm_id(value)
        }
    )


def roster_surname_mask(series: pd.Series, surname: str) -> pd.Series:
    return series.str.split().map(
        lambda tokens: surname in tokens if isinstance(tokens, list) else False
    )


def derive_mappings(
    players: pd.DataFrame,
    roster: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    p, pcols = normalized_identity(players, "players.parquet")
    r, rcols = normalized_identity(roster, "roster_weekly_2026.parquet")

    resolved: dict[str, dict[str, Any]] = {}

    for espn_id, target in TARGETS.items():
        target_name = common.normalize_name(target["name"])
        target_team = common.normalize_team(target["team"])
        target_position = target["position"].upper()
        evidence: list[dict[str, Any]] = []

        # Strongest: exact ESPN alias in either official source.
        for source_name, work in (("players", p), ("roster", r)):
            rows = work.loc[
                work["_espn"].eq(espn_id) & work["_gsis"].ne("")
            ]
            ids = unique_gsis(rows)
            if ids:
                if len(ids) != 1:
                    raise ValueError(
                        f"{target['name']}: {source_name} exact ESPN "
                        f"maps to multiple GSIS IDs {ids}"
                    )
                evidence.append(
                    {
                        "source": f"{source_name}_exact_espn",
                        "gsis_id": ids[0],
                        "rows": int(len(rows)),
                    }
                )

        # Official players table: exact full name + position.
        rows = p.loc[
            p["_name"].eq(target_name)
            & p["_position"].eq(target_position)
            & p["_gsis"].ne("")
        ]
        ids = unique_gsis(rows)
        if ids:
            if len(ids) != 1:
                raise ValueError(
                    f"{target['name']}: players exact name+position "
                    f"maps to multiple GSIS IDs {ids}"
                )
            evidence.append(
                {
                    "source": "players_exact_name_position",
                    "gsis_id": ids[0],
                    "rows": int(len(rows)),
                }
            )

        # Official weekly roster: exact full name + team + position.
        rows = r.loc[
            r["_name"].eq(target_name)
            & r["_team"].eq(target_team)
            & r["_position"].eq(target_position)
            & r["_gsis"].ne("")
        ]
        ids = unique_gsis(rows)
        if ids:
            if len(ids) != 1:
                raise ValueError(
                    f"{target['name']}: roster exact identity "
                    f"maps to multiple GSIS IDs {ids}"
                )
            evidence.append(
                {
                    "source": "roster_exact_name_team_position",
                    "gsis_id": ids[0],
                    "rows": int(len(rows)),
                }
            )

        # Name variants such as Greg/Gregory and suffix differences are
        # resolved only with surname + exact team + exact position and a
        # unique nonblank authoritative GSIS.
        rows = r.loc[
            roster_surname_mask(r["_name"], target["surname"])
            & r["_team"].eq(target_team)
            & r["_position"].eq(target_position)
            & r["_gsis"].ne("")
        ]
        ids = unique_gsis(rows)
        if ids:
            if len(ids) != 1:
                raise ValueError(
                    f"{target['name']}: roster surname+team+position "
                    f"maps to multiple GSIS IDs {ids}"
                )
            evidence.append(
                {
                    "source": "roster_surname_team_position",
                    "gsis_id": ids[0],
                    "rows": int(len(rows)),
                }
            )

        ids = sorted({item["gsis_id"] for item in evidence})
        if len(ids) != 1:
            p_sample = p.loc[
                p["_name"].str.contains(target["surname"], na=False)
                & p["_position"].eq(target_position)
            ][
                [
                    c
                    for c in [
                        pcols["name"],
                        pcols["position"],
                        pcols["team"],
                        pcols["gsis"],
                        pcols["espn"],
                    ]
                    if c
                ]
            ].drop_duplicates().head(20).to_dict("records")

            r_sample = r.loc[
                roster_surname_mask(r["_name"], target["surname"])
                & r["_position"].eq(target_position)
                & r["_team"].eq(target_team)
            ][
                [
                    c
                    for c in [
                        rcols["name"],
                        rcols["position"],
                        rcols["team"],
                        rcols["gsis"],
                        rcols["espn"],
                        "week" if "week" in r.columns else None,
                    ]
                    if c
                ]
            ].drop_duplicates().head(20).to_dict("records")

            raise ValueError(
                f"{target['name']}: expected one authoritative GSIS; "
                f"got {ids}. players_sample={p_sample} roster_sample={r_sample}"
            )

        resolved[espn_id] = {
            **target,
            "gsis_id": ids[0],
            "evidence": evidence,
            "resolution_rule": "official_nflverse_unique_identity_evidence",
        }

    return resolved


def enrich_players(
    players: pd.DataFrame,
    mappings: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    common.require_columns(players, ["gsis_id", "espn_id"], "players.parquet")
    result = players.copy()
    gsis = result["gsis_id"].map(norm_id)

    updated = 0
    absent = 0

    for espn_id, mapping in mappings.items():
        indexes = result.index[gsis.eq(mapping["gsis_id"])].tolist()
        if not indexes:
            absent += 1
            continue
        if len(indexes) != 1:
            raise ValueError(
                f"{mapping['name']}: players source has {len(indexes)} rows "
                f"for GSIS {mapping['gsis_id']}"
            )

        idx = indexes[0]
        existing = norm_id(result.at[idx, "espn_id"])
        if existing and existing != espn_id:
            raise ValueError(
                f"{mapping['name']}: players GSIS {mapping['gsis_id']} "
                f"has conflicting ESPN alias {existing}"
            )
        if existing != espn_id:
            result.at[idx, "espn_id"] = espn_id
            updated += 1

    nonblank = result["gsis_id"].map(norm_id)
    if nonblank[nonblank.ne("")].duplicated().any():
        raise ValueError("Repaired players.parquet has duplicate nonblank GSIS IDs")

    return result, {"updated_aliases": updated, "mapping_gsis_absent": absent}


def enrich_roster(
    roster: pd.DataFrame,
    mappings: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    common.require_columns(
        roster,
        ["gsis_id", "espn_id"],
        "roster_weekly_2026.parquet",
    )
    result = roster.copy()
    gsis = result["gsis_id"].map(norm_id)

    updated = 0
    matched = 0

    for espn_id, mapping in mappings.items():
        indexes = result.index[gsis.eq(mapping["gsis_id"])].tolist()
        if not indexes:
            continue

        matched += 1
        for idx in indexes:
            existing = norm_id(result.at[idx, "espn_id"])
            if existing and existing != espn_id:
                raise ValueError(
                    f"{mapping['name']}: roster GSIS {mapping['gsis_id']} "
                    f"has conflicting ESPN alias {existing}"
                )
            if existing != espn_id:
                result.at[idx, "espn_id"] = espn_id
                updated += 1

    return result, {"gsis_mappings_present": matched, "updated_alias_rows": updated}


def verify_upstream_aliases(
    players: pd.DataFrame,
    roster: pd.DataFrame,
    mappings: dict[str, dict[str, Any]],
) -> None:
    for espn_id, mapping in mappings.items():
        gsis_id = mapping["gsis_id"]
        found = False

        for frame in (players, roster):
            if "gsis_id" not in frame.columns or "espn_id" not in frame.columns:
                continue
            mask = (
                frame["gsis_id"].map(norm_id).eq(gsis_id)
                & frame["espn_id"].map(norm_id).eq(espn_id)
            )
            if mask.any():
                found = True
                break

        if not found:
            raise ValueError(
                f"{mapping['name']}: repaired upstream sources do not contain "
                f"ESPN {espn_id} -> GSIS {gsis_id}"
            )


def atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
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
        raise ValueError("Configured historical_players path differs from authorized path")
    if (
        Path(
            str(config["paths"]["historical_rosters_pattern"]).format(season=2026)
        )
        != ROSTER_REL
    ):
        raise ValueError("Configured 2026 roster path differs from authorized path")

    players_asset = current_release_asset("players", "players.parquet")
    roster_asset = current_release_asset(
        "weekly_rosters",
        "roster_weekly_2026.parquet",
    )

    players_temp = None
    roster_temp = None

    try:
        players_temp = download_verified(players_asset, players_dest.parent)
        roster_temp = download_verified(roster_asset, roster_dest.parent)

        official_players = pd.read_parquet(players_temp)
        official_roster = pd.read_parquet(roster_temp)

        mappings = derive_mappings(official_players, official_roster)

        repaired_players, players_stats = enrich_players(
            official_players,
            mappings,
        )
        repaired_roster, roster_stats = enrich_roster(
            official_roster,
            mappings,
        )
        verify_upstream_aliases(repaired_players, repaired_roster, mappings)

        atomic_write_parquet(repaired_players, players_dest)
        atomic_write_parquet(repaired_roster, roster_dest)

        completed = subprocess.run(
            [sys.executable, str(BUILDER)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )

        payload = {
            "script": "apply_gsis_identity_source_refresh.py",
            "status": "passed" if completed.returncode == 0 else "failed",
            "refreshed_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "manual_identity_override": False,
            "canonical_source_repair": True,
            "target_count": len(TARGETS),
            "mappings": mappings,
            "official_assets": {
                "players": players_asset,
                "weekly_roster_2026": roster_asset,
            },
            "repository_sources": {
                "players": {
                    "path": str(PLAYERS_REL).replace("\\", "/"),
                    "sha256": sha256_file(players_dest),
                    **players_stats,
                },
                "weekly_roster_2026": {
                    "path": str(ROSTER_REL).replace("\\", "/"),
                    "sha256": sha256_file(roster_dest),
                    **roster_stats,
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
            evidence = ",".join(item["source"] for item in mapping["evidence"])
            print(
                f"espn_id={espn_id} name={mapping['name']} "
                f"authoritative_gsis={mapping['gsis_id']} evidence={evidence}"
            )

        print(f"official_players_sha256={players_asset['sha256']}")
        print(f"official_roster_sha256={roster_asset['sha256']}")
        print(f"repaired_players_sha256={sha256_file(players_dest)}")
        print(f"repaired_roster_sha256={sha256_file(roster_dest)}")
        print("manual_identity_override=false")

        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise RuntimeError(
                f"build_player_identity.py failed with return code "
                f"{completed.returncode}"
            )

        print("AUTHORITATIVE GSIS SOURCE REPAIR: PASS")
        return 0
    finally:
        for temp in (players_temp, roster_temp):
            if temp is not None and temp.exists():
                temp.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
