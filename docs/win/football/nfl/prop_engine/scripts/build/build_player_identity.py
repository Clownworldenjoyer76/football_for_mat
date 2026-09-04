#!/usr/bin/env python3
"""
Build the canonical Prop Engine player identity crosswalk.

READS:
    docs/win/football/nfl/data/historic_data/players/players.parquet
    docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_{season}.parquet
    docs/win/football/nfl/data/master/roster_master.csv
    docs/win/football/nfl/data/master/depth_charts/{TEAM}/{TEAM}_depth.csv

WRITES:
    docs/win/football/nfl/prop_engine/data/identity/player_crosswalk.parquet
    docs/win/football/nfl/prop_engine/logs/build_player_identity.json

CANONICAL ID:
    GSIS ID.

RULES:
    GSIS is authoritative. ESPN and PFR are aliases. Name matching is
    allowed only when the normalized name resolves uniquely.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


OUTPUT_COLUMNS = [
    "player_id",
    "gsis_id",
    "espn_id",
    "pfr_id",
    "display_name",
    "normalized_name",
    "position",
    "position_group",
    "current_team",
    "current_espn_id",
    "current_status",
    "first_season",
    "last_season",
    "identity_source",
    "resolution_method",
    "resolution_confidence",
    "resolution_status",
]


PLAYER_ALIASES = {
    "gsis_id": ["gsis_id"],
    "espn_id": ["espn_id"],
    "pfr_id": ["pfr_id", "pfr_player_id"],
    "name": ["display_name", "full_name", "player_name"],
    "position": ["position", "position_group"],
    "position_group": ["position_group"],
}


HISTORICAL_ROSTER_ALIASES = {
    "gsis_id": ["gsis_id"],
    "espn_id": ["espn_id"],
    "pfr_id": ["pfr_id", "pfr_player_id"],
    "name": ["full_name", "display_name", "player_name"],
    "position": [
        "position",
        "depth_chart_position",
        "ngs_position",
    ],
    "position_group": ["position_group"],
    "season": ["season"],
}


CURRENT_ROSTER_ALIASES = {
    "espn_id": ["id", "espn_id", "player_id"],
    "name": [
        "displayName",
        "fullName",
        "display_name",
        "full_name",
        "player_name",
    ],
    "position": [
        "position.abbreviation",
        "position_abb",
        "position",
        "position_abbreviation",
    ],
    "position_group": [
        "position.parent.abbreviation",
        "position_group",
    ],
    "status": [
        "status.name",
        "status.type",
        "status.abbreviation",
        "status",
    ],
    "team_id": ["team_id"],
}


DEPTH_ALIASES = {
    "espn_id": ["player_id", "id", "espn_id"],
    "name": [
        "name",
        "displayName",
        "display_name",
        "full_name",
        "player_name",
    ],
    "team": ["team", "team_abbr"],
    "position": [
        "position_abb",
        "position.abbreviation",
        "position",
        "position_abbreviation",
    ],
    "starter": [
        "starter_flag",
        "starter",
        "is_starter",
    ],
    "team_id": ["team_id"],
}


CRITICAL_SKILL_POSITIONS = {
    "QB",
    "RB",
    "WR",
    "TE",
    "K",
    "PK",
}


DEFENSIVE_POSITIONS = {
    "DB",
    "CB",
    "LCB",
    "RCB",
    "NB",
    "S",
    "FS",
    "SS",
    "LB",
    "ILB",
    "OLB",
    "MLB",
    "WLB",
    "SLB",
    "DL",
    "DE",
    "LDE",
    "RDE",
    "DT",
    "LDT",
    "RDT",
    "NT",
    "EDGE",
}


SEASON_FROM_FILENAME = re.compile(
    r"roster_weekly_(\d{4})\.parquet$"
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

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def normalize_position(value: Any) -> str:
    return clean(value).upper()


def choose_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    label: str,
    required: bool = False,
) -> str | None:
    choices = list(aliases)

    for column in choices:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"{label}: none of the required aliases exist: "
            f"{choices}"
        )

    return None


def selected_columns(
    df: pd.DataFrame,
    aliases: dict[str, list[str]],
    *,
    label: str,
    required: Iterable[str] = (),
) -> dict[str, str | None]:
    required_set = set(required)

    return {
        key: choose_column(
            df,
            values,
            label=label,
            required=key in required_set,
        )
        for key, values in aliases.items()
    }


def new_canonical(
    gsis_id: str,
) -> dict[str, Any]:
    return {
        "gsis_id": gsis_id,
        "espn_ids": set(),
        "pfr_ids": set(),
        "primary_espn_id": "",
        "primary_pfr_id": "",
        "names": set(),
        "display_name": "",
        "position": "",
        "position_group": "",
        "seasons": set(),
        "sources": set(),
        "current": None,
        "current_resolution_method": "",
        "current_resolution_confidence": None,
    }


def add_canonical(
    canonical: dict[str, dict[str, Any]],
    *,
    gsis_id: Any,
    espn_id: Any = "",
    pfr_id: Any = "",
    name: Any = "",
    position: Any = "",
    position_group: Any = "",
    season: int | None = None,
    source: str,
    prefer_metadata: bool = False,
) -> None:
    gsis = common.normalize_player_id(
        gsis_id
    )

    if not gsis:
        return

    record = canonical.setdefault(
        gsis,
        new_canonical(gsis),
    )

    espn = common.normalize_player_id(
        espn_id
    )

    pfr = common.normalize_player_id(
        pfr_id
    )

    display_name = clean(name)
    pos = normalize_position(position)
    pos_group = normalize_position(
        position_group
    )

    if espn:
        record["espn_ids"].add(espn)

        if (
            prefer_metadata
            and not record["primary_espn_id"]
        ):
            record["primary_espn_id"] = espn

    if pfr:
        record["pfr_ids"].add(pfr)

        if (
            prefer_metadata
            and not record["primary_pfr_id"]
        ):
            record["primary_pfr_id"] = pfr

    if display_name:
        record["names"].add(display_name)

        if (
            prefer_metadata
            or not record["display_name"]
        ):
            record["display_name"] = display_name

    if pos and (
        prefer_metadata
        or not record["position"]
    ):
        record["position"] = pos

    if pos_group and (
        prefer_metadata
        or not record["position_group"]
    ):
        record["position_group"] = pos_group

    if season is not None:
        record["seasons"].add(
            int(season)
        )

    record["sources"].add(source)


def unique_alias_index(
    canonical: dict[str, dict[str, Any]],
    field: str,
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
]:
    candidates: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for gsis_id, record in canonical.items():
        for alias in record[field]:
            if alias:
                candidates[alias].add(
                    gsis_id
                )

    unique = {
        alias: next(iter(gsis_ids))
        for alias, gsis_ids
        in candidates.items()
        if len(gsis_ids) == 1
    }

    conflicts = {
        alias: sorted(gsis_ids)
        for alias, gsis_ids
        in candidates.items()
        if len(gsis_ids) > 1
    }

    return unique, conflicts


def authoritative_espn_index(
    canonical: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
]:
    """
    Build the authoritative ESPN alias index from players.parquet aliases.

    primary_espn_id is populated only from the authoritative historical
    players table. Historical weekly roster ESPN IDs remain secondary
    evidence because those IDs can change, be corrected, or be reused.
    """
    candidates: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for gsis_id, record in canonical.items():
        alias = record["primary_espn_id"]

        if alias:
            candidates[alias].add(
                gsis_id
            )

    unique = {
        alias: next(iter(gsis_ids))
        for alias, gsis_ids
        in candidates.items()
        if len(gsis_ids) == 1
    }

    conflicts = {
        alias: sorted(gsis_ids)
        for alias, gsis_ids
        in candidates.items()
        if len(gsis_ids) > 1
    }

    return unique, conflicts


def unique_name_index(
    canonical: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
]:
    candidates: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for gsis_id, record in canonical.items():
        for raw_name in record["names"]:
            normalized = common.normalize_name(
                raw_name
            )

            if normalized:
                candidates[
                    normalized
                ].add(gsis_id)

    unique = {
        normalized: next(iter(gsis_ids))
        for normalized, gsis_ids
        in candidates.items()
        if len(gsis_ids) == 1
    }

    ambiguous = {
        normalized: sorted(gsis_ids)
        for normalized, gsis_ids
        in candidates.items()
        if len(gsis_ids) > 1
    }

    return unique, ambiguous


def parse_optional_season(
    value: Any,
) -> int | None:
    text = clean(value)

    if not text:
        return None

    try:
        number = int(float(text))
    except ValueError:
        return None

    if 1900 <= number <= 2200:
        return number

    return None


def truthy(value: Any) -> bool:
    return clean(value).casefold() in {
        "1",
        "true",
        "yes",
        "y",
        "starter",
    }


def write_json_atomic(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )

    temp_path = Path(handle.name)

    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def merge_current(
    current: dict[str, dict[str, Any]],
    espn_id: str,
    *,
    source: str,
    name: str = "",
    position: str = "",
    position_group: str = "",
    team: str = "",
    status: str = "",
    depth_starter: bool = False,
    depth_position: str = "",
) -> None:
    if not espn_id:
        return

    record = current.setdefault(
        espn_id,
        {
            "current_espn_id": espn_id,
            "display_name": "",
            "position": "",
            "position_group": "",
            "current_team": "",
            "current_status": "",
            "sources": set(),
            "in_roster": False,
            "depth_starter": False,
            "depth_positions": set(),
        },
    )

    record["sources"].add(
        source
    )

    if source == "current_roster":
        record["in_roster"] = True

    if name and (
        source == "current_roster"
        or not record["display_name"]
    ):
        record["display_name"] = name

    if position and (
        source == "current_roster"
        or not record["position"]
    ):
        record["position"] = position

    if position_group and (
        source == "current_roster"
        or not record["position_group"]
    ):
        record["position_group"] = (
            position_group
        )

    if team and (
        source == "current_roster"
        or not record["current_team"]
    ):
        record["current_team"] = team

    if status and (
        source == "current_roster"
        or not record["current_status"]
    ):
        record["current_status"] = status

    if depth_starter:
        record["depth_starter"] = True

    if depth_position:
        record["depth_positions"].add(
            depth_position
        )


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()

    paths = config["paths"]

    players_path = (
        repo
        / paths["historical_players"]
    )

    roster_pattern = paths[
        "historical_rosters_pattern"
    ]

    current_roster_path = (
        repo
        / paths["current_roster"]
    )

    depth_root = (
        repo
        / paths["current_depth_root"]
    )

    output_path = (
        repo
        / paths["identity_crosswalk"]
    )

    log_path = (
        repo
        / paths["log_root"]
        / "build_player_identity.json"
    )

    if "{season}" not in roster_pattern:
        raise ValueError(
            "paths.historical_rosters_pattern "
            "must contain {season}"
        )

    roster_files = sorted(
        repo.glob(
            roster_pattern.replace(
                "{season}",
                "*",
            )
        )
    )

    if not roster_files:
        raise FileNotFoundError(
            "No historical weekly roster "
            f"files matched: {roster_pattern}"
        )

    depth_files = sorted(
        depth_root.glob(
            "*/*_depth.csv"
        )
    )

    if not depth_files:
        raise FileNotFoundError(
            "No current depth files found "
            f"under: {depth_root}"
        )

    canonical: dict[
        str,
        dict[str, Any],
    ] = {}

    chosen_columns: dict[
        str,
        Any,
    ] = {}

    players = (
        common.read_parquet_required(
            players_path
        )
    )

    player_cols = selected_columns(
        players,
        PLAYER_ALIASES,
        label=str(players_path),
        required={"gsis_id"},
    )

    chosen_columns[
        "historical_players"
    ] = player_cols

    for row in players.to_dict(
        orient="records"
    ):
        add_canonical(
            canonical,
            gsis_id=row[
                player_cols["gsis_id"]
            ],
            espn_id=(
                row[player_cols["espn_id"]]
                if player_cols["espn_id"]
                else ""
            ),
            pfr_id=(
                row[player_cols["pfr_id"]]
                if player_cols["pfr_id"]
                else ""
            ),
            name=(
                row[player_cols["name"]]
                if player_cols["name"]
                else ""
            ),
            position=(
                row[player_cols["position"]]
                if player_cols["position"]
                else ""
            ),
            position_group=(
                row[
                    player_cols[
                        "position_group"
                    ]
                ]
                if player_cols[
                    "position_group"
                ]
                else ""
            ),
            source="historical_players",
            prefer_metadata=True,
        )

    historical_roster_rows = 0

    roster_chosen_by_schema: dict[
        str,
        dict[str, str | None],
    ] = {}

    for roster_path in roster_files:
        roster = (
            common.read_parquet_required(
                roster_path
            )
        )

        historical_roster_rows += len(
            roster
        )

        roster_cols = selected_columns(
            roster,
            HISTORICAL_ROSTER_ALIASES,
            label=str(roster_path),
        )

        schema_key = "|".join(
            str(column)
            for column in roster.columns
        )

        roster_chosen_by_schema.setdefault(
            schema_key,
            roster_cols,
        )

        filename_match = (
            SEASON_FROM_FILENAME.search(
                roster_path.name
            )
        )

        filename_season = (
            int(filename_match.group(1))
            if filename_match
            else None
        )

        if not roster_cols["gsis_id"]:
            continue

        for row in roster.to_dict(
            orient="records"
        ):
            season = (
                parse_optional_season(
                    row[
                        roster_cols[
                            "season"
                        ]
                    ]
                )
                if roster_cols["season"]
                else filename_season
            )

            add_canonical(
                canonical,
                gsis_id=row[
                    roster_cols["gsis_id"]
                ],
                espn_id=(
                    row[
                        roster_cols[
                            "espn_id"
                        ]
                    ]
                    if roster_cols[
                        "espn_id"
                    ]
                    else ""
                ),
                pfr_id=(
                    row[
                        roster_cols[
                            "pfr_id"
                        ]
                    ]
                    if roster_cols[
                        "pfr_id"
                    ]
                    else ""
                ),
                name=(
                    row[
                        roster_cols["name"]
                    ]
                    if roster_cols["name"]
                    else ""
                ),
                position=(
                    row[
                        roster_cols[
                            "position"
                        ]
                    ]
                    if roster_cols[
                        "position"
                    ]
                    else ""
                ),
                position_group=(
                    row[
                        roster_cols[
                            "position_group"
                        ]
                    ]
                    if roster_cols[
                        "position_group"
                    ]
                    else ""
                ),
                season=season,
                source=(
                    "historical_rosters"
                ),
                prefer_metadata=False,
            )

    chosen_columns[
        "historical_rosters_by_schema"
    ] = list(
        roster_chosen_by_schema.values()
    )

    espn_index, espn_conflicts = (
        unique_alias_index(
            canonical,
            "espn_ids",
        )
    )

    (
        authoritative_espn,
        authoritative_espn_conflicts,
    ) = authoritative_espn_index(
        canonical
    )

    _, pfr_conflicts = (
        unique_alias_index(
            canonical,
            "pfr_ids",
        )
    )

    name_index, ambiguous_names = (
        unique_name_index(
            canonical
        )
    )

    current: dict[
        str,
        dict[str, Any],
    ] = {}

    team_id_to_team: dict[
        str,
        str,
    ] = {}

    depth_chosen_by_schema: dict[
        str,
        dict[str, str | None],
    ] = {}

    depth_rows = 0

    for depth_path in depth_files:
        depth = (
            common.read_csv_required(
                depth_path
            )
        )

        depth_rows += len(depth)

        depth_cols = selected_columns(
            depth,
            DEPTH_ALIASES,
            label=str(depth_path),
            required={
                "espn_id",
                "team",
            },
        )

        schema_key = "|".join(
            str(column)
            for column in depth.columns
        )

        depth_chosen_by_schema.setdefault(
            schema_key,
            depth_cols,
        )

        for row in depth.to_dict(
            orient="records"
        ):
            espn_id = (
                common.normalize_player_id(
                    row[
                        depth_cols[
                            "espn_id"
                        ]
                    ]
                )
            )

            team = common.normalize_team(
                row[
                    depth_cols["team"]
                ]
            )

            position = (
                normalize_position(
                    row[
                        depth_cols[
                            "position"
                        ]
                    ]
                )
                if depth_cols["position"]
                else ""
            )

            starter = (
                truthy(
                    row[
                        depth_cols[
                            "starter"
                        ]
                    ]
                )
                if depth_cols["starter"]
                else False
            )

            team_id = (
                common.normalize_player_id(
                    row[
                        depth_cols[
                            "team_id"
                        ]
                    ]
                )
                if depth_cols["team_id"]
                else ""
            )

            if team_id and team:
                previous = (
                    team_id_to_team.get(
                        team_id
                    )
                )

                if (
                    previous
                    and previous != team
                ):
                    raise ValueError(
                        f"Depth team_id "
                        f"{team_id} maps to "
                        f"both {previous} "
                        f"and {team}"
                    )

                team_id_to_team[
                    team_id
                ] = team

            merge_current(
                current,
                espn_id,
                source="current_depth",
                name=(
                    clean(
                        row[
                            depth_cols[
                                "name"
                            ]
                        ]
                    )
                    if depth_cols["name"]
                    else ""
                ),
                position=position,
                team=team,
                depth_starter=starter,
                depth_position=position,
            )

    chosen_columns[
        "current_depth_by_schema"
    ] = list(
        depth_chosen_by_schema.values()
    )

    current_roster = (
        common.read_csv_required(
            current_roster_path
        )
    )

    current_roster_cols = (
        selected_columns(
            current_roster,
            CURRENT_ROSTER_ALIASES,
            label=str(
                current_roster_path
            ),
            required={"espn_id"},
        )
    )

    chosen_columns[
        "current_roster"
    ] = current_roster_cols

    for row in current_roster.to_dict(
        orient="records"
    ):
        espn_id = (
            common.normalize_player_id(
                row[
                    current_roster_cols[
                        "espn_id"
                    ]
                ]
            )
        )

        team_id = (
            common.normalize_player_id(
                row[
                    current_roster_cols[
                        "team_id"
                    ]
                ]
            )
            if current_roster_cols[
                "team_id"
            ]
            else ""
        )

        team = team_id_to_team.get(
            team_id,
            "",
        )

        merge_current(
            current,
            espn_id,
            source="current_roster",
            name=(
                clean(
                    row[
                        current_roster_cols[
                            "name"
                        ]
                    ]
                )
                if current_roster_cols[
                    "name"
                ]
                else ""
            ),
            position=(
                normalize_position(
                    row[
                        current_roster_cols[
                            "position"
                        ]
                    ]
                )
                if current_roster_cols[
                    "position"
                ]
                else ""
            ),
            position_group=(
                normalize_position(
                    row[
                        current_roster_cols[
                            "position_group"
                        ]
                    ]
                )
                if current_roster_cols[
                    "position_group"
                ]
                else ""
            ),
            team=team,
            status=(
                clean(
                    row[
                        current_roster_cols[
                            "status"
                        ]
                    ]
                )
                if current_roster_cols[
                    "status"
                ]
                else ""
            ),
        )

    proposed: dict[
        str,
        dict[str, Any],
    ] = {}

    for espn_id, record in sorted(
        current.items()
    ):
        gsis_id = authoritative_espn.get(
            espn_id,
            "",
        )

        method = ""
        confidence = 0.0

        if gsis_id:
            method = (
                "historical_players_espn_alias"
            )
            confidence = 1.0

        else:
            gsis_id = espn_index.get(
                espn_id,
                "",
            )

            if gsis_id:
                method = (
                    "unique_historical_roster_"
                    "espn_alias"
                )
                confidence = 0.95

        if not gsis_id:
            normalized_name = (
                common.normalize_name(
                    record[
                        "display_name"
                    ]
                )
            )

            if (
                normalized_name
                and normalized_name
                in name_index
            ):
                gsis_id = name_index[
                    normalized_name
                ]

                method = (
                    "unique_normalized_name"
                )

                confidence = 0.8

        proposed[espn_id] = {
            "gsis_id": gsis_id,
            "method": method,
            "confidence": confidence,
        }

    by_gsis: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for (
        espn_id,
        resolution,
    ) in proposed.items():
        if resolution["gsis_id"]:
            by_gsis[
                resolution["gsis_id"]
            ].append(espn_id)

    unresolved_reason: dict[
        str,
        str,
    ] = {}

    for (
        gsis_id,
        espn_ids,
    ) in by_gsis.items():
        if len(espn_ids) <= 1:
            continue

        normalized_names = {
            common.normalize_name(
                current[
                    espn_id
                ]["display_name"]
            )
            for espn_id in espn_ids
            if common.normalize_name(
                current[
                    espn_id
                ]["display_name"]
            )
        }

        current_teams = {
            current[
                espn_id
            ]["current_team"]
            for espn_id in espn_ids
            if current[
                espn_id
            ]["current_team"]
        }

        # Multiple current ESPN aliases may legitimately identify the
        # same player when the name is identical and the team does not
        # conflict. The canonical record remains one row per GSIS ID.
        if (
            len(normalized_names) == 1
            and len(current_teams) <= 1
        ):
            continue

        direct = [
            espn_id
            for espn_id in espn_ids
            if proposed[
                espn_id
            ]["method"]
            in {
                "historical_players_espn_alias",
                "unique_historical_roster_espn_alias",
            }
        ]

        if len(direct) == 1:
            keep = direct[0]

            for espn_id in espn_ids:
                if espn_id == keep:
                    continue

                proposed[
                    espn_id
                ]["gsis_id"] = ""

                proposed[
                    espn_id
                ]["method"] = ""

                proposed[
                    espn_id
                ]["confidence"] = 0.0

                unresolved_reason[
                    espn_id
                ] = (
                    "current_identity_"
                    "conflicts_with_direct_"
                    "espn_alias"
                )

        else:
            for espn_id in espn_ids:
                proposed[
                    espn_id
                ]["gsis_id"] = ""

                proposed[
                    espn_id
                ]["method"] = ""

                proposed[
                    espn_id
                ]["confidence"] = 0.0

                unresolved_reason[
                    espn_id
                ] = (
                    "ambiguous_current_identity"
                )

    for (
        espn_id,
        resolution,
    ) in proposed.items():
        gsis_id = resolution[
            "gsis_id"
        ]

        if not gsis_id:
            if (
                espn_id
                not in unresolved_reason
            ):
                normalized_name = (
                    common.normalize_name(
                        current[
                            espn_id
                        ]["display_name"]
                    )
                )

                if (
                    normalized_name
                    in ambiguous_names
                ):
                    unresolved_reason[
                        espn_id
                    ] = (
                        "ambiguous_normalized_name"
                    )
                else:
                    unresolved_reason[
                        espn_id
                    ] = (
                        "no_unique_gsis_match"
                    )

            continue

        record = canonical[
            gsis_id
        ]

        current_record = current[
            espn_id
        ]

        record["current"] = (
            current_record
        )

        record[
            "current_resolution_method"
        ] = resolution["method"]

        record[
            "current_resolution_confidence"
        ] = resolution[
            "confidence"
        ]

        record["sources"].update(
            current_record["sources"]
        )

        if current_record["position"]:
            record["position"] = (
                current_record[
                    "position"
                ]
            )

        if current_record[
            "position_group"
        ]:
            record[
                "position_group"
            ] = current_record[
                "position_group"
            ]

    rows: list[
        dict[str, Any]
    ] = []

    multiple_espn_aliases: dict[
        str,
        list[str],
    ] = {}

    multiple_pfr_aliases: dict[
        str,
        list[str],
    ] = {}

    for gsis_id in sorted(
        canonical
    ):
        record = canonical[
            gsis_id
        ]

        espn_ids = sorted(
            record["espn_ids"]
        )

        pfr_ids = sorted(
            record["pfr_ids"]
        )

        if len(espn_ids) > 1:
            multiple_espn_aliases[
                gsis_id
            ] = espn_ids

        if len(pfr_ids) > 1:
            multiple_pfr_aliases[
                gsis_id
            ] = pfr_ids

        espn_id = (
            record["primary_espn_id"]
            or (
                espn_ids[0]
                if len(espn_ids) == 1
                else ""
            )
        )

        pfr_id = (
            record["primary_pfr_id"]
            or (
                pfr_ids[0]
                if len(pfr_ids) == 1
                else ""
            )
        )

        current_record = record[
            "current"
        ]

        current_method = record[
            "current_resolution_method"
        ]

        rows.append(
            {
                "player_id": gsis_id,
                "gsis_id": gsis_id,
                "espn_id": espn_id,
                "pfr_id": pfr_id,
                "display_name": (
                    record[
                        "display_name"
                    ]
                ),
                "normalized_name": (
                    common.normalize_name(
                        record[
                            "display_name"
                        ]
                    )
                ),
                "position": (
                    record["position"]
                ),
                "position_group": (
                    record[
                        "position_group"
                    ]
                ),
                "current_team": (
                    current_record[
                        "current_team"
                    ]
                    if current_record
                    else ""
                ),
                "current_espn_id": (
                    current_record[
                        "current_espn_id"
                    ]
                    if current_record
                    else ""
                ),
                "current_status": (
                    current_record[
                        "current_status"
                    ]
                    if current_record
                    else ""
                ),
                "first_season": (
                    min(
                        record[
                            "seasons"
                        ]
                    )
                    if record["seasons"]
                    else pd.NA
                ),
                "last_season": (
                    max(
                        record[
                            "seasons"
                        ]
                    )
                    if record["seasons"]
                    else pd.NA
                ),
                "identity_source": (
                    ";".join(
                        sorted(
                            record[
                                "sources"
                            ]
                        )
                    )
                ),
                "resolution_method": (
                    current_method
                    if current_method
                    else (
                        "gsis_authoritative"
                    )
                ),
                "resolution_confidence": (
                    record[
                        "current_resolution_confidence"
                    ]
                    if current_record
                    else 1.0
                ),
                "resolution_status": (
                    "resolved"
                ),
            }
        )

    unresolved_current: list[
        dict[str, Any]
    ] = []

    for espn_id in sorted(
        current
    ):
        if proposed[
            espn_id
        ]["gsis_id"]:
            continue

        record = current[
            espn_id
        ]

        row = {
            "player_id": "",
            "gsis_id": "",
            "espn_id": "",
            "pfr_id": "",
            "display_name": (
                record[
                    "display_name"
                ]
            ),
            "normalized_name": (
                common.normalize_name(
                    record[
                        "display_name"
                    ]
                )
            ),
            "position": (
                record["position"]
            ),
            "position_group": (
                record[
                    "position_group"
                ]
            ),
            "current_team": (
                record[
                    "current_team"
                ]
            ),
            "current_espn_id": (
                espn_id
            ),
            "current_status": (
                record[
                    "current_status"
                ]
            ),
            "first_season": pd.NA,
            "last_season": pd.NA,
            "identity_source": (
                ";".join(
                    sorted(
                        record[
                            "sources"
                        ]
                    )
                )
            ),
            "resolution_method": (
                unresolved_reason[
                    espn_id
                ]
            ),
            "resolution_confidence": 0.0,
            "resolution_status": (
                "unresolved"
            ),
        }

        rows.append(row)

        unresolved_current.append(
            row
        )

    output = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    output[
        "first_season"
    ] = pd.array(
        output["first_season"],
        dtype="Int64",
    )

    output[
        "last_season"
    ] = pd.array(
        output["last_season"],
        dtype="Int64",
    )

    canonical_output = output.loc[
        output["gsis_id"]
        .astype(str)
        .str.len()
        .gt(0)
    ]

    common.ensure_unique(
        canonical_output,
        ["gsis_id"],
        (
            "canonical player identity "
            "crosswalk"
        ),
    )

    unresolved_output = (
        output.loc[
            output[
                "resolution_status"
            ].eq("unresolved")
        ]
    )

    if not unresolved_output.empty:
        common.ensure_unique(
            unresolved_output,
            ["current_espn_id"],
            (
                "unresolved current "
                "ESPN identities"
            ),
        )

    common.require_columns(
        output,
        OUTPUT_COLUMNS,
        (
            "player identity "
            "crosswalk"
        ),
    )

    common.write_parquet_atomic(
        output,
        output_path,
    )

    critical_unresolved: list[
        dict[str, Any]
    ] = []

    for row in unresolved_current:
        source = current[
            row[
                "current_espn_id"
            ]
        ]

        position = (
            normalize_position(
                row["position"]
            )
        )

        position_group = (
            normalize_position(
                row[
                    "position_group"
                ]
            )
        )

        depth_positions = {
            normalize_position(
                value
            )
            for value
            in source[
                "depth_positions"
            ]
            if value
        }

        critical_skill = (
            position
            in CRITICAL_SKILL_POSITIONS
        )

        defensive_starter = (
            source["depth_starter"]
            and (
                position_group == "DEF"
                or bool(
                    depth_positions
                    & DEFENSIVE_POSITIONS
                )
            )
        )

        if (
            critical_skill
            or defensive_starter
        ):
            critical_unresolved.append(
                {
                    "current_espn_id": (
                        row[
                            "current_espn_id"
                        ]
                    ),
                    "display_name": (
                        row[
                            "display_name"
                        ]
                    ),
                    "position": (
                        row["position"]
                    ),
                    "position_group": (
                        row[
                            "position_group"
                        ]
                    ),
                    "current_team": (
                        row[
                            "current_team"
                        ]
                    ),
                    (
                        "critical_skill_"
                        "position"
                    ): critical_skill,
                    (
                        "defensive_depth_"
                        "starter"
                    ): defensive_starter,
                    "resolution_method": (
                        row[
                            "resolution_method"
                        ]
                    ),
                }
            )

    payload = {
        "script": (
            "build_player_identity.py"
        ),
        "inputs": {
            "historical_players": str(
                players_path.relative_to(
                    repo
                )
            ),
            "historical_roster_files": [
                str(
                    path.relative_to(
                        repo
                    )
                )
                for path in roster_files
            ],
            "current_roster": str(
                current_roster_path.relative_to(
                    repo
                )
            ),
            "current_depth_files": [
                str(
                    path.relative_to(
                        repo
                    )
                )
                for path in depth_files
            ],
        },
        "outputs": {
            "crosswalk": str(
                output_path.relative_to(
                    repo
                )
            ),
            "log": str(
                log_path.relative_to(
                    repo
                )
            ),
        },
        "chosen_columns": (
            chosen_columns
        ),
        "counts": {
            "historical_player_rows": (
                len(players)
            ),
            "historical_roster_files": (
                len(roster_files)
            ),
            "historical_roster_rows": (
                historical_roster_rows
            ),
            "current_roster_rows": (
                len(current_roster)
            ),
            "current_depth_files": (
                len(depth_files)
            ),
            "current_depth_rows": (
                depth_rows
            ),
            "canonical_gsis_records": (
                len(canonical_output)
            ),
            "current_espn_records": (
                len(current)
            ),
            "resolved_current_records": (
                sum(
                    1
                    for value
                    in proposed.values()
                    if value[
                        "gsis_id"
                    ]
                )
            ),
            "unresolved_current_records": (
                len(
                    unresolved_current
                )
            ),
            (
                "critical_unresolved_"
                "records"
            ): len(
                critical_unresolved
            ),
        },
        "alias_conflicts": {
            (
                "espn_alias_to_"
                "multiple_gsis"
            ): espn_conflicts,
            (
                "authoritative_espn_"
                "alias_to_multiple_gsis"
            ): authoritative_espn_conflicts,
            (
                "pfr_alias_to_"
                "multiple_gsis"
            ): pfr_conflicts,
            (
                "normalized_name_to_"
                "multiple_gsis"
            ): ambiguous_names,
            (
                "multiple_espn_"
                "aliases_on_gsis"
            ): multiple_espn_aliases,
            (
                "multiple_pfr_"
                "aliases_on_gsis"
            ): multiple_pfr_aliases,
        },
        "critical_unresolved": (
            critical_unresolved
        ),
        "resolution_policy": {
            "canonical_id": (
                "gsis_id"
            ),
            "current_match_order": [
                (
                    "unique_espn_id_"
                    "alias"
                ),
                (
                    "unique_normalized_"
                    "name"
                ),
            ],
            "ambiguous_name_matches": (
                "unresolved"
            ),
            "name_match_confidence": (
                0.8
            ),
            (
                "direct_espn_match_"
                "confidence"
            ): 1.0,
        },
        "status": (
            "failed"
            if critical_unresolved
            else "passed"
        ),
    }

    write_json_atomic(
        payload,
        log_path,
    )

    common.log_run(
        "build_player_identity.py",
        {
            (
                "canonical_gsis_"
                "records"
            ): len(
                canonical_output
            ),
            (
                "unresolved_current_"
                "records"
            ): len(
                unresolved_current
            ),
            (
                "critical_unresolved_"
                "records"
            ): len(
                critical_unresolved
            ),
            "status": payload[
                "status"
            ],
        },
    )

    if critical_unresolved:
        sample = (
            critical_unresolved[
                :20
            ]
        )

        raise RuntimeError(
            "Identity validation failed: "
            "unresolved current "
            "QB/RB/WR/TE/K or defensive "
            "depth starter(s) remain. "
            f"Count="
            f"{len(critical_unresolved)}. "
            f"Sample={sample}. "
            f"See {log_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
