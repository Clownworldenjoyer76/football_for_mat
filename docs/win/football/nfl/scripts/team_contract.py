#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def is_nfl_mapping_row(
    row: dict[str, Any],
    *,
    clean: Callable[[Any], str],
    allow_blank_scope: bool,
) -> bool:
    sport = clean(row.get("sport")).casefold()
    league = clean(row.get("league")).casefold()

    valid_sports = {"football"}
    valid_leagues = {"nfl"}
    if allow_blank_scope:
        valid_sports.add("")
        valid_leagues.add("")

    return sport in valid_sports and league in valid_leagues


def build_team_abbr_map(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    clean: Callable[[Any], str],
    fail: Callable[[str], Any],
) -> dict[str, str]:
    by_id: dict[str, str] = {}
    by_abbr: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        if not is_nfl_mapping_row(
            row,
            clean=clean,
            allow_blank_scope=False,
        ):
            continue

        team_id = clean(row.get("team_id"))
        team_abbr = clean(row.get("team_abbr")).upper()

        if not team_id or not team_abbr:
            fail(
                f"{path} line {line_number} has "
                "blank team_id/team_abbr"
            )

        previous_abbr = by_id.get(team_id)
        if previous_abbr and previous_abbr != team_abbr:
            fail(
                f"{path} has conflicting abbreviation "
                f"for team_id={team_id}: "
                f"{previous_abbr!r} vs {team_abbr!r}"
            )

        previous_id = by_abbr.get(team_abbr)
        if previous_id and previous_id != team_id:
            fail(
                f"{path} has conflicting team ID "
                f"for team_abbr={team_abbr}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        by_id[team_id] = team_abbr
        by_abbr[team_abbr] = team_id

    if len(by_id) != 32 or len(by_abbr) != 32:
        fail(
            "NFL team map must resolve exactly 32 teams; "
            f"ids={len(by_id)} abbrs={len(by_abbr)}"
        )

    return by_id


def build_team_name_maps(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    clean: Callable[[Any], str],
    fail: Callable[[str], Any],
) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    by_id: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        if not is_nfl_mapping_row(
            row,
            clean=clean,
            allow_blank_scope=False,
        ):
            continue

        team_id = clean(row.get("team_id"))
        canonical_team = clean(row.get("canonical_team"))

        if not team_id or not canonical_team:
            fail(
                f"{path} line {line_number} has "
                "blank team_id/canonical_team"
            )

        previous_id = by_name.get(canonical_team)
        if previous_id and previous_id != team_id:
            fail(
                f"{path} has conflicting team IDs for "
                f"canonical_team={canonical_team!r}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        previous_name = by_id.get(team_id)
        if previous_name and previous_name != canonical_team:
            fail(
                f"{path} has conflicting canonical team "
                f"for team_id={team_id}: "
                f"{previous_name!r} vs {canonical_team!r}"
            )

        by_name[canonical_team] = team_id
        by_id[team_id] = canonical_team

    if len(by_name) != 32 or len(by_id) != 32:
        fail(
            "NFL team map must resolve exactly 32 canonical teams; "
            f"names={len(by_name)} ids={len(by_id)}"
        )

    return by_name, by_id
