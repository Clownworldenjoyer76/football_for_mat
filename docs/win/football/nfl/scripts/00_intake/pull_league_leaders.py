#!/usr/bin/env python3
"""Pull ESPN NFL regular-season league leaders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

SEASON_TYPE = 2

LEADERS_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/nfl/seasons/{season}/types/{season_type}/leaders"
)

TEAM_MASTER_PATH = NFL_ROOT / "data/master/team_master.csv"
OUTPUT_ROOT = NFL_ROOT / "data/league_leaders"
REPORT_ROOT = NFL_ROOT / "errors"

OUTPUT_HEADER = [
    "season",
    "category",
    "rank",
    "athlete_id",
    "team_id",
    "value",
    "displayValue",
]


class LeagueLeadersError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)

    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LeagueLeadersError(
            f"Failed ESPN request {url}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise LeagueLeadersError(
            f"Unexpected ESPN response type for {url}"
        )

    return payload


def extract_id(ref_url: str, segment: str) -> str:
    match = re.search(
        rf"/{re.escape(segment)}/([^/?]+)",
        clean(ref_url),
    )

    return clean(match.group(1)) if match else ""


def load_team_master_ids() -> tuple[set[str], int]:
    if not TEAM_MASTER_PATH.is_file():
        raise LeagueLeadersError(
            f"Missing team master: {TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if "team_id" not in set(reader.fieldnames or []):
            raise LeagueLeadersError(
                f"{TEAM_MASTER_PATH} missing team_id"
            )

        rows = list(reader)

    if not rows:
        raise LeagueLeadersError(
            f"{TEAM_MASTER_PATH} contains no rows"
        )

    team_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))

        if not team_id:
            raise LeagueLeadersError(
                f"{TEAM_MASTER_PATH} line {line_number} "
                "has blank team_id"
            )

        team_ids.add(team_id)

    if len(team_ids) != 32:
        raise LeagueLeadersError(
            f"{TEAM_MASTER_PATH} must resolve 32 unique team IDs; "
            f"found {len(team_ids)}"
        )

    return team_ids, len(rows)


def build_rows(
    data: dict[str, Any],
    season: int,
) -> tuple[list[dict[str, Any]], int, int]:
    categories = data.get("categories")

    if not isinstance(categories, list):
        raise LeagueLeadersError(
            "ESPN leaders response has invalid categories"
        )

    rows: list[dict[str, Any]] = []
    source_leader_count = 0

    for category_index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise LeagueLeadersError(
                f"Category {category_index} is not an object"
            )

        category_name = clean(category.get("name"))

        if not category_name:
            raise LeagueLeadersError(
                f"Category {category_index} has blank name"
            )

        leaders = category.get("leaders")

        if not isinstance(leaders, list):
            raise LeagueLeadersError(
                f"Category {category_name!r} has invalid leaders"
            )

        source_leader_count += len(leaders)

        for rank, leader in enumerate(leaders, start=1):
            if not isinstance(leader, dict):
                raise LeagueLeadersError(
                    f"Category {category_name!r} rank {rank} "
                    "is not an object"
                )

            athlete = leader.get("athlete")
            team = leader.get("team")

            athlete_ref = (
                athlete.get("$ref")
                if isinstance(athlete, dict)
                else ""
            )

            team_ref = (
                team.get("$ref")
                if isinstance(team, dict)
                else ""
            )

            athlete_id = extract_id(
                athlete_ref,
                "athletes",
            )

            team_id = extract_id(
                team_ref,
                "teams",
            )

            if not athlete_id:
                raise LeagueLeadersError(
                    f"Category {category_name!r} rank {rank} "
                    "has no usable athlete ID"
                )

            if not team_id:
                raise LeagueLeadersError(
                    f"Category {category_name!r} rank {rank} "
                    "has no usable team ID"
                )

            rows.append(
                {
                    "season": season,
                    "category": category_name,
                    "rank": rank,
                    "athlete_id": athlete_id,
                    "team_id": team_id,
                    "value": clean(leader.get("value")),
                    "displayValue": clean(
                        leader.get("displayValue")
                    ),
                }
            )

    return rows, len(categories), source_leader_count


def validate_rows(
    rows: list[dict[str, Any]],
    season: int,
    valid_team_ids: set[str],
) -> None:
    seen_keys: set[tuple[str, str, str]] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        category = clean(row.get("category"))
        rank = clean(row.get("rank"))
        athlete_id = clean(row.get("athlete_id"))
        team_id = clean(row.get("team_id"))

        if row_season != str(season):
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                f"has season={row_season!r}; expected {season}"
            )

        if not category:
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                "has blank category"
            )

        if not rank:
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                "has blank rank"
            )

        if not athlete_id:
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                "has blank athlete_id"
            )

        if not team_id:
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                "has blank team_id"
            )

        if team_id not in valid_team_ids:
            raise LeagueLeadersError(
                f"League leaders row {line_number} "
                f"has unknown team_id={team_id!r}"
            )

        key = (
            row_season,
            category,
            rank,
        )

        if key in seen_keys:
            raise LeagueLeadersError(
                f"Duplicate league-leader key: {key}"
            )

        seen_keys.add(key)


def publish(
    output_path: Path,
    rows: list[dict[str, Any]],
    season: int,
    valid_team_ids: set[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".league_leaders_stage_",
        dir=output_path.parent,
    ) as staging_dir:
        staged_path = (
            Path(staging_dir)
            / output_path.name
        )

        with staged_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_HEADER,
                extrasaction="ignore",
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(rows)

            handle.flush()
            os.fsync(handle.fileno())

        if staged_path.stat().st_size == 0:
            raise LeagueLeadersError(
                "Staged league leaders output is zero bytes"
            )

        with staged_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            staged_header = reader.fieldnames or []
            staged_rows = list(reader)

        if staged_header != OUTPUT_HEADER:
            raise LeagueLeadersError(
                "Staged league leaders headers changed"
            )

        if len(staged_rows) != len(rows):
            raise LeagueLeadersError(
                "Staged league leaders row count changed"
            )

        validate_rows(
            staged_rows,
            season,
            valid_team_ids,
        )

        os.replace(
            staged_path,
            output_path,
        )


def run(
    season: int,
    reporter: PipelineReporter,
) -> None:
    leaders_url = LEADERS_URL_TEMPLATE.format(
        season=season,
        season_type=SEASON_TYPE,
    )

    output_path = (
        OUTPUT_ROOT
        / f"league_leaders_{season}.csv"
    )

    reporter.add_input(
        TEAM_MASTER_PATH
    )

    reporter.add_input(
        leaders_url
    )

    reporter.update_details(
        {
            "season": season,
            "season_type": SEASON_TYPE,
            "leaders_url": leaders_url,
            "empty_data": False,
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    valid_team_ids, team_master_rows = (
        load_team_master_ids()
    )

    data = fetch_json(
        leaders_url
    )

    (
        rows,
        category_count,
        source_leader_count,
    ) = build_rows(
        data,
        season,
    )

    validate_rows(
        rows,
        season,
        valid_team_ids,
    )

    reporter.set_rows(
        rows_in=source_leader_count,
        rows_out=0,
    )

    reporter.update_details(
        {
            "team_master_rows": team_master_rows,
            "category_count": category_count,
            "source_leader_count": source_leader_count,
            "output_path": str(output_path),
        }
    )

    if not rows:
        reporter.warning(
            "ESPN regular-season leaders returned no rows; "
            "publishing header-only output",
            season=season,
            season_type=SEASON_TYPE,
        )

        reporter.set_detail(
            "empty_data",
            True,
        )

    publish(
        output_path,
        rows,
        season,
        valid_team_ids,
    )

    reporter.add_output(
        output_path
    )

    reporter.set_rows(
        rows_in=source_leader_count,
        rows_out=len(rows),
    )

    reporter.update_details(
        {
            "staged_roundtrip_verified": True,
            "rows_published": len(rows),
            "publication_completed": True,
        }
    )

    print(
        f"rows={len(rows)} output={output_path}"
    )


def main() -> int:
    args = parse_args()

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            extra_context={
                "component": "league leaders",
                "season_type": SEASON_TYPE,
            },
        ) as reporter:
            run(
                args.season,
                reporter,
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
