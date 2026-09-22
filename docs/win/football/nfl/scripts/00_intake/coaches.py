#!/usr/bin/env python3
"""
Pull each NFL team's current head coach from ESPN and build coaches_master.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

COACHES_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/teams/{team_id}/coaches"
)

REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    ),
}
HTTP_ATTEMPTS = 4
HTTP_TIMEOUT = 30
RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

TEAM_MASTER_PATH = NFL_ROOT / "data/master/team_master.csv"
OUTPUT_PATH = NFL_ROOT / "data/master/coaches_master.csv"
REPORT_ROOT = NFL_ROOT / "errors"

HEADER = [
    "sport",
    "league",
    "name",
    "team",
    "team_id",
    "experience",
    "career_record",
    "post_season_career_record",
    "id",
    "uid",
]

REQUIRED_OUTPUT_FIELDS = [
    "sport",
    "league",
    "name",
    "team",
    "team_id",
    "experience",
    "career_record",
    "id",
    "uid",
]


class CoachesError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NFL head-coach master data from ESPN."
    )
    parser.add_argument("--season", required=True, type=int)

    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def fetch_json(
    url: str,
    *,
    attempts: int = HTTP_ATTEMPTS,
    timeout: int = HTTP_TIMEOUT,
) -> dict[str, Any]:
    if not clean(url):
        raise CoachesError("Cannot fetch blank ESPN URL")

    if attempts < 1:
        raise CoachesError("ESPN request attempts must be at least 1")

    request = urllib.request.Request(
        url,
        headers=REQUEST_HEADERS,
    )
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read().decode("utf-8")

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CoachesError(
                    f"ESPN returned invalid JSON {url}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise CoachesError(
                    f"Unexpected ESPN response type for {url}"
                )

            return payload

        except urllib.error.HTTPError as exc:
            last_error = exc

            if (
                exc.code not in RETRYABLE_HTTP_CODES
                or attempt == attempts
            ):
                raise CoachesError(
                    f"Failed ESPN request {url}: "
                    f"HTTPError: status={exc.code}"
                ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

            if attempt == attempts:
                raise CoachesError(
                    f"Failed ESPN request {url}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        if attempt < attempts:
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"retry_espn_request "
                f"attempt={attempt}/{attempts} "
                f"delay={delay}s url={url}"
            )
            time.sleep(delay)

    if last_error is not None:
        raise CoachesError(
            f"Failed ESPN request {url}: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    raise CoachesError(
        f"Failed ESPN request {url} without an exception"
    )


def load_team_master() -> tuple[dict[str, str], int]:
    if not TEAM_MASTER_PATH.is_file():
        raise CoachesError(f"Missing team master: {TEAM_MASTER_PATH}")

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])

        missing = sorted(
            {"team_abbr", "team_id"} - columns
        )

        if missing:
            raise CoachesError(
                f"{TEAM_MASTER_PATH} missing columns: {missing}"
            )

        rows = list(reader)

    if not rows:
        raise CoachesError(
            f"{TEAM_MASTER_PATH} contains no data rows"
        )

    lookup: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        abbr = clean(row.get("team_abbr"))
        team_id = clean(row.get("team_id"))

        if not abbr or not team_id:
            raise CoachesError(
                f"{TEAM_MASTER_PATH} line {line_number} "
                "has blank team_abbr/team_id"
            )

        previous = lookup.get(abbr)

        if previous is not None and previous != team_id:
            raise CoachesError(
                f"{TEAM_MASTER_PATH} maps {abbr!r} to conflicting IDs "
                f"{previous!r} and {team_id!r}"
            )

        lookup[abbr] = team_id

    if len(lookup) != 32:
        raise CoachesError(
            f"{TEAM_MASTER_PATH} must resolve exactly 32 teams; "
            f"found {len(lookup)}"
        )

    return lookup, len(rows)



def get_career_records(
    coach: dict[str, Any],
) -> tuple[str, str]:
    person = coach.get("person")

    person_ref = (
        clean(person.get("$ref"))
        if isinstance(person, dict)
        else ""
    )

    if not person_ref:
        raise CoachesError(
            "Coach payload is missing person.$ref"
        )

    person_data = fetch_json(person_ref)

    career_records = person_data.get("careerRecords")

    if not isinstance(career_records, list):
        raise CoachesError(
            "Coach person payload has invalid careerRecords"
        )

    total = ""
    postseason = ""

    for index, item in enumerate(career_records):
        ref = (
            clean(item.get("$ref"))
            if isinstance(item, dict)
            else ""
        )

        if not ref:
            raise CoachesError(
                f"careerRecords item {index} is missing $ref"
            )

        record = fetch_json(ref)

        record_type = clean(record.get("type"))
        summary = clean(record.get("summary"))

        if record_type == "Total":
            total = summary
        elif record_type == "Post Season":
            postseason = summary

    if not total:
        raise CoachesError(
            "Coach career records contain no usable Total record"
        )

    return total, postseason


def resolve_head_coach(
    season: int,
    team_id: str,
    abbr: str,
) -> dict[str, str]:
    url = COACHES_URL_TEMPLATE.format(
        season=season,
        team_id=team_id,
    )

    coaches_list = fetch_json(url)
    items = coaches_list.get("items")

    if not isinstance(items, list) or not items:
        raise CoachesError(
            f"team={abbr} returned no coach items"
        )

    first = items[0]

    coach_ref = (
        clean(first.get("$ref"))
        if isinstance(first, dict)
        else ""
    )

    if not coach_ref:
        raise CoachesError(
            f"team={abbr} first coach item is missing $ref"
        )

    coach = fetch_json(coach_ref)

    career_record, postseason_record = (
        get_career_records(coach)
    )

    name = (
        f"{clean(coach.get('firstName'))} "
        f"{clean(coach.get('lastName'))}"
    ).strip()

    return {
        "sport": "football",
        "league": "nfl",
        "name": name,
        "team": abbr,
        "team_id": team_id,
        "experience": clean(coach.get("experience")),
        "career_record": career_record,
        "post_season_career_record": postseason_record,
        "id": clean(coach.get("id")),
        "uid": clean(coach.get("uid")),
    }


def validate_rows(
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) != 32:
        raise CoachesError(
            f"coaches_master must contain 32 rows; found {len(rows)}"
        )

    team_ids: set[str] = set()
    teams: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        missing = [
            field
            for field in REQUIRED_OUTPUT_FIELDS
            if not clean(row.get(field))
        ]

        if missing:
            raise CoachesError(
                f"coaches_master row {line_number} "
                f"has blank required fields: {missing}"
            )

        if (
            clean(row.get("sport")) != "football"
            or clean(row.get("league")) != "nfl"
        ):
            raise CoachesError(
                f"coaches_master row {line_number} "
                "has invalid sport/league"
            )

        team_id = clean(row.get("team_id"))
        team = clean(row.get("team"))

        if team_id in team_ids:
            raise CoachesError(
                f"coaches_master contains duplicate team_id={team_id!r}"
            )

        if team in teams:
            raise CoachesError(
                f"coaches_master contains duplicate team={team!r}"
            )

        team_ids.add(team_id)
        teams.add(team)

    if len(team_ids) != 32 or len(teams) != 32:
        raise CoachesError(
            "coaches_master does not contain 32 unique teams"
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HEADER,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            {
                field: clean(row.get(field))
                for field in HEADER
            }
            for row in rows
        )

        handle.flush()
        os.fsync(handle.fileno())


def stage_and_publish(
    rows: list[dict[str, Any]],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".coaches_stage_",
        dir=OUTPUT_PATH.parent,
    ) as staging_dir:
        staged = (
            Path(staging_dir)
            / OUTPUT_PATH.name
        )

        write_csv(staged, rows)

        with staged.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)

            if (reader.fieldnames or []) != HEADER:
                raise CoachesError(
                    "Staged output has invalid headers"
                )

            staged_rows = list(reader)

        validate_rows(staged_rows)

        if len(staged_rows) != len(rows):
            raise CoachesError(
                "Staged row count changed during "
                "round-trip verification"
            )

        os.replace(
            staged,
            OUTPUT_PATH,
        )


def run(
    season: int,
    reporter: PipelineReporter,
) -> None:
    reporter.add_input(
        TEAM_MASTER_PATH
    )

    reporter.update_details(
        {
            "team_universe_source": str(TEAM_MASTER_PATH),
            "coach_url_template": COACHES_URL_TEMPLATE,
            "publication_completed": False,
            "staged_roundtrip_verified": False,
            "teams_resolved": 0,
        }
    )

    team_master, team_master_rows = (
        load_team_master()
    )

    reporter.set_rows(
        rows_in=team_master_rows,
        rows_out=0,
    )

    reporter.update_details(
        {
            "team_master_rows": team_master_rows,
            "team_master_unique_teams": len(team_master),
        }
    )

    rows: list[dict[str, str]] = []

    for abbr, team_id in sorted(team_master.items()):
        rows.append(
            resolve_head_coach(
                season,
                team_id,
                abbr,
            )
        )

        reporter.set_detail(
            "teams_resolved",
            len(rows),
        )

    validate_rows(rows)

    stage_and_publish(rows)

    reporter.set_detail(
        "staged_roundtrip_verified",
        True,
    )

    reporter.add_output(
        OUTPUT_PATH
    )

    reporter.set_rows(
        rows_in=team_master_rows,
        rows_out=len(rows),
    )

    reporter.update_details(
        {
            "rows_published": len(rows),
            "publication_completed": True,
        }
    )

    print(
        f"rows={len(rows)} output={OUTPUT_PATH}"
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
                "component": "head coaches",
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
