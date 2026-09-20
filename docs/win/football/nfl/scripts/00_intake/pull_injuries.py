#!/usr/bin/env python3
"""
Pull the configured NFL season's injury data from ESPN's Site API and publish
a validated seven-column season injury CSV.

Endpoint:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/injuries"
)
OUTPUT_DIR = NFL_ROOT / "00_intake" / "injuries"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"

OUTPUT_HEADERS = [
    "season",
    "team",
    "player_id",
    "player_name",
    "position",
    "game_status",
    "report_date",
]

RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


class InjuryPullError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise InjuryPullError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull and validate ESPN NFL injury data for "
            "one configured season."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year to require from the ESPN response.",
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def read_csv(
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(
            f"Could not read {label} {path}: "
            f"{type(exc).__name__}: {exc}"
        )

    if not fieldnames:
        fail(f"{label} has no CSV header: {path}")

    if len(fieldnames) != len(set(fieldnames)):
        fail(f"{label} contains duplicate CSV columns: {path}")

    if required_columns:
        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            fail(
                f"{label} missing expected columns: {missing}"
            )

    if exact_columns is not None and fieldnames != exact_columns:
        fail(
            f"{label} has unexpected column order/schema. "
            f"Expected={exact_columns} actual={fieldnames}"
        )

    return fieldnames, rows


def load_canonical_team_names() -> set[str]:
    _, rows = read_csv(
        TEAM_MAP_PATH,
        label="NFL team map",
        required_columns=[
            "sport",
            "league",
            "team_id",
            "canonical_team",
        ],
    )

    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}

    for line_number, row in enumerate(rows, start=2):
        sport = clean(row.get("sport")).casefold()
        league = clean(row.get("league")).casefold()

        if sport != "football" or league != "nfl":
            continue

        team_id = clean(row.get("team_id"))
        canonical_team = clean(row.get("canonical_team"))

        if not team_id or not canonical_team:
            fail(
                f"{TEAM_MAP_PATH} line {line_number} has "
                "blank team_id/canonical_team"
            )

        previous_name = by_id.get(team_id)
        if previous_name and previous_name != canonical_team:
            fail(
                f"{TEAM_MAP_PATH} has conflicting canonical team "
                f"names for team_id={team_id}: "
                f"{previous_name!r} vs {canonical_team!r}"
            )

        previous_id = by_name.get(canonical_team)
        if previous_id and previous_id != team_id:
            fail(
                f"{TEAM_MAP_PATH} has conflicting team IDs for "
                f"canonical_team={canonical_team!r}: "
                f"{previous_id!r} vs {team_id!r}"
            )

        by_id[team_id] = canonical_team
        by_name[canonical_team] = team_id

    if len(by_id) != 32 or len(by_name) != 32:
        fail(
            "Canonical NFL team map must resolve exactly 32 teams; "
            f"ids={len(by_id)} names={len(by_name)}"
        )

    return set(by_name)


def fetch_json(
    url: str,
    *,
    timeout: int = 15,
    attempts: int = 4,
) -> tuple[dict[str, Any], int]:
    if attempts < 1:
        fail("fetch attempts must be at least 1")

    headers = {
        "Accept": "application/json",
        "User-Agent": "football_for_mat/1.0",
    }
    request = urllib.request.Request(
        url,
        headers=headers,
    )

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read()

            try:
                payload = json.loads(
                    raw.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                fail(
                    "ESPN injury response was not valid UTF-8 JSON: "
                    f"{exc}"
                )

            if not isinstance(payload, dict):
                fail(
                    "ESPN injury response root must be an object"
                )

            return payload, attempt

        except urllib.error.HTTPError as exc:
            last_error = exc

            if (
                exc.code not in RETRYABLE_HTTP_CODES
                or attempt == attempts
            ):
                raise

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = exc

            if attempt == attempts:
                raise

        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))

    if last_error is not None:
        raise last_error

    fail("ESPN injury request failed without an exception")


def extract_player_id(athlete: dict[str, Any]) -> str:
    links = athlete.get("links", [])

    if not isinstance(links, list):
        return ""

    for link in links:
        if not isinstance(link, dict):
            continue

        rel = link.get("rel", [])
        if not isinstance(rel, list):
            continue

        if "playercard" not in rel:
            continue

        href = clean(link.get("href"))
        match = re.search(r"/id/(\d+)/", href)

        if match:
            return match.group(1)

    return ""


def build_rows(
    data: dict[str, Any],
    *,
    requested_season: int,
    canonical_teams: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    season = data.get("season")

    if not isinstance(season, dict):
        fail("ESPN injury response missing season object")

    response_season = clean(season.get("year"))

    if response_season != str(requested_season):
        fail(
            "ESPN injury response season does not match "
            f"requested season. requested={requested_season} "
            f"received={response_season!r}"
        )

    team_entries = data.get("injuries")

    if not isinstance(team_entries, list):
        fail("ESPN injury response missing injuries list")

    if len(team_entries) != 32:
        fail(
            "ESPN injury response must contain exactly 32 "
            f"team groups; received={len(team_entries)}"
        )

    seen_teams: set[str] = set()
    rows: list[dict[str, str]] = []
    team_row_counts: dict[str, int] = {}

    for team_index, team_entry in enumerate(
        team_entries,
        start=1,
    ):
        if not isinstance(team_entry, dict):
            fail(
                "ESPN injury team entry must be an object "
                f"index={team_index}"
            )

        team_name = clean(
            team_entry.get("displayName")
        )

        if not team_name:
            fail(
                "ESPN injury team entry has blank displayName "
                f"index={team_index}"
            )

        if team_name not in canonical_teams:
            fail(
                "ESPN injury response contains unexpected team "
                f"{team_name!r}"
            )

        if team_name in seen_teams:
            fail(
                "ESPN injury response contains duplicate team "
                f"{team_name!r}"
            )

        team_injuries = team_entry.get("injuries")

        if not isinstance(team_injuries, list):
            fail(
                "ESPN injury team entry missing injuries list "
                f"team={team_name}"
            )

        seen_teams.add(team_name)
        team_row_counts[team_name] = len(
            team_injuries
        )

        for injury_index, injury in enumerate(
            team_injuries,
            start=1,
        ):
            if not isinstance(injury, dict):
                fail(
                    "ESPN injury item must be an object "
                    f"team={team_name} "
                    f"index={injury_index}"
                )

            athlete = injury.get("athlete")
            if not isinstance(athlete, dict):
                fail(
                    "ESPN injury item missing athlete object "
                    f"team={team_name} "
                    f"index={injury_index}"
                )

            position = athlete.get("position")
            if not isinstance(position, dict):
                fail(
                    "ESPN injury athlete missing position object "
                    f"team={team_name} "
                    f"index={injury_index}"
                )

            rows.append(
                {
                    "season": str(requested_season),
                    "team": team_name,
                    "player_id": extract_player_id(
                        athlete
                    ),
                    "player_name": clean(
                        athlete.get("displayName")
                    ),
                    "position": clean(
                        position.get("abbreviation")
                    ),
                    "game_status": clean(
                        injury.get("status")
                    ),
                    "report_date": clean(
                        injury.get("date")
                    ),
                }
            )

    if seen_teams != canonical_teams:
        fail(
            "ESPN injury team universe mismatch "
            f"missing={sorted(canonical_teams - seen_teams)} "
            f"extra={sorted(seen_teams - canonical_teams)}"
        )

    if not rows:
        fail(
            "ESPN injury response produced no injury rows"
        )

    return rows, team_row_counts


def validate_rows(
    rows: list[dict[str, str]],
    *,
    requested_season: int,
    canonical_teams: set[str],
) -> None:
    if not rows:
        fail("Injury output contains no data rows")

    seen_player_ids: set[str] = set()
    represented_teams: set[str] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        for field in OUTPUT_HEADERS:
            if not clean(row.get(field)):
                fail(
                    "Injury output row has blank required field "
                    f"line={line_number} field={field}"
                )

        season = clean(row.get("season"))
        team = clean(row.get("team"))
        player_id = clean(row.get("player_id"))

        if season != str(requested_season):
            fail(
                "Injury output season mismatch "
                f"line={line_number} "
                f"expected={requested_season} "
                f"received={season!r}"
            )

        if team not in canonical_teams:
            fail(
                "Injury output contains noncanonical team "
                f"line={line_number} team={team!r}"
            )

        if not player_id.isdigit():
            fail(
                "Injury output player_id must be numeric "
                f"line={line_number} "
                f"player_id={player_id!r}"
            )

        if player_id in seen_player_ids:
            fail(
                "Injury output contains duplicate player_id="
                f"{player_id}"
            )

        seen_player_ids.add(player_id)
        represented_teams.add(team)

    if not represented_teams:
        fail("Injury output represents no NFL teams")


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            column: clean(row.get(column))
            for column in OUTPUT_HEADERS
        }
        for row in rows
    ]


def publish(
    rows: list[dict[str, str]],
    *,
    output_file: Path,
    requested_season: int,
    canonical_teams: set[str],
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{output_file.stem}_",
        suffix=".tmp",
        dir=output_file.parent,
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        staged_columns, staged_rows = read_csv(
            staged_path,
            label="staged injury CSV",
            exact_columns=OUTPUT_HEADERS,
        )

        if staged_columns != OUTPUT_HEADERS:
            fail(
                "Staged injury schema changed during "
                "round-trip validation"
            )

        validate_rows(
            staged_rows,
            requested_season=requested_season,
            canonical_teams=canonical_teams,
        )

        if staged_rows != normalize_rows(rows):
            fail(
                "Staged injury CSV differs from validated "
                "in-memory projection"
            )

        os.replace(
            staged_path,
            output_file,
        )

    finally:
        if staged_path.exists():
            staged_path.unlink()


def run(
    reporter: PipelineReporter,
    *,
    requested_season: int,
) -> None:
    output_file = (
        OUTPUT_DIR
        / f"{requested_season}_injuries.csv"
    )

    reporter.add_input(TEAM_MAP_PATH)
    reporter.update_details(
        {
            "endpoint": INJURIES_URL,
            "requested_season": requested_season,
            "output_file": str(output_file),
            "expected_output_columns": len(
                OUTPUT_HEADERS
            ),
            "expected_team_groups": 32,
            "publication_mode": "staged_atomic_replace",
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    canonical_teams = (
        load_canonical_team_names()
    )

    data, fetch_attempt = fetch_json(
        INJURIES_URL
    )

    rows, team_row_counts = build_rows(
        data,
        requested_season=requested_season,
        canonical_teams=canonical_teams,
    )

    validate_rows(
        rows,
        requested_season=requested_season,
        canonical_teams=canonical_teams,
    )

    min_team_rows = min(
        team_row_counts.values()
    )
    max_team_rows = max(
        team_row_counts.values()
    )

    reporter.set_rows(
        rows_in=len(rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            "response_season": requested_season,
            "fetch_attempt": fetch_attempt,
            "team_groups": len(
                team_row_counts
            ),
            "injury_rows": len(rows),
            "min_team_rows": min_team_rows,
            "max_team_rows": max_team_rows,
            "team_row_counts": dict(
                sorted(team_row_counts.items())
            ),
            "unique_player_ids": len(
                {
                    clean(row.get("player_id"))
                    for row in rows
                }
            ),
        }
    )

    zero_row_teams = sorted(
        team
        for team, count
        in team_row_counts.items()
        if count == 0
    )
    if zero_row_teams:
        reporter.warning(
            "ESPN returned team groups with zero injury rows",
            teams=zero_row_teams,
        )

    publish(
        rows,
        output_file=output_file,
        requested_season=requested_season,
        canonical_teams=canonical_teams,
    )

    reporter.add_output(output_file)
    reporter.set_rows(
        rows_in=len(rows),
        rows_out=len(rows),
    )
    reporter.update_details(
        {
            "rows_published": len(rows),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    print(
        f"season={requested_season} "
        f"teams={len(team_row_counts)} "
        f"rows={len(rows)} "
        f"output={output_file}"
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
                "component": "injury pull",
            },
        ) as reporter:
            run(
                reporter,
                requested_season=args.season,
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
