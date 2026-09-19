#!/usr/bin/env python3
"""
Pull the configured NFL season roster from ESPN Core and publish one flattened
raw CSV compatible with roster_cleanup.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

CORE_BASE = "https://sports.core.api.espn.com/v2"
OUTPUT_PATH = NFL_ROOT / "data" / "raw" / "raw_roster.csv"
TEAM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "team_map.csv"
REPORT_ROOT = NFL_ROOT / "errors"

REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    ),
}

RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

COMPATIBILITY_COLUMNS = [
    "age",
    "alternateIds.sdr",
    "birthPlace.city",
    "birthPlace.country",
    "birthPlace.state",
    "college.abbrev",
    "college.guid",
    "college.id",
    "college.name",
    "college.shortName",
    "contract.active",
    "contract.bonus",
    "contract.optionType",
    "contract.salary",
    "contract.salaryRemaining",
    "contract.season.endDate",
    "contract.season.startDate",
    "contract.season.year",
    "contract.signedThrough",
    "dateOfBirth",
    "debutYear",
    "displayHeight",
    "displayName",
    "displayWeight",
    "experience.years",
    "firstName",
    "fullName",
    "guid",
    "hand.abbreviation",
    "hand.displayValue",
    "hand.type",
    "headshot.alt",
    "headshot.href",
    "height",
    "id",
    "injuries.0.date",
    "injuries.0.status",
    "jersey",
    "lastName",
    "position.abbreviation",
    "position.displayName",
    "position.id",
    "position.leaf",
    "position.name",
    "position.parent.abbreviation",
    "position.parent.displayName",
    "position.parent.id",
    "position.parent.leaf",
    "position.parent.name",
    "shortName",
    "slug",
    "status.abbreviation",
    "status.id",
    "status.name",
    "status.type",
    "team_id",
    "uid",
    "weight",
]

CORE_REQUIRED_FIELDS = [
    "id",
    "displayName",
    "position.id",
    "position.abbreviation",
    "team_id",
]


class RosterPullError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise RosterPullError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")

    return args


def get_workers() -> int:
    raw = clean(os.environ.get("NFL_ROSTER_WORKERS", "8"))

    try:
        workers = int(raw)
    except ValueError:
        fail(
            "NFL_ROSTER_WORKERS must be an integer; "
            f"received {raw!r}"
        )

    if not 1 <= workers <= 32:
        fail(
            "NFL_ROSTER_WORKERS must be between 1 and 32; "
            f"received {workers}"
        )

    return workers


def normalize_ref(url: Any) -> str:
    text = clean(url)

    if text.startswith("http://"):
        return "https://" + text[len("http://"):]

    return text


def fetch_json(
    url: str,
    *,
    retries: int = 4,
    timeout: int = 30,
) -> dict[str, Any]:
    url = normalize_ref(url)

    if not url:
        fail("Cannot fetch blank ESPN URL")

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers=REQUEST_HEADERS,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read().decode("utf-8")

            payload = json.loads(raw)

            if not isinstance(payload, dict):
                fail(
                    "ESPN response must be a JSON object "
                    f"url={url}"
                )

            return payload

        except urllib.error.HTTPError as exc:
            if (
                exc.code in RETRYABLE_HTTP_CODES
                and attempt < retries
            ):
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_http status={exc.code} "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url}"
                )
                time.sleep(delay)
                continue

            raise RosterPullError(
                f"ESPN HTTP error status={exc.code} "
                f"url={url}"
            ) from exc

        except urllib.error.URLError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_url_error "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url} "
                    f"reason={exc.reason}"
                )
                time.sleep(delay)
                continue

            raise RosterPullError(
                f"ESPN request failed url={url} "
                f"reason={exc.reason}"
            ) from exc

        except TimeoutError as exc:
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"retry_timeout "
                    f"attempt={attempt}/{retries} "
                    f"delay={delay}s url={url}"
                )
                time.sleep(delay)
                continue

            raise RosterPullError(
                f"ESPN request timed out url={url}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RosterPullError(
                f"ESPN returned invalid JSON url={url}"
            ) from exc

    fail(f"Unable to fetch ESPN URL: {url}")


def flatten(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    items: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else str(key)
            )
            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            new_key = (
                f"{parent_key}{sep}{index}"
                if parent_key
                else str(index)
            )
            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )

    else:
        items[parent_key] = obj

    return items


def id_from_ref(
    ref: Any,
    resource_name: str,
) -> str:
    normalized = normalize_ref(ref)

    if not normalized:
        fail(f"Missing {resource_name} $ref")

    path = urlparse(normalized).path.rstrip("/")
    value = path.split("/")[-1]

    if not value.isdigit():
        fail(
            f"Unable to extract {resource_name} ID "
            f"from ref={normalized}"
        )

    return value


def load_canonical_team_ids(
    reporter: PipelineReporter,
) -> set[str]:
    reporter.add_input(TEAM_MAP_PATH)

    if not TEAM_MAP_PATH.is_file():
        fail(f"Missing team map: {TEAM_MAP_PATH}")

    with TEAM_MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []

        required = [
            "sport",
            "league",
            "team_id",
        ]
        missing = [
            column
            for column in required
            if column not in columns
        ]

        if missing:
            fail(
                f"{TEAM_MAP_PATH} missing columns: {missing}"
            )

        team_ids: set[str] = set()

        for row in reader:
            sport = clean(row.get("sport")).casefold()
            league = clean(row.get("league")).casefold()

            if sport not in {"", "football"}:
                continue
            if league not in {"", "nfl"}:
                continue

            team_id = clean(row.get("team_id"))
            if team_id:
                team_ids.add(team_id)

    if len(team_ids) != 32:
        fail(
            f"{TEAM_MAP_PATH} must resolve exactly 32 "
            f"NFL team IDs; found {len(team_ids)}"
        )

    return team_ids


def season_url(season: int) -> str:
    return (
        f"{CORE_BASE}/sports/football/leagues/nfl/"
        f"seasons/{season}?lang=en&region=us"
    )


def teams_url(season: int) -> str:
    return (
        f"{CORE_BASE}/sports/football/leagues/nfl/"
        f"seasons/{season}/teams?limit=100"
    )


def team_athletes_url(
    season: int,
    team_id: str,
) -> str:
    return (
        f"{CORE_BASE}/sports/football/leagues/nfl/"
        f"seasons/{season}/teams/{team_id}/athletes"
        "?limit=200"
    )


def get_team_ids(
    season: int,
    canonical_team_ids: set[str],
) -> list[str]:
    data = fetch_json(
        teams_url(season)
    )
    items = data.get("items", [])

    if not isinstance(items, list) or not items:
        fail(
            f"No NFL teams returned for season={season}"
        )

    team_ids: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            fail(
                f"Invalid ESPN team item for season={season}"
            )

        ref = item.get("$ref")
        if not ref:
            fail(
                f"ESPN team item missing $ref for season={season}"
            )

        team_id = id_from_ref(
            ref,
            "team",
        )

        if team_id in team_ids:
            fail(
                f"Duplicate ESPN team_id={team_id} "
                f"for season={season}"
            )

        team_ids.append(team_id)

    expected_count = data.get("count")
    if expected_count is not None:
        try:
            expected = int(expected_count)
        except (TypeError, ValueError):
            fail(
                "ESPN teams response has invalid count="
                f"{expected_count!r}"
            )

        if len(team_ids) != expected:
            fail(
                "Team count mismatch "
                f"expected={expected} "
                f"received={len(team_ids)}"
            )

    observed = set(team_ids)

    if observed != canonical_team_ids:
        fail(
            "ESPN team IDs do not exactly match canonical "
            "NFL team IDs. "
            f"missing={sorted(canonical_team_ids - observed)} "
            f"extra={sorted(observed - canonical_team_ids)}"
        )

    print(
        f"teams={len(team_ids)} "
        f"season={season}"
    )

    return team_ids


def get_team_athlete_refs(
    season: int,
    team_id: str,
) -> list[str]:
    url = team_athletes_url(
        season,
        team_id,
    )
    data = fetch_json(url)

    items = data.get("items", [])
    if not isinstance(items, list):
        fail(
            f"Invalid athlete item collection "
            f"team_id={team_id}"
        )

    refs: list[str] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            fail(
                f"Invalid athlete item team_id={team_id}"
            )

        ref = normalize_ref(
            item.get("$ref")
        )

        if not ref:
            fail(
                f"Athlete item missing $ref "
                f"team_id={team_id}"
            )

        if ref in seen:
            fail(
                f"Duplicate athlete $ref "
                f"team_id={team_id} ref={ref}"
            )

        seen.add(ref)
        refs.append(ref)

    expected_count = data.get("count")
    if expected_count is not None:
        try:
            expected = int(expected_count)
        except (TypeError, ValueError):
            fail(
                f"Invalid athlete count team_id={team_id} "
                f"count={expected_count!r}"
            )

        if len(refs) != expected:
            fail(
                f"Athlete count mismatch "
                f"team_id={team_id} "
                f"expected={expected} "
                f"received={len(refs)} "
                f"url={url}"
            )

    if not refs:
        fail(
            f"No athletes returned for team_id={team_id} "
            f"season={season}"
        )

    print(
        f"team_id={team_id} "
        f"athletes={len(refs)}"
    )

    return refs


def fetch_athlete_entry(
    entry: tuple[str, str],
) -> dict[str, Any]:
    team_id, athlete_ref = entry
    athlete = fetch_json(athlete_ref)

    athlete_id = clean(
        athlete.get("id")
    )

    if not athlete_id:
        fail(
            f"Athlete response missing id "
            f"team_id={team_id} "
            f"url={athlete_ref}"
        )

    return {
        "team_id": team_id,
        "athlete_ref": athlete_ref,
        "athlete": athlete,
    }


def fetch_optional_json(
    url: str,
) -> tuple[str, dict[str, Any] | None, str]:
    try:
        return url, fetch_json(url), ""
    except Exception as exc:
        return (
            url,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def fetch_reference_map(
    refs: set[str],
    *,
    workers: int,
) -> tuple[
    dict[str, dict[str, Any]],
    list[tuple[str, str]],
]:
    unique_refs = sorted(
        {
            normalize_ref(ref)
            for ref in refs
            if normalize_ref(ref)
        }
    )

    if not unique_refs:
        return {}, []

    result: dict[str, dict[str, Any]] = {}
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        for ref, data, error in executor.map(
            fetch_optional_json,
            unique_refs,
        ):
            if data is None:
                failures.append(
                    (
                        ref,
                        error or "unknown optional-ref failure",
                    )
                )
            else:
                result[ref] = data

    return result, failures


def validate_athlete_records(
    records: list[dict[str, Any]],
    *,
    canonical_team_ids: set[str],
) -> None:
    if not records:
        fail("No athlete records were fetched")

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    athlete_teams: dict[str, set[str]] = {}
    represented_teams: set[str] = set()

    for record in records:
        team_id = clean(record.get("team_id"))
        athlete = record.get("athlete")

        if team_id not in canonical_team_ids:
            fail(
                f"Fetched athlete record has unknown "
                f"team_id={team_id!r}"
            )

        if not isinstance(athlete, dict):
            fail(
                f"Fetched athlete record for team_id={team_id} "
                "is not an object"
            )

        athlete_id = clean(
            athlete.get("id")
        )

        if not athlete_id:
            fail(
                f"Fetched athlete record team_id={team_id} "
                "has blank athlete id"
            )

        pair = (athlete_id, team_id)

        if pair in seen_pairs:
            fail(
                "Duplicate athlete/team membership: "
                f"athlete_id={athlete_id} team_id={team_id}"
            )

        seen_pairs.add(pair)
        represented_teams.add(team_id)

        athlete_teams.setdefault(
            athlete_id,
            set(),
        ).add(team_id)

        if athlete_id in seen_ids:
            continue

        seen_ids.add(athlete_id)

    multi_team = {
        athlete_id: sorted(team_ids)
        for athlete_id, team_ids in athlete_teams.items()
        if len(team_ids) > 1
    }

    if multi_team:
        examples = list(
            sorted(multi_team.items())
        )[:10]
        fail(
            "Athletes simultaneously assigned to multiple "
            f"NFL teams. Examples={examples}"
        )

    if represented_teams != canonical_team_ids:
        fail(
            "Fetched athlete records do not cover all "
            "canonical NFL teams"
        )

    if len(seen_ids) != len(records):
        fail(
            "Fetched athlete IDs are not unique: "
            f"records={len(records)} "
            f"unique_ids={len(seen_ids)}"
        )


def build_rows(
    athlete_records: list[dict[str, Any]],
    *,
    college_map: dict[str, dict[str, Any]],
    position_parent_map: dict[str, dict[str, Any]],
    season_year: int,
    season_start_date: str,
    season_end_date: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    all_rows: list[dict[str, Any]] = []
    all_columns = set(COMPATIBILITY_COLUMNS)

    for record in athlete_records:
        team_id = clean(record["team_id"])
        athlete = dict(record["athlete"])

        college = athlete.get("college")
        if isinstance(college, dict):
            college_ref = normalize_ref(
                college.get("$ref")
            )
            if (
                college_ref
                and college_ref in college_map
            ):
                athlete["college"] = dict(
                    college_map[college_ref]
                )

        position = athlete.get("position")
        if isinstance(position, dict):
            position = dict(position)
            athlete["position"] = position
            parent = position.get("parent")

            if isinstance(parent, dict):
                parent_ref = normalize_ref(
                    parent.get("$ref")
                )
                if (
                    parent_ref
                    and parent_ref
                    in position_parent_map
                ):
                    position["parent"] = dict(
                        position_parent_map[
                            parent_ref
                        ]
                    )

        contract = athlete.get("contract")
        if isinstance(contract, dict):
            contract = dict(contract)
            athlete["contract"] = contract

            contract_season = contract.get("season")
            if not isinstance(contract_season, dict):
                contract_season = {}
            else:
                contract_season = dict(
                    contract_season
                )

            contract_season["year"] = season_year
            contract_season["startDate"] = (
                season_start_date
            )
            contract_season["endDate"] = (
                season_end_date
            )
            contract["season"] = contract_season

        flat_row = flatten(athlete)
        flat_row["team_id"] = team_id

        flat_row.setdefault(
            "hand.abbreviation",
            "",
        )
        flat_row.setdefault(
            "hand.displayValue",
            "",
        )
        flat_row.setdefault(
            "hand.type",
            "",
        )

        for column in COMPATIBILITY_COLUMNS:
            flat_row.setdefault(
                column,
                "",
            )

        all_rows.append(flat_row)
        all_columns.update(
            flat_row.keys()
        )

    if not all_rows:
        fail(
            f"No roster rows created "
            f"for season={season_year}"
        )

    return (
        all_rows,
        sorted(all_columns),
    )


def validate_flat_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    season: int,
    canonical_team_ids: set[str],
) -> None:
    if not rows:
        fail("Raw roster contains no rows")

    missing_columns = [
        column
        for column in COMPATIBILITY_COLUMNS
        if column not in fieldnames
    ]

    if missing_columns:
        fail(
            "Raw roster missing compatibility columns: "
            f"{missing_columns}"
        )

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    represented_teams: set[str] = set()

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        for field in CORE_REQUIRED_FIELDS:
            if not clean(row.get(field)):
                fail(
                    f"Raw roster line {line_number} "
                    f"has blank {field}"
                )

        athlete_id = clean(
            row.get("id")
        )
        team_id = clean(
            row.get("team_id")
        )

        if team_id not in canonical_team_ids:
            fail(
                f"Raw roster line {line_number} "
                f"has unknown team_id={team_id!r}"
            )

        if athlete_id in seen_ids:
            fail(
                f"Raw roster duplicate athlete id="
                f"{athlete_id}"
            )

        pair = (athlete_id, team_id)
        if pair in seen_pairs:
            fail(
                f"Raw roster duplicate athlete/team "
                f"pair={pair}"
            )

        seen_ids.add(athlete_id)
        seen_pairs.add(pair)
        represented_teams.add(team_id)

        contract_year = clean(
            row.get("contract.season.year")
        )

        if contract_year:
            try:
                parsed_year = int(
                    float(contract_year)
                )
            except ValueError:
                fail(
                    f"Raw roster athlete_id={athlete_id} "
                    "has invalid contract.season.year="
                    f"{contract_year!r}"
                )

            if parsed_year != season:
                fail(
                    f"Raw roster athlete_id={athlete_id} "
                    f"has contract season={parsed_year}; "
                    f"expected {season}"
                )

    if represented_teams != canonical_team_ids:
        fail(
            "Raw roster team IDs do not exactly match "
            "the canonical 32-team universe"
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def publish(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    season: int,
    canonical_team_ids: set[str],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".raw_roster_stage_",
        dir=OUTPUT_PATH.parent,
    ) as staging_dir:
        staged_path = (
            Path(staging_dir)
            / OUTPUT_PATH.name
        )

        write_csv(
            staged_path,
            rows,
            fieldnames,
        )

        if staged_path.stat().st_size == 0:
            fail(
                "Staged raw roster CSV is zero bytes"
            )

        with staged_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            staged_fieldnames = (
                reader.fieldnames
                or []
            )
            staged_rows = list(reader)

        if staged_fieldnames != fieldnames:
            fail(
                "Staged raw roster headers changed "
                "during round-trip validation"
            )

        if len(staged_rows) != len(rows):
            fail(
                "Staged raw roster row count changed "
                "during round-trip validation"
            )

        validate_flat_rows(
            staged_rows,
            staged_fieldnames,
            season=season,
            canonical_team_ids=(
                canonical_team_ids
            ),
        )

        os.replace(
            staged_path,
            OUTPUT_PATH,
        )


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    season = args.season
    workers = get_workers()

    reporter.update_details(
        {
            "configured_season": season,
            "workers": workers,
            "output_path": str(OUTPUT_PATH),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    canonical_team_ids = (
        load_canonical_team_ids(
            reporter
        )
    )

    season_data = fetch_json(
        season_url(season)
    )

    season_year_raw = season_data.get(
        "year"
    )

    try:
        season_year = int(
            season_year_raw
        )
    except (TypeError, ValueError):
        fail(
            "ESPN season metadata has invalid year="
            f"{season_year_raw!r}"
        )

    if season_year != season:
        fail(
            f"ESPN season metadata year={season_year}; "
            f"expected {season}"
        )

    season_start_date = clean(
        season_data.get("startDate")
    )
    season_end_date = clean(
        season_data.get("endDate")
    )

    team_ids = get_team_ids(
        season,
        canonical_team_ids,
    )

    athlete_entries: list[
        tuple[str, str]
    ] = []

    for team_id in team_ids:
        athlete_refs = (
            get_team_athlete_refs(
                season,
                team_id,
            )
        )

        for athlete_ref in athlete_refs:
            athlete_entries.append(
                (
                    team_id,
                    athlete_ref,
                )
            )

    if not athlete_entries:
        fail(
            f"No NFL athletes found "
            f"for season={season}"
        )

    print(
        f"athlete_refs="
        f"{len(athlete_entries)}"
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        athlete_records = list(
            executor.map(
                fetch_athlete_entry,
                athlete_entries,
            )
        )

    validate_athlete_records(
        athlete_records,
        canonical_team_ids=(
            canonical_team_ids
        ),
    )

    print(
        f"athlete_records="
        f"{len(athlete_records)}"
    )

    college_refs: set[str] = set()
    position_parent_refs: set[str] = set()

    for record in athlete_records:
        athlete = record["athlete"]

        college = athlete.get("college")
        if isinstance(college, dict):
            college_ref = normalize_ref(
                college.get("$ref")
            )
            if college_ref:
                college_refs.add(
                    college_ref
                )

        position = athlete.get("position")
        if isinstance(position, dict):
            parent = position.get("parent")
            if isinstance(parent, dict):
                parent_ref = normalize_ref(
                    parent.get("$ref")
                )
                if parent_ref:
                    position_parent_refs.add(
                        parent_ref
                    )

    (
        college_map,
        college_failures,
    ) = fetch_reference_map(
        college_refs,
        workers=workers,
    )

    (
        position_parent_map,
        position_parent_failures,
    ) = fetch_reference_map(
        position_parent_refs,
        workers=workers,
    )

    for ref, error in college_failures:
        reporter.warning(
            "Optional college reference fetch failed",
            ref=ref,
            error=error,
        )
        print(
            "warning_optional_ref_failed "
            f"url={ref} error={error}"
        )

    for ref, error in position_parent_failures:
        reporter.warning(
            "Optional position-parent reference fetch failed",
            ref=ref,
            error=error,
        )
        print(
            "warning_optional_ref_failed "
            f"url={ref} error={error}"
        )

    print(
        f"college_refs={len(college_refs)} "
        f"college_records={len(college_map)}"
    )

    print(
        f"position_parent_refs="
        f"{len(position_parent_refs)} "
        f"position_parent_records="
        f"{len(position_parent_map)}"
    )

    (
        all_rows,
        fieldnames,
    ) = build_rows(
        athlete_records,
        college_map=college_map,
        position_parent_map=(
            position_parent_map
        ),
        season_year=season_year,
        season_start_date=(
            season_start_date
        ),
        season_end_date=season_end_date,
    )

    validate_flat_rows(
        all_rows,
        fieldnames,
        season=season,
        canonical_team_ids=(
            canonical_team_ids
        ),
    )

    reporter.set_rows(
        rows_in=len(athlete_entries),
        rows_out=0,
    )
    reporter.update_details(
        {
            "canonical_team_count": (
                len(canonical_team_ids)
            ),
            "espn_team_count": len(team_ids),
            "athlete_refs": len(
                athlete_entries
            ),
            "athlete_records": len(
                athlete_records
            ),
            "college_refs": len(
                college_refs
            ),
            "college_records": len(
                college_map
            ),
            "college_failures": len(
                college_failures
            ),
            "position_parent_refs": len(
                position_parent_refs
            ),
            "position_parent_records": len(
                position_parent_map
            ),
            "position_parent_failures": len(
                position_parent_failures
            ),
            "output_columns": len(
                fieldnames
            ),
        }
    )

    publish(
        all_rows,
        fieldnames,
        season=season,
        canonical_team_ids=(
            canonical_team_ids
        ),
    )

    reporter.add_output(
        OUTPUT_PATH
    )
    reporter.set_rows(
        rows_in=len(athlete_entries),
        rows_out=len(all_rows),
    )
    reporter.update_details(
        {
            "rows_published": len(
                all_rows
            ),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    print(
        f"rows={len(all_rows)} "
        f"columns={len(fieldnames)} "
        f"teams={len(team_ids)} "
        f"season={season} "
        f"output={OUTPUT_PATH}"
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
                "component": "raw roster",
            },
        ) as reporter:
            run(
                args,
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
