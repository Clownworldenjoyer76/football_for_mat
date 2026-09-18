#!/usr/bin/env python3
"""Pull ESPN season-level NFL Football Power Index data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

POWERINDEX_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/nfl/seasons/{season}/powerindex"
)

TEAM_MASTER_PATH = NFL_ROOT / "data/master/team_master.csv"
OUTPUT_ROOT = NFL_ROOT / "data/team_power_index"
REPORT_ROOT = NFL_ROOT / "errors"

BASE_FIELDS = [
    "season",
    "team_id",
    "lastUpdated",
]


class PowerIndexError(RuntimeError):
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


def fetch_json(url: str, timeout: int = 15) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise PowerIndexError(
            f"Failed ESPN request {url}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise PowerIndexError(
            f"Unexpected ESPN response type for {url}"
        )

    return payload


def optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None or clean(value) == "":
        return None

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PowerIndexError(
            f"Invalid {field}: {value!r}"
        ) from exc

    if result < 0:
        raise PowerIndexError(
            f"Invalid {field}: {result}"
        )

    return result


def build_page_url(base_url: str, page: int) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    query["page"] = [str(page)]

    return urllib.parse.urlunparse(
        parsed._replace(
            query=urllib.parse.urlencode(
                query,
                doseq=True,
            )
        )
    )


def extract_team_id(ref_url: str) -> str:
    match = re.search(
        r"/teams/([^/?]+)",
        clean(ref_url),
    )

    return clean(match.group(1)) if match else ""


def load_team_master_ids() -> tuple[set[str], int]:
    if not TEAM_MASTER_PATH.is_file():
        raise PowerIndexError(
            f"Missing team master: {TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])

        if "team_id" not in columns:
            raise PowerIndexError(
                f"{TEAM_MASTER_PATH} missing team_id"
            )

        rows = list(reader)

    if not rows:
        raise PowerIndexError(
            f"{TEAM_MASTER_PATH} contains no rows"
        )

    team_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        team_id = clean(row.get("team_id"))

        if not team_id:
            raise PowerIndexError(
                f"{TEAM_MASTER_PATH} line {line_number} "
                "has blank team_id"
            )

        team_ids.add(team_id)

    if len(team_ids) != 32:
        raise PowerIndexError(
            f"{TEAM_MASTER_PATH} must resolve 32 unique team IDs; "
            f"found {len(team_ids)}"
        )

    return team_ids, len(rows)


def get_all_items(
    base_url: str,
) -> tuple[list[dict[str, Any]], int, int, int | None]:
    first_page = fetch_json(base_url)

    first_items = first_page.get("items")

    if not isinstance(first_items, list):
        raise PowerIndexError(
            "Power index page 1 has invalid items"
        )

    raw_page_count = optional_nonnegative_int(
        first_page.get("pageCount", 1),
        "pageCount",
    )

    page_count = max(
        1,
        raw_page_count if raw_page_count is not None else 1,
    )

    expected_count = optional_nonnegative_int(
        first_page.get("count"),
        "count",
    )

    items: list[dict[str, Any]] = []

    for index, item in enumerate(first_items):
        if not isinstance(item, dict):
            raise PowerIndexError(
                f"Power index page 1 item {index} is not an object"
            )

        items.append(item)

    pages_fetched = 1

    for page in range(2, page_count + 1):
        page_data = fetch_json(
            build_page_url(
                base_url,
                page,
            )
        )

        page_items = page_data.get("items")

        if not isinstance(page_items, list):
            raise PowerIndexError(
                f"Power index page {page} has invalid items"
            )

        for index, item in enumerate(page_items):
            if not isinstance(item, dict):
                raise PowerIndexError(
                    f"Power index page {page} item {index} "
                    "is not an object"
                )

            items.append(item)

        pages_fetched += 1

    if expected_count is not None and len(items) != expected_count:
        raise PowerIndexError(
            "Incomplete power index collection: "
            f"expected={expected_count} fetched={len(items)}"
        )

    if not items:
        raise PowerIndexError(
            "No team power index data was returned"
        )

    return (
        items,
        page_count,
        pages_fetched,
        expected_count,
    )


def build_rows(
    items: list[dict[str, Any]],
    season: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    fieldnames = list(BASE_FIELDS)
    seen_fieldnames = set(fieldnames)
    seen_team_ids: set[str] = set()

    for index, item in enumerate(items):
        team = item.get("team")

        team_ref = (
            team.get("$ref")
            if isinstance(team, dict)
            else ""
        )

        team_id = extract_team_id(team_ref)

        if not team_id:
            raise PowerIndexError(
                f"Power index item {index} has no usable team ID"
            )

        if team_id in seen_team_ids:
            raise PowerIndexError(
                f"Duplicate ESPN power index team_id={team_id}"
            )

        item_season = clean(item.get("season"))

        if item_season != str(season):
            raise PowerIndexError(
                f"Power index item {index} has season={item_season!r}; "
                f"expected {season}"
            )

        predictives = item.get("predictives")

        if not isinstance(predictives, list):
            raise PowerIndexError(
                f"Power index item {index} has invalid predictives"
            )

        row: dict[str, Any] = {
            "season": season,
            "team_id": team_id,
            "lastUpdated": clean(item.get("lastUpdated")),
        }

        for stat_index, stat in enumerate(predictives):
            if not isinstance(stat, dict):
                raise PowerIndexError(
                    f"Power index item {index} predictive "
                    f"{stat_index} is not an object"
                )

            name = clean(stat.get("name"))

            if not name:
                raise PowerIndexError(
                    f"Power index item {index} contains "
                    "a blank predictive name"
                )

            row[name] = clean(stat.get("value"))

            if name not in seen_fieldnames:
                fieldnames.append(name)
                seen_fieldnames.add(name)

        rows.append(row)
        seen_team_ids.add(team_id)

    if len(fieldnames) <= len(BASE_FIELDS):
        raise PowerIndexError(
            "Power index collection contains no predictive columns"
        )

    return rows, fieldnames


def validate_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    season: int,
    expected_team_ids: set[str],
) -> None:
    if fieldnames[:3] != BASE_FIELDS:
        raise PowerIndexError(
            "Power index base headers are invalid"
        )

    if len(fieldnames) <= len(BASE_FIELDS):
        raise PowerIndexError(
            "Power index has no predictive columns"
        )

    if len(rows) != 32:
        raise PowerIndexError(
            f"Expected exactly 32 NFL teams; found {len(rows)}"
        )

    team_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        team_id = clean(row.get("team_id"))

        if row_season != str(season):
            raise PowerIndexError(
                f"Power index row {line_number} "
                f"has season={row_season!r}"
            )

        if not team_id:
            raise PowerIndexError(
                f"Power index row {line_number} has blank team_id"
            )

        if team_id in team_ids:
            raise PowerIndexError(
                f"Power index contains duplicate team_id={team_id}"
            )

        team_ids.add(team_id)

    if team_ids != expected_team_ids:
        raise PowerIndexError(
            "Power index team IDs do not exactly match team_master: "
            f"missing={sorted(expected_team_ids - team_ids)} "
            f"unexpected={sorted(team_ids - expected_team_ids)}"
        )


def sort_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            0,
            int(clean(row.get("team_id"))),
        )
        if clean(row.get("team_id")).isdigit()
        else (
            1,
            clean(row.get("team_id")),
        ),
    )


def publish(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    season: int,
    expected_team_ids: set[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".team_power_index_stage_",
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
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field: clean(row.get(field))
                        for field in fieldnames
                    }
                )

            handle.flush()
            os.fsync(handle.fileno())

        if staged_path.stat().st_size == 0:
            raise PowerIndexError(
                "Staged power index output is zero bytes"
            )

        with staged_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            staged_fields = reader.fieldnames or []
            staged_rows = list(reader)

        if staged_fields != fieldnames:
            raise PowerIndexError(
                "Staged power index headers changed"
            )

        if len(staged_rows) != len(rows):
            raise PowerIndexError(
                "Staged power index row count changed"
            )

        validate_rows(
            staged_rows,
            staged_fields,
            season,
            expected_team_ids,
        )

        os.replace(
            staged_path,
            output_path,
        )


def run(
    season: int,
    reporter: PipelineReporter,
) -> None:
    powerindex_url = POWERINDEX_URL_TEMPLATE.format(
        season=season
    )

    output_path = (
        OUTPUT_ROOT
        / f"team_power_index_{season}.csv"
    )

    reporter.add_input(
        TEAM_MASTER_PATH
    )
    reporter.add_input(
        powerindex_url
    )

    reporter.update_details(
        {
            "season": season,
            "powerindex_url": powerindex_url,
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    expected_team_ids, team_master_rows = (
        load_team_master_ids()
    )

    (
        items,
        pages_expected,
        pages_fetched,
        items_expected,
    ) = get_all_items(
        powerindex_url
    )

    reporter.set_rows(
        rows_in=len(items),
        rows_out=0,
    )

    reporter.update_details(
        {
            "team_master_rows": team_master_rows,
            "pages_expected": pages_expected,
            "pages_fetched": pages_fetched,
            "items_expected": items_expected,
            "items_fetched": len(items),
        }
    )

    rows, fieldnames = build_rows(
        items,
        season,
    )

    rows = sort_rows(rows)

    validate_rows(
        rows,
        fieldnames,
        season,
        expected_team_ids,
    )

    reporter.update_details(
        {
            "unique_teams": len(rows),
            "predictive_columns_observed": fieldnames[3:],
            "output_path": str(output_path),
        }
    )

    publish(
        output_path,
        rows,
        fieldnames,
        season,
        expected_team_ids,
    )

    reporter.add_output(
        output_path
    )

    reporter.set_rows(
        rows_in=len(items),
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
                "component": "team power index",
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
