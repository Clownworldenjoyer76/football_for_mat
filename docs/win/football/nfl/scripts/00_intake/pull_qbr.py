#!/usr/bin/env python3
"""Pull ESPN QBR data for ESPN's active NFL week."""

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

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
QBR_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/types/{season_type}/weeks/{week}/qbr/0"
)

OUTPUT_ROOT = NFL_ROOT / "data/qb_data/qbr_data"
REPORT_ROOT = NFL_ROOT / "errors"
BASE_FIELDS = ["season", "week", "athlete_id", "team_id"]


class QBRError(RuntimeError):
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
        raise QBRError(
            f"Failed ESPN request {url}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise QBRError(f"Unexpected ESPN response type for {url}")

    return payload


def positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise QBRError(f"Invalid {field}: {value!r}") from exc

    if result < 1:
        raise QBRError(f"Invalid {field}: {result}")

    return result


def optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None or clean(value) == "":
        return None

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise QBRError(f"Invalid {field}: {value!r}") from exc

    if result < 0:
        raise QBRError(f"Invalid {field}: {result}")

    return result


def get_current_week(expected_season: int) -> tuple[int, int, int]:
    data = fetch_json(SCOREBOARD_URL)
    season = data.get("season")
    week = data.get("week")

    if not isinstance(season, dict) or not isinstance(week, dict):
        raise QBRError("ESPN scoreboard is missing season/week data")

    detected_season = positive_int(season.get("year"), "season.year")
    season_type = positive_int(season.get("type"), "season.type")
    week_number = positive_int(week.get("number"), "week.number")

    if detected_season != expected_season:
        raise QBRError(
            f"Configured season={expected_season} does not match "
            f"ESPN season={detected_season}"
        )

    if season_type not in {1, 2, 3}:
        raise QBRError(f"Unsupported ESPN season type: {season_type}")

    return detected_season, season_type, week_number


def page_url(url: str, page: int) -> str:
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query["page"] = str(page)

    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query),
            parts.fragment,
        )
    )


def get_qbr_items(
    url: str,
) -> tuple[list[dict[str, Any]], int, int, int | None]:
    first = fetch_json(url)
    first_items = first.get("items")

    if not isinstance(first_items, list):
        raise QBRError("QBR response has invalid items")

    raw_page_count = optional_nonnegative_int(
        first.get("pageCount", 1),
        "pageCount",
    )
    page_count = max(1, raw_page_count if raw_page_count is not None else 1)
    expected_count = optional_nonnegative_int(first.get("count"), "count")

    items: list[dict[str, Any]] = []

    for item in first_items:
        if not isinstance(item, dict):
            raise QBRError("QBR response contains a non-object item")
        items.append(item)

    pages_fetched = 1

    for page in range(2, page_count + 1):
        payload = fetch_json(page_url(url, page))
        page_items = payload.get("items")

        if not isinstance(page_items, list):
            raise QBRError(f"QBR page {page} has invalid items")

        for item in page_items:
            if not isinstance(item, dict):
                raise QBRError(f"QBR page {page} contains a non-object item")
            items.append(item)

        pages_fetched += 1

    if expected_count is not None and len(items) != expected_count:
        raise QBRError(
            f"Incomplete QBR collection: expected={expected_count} "
            f"fetched={len(items)}"
        )

    return items, page_count, pages_fetched, expected_count


def extract_id(ref_url: str, segment: str) -> str:
    match = re.search(
        rf"/{re.escape(segment)}/([^/?]+)",
        clean(ref_url),
    )
    return clean(match.group(1)) if match else ""


def build_rows(
    items: list[dict[str, Any]],
    season: int,
    week: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    fields = list(BASE_FIELDS)
    seen_fields = set(fields)

    for index, item in enumerate(items):
        athlete = item.get("athlete")
        team = item.get("team")

        athlete_ref = athlete.get("$ref") if isinstance(athlete, dict) else ""
        team_ref = team.get("$ref") if isinstance(team, dict) else ""

        athlete_id = extract_id(athlete_ref, "athletes")
        team_id = extract_id(team_ref, "teams")

        if not athlete_id or not team_id:
            raise QBRError(
                f"QBR item {index} has blank athlete_id/team_id"
            )

        row: dict[str, Any] = {
            "season": season,
            "week": week,
            "athlete_id": athlete_id,
            "team_id": team_id,
        }

        splits = item.get("splits")
        categories = (
            splits.get("categories")
            if isinstance(splits, dict)
            else None
        )

        if not isinstance(categories, list):
            raise QBRError(f"QBR item {index} has invalid categories")

        for category in categories:
            if not isinstance(category, dict):
                raise QBRError(f"QBR item {index} has invalid category")

            stats = category.get("stats")

            if not isinstance(stats, list):
                raise QBRError(f"QBR item {index} has invalid stats")

            for stat in stats:
                if not isinstance(stat, dict):
                    raise QBRError(f"QBR item {index} has invalid stat")

                name = clean(stat.get("name"))

                if not name:
                    raise QBRError(f"QBR item {index} has blank stat name")

                row[name] = clean(stat.get("value"))

                if name not in seen_fields:
                    fields.append(name)
                    seen_fields.add(name)

        rows.append(row)

    return rows, fields


def validate_rows(
    rows: list[dict[str, Any]],
    fields: list[str],
    season: int,
    week: int,
) -> None:
    if not rows:
        return

    if fields[:4] != BASE_FIELDS or len(fields) <= 4:
        raise QBRError("QBR output contains no valid statistic columns")

    seen: set[tuple[str, str, str, str]] = set()

    for line, row in enumerate(rows, start=2):
        values = (
            clean(row.get("season")),
            clean(row.get("week")),
            clean(row.get("athlete_id")),
            clean(row.get("team_id")),
        )

        if values[0] != str(season) or values[1] != str(week):
            raise QBRError(f"QBR row {line} has wrong season/week")

        if not values[2] or not values[3]:
            raise QBRError(f"QBR row {line} has blank IDs")

        if values in seen:
            raise QBRError(f"Duplicate QBR key: {values}")

        seen.add(values)


def get_output_path(season: int, season_type: int, week: int) -> Path:
    folder = OUTPUT_ROOT / str(season)

    if season_type == 2:
        filename = f"qbr_week{week}.csv"
    elif season_type == 3:
        filename = f"qbr_playoffs_week{week}.csv"
    else:
        filename = f"qbr_type{season_type}_week{week}.csv"

    return folder / filename


def publish(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
    season: int,
    week: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".qbr_stage_",
        dir=path.parent,
    ) as staging_dir:
        staged = Path(staging_dir) / path.name

        with staged.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        with staged.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)

            if (reader.fieldnames or []) != fields:
                raise QBRError("Staged QBR headers changed")

            staged_rows = list(reader)

        if len(staged_rows) != len(rows):
            raise QBRError("Staged QBR row count changed")

        validate_rows(staged_rows, fields, season, week)
        os.replace(staged, path)


def run(expected_season: int, reporter: PipelineReporter) -> None:
    reporter.add_input(SCOREBOARD_URL)
    reporter.update_details(
        {
            "configured_season": expected_season,
            "publication_completed": False,
            "staged_roundtrip_verified": False,
            "empty_data": False,
        }
    )

    season, season_type, week = get_current_week(expected_season)

    qbr_url = QBR_URL_TEMPLATE.format(
        season=season,
        season_type=season_type,
        week=week,
    )

    reporter.add_input(qbr_url)

    items, pages_expected, pages_fetched, expected_count = (
        get_qbr_items(qbr_url)
    )

    reporter.set_rows(rows_in=len(items), rows_out=0)
    reporter.update_details(
        {
            "detected_season": season,
            "detected_season_type": season_type,
            "detected_week": week,
            "qbr_url": qbr_url,
            "pages_expected": pages_expected,
            "pages_fetched": pages_fetched,
            "items_expected": expected_count,
            "items_fetched": len(items),
        }
    )

    if not items:
        reporter.warning(
            "ESPN QBR collection returned no items; existing output unchanged",
            season=season,
            season_type=season_type,
            week=week,
        )
        reporter.set_detail("empty_data", True)
        return

    rows, fields = build_rows(items, season, week)
    validate_rows(rows, fields, season, week)

    output_path = get_output_path(season, season_type, week)

    reporter.set_detail("output_path", str(output_path))
    reporter.set_detail("stat_columns_observed", fields[4:])

    publish(output_path, rows, fields, season, week)

    reporter.add_output(output_path)
    reporter.set_rows(rows_in=len(items), rows_out=len(rows))
    reporter.update_details(
        {
            "staged_roundtrip_verified": True,
            "rows_published": len(rows),
            "publication_completed": True,
        }
    )

    print(f"rows={len(rows)} output={output_path}")


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
            extra_context={"component": "ESPN QBR"},
        ) as reporter:
            run(args.season, reporter)

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
