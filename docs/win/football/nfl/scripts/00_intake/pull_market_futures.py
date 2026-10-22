#!/usr/bin/env python3
"""Pull ESPN NFL season-long futures/props betting markets."""

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

FUTURES_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/nfl/seasons/{season}/futures"
)

TEAM_MASTER_PATH = NFL_ROOT / "data/master/team_master.csv"
OUTPUT_ROOT = NFL_ROOT / "data/market_futures"
REPORT_ROOT = NFL_ROOT / "errors"

OUTPUT_HEADER = [
    "season",
    "future_id",
    "future_name",
    "provider_id",
    "provider_name",
    "athlete_id",
    "team_id",
    "value",
]


class MarketFuturesError(RuntimeError):
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
        raise MarketFuturesError(
            f"Failed ESPN request {url}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise MarketFuturesError(
            f"Unexpected ESPN response type for {url}"
        )

    return payload


def metadata_int(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise MarketFuturesError(f"ESPN {field} must be an integer")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MarketFuturesError(
            f"ESPN {field} must be an integer; got {value!r}"
        ) from exc

    if parsed < minimum:
        raise MarketFuturesError(
            f"ESPN {field} must be >= {minimum}; got {parsed}"
        )

    return parsed


def fetch_all_futures(
    base_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = fetch_json(base_url)
    first_items = first.get("items")

    if not isinstance(first_items, list):
        raise MarketFuturesError(
            "ESPN futures response has invalid items"
        )

    page_count = metadata_int(
        first.get("pageCount", 1),
        "pageCount",
        1,
    )

    expected_count = None
    if first.get("count") is not None:
        expected_count = metadata_int(
            first["count"],
            "count",
            0,
        )

    all_items = list(first_items)
    page_item_counts = [len(first_items)]

    for page in range(2, page_count + 1):
        separator = "&" if "?" in base_url else "?"
        payload = fetch_json(
            f"{base_url}{separator}page={page}"
        )

        items = payload.get("items")
        if not isinstance(items, list):
            raise MarketFuturesError(
                f"ESPN futures page {page} has invalid items"
            )

        if payload.get("pageCount") is not None:
            reported_page_count = metadata_int(
                payload["pageCount"],
                f"page {page} pageCount",
                1,
            )
            if reported_page_count != page_count:
                raise MarketFuturesError(
                    f"ESPN futures page {page} reported "
                    f"pageCount={reported_page_count}; "
                    f"expected {page_count}"
                )

        if (
            expected_count is not None
            and payload.get("count") is not None
        ):
            reported_count = metadata_int(
                payload["count"],
                f"page {page} count",
                0,
            )
            if reported_count != expected_count:
                raise MarketFuturesError(
                    f"ESPN futures page {page} reported "
                    f"count={reported_count}; "
                    f"expected {expected_count}"
                )

        all_items.extend(items)
        page_item_counts.append(len(items))

    if (
        expected_count is not None
        and len(all_items) != expected_count
    ):
        raise MarketFuturesError(
            "ESPN futures pagination incomplete: "
            f"received {len(all_items)} items; "
            f"expected count={expected_count}"
        )

    return all_items, {
        "page_count": page_count,
        "page_item_counts": page_item_counts,
        "expected_count": expected_count,
        "source_future_count": len(all_items),
    }


def extract_id(ref_url: Any, segment: str) -> str:
    match = re.search(
        rf"/{re.escape(segment)}/([^/?]+)",
        clean(ref_url),
    )
    return clean(match.group(1)) if match else ""


def load_team_master_ids() -> tuple[set[str], int]:
    if not TEAM_MASTER_PATH.is_file():
        raise MarketFuturesError(
            f"Missing team master: {TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if "team_id" not in set(reader.fieldnames or []):
            raise MarketFuturesError(
                f"{TEAM_MASTER_PATH} missing team_id"
            )

        rows = list(reader)

    if not rows:
        raise MarketFuturesError(
            f"{TEAM_MASTER_PATH} contains no rows"
        )

    team_ids = {
        clean(row.get("team_id"))
        for row in rows
        if clean(row.get("team_id"))
    }

    if len(team_ids) != 32:
        raise MarketFuturesError(
            f"{TEAM_MASTER_PATH} must resolve 32 unique "
            f"team IDs; found {len(team_ids)}"
        )

    return team_ids, len(rows)


def build_rows(
    futures: list[dict[str, Any]],
    season: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen_future_ids: set[str] = set()

    provider_entry_count = 0
    source_book_count = 0
    futures_without_providers = 0
    providers_without_books = 0

    for future_index, future in enumerate(futures, start=1):
        if not isinstance(future, dict):
            raise MarketFuturesError(
                f"Future item {future_index} is not an object"
            )

        future_id = clean(future.get("id"))
        future_name = clean(future.get("name"))

        if not future_id:
            raise MarketFuturesError(
                f"Future item {future_index} has blank id"
            )

        if not future_name:
            raise MarketFuturesError(
                f"Future {future_id!r} has blank name"
            )

        if future_id in seen_future_ids:
            raise MarketFuturesError(
                f"Duplicate future_id returned by ESPN: {future_id}"
            )

        seen_future_ids.add(future_id)

        provider_entries = future.get("futures")
        if not isinstance(provider_entries, list):
            raise MarketFuturesError(
                f"Future {future_id!r} has invalid futures list"
            )

        if not provider_entries:
            futures_without_providers += 1
            continue

        for provider_index, provider_entry in enumerate(
            provider_entries,
            start=1,
        ):
            if not isinstance(provider_entry, dict):
                raise MarketFuturesError(
                    f"Future {future_id!r} provider entry "
                    f"{provider_index} is not an object"
                )

            provider = provider_entry.get("provider")
            if not isinstance(provider, dict):
                raise MarketFuturesError(
                    f"Future {future_id!r} provider entry "
                    f"{provider_index} has invalid provider"
                )

            provider_id = clean(provider.get("id"))
            provider_name = clean(provider.get("name"))

            if not provider_id:
                raise MarketFuturesError(
                    f"Future {future_id!r} provider entry "
                    f"{provider_index} has blank provider_id"
                )

            if not provider_name:
                raise MarketFuturesError(
                    f"Future {future_id!r} provider "
                    f"{provider_id!r} has blank provider_name"
                )

            provider_entry_count += 1

            books = provider_entry.get("books")
            if not isinstance(books, list):
                raise MarketFuturesError(
                    f"Future {future_id!r} provider "
                    f"{provider_id!r} has invalid books"
                )

            if not books:
                providers_without_books += 1
                continue

            source_book_count += len(books)

            for book_index, book in enumerate(books, start=1):
                if not isinstance(book, dict):
                    raise MarketFuturesError(
                        f"Future {future_id!r} provider "
                        f"{provider_id!r} book {book_index} "
                        "is not an object"
                    )

                athlete = book.get("athlete")
                team = book.get("team")

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

                if bool(athlete_id) == bool(team_id):
                    raise MarketFuturesError(
                        f"Future {future_id!r} provider "
                        f"{provider_id!r} book {book_index} "
                        "must resolve exactly one of "
                        "athlete_id or team_id"
                    )

                value = clean(book.get("value"))
                if not value:
                    raise MarketFuturesError(
                        f"Future {future_id!r} provider "
                        f"{provider_id!r} book {book_index} "
                        "has blank value"
                    )

                rows.append(
                    {
                        "season": season,
                        "future_id": future_id,
                        "future_name": future_name,
                        "provider_id": provider_id,
                        "provider_name": provider_name,
                        "athlete_id": athlete_id,
                        "team_id": team_id,
                        "value": value,
                    }
                )

    if not rows:
        raise MarketFuturesError(
            "ESPN futures response produced no usable market rows"
        )

    return rows, {
        "provider_entry_count": provider_entry_count,
        "source_book_count": source_book_count,
        "futures_without_providers": futures_without_providers,
        "providers_without_books": providers_without_books,
    }


def validate_rows(
    rows: list[dict[str, Any]],
    season: int,
    valid_team_ids: set[str],
) -> None:
    if not rows:
        raise MarketFuturesError(
            "Market futures output contains no rows"
        )

    seen_keys: set[
        tuple[str, str, str, str, str]
    ] = set()

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        future_id = clean(row.get("future_id"))
        future_name = clean(row.get("future_name"))
        provider_id = clean(row.get("provider_id"))
        provider_name = clean(row.get("provider_name"))
        athlete_id = clean(row.get("athlete_id"))
        team_id = clean(row.get("team_id"))
        value = clean(row.get("value"))

        if row_season != str(season):
            raise MarketFuturesError(
                f"Market futures row {line_number} has "
                f"season={row_season!r}; expected {season}"
            )

        required = {
            "future_id": future_id,
            "future_name": future_name,
            "provider_id": provider_id,
            "provider_name": provider_name,
            "value": value,
        }

        for field, field_value in required.items():
            if not field_value:
                raise MarketFuturesError(
                    f"Market futures row {line_number} "
                    f"has blank {field}"
                )

        if bool(athlete_id) == bool(team_id):
            raise MarketFuturesError(
                f"Market futures row {line_number} "
                "must contain exactly one of athlete_id "
                "or team_id"
            )

        if team_id and team_id not in valid_team_ids:
            raise MarketFuturesError(
                f"Market futures row {line_number} "
                f"has unknown team_id={team_id!r}"
            )

        key = (
            row_season,
            future_id,
            provider_id,
            athlete_id,
            team_id,
        )

        if key in seen_keys:
            raise MarketFuturesError(
                f"Duplicate market-futures key: {key}"
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
        prefix=".market_futures_stage_",
        dir=output_path.parent,
    ) as staging_dir:
        staged_path = Path(staging_dir) / output_path.name

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
            raise MarketFuturesError(
                "Staged market futures output is zero bytes"
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
            raise MarketFuturesError(
                "Staged market futures headers changed"
            )

        if len(staged_rows) != len(rows):
            raise MarketFuturesError(
                "Staged market futures row count changed"
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
    futures_url = FUTURES_URL_TEMPLATE.format(
        season=season,
    )
    output_path = (
        OUTPUT_ROOT
        / f"market_futures_{season}.csv"
    )

    reporter.add_input(TEAM_MASTER_PATH)
    reporter.add_input(futures_url)
    reporter.update_details(
        {
            "season": season,
            "futures_url": futures_url,
            "output_path": str(output_path),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    valid_team_ids, team_master_rows = (
        load_team_master_ids()
    )
    futures, pagination = fetch_all_futures(
        futures_url
    )
    rows, source_counts = build_rows(
        futures,
        season,
    )

    validate_rows(
        rows,
        season,
        valid_team_ids,
    )

    reporter.set_rows(
        rows_in=source_counts["source_book_count"],
        rows_out=0,
    )
    reporter.update_details(
        {
            "team_master_rows": team_master_rows,
            **pagination,
            **source_counts,
            "rows_validated": len(rows),
        }
    )

    if source_counts["futures_without_providers"]:
        reporter.warning(
            "ESPN returned futures with no provider entries",
            count=source_counts["futures_without_providers"],
        )

    if source_counts["providers_without_books"]:
        reporter.warning(
            "ESPN returned provider entries with no books",
            count=source_counts["providers_without_books"],
        )

    publish(
        output_path,
        rows,
        season,
        valid_team_ids,
    )

    reporter.add_output(output_path)
    reporter.set_rows(
        rows_in=source_counts["source_book_count"],
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
                "component": "market futures",
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
