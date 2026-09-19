#!/usr/bin/env python3
"""Build cumulative NFL opening-odds and market-movement history."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

WEEKLY_DIR = NFL_ROOT / "00_intake" / "schedule" / "weekly"
OPENERS_DIR = NFL_ROOT / "00_intake" / "odds" / "openers"
SNAPSHOT_DIR = NFL_ROOT / "00_intake" / "odds" / "snapshots"
REPORT_ROOT = NFL_ROOT / "errors"
LOG_FILE = REPORT_ROOT / "00_intake" / "pull_opening_odds.txt"

ESPN_CORE_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
)

HTTP_RETRIES = 4
HTTP_TIMEOUT = 45

OUTPUT_COLUMNS = [
    "game_id",
    "odds_provider_game_id",
    "market_type",
    "bet_side",
    "opening_line",
    "opening_odds_american",
    "opening_timestamp",
    "bookmaker",
    "opening_spread",
    "current_spread",
    "spread_movement",
    "opening_total",
    "current_total",
    "total_movement",
    "opening_moneyline",
    "current_moneyline",
    "moneyline_movement",
    "opener_status",
    "opener_missing_reason",
    "opener_http_status",
]

WEEKLY_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "odds_provider_game_id",
    "away_team",
    "home_team",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "total",
    "odds_available",
]

SNAPSHOT_REQUIRED_COLUMNS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "bookmaker",
]

EXPECTED_MARKET_SIDES = {
    ("h2h", "home"),
    ("h2h", "away"),
    ("spreads", "home"),
    ("spreads", "away"),
    ("totals", "over"),
    ("totals", "under"),
}


class OpeningOddsError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] {message}\n")


def fail(message: str) -> None:
    log(f"ERROR: {message}")
    raise OpeningOddsError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    args = parser.parse_args()

    if not 2000 <= args.season <= 2100:
        parser.error("--season must be between 2000 and 2100")
    if not 1 <= args.week <= 25:
        parser.error("--week must be between 1 and 25")

    return args


def parse_iso_timestamp(value: Any, label: str) -> datetime:
    text = clean(value)
    if not text:
        fail(f"{label} is blank")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        fail(f"{label} is not a valid ISO timestamp: {value!r}")

    if parsed.tzinfo is None:
        fail(f"{label} has no timezone: {value!r}")

    return parsed


def read_csv(
    path: Path,
    required_columns: list[str],
    label: str,
    *,
    allow_empty: bool = False,
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

    missing = [
        column
        for column in required_columns
        if column not in fieldnames
    ]
    if missing:
        fail(f"{label} missing columns: {missing}")

    if exact_columns is not None and fieldnames != exact_columns:
        fail(
            f"{label} has unexpected headers: "
            f"{fieldnames}; expected {exact_columns}"
        )

    if not rows and not allow_empty:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows


def to_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def clean_number(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number)


def clean_american(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return str(int(round(number)))


def numeric_movement(
    current_value: Any,
    opening_value: Any,
) -> str:
    current = to_float(current_value)
    opening = to_float(opening_value)

    if current is None or opening is None:
        return ""

    movement = current - opening
    if movement.is_integer():
        return str(int(movement))
    return str(round(movement, 4))


def normalize_bookmaker(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def validate_weekly_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    week: int,
) -> None:
    seen: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        row_season = clean(row.get("season"))
        row_week = clean(row.get("week"))
        odds_available = clean(row.get("odds_available"))

        if row_season != str(season):
            fail(
                f"Weekly row {line_number} has "
                f"season={row_season!r}; expected {season}"
            )
        if row_week != str(week):
            fail(
                f"Weekly row {line_number} has "
                f"week={row_week!r}; expected {week}"
            )
        if not game_id:
            fail(f"Weekly row {line_number} has blank game_id")
        if game_id in seen:
            fail(f"Duplicate weekly game_id: {game_id}")
        seen.add(game_id)

        if not clean(row.get("away_team")) or not clean(row.get("home_team")):
            fail(f"game_id={game_id} has blank away/home team")

        if odds_available not in {"0", "1"}:
            fail(
                f"game_id={game_id} has invalid "
                f"odds_available={odds_available!r}"
            )

        if odds_available == "1":
            if not clean(row.get("odds_provider_game_id")):
                fail(
                    f"game_id={game_id} has odds_available=1 "
                    "but blank odds_provider_game_id"
                )

            for field in (
                "home_moneyline_american",
                "away_moneyline_american",
                "home_spread",
                "away_spread",
                "total",
            ):
                value = clean(row.get(field))
                if value and to_float(value) is None:
                    fail(
                        f"game_id={game_id} has nonnumeric "
                        f"{field}={value!r}"
                    )


def load_snapshot_index(
    reporter: PipelineReporter,
) -> tuple[dict[tuple[str, str], str], int, int]:
    reporter.add_input(SNAPSHOT_DIR)

    paths = sorted(SNAPSHOT_DIR.glob("*_NFL_odds.csv"))
    if not paths:
        fail(f"No odds snapshots found in {SNAPSHOT_DIR}")

    earliest: dict[tuple[str, str], tuple[datetime, str]] = {}
    total_rows = 0

    for path in paths:
        _, rows = read_csv(
            path,
            SNAPSHOT_REQUIRED_COLUMNS,
            "odds snapshot CSV",
        )

        for line_number, row in enumerate(rows, start=2):
            snapshot_id = clean(row.get("snapshot_id"))
            stamp = clean(row.get("snapshot_fetched_at"))
            game_id = clean(row.get("game_id"))
            bookmaker = clean(row.get("bookmaker"))

            if not snapshot_id:
                fail(
                    f"{path} line {line_number} has "
                    "blank snapshot_id"
                )
            if not game_id:
                fail(
                    f"{path} line {line_number} has blank game_id"
                )
            if not bookmaker:
                fail(
                    f"{path} line {line_number} has blank bookmaker"
                )

            parsed = parse_iso_timestamp(
                stamp,
                f"{path} line {line_number} snapshot_fetched_at",
            )

            key = (game_id, normalize_bookmaker(bookmaker))
            previous = earliest.get(key)
            if previous is None or parsed < previous[0]:
                earliest[key] = (parsed, stamp)

            total_rows += 1

    return (
        {key: value[1] for key, value in earliest.items()},
        len(paths),
        total_rows,
    )


def odds_url(event_id: str) -> str:
    return (
        f"{ESPN_CORE_BASE}/events/{event_id}"
        f"/competitions/{event_id}/odds"
        "?lang=en&region=us"
    )


def http_get_json(
    url: str,
) -> tuple[int | None, object, str]:
    last_error = ""

    for attempt in range(1, HTTP_RETRIES + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "nfl-pull-opening-odds-espn/1.0",
            },
        )

        try:
            with urlopen(
                request,
                timeout=HTTP_TIMEOUT,
            ) as response:
                body = response.read().decode("utf-8")
                try:
                    return response.status, json.loads(body), ""
                except Exception as exc:
                    return (
                        response.status,
                        {},
                        f"JSON parse failed: {exc}",
                    )

        except HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            last_error = body or str(exc)

            if exc.code not in {
                408,
                425,
                429,
                500,
                502,
                503,
                504,
            }:
                return exc.code, {}, last_error

        except URLError as exc:
            last_error = str(exc.reason)

        except Exception as exc:
            last_error = str(exc)

        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))

    return None, {}, last_error or "request failed"


def nested(data: object, *keys: str) -> object:
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def display_value(value: object) -> object:
    if isinstance(value, dict):
        for key in (
            "american",
            "alternateDisplayValue",
            "value",
        ):
            if value.get(key) not in (None, ""):
                return value.get(key)

    return value


def select_bookmaker_item(
    payload: object,
    bookmaker: str,
) -> tuple[dict | None, bool, str]:
    if not isinstance(payload, dict):
        return None, False, ""

    items = payload.get("items")
    if not isinstance(items, list):
        return None, False, ""

    valid = [
        item
        for item in items
        if isinstance(item, dict)
    ]

    for item in valid:
        provider = item.get("provider")
        if (
            isinstance(provider, dict)
            and clean(provider.get("name")) == bookmaker
        ):
            return item, False, bookmaker

    if not valid:
        return None, False, ""

    fallback = valid[0]
    provider = (
        fallback.get("provider")
        if isinstance(fallback.get("provider"), dict)
        else {}
    )
    resolved = clean(provider.get("name")) or bookmaker

    return fallback, True, resolved


def status_fields(
    status: str,
    reason: str = "",
    http_status: object = "",
) -> dict[str, str]:
    return {
        "opener_status": status,
        "opener_missing_reason": reason,
        "opener_http_status": clean(http_status),
    }


def base_row(
    weekly_row: dict[str, str],
    market_type: str,
    bet_side: str,
    opening_timestamp: str,
) -> dict[str, str]:
    return {
        "game_id": clean(weekly_row.get("game_id")),
        "odds_provider_game_id": clean(
            weekly_row.get("odds_provider_game_id")
        ),
        "market_type": market_type,
        "bet_side": bet_side,
        "opening_line": "",
        "opening_odds_american": "",
        "opening_timestamp": opening_timestamp,
        "bookmaker": clean(weekly_row.get("bookmaker")),
        "opening_spread": "",
        "current_spread": "",
        "spread_movement": "",
        "opening_total": "",
        "current_total": "",
        "total_movement": "",
        "opening_moneyline": "",
        "current_moneyline": "",
        "moneyline_movement": "",
        "opener_status": "",
        "opener_missing_reason": "",
        "opener_http_status": "",
    }


def empty_market_rows(
    weekly_row: dict[str, str],
    status: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for market_type, side in sorted(EXPECTED_MARKET_SIDES):
        row = base_row(
            weekly_row,
            market_type,
            side,
            "",
        )
        row.update(status)
        rows.append(row)

    return rows


def build_rows_for_weekly_row(
    weekly_row: dict[str, str],
    snapshot_index: dict[tuple[str, str], str],
) -> tuple[list[dict[str, str]], bool, str, str]:
    event_id = clean(
        weekly_row.get("odds_provider_game_id")
    )
    requested_bookmaker = (
        clean(weekly_row.get("bookmaker"))
        or "DraftKings"
    )

    if not event_id:
        fail(
            f"game_id={weekly_row.get('game_id')} "
            "cannot fetch openers without provider event ID"
        )

    http_status, payload, request_error = http_get_json(
        odds_url(event_id)
    )

    if request_error:
        rows = empty_market_rows(
            weekly_row,
            status_fields(
                "error",
                request_error,
                http_status,
            ),
        )
        return rows, False, requested_bookmaker, request_error

    item, used_fallback, resolved_bookmaker = (
        select_bookmaker_item(
            payload,
            requested_bookmaker,
        )
    )

    if not item:
        rows = empty_market_rows(
            weekly_row,
            status_fields(
                "missing",
                "no_espn_odds_item",
                http_status,
            ),
        )
        return (
            rows,
            False,
            requested_bookmaker,
            "no_espn_odds_item",
        )

    working = dict(weekly_row)
    working["bookmaker"] = resolved_bookmaker

    opening_timestamp = snapshot_index.get(
        (
            event_id,
            normalize_bookmaker(resolved_bookmaker),
        ),
        "",
    )

    home_open_ml = clean_american(
        nested(
            item,
            "homeTeamOdds",
            "open",
            "moneyLine",
            "american",
        )
    )
    away_open_ml = clean_american(
        nested(
            item,
            "awayTeamOdds",
            "open",
            "moneyLine",
            "american",
        )
    )

    home_current_ml = (
        clean_american(
            nested(
                item,
                "homeTeamOdds",
                "current",
                "moneyLine",
                "american",
            )
        )
        or clean(weekly_row.get("home_moneyline_american"))
    )
    away_current_ml = (
        clean_american(
            nested(
                item,
                "awayTeamOdds",
                "current",
                "moneyLine",
                "american",
            )
        )
        or clean(weekly_row.get("away_moneyline_american"))
    )

    home_open_spread = clean_number(
        display_value(
            nested(
                item,
                "homeTeamOdds",
                "open",
                "pointSpread",
            )
        )
    )
    away_open_spread = clean_number(
        display_value(
            nested(
                item,
                "awayTeamOdds",
                "open",
                "pointSpread",
            )
        )
    )

    home_current_spread = (
        clean_number(
            display_value(
                nested(
                    item,
                    "homeTeamOdds",
                    "current",
                    "pointSpread",
                )
            )
        )
        or clean(weekly_row.get("home_spread"))
    )
    away_current_spread = (
        clean_number(
            display_value(
                nested(
                    item,
                    "awayTeamOdds",
                    "current",
                    "pointSpread",
                )
            )
        )
        or clean(weekly_row.get("away_spread"))
    )

    home_open_spread_odds = clean_american(
        nested(
            item,
            "homeTeamOdds",
            "open",
            "spread",
            "american",
        )
    )
    away_open_spread_odds = clean_american(
        nested(
            item,
            "awayTeamOdds",
            "open",
            "spread",
            "american",
        )
    )

    opening_total = clean_number(
        display_value(
            nested(
                item,
                "open",
                "total",
            )
        )
    )
    current_total = (
        clean_number(
            display_value(
                nested(
                    item,
                    "current",
                    "total",
                )
            )
        )
        or clean(weekly_row.get("total"))
    )

    over_open_odds = clean_american(
        nested(
            item,
            "open",
            "over",
            "american",
        )
    )
    under_open_odds = clean_american(
        nested(
            item,
            "open",
            "under",
            "american",
        )
    )

    rows: list[dict[str, str]] = []

    for side, opening_ml, current_ml in (
        ("home", home_open_ml, home_current_ml),
        ("away", away_open_ml, away_current_ml),
    ):
        row = base_row(
            working,
            "h2h",
            side,
            opening_timestamp,
        )
        row.update(
            {
                "opening_odds_american": opening_ml,
                "opening_moneyline": opening_ml,
                "current_moneyline": current_ml,
                "moneyline_movement": numeric_movement(
                    current_ml,
                    opening_ml,
                ),
            }
        )
        row.update(
            status_fields(
                "ok" if opening_ml else "missing",
                "" if opening_ml else "no_embedded_open_moneyline",
            )
        )
        rows.append(row)

    for side, opening_spread, current_spread, opening_price in (
        (
            "home",
            home_open_spread,
            home_current_spread,
            home_open_spread_odds,
        ),
        (
            "away",
            away_open_spread,
            away_current_spread,
            away_open_spread_odds,
        ),
    ):
        row = base_row(
            working,
            "spreads",
            side,
            opening_timestamp,
        )
        row.update(
            {
                "opening_line": opening_spread,
                "opening_odds_american": opening_price,
                "opening_spread": opening_spread,
                "current_spread": current_spread,
                "spread_movement": numeric_movement(
                    current_spread,
                    opening_spread,
                ),
            }
        )
        row.update(
            status_fields(
                "ok" if opening_spread else "missing",
                "" if opening_spread else "no_embedded_open_spread",
            )
        )
        rows.append(row)

    for side, opening_price in (
        ("over", over_open_odds),
        ("under", under_open_odds),
    ):
        row = base_row(
            working,
            "totals",
            side,
            opening_timestamp,
        )
        row.update(
            {
                "opening_line": opening_total,
                "opening_odds_american": opening_price,
                "opening_total": opening_total,
                "current_total": current_total,
                "total_movement": numeric_movement(
                    current_total,
                    opening_total,
                ),
            }
        )
        row.update(
            status_fields(
                "ok" if opening_total else "missing",
                "" if opening_total else "no_embedded_open_total",
            )
        )
        rows.append(row)

    return (
        rows,
        used_fallback,
        resolved_bookmaker,
        "",
    )


def row_key(
    row: dict[str, str],
) -> tuple[str, str, str, str]:
    return (
        clean(row.get("game_id")),
        clean(row.get("market_type")),
        clean(row.get("bet_side")),
        clean(row.get("bookmaker")),
    )


def game_groups(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            clean(row.get("game_id")),
            [],
        ).append(row)
    return grouped


def expected_movement_fields(
    market_type: str,
) -> tuple[str, str, str]:
    if market_type == "h2h":
        return (
            "current_moneyline",
            "opening_moneyline",
            "moneyline_movement",
        )
    if market_type == "spreads":
        return (
            "current_spread",
            "opening_spread",
            "spread_movement",
        )
    if market_type == "totals":
        return (
            "current_total",
            "opening_total",
            "total_movement",
        )
    fail(f"Unsupported market_type={market_type!r}")


def validate_opener_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
    allow_empty: bool = True,
) -> None:
    if not rows:
        if allow_empty:
            return
        fail(f"{label} contains no rows")

    seen_keys: set[tuple[str, str, str, str]] = set()
    grouped_pairs: dict[str, set[tuple[str, str]]] = {}
    grouped_bookmakers: dict[str, set[str]] = {}
    grouped_provider_ids: dict[str, set[str]] = {}

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        provider_id = clean(row.get("odds_provider_game_id"))
        market_type = clean(row.get("market_type"))
        bet_side = clean(row.get("bet_side"))
        bookmaker = clean(row.get("bookmaker"))
        status = clean(row.get("opener_status"))
        reason = clean(row.get("opener_missing_reason"))

        if not game_id:
            fail(f"{label} row {line_number} has blank game_id")
        if not provider_id:
            fail(
                f"{label} game_id={game_id} has blank "
                "odds_provider_game_id"
            )
        if not bookmaker:
            fail(f"{label} game_id={game_id} has blank bookmaker")

        pair = (market_type, bet_side)
        if pair not in EXPECTED_MARKET_SIDES:
            fail(
                f"{label} game_id={game_id} has invalid "
                f"market/side={pair}"
            )

        key = row_key(row)
        if key in seen_keys:
            fail(f"{label} contains duplicate key={key}")
        seen_keys.add(key)

        if status not in {"ok", "missing", "error"}:
            fail(
                f"{label} game_id={game_id} has invalid "
                f"opener_status={status!r}"
            )
        if status == "ok" and reason:
            fail(
                f"{label} game_id={game_id} status=ok "
                "has a missing reason"
            )
        if status in {"missing", "error"} and not reason:
            fail(
                f"{label} game_id={game_id} status={status} "
                "has blank missing reason"
            )

        opening_timestamp = clean(row.get("opening_timestamp"))
        if opening_timestamp:
            parse_iso_timestamp(
                opening_timestamp,
                f"{label} game_id={game_id} opening_timestamp",
            )

        numeric_fields = (
            "opening_line",
            "opening_odds_american",
            "opening_spread",
            "current_spread",
            "spread_movement",
            "opening_total",
            "current_total",
            "total_movement",
            "opening_moneyline",
            "current_moneyline",
            "moneyline_movement",
        )
        for field in numeric_fields:
            value = clean(row.get(field))
            if value and to_float(value) is None:
                fail(
                    f"{label} game_id={game_id} has "
                    f"nonnumeric {field}={value!r}"
                )

        if status == "ok":
            if market_type == "h2h":
                if not clean(row.get("opening_moneyline")):
                    fail(
                        f"{label} game_id={game_id} ok h2h row "
                        "has blank opening_moneyline"
                    )
            elif market_type == "spreads":
                if not clean(row.get("opening_spread")):
                    fail(
                        f"{label} game_id={game_id} ok spread row "
                        "has blank opening_spread"
                    )
                if clean(row.get("opening_line")) != clean(
                    row.get("opening_spread")
                ):
                    fail(
                        f"{label} game_id={game_id} spread "
                        "opening_line/opening_spread mismatch"
                    )
            elif market_type == "totals":
                if not clean(row.get("opening_total")):
                    fail(
                        f"{label} game_id={game_id} ok total row "
                        "has blank opening_total"
                    )
                if clean(row.get("opening_line")) != clean(
                    row.get("opening_total")
                ):
                    fail(
                        f"{label} game_id={game_id} total "
                        "opening_line/opening_total mismatch"
                    )

        current_field, opening_field, movement_field = (
            expected_movement_fields(market_type)
        )
        current = clean(row.get(current_field))
        opening = clean(row.get(opening_field))
        movement = clean(row.get(movement_field))

        if current and opening:
            expected = numeric_movement(current, opening)
            if not movement:
                fail(
                    f"{label} game_id={game_id} has blank "
                    f"{movement_field} with current/opening values"
                )
            actual_number = to_float(movement)
            expected_number = to_float(expected)
            if (
                actual_number is None
                or expected_number is None
                or abs(actual_number - expected_number) > 1e-9
            ):
                fail(
                    f"{label} game_id={game_id} has incorrect "
                    f"{movement_field}={movement!r}; "
                    f"expected {expected!r}"
                )
        elif movement:
            fail(
                f"{label} game_id={game_id} has {movement_field} "
                "without both current/opening values"
            )

        grouped_pairs.setdefault(game_id, set()).add(pair)
        grouped_bookmakers.setdefault(game_id, set()).add(bookmaker)
        grouped_provider_ids.setdefault(game_id, set()).add(provider_id)

    for game_id, pairs in grouped_pairs.items():
        if pairs != EXPECTED_MARKET_SIDES:
            fail(
                f"{label} game_id={game_id} does not have "
                "exactly the expected six market/side rows"
            )
        if len(grouped_bookmakers[game_id]) != 1:
            fail(
                f"{label} game_id={game_id} has multiple "
                f"bookmakers={sorted(grouped_bookmakers[game_id])}"
            )
        if len(grouped_provider_ids[game_id]) != 1:
            fail(
                f"{label} game_id={game_id} has multiple "
                "provider event IDs"
            )


def read_existing_openers(
    path: Path,
    reporter: PipelineReporter,
) -> list[dict[str, str]]:
    if not path.exists():
        return []

    reporter.add_input(path)

    _, rows = read_csv(
        path,
        OUTPUT_COLUMNS,
        "existing cumulative opener CSV",
        allow_empty=True,
        exact_columns=OUTPUT_COLUMNS,
    )

    normalized = [
        {
            column: clean(row.get(column))
            for column in OUTPUT_COLUMNS
        }
        for row in rows
    ]

    validate_opener_rows(
        normalized,
        label="existing cumulative opener CSV",
    )
    return normalized


def preserve_existing_timestamp(
    new_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
) -> None:
    existing_by_key = {
        row_key(row): row
        for row in existing_rows
    }

    for row in new_rows:
        previous = existing_by_key.get(row_key(row))
        if previous is None:
            continue

        previous_timestamp = clean(
            previous.get("opening_timestamp")
        )
        if previous_timestamp:
            row["opening_timestamp"] = previous_timestamp


def refresh_preserved_current_fields(
    rows: list[dict[str, str]],
    weekly_row: dict[str, str],
    *,
    current_odds_available: bool,
) -> list[dict[str, str]]:
    weekly_bookmaker = clean(weekly_row.get("bookmaker"))
    refreshed: list[dict[str, str]] = []

    for original in rows:
        row = dict(original)
        same_bookmaker = (
            not weekly_bookmaker
            or normalize_bookmaker(row.get("bookmaker"))
            == normalize_bookmaker(weekly_bookmaker)
        )

        market_type = clean(row.get("market_type"))
        bet_side = clean(row.get("bet_side"))

        if current_odds_available and same_bookmaker:
            if market_type == "h2h":
                current = clean(
                    weekly_row.get(
                        "home_moneyline_american"
                        if bet_side == "home"
                        else "away_moneyline_american"
                    )
                )
                row["current_moneyline"] = current
                row["moneyline_movement"] = numeric_movement(
                    current,
                    row.get("opening_moneyline"),
                )
            elif market_type == "spreads":
                current = clean(
                    weekly_row.get(
                        "home_spread"
                        if bet_side == "home"
                        else "away_spread"
                    )
                )
                row["current_spread"] = current
                row["spread_movement"] = numeric_movement(
                    current,
                    row.get("opening_spread"),
                )
            else:
                current = clean(weekly_row.get("total"))
                row["current_total"] = current
                row["total_movement"] = numeric_movement(
                    current,
                    row.get("opening_total"),
                )
        else:
            if market_type == "h2h":
                row["current_moneyline"] = ""
                row["moneyline_movement"] = ""
            elif market_type == "spreads":
                row["current_spread"] = ""
                row["spread_movement"] = ""
            else:
                row["current_total"] = ""
                row["total_movement"] = ""

        refreshed.append(row)

    return refreshed


def fully_valid_existing_game(
    rows: list[dict[str, str]],
) -> bool:
    if len(rows) != 6:
        return False
    return all(
        clean(row.get("opener_status")) == "ok"
        for row in rows
    )


def merge_current_week(
    *,
    weekly_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    snapshot_index: dict[tuple[str, str], str],
    reporter: PipelineReporter,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    current_game_ids = {
        clean(row.get("game_id"))
        for row in weekly_rows
    }
    existing_by_game = game_groups(existing_rows)

    final_rows = [
        dict(row)
        for row in existing_rows
        if clean(row.get("game_id")) not in current_game_ids
    ]

    stats = {
        "available_games": 0,
        "unavailable_games": 0,
        "refreshed_games": 0,
        "preserved_games": 0,
        "new_degraded_games": 0,
        "bookmaker_fallbacks": 0,
        "fetch_or_missing_games": 0,
    }

    for weekly_row in weekly_rows:
        game_id = clean(weekly_row.get("game_id"))
        prior_rows = existing_by_game.get(game_id, [])
        odds_available = (
            clean(weekly_row.get("odds_available")) == "1"
        )

        if not odds_available:
            stats["unavailable_games"] += 1

            if prior_rows:
                preserved = refresh_preserved_current_fields(
                    prior_rows,
                    weekly_row,
                    current_odds_available=False,
                )
                final_rows.extend(preserved)
                stats["preserved_games"] += 1

            reporter.warning(
                "Current weekly odds unavailable; preserved "
                "known opening history when available",
                game_id=game_id,
                preserved_existing_rows=len(prior_rows),
            )
            continue

        stats["available_games"] += 1

        (
            new_rows,
            used_fallback,
            resolved_bookmaker,
            degraded_reason,
        ) = build_rows_for_weekly_row(
            weekly_row,
            snapshot_index,
        )

        validate_opener_rows(
            new_rows,
            label=f"generated opener rows game_id={game_id}",
            allow_empty=False,
        )

        if used_fallback:
            stats["bookmaker_fallbacks"] += 1
            reporter.warning(
                "Preferred bookmaker unavailable; used existing "
                "ESPN fallback behavior",
                game_id=game_id,
                requested_bookmaker=clean(
                    weekly_row.get("bookmaker")
                ),
                resolved_bookmaker=resolved_bookmaker,
            )

        all_ok = all(
            clean(row.get("opener_status")) == "ok"
            for row in new_rows
        )

        if all_ok:
            preserve_existing_timestamp(
                new_rows,
                prior_rows,
            )
            final_rows.extend(new_rows)
            stats["refreshed_games"] += 1

            if any(
                not clean(row.get("opening_timestamp"))
                for row in new_rows
            ):
                reporter.warning(
                    "Opening line was available but no matching "
                    "historical snapshot timestamp was found",
                    game_id=game_id,
                    bookmaker=resolved_bookmaker,
                )

            continue

        stats["fetch_or_missing_games"] += 1

        if fully_valid_existing_game(prior_rows):
            preserved = refresh_preserved_current_fields(
                prior_rows,
                weekly_row,
                current_odds_available=True,
            )
            final_rows.extend(preserved)
            stats["preserved_games"] += 1

            reporter.warning(
                "Degraded ESPN opener response; preserved "
                "previously valid opening history",
                game_id=game_id,
                reason=degraded_reason or "embedded_opening_fields_missing",
                generated_statuses=sorted(
                    {
                        clean(row.get("opener_status"))
                        for row in new_rows
                    }
                ),
            )
        else:
            final_rows.extend(new_rows)
            stats["new_degraded_games"] += 1

            reporter.warning(
                "Degraded ESPN opener response with no complete "
                "previous valid opener history to preserve",
                game_id=game_id,
                reason=degraded_reason or "embedded_opening_fields_missing",
                generated_statuses=sorted(
                    {
                        clean(row.get("opener_status"))
                        for row in new_rows
                    }
                ),
            )

    final_rows.sort(
        key=lambda row: (
            clean(row.get("game_id")),
            clean(row.get("market_type")),
            clean(row.get("bet_side")),
            clean(row.get("bookmaker")),
        )
    )

    validate_opener_rows(
        final_rows,
        label="final cumulative opener rows",
    )

    return final_rows, stats


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
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in OUTPUT_COLUMNS
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def publish(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".opening_odds_stage_",
        dir=output_path.parent,
    ) as staging_dir:
        staged_path = Path(staging_dir) / output_path.name

        write_csv(staged_path, rows)

        _, staged_rows = read_csv(
            staged_path,
            OUTPUT_COLUMNS,
            "staged opener CSV",
            allow_empty=True,
            exact_columns=OUTPUT_COLUMNS,
        )

        validate_opener_rows(
            staged_rows,
            label="staged opener CSV",
        )

        if len(staged_rows) != len(rows):
            fail(
                "Staged opener row count changed during "
                "round-trip validation"
            )

        os.replace(staged_path, output_path)


def run(
    args: argparse.Namespace,
    reporter: PipelineReporter,
) -> None:
    weekly_path = (
        WEEKLY_DIR
        / f"week_{args.week}_NFL_weekly_schedule.csv"
    )
    output_path = (
        OPENERS_DIR
        / f"{args.season}_NFL_openers.csv"
    )

    reporter.add_input(weekly_path)
    reporter.update_details(
        {
            "configured_season": args.season,
            "configured_week": args.week,
            "weekly_path": str(weekly_path),
            "output_path": str(output_path),
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    _, weekly_rows = read_csv(
        weekly_path,
        WEEKLY_REQUIRED_COLUMNS,
        "configured weekly schedule CSV",
    )
    validate_weekly_rows(
        weekly_rows,
        season=args.season,
        week=args.week,
    )

    existing_rows = read_existing_openers(
        output_path,
        reporter,
    )

    (
        snapshot_index,
        snapshot_files,
        snapshot_rows,
    ) = load_snapshot_index(reporter)

    reporter.update_details(
        {
            "weekly_rows": len(weekly_rows),
            "existing_opener_rows": len(existing_rows),
            "snapshot_files_scanned": snapshot_files,
            "snapshot_rows_scanned": snapshot_rows,
            "snapshot_game_bookmaker_keys": len(snapshot_index),
        }
    )

    final_rows, merge_stats = merge_current_week(
        weekly_rows=weekly_rows,
        existing_rows=existing_rows,
        snapshot_index=snapshot_index,
        reporter=reporter,
    )

    reporter.set_rows(
        rows_in=len(weekly_rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            **merge_stats,
            "final_rows_before_publish": len(final_rows),
            "final_games_before_publish": len(
                {
                    clean(row.get("game_id"))
                    for row in final_rows
                }
            ),
        }
    )

    publish(
        output_path,
        final_rows,
    )

    reporter.add_output(output_path)
    reporter.add_output(LOG_FILE)
    reporter.set_rows(
        rows_in=len(weekly_rows),
        rows_out=len(final_rows),
    )
    reporter.update_details(
        {
            "rows_published": len(final_rows),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    ok_rows = sum(
        clean(row.get("opener_status")) == "ok"
        for row in final_rows
    )
    missing_rows = sum(
        clean(row.get("opener_status")) == "missing"
        for row in final_rows
    )
    error_rows = sum(
        clean(row.get("opener_status")) == "error"
        for row in final_rows
    )

    log(f"Weekly schedule input: {weekly_path}")
    log(f"Configured season: {args.season}")
    log(f"Configured week: {args.week}")
    log(f"Weekly rows loaded: {len(weekly_rows)}")
    log(f"Existing opener rows loaded: {len(existing_rows)}")
    log(f"Snapshot files scanned: {snapshot_files}")
    log(f"Snapshot rows scanned: {snapshot_rows}")
    log(f"Available current-week games: {merge_stats['available_games']}")
    log(f"Unavailable current-week games: {merge_stats['unavailable_games']}")
    log(f"Successfully refreshed games: {merge_stats['refreshed_games']}")
    log(f"Preserved historical games: {merge_stats['preserved_games']}")
    log(f"New degraded games: {merge_stats['new_degraded_games']}")
    log(f"Bookmaker fallbacks: {merge_stats['bookmaker_fallbacks']}")
    log(f"Final season opener rows written: {len(final_rows)}")
    log(f"Final season opener ok rows: {ok_rows}")
    log(f"Final season opener missing rows: {missing_rows}")
    log(f"Final season opener error rows: {error_rows}")
    log(f"Output written: {output_path}")

    print(f"Opening odds written: {output_path}")
    print(f"Weekly rows loaded: {len(weekly_rows)}")
    print(f"Successfully refreshed games: {merge_stats['refreshed_games']}")
    print(f"Preserved historical games: {merge_stats['preserved_games']}")
    print(f"Bookmaker fallbacks: {merge_stats['bookmaker_fallbacks']}")
    print(f"Final season opener rows written: {len(final_rows)}")
    print(f"Final season opener ok rows: {ok_rows}")
    print(f"Final season opener missing rows: {missing_rows}")
    print(f"Final season opener error rows: {error_rows}")


def main() -> int:
    args = parse_args()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="00_intake",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            season=args.season,
            week=args.week,
            extra_context={
                "component": "opening odds",
            },
        ) as reporter:
            run(args, reporter)

        return 0

    except Exception as exc:
        log(traceback.format_exc())
        print(
            f"ERROR: {type(exc).__name__}: {exc}; "
            f"see {LOG_FILE}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
