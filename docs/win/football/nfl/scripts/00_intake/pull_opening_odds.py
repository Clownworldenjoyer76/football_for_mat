#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path("docs/win/football/nfl")
WEEKLY_DIR = BASE_DIR / "00_intake" / "schedule" / "weekly"
OPENERS_DIR = BASE_DIR / "00_intake" / "odds" / "openers"
SNAPSHOT_DIR = BASE_DIR / "00_intake" / "odds" / "snapshots"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
OPENERS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_opening_odds.txt"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message: str) -> None:
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def latest_file(directory: Path, pattern: str, label: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        fail(f"No {label} found in {directory} matching {pattern}")
    return files[0]


def read_csv(path: Path, required_columns: list[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"Missing {label}: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            fail(f"{label} missing columns: {missing}")
        return list(reader)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def http_get_json(url: str) -> tuple[int | None, object, str]:
    last_error = ""
    for attempt in range(1, HTTP_RETRIES + 1):
        request = Request(url, headers={"User-Agent": "nfl-pull-opening-odds-espn/1.0"})
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                try:
                    return response.status, json.loads(body), ""
                except Exception as exc:
                    return response.status, {}, f"JSON parse failed: {exc}"
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = body or str(exc)
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                return exc.code, {}, last_error
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:
            last_error = str(exc)

        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))

    return None, {}, last_error or "request failed"


def odds_url(event_id: str) -> str:
    return (
        f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}/odds"
        "?lang=en&region=us"
    )


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def clean_number(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number)


def clean_american(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return str(int(round(number)))


def numeric_movement(current_value: object, opening_value: object) -> str:
    current = to_float(current_value)
    opening = to_float(opening_value)
    if current is None or opening is None:
        return ""
    movement = current - opening
    if movement.is_integer():
        return str(int(movement))
    return str(round(movement, 4))


def nested(data: object, *keys: str) -> object:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def display_value(value: object) -> object:
    if isinstance(value, dict):
        for key in ("american", "alternateDisplayValue", "value"):
            if value.get(key) not in (None, ""):
                return value.get(key)
    return value


def select_bookmaker_item(payload: object, bookmaker: str) -> dict | None:
    if not isinstance(payload, dict):
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    valid = [item for item in items if isinstance(item, dict)]
    for item in valid:
        provider = item.get("provider")
        if isinstance(provider, dict) and str(provider.get("name", "")).strip() == bookmaker:
            return item
    return valid[0] if valid else None


def earliest_snapshot_timestamp(event_id: str, bookmaker: str) -> str:
    earliest = ""
    for path in sorted(SNAPSHOT_DIR.glob("*_NFL_odds.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("game_id", "")).strip() != event_id:
                        continue
                    if bookmaker and str(row.get("bookmaker", "")).strip() != bookmaker:
                        continue
                    stamp = str(row.get("snapshot_fetched_at", "")).strip()
                    if stamp and (not earliest or stamp < earliest):
                        earliest = stamp
        except Exception as exc:
            log(f"SNAPSHOT_READ_FAILED path={path} error={exc}")
    return earliest


def status_fields(status: str, reason: str = "", http_status: object = "") -> dict[str, str]:
    return {
        "opener_status": status,
        "opener_missing_reason": reason,
        "opener_http_status": str(http_status or ""),
    }


def base_row(weekly_row: dict[str, str], market_type: str, bet_side: str, opening_timestamp: str) -> dict[str, str]:
    return {
        "game_id": weekly_row.get("game_id", ""),
        "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
        "market_type": market_type,
        "bet_side": bet_side,
        "opening_line": "",
        "opening_odds_american": "",
        "opening_timestamp": opening_timestamp,
        "bookmaker": weekly_row.get("bookmaker", ""),
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


def build_rows_for_weekly_row(weekly_row: dict[str, str]) -> list[dict[str, str]]:
    event_id = str(weekly_row.get("odds_provider_game_id", "")).strip()
    bookmaker = str(weekly_row.get("bookmaker", "")).strip() or "DraftKings"

    if not event_id:
        return []

    http_status, payload, request_error = http_get_json(odds_url(event_id))
    if request_error:
        status = status_fields("error", request_error, http_status)
        rows = []
        for market_type, sides in (("h2h", ("home", "away")), ("spreads", ("home", "away")), ("totals", ("over", "under"))):
            for side in sides:
                row = base_row(weekly_row, market_type, side, "")
                row.update(status)
                rows.append(row)
        return rows

    item = select_bookmaker_item(payload, bookmaker)
    if not item:
        status = status_fields("missing", "no_espn_odds_item", http_status)
        rows = []
        for market_type, sides in (("h2h", ("home", "away")), ("spreads", ("home", "away")), ("totals", ("over", "under"))):
            for side in sides:
                row = base_row(weekly_row, market_type, side, "")
                row.update(status)
                rows.append(row)
        return rows

    provider = item.get("provider") if isinstance(item.get("provider"), dict) else {}
    resolved_bookmaker = str(provider.get("name", "")).strip() or bookmaker
    weekly_row = dict(weekly_row)
    weekly_row["bookmaker"] = resolved_bookmaker
    opening_timestamp = earliest_snapshot_timestamp(event_id, resolved_bookmaker)

    home_open_ml = clean_american(nested(item, "homeTeamOdds", "open", "moneyLine", "american"))
    away_open_ml = clean_american(nested(item, "awayTeamOdds", "open", "moneyLine", "american"))
    home_current_ml = clean_american(nested(item, "homeTeamOdds", "current", "moneyLine", "american")) or str(weekly_row.get("home_moneyline_american", "")).strip()
    away_current_ml = clean_american(nested(item, "awayTeamOdds", "current", "moneyLine", "american")) or str(weekly_row.get("away_moneyline_american", "")).strip()

    home_open_spread = clean_number(display_value(nested(item, "homeTeamOdds", "open", "pointSpread")))
    away_open_spread = clean_number(display_value(nested(item, "awayTeamOdds", "open", "pointSpread")))
    home_current_spread = clean_number(display_value(nested(item, "homeTeamOdds", "current", "pointSpread"))) or str(weekly_row.get("home_spread", "")).strip()
    away_current_spread = clean_number(display_value(nested(item, "awayTeamOdds", "current", "pointSpread"))) or str(weekly_row.get("away_spread", "")).strip()
    home_open_spread_odds = clean_american(nested(item, "homeTeamOdds", "open", "spread", "american"))
    away_open_spread_odds = clean_american(nested(item, "awayTeamOdds", "open", "spread", "american"))

    opening_total = clean_number(display_value(nested(item, "open", "total")))
    current_total = clean_number(display_value(nested(item, "current", "total"))) or str(weekly_row.get("total", "")).strip()
    over_open_odds = clean_american(nested(item, "open", "over", "american"))
    under_open_odds = clean_american(nested(item, "open", "under", "american"))

    rows: list[dict[str, str]] = []

    for side, opening_ml, current_ml in (
        ("home", home_open_ml, home_current_ml),
        ("away", away_open_ml, away_current_ml),
    ):
        row = base_row(weekly_row, "h2h", side, opening_timestamp)
        row.update(
            {
                "opening_odds_american": opening_ml,
                "opening_moneyline": opening_ml,
                "current_moneyline": current_ml,
                "moneyline_movement": numeric_movement(current_ml, opening_ml),
            }
        )
        row.update(status_fields("ok" if opening_ml else "missing", "" if opening_ml else "no_embedded_open_moneyline"))
        rows.append(row)

    for side, opening_spread, current_spread, opening_price in (
        ("home", home_open_spread, home_current_spread, home_open_spread_odds),
        ("away", away_open_spread, away_current_spread, away_open_spread_odds),
    ):
        row = base_row(weekly_row, "spreads", side, opening_timestamp)
        row.update(
            {
                "opening_line": opening_spread,
                "opening_odds_american": opening_price,
                "opening_spread": opening_spread,
                "current_spread": current_spread,
                "spread_movement": numeric_movement(current_spread, opening_spread),
            }
        )
        row.update(status_fields("ok" if opening_spread else "missing", "" if opening_spread else "no_embedded_open_spread"))
        rows.append(row)

    for side, opening_price in (("over", over_open_odds), ("under", under_open_odds)):
        row = base_row(weekly_row, "totals", side, opening_timestamp)
        row.update(
            {
                "opening_line": opening_total,
                "opening_odds_american": opening_price,
                "opening_total": opening_total,
                "current_total": current_total,
                "total_movement": numeric_movement(current_total, opening_total),
            }
        )
        row.update(status_fields("ok" if opening_total else "missing", "" if opening_total else "no_embedded_open_total"))
        rows.append(row)

    return rows


def build_opening_rows(weekly_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in weekly_rows:
        if str(row.get("odds_available", "")).strip() != "1":
            continue
        try:
            output_rows.extend(build_rows_for_weekly_row(row))
        except Exception as exc:
            log(
                "OPENING_ROW_FAILED "
                f"game_id={row.get('game_id', '')} "
                f"event_id={row.get('odds_provider_game_id', '')} "
                f"error={exc}"
            )
    return output_rows


def read_existing_openers(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [{column: row.get(column, "") for column in OUTPUT_COLUMNS} for row in reader]
        missing = [column for column in OUTPUT_COLUMNS if column not in fieldnames]
        if missing:
            log(f"Existing opener file missing columns; blanks inserted: {missing}")
        return rows


def row_has_opening_data(row: dict[str, str]) -> bool:
    return any(
        str(row.get(column, "")).strip()
        for column in ("opening_line", "opening_odds_american", "opening_spread", "opening_total", "opening_moneyline")
    )


def row_status_rank(row: dict[str, str]) -> int:
    status = str(row.get("opener_status", "")).strip()
    if status == "ok":
        return 3
    if row_has_opening_data(row):
        return 2
    if status == "missing":
        return 1
    return 0


def upsert_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keyed: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for row in existing_rows:
        key = (
            str(row.get("game_id", "")).strip(),
            str(row.get("market_type", "")).strip(),
            str(row.get("bet_side", "")).strip(),
            str(row.get("bookmaker", "")).strip(),
        )
        keyed[key] = row

    for row in new_rows:
        key = (
            str(row.get("game_id", "")).strip(),
            str(row.get("market_type", "")).strip(),
            str(row.get("bet_side", "")).strip(),
            str(row.get("bookmaker", "")).strip(),
        )
        existing = keyed.get(key)
        if existing is None:
            keyed[key] = row
            continue

        if row_status_rank(row) >= row_status_rank(existing):
            merged = dict(row)
            if str(existing.get("opening_timestamp", "")).strip():
                merged["opening_timestamp"] = existing["opening_timestamp"]
            keyed[key] = merged

    rows = list(keyed.values())
    rows.sort(key=lambda row: (row.get("game_id", ""), row.get("market_type", ""), row.get("bet_side", ""), row.get("bookmaker", "")))
    return rows


def detect_season(weekly_rows: list[dict[str, str]]) -> str:
    seasons = sorted({str(row.get("season", "")).strip() for row in weekly_rows if str(row.get("season", "")).strip()})
    if len(seasons) != 1:
        fail(f"Expected exactly one season in weekly schedule, found: {seasons}")
    return seasons[0]


def main() -> None:
    LOG_FILE.write_text("", encoding="utf-8")

    weekly_path = latest_file(WEEKLY_DIR, "week_*_NFL_weekly_schedule.csv", "weekly schedule CSV")
    log(f"Weekly schedule input: {weekly_path}")

    weekly_rows = read_csv(weekly_path, WEEKLY_REQUIRED_COLUMNS, "weekly schedule CSV")
    season = detect_season(weekly_rows)
    output_path = OPENERS_DIR / f"{season}_NFL_openers.csv"

    existing_rows = read_existing_openers(output_path)
    new_rows = build_opening_rows(weekly_rows)
    final_rows = upsert_rows(existing_rows, new_rows)
    write_csv(output_path, final_rows)

    ok_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "ok")
    missing_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "missing")
    error_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "error")

    log(f"Weekly rows loaded: {len(weekly_rows)}")
    log(f"Existing opener rows loaded: {len(existing_rows)}")
    log(f"New opener rows built: {len(new_rows)}")
    log(f"Final opener rows written: {len(final_rows)}")
    log(f"Final opener ok rows: {ok_rows}")
    log(f"Final opener missing rows: {missing_rows}")
    log(f"Final opener error rows: {error_rows}")
    log(f"Output written: {output_path}")

    print(f"Opening odds written: {output_path}")
    print(f"Weekly rows loaded: {len(weekly_rows)}")
    print(f"New opener rows built: {len(new_rows)}")
    print(f"Final opener rows written: {len(final_rows)}")
    print(f"Final opener ok rows: {ok_rows}")
    print(f"Final opener missing rows: {missing_rows}")
    print(f"Final opener error rows: {error_rows}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        print(f"ERROR: see {LOG_FILE}", file=sys.stderr)
        raise
