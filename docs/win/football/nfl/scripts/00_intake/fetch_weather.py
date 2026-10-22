#!/usr/bin/env python3
"""
Fetch validated weather data from met.no for one configured NFL season/week.

Input:
    docs/win/football/nfl/00_intake/schedule/weekly/
        week_{week}_NFL_weekly_schedule.csv

Output:
    docs/win/football/nfl/data/weather/
        week_{week}_NFL_weekly_weather.csv

Existing past-game rows are preserved unchanged. Future games are refreshed
when provider data is available. A transient provider failure preserves an
existing valid forecast when one exists. Legitimately unavailable forecasts
remain blank.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

METNO_URL = (
    "https://api.met.no/weatherapi/"
    "locationforecast/2.0/complete"
)
METNO_USER_AGENT = os.environ.get(
    "METNO_USER_AGENT",
    "MatsPicksWeather/1.0 local-dev",
)
REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECONDS = 1.25
REQUEST_ATTEMPTS = 4

SCHEDULE_DIR = NFL_ROOT / "00_intake" / "schedule" / "weekly"
STADIUM_MAP_PATH = (
    NFL_ROOT / "config" / "mapping" / "stadium_map_nfl.csv"
)
OUTPUT_DIR = NFL_ROOT / "data" / "weather"
ERROR_LOG_DIR = NFL_ROOT / "errors" / "00_intake"
ERROR_LOG_PATH = ERROR_LOG_DIR / "fetch_weather.txt"
REPORT_ROOT = NFL_ROOT / "errors"

OUTPUT_HEADERS = [
    "game_id",
    "stadium",
    "latitude",
    "longitude",
    "game_time",
    "game_timezone",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_fetched_at",
]

SCHEDULE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "stadium",
    "game_timezone",
]

STADIUM_REQUIRED_COLUMNS = [
    "team",
    "stadium",
    "latitude",
    "longitude",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
]

WEATHER_VALUE_FIELDS = [
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
]

STATIC_MAP_FIELDS = [
    "latitude",
    "longitude",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
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


class WeatherError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise WeatherError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch validated met.no weather for one configured "
            "NFL season/week."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")

    if args.week < 1 or args.week > 22:
        parser.error("--week must be between 1 and 22")

    return args


def read_csv(
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    exact_columns: list[str] | None = None,
    allow_empty: bool = False,
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

    if not rows and not allow_empty:
        fail(f"{label} contains no data rows: {path}")

    return fieldnames, rows


def parse_game_datetime(game: dict[str, str]) -> datetime:
    game_id = clean(game.get("game_id"))
    game_date = clean(game.get("game_date"))
    game_time = clean(game.get("game_time"))
    game_timezone = clean(game.get("game_timezone"))

    if not game_date:
        fail(f"game_id={game_id} has blank game_date")

    if not game_time:
        fail(f"game_id={game_id} has blank game_time")

    if not game_timezone:
        fail(f"game_id={game_id} has blank game_timezone")

    try:
        naive = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError as exc:
        fail(
            f"game_id={game_id} has invalid game_date/game_time "
            f"{game_date!r} {game_time!r}: {exc}"
        )

    try:
        tz = ZoneInfo(game_timezone)
    except Exception as exc:
        fail(
            f"game_id={game_id} has invalid game_timezone "
            f"{game_timezone!r}: {exc}"
        )

    return naive.replace(tzinfo=tz)


def load_schedule(
    *,
    season: int,
    week: int,
) -> tuple[Path, list[dict[str, str]], dict[str, datetime]]:
    path = (
        SCHEDULE_DIR
        / f"week_{week}_NFL_weekly_schedule.csv"
    )

    _, rows = read_csv(
        path,
        label="configured weekly schedule",
        required_columns=SCHEDULE_REQUIRED_COLUMNS,
    )

    seen_ids: set[str] = set()
    game_datetimes: dict[str, datetime] = {}

    for line_number, row in enumerate(rows, start=2):
        row_season = clean(row.get("season"))
        row_week = clean(row.get("week"))
        game_id = clean(row.get("game_id"))
        home_team = clean(row.get("home_team"))

        if row_season != str(season):
            fail(
                f"{path} line {line_number} has season="
                f"{row_season!r}; expected={season}"
            )

        if row_week != str(week):
            fail(
                f"{path} line {line_number} has week="
                f"{row_week!r}; expected={week}"
            )

        if not game_id:
            fail(
                f"{path} line {line_number} has blank game_id"
            )

        if game_id in seen_ids:
            fail(
                f"{path} contains duplicate game_id={game_id}"
            )

        if not home_team:
            fail(
                f"{path} line {line_number} has blank home_team"
            )

        seen_ids.add(game_id)
        game_datetimes[game_id] = parse_game_datetime(row)

    return path, rows, game_datetimes


def _validate_optional_float(
    value: str,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    text = clean(value)
    if not text:
        return

    try:
        number = float(text)
    except ValueError:
        fail(f"{label} must be numeric; received={text!r}")

    if minimum is not None and number < minimum:
        fail(
            f"{label} must be >= {minimum}; received={number}"
        )

    if maximum is not None and number > maximum:
        fail(
            f"{label} must be <= {maximum}; received={number}"
        )


def load_stadium_map() -> dict[tuple[str, str], dict[str, str]]:
    _, rows = read_csv(
        STADIUM_MAP_PATH,
        label="NFL stadium map",
        required_columns=STADIUM_REQUIRED_COLUMNS,
    )

    lookup: dict[tuple[str, str], dict[str, str]] = {}

    for line_number, row in enumerate(rows, start=2):
        stadium = clean(row.get("stadium"))
        team = clean(row.get("team"))

        if not stadium:
            fail(
                f"{STADIUM_MAP_PATH} line {line_number} "
                "has blank stadium"
            )

        key = (stadium, team)

        if key in lookup:
            fail(
                f"{STADIUM_MAP_PATH} contains duplicate "
                f"(stadium, team) key={key!r}"
            )

        latitude = clean(row.get("latitude"))
        longitude = clean(row.get("longitude"))

        if bool(latitude) != bool(longitude):
            fail(
                f"{STADIUM_MAP_PATH} line {line_number} has "
                "only one of latitude/longitude populated"
            )

        _validate_optional_float(
            latitude,
            label=(
                f"{STADIUM_MAP_PATH} line {line_number} latitude"
            ),
            minimum=-90.0,
            maximum=90.0,
        )
        _validate_optional_float(
            longitude,
            label=(
                f"{STADIUM_MAP_PATH} line {line_number} longitude"
            ),
            minimum=-180.0,
            maximum=180.0,
        )

        for flag in (
            "dome_flag",
            "retractable_roof_flag",
            "open_air_flag",
        ):
            value = clean(row.get(flag))
            if value and value not in {"0", "1"}:
                fail(
                    f"{STADIUM_MAP_PATH} line {line_number} "
                    f"has invalid {flag}={value!r}"
                )

        lookup[key] = row

    return lookup


def load_existing_output(
    output_path: Path,
    *,
    schedule_ids: set[str],
    reporter: PipelineReporter,
    log_lines: list[str],
) -> dict[str, dict[str, str]]:
    if not output_path.exists():
        return {}

    reporter.add_input(output_path)

    _, rows = read_csv(
        output_path,
        label="existing weekly weather",
        exact_columns=OUTPUT_HEADERS,
        allow_empty=True,
    )

    by_id: dict[str, dict[str, str]] = {}
    extra_ids: list[str] = []

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))

        if not game_id:
            fail(
                f"{output_path} line {line_number} "
                "has blank game_id"
            )

        if game_id in by_id:
            fail(
                f"{output_path} contains duplicate game_id="
                f"{game_id}"
            )

        validate_weather_row_values(
            row,
            label=f"{output_path} line {line_number}",
        )

        if game_id not in schedule_ids:
            extra_ids.append(game_id)

        by_id[game_id] = row

    if extra_ids:
        log_lines.append(
            "WARNING: existing weather file contains stale "
            f"game IDs not in configured schedule: {extra_ids[:10]}"
        )
        reporter.warning(
            "Existing weather file contains stale game IDs; "
            "they will not be republished",
            count=len(extra_ids),
            examples=extra_ids[:10],
        )

    return by_id


def is_future_game(game_dt: datetime) -> bool:
    return game_dt.astimezone(timezone.utc) > datetime.now(
        timezone.utc
    )


def fetch_weather_json(
    lat: str,
    lon: str,
    *,
    game_id: str,
) -> tuple[dict[str, Any] | None, int, str]:
    url = f"{METNO_URL}?lat={lat}&lon={lon}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": METNO_USER_AGENT},
    )

    last_error = ""

    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:
                raw = response.read()

            try:
                payload = json.loads(raw.decode("utf-8"))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = (
                    "invalid UTF-8/JSON response: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if isinstance(payload, dict):
                    return payload, attempt, ""

                last_error = (
                    "response root was not a JSON object"
                )

        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"

            if exc.code not in RETRYABLE_HTTP_CODES:
                return None, attempt, last_error

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            return None, attempt, last_error

        if attempt < REQUEST_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))

    return None, REQUEST_ATTEMPTS, last_error


def find_closest_timestep(
    weather_json: dict[str, Any],
    target_dt_utc: datetime,
) -> tuple[dict[str, Any] | None, str]:
    properties = weather_json.get("properties")

    if not isinstance(properties, dict):
        return None, "malformed_response"

    timeseries = properties.get("timeseries")

    if not isinstance(timeseries, list):
        return None, "malformed_response"

    if not timeseries:
        return None, "forecast_unavailable"

    best: dict[str, Any] | None = None
    best_diff: float | None = None
    valid_times = 0

    for entry in timeseries:
        if not isinstance(entry, dict):
            continue

        entry_time_text = clean(entry.get("time"))

        try:
            entry_time = datetime.strptime(
                entry_time_text,
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue

        valid_times += 1
        diff = abs(
            (
                entry_time - target_dt_utc
            ).total_seconds()
        )

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = entry

    if valid_times == 0 or best is None or best_diff is None:
        return None, "malformed_response"

    if best_diff > 12 * 3600:
        return None, "forecast_unavailable"

    return best, "ok"


def extract_precip_probability(
    entry: dict[str, Any],
) -> Any:
    details = entry.get("data", {})

    if not isinstance(details, dict):
        return None

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        period = details.get(period_key, {})

        if not isinstance(period, dict):
            continue

        period_details = period.get("details", {})

        if not isinstance(period_details, dict):
            continue

        probability = period_details.get(
            "probability_of_precipitation"
        )

        if probability is not None:
            return probability

    return None


def extract_symbol_code(
    entry: dict[str, Any],
) -> str:
    details = entry.get("data", {})

    if not isinstance(details, dict):
        return ""

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        period = details.get(period_key, {})

        if not isinstance(period, dict):
            continue

        summary = period.get("summary", {})

        if not isinstance(summary, dict):
            continue

        code = clean(summary.get("symbol_code"))

        if code:
            return code

    return ""


def derive_rain_snow_flags(
    symbol_code: str,
) -> tuple[int, int]:
    code = symbol_code.lower()
    rain_flag = (
        1
        if "rain" in code or "sleet" in code
        else 0
    )
    snow_flag = 1 if "snow" in code else 0
    return rain_flag, snow_flag


def extract_weather_values(
    entry: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    data = entry.get("data")

    if not isinstance(data, dict):
        return None, "malformed_response"

    instant = data.get("instant")

    if not isinstance(instant, dict):
        return None, "malformed_response"

    instant_details = instant.get("details")

    if not isinstance(instant_details, dict):
        return None, "malformed_response"

    precip_probability = extract_precip_probability(
        entry
    )
    symbol_code = extract_symbol_code(entry)
    rain_flag, snow_flag = derive_rain_snow_flags(
        symbol_code
    )

    values = {
        "temperature": instant_details.get(
            "air_temperature",
            "",
        ),
        "wind_speed": instant_details.get(
            "wind_speed",
            "",
        ),
        "wind_gust": instant_details.get(
            "wind_speed_of_gust",
            "",
        ),
        "precip_probability": (
            precip_probability
            if precip_probability is not None
            else ""
        ),
        "rain_flag": rain_flag,
        "snow_flag": snow_flag,
        "humidity": instant_details.get(
            "relative_humidity",
            "",
        ),
    }

    return values, "ok"


def blank_weather_values() -> dict[str, str]:
    return {
        field: ""
        for field in WEATHER_VALUE_FIELDS
    }


def existing_has_weather(
    row: dict[str, str] | None,
) -> bool:
    if row is None:
        return False

    return any(
        clean(row.get(field))
        for field in WEATHER_VALUE_FIELDS
    )


def build_static_row(
    game: dict[str, str],
    stadium_row: dict[str, str] | None,
    *,
    weather_fetched_at: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": clean(game.get("game_id")),
        "stadium": clean(game.get("stadium")),
        "latitude": "",
        "longitude": "",
        "game_time": clean(game.get("game_time")),
        "game_timezone": clean(
            game.get("game_timezone")
        ),
        **blank_weather_values(),
        "roof_type": "",
        "dome_flag": "",
        "retractable_roof_flag": "",
        "open_air_flag": "",
        "weather_fetched_at": weather_fetched_at,
    }

    if stadium_row is None:
        return row

    row.update(
        {
            "latitude": clean(
                stadium_row.get("latitude")
            ),
            "longitude": clean(
                stadium_row.get("longitude")
            ),
            "roof_type": clean(
                stadium_row.get("roof_type")
            ),
            "dome_flag": clean(
                stadium_row.get("dome_flag")
            ),
            "retractable_roof_flag": clean(
                stadium_row.get(
                    "retractable_roof_flag"
                )
            ),
            "open_air_flag": clean(
                stadium_row.get("open_air_flag")
            ),
        }
    )

    return row


def preserve_existing_weather(
    row: dict[str, Any],
    existing_row: dict[str, str],
) -> None:
    for field in WEATHER_VALUE_FIELDS:
        row[field] = clean(existing_row.get(field))

    existing_stamp = clean(
        existing_row.get("weather_fetched_at")
    )
    if existing_stamp:
        row["weather_fetched_at"] = existing_stamp


def validate_weather_row_values(
    row: dict[str, Any],
    *,
    label: str,
) -> None:
    if not clean(row.get("game_id")):
        fail(f"{label} has blank game_id")

    for field in (
        "rain_flag",
        "snow_flag",
        "dome_flag",
        "retractable_roof_flag",
        "open_air_flag",
    ):
        value = clean(row.get(field))
        if value and value not in {"0", "1"}:
            fail(
                f"{label} has invalid {field}={value!r}"
            )

    numeric_checks = {
        "latitude": (-90.0, 90.0),
        "longitude": (-180.0, 180.0),
        "temperature": (None, None),
        "wind_speed": (0.0, None),
        "wind_gust": (0.0, None),
        "precip_probability": (0.0, 100.0),
        "humidity": (0.0, 100.0),
    }

    for field, (minimum, maximum) in numeric_checks.items():
        _validate_optional_float(
            clean(row.get(field)),
            label=f"{label} {field}",
            minimum=minimum,
            maximum=maximum,
        )


def process_games(
    schedule_rows: list[dict[str, str]],
    *,
    game_datetimes: dict[str, datetime],
    stadium_lookup: dict[
        tuple[str, str],
        dict[str, str],
    ],
    existing_rows: dict[str, dict[str, str]],
    reporter: PipelineReporter,
    log_lines: list[str],
    weather_fetched_at: str,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    dict[str, int],
]:
    output_rows: list[dict[str, Any]] = []
    preserved_past_ids: set[str] = set()

    metrics = {
        "past_rows_preserved": 0,
        "past_rows_created_without_fetch": 0,
        "future_fetch_success": 0,
        "forecast_unavailable": 0,
        "provider_failures": 0,
        "provider_failures_preserved_existing": 0,
        "missing_stadium_matches": 0,
        "missing_coordinates": 0,
        "provider_request_attempts": 0,
    }

    provider_failures: list[dict[str, str]] = []
    unavailable_games: list[str] = []
    stadium_warnings: list[dict[str, str]] = []

    for game in schedule_rows:
        game_id = clean(game.get("game_id"))
        game_dt = game_datetimes[game_id]
        existing_row = existing_rows.get(game_id)

        if not is_future_game(game_dt):
            if existing_row is not None:
                output_rows.append(
                    {
                        header: clean(
                            existing_row.get(header)
                        )
                        for header in OUTPUT_HEADERS
                    }
                )
                preserved_past_ids.add(game_id)
                metrics["past_rows_preserved"] += 1
                continue

        stadium_key = (
            clean(game.get("stadium")),
            clean(game.get("home_team")),
        )
        stadium_row = stadium_lookup.get(
            stadium_key
        )

        row = build_static_row(
            game,
            stadium_row,
            weather_fetched_at=weather_fetched_at,
        )

        if not is_future_game(game_dt):
            log_lines.append(
                f"INFO: game_id={game_id} game is in the past "
                "and has no existing output row; recorded "
                "non-weather fields without a provider request."
            )
            metrics[
                "past_rows_created_without_fetch"
            ] += 1
            output_rows.append(row)
            continue

        if stadium_row is None:
            metrics["missing_stadium_matches"] += 1
            warning = {
                "game_id": game_id,
                "stadium": clean(game.get("stadium")),
                "home_team": clean(game.get("home_team")),
            }
            stadium_warnings.append(warning)
            log_lines.append(
                "WARNING: game_id="
                f"{game_id} no stadium_map match for "
                f"stadium={warning['stadium']!r} "
                f"home_team={warning['home_team']!r}; "
                "weather fields left blank."
            )
            output_rows.append(row)
            continue

        lat = clean(stadium_row.get("latitude"))
        lon = clean(stadium_row.get("longitude"))

        if not lat or not lon:
            metrics["missing_coordinates"] += 1
            warning = {
                "game_id": game_id,
                "stadium": clean(game.get("stadium")),
                "home_team": clean(game.get("home_team")),
            }
            stadium_warnings.append(warning)
            log_lines.append(
                f"WARNING: game_id={game_id} matched stadium "
                "row has no latitude/longitude; weather fields "
                "left blank."
            )
            output_rows.append(row)
            continue

        payload, attempts, fetch_error = (
            fetch_weather_json(
                lat,
                lon,
                game_id=game_id,
            )
        )
        metrics[
            "provider_request_attempts"
        ] += attempts

        if payload is None:
            metrics["provider_failures"] += 1
            failure = {
                "game_id": game_id,
                "error": fetch_error,
            }
            provider_failures.append(failure)

            if existing_has_weather(existing_row):
                preserve_existing_weather(
                    row,
                    existing_row,
                )
                metrics[
                    "provider_failures_preserved_existing"
                ] += 1
                log_lines.append(
                    f"WARNING: game_id={game_id} met.no "
                    f"request failed ({fetch_error}); preserved "
                    "existing weather values."
                )
            else:
                log_lines.append(
                    f"WARNING: game_id={game_id} met.no "
                    f"request failed ({fetch_error}); no existing "
                    "weather was available, so weather fields "
                    "remain blank."
                )

            output_rows.append(row)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        entry, timestep_status = (
            find_closest_timestep(
                payload,
                game_dt.astimezone(timezone.utc),
            )
        )

        if timestep_status == "forecast_unavailable":
            metrics["forecast_unavailable"] += 1
            unavailable_games.append(game_id)
            log_lines.append(
                f"INFO: game_id={game_id} no met.no timestep "
                "within 12 hours of kickoff; weather fields "
                "left blank."
            )
            output_rows.append(row)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        if (
            timestep_status != "ok"
            or entry is None
        ):
            metrics["provider_failures"] += 1
            failure = {
                "game_id": game_id,
                "error": "malformed met.no timeseries response",
            }
            provider_failures.append(failure)

            if existing_has_weather(existing_row):
                preserve_existing_weather(
                    row,
                    existing_row,
                )
                metrics[
                    "provider_failures_preserved_existing"
                ] += 1

            log_lines.append(
                f"WARNING: game_id={game_id} met.no response "
                "was malformed; "
                + (
                    "preserved existing weather values."
                    if existing_has_weather(existing_row)
                    else "weather fields remain blank."
                )
            )
            output_rows.append(row)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        weather_values, values_status = (
            extract_weather_values(entry)
        )

        if (
            values_status != "ok"
            or weather_values is None
        ):
            metrics["provider_failures"] += 1
            failure = {
                "game_id": game_id,
                "error": "malformed met.no weather details",
            }
            provider_failures.append(failure)

            if existing_has_weather(existing_row):
                preserve_existing_weather(
                    row,
                    existing_row,
                )
                metrics[
                    "provider_failures_preserved_existing"
                ] += 1

            log_lines.append(
                f"WARNING: game_id={game_id} met.no weather "
                "details were malformed; "
                + (
                    "preserved existing weather values."
                    if existing_has_weather(existing_row)
                    else "weather fields remain blank."
                )
            )
            output_rows.append(row)
            time.sleep(REQUEST_SLEEP_SECONDS)
            continue

        row.update(weather_values)
        metrics["future_fetch_success"] += 1
        output_rows.append(row)
        time.sleep(REQUEST_SLEEP_SECONDS)

    if provider_failures:
        reporter.warning(
            "One or more met.no requests/responses failed; "
            "existing forecasts were preserved where possible",
            count=len(provider_failures),
            examples=provider_failures[:10],
        )

    if unavailable_games:
        reporter.warning(
            "One or more games were outside the usable met.no "
            "forecast timestep range",
            count=len(unavailable_games),
            game_ids=unavailable_games[:20],
        )

    if stadium_warnings:
        reporter.warning(
            "One or more games could not be fully joined to "
            "usable stadium coordinates",
            count=len(stadium_warnings),
            examples=stadium_warnings[:10],
        )

    return (
        output_rows,
        preserved_past_ids,
        metrics,
    )


def validate_output_rows(
    rows: list[dict[str, Any]],
    *,
    schedule_rows: list[dict[str, str]],
    stadium_lookup: dict[
        tuple[str, str],
        dict[str, str],
    ],
    preserved_past_ids: set[str],
) -> None:
    schedule_by_id = {
        clean(game.get("game_id")): game
        for game in schedule_rows
    }

    if len(rows) != len(schedule_rows):
        fail(
            "Weather output row count does not match schedule "
            f"output={len(rows)} schedule={len(schedule_rows)}"
        )

    seen_ids: set[str] = set()

    for line_number, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))

        validate_weather_row_values(
            row,
            label=f"weather output line {line_number}",
        )

        if game_id in seen_ids:
            fail(
                "Weather output contains duplicate game_id="
                f"{game_id}"
            )
        seen_ids.add(game_id)

        game = schedule_by_id.get(game_id)
        if game is None:
            fail(
                "Weather output contains game absent from "
                f"configured schedule game_id={game_id}"
            )

        if not clean(row.get("weather_fetched_at")):
            fail(
                "Weather output has blank weather_fetched_at "
                f"game_id={game_id}"
            )

        if game_id in preserved_past_ids:
            continue

        expected_stadium = clean(
            game.get("stadium")
        )
        expected_time = clean(
            game.get("game_time")
        )
        expected_timezone = clean(
            game.get("game_timezone")
        )

        if clean(row.get("stadium")) != expected_stadium:
            fail(
                "Weather output stadium mismatch "
                f"game_id={game_id}"
            )

        if clean(row.get("game_time")) != expected_time:
            fail(
                "Weather output game_time mismatch "
                f"game_id={game_id}"
            )

        if (
            clean(row.get("game_timezone"))
            != expected_timezone
        ):
            fail(
                "Weather output game_timezone mismatch "
                f"game_id={game_id}"
            )

        stadium_key = (
            expected_stadium,
            clean(game.get("home_team")),
        )
        stadium_row = stadium_lookup.get(
            stadium_key
        )

        if stadium_row is None:
            for field in STATIC_MAP_FIELDS:
                if clean(row.get(field)):
                    fail(
                        "Weather output contains stadium-map "
                        f"field {field} without a stadium match "
                        f"game_id={game_id}"
                    )
            continue

        for field in STATIC_MAP_FIELDS:
            expected = clean(
                stadium_row.get(field)
            )
            actual = clean(row.get(field))

            if actual != expected:
                fail(
                    "Weather output stadium-map mismatch "
                    f"game_id={game_id} field={field} "
                    f"expected={expected!r} actual={actual!r}"
                )

    if seen_ids != set(schedule_by_id):
        missing = sorted(
            set(schedule_by_id) - seen_ids
        )
        extra = sorted(
            seen_ids - set(schedule_by_id)
        )
        fail(
            "Weather output game universe mismatch "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def normalize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            header: clean(row.get(header))
            for header in OUTPUT_HEADERS
        }
        for row in rows
    ]


def publish(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    schedule_rows: list[dict[str, str]],
    stadium_lookup: dict[
        tuple[str, str],
        dict[str, str],
    ],
    preserved_past_ids: set[str],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{output_path.stem}_",
        suffix=".tmp",
        dir=OUTPUT_DIR,
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
        )
        writer.writeheader()
        writer.writerows(
            normalize_rows(rows)
        )
        handle.flush()
        os.fsync(handle.fileno())

    try:
        staged_columns, staged_rows = read_csv(
            staged_path,
            label="staged weekly weather",
            exact_columns=OUTPUT_HEADERS,
        )

        if staged_columns != OUTPUT_HEADERS:
            fail(
                "Staged weather schema changed during "
                "round-trip validation"
            )

        validate_output_rows(
            staged_rows,
            schedule_rows=schedule_rows,
            stadium_lookup=stadium_lookup,
            preserved_past_ids=preserved_past_ids,
        )

        if staged_rows != normalize_rows(rows):
            fail(
                "Staged weather CSV differs from validated "
                "in-memory projection"
            )

        os.replace(
            staged_path,
            output_path,
        )

    finally:
        if staged_path.exists():
            staged_path.unlink()


def append_legacy_log(
    *,
    weather_fetched_at: str,
    log_lines: list[str],
) -> None:
    ERROR_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ERROR_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"\n--- Run at {weather_fetched_at} ---\n"
        )

        if log_lines:
            for line in log_lines:
                handle.write(line + "\n")
        else:
            handle.write("No issues.\n")


def run(
    reporter: PipelineReporter,
    *,
    season: int,
    week: int,
) -> None:
    weather_fetched_at = datetime.now(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    log_lines: list[str] = []

    schedule_path, schedule_rows, game_datetimes = (
        load_schedule(
            season=season,
            week=week,
        )
    )
    schedule_ids = {
        clean(row.get("game_id"))
        for row in schedule_rows
    }

    output_path = (
        OUTPUT_DIR
        / f"week_{week}_NFL_weekly_weather.csv"
    )

    reporter.add_input(schedule_path)
    reporter.add_input(STADIUM_MAP_PATH)
    reporter.update_details(
        {
            "season": season,
            "week": week,
            "schedule_path": str(schedule_path),
            "stadium_map_path": str(STADIUM_MAP_PATH),
            "output_path": str(output_path),
            "provider_endpoint": METNO_URL,
            "request_attempts": REQUEST_ATTEMPTS,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "expected_output_columns": len(
                OUTPUT_HEADERS
            ),
            "publication_mode": "staged_atomic_replace",
            "publication_completed": False,
            "staged_roundtrip_verified": False,
        }
    )

    stadium_lookup = load_stadium_map()

    existing_rows = load_existing_output(
        output_path,
        schedule_ids=schedule_ids,
        reporter=reporter,
        log_lines=log_lines,
    )

    (
        output_rows,
        preserved_past_ids,
        metrics,
    ) = process_games(
        schedule_rows,
        game_datetimes=game_datetimes,
        stadium_lookup=stadium_lookup,
        existing_rows=existing_rows,
        reporter=reporter,
        log_lines=log_lines,
        weather_fetched_at=weather_fetched_at,
    )

    validate_output_rows(
        output_rows,
        schedule_rows=schedule_rows,
        stadium_lookup=stadium_lookup,
        preserved_past_ids=preserved_past_ids,
    )

    reporter.set_rows(
        rows_in=len(schedule_rows),
        rows_out=0,
    )
    reporter.update_details(
        {
            "schedule_rows": len(schedule_rows),
            "schedule_game_ids": len(schedule_ids),
            "existing_weather_rows": len(
                existing_rows
            ),
            **metrics,
        }
    )

    publish(
        output_path,
        output_rows,
        schedule_rows=schedule_rows,
        stadium_lookup=stadium_lookup,
        preserved_past_ids=preserved_past_ids,
    )

    reporter.add_output(output_path)
    reporter.set_rows(
        rows_in=len(schedule_rows),
        rows_out=len(output_rows),
    )
    reporter.update_details(
        {
            "rows_published": len(output_rows),
            "staged_roundtrip_verified": True,
            "publication_completed": True,
        }
    )

    try:
        append_legacy_log(
            weather_fetched_at=weather_fetched_at,
            log_lines=log_lines,
        )
        reporter.add_output(ERROR_LOG_PATH)
        reporter.set_detail(
            "legacy_log_written",
            True,
        )
    except Exception as exc:
        reporter.warning(
            "Weather output was published but legacy text log "
            "could not be appended",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        reporter.set_detail(
            "legacy_log_written",
            False,
        )

    print(
        f"season={season} week={week} "
        f"rows={len(output_rows)} "
        f"output={output_path}"
    )
    print(
        f"Log written to {ERROR_LOG_PATH}"
        if reporter.status != "FAILED"
        else f"Log target: {ERROR_LOG_PATH}"
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
            week=args.week,
            extra_context={
                "component": "weather fetch",
            },
        ) as reporter:
            run(
                reporter,
                season=args.season,
                week=args.week,
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
