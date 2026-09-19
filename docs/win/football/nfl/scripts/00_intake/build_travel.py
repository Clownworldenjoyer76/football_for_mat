#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

SCHEDULE_DIR = NFL_ROOT / "00_intake" / "schedule" / "weekly"
STADIUM_MAP_PATH = NFL_ROOT / "config" / "mapping" / "stadium_map_nfl.csv"
OUTPUT_DIR = NFL_ROOT / "data" / "travel"
REPORT_ROOT = NFL_ROOT / "errors"
EARTH_RADIUS_MILES = 3958.8

OUTPUT_HEADERS = [
    "game_id", "away_team", "home_team", "away_lat", "away_lon",
    "home_lat", "home_lon", "miles_traveled", "time_zones_crossed",
    "east_to_west", "west_to_east", "international_flag",
    "neutral_site_flag",
]
SCHEDULE_REQUIRED_COLUMNS = [
    "season", "week", "game_id", "game_date", "away_team", "home_team",
    "neutral_site",
]
STADIUM_REQUIRED_COLUMNS = [
    "team", "latitude", "longitude", "timezone", "venue_country",
]


class TravelBuildError(RuntimeError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fail(message: str) -> None:
    raise TravelBuildError(message)


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


def read_csv(path: Path, required: list[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        fail(f"Zero-byte {label}: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except Exception as exc:
        fail(f"Could not read {label} {path}: {type(exc).__name__}: {exc}")
    missing = [c for c in required if c not in columns]
    if missing:
        fail(f"{label} missing columns: {missing}")
    if not rows:
        fail(f"{label} contains no data rows: {path}")
    return rows


def parse_coordinate(value: Any, label: str, minimum: float, maximum: float) -> float:
    text = clean(value)
    try:
        number = float(text)
    except ValueError:
        fail(f"{label} is not numeric: {text!r}")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        fail(f"{label} is outside valid range: {text!r}")
    return number


def validate_schedule(rows: list[dict[str, str]], season: int, week: int) -> None:
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        away = clean(row.get("away_team"))
        home = clean(row.get("home_team"))
        neutral = clean(row.get("neutral_site"))
        if clean(row.get("season")) != str(season):
            fail(f"Weekly schedule row {line} has wrong season")
        if clean(row.get("week")) != str(week):
            fail(f"Weekly schedule row {line} has wrong week")
        if not game_id:
            fail(f"Weekly schedule row {line} has blank game_id")
        if game_id in seen:
            fail(f"Duplicate weekly schedule game_id: {game_id}")
        seen.add(game_id)
        if not away or not home or away == home:
            fail(f"game_id={game_id} has invalid away/home teams")
        try:
            datetime.strptime(clean(row.get("game_date")), "%Y-%m-%d")
        except ValueError:
            fail(f"game_id={game_id} has invalid game_date")
        if neutral not in {"0", "1"}:
            fail(f"game_id={game_id} has invalid neutral_site={neutral!r}")


def load_stadium_map() -> dict[str, dict[str, str]]:
    rows = read_csv(STADIUM_MAP_PATH, STADIUM_REQUIRED_COLUMNS, "stadium map")
    lookup: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        team = clean(row.get("team"))
        if not team:
            continue
        if team in lookup:
            fail(f"Duplicate stadium-map team={team!r} at line {line}")
        parse_coordinate(row.get("latitude"), f"{team} latitude", -90.0, 90.0)
        parse_coordinate(row.get("longitude"), f"{team} longitude", -180.0, 180.0)
        timezone_name = clean(row.get("timezone"))
        if not timezone_name:
            fail(f"Stadium map team={team!r} has blank timezone")
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            fail(f"Stadium map team={team!r} has invalid timezone: {exc}")
        if not clean(row.get("venue_country")):
            fail(f"Stadium map team={team!r} has blank venue_country")
        lookup[team] = row
    if len(lookup) != 32:
        fail(f"Stadium map must contain exactly 32 unique NFL team rows; found {len(lookup)}")
    return lookup


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return EARTH_RADIUS_MILES * (2 * math.asin(math.sqrt(a)))


def utc_offset_hours(timezone_name: str, game_date: str) -> float:
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo(timezone_name)
        )
    except Exception as exc:
        fail(
            f"Could not compute UTC offset for timezone={timezone_name!r}, "
            f"game_date={game_date!r}: {exc}"
        )
    offset = dt.utcoffset()
    if offset is None:
        fail(f"No UTC offset for timezone={timezone_name!r}, game_date={game_date!r}")
    return offset.total_seconds() / 3600


def build_row(game: dict[str, str], stadiums: dict[str, dict[str, str]]) -> dict[str, Any]:
    game_id = clean(game.get("game_id"))
    away_team = clean(game.get("away_team"))
    home_team = clean(game.get("home_team"))
    game_date = clean(game.get("game_date"))
    neutral_site = clean(game.get("neutral_site"))
    away = stadiums.get(away_team)
    home = stadiums.get(home_team)
    if away is None:
        fail(f"game_id={game_id} no stadium-map match for away_team={away_team!r}")
    if home is None:
        fail(f"game_id={game_id} no stadium-map match for home_team={home_team!r}")

    away_lat_s, away_lon_s = clean(away["latitude"]), clean(away["longitude"])
    home_lat_s, home_lon_s = clean(home["latitude"]), clean(home["longitude"])
    away_lat = parse_coordinate(away_lat_s, f"{game_id} away latitude", -90.0, 90.0)
    away_lon = parse_coordinate(away_lon_s, f"{game_id} away longitude", -180.0, 180.0)
    home_lat = parse_coordinate(home_lat_s, f"{game_id} home latitude", -90.0, 90.0)
    home_lon = parse_coordinate(home_lon_s, f"{game_id} home longitude", -180.0, 180.0)

    miles = round(haversine_miles(away_lat, away_lon, home_lat, home_lon), 1)
    away_offset = utc_offset_hours(clean(away["timezone"]), game_date)
    home_offset = utc_offset_hours(clean(home["timezone"]), game_date)
    zones = abs(home_offset - away_offset)

    if home_lon > away_lon:
        west_to_east, east_to_west = 1, 0
    elif home_lon < away_lon:
        west_to_east, east_to_west = 0, 1
    else:
        west_to_east, east_to_west = 0, 0

    return {
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "away_lat": away_lat_s,
        "away_lon": away_lon_s,
        "home_lat": home_lat_s,
        "home_lon": home_lon_s,
        "miles_traveled": miles,
        "time_zones_crossed": zones,
        "east_to_west": east_to_west,
        "west_to_east": west_to_east,
        "international_flag": 0 if clean(home["venue_country"]) == "USA" else 1,
        "neutral_site_flag": neutral_site,
    }


def validate_output(rows: list[dict[str, Any]], schedule_rows: list[dict[str, str]]) -> None:
    if not rows or len(rows) != len(schedule_rows):
        fail("Travel output row count does not match weekly schedule")
    schedule_by_id = {clean(r["game_id"]): r for r in schedule_rows}
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        game_id = clean(row.get("game_id"))
        if not game_id or game_id in seen:
            fail(f"Travel row {line} has blank/duplicate game_id={game_id!r}")
        seen.add(game_id)
        source = schedule_by_id.get(game_id)
        if source is None:
            fail(f"Travel game_id={game_id} absent from weekly schedule")
        if clean(row.get("away_team")) != clean(source.get("away_team")):
            fail(f"Travel game_id={game_id} away_team mismatch")
        if clean(row.get("home_team")) != clean(source.get("home_team")):
            fail(f"Travel game_id={game_id} home_team mismatch")
        if clean(row.get("neutral_site_flag")) != clean(source.get("neutral_site")):
            fail(f"Travel game_id={game_id} neutral_site_flag mismatch")

        for column in OUTPUT_HEADERS:
            if clean(row.get(column)) == "":
                fail(f"Travel game_id={game_id} has blank {column}")

        parse_coordinate(row["away_lat"], f"{game_id} away_lat", -90.0, 90.0)
        parse_coordinate(row["away_lon"], f"{game_id} away_lon", -180.0, 180.0)
        parse_coordinate(row["home_lat"], f"{game_id} home_lat", -90.0, 90.0)
        parse_coordinate(row["home_lon"], f"{game_id} home_lon", -180.0, 180.0)

        try:
            miles = float(clean(row["miles_traveled"]))
            zones = float(clean(row["time_zones_crossed"]))
        except ValueError:
            fail(f"Travel game_id={game_id} has nonnumeric travel metrics")
        if not math.isfinite(miles) or miles < 0 or not math.isfinite(zones) or zones < 0:
            fail(f"Travel game_id={game_id} has invalid travel metrics")

        east = clean(row["east_to_west"])
        west = clean(row["west_to_east"])
        international = clean(row["international_flag"])
        neutral = clean(row["neutral_site_flag"])
        if east not in {"0", "1"} or west not in {"0", "1"}:
            fail(f"Travel game_id={game_id} has invalid direction flags")
        if east == "1" and west == "1":
            fail(f"Travel game_id={game_id} has both direction flags set")
        if international not in {"0", "1"} or neutral not in {"0", "1"}:
            fail(f"Travel game_id={game_id} has invalid binary flags")

    if seen != set(schedule_by_id):
        fail("Travel game IDs do not exactly match weekly schedule game IDs")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_HEADERS})
        handle.flush()
        os.fsync(handle.fileno())


def publish(
    output_path: Path,
    output_rows: list[dict[str, Any]],
    schedule_rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".travel_stage_", dir=output_path.parent) as stage:
        staged = Path(stage) / output_path.name
        write_csv(staged, output_rows)
        if staged.stat().st_size == 0:
            fail("Staged travel CSV is zero bytes")
        with staged.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames or []) != OUTPUT_HEADERS:
                fail("Staged travel CSV headers changed")
            staged_rows = list(reader)
        validate_output(staged_rows, schedule_rows)
        os.replace(staged, output_path)


def run(args: argparse.Namespace, reporter: PipelineReporter) -> None:
    schedule_path = SCHEDULE_DIR / f"week_{args.week}_NFL_weekly_schedule.csv"
    output_path = OUTPUT_DIR / f"{args.season}_week_{args.week}_travel.csv"

    reporter.add_input(schedule_path)
    reporter.add_input(STADIUM_MAP_PATH)
    reporter.update_details({
        "configured_season": args.season,
        "configured_week": args.week,
        "schedule_path": str(schedule_path),
        "stadium_map_path": str(STADIUM_MAP_PATH),
        "output_path": str(output_path),
        "earth_radius_miles": EARTH_RADIUS_MILES,
        "publication_completed": False,
        "staged_roundtrip_verified": False,
    })

    schedule_rows = read_csv(
        schedule_path, SCHEDULE_REQUIRED_COLUMNS, "configured weekly schedule"
    )
    validate_schedule(schedule_rows, args.season, args.week)
    stadiums = load_stadium_map()

    missing = sorted({
        team
        for row in schedule_rows
        for team in (clean(row["away_team"]), clean(row["home_team"]))
        if team not in stadiums
    })
    if missing:
        fail(f"Weekly schedule teams missing from stadium map: {missing}")

    output_rows = [build_row(row, stadiums) for row in schedule_rows]
    validate_output(output_rows, schedule_rows)

    reporter.set_rows(rows_in=len(schedule_rows), rows_out=0)
    reporter.update_details({
        "schedule_rows": len(schedule_rows),
        "stadium_team_rows": len(stadiums),
        "travel_rows_built": len(output_rows),
        "missing_schedule_teams": missing,
    })

    publish(output_path, output_rows, schedule_rows)

    reporter.add_output(output_path)
    reporter.set_rows(rows_in=len(schedule_rows), rows_out=len(output_rows))
    reporter.update_details({
        "rows_published": len(output_rows),
        "staged_roundtrip_verified": True,
        "publication_completed": True,
    })
    print(f"rows={len(output_rows)} output={output_path}")


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
            extra_context={"component": "travel"},
        ) as reporter:
            run(args, reporter)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
