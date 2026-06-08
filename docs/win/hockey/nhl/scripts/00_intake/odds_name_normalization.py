#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/odds_name_normalization.py.py

import csv
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SPORTSBOOK_DIR = Path("docs/win/hockey/nhl/00_intake/sportsbook")
TEAM_MAP_PATH = Path("docs/win/hockey/nhl/config/mapping/team_map_nhl.csv")
NO_MAP_PATH = Path("docs/win/hockey/nhl/config/mapping/no_map_nhl_odds.csv")
LOG_DIR = Path("docs/win/hockey/nhl/errors/00_intake")
LOG_FILE = LOG_DIR / "odds_name_normalization.txt"

TARGET_LEAGUE = "nhl"
TEAM_COLUMNS = ["home_team", "away_team"]


def setup_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {message}\n")


def start_log() -> None:
    setup_dirs()
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"=== odds_name_normalization RUN {datetime.now(ET).isoformat()} ===\n"
        )


def clean_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_team_map() -> dict[str, str]:
    if not TEAM_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing team map: {TEAM_MAP_PATH}")

    team_map = {}

    with open(TEAM_MAP_PATH, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"league", "alias", "canonical_team"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"{TEAM_MAP_PATH} missing required columns: {sorted(missing)}"
            )

        for row in reader:
            league = clean_value(row.get("league")).lower()
            alias = clean_value(row.get("alias"))
            canonical_team = clean_value(row.get("canonical_team"))

            if league != TARGET_LEAGUE:
                continue

            if not alias or not canonical_team:
                continue

            team_map[alias.lower()] = canonical_team

    log(f"Loaded {len(team_map)} NHL team-map aliases from {TEAM_MAP_PATH}")
    return team_map


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_no_map(unmapped: set[str]) -> None:
    with open(NO_MAP_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["league", "team"])
        writer.writeheader()

        for team in sorted(unmapped):
            writer.writerow(
                {
                    "league": TARGET_LEAGUE,
                    "team": team,
                }
            )

    log(f"WROTE {NO_MAP_PATH} ({len(unmapped)} unmapped team names)")


def normalize_file(path: Path, team_map: dict[str, str]) -> tuple[int, int, set[str]]:
    fieldnames, rows = read_csv(path)

    missing_columns = [col for col in TEAM_COLUMNS if col not in fieldnames]
    if missing_columns:
        raise RuntimeError(f"{path} missing required columns: {missing_columns}")

    changed_count = 0
    unmapped = set()

    for row in rows:
        for col in TEAM_COLUMNS:
            original = clean_value(row.get(col))

            if not original:
                continue

            canonical = team_map.get(original.lower())

            if canonical:
                if canonical != original:
                    row[col] = canonical
                    changed_count += 1
            else:
                unmapped.add(original)

    write_csv(path, fieldnames, rows)

    return len(rows), changed_count, unmapped


def main() -> None:
    start_log()

    log(f"Sportsbook input dir: {SPORTSBOOK_DIR}")
    log(f"Team map: {TEAM_MAP_PATH}")
    log(f"No-map output: {NO_MAP_PATH}")

    team_map = load_team_map()

    sportsbook_files = sorted(SPORTSBOOK_DIR.glob("NHL_*.csv"))

    log(f"Sportsbook files found: {len(sportsbook_files)}")

    all_unmapped = set()
    total_rows = 0
    total_changed = 0

    for path in sportsbook_files:
        rows_count, changed_count, unmapped = normalize_file(path, team_map)

        total_rows += rows_count
        total_changed += changed_count
        all_unmapped.update(unmapped)

        log(
            f"NORMALIZED {path} | rows={rows_count} "
            f"changed={changed_count} unmapped={len(unmapped)}"
        )

    write_no_map(all_unmapped)

    log("--- SUMMARY ---")
    log(f"Files processed: {len(sportsbook_files)}")
    log(f"Rows processed: {total_rows}")
    log(f"Team values changed: {total_changed}")
    log(f"Unmapped unique teams: {len(all_unmapped)}")
    log("STATUS: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise
