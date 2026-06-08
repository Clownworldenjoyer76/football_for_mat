from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=probablePitcher"
)
LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
LINEUP_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/lineups"

OUTPUT_DIR = Path("docs/win/baseball/00_intake/mlb_raw")

CSV_HEADERS = [
    "gamePk",
    "gameGuid",
    "game_date",
    "game_time",
    "venue_id",
    "doubleheader",
    "gameNumber",
    "home_team_id",
    "away_team_id",
    "home_pitcher_id",
    "away_pitcher_id",
    "day_night",
    "home_bat_1_id",
    "home_bat_2_id",
    "home_bat_3_id",
    "home_bat_4_id",
    "home_bat_5_id",
    "home_bat_6_id",
    "home_bat_7_id",
    "home_bat_8_id",
    "home_bat_9_id",
    "away_bat_1_id",
    "away_bat_2_id",
    "away_bat_3_id",
    "away_bat_4_id",
    "away_bat_5_id",
    "away_bat_6_id",
    "away_bat_7_id",
    "away_bat_8_id",
    "away_bat_9_id",
]


def fetch_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error for {url}: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error for {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned for {url}") from exc


def safe_get(mapping: dict, *keys, default=""):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current or current[key] is None:
            return default
        current = current[key]
    return current


def batting_slot(batting_order: list, idx: int) -> str:
    if idx < len(batting_order):
        return str(batting_order[idx])
    return ""


def get_lineup(game_pk) -> tuple[list, list]:
    """
    Fetch pre-game lineup from /lineups endpoint.
    Returns (home_order, away_order) as lists of player IDs.
    Falls back to empty lists if not yet available.
    """
    try:
        data = fetch_json(LINEUP_URL.format(game_pk=game_pk))
        home = [p["id"] for p in data.get("homePlayers", []) if "id" in p]
        away = [p["id"] for p in data.get("awayPlayers", []) if "id" in p]
        return home, away
    except RuntimeError:
        return [], []


def build_row(game: dict, live: dict) -> dict:
    game_pk = safe_get(game, "gamePk")

    home_lineup, away_lineup = get_lineup(game_pk)

    # Fall back to boxscore battingOrder if lineups endpoint is empty
    if not home_lineup:
        home_lineup = safe_get(
            live, "liveData", "boxscore", "teams", "home", "battingOrder", default=[]
        )
        if not isinstance(home_lineup, list):
            home_lineup = []

    if not away_lineup:
        away_lineup = safe_get(
            live, "liveData", "boxscore", "teams", "away", "battingOrder", default=[]
        )
        if not isinstance(away_lineup, list):
            away_lineup = []

    return {
        "gamePk": game_pk,
        "gameGuid": safe_get(game, "gameGuid"),
        "game_date": safe_get(game, "officialDate"),
        "game_time": safe_get(game, "gameDate"),
        "venue_id": safe_get(game, "venue", "id"),
        "doubleheader": safe_get(game, "doubleHeader"),
        "gameNumber": safe_get(game, "gameNumber"),
        "home_team_id": safe_get(game, "teams", "home", "team", "id"),
        "away_team_id": safe_get(game, "teams", "away", "team", "id"),
        "home_pitcher_id": safe_get(live, "gameData", "probablePitchers", "home", "id"),
        "away_pitcher_id": safe_get(live, "gameData", "probablePitchers", "away", "id"),
        "day_night": safe_get(game, "dayNight"),
        "home_bat_1_id": batting_slot(home_lineup, 0),
        "home_bat_2_id": batting_slot(home_lineup, 1),
        "home_bat_3_id": batting_slot(home_lineup, 2),
        "home_bat_4_id": batting_slot(home_lineup, 3),
        "home_bat_5_id": batting_slot(home_lineup, 4),
        "home_bat_6_id": batting_slot(home_lineup, 5),
        "home_bat_7_id": batting_slot(home_lineup, 6),
        "home_bat_8_id": batting_slot(home_lineup, 7),
        "home_bat_9_id": batting_slot(home_lineup, 8),
        "away_bat_1_id": batting_slot(away_lineup, 0),
        "away_bat_2_id": batting_slot(away_lineup, 1),
        "away_bat_3_id": batting_slot(away_lineup, 2),
        "away_bat_4_id": batting_slot(away_lineup, 3),
        "away_bat_5_id": batting_slot(away_lineup, 4),
        "away_bat_6_id": batting_slot(away_lineup, 5),
        "away_bat_7_id": batting_slot(away_lineup, 6),
        "away_bat_8_id": batting_slot(away_lineup, 7),
        "away_bat_9_id": batting_slot(away_lineup, 8),
    }


def load_existing_rows(out_path: Path) -> dict:
    """Return existing rows as a dict keyed by gamePk string."""
    if not out_path.exists():
        return {}
    with out_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["gamePk"]: row for row in reader}


def merge_row(existing: dict, new: dict) -> dict:
    """Update existing row with non-empty values from new row."""
    merged = dict(existing)
    for key, value in new.items():
        if value != "":
            merged[key] = value
    return merged


def main() -> int:
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{target_date.replace('-', '_')}_mlb_raw.csv"

    existing_rows = load_existing_rows(out_path)

    schedule = fetch_json(SCHEDULE_URL.format(date=target_date))
    dates = schedule.get("dates", [])
    games = dates[0].get("games", []) if dates else []

    rows_written = 0
    for game in games:
        detailed_state = safe_get(game, "status", "detailedState")
        if detailed_state not in {"Pre-Game", "Scheduled"}:
            continue

        game_pk = str(safe_get(game, "gamePk"))
        if not game_pk:
            continue

        live = fetch_json(LIVE_URL.format(game_pk=game_pk))
        new_row = build_row(game, live)
        new_row["gamePk"] = game_pk

        if game_pk in existing_rows:
            existing_rows[game_pk] = merge_row(existing_rows[game_pk], new_row)
        else:
            existing_rows[game_pk] = new_row

        rows_written += 1

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(out_path.as_posix())
    print(f"rows_written={rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
