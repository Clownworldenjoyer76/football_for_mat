#!/usr/bin/env python3
"""Pull ESPN Core NFL player props and write weekly category CSVs.

Outputs are written under:
  docs/win/football/nfl/prop_engine/output/{season}/week_{week}_props/

Each CSV is one row per game/player and includes:
  game_date
  game_id
  player_name
  player_id
  requested prop columns
  ESPN odds total

ESPN occasionally returns duplicate prop rows. Exact duplicate offers are collapsed.

If a requested market has multiple distinct offers for the same player,
the target values are retained as pipe-delimited lists in the same order.

For selection-style markets that have no numeric target, such as touchdown
scorer markets, the market column is written as AVAILABLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SEASON_TYPE = 2
MAX_REGULAR_WEEKS = 18
HTTP_RETRIES = 4
HTTP_TIMEOUT = 45
WORKERS = max(1, min(int(os.getenv("NFL_PROP_WORKERS", "8")), 16))
TARGET_WEEK_GRACE = timedelta(hours=5)

PROP_ENGINE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROP_ENGINE_DIR / "output"

BASE_COLUMNS = [
    "game_date",
    "game_id",
    "player_name",
    "player_id",
]

CATEGORY_CONFIG = {
    "passing": {
        "filename": "passing_props.csv",
        "columns": [
            "Total Passing Yards",
            "Total Pass Completions",
            "Total Passing Attempts",
            "Total Passing Touchdowns",
            "Total Passing Interceptions",
        ],
        "markets": {
            "Total Passing Yards (incl. overtime)": "Total Passing Yards",
            "Total Pass Completions (incl. overtime)": "Total Pass Completions",
            "Total Passing Attempts (incl. overtime)": "Total Passing Attempts",
            "Total Passing Touchdowns (incl. overtime)": "Total Passing Touchdowns",
            "Total Passing Interceptions (incl. overtime)": "Total Passing Interceptions",
        },
    },
    "rushing": {
        "filename": "rushing_props.csv",
        "columns": [
            "Total Carries",
            "Total Rushing Yards",
            "Longest Rush",
        ],
        "markets": {
            "Total Carries (incl. overtime)": "Total Carries",
            "Total Rushing Yards (incl. overtime)": "Total Rushing Yards",
            "Longest Rush (incl. overtime)": "Longest Rush",
        },
    },
    "receiving": {
        "filename": "receiving_props.csv",
        "columns": [
            "Total Receiving Yards",
            "Total Receptions",
            "Longest Reception",
            "Receiving Yards",
        ],
        "markets": {
            "Total Receiving Yards (incl. overtime)": "Total Receiving Yards",
            "Total Receptions (incl. overtime)": "Total Receptions",
            "Longest Reception (incl. overtime)": "Longest Reception",
            "Receiving Yards Milestones": "Receiving Yards",
        },
    },
    "combo": {
        "filename": "combo_props.csv",
        "columns": [
            "Total Passing Plus Rushing Yards",
            "Total Rushing Plus Receiving Yards",
        ],
        "markets": {
            "Total Passing Plus Rushing Yards (incl. overtime)": "Total Passing Plus Rushing Yards",
            "Total Rushing Plus Receiving Yards (incl. overtime)": "Total Rushing Plus Receiving Yards",
        },
    },
    "tds": {
        "filename": "td_props.csv",
        "columns": [
            "Anytime Touchdown Scorer",
            "First Touchdown Scorer",
            "Last Touchdown Scorer",
            "First Team Touchdown Scorer",
            "Player to score 2+ touchdowns",
            "Player to score 3+ touchdowns",
        ],
        "markets": {
            "Anytime Touchdown Scorer": "Anytime Touchdown Scorer",
            "First Touchdown Scorer": "First Touchdown Scorer",
            "Last Touchdown Scorer": "Last Touchdown Scorer",
            "First Team Touchdown Scorer": "First Team Touchdown Scorer",
            "Player to score 2 or more touchdowns": "Player to score 2+ touchdowns",
            "Player to score 3 or more touchdowns": "Player to score 3+ touchdowns",
        },
    },
    "1sthalf": {
        "filename": "1sthalf_props.csv",
        "columns": [
            "Total Passing Yards",
            "Total Receiving Yards",
            "Total Rushing Yards",
            "Touchdown Scorer",
        ],
        "markets": {
            "1st Half Total Passing Yards": "Total Passing Yards",
            "1st Half Total Receiving Yards": "Total Receiving Yards",
            "1st Half Total Rushing Yards": "Total Rushing Yards",
            "1st Half Touchdown Scorer": "Touchdown Scorer",
        },
    },
    "1stquarter": {
        "filename": "1stquarter_props.csv",
        "columns": [
            "Total Passing Yards",
            "Total Receiving Yards",
            "Total Rushing Yards",
        ],
        "markets": {
            "1st Quarter Total Passing Yards": "Total Passing Yards",
            "1st Quarter Total Receiving Yards": "Total Receiving Yards",
            "1st Quarter Total Rushing Yards": "Total Rushing Yards",
        },
    },
    "defense": {
        "filename": "defense_props.csv",
        "columns": [
            "Total Tackles",
            "Total Assists",
            "Total Tackles Plus Assists",
            "Total Sacks",
        ],
        "markets": {
            "Total Tackles (incl. overtime)": "Total Tackles",
            "Total Assists (incl. overtime)": "Total Assists",
            "Total Tackles Plus Assists (incl. overtime)": "Total Tackles Plus Assists",
            "Total Sacks (incl. overtime)": "Total Sacks",
        },
    },
    "kicking": {
        "filename": "kicking_props.csv",
        "columns": [
            "Total Kicking Points",
            "Total Field Goals Made",
            "Total Extra Points Made",
        ],
        "markets": {
            "Total Kicking Points (incl. overtime)": "Total Kicking Points",
            "Total Field Goals Made (incl. overtime)": "Total Field Goals Made",
            "Total Extra Points Made (incl. overtime)": "Total Extra Points Made",
        },
    },
}


def secure_ref(value: object) -> str:
    return str(value or "").strip().replace("http://", "https://", 1)


def clean_value(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def http_get_json(url: str) -> object:
    last_error: Exception | None = None

    for attempt in range(1, HTTP_RETRIES + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "nfl-prop-odds-espn/1.0",
            },
        )

        try:
            with urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))

        except (
            HTTPError,
            URLError,
            json.JSONDecodeError,
            TimeoutError,
        ) as exc:
            last_error = exc

            if (
                isinstance(exc, HTTPError)
                and exc.code
                not in {
                    408,
                    425,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
            ):
                raise

            if attempt < HTTP_RETRIES:
                time.sleep(
                    min(
                        2 ** (attempt - 1),
                        8,
                    )
                )

    raise RuntimeError(
        f"ESPN request failed: {url}: {last_error}"
    )


def parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def week_events_url(
    season: int,
    week: int,
) -> str:
    return (
        f"{ESPN_CORE_BASE}/seasons/{season}/types/"
        f"{SEASON_TYPE}/weeks/{week}/events"
        "?limit=100&lang=en&region=us"
    )


def event_id_from_ref(ref: str) -> str:
    match = re.search(
        r"/events/(\d+)",
        ref,
    )

    return match.group(1) if match else ""


def athlete_id_from_ref(ref: str) -> str:
    match = re.search(
        r"/athletes/(\d+)",
        ref,
    )

    return match.group(1) if match else ""


def event_odds_url(
    event_id: str,
) -> str:
    return (
        f"{ESPN_CORE_BASE}/events/{event_id}/"
        f"competitions/{event_id}/odds"
        "?lang=en&region=us"
    )


def fetch_week_events(
    season: int,
    week: int,
) -> list[dict]:
    payload = http_get_json(
        week_events_url(
            season,
            week,
        )
    )

    refs = (
        payload.get("items", [])
        if isinstance(payload, dict)
        else []
    )

    event_refs = [
        secure_ref(
            item.get("$ref")
        )
        for item in refs
        if isinstance(item, dict)
        and item.get("$ref")
    ]

    events: list[dict] = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:
        futures = {
            executor.submit(
                http_get_json,
                ref,
            ): ref
            for ref in event_refs
        }

        for future in as_completed(
            futures
        ):
            ref = futures[future]

            try:
                event = future.result()
            except Exception as exc:
                print(
                    f"WARN event fetch failed: "
                    f"{ref}: {exc}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(
                event,
                dict,
            ):
                continue

            event_id = str(
                event.get("id")
                or event_id_from_ref(ref)
            ).strip()

            if not event_id:
                continue

            events.append(
                {
                    "event_id": event_id,
                    "date": str(
                        event.get(
                            "date",
                            "",
                        )
                    ).strip(),
                    "event": event,
                }
            )

    events.sort(
        key=lambda row: (
            row["date"],
            row["event_id"],
        )
    )

    return events


def choose_target_week(
    season: int,
) -> int:
    now = datetime.now(
        timezone.utc
    )

    threshold = (
        now
        - TARGET_WEEK_GRACE
    )

    last_week_with_events = 1

    for week in range(
        1,
        MAX_REGULAR_WEEKS + 1,
    ):
        events = fetch_week_events(
            season,
            week,
        )

        if not events:
            continue

        last_week_with_events = week

        for event in events:
            event_dt = parse_datetime(
                event.get("date")
            )

            if (
                event_dt
                and event_dt >= threshold
            ):
                return week

    return last_week_with_events


def select_prop_ref(
    odds_payload: object,
) -> str:
    if not isinstance(
        odds_payload,
        dict,
    ):
        return ""

    items = [
        item
        for item in odds_payload.get(
            "items",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    ]

    ordered = sorted(
        items,
        key=lambda item: (
            0
            if (
                str(
                    (
                        item.get("provider")
                        or {}
                    ).get(
                        "id",
                        "",
                    )
                )
                == "100"
                or str(
                    (
                        item.get("provider")
                        or {}
                    ).get(
                        "name",
                        "",
                    )
                )
                == "DraftKings"
            )
            else 1
        ),
    )

    for item in ordered:
        prop_bets = item.get(
            "propBets"
        )

        if (
            isinstance(
                prop_bets,
                dict,
            )
            and prop_bets.get("$ref")
        ):
            ref = secure_ref(
                prop_bets["$ref"]
            )

            separator = (
                "&"
                if "?" in ref
                else "?"
            )

            return (
                f"{ref}"
                f"{separator}"
                f"limit=1000"
            )

    return ""


def fetch_game_props(
    event: dict,
) -> dict:
    event_id = event[
        "event_id"
    ]

    odds = http_get_json(
        event_odds_url(
            event_id
        )
    )

    prop_ref = select_prop_ref(
        odds
    )

    if not prop_ref:
        return {
            **event,
            "props": [],
        }

    payload = http_get_json(
        prop_ref
    )

    props = (
        payload.get(
            "items",
            [],
        )
        if isinstance(
            payload,
            dict,
        )
        else []
    )

    return {
        **event,
        "props": [
            row
            for row in props
            if isinstance(
                row,
                dict,
            )
        ],
    }


def fetch_all_game_props(
    events: list[dict],
) -> list[dict]:
    results: list[dict] = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:
        futures = {
            executor.submit(
                fetch_game_props,
                event,
            ): event
            for event in events
        }

        for future in as_completed(
            futures
        ):
            event = futures[future]

            try:
                results.append(
                    future.result()
                )

            except Exception as exc:
                print(
                    f"WARN props fetch failed: "
                    f"game_id="
                    f"{event['event_id']}: "
                    f"{exc}",
                    file=sys.stderr,
                )

                results.append(
                    {
                        **event,
                        "props": [],
                    }
                )

    results.sort(
        key=lambda row: (
            row["date"],
            row["event_id"],
        )
    )

    return results


def resolve_athletes(
    game_props: list[dict],
) -> dict[str, str]:
    refs: dict[
        str,
        str,
    ] = {}

    for game in game_props:
        for prop in game["props"]:
            athlete = prop.get(
                "athlete"
            )

            if not isinstance(
                athlete,
                dict,
            ):
                continue

            ref = secure_ref(
                athlete.get(
                    "$ref"
                )
            )

            athlete_id = (
                athlete_id_from_ref(
                    ref
                )
            )

            if (
                ref
                and athlete_id
            ):
                refs[
                    athlete_id
                ] = ref

    names: dict[
        str,
        str,
    ] = {}

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:
        futures = {
            executor.submit(
                http_get_json,
                ref,
            ): athlete_id
            for (
                athlete_id,
                ref,
            ) in refs.items()
        }

        for future in as_completed(
            futures
        ):
            athlete_id = futures[
                future
            ]

            try:
                payload = (
                    future.result()
                )

            except Exception as exc:
                print(
                    f"WARN athlete fetch failed: "
                    f"athlete_id={athlete_id}: "
                    f"{exc}",
                    file=sys.stderr,
                )

                names[
                    athlete_id
                ] = ""

                continue

            if isinstance(
                payload,
                dict,
            ):
                names[
                    athlete_id
                ] = str(
                    payload.get(
                        "fullName"
                    )
                    or payload.get(
                        "displayName"
                    )
                    or payload.get(
                        "shortName"
                    )
                    or ""
                ).strip()

            else:
                names[
                    athlete_id
                ] = ""

    return names


def current_prop_value(
    prop: dict,
) -> str:
    current = prop.get(
        "current"
    )

    if isinstance(
        current,
        dict,
    ):
        target = current.get(
            "target"
        )

        if isinstance(
            target,
            dict,
        ):
            value = target.get(
                "value"
            )

            if value not in (
                None,
                "",
            ):
                return clean_value(
                    value
                )

            display = target.get(
                "displayValue"
            )

            if display not in (
                None,
                "",
            ):
                return clean_value(
                    display
                )

        elif target not in (
            None,
            "",
        ):
            return clean_value(
                target
            )

    return "AVAILABLE"


def current_prop_odds(
    prop: dict,
) -> dict[str, str]:
    odds = prop.get(
        "odds"
    )

    if not isinstance(
        odds,
        dict,
    ):
        return {
            "american": "",
            "decimal": "",
            "fraction": "",
            "total": "",
        }

    return {
        "american": clean_value(
            odds.get(
                "american"
            )
        ),
        "decimal": clean_value(
            odds.get(
                "decimal"
            )
        ),
        "fraction": clean_value(
            odds.get(
                "fraction"
            )
        ),
        "total": clean_value(
            odds.get(
                "total"
            )
        ),
    }


def game_date(
    value: str,
) -> str:
    parsed = parse_datetime(
        value
    )

    return (
        parsed.date().isoformat()
        if parsed
        else value
    )


def offer_sort_key(
    offer: tuple[
        str,
        str,
        str,
        str,
        str,
    ],
) -> tuple[
    int,
    float | str,
    str,
    str,
    str,
    str,
]:
    target = offer[0]

    try:
        return (
            0,
            float(target),
            offer[1],
            offer[2],
            offer[3],
            offer[4],
        )

    except ValueError:
        return (
            1,
            target,
            offer[1],
            offer[2],
            offer[3],
            offer[4],
        )


def add_offer(
    offers: dict[
        str,
        list[
            tuple[
                str,
                str,
                str,
                str,
                str,
            ]
        ],
    ],
    market_column: str,
    target: str,
    odds: dict[str, str],
) -> None:
    offer = (
        target,
        odds["american"],
        odds["decimal"],
        odds["fraction"],
        odds["total"],
    )

    bucket = offers.setdefault(
        market_column,
        [],
    )

    if offer not in bucket:
        bucket.append(
            offer
        )


def category_fieldnames(
    category: str,
) -> list[str]:
    fieldnames = list(
        BASE_COLUMNS
    )

    for column in CATEGORY_CONFIG[
        category
    ]["columns"]:
        fieldnames.append(
            column
        )

    return fieldnames


def build_category_rows(
    category: str,
    game_props: list[dict],
    athlete_names: dict[
        str,
        str,
    ],
) -> list[dict]:
    config = CATEGORY_CONFIG[
        category
    ]

    market_map: dict[
        str,
        str,
    ] = config["markets"]

    columns: list[str] = (
        config["columns"]
    )

    rows: dict[
        tuple[
            str,
            str,
        ],
        dict,
    ] = {}

    offers_by_player: dict[
        tuple[
            str,
            str,
        ],
        dict[
            str,
            list[
                tuple[
                    str,
                    str,
                    str,
                    str,
                    str,
                ]
            ],
        ],
    ] = {}

    for game in game_props:
        event_id = game[
            "event_id"
        ]

        date_value = game_date(
            game["date"]
        )

        for prop in game["props"]:
            market_type = prop.get(
                "type"
            )

            if not isinstance(
                market_type,
                dict,
            ):
                continue

            market_name = str(
                market_type.get(
                    "name",
                    "",
                )
            ).strip()

            output_column = (
                market_map.get(
                    market_name
                )
            )

            if not output_column:
                continue

            athlete = prop.get(
                "athlete"
            )

            if not isinstance(
                athlete,
                dict,
            ):
                continue

            athlete_ref = (
                secure_ref(
                    athlete.get(
                        "$ref"
                    )
                )
            )

            player_id = (
                athlete_id_from_ref(
                    athlete_ref
                )
            )

            if not player_id:
                continue

            key = (
                event_id,
                player_id,
            )

            if key not in rows:
                row = {
                    "game_date": date_value,
                    "game_id": event_id,
                    "player_name": (
                        athlete_names.get(
                            player_id,
                            "",
                        )
                    ),
                    "player_id": player_id,
                }

                for column in columns:
                    row[column] = ""

                rows[
                    key
                ] = row

            target = (
                current_prop_value(
                    prop
                )
            )

            odds = (
                current_prop_odds(
                    prop
                )
            )

            player_offers = (
                offers_by_player.setdefault(
                    key,
                    {},
                )
            )

            add_offer(
                player_offers,
                output_column,
                target,
                odds,
            )

    for (
        key,
        market_offers,
    ) in offers_by_player.items():
        row = rows[key]

        for (
            market_column,
            offers,
        ) in market_offers.items():
            ordered = sorted(
                offers,
                key=offer_sort_key,
            )

            row[
                market_column
            ] = "|".join(
                offer[0]
                for offer in ordered
            )

    return sorted(
        rows.values(),
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_id"
            ],
            row[
                "player_name"
            ],
            row[
                "player_id"
            ],
        ),
    )


def write_csv(
    path: Path,
    category: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        category_fieldnames(
            category
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull ESPN NFL weekly player props."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=int(
            os.getenv(
                "NFL_SEASON",
                "2026",
            )
        ),
        help=(
            "NFL season year "
            "(default: NFL_SEASON or 2026)"
        ),
    )

    parser.add_argument(
        "--week",
        type=int,
        default=(
            int(
                os.getenv(
                    "NFL_WEEK",
                    "0",
                )
            )
            or None
        ),
        help=(
            "Regular-season week. "
            "If omitted, the current/upcoming "
            "week is detected."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    season = args.season

    week = (
        args.week
        or choose_target_week(
            season
        )
    )

    if not (
        1
        <= week
        <= MAX_REGULAR_WEEKS
    ):
        raise ValueError(
            f"Week must be between "
            f"1 and "
            f"{MAX_REGULAR_WEEKS}: "
            f"{week}"
        )

    events = fetch_week_events(
        season,
        week,
    )

    if not events:
        raise RuntimeError(
            f"No ESPN NFL events found for "
            f"season={season} "
            f"week={week}"
        )

    games = fetch_all_game_props(
        events
    )

    athlete_names = (
        resolve_athletes(
            games
        )
    )

    week_root = (
        OUTPUT_ROOT
        / str(season)
        / f"week_{week}_props"
    )

    total_props = sum(
        len(
            game["props"]
        )
        for game in games
    )

    for (
        category,
        config,
    ) in CATEGORY_CONFIG.items():
        rows = (
            build_category_rows(
                category,
                games,
                athlete_names,
            )
        )

        output_path = (
            week_root
            / category
            / config["filename"]
        )

        write_csv(
            output_path,
            category,
            rows,
        )

        print(
            f"{category}: "
            f"{len(rows)} rows -> "
            f"{output_path}"
        )

    print(
        f"season={season}"
    )

    print(
        f"week={week}"
    )

    print(
        f"games={len(events)}"
    )

    print(
        f"raw_prop_rows="
        f"{total_props}"
    )

    print(
        f"output_root="
        f"{week_root}"
    )


if __name__ == "__main__":
    main()
