#docs/win/baseball/scripts/00_parsing/mlb_odds_pull.py

import requests
import os
import json
from pathlib import Path
from datetime import datetime, timezone


API_KEY = os.getenv("API_ODDS")

if not API_KEY:
    raise RuntimeError("API_ODDS environment variable is not set")


BASE_URL = "https://api.odds-api.io/v3"

SPORT = "baseball"
LEAGUE = "usa-mlb"
BOOKMAKER = "DraftKings"

today = datetime.now(timezone.utc).strftime("%Y_%m_%d")
path = f"docs/win/baseball/odds/{today}.json"


def get_json(endpoint, params):
    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        print(f"{endpoint} error: {response.status_code}")
        print(response.text)
        raise SystemExit(1)

    return response.json()


def fetch_events():
    events = []
    skip = 0
    limit = 100

    while True:
        batch = get_json(
            "/events",
            {
                "apiKey": API_KEY,
                "sport": SPORT,
                "league": LEAGUE,
                "status": "pending,live",
                "bookmaker": BOOKMAKER,
                "limit": limit,
                "skip": skip,
            },
        )

        if not isinstance(batch, list):
            print("Unexpected /events response:")
            print(json.dumps(batch, indent=2))
            raise SystemExit(1)

        events.extend(batch)

        if len(batch) < limit:
            break

        skip += limit

    return events


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_odds(event_ids):
    odds = []

    for batch_ids in chunks(event_ids, 10):
        batch = get_json(
            "/odds/multi",
            {
                "apiKey": API_KEY,
                "eventIds": ",".join(str(event_id) for event_id in batch_ids),
                "bookmakers": BOOKMAKER,
            },
        )

        if not isinstance(batch, list):
            print("Unexpected /odds/multi response:")
            print(json.dumps(batch, indent=2))
            raise SystemExit(1)

        odds.extend(batch)

    return odds


def to_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def market_key(name):
    normalized = str(name or "").strip().lower()

    if normalized == "ml":
        return "h2h"

    if normalized in {"spread", "spreads", "asian handicap", "handicap"}:
        return "spreads"

    if normalized in {"totals", "total", "over/under", "over under"}:
        return "totals"

    return None


def convert_market(event, market):
    key = market_key(market.get("name"))

    if key not in {"h2h", "spreads", "totals"}:
        return None

    converted = {
        "key": key,
        "last_update": market.get("updatedAt"),
        "outcomes": [],
    }

    odds_rows = market.get("odds") or []

    for row in odds_rows:
        if key == "h2h":
            home_price = to_float(row.get("home"))
            away_price = to_float(row.get("away"))

            if home_price is not None:
                converted["outcomes"].append(
                    {
                        "name": event.get("home"),
                        "price": home_price,
                    }
                )

            if away_price is not None:
                converted["outcomes"].append(
                    {
                        "name": event.get("away"),
                        "price": away_price,
                    }
                )

        elif key == "spreads":
            point = to_float(row.get("hdp"))
            home_price = to_float(row.get("home"))
            away_price = to_float(row.get("away"))

            if home_price is not None:
                converted["outcomes"].append(
                    {
                        "name": event.get("home"),
                        "price": home_price,
                        "point": point,
                    }
                )

            if away_price is not None:
                converted["outcomes"].append(
                    {
                        "name": event.get("away"),
                        "price": away_price,
                        "point": -point if point is not None else None,
                    }
                )

        elif key == "totals":
            point = to_float(row.get("hdp"))
            over_price = to_float(row.get("over"))
            under_price = to_float(row.get("under"))

            if over_price is not None:
                converted["outcomes"].append(
                    {
                        "name": "Over",
                        "price": over_price,
                        "point": point,
                    }
                )

            if under_price is not None:
                converted["outcomes"].append(
                    {
                        "name": "Under",
                        "price": under_price,
                        "point": point,
                    }
                )

    if not converted["outcomes"]:
        return None

    return converted


def convert_event(event):
    bookmakers = event.get("bookmakers") or {}
    bookmaker_markets = bookmakers.get(BOOKMAKER, [])

    converted_markets = []

    for market in bookmaker_markets:
        converted_market = convert_market(event, market)
        if converted_market:
            converted_markets.append(converted_market)

    if not converted_markets:
        return None

    last_update_values = [
        market.get("last_update")
        for market in converted_markets
        if market.get("last_update")
    ]

    bookmaker = {
        "key": "draftkings",
        "title": BOOKMAKER,
        "last_update": max(last_update_values) if last_update_values else None,
        "markets": converted_markets,
    }

    return {
        "id": str(event.get("id")),
        "sport_key": "baseball_mlb",
        "sport_title": "MLB",
        "commence_time": event.get("date"),
        "home_team": event.get("home"),
        "away_team": event.get("away"),
        "bookmakers": [bookmaker],
    }


events = fetch_events()

if not events:
    print("No MLB events found with DraftKings odds.")
    data = []
else:
    event_ids = [event["id"] for event in events if event.get("id") is not None]
    raw_odds = fetch_odds(event_ids)

    data = []

    for event in raw_odds:
        converted = convert_event(event)
        if converted:
            data.append(converted)

Path(path).parent.mkdir(parents=True, exist_ok=True)

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved {path}")
print(f"Events found: {len(events)}")
print(f"Events with converted DraftKings odds: {len(data)}")
