#!/usr/bin/env python3
"""Direct ESPN Core probe for embedded NFL open/current odds and movement collections."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SEASON = 2025
SEASON_TYPE = 2
WEEK = 1
TARGET_HOME = "Philadelphia Eagles"
TARGET_AWAY = "Dallas Cowboys"
OUTPUT = Path("docs/win/football/nfl/training/market_timing_direct_movement_probe.json")
HTTP_RETRIES = 4


def get_json(url: str):
    last_error = ""
    for attempt in range(1, HTTP_RETRIES + 1):
        req = Request(url, headers={"User-Agent": "nfl-market-timing-direct-espn/1.0"})
        try:
            with urlopen(req, timeout=45) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace") or str(exc)
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                return exc.code, {"error": last_error[:500]}
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:
            last_error = str(exc)
        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))
    return None, {"error": last_error}


def secure_ref(value: object) -> str:
    return str(value or "").replace("http://", "https://", 1)


def nested(data: object, *keys: str) -> object:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def find_event_id() -> str:
    _, payload = get_json(f"{ESPN_CORE}/seasons/{SEASON}/types/{SEASON_TYPE}/weeks/{WEEK}/events?limit=100&lang=en&region=us")
    items = payload.get("items", []) if isinstance(payload, dict) else []
    for item in items:
        ref = secure_ref(item.get("$ref")) if isinstance(item, dict) else ""
        if not ref:
            continue
        _, event = get_json(ref)
        if isinstance(event, dict):
            name = str(event.get("name", ""))
            if TARGET_HOME in name and TARGET_AWAY in name:
                return str(event.get("id", "")).strip()
    return ""


def main():
    event_id = find_event_id()
    report = {
        "probe": "finished_2025_nfl_direct_espn_open_current",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "provider": None,
        "provider_id": None,
        "embedded_open_present": False,
        "embedded_current_present": False,
        "movement_counts": {},
        "reason": "event_not_found",
    }

    if event_id:
        status, payload = get_json(f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/odds?lang=en&region=us")
        report["http_status"] = status
        items = payload.get("items", []) if isinstance(payload, dict) else []
        item = next((x for x in items if isinstance(x, dict) and str(nested(x, "provider", "name")) == "DraftKings"), None)
        if item is None and items:
            item = next((x for x in items if isinstance(x, dict)), None)
        if isinstance(item, dict):
            report["provider"] = nested(item, "provider", "name")
            report["provider_id"] = str(nested(item, "provider", "id") or "")
            report["embedded_open_present"] = bool(item.get("open") or nested(item, "homeTeamOdds", "open") or nested(item, "awayTeamOdds", "open"))
            report["embedded_current_present"] = bool(item.get("current") or nested(item, "homeTeamOdds", "current") or nested(item, "awayTeamOdds", "current"))
            provider_id = report["provider_id"]
            if provider_id:
                for idx in (0, 1, 2):
                    h_status, h_payload = get_json(f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/odds/{provider_id}/history/{idx}/movement?limit=100")
                    report["movement_counts"][str(idx)] = {
                        "http_status": h_status,
                        "count": h_payload.get("count") if isinstance(h_payload, dict) else None,
                    }
            report["reason"] = "embedded_open_current_available; movement_endpoint_counts_reported_separately"
        else:
            report["reason"] = "odds_item_not_available"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
