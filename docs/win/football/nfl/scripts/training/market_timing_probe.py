#!/usr/bin/env python3
"""Probe ESPN Core NFL opening/current market timing capability without API keys."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path("docs/win/football/nfl")
OUTPUT = BASE_DIR / "training" / "market_timing_provider_probe.json"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SEASON = 2025
SEASON_TYPE = 2
WEEK = 1
TARGET_HOME = "Philadelphia Eagles"
TARGET_AWAY = "Dallas Cowboys"
HTTP_TIMEOUT = 45
HTTP_RETRIES = 4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def secure_ref(value: object) -> str:
    return str(value or "").replace("http://", "https://", 1)


def request_json(url: str) -> tuple[int | None, object, str]:
    last_error = ""
    for attempt in range(1, HTTP_RETRIES + 1):
        req = Request(url, headers={"User-Agent": "nfl-market-timing-espn-probe/1.0"})
        try:
            with urlopen(req, timeout=HTTP_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                return response.status, json.loads(body), "ok"
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = body or str(exc)
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                return exc.code, {"error": last_error[:500]}, "http_error"
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:
            last_error = str(exc)
        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))
    return None, {"error": last_error}, "request_error"


def event_id_from_ref(ref: str) -> str:
    return ref.split("/events/", 1)[1].split("?", 1)[0].split("/", 1)[0] if "/events/" in ref else ""


def find_target_event() -> tuple[str, dict | None]:
    url = f"{ESPN_CORE}/seasons/{SEASON}/types/{SEASON_TYPE}/weeks/{WEEK}/events?limit=100&lang=en&region=us"
    _, payload, _ = request_json(url)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = secure_ref(item.get("$ref"))
        if not ref:
            continue
        _, event, _ = request_json(ref)
        if not isinstance(event, dict):
            continue
        name = str(event.get("name", ""))
        if TARGET_HOME in name and TARGET_AWAY in name:
            return str(event.get("id") or event_id_from_ref(ref)), event
    return "", None


def nested(data: object, *keys: str) -> object:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def main() -> int:
    event_id, event = find_target_event()
    report = {
        "probe": "nfl_market_timing_espn_capability",
        "created_utc": now_iso(),
        "source": "espn_core",
        "target_game": {"away": TARGET_AWAY, "home": TARGET_HOME},
        "target_event_found": bool(event_id),
        "embedded_open_available": False,
        "embedded_current_available": False,
        "movement_endpoints": [],
        "asof_backfill_capability": False,
        "asof_backfill_reason": "target_event_not_found",
    }

    if not event_id:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    report["target_event_id"] = event_id
    report["target_event_date"] = event.get("date") if isinstance(event, dict) else None

    odds_url = f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/odds?lang=en&region=us"
    status, payload, kind = request_json(odds_url)
    report["odds_collection_http_status"] = status
    report["odds_collection_result_kind"] = kind

    items = payload.get("items", []) if isinstance(payload, dict) else []
    item = next((x for x in items if isinstance(x, dict) and str(nested(x, "provider", "name")) == "DraftKings"), None)
    if item is None and items:
        item = next((x for x in items if isinstance(x, dict)), None)

    if isinstance(item, dict):
        provider_id = str(nested(item, "provider", "id") or "")
        report["provider_id"] = provider_id
        report["provider"] = nested(item, "provider", "name")
        report["embedded_open_available"] = bool(item.get("open") or nested(item, "homeTeamOdds", "open") or nested(item, "awayTeamOdds", "open"))
        report["embedded_current_available"] = bool(item.get("current") or nested(item, "homeTeamOdds", "current") or nested(item, "awayTeamOdds", "current"))
        report["open_total"] = nested(item, "open", "total", "american")
        report["current_total"] = nested(item, "current", "total", "american")
        report["open_home_moneyline"] = nested(item, "homeTeamOdds", "open", "moneyLine", "american")
        report["current_home_moneyline"] = nested(item, "homeTeamOdds", "current", "moneyLine", "american")
        report["open_home_spread"] = nested(item, "homeTeamOdds", "open", "pointSpread", "american")
        report["current_home_spread"] = nested(item, "homeTeamOdds", "current", "pointSpread", "american")

        if provider_id:
            for history_index in (0, 1, 2):
                url = f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/odds/{provider_id}/history/{history_index}/movement?limit=100"
                h_status, h_payload, h_kind = request_json(url)
                report["movement_endpoints"].append({
                    "history_index": history_index,
                    "http_status": h_status,
                    "result_kind": h_kind,
                    "count": h_payload.get("count") if isinstance(h_payload, dict) else None,
                })

    if report["embedded_open_available"] and report["embedded_current_available"]:
        report["asof_backfill_reason"] = "embedded_open_and_current_available; timestamped_local_snapshots_required_for_time_series"
    else:
        report["asof_backfill_reason"] = "embedded_open_or_current_missing"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
