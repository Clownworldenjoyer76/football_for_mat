#!/usr/bin/env python3
"""Pull current NFL odds from ESPN Core and preserve point-in-time snapshots."""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_reporter import PipelineReporter
BASE_DIR = Path('docs/win/football/nfl')
ODDS_DIR = BASE_DIR / '00_intake' / 'odds'
RAW_ODDS_DIR = ODDS_DIR / 'raw'
SNAPSHOT_DIR = ODDS_DIR / 'snapshots'
RAW_SNAPSHOT_DIR = RAW_ODDS_DIR / 'snapshots'
ERROR_DIR = BASE_DIR / 'errors' / '00_intake'
REPORT_ROOT = BASE_DIR / 'errors'
LOG_FILE = ERROR_DIR / 'pull_odds.txt'
ESPN_CORE_BASE = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl'
HTTP_RETRIES = 4
HTTP_TIMEOUT = 45
WORKERS = max(1, min(int(os.getenv('NFL_ODDS_WORKERS', '8')), 16))
TARGET_WEEK_GRACE = timedelta(hours=5)
OUTPUT_COLUMNS = ['snapshot_id', 'snapshot_fetched_at', 'game_id', 'commence_time', 'home_team', 'away_team', 'bookmaker', 'market_type', 'bet_side', 'line', 'odds_american', 'odds_decimal', 'last_update', 'home_moneyline_american', 'away_moneyline_american', 'home_spread', 'away_spread', 'home_spread_american', 'away_spread_american', 'total', 'over_american', 'under_american']
EXPECTED_MARKET_SIDES = {('h2h', 'home'), ('h2h', 'away'), ('spreads', 'home'), ('spreads', 'away'), ('totals', 'over'), ('totals', 'under')}

class OddsError(RuntimeError):
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', required=True, type=int)
    parser.add_argument('--season-type', required=True, type=int, choices=(1, 2, 3))
    parser.add_argument('--week', required=True, type=int)
    parser.add_argument('--sportsbook', required=True)
    args = parser.parse_args()
    if not 2000 <= args.season <= 2100:
        parser.error('--season must be between 2000 and 2100')
    if not 1 <= args.week <= 25:
        parser.error('--week must be between 1 and 25')
    args.sportsbook = str(args.sportsbook).strip()
    if not args.sportsbook:
        parser.error('--sportsbook must not be blank')
    return args

def ensure_directories() -> None:
    for directory in (RAW_ODDS_DIR, SNAPSHOT_DIR, RAW_SNAPSHOT_DIR, ODDS_DIR, ERROR_DIR):
        directory.mkdir(parents=True, exist_ok=True)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat()

def log(message: str) -> None:
    with LOG_FILE.open('a', encoding='utf-8') as handle:
        handle.write(f'[{utc_now_iso()}] {message}\n')

def fail(message: str) -> None:
    log(f'ERROR: {message}')
    raise OddsError(message)

def secure_ref(value: object) -> str:
    return str(value or '').strip().replace('http://', 'https://', 1)

def http_get_json(url: str) -> object:
    last_error = None
    for attempt in range(1, HTTP_RETRIES + 1):
        request = Request(url, headers={'User-Agent': 'nfl-pull-odds-espn/1.0'})
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            last_error = f'HTTP {exc.code}: {body[:500]}'
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise OddsError(last_error) from exc
        except URLError as exc:
            last_error = f'network error: {exc.reason}'
        except json.JSONDecodeError as exc:
            last_error = f'JSON parse error: {exc}'
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
        if attempt < HTTP_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 8))
    raise OddsError(f'ESPN request failed after {HTTP_RETRIES} attempts: {last_error}')

def parse_espn_datetime(value: object) -> datetime | None:
    text = str(value or '').strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

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
        return ''
    if number.is_integer():
        return str(int(number))
    return str(number)

def clean_american(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ''
    return str(int(round(number)))

def nested(data: object, *keys: str) -> object:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

def value_from_display_object(value: object) -> object:
    if isinstance(value, dict):
        for key in ('american', 'alternateDisplayValue', 'value'):
            if value.get(key) not in (None, ''):
                return value.get(key)
    return value

def team_id_from_ref(ref: object) -> str:
    text = secure_ref(ref)
    match = re.search('/teams/([^/?]+)', text)
    return match.group(1).strip() if match else ''

def ref_id(ref: object, segment: str) -> str:
    text = secure_ref(ref)
    match = re.search(f'/{re.escape(segment)}/([^/?]+)', text)
    return match.group(1).strip() if match else ''

def week_events_url(season: int, season_type: int, week: int) -> str:
    return f'{ESPN_CORE_BASE}/seasons/{season}/types/{season_type}/weeks/{week}/events?limit=100&lang=en&region=us'

def odds_url(event_id: str) -> str:
    return f'{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}/odds?lang=en&region=us'

def get_event_id(event_ref: str) -> str:
    return ref_id(event_ref, 'events')

def discover_event_refs(season: int, season_type: int, week: int) -> list[str]:
    payload = http_get_json(week_events_url(season, season_type, week))
    if not isinstance(payload, dict):
        fail('Invalid ESPN week-events response')
    items = payload.get('items')
    if not isinstance(items, list):
        fail('ESPN week-events response has invalid items')
    refs: list[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            fail(f'ESPN week-events item {index} is not an object')
        ref = secure_ref(item.get('$ref'))
        event_id = get_event_id(ref)
        if not ref or not event_id:
            fail(f'ESPN week-events item {index} has no usable event reference')
        if event_id in seen_ids:
            fail(f'Duplicate ESPN event reference for event_id={event_id}')
        refs.append(ref)
        seen_ids.add(event_id)
    if not refs:
        fail(f'No ESPN NFL event references returned for season={season}, season_type={season_type}, week={week}')
    return refs

def validate_event_context(event: dict[str, Any], season: int, season_type: int, week: int, event_id: str) -> None:
    event_season = ref_id(nested(event, 'season', '$ref'), 'seasons')
    event_type = ref_id(nested(event, 'seasonType', '$ref'), 'types')
    event_week = ref_id(nested(event, 'week', '$ref'), 'weeks')
    expected = (str(season), str(season_type), str(week))
    actual = (event_season, event_type, event_week)
    if actual != expected:
        fail(f'ESPN event {event_id} context mismatch: expected={expected} actual={actual}')

def fetch_event_record(event_ref: str, season: int, season_type: int, week: int) -> dict[str, Any]:
    event = http_get_json(event_ref)
    if not isinstance(event, dict):
        raise OddsError(f'Invalid ESPN event response: {event_ref}')
    ref_event_id = get_event_id(event_ref)
    event_id = str(event.get('id') or ref_event_id).strip()
    if not event_id:
        raise OddsError(f'ESPN event has blank ID: {event_ref}')
    if ref_event_id and event_id != ref_event_id:
        raise OddsError(f'ESPN event ID mismatch: ref={ref_event_id} response={event_id}')
    event_date = str(event.get('date', '')).strip()
    if parse_espn_datetime(event_date) is None:
        raise OddsError(f'ESPN event {event_id} has invalid date={event_date!r}')
    validate_event_context(event, season, season_type, week, event_id)
    return {'week': week, 'event_id': event_id, 'event_ref': event_ref, 'date': event_date, 'event': event}

def fetch_all_event_records(event_refs: list[str], season: int, season_type: int, week: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(fetch_event_record, ref, season, season_type, week): ref for ref in event_refs}
        for future in as_completed(future_map):
            ref = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append(f'ref={ref} error={type(exc).__name__}: {exc}')
    if errors:
        fail('One or more ESPN event requests failed: ' + ' | '.join(errors[:3]))
    if len(records) != len(event_refs):
        fail(f'Incomplete ESPN event collection: refs={len(event_refs)} records={len(records)}')
    event_ids = [record['event_id'] for record in records]
    if len(event_ids) != len(set(event_ids)):
        fail('Duplicate ESPN event IDs after event fetch')
    records.sort(key=lambda record: (record.get('date', ''), record['event_id']))
    return records

def bookmaker_key(value: object) -> str:
    return re.sub('[^a-z0-9]+', '', str(value or '').strip().lower())

def select_bookmaker_item(items: object, preferred_bookmaker: str) -> tuple[dict[str, Any] | None, str, bool]:
    if not isinstance(items, list):
        raise OddsError('ESPN odds response has invalid items')
    valid: list[tuple[dict[str, Any], str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise OddsError(f'ESPN odds item {index} is not an object')
        provider = item.get('provider')
        provider_name = str(provider.get('name', '')).strip() if isinstance(provider, dict) else ''
        if provider_name:
            valid.append((item, provider_name))
    preferred_key = bookmaker_key(preferred_bookmaker)
    for item, provider_name in valid:
        if bookmaker_key(provider_name) == preferred_key:
            return (item, provider_name, False)
    if valid:
        item, provider_name = valid[0]
        return (item, provider_name, True)
    return (None, '', False)

def fetch_odds_for_record(record: dict[str, Any], preferred_bookmaker: str) -> dict[str, Any] | None:
    event_id = record['event_id']
    payload = http_get_json(odds_url(event_id))
    if not isinstance(payload, dict):
        raise OddsError(f'Invalid ESPN odds response for event_id={event_id}')
    selected, provider_name, fallback_used = select_bookmaker_item(payload.get('items'), preferred_bookmaker)
    if selected is None:
        return None
    return {**record, 'odds_collection': payload, 'odds_item': selected, 'selected_bookmaker': provider_name, 'bookmaker_fallback': fallback_used}

def eligible_event_records(records: list[dict[str, Any]], now: datetime) -> tuple[list[dict[str, Any]], list[str]]:
    threshold = now - TARGET_WEEK_GRACE
    eligible: list[dict[str, Any]] = []
    excluded_ids: list[str] = []
    for record in records:
        event_time = parse_espn_datetime(record.get('date'))
        if event_time is None:
            fail(f"Event {record.get('event_id', '')} has invalid date")
        if event_time >= threshold:
            eligible.append(record)
        else:
            excluded_ids.append(record['event_id'])
    return (eligible, excluded_ids)

def fetch_odds_records(records: list[dict[str, Any]], preferred_bookmaker: str) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    missing_odds: list[str] = []
    fallbacks: list[dict[str, str]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(fetch_odds_for_record, record, preferred_bookmaker): record for record in records}
        for future in as_completed(future_map):
            record = future_map[future]
            try:
                value = future.result()
                if value is None:
                    missing_odds.append(record['event_id'])
                    continue
                results.append(value)
                if value.get('bookmaker_fallback'):
                    fallbacks.append({'event_id': value['event_id'], 'selected_bookmaker': value['selected_bookmaker']})
            except Exception as exc:
                errors.append(f"event_id={record['event_id']} error={type(exc).__name__}: {exc}")
    if errors:
        fail('One or more ESPN odds requests failed: ' + ' | '.join(errors[:3]))
    results.sort(key=lambda record: (record.get('date', ''), record['event_id']))
    missing_odds.sort()
    fallbacks.sort(key=lambda item: item['event_id'])
    return (results, missing_odds, fallbacks)

def team_name_from_payload(team_payload: object) -> str:
    if not isinstance(team_payload, dict):
        return ''
    for key in ('displayName', 'name', 'shortDisplayName', 'location'):
        text = str(team_payload.get(key, '')).strip()
        if text:
            return text
    return ''

def fetch_team_names(odds_records: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    refs: dict[str, str] = {}
    for record in odds_records:
        item = record['odds_item']
        for side in ('homeTeamOdds', 'awayTeamOdds'):
            ref = secure_ref(nested(item, side, 'team', '$ref'))
            team_id = team_id_from_ref(ref)
            if team_id and ref:
                refs[team_id] = ref
    names: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_map = {executor.submit(http_get_json, ref): (team_id, ref) for team_id, ref in refs.items()}
        for future in as_completed(future_map):
            team_id, ref = future_map[future]
            try:
                names[team_id] = team_name_from_payload(future.result())
            except Exception as exc:
                failures.append(f'team_id={team_id} ref={ref} error={type(exc).__name__}: {exc}')
    failures.sort()
    return (names, failures)

def fallback_team_names(event: dict[str, Any]) -> tuple[str, str]:
    name = str(event.get('name', '')).strip()
    if ' at ' in name:
        away, home = name.split(' at ', 1)
        return (home.strip(), away.strip())
    if ' vs ' in name:
        away, home = name.split(' vs ', 1)
        return (home.strip(), away.strip())
    return ('', '')

def enrich_names(record: dict[str, Any], team_names: dict[str, str]) -> tuple[str, str]:
    item = record['odds_item']
    home_ref = nested(item, 'homeTeamOdds', 'team', '$ref')
    away_ref = nested(item, 'awayTeamOdds', 'team', '$ref')
    home = team_names.get(team_id_from_ref(home_ref), '')
    away = team_names.get(team_id_from_ref(away_ref), '')
    if home and away:
        return (home, away)
    fallback_home, fallback_away = fallback_team_names(record.get('event', {}))
    return (home or fallback_home, away or fallback_away)

def market_values(item: dict[str, Any]) -> dict[str, str]:
    home_current_spread = value_from_display_object(nested(item, 'homeTeamOdds', 'current', 'pointSpread'))
    away_current_spread = value_from_display_object(nested(item, 'awayTeamOdds', 'current', 'pointSpread'))
    current_total = value_from_display_object(nested(item, 'current', 'total'))
    return {'home_moneyline_american': clean_american(nested(item, 'homeTeamOdds', 'moneyLine')), 'away_moneyline_american': clean_american(nested(item, 'awayTeamOdds', 'moneyLine')), 'home_spread': clean_number(home_current_spread if home_current_spread not in (None, '') else item.get('spread')), 'away_spread': clean_number(away_current_spread), 'home_spread_american': clean_american(nested(item, 'homeTeamOdds', 'spreadOdds')), 'away_spread_american': clean_american(nested(item, 'awayTeamOdds', 'spreadOdds')), 'total': clean_number(current_total if current_total not in (None, '') else item.get('overUnder')), 'over_american': clean_american(item.get('overOdds')), 'under_american': clean_american(item.get('underOdds'))}

def add_row(rows: list[dict[str, Any]], *, record: dict[str, Any], home: str, away: str, bookmaker: str, market_type: str, bet_side: str, line: object, american: object, decimal: object, current_fields: dict[str, str], snapshot_id: str, snapshot_fetched_at: str) -> None:
    row = {'snapshot_id': snapshot_id, 'snapshot_fetched_at': snapshot_fetched_at, 'game_id': record['event_id'], 'commence_time': record.get('date', ''), 'home_team': home, 'away_team': away, 'bookmaker': bookmaker, 'market_type': market_type, 'bet_side': bet_side, 'line': clean_number(line), 'odds_american': clean_american(american), 'odds_decimal': clean_number(decimal), 'last_update': snapshot_fetched_at}
    row.update(current_fields)
    rows.append(row)

def normalize_record(record: dict[str, Any], team_names: dict[str, str], snapshot_id: str, snapshot_fetched_at: str) -> list[dict[str, Any]]:
    item = record['odds_item']
    bookmaker = str(record.get('selected_bookmaker', '')).strip()
    home, away = enrich_names(record, team_names)
    current_fields = market_values(item)
    rows: list[dict[str, Any]] = []
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='h2h', bet_side='home', line='', american=nested(item, 'homeTeamOdds', 'moneyLine'), decimal=nested(item, 'homeTeamOdds', 'current', 'moneyLine', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='h2h', bet_side='away', line='', american=nested(item, 'awayTeamOdds', 'moneyLine'), decimal=nested(item, 'awayTeamOdds', 'current', 'moneyLine', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='spreads', bet_side='home', line=current_fields['home_spread'], american=nested(item, 'homeTeamOdds', 'spreadOdds'), decimal=nested(item, 'homeTeamOdds', 'current', 'spread', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='spreads', bet_side='away', line=current_fields['away_spread'], american=nested(item, 'awayTeamOdds', 'spreadOdds'), decimal=nested(item, 'awayTeamOdds', 'current', 'spread', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='totals', bet_side='over', line=current_fields['total'], american=item.get('overOdds'), decimal=nested(item, 'current', 'over', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    add_row(rows, record=record, home=home, away=away, bookmaker=bookmaker, market_type='totals', bet_side='under', line=current_fields['total'], american=item.get('underOdds'), decimal=nested(item, 'current', 'under', 'decimal'), current_fields=current_fields, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    return rows

def compatibility_event(record: dict[str, Any], team_names: dict[str, str]) -> dict[str, Any]:
    if 'odds_item' in record:
        home, away = enrich_names(record, team_names)
    else:
        home, away = fallback_team_names(record.get('event', {}))
    return {'id': record['event_id'], 'date': record.get('date', ''), 'home': home, 'away': away, 'week': record.get('week', ''), 'source': 'espn_core'}

def raw_odds_record(record: dict[str, Any], team_names: dict[str, str]) -> dict[str, Any]:
    home, away = enrich_names(record, team_names)
    return {'id': record['event_id'], 'date': record.get('date', ''), 'week': record.get('week', ''), 'home': home, 'away': away, 'provider': record['odds_item'].get('provider', {}), 'odds': record['odds_item'], 'odds_collection': record['odds_collection'], 'event': record['event']}

def validate_normalized_rows(rows: list[dict[str, Any]], odds_records: list[dict[str, Any]], snapshot_id: str, snapshot_fetched_at: str) -> None:
    if len(rows) != len(odds_records) * 6:
        fail(f'Normalized odds row count mismatch: rows={len(rows)} odds_events={len(odds_records)}')
    expected_ids = {record['event_id'] for record in odds_records}
    actual_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str, str]] = set()
    pairs_by_game: dict[str, set[tuple[str, str]]] = {}
    for index, row in enumerate(rows, start=2):
        game_id = str(row.get('game_id', '')).strip()
        market_type = str(row.get('market_type', '')).strip()
        bet_side = str(row.get('bet_side', '')).strip()
        bookmaker = str(row.get('bookmaker', '')).strip()
        home = str(row.get('home_team', '')).strip()
        away = str(row.get('away_team', '')).strip()
        if not game_id or not bookmaker or (not home) or (not away):
            fail(f'Normalized odds row {index} has blank identity fields')
        if parse_espn_datetime(row.get('commence_time')) is None:
            fail(f'Normalized odds row {index} has invalid commence_time')
        if str(row.get('snapshot_id', '')).strip() != snapshot_id:
            fail(f'Normalized odds row {index} has wrong snapshot_id')
        if str(row.get('snapshot_fetched_at', '')).strip() != snapshot_fetched_at:
            fail(f'Normalized odds row {index} has wrong snapshot_fetched_at')
        if str(row.get('last_update', '')).strip() != snapshot_fetched_at:
            fail(f'Normalized odds row {index} has wrong capture-time last_update')
        pair = (market_type, bet_side)
        if pair not in EXPECTED_MARKET_SIDES:
            fail(f'Normalized odds row {index} has invalid market/side={pair}')
        key = (game_id, market_type, bet_side, bookmaker)
        if key in seen_keys:
            fail(f'Duplicate normalized odds key: {key}')
        seen_keys.add(key)
        actual_ids.add(game_id)
        pairs_by_game.setdefault(game_id, set()).add(pair)
    if actual_ids != expected_ids:
        fail(f'Normalized odds game IDs do not match odds records: missing={sorted(expected_ids - actual_ids)} unexpected={sorted(actual_ids - expected_ids)}')
    for game_id in sorted(expected_ids):
        pairs = pairs_by_game.get(game_id, set())
        if pairs != EXPECTED_MARKET_SIDES:
            fail(f'Normalized odds event {game_id} missing market rows: {sorted(EXPECTED_MARKET_SIDES - pairs)}')

def validate_raw_payload(payload: dict[str, Any], *, season: int, season_type: int, week: int, snapshot_id: str, snapshot_fetched_at: str, event_records: list[dict[str, Any]], odds_records: list[dict[str, Any]]) -> None:
    if payload.get('snapshot_id') != snapshot_id:
        fail('Raw odds payload has wrong snapshot_id')
    if payload.get('fetched_at') != snapshot_fetched_at:
        fail('Raw odds payload has wrong fetched_at')
    if payload.get('season') != season:
        fail('Raw odds payload has wrong season')
    if payload.get('season_type') != season_type:
        fail('Raw odds payload has wrong season_type')
    if payload.get('target_week') != week:
        fail('Raw odds payload has wrong target_week')
    events = payload.get('events')
    all_events = payload.get('all_events')
    odds = payload.get('odds')
    if not isinstance(events, list) or not isinstance(all_events, list):
        fail('Raw odds payload events/all_events must be lists')
    if not isinstance(odds, list):
        fail('Raw odds payload odds must be a list')
    if payload.get('events_count') != len(events):
        fail('Raw odds payload events_count mismatch')
    if payload.get('odds_events_count') != len(odds):
        fail('Raw odds payload odds_events_count mismatch')
    expected_event_ids = [record['event_id'] for record in event_records]
    actual_event_ids = [str(item.get('id', '')).strip() for item in events if isinstance(item, dict)]
    all_event_ids = [str(item.get('id', '')).strip() for item in all_events if isinstance(item, dict)]
    if len(actual_event_ids) != len(events) or len(all_event_ids) != len(all_events):
        fail('Raw odds payload contains invalid event objects')
    if actual_event_ids != expected_event_ids or all_event_ids != expected_event_ids:
        fail('Raw odds payload event IDs do not match configured-week events')
    if len(actual_event_ids) != len(set(actual_event_ids)):
        fail('Raw odds payload contains duplicate event IDs')
    expected_odds_ids = [record['event_id'] for record in odds_records]
    actual_odds_ids = [str(item.get('id', '')).strip() for item in odds if isinstance(item, dict)]
    if len(actual_odds_ids) != len(odds):
        fail('Raw odds payload contains invalid odds objects')
    if actual_odds_ids != expected_odds_ids:
        fail('Raw odds payload odds IDs do not match normalized odds records')
    if len(actual_odds_ids) != len(set(actual_odds_ids)):
        fail('Raw odds payload contains duplicate odds event IDs')

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in OUTPUT_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())

def write_json(path: Path, payload: object) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())

def make_stage_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
    os.close(fd)
    return Path(name)

def validate_staged_csv(path: Path, odds_records: list[dict[str, Any]], snapshot_id: str, snapshot_fetched_at: str) -> None:
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if (reader.fieldnames or []) != OUTPUT_COLUMNS:
            fail('Staged odds CSV headers changed')
        rows = list(reader)
    validate_normalized_rows(rows, odds_records, snapshot_id, snapshot_fetched_at)

def validate_staged_json(path: Path, **validation_kwargs: Any) -> None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        fail(f'Invalid staged odds JSON: {type(exc).__name__}: {exc}')
    if not isinstance(payload, dict):
        fail('Staged odds JSON root is not an object')
    validate_raw_payload(payload, **validation_kwargs)

def publish_outputs(*, raw_path: Path, csv_path: Path, raw_snapshot_path: Path, csv_snapshot_path: Path, raw_payload: dict[str, Any], rows: list[dict[str, Any]], event_records: list[dict[str, Any]], odds_records: list[dict[str, Any]], season: int, season_type: int, week: int, snapshot_id: str, snapshot_fetched_at: str) -> None:
    snapshot_targets = [raw_snapshot_path, csv_snapshot_path]
    for target in snapshot_targets:
        if target.exists():
            fail(f'Snapshot output already exists: {target}')
    targets = [raw_path, csv_path, raw_snapshot_path, csv_snapshot_path]
    staged = {target: make_stage_path(target) for target in targets}
    try:
        write_json(staged[raw_path], raw_payload)
        write_json(staged[raw_snapshot_path], raw_payload)
        write_csv(staged[csv_path], rows)
        write_csv(staged[csv_snapshot_path], rows)
        validation_kwargs = {'season': season, 'season_type': season_type, 'week': week, 'snapshot_id': snapshot_id, 'snapshot_fetched_at': snapshot_fetched_at, 'event_records': event_records, 'odds_records': odds_records}
        validate_staged_json(staged[raw_path], **validation_kwargs)
        validate_staged_json(staged[raw_snapshot_path], **validation_kwargs)
        validate_staged_csv(staged[csv_path], odds_records, snapshot_id, snapshot_fetched_at)
        validate_staged_csv(staged[csv_snapshot_path], odds_records, snapshot_id, snapshot_fetched_at)
        if staged[raw_path].read_bytes() != staged[raw_snapshot_path].read_bytes():
            fail('Raw compatibility and snapshot JSON differ before publication')
        if staged[csv_path].read_bytes() != staged[csv_snapshot_path].read_bytes():
            fail('CSV compatibility and snapshot outputs differ before publication')
        compatibility_targets = [raw_path, csv_path]
        backup_paths = {target: target.with_name(f'.{target.name}.{snapshot_id}.bak') for target in compatibility_targets if target.exists()}
        for backup in backup_paths.values():
            if backup.exists():
                fail(f'Backup path already exists: {backup}')
        published: list[Path] = []
        moved_backups: dict[Path, Path] = {}
        try:
            for target, backup in backup_paths.items():
                os.replace(target, backup)
                moved_backups[target] = backup
            for target in (raw_snapshot_path, csv_snapshot_path, raw_path, csv_path):
                os.replace(staged[target], target)
                published.append(target)
        except Exception:
            for target in reversed(published):
                if target.exists():
                    target.unlink()
            for target, backup in moved_backups.items():
                if backup.exists():
                    os.replace(backup, target)
            raise
        else:
            for backup in moved_backups.values():
                if backup.exists():
                    backup.unlink()
    finally:
        for stage_path in staged.values():
            if stage_path.exists():
                stage_path.unlink()

def run(args: argparse.Namespace, reporter: PipelineReporter) -> None:
    captured_at = utc_now()
    run_date = captured_at.strftime('%Y_%m_%d')
    snapshot_id = captured_at.strftime('%Y_%m_%d_%H%M%S_%f')
    snapshot_fetched_at = captured_at.isoformat()
    raw_path = RAW_ODDS_DIR / f'{run_date}_nfl_odds.json'
    csv_path = ODDS_DIR / f'{run_date}_NFL_odds.csv'
    raw_snapshot_path = RAW_SNAPSHOT_DIR / f'{snapshot_id}_nfl_odds.json'
    csv_snapshot_path = SNAPSHOT_DIR / f'{snapshot_id}_NFL_odds.csv'
    reporter.update_details({'configured_season': args.season, 'configured_season_type': args.season_type, 'configured_week': args.week, 'preferred_sportsbook': args.sportsbook, 'sportsbook_fallback_allowed': True, 'snapshot_id': snapshot_id, 'snapshot_fetched_at': snapshot_fetched_at, 'last_update_semantics': 'capture_time_no_espn_market_timestamp', 'publication_completed': False, 'staged_roundtrip_verified': False})
    events_url = week_events_url(args.season, args.season_type, args.week)
    reporter.add_input(events_url)
    event_refs = discover_event_refs(args.season, args.season_type, args.week)
    event_records = fetch_all_event_records(event_refs, args.season, args.season_type, args.week)
    eligible_records, excluded_ids = eligible_event_records(event_records, captured_at)
    if not eligible_records:
        fail('No current/future ESPN NFL events are eligible for odds retrieval')
    odds_records, missing_odds_ids, bookmaker_fallbacks = fetch_odds_records(eligible_records, args.sportsbook)
    if not odds_records:
        fail('No current/future ESPN NFL odds were returned')
    team_names, team_fetch_failures = fetch_team_names(odds_records)
    rows: list[dict[str, Any]] = []
    for record in odds_records:
        rows.extend(normalize_record(record, team_names, snapshot_id, snapshot_fetched_at))
    validate_normalized_rows(rows, odds_records, snapshot_id, snapshot_fetched_at)
    compatibility_events = [compatibility_event(record, team_names) for record in event_records]
    raw_payload = {'snapshot_id': snapshot_id, 'fetched_at': snapshot_fetched_at, 'source': 'espn_core', 'season': args.season, 'season_type': args.season_type, 'target_week': args.week, 'preferred_bookmakers': [args.sportsbook], 'events_count': len(event_records), 'odds_events_count': len(odds_records), 'events': compatibility_events, 'all_events': list(compatibility_events), 'odds': [raw_odds_record(record, team_names) for record in odds_records]}
    validate_raw_payload(raw_payload, season=args.season, season_type=args.season_type, week=args.week, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at, event_records=event_records, odds_records=odds_records)
    if bookmaker_fallbacks:
        reporter.warning('Preferred sportsbook was unavailable for some events; another ESPN sportsbook was used', events=bookmaker_fallbacks)
    if missing_odds_ids:
        reporter.warning('ESPN returned no usable sportsbook odds for some eligible events', event_ids=missing_odds_ids)
    if team_fetch_failures:
        reporter.warning('Some ESPN team-name requests failed; event-name fallback was used', failures=team_fetch_failures[:10])
    publish_outputs(raw_path=raw_path, csv_path=csv_path, raw_snapshot_path=raw_snapshot_path, csv_snapshot_path=csv_snapshot_path, raw_payload=raw_payload, rows=rows, event_records=event_records, odds_records=odds_records, season=args.season, season_type=args.season_type, week=args.week, snapshot_id=snapshot_id, snapshot_fetched_at=snapshot_fetched_at)
    for output in (raw_path, csv_path, raw_snapshot_path, csv_snapshot_path):
        reporter.add_output(output)
    reporter.set_rows(rows_in=len(event_records), rows_out=len(rows))
    reporter.update_details({'event_refs_discovered': len(event_refs), 'event_records_resolved': len(event_records), 'eligible_current_future_events': len(eligible_records), 'excluded_past_events': len(excluded_ids), 'excluded_past_event_ids': excluded_ids, 'odds_events_returned': len(odds_records), 'missing_odds_events': len(missing_odds_ids), 'missing_odds_event_ids': missing_odds_ids, 'bookmaker_fallback_events': len(bookmaker_fallbacks), 'bookmaker_fallback_details': bookmaker_fallbacks, 'team_name_fetch_failures': len(team_fetch_failures), 'csv_rows': len(rows), 'current_raw_json': str(raw_path), 'current_normalized_csv': str(csv_path), 'snapshot_raw_json': str(raw_snapshot_path), 'snapshot_normalized_csv': str(csv_snapshot_path), 'staged_roundtrip_verified': True, 'publication_completed': True})
    log(f'ESPN season={args.season} season_type={args.season_type} week={args.week} preferred_sportsbook={args.sportsbook}')
    log(f'Configured-week events returned: {len(event_records)}')
    log(f'Eligible current/future events: {len(eligible_records)}')
    log(f'Odds events returned: {len(odds_records)}')
    log(f'Preferred-bookmaker fallback events: {len(bookmaker_fallbacks)}')
    log(f'Events with no usable odds: {len(missing_odds_ids)}')
    log(f'CSV rows written: {len(rows)}')
    log(f'Current raw JSON written: {raw_path}')
    log(f'Current normalized CSV written: {csv_path}')
    log(f'Archived raw JSON written: {raw_snapshot_path}')
    log(f'Archived normalized CSV written: {csv_snapshot_path}')
    print(f'rows={len(rows)} events={len(event_records)} odds_events={len(odds_records)} snapshot_id={snapshot_id}')

def main() -> int:
    args = parse_args()
    ensure_directories()
    LOG_FILE.write_text('', encoding='utf-8')
    try:
        with PipelineReporter(script=SCRIPT_PATH, stage='00_intake', report_root=REPORT_ROOT, pipeline='NFL', league='NFL', season=args.season, extra_context={'component': 'current odds', 'season_type': args.season_type, 'week': args.week}) as reporter:
            run(args, reporter)
        return 0
    except Exception:
        log(traceback.format_exc())
        print(f'ERROR: see {LOG_FILE}', file=sys.stderr)
        return 1
if __name__ == '__main__':
    sys.exit(main())
