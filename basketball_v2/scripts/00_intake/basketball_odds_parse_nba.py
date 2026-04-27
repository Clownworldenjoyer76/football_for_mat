#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_odds_parse_nba.py

import csv
import json
import traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

# =========================
# PATHS
# =========================

ODDS_DIR   = Path('docs/win/basketball/odds/nba')
OUTPUT_DIR = Path('docs/win/basketball/00_intake/sportsbook/nba')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NY_TZ  = ZoneInfo('America/New_York')
UTC_TZ = ZoneInfo('UTC')

BOOKMAKER_KEY = 'draftkings'

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_odds_parse_nba.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_odds_parse_nba RUN {datetime.now().isoformat()} ===\n")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

# =========================
# DECIMAL TO AMERICAN
# =========================

def decimal_to_american(dec):
    if dec is None:
        return None
    try:
        dec = float(dec)
    except Exception:
        return None
    if dec >= 2.0:
        return round((dec - 1) * 100)
    else:
        return round(-100 / (dec - 1))

# =========================
# PARSE ONE GAME
# =========================

def parse_game(game):
    game_id   = game.get('id')
    home_team = game.get('home_team')
    away_team = game.get('away_team')

    commence_utc = datetime.fromisoformat(
        game['commence_time'].replace('Z', '+00:00')
    ).replace(tzinfo=UTC_TZ)

    commence_ny = commence_utc.astimezone(NY_TZ)
    game_date   = commence_ny.strftime('%Y_%m_%d')
    game_time   = commence_ny.strftime('%I:%M %p')

    dk = None
    for bk in game.get('bookmakers', []):
        if bk.get('key') == BOOKMAKER_KEY:
            dk = bk
            break

    if dk is None:
        return game_date, None

    odds_last_update = dk.get('last_update')

    away_ml_dec = home_ml_dec = None
    away_spread_dec = home_spread_dec = None
    away_spread = home_spread = None
    over_dec = under_dec = total = None

    for market in dk.get('markets', []):
        key      = market.get('key')
        outcomes = market.get('outcomes', [])

        if key == 'h2h':
            for o in outcomes:
                if o.get('name') == home_team:
                    home_ml_dec = o.get('price')
                elif o.get('name') == away_team:
                    away_ml_dec = o.get('price')

        elif key == 'spreads':
            for o in outcomes:
                if o.get('name') == home_team:
                    home_spread_dec = o.get('price')
                    home_spread     = o.get('point')
                elif o.get('name') == away_team:
                    away_spread_dec = o.get('price')
                    away_spread     = o.get('point')

        elif key == 'totals':
            for o in outcomes:
                if o.get('name') == 'Over':
                    over_dec = o.get('price')
                    total    = o.get('point')
                elif o.get('name') == 'Under':
                    under_dec = o.get('price')

    row = {
        'sport':                      'Basketball',
        'league':                     'NBA',
        'game_date':                  game_date,
        'game_id':                    game_id,
        'odds_last_update':           odds_last_update,
        'game_time':                  game_time,
        'home_team':                  home_team,
        'away_team':                  away_team,
        'home_spread':                home_spread,
        'away_spread':                away_spread,
        'total':                      total,
        'home_dk_moneyline_american': decimal_to_american(home_ml_dec),
        'away_dk_moneyline_american': decimal_to_american(away_ml_dec),
        'home_dk_spread_american':    decimal_to_american(home_spread_dec),
        'away_dk_spread_american':    decimal_to_american(away_spread_dec),
        'dk_total_over_american':     decimal_to_american(over_dec),
        'dk_total_under_american':    decimal_to_american(under_dec),
        'home_dk_moneyline_decimal':  home_ml_dec,
        'away_dk_moneyline_decimal':  away_ml_dec,
        'home_dk_spread_decimal':     home_spread_dec,
        'away_dk_spread_decimal':     away_spread_dec,
        'dk_total_over_decimal':      over_dec,
        'dk_total_under_decimal':     under_dec,
    }

    return game_date, row

# =========================
# MAIN
# =========================

def main():
    files_written = []
    games_parsed  = 0
    games_skipped = 0

    fieldnames = [
        'sport', 'league', 'game_date', 'game_id', 'odds_last_update',
        'game_time', 'home_team', 'away_team',
        'home_spread', 'away_spread', 'total',
        'home_dk_moneyline_american', 'away_dk_moneyline_american',
        'home_dk_spread_american', 'away_dk_spread_american',
        'dk_total_over_american', 'dk_total_under_american',
        'home_dk_moneyline_decimal', 'away_dk_moneyline_decimal',
        'home_dk_spread_decimal', 'away_dk_spread_decimal',
        'dk_total_over_decimal', 'dk_total_under_decimal',
    ]

    try:
        json_files = sorted(ODDS_DIR.glob('*.json'))
        if not json_files:
            log(f'No JSON files found in {ODDS_DIR}')
            log("STATUS: SUCCESS (nothing to do)")
            return

        for json_path in json_files:
            log(f'Processing {json_path.name}')

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    games = json.load(f)

                by_date = defaultdict(list)

                for game in games:
                    try:
                        game_date, row = parse_game(game)
                        if row is not None:
                            by_date[game_date].append(row)
                            games_parsed += 1
                        else:
                            games_skipped += 1
                    except Exception as e:
                        games_skipped += 1
                        log(f"  ERROR parsing game {game.get('id', '?')}: {e}\n{traceback.format_exc()}")

                for game_date, rows in by_date.items():
                    out_path = OUTPUT_DIR / f'{game_date}_NBA_odds.csv'

                    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)

                    files_written.append((str(out_path), len(rows)))
                    log(f'  WROTE {out_path.name} ({len(rows)} games)')

            except Exception as e:
                log(f"ERROR processing {json_path.name}: {e}\n{traceback.format_exc()}")

        log("--- SUMMARY ---")
        log(f"Games parsed: {games_parsed}")
        log(f"Games skipped (no DraftKings odds): {games_skipped}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} games)")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

if __name__ == '__main__':
    main()