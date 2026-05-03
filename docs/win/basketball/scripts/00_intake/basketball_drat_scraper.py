# docs/win/basketball/scripts/00_intake/basketball_drat_scraper.py

import json
import time
import random
import traceback
import re
from pathlib import Path
from datetime import datetime
import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "nba":  "https://www.dratings.com/predictor/nba-basketball-predictions/",
    "ncaa": "https://www.dratings.com/predictor/ncaa-basketball-predictions/",
    "wnba": "https://www.dratings.com/predictor/wnba-basketball-predictions/",
}

UTC = pytz.utc
ET  = pytz.timezone("America/New_York")

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_drat_scraper.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== basketball_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt     = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et  = dt_utc.astimezone(ET)
        return dt_et.strftime("%m/%d/%Y %I:%M %p")
    except Exception:
        return date_time_str


def split_pair(value: str):
    value = (value or "").strip()

    if not value:
        return "", ""

    if "\n" in value:
        parts = [p.strip() for p in value.split("\n") if p.strip()]
    elif "|" in value:
        parts = [p.strip() for p in value.split("|") if p.strip()]
    else:
        parts = [value.strip()]

    if len(parts) >= 2:
        return parts[0], parts[1]

    if len(parts) == 1:
        return parts[0], ""

    return "", ""


def strip_wnba_record(team: str) -> str:
    return re.sub(r"\s*\(\d+-\d+\)\s*$", "", (team or "").strip()).strip()


def is_game_row(row):
    return len(row) >= 5 and "\n" in row[1]


def is_score(s):
    try:
        v = float(s)
        return v >= 0 and v == int(v) and v < 250
    except (ValueError, TypeError):
        return False


def parse_nba_ncaa(row, sport):
    if not is_game_row(row):
        return None

    try:
        date_time = convert_utc_to_et(row[0].replace("\n", " "))

        t = row[1].split("\n")
        team1, team2 = t[0].strip(), t[1].strip()

        w = row[2].split("\n")
        wp1, wp2 = w[0].strip(), w[1].strip()

        m = row[3].split("\n")
        ml1, ml2 = m[0].strip(), m[1].strip()

        s = row[4].split("\n")
        sp1, sp2 = s[0].strip(), s[1].strip()

        proj1 = proj2 = total = over_line = under_line = ""
        score1 = score2 = game_status = ""

        if len(row) >= 10 and "\n" in row[5] and not is_score(row[5].split("\n")[0]):
            ps = row[5].split("\n")
            proj1, proj2 = ps[0], ps[1]
            total = row[6]
            ou = row[7].split("\n")
            over_line, under_line = ou[0], ou[1]

        elif len(row) >= 9 and not is_score(row[5]):
            total = row[5]
            ou = row[6].split("\n")
            over_line, under_line = ou[0], ou[1]
            game_status = " ".join(row[7].split("\n"))
            sc = row[8].split("\n")
            score1, score2 = sc[0], sc[1]

        elif len(row) >= 7:
            sc = row[5].split("\n")
            score1 = sc[0].strip()
            if len(sc) > 1:
                score2 = sc[1].strip()
            elif len(row) > 6 and is_score(row[6]):
                score2 = row[6].strip()

        return {
            "sport":           sport,
            "date_time":       date_time,
            "team1":           team1,
            "team2":           team2,
            "team1_win_pct":   wp1,
            "team2_win_pct":   wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread":    sp1,
            "team2_spread":    sp2,
            "proj_score_1":    proj1,
            "proj_score_2":    proj2,
            "total":           total,
            "over_line":       over_line,
            "under_line":      under_line,
            "score1":          score1,
            "score2":          score2,
            "game_status":     game_status,
        }

    except Exception:
        return None


def parse_wnba(row):
    """
    WNBA-specific parser.

    Confirmed WNBA page shapes:
      Future rows:    10 cells
        0 date/time
        1 teams, with records
        2 win %
        3 blank/moneyline
        4 blank/spread
        5 projected scores
        6 total
        7 blank/O-U
        8 blank
        9 blank

      Completed rows: 8 cells
        0 date/time
        1 teams
        2 win %
        3 moneyline
        4 spread
        5 final score
        6 rating/stat junk
        7 rating/stat junk

      Summary rows:   6 cells
        rejected
    """
    if len(row) not in (8, 10):
        return None

    try:
        date_time = convert_utc_to_et(row[0].replace("\n", " "))

        team1, team2 = split_pair(row[1])
        team1 = strip_wnba_record(team1)
        team2 = strip_wnba_record(team2)

        if not team1 or not team2:
            return None

        if team1.lower() == "sportsbooks" or team1.lower() == "dratings":
            return None

        wp1, wp2 = split_pair(row[2])
        ml1, ml2 = split_pair(row[3]) if len(row) > 3 else ("", "")
        sp1, sp2 = split_pair(row[4]) if len(row) > 4 else ("", "")

        proj1 = proj2 = total = over_line = under_line = ""
        score1 = score2 = game_status = ""

        if len(row) == 10:
            proj1, proj2 = split_pair(row[5])
            total = row[6].strip() if len(row) > 6 else ""
            over_line, under_line = split_pair(row[7]) if len(row) > 7 else ("", "")

        elif len(row) == 8:
            score1, score2 = split_pair(row[5])

        return {
            "sport":           "wnba",
            "date_time":       date_time,
            "team1":           team1,
            "team2":           team2,
            "team1_win_pct":   wp1,
            "team2_win_pct":   wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread":    sp1,
            "team2_spread":    sp2,
            "proj_score_1":    proj1,
            "proj_score_2":    proj2,
            "total":           total,
            "over_line":       over_line,
            "under_line":      under_line,
            "score1":          score1,
            "score2":          score2,
            "game_status":     game_status,
        }

    except Exception:
        return None


def parse_row(row, sport):
    if sport == "wnba":
        return parse_wnba(row)

    return parse_nba_ncaa(row, sport)


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    rows = page.query_selector_all("table tbody tr")
    return [[c.inner_text().strip() for c in r.query_selector_all("td")] for r in rows]


def main():
    files_written = []

    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")
        base_out_dir = Path("docs/win/basketball/00_intake/drat_raw")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            })

            for sport, url in URLS.items():
                log(f"Scraping {sport.upper()}")

                try:
                    raw = scrape_page(page, url)
                    log(f"  RAW ROWS: {len(raw)}")

                    games = [parse_row(r, sport) for r in raw]
                    games = [g for g in games if g]

                    label = "ncaam" if sport == "ncaa" else sport
                    out_dir = base_out_dir / label
                    out_dir.mkdir(parents=True, exist_ok=True)

                    path = out_dir / f"{date}_{label}_raw.json"

                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(games, f, indent=2)

                    files_written.append((str(path), len(games)))
                    log(f"  WROTE {path} ({len(games)} games)")

                except Exception as e:
                    log(f"ERROR scraping {sport}: {e}\n{traceback.format_exc()}")

                time.sleep(random.uniform(2, 4))

            browser.close()

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"  FILE: {path} ({count} games)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Basketball drat scraper complete.")


if __name__ == "__main__":
    main()
