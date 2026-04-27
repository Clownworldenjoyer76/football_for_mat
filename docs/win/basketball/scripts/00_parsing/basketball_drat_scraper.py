# docs/win/basketball/scripts/00_intake/basketball_drat_scraper.py

import json
import time
import random
from pathlib import Path
from datetime import datetime
import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "nba":  "https://www.dratings.com/predictor/nba-basketball-predictions/",
    "ncaa": "https://www.dratings.com/predictor/ncaa-basketball-predictions/",
}

UTC = pytz.utc
ET  = pytz.timezone("America/New_York")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt     = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et  = dt_utc.astimezone(ET)
        return dt_et.strftime("%m/%d/%Y %I:%M %p")
    except Exception:
        return date_time_str


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
            # Upcoming: proj scores in col[5]
            ps = row[5].split("\n")
            proj1, proj2 = ps[0], ps[1]
            total = row[6]
            ou = row[7].split("\n")
            over_line, under_line = ou[0], ou[1]
        elif len(row) >= 9 and not is_score(row[5]):
            # Live: total in col[5], scores in col[8]
            total = row[5]
            ou = row[6].split("\n")
            over_line, under_line = ou[0], ou[1]
            game_status = " ".join(row[7].split("\n"))
            sc = row[8].split("\n")
            score1, score2 = sc[0], sc[1]
        elif len(row) >= 7:
            # Completed: scores in col[5] as "s1\ns2" or col[5]/col[6]
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
    except:
        return None


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    rows = page.query_selector_all("table tbody tr")
    return [[c.inner_text().strip() for c in r.query_selector_all("td")] for r in rows]


def main():
    date    = datetime.now(ET).strftime("%Y_%m_%d")
    out_dir = Path("docs/win/basketball/00_intake/drat_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        })

        for sport, url in URLS.items():
            print(f"Scraping {sport.upper()}...")
            try:
                raw   = scrape_page(page, url)
                games = [parse_nba_ncaa(r, sport) for r in raw]
                games = [g for g in games if g]

                label = "ncaab" if sport == "ncaa" else sport
                path  = out_dir / f"{date}_{label}_raw.json"
                with open(path, "w") as f:
                    json.dump(games, f, indent=2)
                print(f"  Saved {len(games)} games -> {path}")
            except Exception as e:
                print(f"  ERROR scraping {sport}: {e}")

            time.sleep(random.uniform(2, 4))

        browser.close()


if __name__ == "__main__":
    main()
