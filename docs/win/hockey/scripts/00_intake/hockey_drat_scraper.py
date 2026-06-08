# docs/win/hockey/scripts/00_intake/hockey_drat_scraper.py

import json
import traceback
from pathlib import Path
from datetime import datetime
import pytz
import pandas as pd
from playwright.sync_api import sync_playwright

URLS = {
    "nhl": "https://www.dratings.com/predictor/nhl-hockey-predictions/",
}

UTC = pytz.utc
ET  = pytz.timezone("America/New_York")

ERROR_DIR = Path("docs/win/hockey/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "hockey_drat_scraper.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== hockey_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")

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


def parse_date(date_time_str: str) -> str:
    try:
        dt = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        return dt.strftime("%Y_%m_%d")
    except Exception:
        return date_time_str.strip().replace("/", "_").replace(" ", "_")


def parse_time(date_time_str: str) -> str:
    parts = date_time_str.strip().split(" ")
    if len(parts) >= 2:
        return " ".join(parts[1:])
    return ""


def normalize_team(name: str) -> str:
    name = str(name).strip().lower()
    replacements = {
        "st. louis": "st louis",
        "ny rangers": "new york rangers",
        "ny islanders": "new york islanders",
        "nj devils": "new jersey devils",
        "la kings": "los angeles kings",
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name


def is_game_row(row):
    return len(row) >= 6 and "\n" in row[1]


def parse_nhl(row):
    if not is_game_row(row):
        return None

    try:
        if len(row) == 11:
            date_time = convert_utc_to_et(row[0].replace("\n", " "))
            t = row[1].split("\n")
            team1, team2 = t[0].strip(), t[1].strip()
            wp = row[3].split("\n")
            wp1, wp2 = wp[0], wp[1]
            ml = row[4].split("\n")
            ml1, ml2 = ml[0], ml[1]
            sp = row[5].split("\n")
            sp1, sp2 = sp[0], sp[1]
            ps = row[6].split("\n")
            proj1, proj2 = ps[0], ps[1]
            total = row[7]
            ou = row[8].split("\n")
            over_line, under_line = ou[0], ou[1]

            return {
                "sport":           "NHL",
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
                "score1":          "",
                "score2":          "",
                "game_status":     "upcoming",
            }

        elif len(row) == 8:
            date_time = convert_utc_to_et(row[0].replace("\n", " "))
            t = row[1].split("\n")
            team1, team2 = t[0].strip(), t[1].strip()
            wp = row[2].split("\n")
            wp1, wp2 = wp[0], wp[1]
            ml = row[3].split("\n")
            ml1, ml2 = ml[0], ml[1]
            sp = row[4].split("\n")
            sp1, sp2 = sp[0], sp[1]
            sc = row[5].split("\n")
            score1, score2 = sc[0].strip(), sc[1].strip()

            return {
                "sport":           "NHL",
                "date_time":       date_time,
                "team1":           team1,
                "team2":           team2,
                "team1_win_pct":   wp1,
                "team2_win_pct":   wp2,
                "team1_moneyline": ml1,
                "team2_moneyline": ml2,
                "team1_spread":    sp1,
                "team2_spread":    sp2,
                "proj_score_1":    "",
                "proj_score_2":    "",
                "total":           "",
                "over_line":       "",
                "under_line":      "",
                "score1":          score1,
                "score2":          score2,
                "game_status":     "completed",
            }

    except Exception as e:
        log(f"WARNING: parse_nhl failed on row (len={len(row)}): {e}")

    return None


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    rows = page.query_selector_all("table tbody tr")
    return [[c.inner_text().strip() for c in r.query_selector_all("td")] for r in rows]


def save_final_scores(completed: list, files_written: list):
    if not completed:
        return

    rows = []
    for g in completed:
        try:
            game_date  = parse_date(g["date_time"])
            away_score = int(float(g["score1"]))
            home_score = int(float(g["score2"]))
            total      = away_score + home_score
            away_pl    = away_score - home_score
            home_pl    = home_score - away_score

            rows.append({
                "game_date":      game_date,
                "league":         "Hockey",
                "market":         "NHL",
                "away_team":      normalize_team(g["team1"]),
                "home_team":      normalize_team(g["team2"]),
                "away_score":     away_score,
                "home_score":     home_score,
                "total":          total,
                "away_spread":    "",
                "home_spread":    "",
                "away_puck_line": away_pl,
                "home_puck_line": home_pl,
            })
        except Exception as e:
            log(f"WARNING: could not process completed game {g.get('team1')} vs {g.get('team2')}: {e}")

    if not rows:
        return

    df = pd.DataFrame(rows)

    for date_val, group in df.groupby("game_date"):
        out_dir  = Path("docs/win/final_scores/results/nhl/final_scores")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_val}_final_scores_NHL.csv"

        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["game_date", "away_team", "home_team"], keep="last"
            )
            combined.to_csv(out_path, index=False)
        else:
            group.to_csv(out_path, index=False)

        files_written.append((str(out_path), len(group)))
        log(f"WROTE final scores -> {out_path} ({len(group)} rows)")


def main():
    files_written  = []
    parse_errors   = 0

    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")

        raw_dir = Path("docs/win/hockey/00_intake/drat_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        pred_dir = Path("docs/win/hockey/00_intake/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)

        scraper_dir = pred_dir / "scraper"
        scraper_dir.mkdir(parents=True, exist_ok=True)

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

            raw = scrape_page(page, URLS["nhl"])

            raw_rows_path = raw_dir / f"{date}_nhl_raw_rows.json"
            with open(raw_rows_path, "w") as f:
                json.dump(raw, f, indent=2)
            files_written.append((str(raw_rows_path), len(raw)))

            col_counts = {}
            for r in raw:
                n = len(r)
                col_counts[n] = col_counts.get(n, 0) + 1
            log(f"Column count distribution: {col_counts}")

            games = []
            for r in raw:
                result = parse_nhl(r)
                if result:
                    games.append(result)
                elif is_game_row(r):
                    parse_errors += 1

            raw_path = raw_dir / f"{date}_nhl_raw.json"
            with open(raw_path, "w") as f:
                json.dump(games, f, indent=2)
            files_written.append((str(raw_path), len(games)))

            upcoming  = [g for g in games if g["game_status"] == "upcoming"]
            completed = [g for g in games if g["game_status"] == "completed"]

            log(f"Upcoming games: {len(upcoming)}")
            log(f"Completed games: {len(completed)}")

            if upcoming:
                df_up        = pd.DataFrame(upcoming)
                scraper_path = scraper_dir / f"{date}_nhl_predictions.csv"
                df_up.to_csv(scraper_path, index=False)
                files_written.append((str(scraper_path), len(df_up)))
                log(f"WROTE upcoming scraper copy -> {scraper_path} ({len(df_up)} rows)")
            else:
                log("No upcoming games found.")

            if completed:
                save_final_scores(completed, files_written)
            else:
                log("No completed games found.")

            browser.close()

        log("--- SUMMARY ---")
        log(f"Raw rows scraped: {len(raw)}")
        log(f"Games parsed: {len(games)}")
        log(f"Parse errors: {parse_errors}")
        log(f"Upcoming: {len(upcoming)}")
        log(f"Completed: {len(completed)}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
