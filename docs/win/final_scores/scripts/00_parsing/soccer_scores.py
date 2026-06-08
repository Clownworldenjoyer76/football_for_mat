#!/usr/bin/env python3
# docs/win/final_scores/scripts/00_parsing/scores.py

import sys
import csv
import re
from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd

# =========================
# LOGGER UTILITY
# =========================

def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_mode = "a" if log_path.exists() else "w"
    with open(log_path, log_mode) as f:
        f.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            f.write(f"  MSG: {msg}\n")
        if df is not None:
            f.write(f"  STATS: {len(df)} rows\n")
        f.write("-" * 40 + "\n")

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path("docs/win/final_scores")
ERR_DIR = BASE_DIR / "errors"
AUDIT_LOG = BASE_DIR / "scripts/00_parsing/parsing_audit.txt"

ERR_DIR.mkdir(parents=True, exist_ok=True)

SOCCER_HEADERS = [
    "league",
    "market",
    "game_date",
    "match_time",
    "home_team",
    "away_team",
    "away_score",
    "home_score"
]

SOCCER_MARKETS = {"epl", "laliga", "ligue1", "bundesliga", "seriea"}

def is_date_line(s: str) -> bool:
    return bool(re.match(r"^\d{2}/\d{2}/\d{4}$", s.strip()))

def convert_to_pipeline_date(s: str) -> str:
    return datetime.strptime(s.strip(), "%m/%d/%Y").strftime("%Y_%m_%d")

# =========================
# PARSING LOGIC
# =========================

def parse_soccer(lines, market):

    games = []
    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not is_date_line(line):
            i += 1
            continue

        # Convert immediately to pipeline format
        game_date = convert_to_pipeline_date(line)

        i += 1
        if i >= len(lines):
            break

        row_data = lines[i].split("\t")

        match_time = row_data[0].strip()
        away_team = row_data[1].strip() if len(row_data) > 1 else ""

        i += 1
        if i >= len(lines):
            break

        home_team = lines[i].split("\t")[0].strip()

        score_count = 0
        away_score = ""
        home_score = ""

        search_limit = i + 10

        while i < len(lines) and i < search_limit:

            potential_score = lines[i].split("\t")[0].strip()

            if potential_score.isdigit():

                if score_count == 0:
                    away_score = potential_score
                    score_count += 1
                else:
                    home_score = potential_score
                    break

            i += 1

        games.append({
            "league": "Soccer",
            "market": market,
            "game_date": game_date,
            "match_time": match_time,
            "home_team": home_team,
            "away_team": away_team,
            "away_score": away_score,
            "home_score": home_score
        })

        i += 1

    return games

# =========================
# SOCCER MASTER BUILDER
# =========================

def build_soccer_master():

    soccer_dir = BASE_DIR / "results/soccer/final_scores"

    files = [
        f for f in soccer_dir.glob("*_final_scores_*.csv")
        if f.name.split("_final_scores_")[-1].replace(".csv", "") in SOCCER_MARKETS
    ]

    dates = {}

    for f in files:

        name = f.name
        m = re.search(r"(\d{4}_\d{2}_\d{2})", name)

        if not m:
            continue

        d = m.group(1)
        dates.setdefault(d, []).append(f)

    for date_str, file_list in dates.items():

        dfs = []

        for f in file_list:

            try:
                df = pd.read_csv(f)

                if not df.empty:
                    dfs.append(df)

            except Exception:
                continue

        if not dfs:
            continue

        master_df = pd.concat(dfs, ignore_index=True)

        out_path = soccer_dir / f"{date_str}_final_scores_SOCCER.csv"

        master_df.to_csv(out_path, index=False)

# =========================
# MAIN
# =========================

def main():

    if len(sys.argv) != 3:
        return 2

    market = sys.argv[1].lower()
    input_path = Path(sys.argv[2])

    if market not in SOCCER_MARKETS:
        print(f"Market {market} not in soccer list.")
        return 1

    try:

        lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()

        all_games = parse_soccer(lines, market)

        if not all_games:
            raise ValueError("No soccer games parsed.")

        df_all = pd.DataFrame(all_games)

        unique_dates = df_all["game_date"].unique()

        for m_date in unique_dates:

            output_dir = BASE_DIR / "results/soccer/final_scores"
            output_dir.mkdir(parents=True, exist_ok=True)

            out_path = output_dir / f"{m_date}_final_scores_{market}.csv"

            date_rows = [g for g in all_games if g["game_date"] == m_date]

            with open(out_path, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=SOCCER_HEADERS)
                writer.writeheader()
                writer.writerows(date_rows)

            print(f"Created: {out_path}")

        build_soccer_master()

        audit(AUDIT_LOG, "SOCCER_PARSE", "SUCCESS", msg=f"Processed {market}", df=df_all)

        return 0

    except Exception as e:

        err_path = ERR_DIR / f"soccer_error_{market}.txt"

        with open(err_path, "w") as f:
            f.write(traceback.format_exc())

        audit(AUDIT_LOG, "SOCCER_PARSE", "ERROR", msg=str(e))

        return 1


if __name__ == "__main__":
    sys.exit(main())
