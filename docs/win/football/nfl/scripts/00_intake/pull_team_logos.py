"""
pull_team_logos.py

Downloads all logo variants for every NFL team from team_master.csv and
saves them into per-team subfolders.

Input:
    docs/win/football/nfl/data/master/team_master.csv

Output:
    docs/win/football/nfl/config/mapping/team_logos/{team_abbr}/{team_abbr}_{variant}.png

Error/run log:
    docs/win/football/nfl/errors/00_intake/pull_team_logos.txt
"""

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone

INPUT_PATH = "docs/win/football/nfl/data/master/team_master.csv"
OUTPUT_ROOT = "docs/win/football/nfl/config/mapping/team_logos"
ERROR_LOG_PATH = "docs/win/football/nfl/errors/00_intake/pull_team_logos.txt"


def log(lines):
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"--- run {timestamp} ---\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def variant_name(rel_list):
    """
    Converts a rel tag list like ["full", "default"] or
    ["full", "scoreboard", "dark"] into a variant string, e.g.
    "default" or "scoreboard_dark". Drops the leading "full" tag.
    """
    parts = [p for p in rel_list if p != "full"]
    return "_".join(parts) if parts else "default"


def main():
    log_lines = []
    total_downloaded = 0
    total_failed = 0

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        team_abbr = row["sports.leagues.teams.team.abbreviation"]
        hrefs = json.loads(row["sports.leagues.teams.team.logos.href"])
        rels = json.loads(row["sports.leagues.teams.team.logos.rel"])

        team_folder = os.path.join(OUTPUT_ROOT, team_abbr)
        os.makedirs(team_folder, exist_ok=True)

        for href, rel in zip(hrefs, rels):
            variant = variant_name(rel)
            filename = f"{team_abbr}_{variant}.png"
            filepath = os.path.join(team_folder, filename)

            try:
                urllib.request.urlretrieve(href, filepath)
                total_downloaded += 1
            except Exception as e:
                total_failed += 1
                msg = f"team={team_abbr} variant={variant} url={href} error={e}"
                print(msg)
                log_lines.append(msg)

    summary = f"downloaded={total_downloaded} failed={total_failed}"
    print(summary)
    log_lines.append(summary)
    log(log_lines)


if __name__ == "__main__":
    main()
