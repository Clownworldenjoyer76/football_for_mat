"""
team_power_index.py

Pulls ESPN's season-level Football Power Index (FPI) data for every
NFL team for the current season.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/powerindex

Output:
    docs/win/football/nfl/data/team_power_index/team_power_index_{season}.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026

POWERINDEX_URL = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{SEASON}/powerindex"

OUTPUT_PATH = f"docs/win/football/nfl/data/team_power_index/team_power_index_{SEASON}.csv"


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_team_id(ref_url):
    if not ref_url:
        return ""
    parts = ref_url.split("/teams/")
    if len(parts) < 2:
        return ""
    return parts[1].split("?")[0]


def main():
    try:
        data = fetch_json(POWERINDEX_URL)
    except Exception as e:
        print(f"failed to pull power index: {e}")
        return

    all_items = list(data.get("items", []))
    page_count = data.get("pageCount", 1)

    for page in range(2, page_count + 1):
        try:
            next_page = fetch_json(f"{POWERINDEX_URL}&page={page}")
            all_items.extend(next_page.get("items", []))
        except Exception as e:
            print(f"failed to pull page {page}: {e}")

    if not all_items:
        print("no power index data available")
        return

    rows = []
    ordered_fieldnames = ["season", "team_id", "lastUpdated"]
    seen = set(ordered_fieldnames)

    for item in all_items:
        team_ref = item.get("team", {}).get("$ref", "")
        team_id = extract_team_id(team_ref)

        row = {
            "season": item.get("season", SEASON),
            "team_id": team_id,
            "lastUpdated": item.get("lastUpdated", ""),
        }

        for stat in item.get("predictives", []):
            name = stat.get("name", "")
            row[name] = stat.get("value", "")
            if name not in seen:
                ordered_fieldnames.append(name)
                seen.add(name)

        rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
