"""
depth_chart.py

Pulls depth chart data for every NFL team from the ESPN API, flattens the
JSON response fully, and writes one combined raw CSV.

Endpoints used:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/depthcharts

Output:
    docs/win/football/nfl/data/raw/raw_depth.csv
"""

import csv
import json
import os
import urllib.request

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
DEPTHCHART_URL_TEMPLATE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"
OUTPUT_PATH = "docs/win/football/nfl/data/raw/raw_depth.csv"


def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def flatten(obj, parent_key="", sep="."):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten(v, new_key, sep))
    else:
        items[parent_key] = obj
    return items


def get_team_ids():
    data = fetch_json(TEAMS_URL)
    team_ids = []
    for league in data.get("sports", [{}])[0].get("leagues", [{}]):
        for team_entry in league.get("teams", []):
            team = team_entry.get("team", {})
            team_id = team.get("id")
            if team_id:
                team_ids.append(team_id)
    return team_ids


def main():
    team_ids = get_team_ids()

    all_rows = []
    all_columns = set()

    for team_id in team_ids:
        url = DEPTHCHART_URL_TEMPLATE.format(team_id=team_id)
        depth_data = fetch_json(url)

        flat_row = flatten(depth_data)
        flat_row["team_id"] = team_id
        all_rows.append(flat_row)
        all_columns.update(flat_row.keys())

    fieldnames = sorted(all_columns)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"rows={len(all_rows)} columns={len(fieldnames)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
