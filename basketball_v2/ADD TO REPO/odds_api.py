import requests
import os
import json
from pathlib import Path
from datetime import datetime

API_KEY = os.getenv("ODDS_API_KEY")

today = datetime.utcnow().strftime("%Y_%m_%d")

targets = {
    "basketball_nba": f"docs/win/basketball/odds/nba/{today}.json",
    "icehockey_nhl": f"docs/win/hockey/odds/{today}.json",
    "baseball_mlb": f"docs/win/baseball/odds/{today}.json"
}

for sport, path in targets.items():
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"

    params = {
        "apiKey": API_KEY,
        "markets": "h2h,spreads,totals",
        "bookmakers": "draftkings"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"{sport} error: {response.status_code}")
        print(response.text)
        continue

    data = response.json()

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {path}")
