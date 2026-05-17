import requests
import os
import json
from pathlib import Path
from datetime import datetime

API_KEY = os.getenv("ODDS_API_KEY")

if not API_KEY:
    raise RuntimeError("ODDS_API_KEY environment variable is not set")

today = datetime.utcnow().strftime("%Y_%m_%d")

sport = "baseball_mlb"
path = f"docs/win/baseball/odds/{today}.json"

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
    raise SystemExit(1)

data = response.json()

Path(path).parent.mkdir(parents=True, exist_ok=True)

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved {path}")
