#!/usr/bin/env python3
"""
Pull NFL game weather data from Open-Meteo API
and save to data/raw/weather/game_weather_latest.csv
"""

import argparse
import os
from datetime import datetime
import pandas as pd
import requests

OUT_DIR = "data/raw/weather"
OUT_FILE = os.path.join(OUT_DIR, "game_weather_latest.csv")


def parse_args():
    ap = argparse.ArgumentParser(description="Pull NFL game weather from Open-Meteo")
    ap.add_argument("--season", type=int, required=False, help="Season year (e.g., 2025)")
    ap.add_argument("--week", type=int, required=False, help="Week number (1–18)")
    return ap.parse_args()


def fetch_weather(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    # Placeholder: you may already have a games schedule to loop through.
    # Here we just demonstrate with a simple call.
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 42.3601,   # Example: Boston
        "longitude": -71.0589,
        "hourly": "temperature_2m,precipitation,cloudcover,windspeed_10m",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data.get("hourly", {}))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Add metadata
    df["season"] = season
    df["week"] = week
    df["fetched_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return df


def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[weather] wrote {path} ({len(df)} rows)", flush=True)


def main():
    args = parse_args()
    df = fetch_weather(args.season, args.week)

    if df.empty:
        raise RuntimeError("Weather fetch returned no data")

    write_csv(df, OUT_FILE)


if __name__ == "__main__":
    main()
