#!/usr/bin/env python3
"""
Fetch day-of weather for upcoming games using Open-Meteo (no API key).
Joins by game (kickoff time + home team stadium) and outputs per-game weather.

Inputs:
  - data/raw/nflverse/schedules_*.csv.gz (used to find upcoming games, stadium, roof)

Outputs:
  - data/raw/weather/game_weather_latest.csv
  - data/raw/weather/game_weather_{season}_wk{week}.csv  (if --season & --week provided)

Usage:
  python scripts/pull_weather_openmeteo.py --season 2025 --week 1
  python scripts/pull_weather_openmeteo.py             # auto-detect next 10 days from schedules
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
NFLVERSE_DIR = RAW_DIR / "nflverse"
WEATHER_DIR = RAW_DIR / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULES_GLOB = NFLVERSE_DIR / "schedules_*.csv.gz"
OUT_LATEST = WEATHER_DIR / "game_weather_latest.csv"

# Basic stadium coordinates & dome flag by home team (current 32 teams).
# Coordinates approximate center of the playing venue; tz handled by UTC kickoff from schedules.
STADIUMS = {
    # AFC East
    "BUF": {"lat": 42.7738, "lon": -78.7875, "dome": False},  # Highmark Stadium
    "MIA": {"lat": 25.9580, "lon": -80.2389, "dome": False},  # Hard Rock Stadium
    "NE":  {"lat": 42.0909, "lon": -71.2643, "dome": False},  # Gillette Stadium
    "NYJ": {"lat": 40.8128, "lon": -74.0742, "dome": False},  # MetLife (Jets)
    # AFC North
    "BAL": {"lat": 39.2780, "lon": -76.6227, "dome": False},  # M&T Bank
    "CIN": {"lat": 39.0954, "lon": -84.5160, "dome": False},  # Paycor
    "CLE": {"lat": 41.5061, "lon": -81.6995, "dome": False},  # Cleveland Browns Stadium
    "PIT": {"lat": 40.4468, "lon": -80.0158, "dome": False},  # Acrisure
    # AFC South
    "HOU": {"lat": 29.6847, "lon": -95.4107, "dome": True},   # NRG Stadium (retractable)
    "IND": {"lat": 39.7601, "lon": -86.1639, "dome": True},   # Lucas Oil (retractable)
    "JAX": {"lat": 30.3240, "lon": -81.6370, "dome": False},  # EverBank
    "TEN": {"lat": 36.1665, "lon": -86.7713, "dome": False},  # Nissan
    # AFC West
    "DEN": {"lat": 39.7439, "lon": -105.0201, "dome": False}, # Empower Field
    "KC":  {"lat": 39.0490, "lon": -94.4839, "dome": False},  # Arrowhead
    "LV":  {"lat": 36.0909, "lon": -115.1833, "dome": True},  # Allegiant (dome)
    "LAC": {"lat": 33.9535, "lon": -118.3391, "dome": True},  # SoFi (indoor)
    # NFC East
    "DAL": {"lat": 32.7473, "lon": -97.0945, "dome": True},   # AT&T (retractable)
    "NYG": {"lat": 40.8128, "lon": -74.0742, "dome": False},  # MetLife (Giants)
    "PHI": {"lat": 39.9008, "lon": -75.1675, "dome": False},  # Lincoln Financial
    "WAS": {"lat": 38.9077, "lon": -76.8645, "dome": False},  # Commanders Field
    # NFC North
    "CHI": {"lat": 41.8623, "lon": -87.6167, "dome": False},  # Soldier Field
    "DET": {"lat": 42.3400, "lon": -83.0456, "dome": True},   # Ford Field (dome)
    "GB":  {"lat": 44.5013, "lon": -88.0622, "dome": False},  # Lambeau
    "MIN": {"lat": 44.9737, "lon": -93.2581, "dome": True},   # U.S. Bank (dome)
    # NFC South
    "ATL": {"lat": 33.7554, "lon": -84.4010, "dome": True},   # Mercedes-Benz (dome)
    "CAR": {"lat": 35.2259, "lon": -80.8528, "dome": False},  # Bank of America
    "NO":  {"lat": 29.9511, "lon": -90.0812, "dome": True},   # Caesars Superdome
    "TB":  {"lat": 27.9759, "lon": -82.5033, "dome": False},  # Raymond James
    # NFC West
    "ARI": {"lat": 33.5276, "lon": -112.2626, "dome": True},  # State Farm (retractable)
    "LAR": {"lat": 33.9535, "lon": -118.3391, "dome": True},  # SoFi (indoor)
    "SF":  {"lat": 37.4030, "lon": -121.9696, "dome": False}, # Levi's
    "SEA": {"lat": 47.5952, "lon": -122.3316, "dome": False}, # Lumen Field
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def load_schedules() -> pd.DataFrame:
    paths = sorted(NFLVERSE_DIR.glob(SCHEDULES_GLOB.name))
    if not paths:
        raise SystemExit("No schedules_* files found under data/raw/nflverse/.")
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        raise SystemExit("Failed to read schedules files.")
    df = pd.concat(frames, ignore_index=True)
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # Expect: game_id, season, week, home_team, away_team, gameday or game_date, kickoff or start_time
    # Try to build a kickoff UTC timestamp
    dt_cols = [c for c in ["gameday", "game_date", "start_time", "kickoff", "game_datetime"] if c in df.columns]
    if "game_datetime" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["game_datetime"], errors="coerce", utc=True)
    else:
        # combine date + time if available; otherwise parse what we can
        date_col = next((c for c in ["gameday", "game_date"] if c in df.columns), None)
        time_col = next((c for c in ["start_time", "kickoff"] if c in df.columns), None)
        if date_col and time_col:
            df["kickoff_utc"] = pd.to_datetime(df[date_col] + " " + df[time_col], errors="coerce", utc=True)
        else:
            df["kickoff_utc"] = pd.to_datetime(df.get("gameday", df.get("game_date")), errors="coerce", utc=True)
    return df


def choose_games(df: pd.DataFrame, season: int | None, week: int | None) -> pd.DataFrame:
    base = df.copy()
    if season is not None:
        base = base[base["season"] == season]
    if week is not None and "week" in base.columns:
        base = base[base["week"] == week]
    if season is None and week is None:
        # default: next 10 days
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        soon = now + pd.Timedelta(days=10)
        base = base[(base["kickoff_utc"] >= now) & (base["kickoff_utc"] <= soon)]
    # keep necessary columns
    keep = [c for c in ["game_id", "season", "week", "home_team", "away_team", "stadium", "roof", "kickoff_utc"] if c in base.columns]
    return base[keep].dropna(subset=["kickoff_utc"])


def fetch_hour_forecast(lat: float, lon: float, when_utc: pd.Timestamp) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,precipitation,cloudcover,windspeed_10m",
        "forecast_days": 16,  # max
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    hours = data.get("hourly", {})
    times = hours.get("time", [])
    # round to the nearest hour
    target = when_utc.floor("H")
    # find index
    try:
        idx = times.index(target.strftime("%Y-%m-%dT%H:00"))
    except ValueError:
        # if target hour not present, pick closest hour
        if not times:
            return {}
        # find closest
        tseries = pd.to_datetime(times, utc=True)
        ix = int((abs(tseries - target)).argmin())
        idx = ix
    out = {
        "temp_c": hours.get("temperature_2m", [None])[idx],
        "wind_kph": hours.get("windspeed_10m", [None])[idx],
        "precip_mm": hours.get("precipitation", [None])[idx],
        "precip_prob": hours.get("precipitation_probability", [None])[idx],
        "cloudcover": hours.get("cloudcover", [None])[idx],
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None, help="Season (e.g., 2025)")
    parser.add_argument("--week", type=int, default=None, help="Week number (e.g., 1)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    sched = load_schedules()
    games = choose_games(sched, args.season, args.week)

    if games.empty:
        print("No games matched the selection window.")
        # still write an empty latest file for predictability
        pd.DataFrame(columns=[
            "game_id","season","week","home_team","away_team",
            "kickoff_utc","is_dome","lat","lon","temp_c","wind_kph",
            "precip_mm","precip_prob","cloudcover","source","fetched_at_utc"
        ]).to_csv(OUT_LATEST, index=False)
        return

    rows = []
    for _, g in games.iterrows():
        home = str(g["home_team"]).upper()
        info = STADIUMS.get(home)
        roof_val = str(g.get("roof", "")).strip().lower() if "roof" in g else ""
        is_dome = bool(info and info["dome"]) or (roof_val in {"dome", "indoor", "closed"} )

        lat = info["lat"] if info else None
        lon = info["lon"] if info else None

        weather = {"temp_c": None, "wind_kph": None, "precip_mm": None, "precip_prob": None, "cloudcover": None}

        if is_dome:
            # Neutralize weather for closed stadiums
            weather = {"temp_c": 21.0, "wind_kph": 0.0, "precip_mm": 0.0, "precip_prob": 0, "cloudcover": 0}
        elif lat is not None and lon is not None and pd.notna(g["kickoff_utc"]):
            try:
                weather = fetch_hour_forecast(float(lat), float(lon), pd.to_datetime(g["kickoff_utc"], utc=True))
                time.sleep(args.sleep)
            except Exception as e:
                sys.stderr.write(f"Weather fetch failed for game {g.get('game_id','?')} ({home}) : {e}\n")

        rows.append({
            "game_id": g.get("game_id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "home_team": home,
            "away_team": g.get("away_team"),
            "kickoff_utc": pd.to_datetime(g["kickoff_utc"], utc=True),
            "is_dome": int(is_dome),
            "lat": lat,
            "lon": lon,
            **weather,
            "source": "open-meteo",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["season","week","kickoff_utc"]).reset_index(drop=True)

    # Write latest
    out.to_csv(OUT_LATEST, index=False)
    print(f"Wrote: {OUT_LATEST}")

    # Optional seasonal-week file
    if args.season is not None and args.week is not None:
        path_sw = WEATHER_DIR / f"game_weather_{args.season}_wk{args.week}.csv"
        out.to_csv(path_sw, index=False)
        print(f"Wrote: {path_sw}")


if __name__ == "__main__":
    main()
