#!/usr/bin/env python3
"""
Pull per-game weather using Open-Meteo (no API key).
- Joins by game_id/home_team from schedules to stadium coords.
- Handles domes/retractables by neutralizing weather.
- Writes:
    data/raw/weather/game_weather_latest.csv
    data/raw/weather/game_weather_{season}_wk{week}.csv   (if --season/--week passed)

Usage:
  python scripts/pull_weather_openmeteo.py --season 2025 --week 1
  python scripts/pull_weather_openmeteo.py                   # next 10 days
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
import sys
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
NFLVERSE_DIR = RAW_DIR / "nflverse"
WEATHER_DIR = RAW_DIR / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULES_GLOB = NFLVERSE_DIR / "schedules_*.csv.gz"
OUT_LATEST = WEATHER_DIR / "game_weather_latest.csv"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Stadium coordinates & roof (approx; good enough for weather)
STADIUMS = {
    "BUF": {"lat": 42.7738, "lon": -78.7875, "dome": False},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "dome": False},
    "NE":  {"lat": 42.0909, "lon": -71.2643, "dome": False},
    "NYJ": {"lat": 40.8128, "lon": -74.0742, "dome": False},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "dome": False},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "dome": False},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "dome": False},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "dome": False},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "dome": True},
    "IND": {"lat": 39.7601, "lon": -86.1639, "dome": True},
    "JAX": {"lat": 30.3240, "lon": -81.6370, "dome": False},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "dome": False},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "dome": False},
    "KC":  {"lat": 39.0490, "lon": -94.4839, "dome": False},
    "LV":  {"lat": 36.0909, "lon": -115.1833, "dome": True},
    "LAC": {"lat": 33.9535, "lon": -118.3391, "dome": True},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "dome": True},
    "NYG": {"lat": 40.8128, "lon": -74.0742, "dome": False},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "dome": False},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "dome": False},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "dome": False},
    "DET": {"lat": 42.3400, "lon": -83.0456, "dome": True},
    "GB":  {"lat": 44.5013, "lon": -88.0622, "dome": False},
    "MIN": {"lat": 44.9737, "lon": -93.2581, "dome": True},
    "ATL": {"lat": 33.7554, "lon": -84.4010, "dome": True},
    "CAR": {"lat": 35.2259, "lon": -80.8528, "dome": False},
    "NO":  {"lat": 29.9511, "lon": -90.0812, "dome": True},
    "TB":  {"lat": 27.9759, "lon": -82.5033, "dome": False},
    "ARI": {"lat": 33.5276, "lon": -112.2626, "dome": True},
    "LAR": {"lat": 33.9535, "lon": -118.3391, "dome": True},
    "SF":  {"lat": 37.4030, "lon": -121.9696, "dome": False},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "dome": False},
}

def read_schedules() -> pd.DataFrame:
    paths = sorted(NFLVERSE_DIR.glob(SCHEDULES_GLOB.name))
    if not paths:
        raise SystemExit("No schedules_* files under data/raw/nflverse/")
    dfs = [pd.read_csv(p) for p in paths if p.exists()]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    if "game_datetime" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["game_datetime"], errors="coerce", utc=True)
    else:
        date_col = next((c for c in ["gameday","game_date"] if c in df.columns), None)
        time_col = next((c for c in ["start_time","kickoff"] if c in df.columns), None)
        if date_col and time_col:
            df["kickoff_utc"] = pd.to_datetime(df[date_col]+" "+df[time_col], errors="coerce", utc=True)
        elif date_col:
            df["kickoff_utc"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        else:
            df["kickoff_utc"] = pd.NaT

    for tcol in ["home_team","away_team"]:
        if tcol in df.columns:
            df[tcol] = df[tcol].astype(str).str.upper()
    return df

def select_games(df: pd.DataFrame, season:int|None, week:int|None) -> pd.DataFrame:
    sel = df.copy()
    if season is not None and "season" in sel.columns:
        sel = sel[sel["season"]==season]
    if week is not None and "week" in sel.columns:
        sel = sel[sel["week"]==week]
    if season is None and week is None:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        soon = now + pd.Timedelta(days=10)
        sel = sel[(sel["kickoff_utc"]>=now) & (sel["kickoff_utc"]<=soon)]
    keep = [c for c in ["game_id","season","week","home_team","away_team","kickoff_utc"] if c in sel.columns]
    return sel[keep].dropna(subset=["kickoff_utc"])

def http_get(url:str, params:dict, retries:int=3, backoff:float=0.6) -> dict:
    last=None
    for i in range(retries):
        try:
            r=requests.get(url, params=params, timeout=15)
            if r.status_code==200:
                return r.json()
            last=f"{r.status_code} {r.text[:120]}"
        except Exception as e:
            last=str(e)
        time.sleep(backoff*(2**i))
    raise RuntimeError(f"Open-Meteo request failed: {last}")

def fetch_hour(lat:float, lon:float, when_utc:pd.Timestamp) -> dict:
    params={
        "latitude":lat,"longitude":lon,
        "hourly":"temperature_2m,precipitation_probability,precipitation,cloudcover,windspeed_10m",
        "forecast_days":16,"timezone":"UTC"
    }
    data=http_get(OPEN_METEO_URL, params)
    hrs=data.get("hourly",{})
    times=hrs.get("time",[])
    if not times: return {}
    target=when_utc.floor("H")
    series=pd.to_datetime(times, utc=True, errors="coerce")
    idx=int((series-target).abs().argmin())
    return {
        "temp_c":safe_get(hrs,"temperature_2m",idx),
        "wind_kph":safe_get(hrs,"windspeed_10m",idx),
        "precip_mm":safe_get(hrs,"precipitation",idx),
        "precip_prob":safe_get(hrs,"precipitation_probability",idx),
        "cloudcover":safe_get(hrs,"cloudcover",idx),
    }

def safe_get(d:dict, key:str, idx:int):
    arr=d.get(key,[])
    try: return float(arr[idx])
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,default=None)
    ap.add_argument("--week",type=int,default=None)
    ap.add_argument("--sleep",type=float,default=0.4)
    args=ap.parse_args()

    sched=read_schedules()
    games=select_games(sched,args.season,args.week)
    if games.empty:
        pd.DataFrame(columns=[
            "game_id","season","week","home_team","away_team","kickoff_utc",
            "is_dome","lat","lon","temp_c","wind_kph","precip_mm","precip_prob","cloudcover",
            "source","fetched_at_utc"
        ]).to_csv(OUT_LATEST,index=False)
        print(f"Wrote empty: {OUT_LATEST}"); return

    rows=[]
    for _,g in games.iterrows():
        home=str(g["home_team"])
        info=STADIUMS.get(home)
        is_dome=bool(info and info["dome"])
        lat,lon=(info["lat"],info["lon"]) if info else (None,None)
        weather={"temp_c":None,"wind_kph":None,"precip_mm":None,"precip_prob":None,"cloudcover":None}
        if is_dome:
            weather={"temp_c":21.0,"wind_kph":0.0,"precip_mm":0.0,"precip_prob":0.0,"cloudcover":0.0}
        elif lat is not None and lon is not None and pd.notna(g["kickoff_utc"]):
            try:
                weather=fetch_hour(float(lat),float(lon),pd.to_datetime(g["kickoff_utc"],utc=True))
                time.sleep(args.sleep)
            except Exception as e:
                sys.stderr.write(f"[weather] {home} {g.get('game_id','?')}: {e}\n")
        rows.append({
            "game_id":g.get("game_id"),
            "season":g.get("season"),"week":g.get("week"),
            "home_team":home,"away_team":g.get("away_team"),
            "kickoff_utc":pd.to_datetime(g["kickoff_utc"],utc=True),
            "is_dome":int(is_dome),"lat":lat,"lon":lon,
            **weather,"source":"open-meteo",
            "fetched_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    out=pd.DataFrame(rows).sort_values(["season","week","kickoff_utc"],na_position="last").reset_index(drop=True)
    out.to_csv(OUT_LATEST,index=False)
    print(f"Wrote: {OUT_LATEST}")
    if args.season is not None and args.week is not None:
        sw=WEATHER_DIR/f"game_weather_{args.season}_wk{args.week}.csv"
        out.to_csv(sw,index=False); print(f"Wrote: {sw}")

if __name__=="__main__":
    main()
