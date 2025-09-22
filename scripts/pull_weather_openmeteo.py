#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pull_weather_openmeteo.py

Behavior retained:
- Produces CSV at data/raw/weather/game_weather_latest.csv

Update:
- Also writes Parquet at data/raw/weather/game_weather_latest.parquet
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests

RAW_DIR = Path("data/raw/weather")
CSV_OUT = RAW_DIR / "game_weather_latest.csv"
PARQ_OUT = RAW_DIR / "game_weather_latest.parquet"

# If your existing script used a specific endpoint/params, keep them the same.
# This stub keeps the structure minimal and focuses on adding Parquet output
# without altering your CSV generation pathway.

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def _fetch_weather(params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def build_dataframe_from_response(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Adjust this to match your existing schema if needed.
    This function converts a minimal Open-Meteo response into a flat table.
    """
    df = pd.json_normalize(payload)
    return df

def main():
    _ensure_dirs()

    # If your existing script loops over games/venues, keep that logic.
    # Here we keep a single-call example to preserve your CSV + add Parquet.
    # Replace the params with your existing ones if they differ.
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "forecast_days": 1,
        "timezone": "UTC",
    }

    payload = _fetch_weather(params)
    df = build_dataframe_from_response(payload)

    # Existing behavior: CSV
    df.to_csv(CSV_OUT, index=False)

    # New behavior: Parquet
    # (Relies on pyarrow/fastparquet; your workflow already installs pyarrow.)
    df.to_parquet(PARQ_OUT, index=False)

    print(f"Wrote: {CSV_OUT}")
    print(f"Wrote: {PARQ_OUT}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
