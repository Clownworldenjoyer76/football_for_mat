#!/usr/bin/env python3
"""Prepare the cumulative NFL graded-bet master for reporting."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
INPUT_FILE = NFL_ROOT / "04_final_results" / "results" / "graded" / "NFL_final.csv"
OUTPUT_DIR = NFL_ROOT / "04_final_results" / "intermediate"
OUTPUT_FILE = OUTPUT_DIR / "work_nfl.csv"


def to_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except Exception:
        return None


def side_group(row: pd.Series) -> str:
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    if market in {"moneyline", "spread"}:
        return {"home": "HOME", "away": "AWAY"}.get(side, "")
    if market == "total":
        return {"over": "OVER", "under": "UNDER"}.get(side, "")
    return ""


def day_night(row: pd.Series) -> str:
    raw = str(row.get("edt_time", "") or row.get("game_time", "")).strip()
    if not raw:
        return ""
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return "Day" if parsed.hour < 17 else "Night"
        except ValueError:
            continue
    return ""


def ev_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value < 0: return "<0"
    if value < 0.01: return "0.00_to_0.0099"
    if value < 0.02: return "0.01_to_0.0199"
    if value < 0.03: return "0.02_to_0.0299"
    if value < 0.04: return "0.03_to_0.0399"
    if value < 0.05: return "0.04_to_0.0499"
    if value < 0.075: return "0.05_to_0.0749"
    if value < 0.10: return "0.075_to_0.0999"
    return "0.10_plus"


def odds_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= -200: return "minus_200_or_lower"
    if value <= -150: return "minus_199_to_minus_150"
    if value <= -125: return "minus_149_to_minus_125"
    if value <= -110: return "minus_124_to_minus_110"
    if value <= -101: return "minus_109_to_minus_101"
    if value <= 100: return "minus_100_to_plus_100"
    if value <= 125: return "plus_101_to_plus_125"
    if value <= 150: return "plus_126_to_plus_150"
    if value <= 200: return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def kelly_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= 0: return "zero_or_below"
    if value < 0.01: return "0.001_to_0.0099"
    if value < 0.02: return "0.01_to_0.0199"
    if value < 0.03: return "0.02_to_0.0299"
    if value < 0.05: return "0.03_to_0.0499"
    if value < 0.10: return "0.05_to_0.0999"
    if value < 0.15: return "0.10_to_0.1499"
    if value < 0.20: return "0.15_to_0.1999"
    return "0.20_plus"


def model_prob_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    pct = value * 100.0 if value <= 1.0 else value
    if pct < 50: return "<50"
    if pct < 55: return "50_to_54.9"
    if pct < 60: return "55_to_59.9"
    if pct < 65: return "60_to_64.9"
    if pct < 70: return "65_to_69.9"
    if pct < 75: return "70_to_74.9"
    if pct < 80: return "75_to_79.9"
    return "80_plus"


def spread_range_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    absolute = abs(value)
    if absolute <= 2.5: return "0_to_2.5"
    if absolute <= 3.5: return "3_to_3.5"
    if absolute <= 6.5: return "4_to_6.5"
    if absolute <= 9.5: return "7_to_9.5"
    if absolute <= 13.5: return "10_to_13.5"
    return "14_plus"


def spread_line_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    return f"{value:+g}"


def total_range_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    if value <= 37.5: return "37.5_or_lower"
    if value <= 40.5: return "38_to_40.5"
    if value <= 43.5: return "41_to_43.5"
    if value <= 46.5: return "44_to_46.5"
    if value <= 49.5: return "47_to_49.5"
    if value <= 52.5: return "50_to_52.5"
    return "53_plus"


def total_line_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None: return "UNBUCKETED"
    return f"{value:g}"


def effective_kelly(row: pd.Series) -> tuple[float | None, str]:
    selected = to_float(row.get("kelly"))
    if selected is not None:
        return selected, "selected_kelly"
    full = to_float(row.get("full_kelly"))
    if full is not None:
        return full, "full_kelly_fallback"
    return None, ""


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ["market_type", "bet_side"]:
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].astype(str).str.strip().str.lower()

    if "bet_result" not in work.columns:
        work["bet_result"] = ""
    work["bet_result"] = work["bet_result"].astype(str).str.strip().str.title()

    work["side_group"] = work.apply(side_group, axis=1)
    work["odds_value"] = work.get("odds_american", pd.Series(index=work.index, dtype=object)).map(to_float)
    work["ev_value"] = work.get("ev", pd.Series(index=work.index, dtype=object)).map(to_float)
    work["model_prob_value"] = work.get("model_prob", pd.Series(index=work.index, dtype=object)).map(to_float)

    kelly_pairs = work.apply(effective_kelly, axis=1)
    work["kelly_value"] = [pair[0] for pair in kelly_pairs]
    work["kelly_value_source"] = [pair[1] for pair in kelly_pairs]

    work["spread_value"] = work.apply(
        lambda row: to_float(row.get("line")) if row.get("market_type") == "spread" else None,
        axis=1,
    )
    work["total_value"] = work.apply(
        lambda row: to_float(row.get("line")) if row.get("market_type") == "total" else None,
        axis=1,
    )
    work["day_night"] = work.apply(day_night, axis=1)

    work["ev_bucket"] = work["ev_value"].map(ev_bucket)
    work["odds_bucket"] = work["odds_value"].map(odds_bucket)
    work["kelly_bucket"] = work["kelly_value"].map(kelly_bucket)
    work["win_prob_bucket"] = work["model_prob_value"].map(model_prob_bucket)
    work["spread_range_bucket"] = work["spread_value"].map(spread_range_bucket)
    work["spread_line_bucket"] = work["spread_value"].map(spread_line_bucket)
    work["total_range_bucket"] = work["total_value"].map(total_range_bucket)
    work["total_line_bucket"] = work["total_value"].map(total_line_bucket)

    work["week_label"] = work.get("week", pd.Series(index=work.index, dtype=object)).map(
        lambda value: f"Week {str(value).strip()}" if str(value).strip() else "UNBUCKETED"
    )
    return work


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    frame = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
    work = prepare(frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work.to_csv(OUTPUT_FILE, index=False, lineterminator="\n")
    print(f"NFL analyze complete. rows={len(work)} output={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
