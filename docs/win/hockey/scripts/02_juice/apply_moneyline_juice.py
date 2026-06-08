#!/usr/bin/env python3
# docs/win/hockey/scripts/02_juice/apply_moneyline_juice.py

import glob
import math
import traceback
from datetime import datetime, UTC
from pathlib import Path
import sys

import pandas as pd

INPUT_DIR  = Path("docs/win/hockey/01_merge/01_merguiced")
OUTPUT_DIR = Path("docs/win/hockey/02_juice")
JUICE_FILE = Path("config/hockey/nhl/nhl_moneyline_juice.csv")

ERROR_DIR = Path("docs/win/hockey/errors/02_juice")
LOG_FILE  = ERROR_DIR / "apply_moneyline_juice.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  files_found    : {summary['files_found']}",
        f"  files_written  : {summary['files_written']}",
        f"  total_rows     : {summary['total_rows']}",
        f"  applied        : {summary['applied']}",
        f"  skipped_bad    : {summary['skipped_bad']}",
        f"  skipped_noband : {summary['skipped_noband']}",
        f"  errors         : {summary['errors']}",
        "",
        f"  {'file':<45} {'rows':>5} {'applied':>8} {'skipped_bad':>12} {'skipped_noband':>15}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<45} {pf['rows']:>5} {pf['applied']:>8} "
            f"{pf['skipped_bad']:>12} {pf['skipped_noband']:>15}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# JUICE LOOKUP
# =========================

def find_band_row(juice_df, american, fav_ud, venue):
    band = juice_df[
        (juice_df["band_min"] <= american) &
        (american <= juice_df["band_max"]) &
        (juice_df["fav_ud"] == fav_ud) &
        (juice_df["venue"] == venue)
    ]
    if len(band) != 1:
        return None
    return float(band.iloc[0]["extra_juice"])


# =========================
# ROW PROCESSOR
# =========================

def process_row(df, juice_df, idx, row):
    try:
        home_american = float(row["home_dk_moneyline_american"])
        away_american = float(row["away_dk_moneyline_american"])
        home_fair     = float(row["home_fair_decimal_moneyline"])
        away_fair     = float(row["away_fair_decimal_moneyline"])
    except Exception:
        _log(f"row={idx} reason=conversion_failed", "SKIP")
        return df, "bad"

    if not math.isfinite(home_fair) or home_fair <= 1:
        _log(f"row={idx} reason=invalid_home_fair val={home_fair}", "SKIP")
        return df, "bad"
    if not math.isfinite(away_fair) or away_fair <= 1:
        _log(f"row={idx} reason=invalid_away_fair val={away_fair}", "SKIP")
        return df, "bad"

    home_fav_ud = "favorite" if home_american < 0 else "underdog"
    away_fav_ud = "favorite" if away_american < 0 else "underdog"

    home_extra = find_band_row(juice_df, home_american, home_fav_ud, "home")
    away_extra = find_band_row(juice_df, away_american, away_fav_ud, "away")

    if home_extra is None or away_extra is None:
        _log(f"row={idx} reason=band_lookup_failed home={home_american} away={away_american}", "SKIP")
        return df, "noband"

    home_juiced_decimal = home_fair * (1 - home_extra)
    away_juiced_decimal = away_fair * (1 - away_extra)

    if home_juiced_decimal <= 1 or away_juiced_decimal <= 1:
        _log(f"row={idx} reason=invalid_juiced_decimal home={home_juiced_decimal} away={away_juiced_decimal}", "SKIP")
        return df, "bad"

    home_juiced_prob = 1 / home_juiced_decimal
    away_juiced_prob = 1 / away_juiced_decimal

    total = home_juiced_prob + away_juiced_prob
    if not math.isfinite(total) or total <= 0:
        _log(f"row={idx} reason=invalid_normalization_total val={total}", "SKIP")
        return df, "bad"

    home_normalized = home_juiced_prob / total
    away_normalized = away_juiced_prob / total

    df.at[idx, "home_juiced_decimal_moneyline"]  = home_juiced_decimal
    df.at[idx, "away_juiced_decimal_moneyline"]  = away_juiced_decimal
    df.at[idx, "home_juiced_prob_moneyline"]      = home_juiced_prob
    df.at[idx, "away_juiced_prob_moneyline"]      = away_juiced_prob
    df.at[idx, "home_normalized_prob_moneyline"]  = home_normalized
    df.at[idx, "away_normalized_prob_moneyline"]  = away_normalized

    return df, "ok"


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== apply_moneyline_juice RUN {_now()} ===\n")

    summary = {
        "files_found": 0, "files_written": 0, "total_rows": 0,
        "applied": 0, "skipped_bad": 0, "skipped_noband": 0, "errors": 0,
    }
    per_file = []

    for f in OUTPUT_DIR.glob("*moneyline.csv"):
        f.unlink()

    try:
        _log(f"INPUT_DIR : {INPUT_DIR}")
        _log(f"JUICE_FILE: {JUICE_FILE}")

        juice_df = pd.read_csv(JUICE_FILE)
        juice_df["band_min"]    = pd.to_numeric(juice_df["band_min"],    errors="coerce")
        juice_df["band_max"]    = pd.to_numeric(juice_df["band_max"],    errors="coerce")
        juice_df["extra_juice"] = pd.to_numeric(juice_df["extra_juice"], errors="coerce")

        files = sorted(glob.glob(str(INPUT_DIR / "*_NHL_moneyline.csv")))
        summary["files_found"] = len(files)
        _log(f"Files found: {len(files)}")

        for file_path in files:
            in_path  = Path(file_path)
            out_path = OUTPUT_DIR / in_path.name
            pf = {"name": in_path.name, "rows": 0, "applied": 0,
                  "skipped_bad": 0, "skipped_noband": 0}

            _log(f"--- FILE: {in_path.name}")

            try:
                df = pd.read_csv(file_path)

                if df.empty:
                    _log(f"{in_path.name} empty — skipping")
                    per_file.append(pf)
                    continue

                for c in ["home_dk_moneyline_american", "away_dk_moneyline_american",
                          "home_fair_decimal_moneyline", "away_fair_decimal_moneyline"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                for c in ["home_juiced_prob_moneyline", "away_juiced_prob_moneyline",
                          "home_juiced_decimal_moneyline", "away_juiced_decimal_moneyline",
                          "home_normalized_prob_moneyline", "away_normalized_prob_moneyline"]:
                    df[c] = pd.NA

                pf["rows"] = len(df)
                summary["total_rows"] += len(df)

                for idx, row in df.iterrows():
                    df, result = process_row(df, juice_df, idx, row)
                    if result == "ok":
                        pf["applied"] += 1
                    elif result == "noband":
                        pf["skipped_noband"] += 1
                    else:
                        pf["skipped_bad"] += 1

                df.to_csv(out_path, index=False)
                summary["files_written"] += 1
                summary["applied"]        += pf["applied"]
                summary["skipped_bad"]    += pf["skipped_bad"]
                summary["skipped_noband"] += pf["skipped_noband"]

                _log(f"{in_path.name} | rows={pf['rows']} applied={pf['applied']} "
                     f"skipped_bad={pf['skipped_bad']} skipped_noband={pf['skipped_noband']}")
                _log(f"WROTE: {out_path}")

            except Exception as e:
                _log(f"{in_path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                summary["errors"] += 1

            per_file.append(pf)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _write_summary(summary, per_file)
        sys.exit(1)

    _write_summary(summary, per_file)
    print("apply_moneyline_juice complete.")


if __name__ == "__main__":
    main()
