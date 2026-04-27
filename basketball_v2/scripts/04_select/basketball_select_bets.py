#!/usr/bin/env python3
# docs/win/basketball/scripts/04_select/basketball_select_bets.py

import re
import traceback
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

INPUT_DIR   = Path("docs/win/basketball/03_edges/ev_kelly")
SELECT_DIR  = Path("docs/win/basketball/04_select")
DAILY_DIR   = SELECT_DIR / "daily_slate"
CONFIG_PATH = Path("docs/win/basketball/config/markets.yaml")

ERROR_DIR = Path("docs/win/basketball/errors/04_select")
LOG_FILE  = ERROR_DIR / "select_bets.txt"

SELECT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEBUG_COUNTS = defaultdict(int)


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
        f"  files_processed : {summary['files_processed']}",
        f"  total_selected  : {summary['total_selected']}",
        f"  nba_bets        : {summary['nba_bets']}",
        f"  ncaab_bets      : {summary['ncaab_bets']}",
        f"  skipped         : {summary['skipped']}",
        f"  errors          : {summary['errors']}",
        "",
        "--- Debug Filter Counts ---",
    ]
    for k, v in sorted(DEBUG_COUNTS.items()):
        lines.append(f"  {k:<20} : {v}")
    lines += [
        "",
        f"  {'file':<50} {'market':<10} {'league':>6} {'selected':>9} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<50} {pf['market']:<10} {pf['league']:>6} "
            f"{pf['selected']:>9} {pf['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fv(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def in_bands(value, bands):
    if value is None:
        DEBUG_COUNTS["fail_null"] += 1
        return False
    ok = any(lo <= value <= hi for lo, hi in bands)
    if not ok:
        DEBUG_COUNTS["fail_band"] += 1
    return ok


def violates_exclude_rules(ev, kelly, odds, line, rules):
    for r in rules.get("exclude_rules", []):
        if "ev_min"   in r and (ev   is None or ev   < r["ev_min"]):   continue
        if "ev_max"   in r and (ev   is None or ev   > r["ev_max"]):   continue
        if "odds_min" in r and (odds is None or odds < r["odds_min"]): continue
        if "odds_max" in r and (odds is None or odds > r["odds_max"]): continue
        if "line_min" in r and (line is None or line < r["line_min"]): continue
        if "line_max" in r and (line is None or line > r["line_max"]): continue
        return True
    return False


def ev_ok(ev, cfg):
    if ev is None:
        DEBUG_COUNTS["fail_null"] += 1
        return False
    if ev < cfg["ev_min"]:
        DEBUG_COUNTS["fail_ev_min"] += 1
        return False
    if ev > cfg.get("ev_max", float("inf")):
        DEBUG_COUNTS["fail_ev_max"] += 1
        return False
    return True


def kelly_ok(kelly, cfg):
    if kelly is None:
        DEBUG_COUNTS["fail_null"] += 1
        return False
    if not (cfg["kelly_min"] <= kelly <= cfg["kelly_max"]):
        DEBUG_COUNTS["fail_kelly"] += 1
        return False
    return True


def detect_league(filename):
    name = filename.lower()
    if "ncaab" in name:
        return "NCAAB"
    if "nba" in name:
        return "NBA"
    raise ValueError(f"Unknown league in filename: {filename}")


def detect_market_type(filename):
    name = filename.lower()
    if "moneyline" in name: return "moneyline"
    if "spread"    in name: return "spread"
    if "total"     in name: return "total"
    return ""


def extract_date(filename):
    m = re.search(r"\d{4}_\d{2}_\d{2}", filename)
    return m.group(0) if m else None


def market_cfg(league, market_type):
    try:
        return CONFIG["markets"][league.lower()][market_type]
    except KeyError:
        raise KeyError(f"No config: league={league!r} market_type={market_type!r}")


def pick_side(valid_sides, preference):
    if not valid_sides:
        return None
    return max(valid_sides,
               key=lambda x: x["kelly"] if preference == "best_kelly" else x["ev"])


def moneyline(row, league):
    cfg  = market_cfg(league, "moneyline")
    if not cfg.get("enabled", True):
        return False, "", "", 0
    pref = cfg.get("pick_preference", "best_ev")
    sides = []
    for side in ["home", "away"]:
        scfg  = cfg[side]
        if not scfg.get("enabled", True):
            continue
        odds  = fv(row.get(f"{side}_dk_moneyline_american"))
        ev    = fv(row.get(f"{side}_ml_ev"))
        kelly = fv(row.get(f"{side}_ml_kelly"))
        if (in_bands(odds, scfg["odds_bands"])
                and ev_ok(ev, scfg)
                and kelly_ok(kelly, scfg)
                and not violates_exclude_rules(ev, kelly, odds, None, scfg)):
            sides.append({"side": side, "line": odds, "ev": ev, "kelly": kelly})
    pick = pick_side(sides, pref)
    return (True, pick["side"], pick["line"], pick["ev"]) if pick else (False, "", "", 0)


def spread(row, league):
    cfg  = market_cfg(league, "spread")
    if not cfg.get("enabled", True):
        return False, "", "", 0
    pref = cfg.get("pick_preference", "best_ev")
    sides = []
    for side in ["home", "away"]:
        scfg  = cfg[side]
        if not scfg.get("enabled", True):
            continue
        line  = fv(row.get(f"{side}_spread"))
        odds  = fv(row.get(f"{side}_dk_spread_american"))
        ev    = fv(row.get(f"{side}_spread_ev"))
        kelly = fv(row.get(f"{side}_spread_kelly"))
        if (in_bands(line, scfg["line_bands"])
                and in_bands(odds, scfg.get("odds_bands", [[-10000, 10000]]))
                and ev_ok(ev, scfg)
                and kelly_ok(kelly, scfg)
                and not violates_exclude_rules(ev, kelly, odds, line, scfg)):
            sides.append({"side": side, "line": line, "ev": ev, "kelly": kelly})
    pick = pick_side(sides, pref)
    return (True, pick["side"], pick["line"], pick["ev"]) if pick else (False, "", "", 0)


def total(row, league):
    cfg  = market_cfg(league, "total")
    if not cfg.get("enabled", True):
        return False, "", "", 0
    pref = cfg.get("pick_preference", "best_ev")
    line = fv(row.get("total"))
    sides = []
    for side in ["over", "under"]:
        scfg  = cfg[side]
        if not scfg.get("enabled", True):
            continue
        odds  = fv(row.get(f"dk_total_{side}_american"))
        ev    = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        if (in_bands(line, scfg["line_bands"])
                and in_bands(odds, scfg.get("odds_bands", [[-10000, 10000]]))
                and ev_ok(ev, scfg)
                and kelly_ok(kelly, scfg)
                and not violates_exclude_rules(ev, kelly, odds, line, scfg)):
            sides.append({"side": side, "line": line, "ev": ev, "kelly": kelly})
    pick = pick_side(sides, pref)
    return (True, pick["side"], pick["line"], pick["ev"]) if pick else (False, "", "", 0)


def process_file(file):
    df = pd.read_csv(file)
    if df.empty:
        _log(f"EMPTY: {file.name}", "WARN")
        return pd.DataFrame(), 0

    league      = detect_league(file.name)
    market_type = detect_market_type(file.name)
    game_date   = extract_date(file.name)

    _log(f"--- FILE: {file.name}  league={league} market={market_type} rows={len(df)}")

    rows = []
    for _, row in df.iterrows():
        if market_type == "moneyline":
            ok, side, line, ev = moneyline(row, league)
        elif market_type == "spread":
            ok, side, line, ev = spread(row, league)
        else:
            ok, side, line, ev = total(row, league)

        if ok:
            DEBUG_COUNTS["selected"] += 1
            r = row.to_dict()
            r.update({"bet_side": side, "line": line, "selected_ev": ev,
                      "market_type": market_type, "market": league,
                      "league": "basketball", "game_date": game_date})
            rows.append(r)
        else:
            DEBUG_COUNTS["rejected"] += 1

    _log(f"{file.name} | {len(rows)} selected from {len(df)} rows")
    return pd.DataFrame(rows), len(rows)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball select_bets RUN {_now()} ===\n")

    summary = {
        "files_processed": 0, "total_selected": 0,
        "nba_bets": 0, "ncaab_bets": 0,
        "skipped": 0, "errors": 0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {SELECT_DIR}")

    try:
        input_files = sorted(INPUT_DIR.glob("*.csv"))
        _log(f"Files found: {len(input_files)}")

        dfs = []
        for file in input_files:
            pf = {"name": file.name, "market": detect_market_type(file.name),
                  "league": "?", "selected": 0, "status": "ok"}
            try:
                pf["league"] = detect_league(file.name)
            except ValueError:
                pass

            try:
                df, n_selected = process_file(file)
                pf["selected"] = n_selected
                summary["files_processed"] += 1
                summary["total_selected"]  += n_selected

                if pf["league"] == "NBA":
                    summary["nba_bets"] += n_selected
                elif pf["league"] == "NCAAB":
                    summary["ncaab_bets"] += n_selected

                if not df.empty:
                    dfs.append(df)

            except ValueError as e:
                _log(f"{file.name} SKIP: {e}", "WARN")
                pf["status"] = "skipped"
                summary["skipped"] += 1
            except Exception as e:
                _log(f"{file.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                pf["status"] = "error"
                summary["errors"] += 1

            per_file.append(pf)

        final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if final_df.empty:
            _log("No selections generated", "WARN")
            _write_summary(summary, per_file)
            return

        nba_df   = final_df[final_df["market"] == "NBA"]
        ncaab_df = final_df[final_df["market"] == "NCAAB"]

        nba_path   = DAILY_DIR / "nba_selected.csv"
        ncaab_path = DAILY_DIR / "ncaab_selected.csv"

        nba_df.to_csv(nba_path,   index=False)
        ncaab_df.to_csv(ncaab_path, index=False)

        _log(f"WROTE: {nba_path}   ({len(nba_df)} rows)")
        _log(f"WROTE: {ncaab_path} ({len(ncaab_df)} rows)")

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    _write_summary(summary, per_file)
    print("basketball select_bets complete.")


if __name__ == "__main__":
    main()
