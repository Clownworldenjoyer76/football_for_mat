#!/usr/bin/env python3
# docs/win/hockey/scripts/04_select/hockey_select_bets.py

import math
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

INPUT_DIR   = Path("docs/win/hockey/03_edges/ev_kelly")
OUTPUT_DIR  = Path("docs/win/hockey/04_select")
CONFIG_PATH = Path("docs/win/hockey/config/markets.yaml")

ERROR_DIR = Path("docs/win/hockey/errors/04_select")
LOG_FILE  = ERROR_DIR / "select_bets.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_CODE = "NHL"

try:
    with open(CONFIG_PATH, "r") as f:
        _raw = yaml.safe_load(f)
    CONFIG = _raw["markets"]["nhl"]
except FileNotFoundError:
    raise SystemExit(f"Config file not found: {CONFIG_PATH}")
except yaml.YAMLError as e:
    raise SystemExit(f"Malformed YAML in {CONFIG_PATH}: {e}")
except KeyError as e:
    raise SystemExit(f"Missing expected key {e} in {CONFIG_PATH}")

VALID_TOTALS = {"5.5", "6.5"}


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_slate: list) -> None:
    lines = [
        "",
        "=" * 60,
        f"SUMMARY  {_now()}",
        "=" * 60,
        f"  slates_found    : {summary['slates_found']}",
        f"  slates_written  : {summary['slates_written']}",
        f"  total_bets      : {summary['total_bets']}",
        f"  moneyline_bets  : {summary['moneyline_bets']}",
        f"  puck_line_bets  : {summary['puck_line_bets']}",
        f"  total_mkt_bets  : {summary['total_mkt_bets']}",
        f"  skipped_slates  : {summary['skipped_slates']}",
        f"  errors          : {summary['errors']}",
        "",
        f"  {'slate':<30} {'bets':>5} {'ml':>5} {'pl':>5} {'tot':>5} {'status':>10}",
    ]
    for ps in per_slate:
        lines.append(
            f"  {ps['slate']:<30} {ps['bets']:>5} {ps['ml']:>5} "
            f"{ps['pl']:>5} {ps['tot']:>5} {ps['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def fv(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def in_range(val, ranges):
    if val is None:
        return False
    return any(lo <= val <= hi for lo, hi in ranges)


def resolve_total_key(row):
    """Return '5.5' or '6.5' string key, or None if invalid/missing."""
    raw = fv(row.get("total"))
    if raw is None:
        return None
    key = f"{raw:.1f}"
    return key if key in VALID_TOTALS else None


def check_rules_flat(ev, kelly, odds, line, rules):
    """Used for total market — flat band rules, no by_total."""
    if ev is None or kelly is None:
        return False
    if not in_range(ev, rules.get("ev_bands", [])):
        return False
    if not in_range(kelly, rules.get("kelly_bands", [])):
        return False
    if "odds_bands" in rules and not in_range(odds, rules["odds_bands"]):
        return False
    if "line_bands" in rules and not in_range(line, rules["line_bands"]):
        return False
    return True


def check_rules_by_total(ev, kelly, odds, line, side_rules, total_key):
    """Used for moneyline and puck_line — rules vary by game total."""
    if total_key is None:
        return False
    by_total = side_rules.get("by_total", {})
    rules = by_total.get(total_key)
    if rules is None:
        return False
    if ev is None or kelly is None:
        return False
    if not in_range(ev, rules.get("ev_bands", [])):
        return False
    if not in_range(kelly, rules.get("kelly_bands", [])):
        return False
    if "odds_bands" in rules and not in_range(odds, rules["odds_bands"]):
        return False
    if "line_bands" in rules and not in_range(line, rules["line_bands"]):
        return False
    return True


def rescale_prob(p, k=3.0):
    if p is None:
        return None
    try:
        p = float(p)
    except Exception:
        return None
    if not (0 < p < 1):
        return p
    logit = math.log(p / (1 - p))
    return 1 / (1 + math.exp(-k * logit))


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(row, config):
    results = []
    total_key = resolve_total_key(row)
    for side in ["home", "away"]:
        side_rules = config[side]
        if not side_rules["enabled"]:
            continue
        ev         = fv(row.get(f"{side}_ml_ev"))
        kelly      = fv(row.get(f"{side}_ml_kelly"))
        odds       = fv(row.get(f"{side}_dk_moneyline_american"))
        dec        = fv(row.get(f"{side}_dk_decimal_moneyline"))
        model_prob = rescale_prob(fv(row.get(f"{side}_prob")))
        if not check_rules_by_total(ev, kelly, odds, None, side_rules, total_key):
            continue
        results.append({
            "market_type": "moneyline", "bet_side": side,
            "line": "", "take_bet": f"{side}_moneyline",
            "dk_odds_american": odds, "dk_odds_decimal": dec,
            "model_prob": model_prob, "ev": ev, "kelly": kelly,
        })
    return results


def process_puck_line(row, config):
    results = []
    total_key = resolve_total_key(row)
    for side in ["home", "away"]:
        side_rules = config[side]
        if not side_rules["enabled"]:
            continue
        ev         = fv(row.get(f"{side}_puck_line_ev"))
        kelly      = fv(row.get(f"{side}_puck_line_kelly"))
        odds       = fv(row.get(f"{side}_dk_puck_line_american"))
        dec        = fv(row.get(f"{side}_dk_puck_line_decimal"))
        line       = fv(row.get(f"{side}_puck_line"))
        model_prob = rescale_prob(fv(row.get(f"{side}_prob_puck_line")))
        if not check_rules_by_total(ev, kelly, odds, line, side_rules, total_key):
            continue
        results.append({
            "market_type": "puck_line", "bet_side": side,
            "line": line, "take_bet": f"{side}_puck_line",
            "dk_odds_american": odds, "dk_odds_decimal": dec,
            "model_prob": model_prob, "ev": ev, "kelly": kelly,
        })
    return results


def process_total(row, config):
    results = []
    for side in ["over", "under"]:
        rules = config[side]
        if not rules["enabled"]:
            continue
        ev         = fv(row.get(f"{side}_ev"))
        kelly      = fv(row.get(f"{side}_kelly"))
        odds       = fv(row.get(f"dk_total_{side}_american"))
        dec        = fv(row.get(f"dk_total_{side}_decimal"))
        line       = fv(row.get("total"))
        model_prob = rescale_prob(fv(row.get(f"{side}_prob")))
        if not check_rules_flat(ev, kelly, odds, line, rules):
            continue
        results.append({
            "market_type": "total", "bet_side": side,
            "line": line, "take_bet": f"{side}_total",
            "dk_odds_american": odds, "dk_odds_decimal": dec,
            "model_prob": model_prob, "ev": ev, "kelly": kelly,
        })
    return results


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== NHL select_bets RUN {_now()} ===\n")

    summary = {
        "slates_found": 0, "slates_written": 0, "total_bets": 0,
        "moneyline_bets": 0, "puck_line_bets": 0, "total_mkt_bets": 0,
        "skipped_slates": 0, "errors": 0,
    }
    per_slate = []

    for old in OUTPUT_DIR.glob("*.csv"):
        old.unlink()

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    try:
        files = sorted(INPUT_DIR.glob("*_NHL_*.csv"))
        slates: dict = {}
        for fpath in files:
            key = fpath.name.split("_NHL_")[0]
            slates.setdefault(key, []).append(fpath)

        summary["slates_found"] = len(slates)
        _log(f"Slates found: {len(slates)}")

        for slate_key in slates:
            ps = {"slate": slate_key, "bets": 0, "ml": 0, "pl": 0, "tot": 0, "status": "ok"}
            _log(f"--- SLATE: {slate_key}")

            try:
                ml_path = INPUT_DIR / f"{slate_key}_NHL_moneyline.csv"
                pl_path = INPUT_DIR / f"{slate_key}_NHL_puck_line.csv"
                td_path = INPUT_DIR / f"{slate_key}_NHL_total.csv"

                ml_df = pd.read_csv(ml_path) if ml_path.exists() else None
                pl_df = pd.read_csv(pl_path) if pl_path.exists() else None
                td_df = pd.read_csv(td_path) if td_path.exists() else None

                if pl_df is None or pl_df.empty:
                    _log(f"{slate_key} no puck_line file — skipping", "WARN")
                    ps["status"] = "skipped"
                    summary["skipped_slates"] += 1
                    per_slate.append(ps)
                    continue

                final_rows = []
                seen: set  = set()

                for _, row in pl_df.iterrows():
                    game_date = str(row.get("game_date"))
                    away      = str(row.get("away_team"))
                    home      = str(row.get("home_team"))
                    game_id   = row.get("game_id")

                    meta = {"game_date": game_date, "league": LEAGUE_CODE,
                            "away_team": away, "home_team": home, "game_id": game_id}

                    # Total
                    if td_df is not None:
                        for _, trow in td_df[(td_df["away_team"] == away) & (td_df["home_team"] == home)].iterrows():
                            for t in process_total(trow, CONFIG["total"]):
                                key = f"{game_id}_{t['market_type']}_{t['bet_side']}"
                                if key not in seen:
                                    final_rows.append({**meta, **t})
                                    seen.add(key)
                                    ps["tot"] += 1

                    # Puck line
                    for p in process_puck_line(row, CONFIG["puck_line"]):
                        key = f"{game_id}_{p['market_type']}_{p['bet_side']}"
                        if key not in seen:
                            final_rows.append({**meta, **p})
                            seen.add(key)
                            ps["pl"] += 1

                    # Moneyline
                    if ml_df is not None:
                        for _, mrow in ml_df[(ml_df["away_team"] == away) & (ml_df["home_team"] == home)].iterrows():
                            for m in process_moneyline(mrow, CONFIG["moneyline"]):
                                key = f"{game_id}_{m['market_type']}_{m['bet_side']}"
                                if key not in seen:
                                    final_rows.append({**meta, **m})
                                    seen.add(key)
                                    ps["ml"] += 1

                ps["bets"] = len(final_rows)

                if final_rows:
                    out_path = OUTPUT_DIR / f"{slate_key}_NHL.csv"
                    pd.DataFrame(final_rows).to_csv(out_path, index=False)
                    summary["slates_written"]  += 1
                    summary["total_bets"]      += len(final_rows)
                    summary["moneyline_bets"]  += ps["ml"]
                    summary["puck_line_bets"]  += ps["pl"]
                    summary["total_mkt_bets"]  += ps["tot"]
                    _log(f"WROTE: {out_path.name} ({len(final_rows)} bets | ml={ps['ml']} pl={ps['pl']} tot={ps['tot']})")
                else:
                    _log(f"{slate_key} no bets passed filters", "WARN")
                    ps["status"] = "no_bets"

            except Exception as e:
                _log(f"{slate_key} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                ps["status"] = "error"
                summary["errors"] += 1

            per_slate.append(ps)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    _write_summary(summary, per_slate)
    print("hockey select_bets complete.")


if __name__ == "__main__":
    main()
