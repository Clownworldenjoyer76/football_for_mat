#!/usr/bin/env python3
# docs/win/baseball/scripts/04_select/baseball_select_bets.py

import math
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import yaml

INPUT_DIR = Path("docs/win/baseball/03_edges/ev_kelly")
OUTPUT_DIR = Path("docs/win/baseball/04_select")
CONFIG_PATH = Path("docs/win/baseball/config/markets.yaml")

ERROR_DIR = Path("docs/win/baseball/errors/04_select")
LOG_FILE = ERROR_DIR / "select_bets.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_CODE = "MLB"
PROB_TOLERANCE = 1e-9

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _yaml = yaml.safe_load(f)["markets"]["mlb"]
    CONFIG = _yaml
    FILTERS = _yaml


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
        f"  slates_found                       : {summary['slates_found']}",
        f"  slates_written                     : {summary['slates_written']}",
        f"  total_bets                         : {summary['total_bets']}",
        f"  moneyline_bets                     : {summary['moneyline_bets']}",
        f"  run_line_bets                      : {summary['run_line_bets']}",
        f"  total_mkt_bets                     : {summary['total_mkt_bets']}",
        f"  skipped_slates                     : {summary['skipped_slates']}",
        f"  missing_moneyline                  : {summary['missing_moneyline']}",
        f"  missing_run_line                   : {summary['missing_run_line']}",
        f"  missing_total                      : {summary['missing_total']}",
        f"  row_count_warnings                 : {summary['row_count_warnings']}",
        f"  selected_nonpositive_kelly         : {summary['selected_nonpositive_kelly']}",
        f"  selected_blank_probability_source  : {summary['selected_blank_probability_source']}",
        f"  selected_probability_source_mismatch: {summary['selected_probability_source_mismatch']}",
        f"  selected_adjusted_only_positive    : {summary['selected_adjusted_only_positive']}",
        f"  schema_errors                      : {summary['schema_errors']}",
        f"  rain_excluded                      : {summary['rain_excluded']}",
        f"  sp_sample_excluded                 : {summary['sp_sample_excluded']}",
        f"  low_confidence                     : {summary['low_confidence']}",
        f"  errors                             : {summary['errors']}",
        "",
        "--- Filter Breakdown ---",
    ]

    for market, sides in summary.get("counters", {}).items():
        for side, c in sides.items():
            lines.append(
                f"  {market}-{side:<10} passed={c['passed']:>4} "
                f"ev_fail={c['ev_fail']:>4} kelly_fail={c['kelly_fail']:>4} "
                f"odds_fail={c['odds_fail']:>4} line_fail={c['line_fail']:>4} "
                f"prob_fail={c['prob_fail']:>4} source_fail={c['source_fail']:>4} "
                f"adjusted_only_fail={c['adjusted_only_fail']:>4} missing={c['missing']:>4} "
                f"excluded={c['excluded']:>4}"
            )

    lines += [
        "",
        f"  {'slate':<30} {'bets':>5} {'ml':>5} {'rl':>5} {'tot':>5} {'status':>14}",
    ]

    for ps in per_slate:
        lines.append(
            f"  {ps['slate']:<30} {ps['bets']:>5} {ps['ml']:>5} "
            f"{ps['rl']:>5} {ps['tot']:>5} {ps['status']:>14}"
        )

    status = "SUCCESS" if summary["errors"] == 0 and summary["schema_errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 60]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# SCHEMA VALIDATION
# =========================

REQUIRED_BASE_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

REQUIRED_MONEYLINE_COLUMNS = REQUIRED_BASE_COLUMNS + [
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_decimal_moneyline",
    "away_dk_decimal_moneyline",
    "home_normalized_prob_moneyline",
    "away_normalized_prob_moneyline",
    "home_prob_for_ev",
    "away_prob_for_ev",
    "home_prob_for_kelly",
    "away_prob_for_kelly",
    "home_ev_probability_source",
    "away_ev_probability_source",
    "home_kelly_probability_source",
    "away_kelly_probability_source",
    "home_ml_raw_ev",
    "away_ml_raw_ev",
    "home_ml_adjusted_ev",
    "away_ml_adjusted_ev",
    "home_ml_ev",
    "away_ml_ev",
    "home_ml_kelly",
    "away_ml_kelly",
]

REQUIRED_RUN_LINE_COLUMNS = REQUIRED_BASE_COLUMNS + [
    "home_run_line",
    "away_run_line",
    "home_dk_run_line_american",
    "away_dk_run_line_american",
    "home_dk_run_line_decimal",
    "away_dk_run_line_decimal",
    "home_normalized_prob_run_line",
    "away_normalized_prob_run_line",
    "home_prob_for_ev",
    "away_prob_for_ev",
    "home_prob_for_kelly",
    "away_prob_for_kelly",
    "home_ev_probability_source",
    "away_ev_probability_source",
    "home_kelly_probability_source",
    "away_kelly_probability_source",
    "home_rl_raw_ev",
    "away_rl_raw_ev",
    "home_rl_adjusted_ev",
    "away_rl_adjusted_ev",
    "home_rl_ev",
    "away_rl_ev",
    "home_rl_kelly",
    "away_rl_kelly",
]

REQUIRED_TOTAL_COLUMNS = REQUIRED_BASE_COLUMNS + [
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "over_normalized_prob_total",
    "under_normalized_prob_total",
    "over_prob_for_ev",
    "under_prob_for_ev",
    "over_prob_for_kelly",
    "under_prob_for_kelly",
    "over_ev_probability_source",
    "under_ev_probability_source",
    "over_kelly_probability_source",
    "under_kelly_probability_source",
    "over_raw_ev",
    "under_raw_ev",
    "over_adjusted_ev",
    "under_adjusted_ev",
    "over_ev",
    "under_ev",
    "over_kelly",
    "under_kelly",
]

FORBIDDEN_RUN_LINE_COLUMNS = [
    "home_run_line_prob",
    "away_run_line_prob",
]


def duplicate_columns(columns):
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    return duplicates


def validate_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    dupes = duplicate_columns(list(df.columns))

    if dupes:
        raise ValueError(f"{label} has duplicate columns: {dupes}")


def validate_required_columns(df: pd.DataFrame, required_columns: list, label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_forbidden_columns(df: pd.DataFrame, forbidden_columns: list, label: str) -> None:
    present = [col for col in forbidden_columns if col in df.columns]

    if present:
        raise ValueError(
            f"{label} contains obsolete forbidden columns: {present}. "
            f"Use home_normalized_prob_run_line / away_normalized_prob_run_line."
        )


def read_market_csv(path: Path, required_columns: list, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    validate_no_duplicate_columns(df, label)
    validate_required_columns(df, required_columns, label)

    return df


def validate_selected_output(df: pd.DataFrame, label: str) -> dict:
    counts = {
        "selected_nonpositive_kelly": 0,
        "selected_blank_probability_source": 0,
        "selected_probability_source_mismatch": 0,
        "selected_adjusted_only_positive": 0,
    }

    if df.empty:
        return counts

    kelly = pd.to_numeric(df["kelly"], errors="coerce")
    counts["selected_nonpositive_kelly"] = int((kelly.isna() | (kelly <= 0)).sum())

    ev_src = df["ev_probability_source"].astype(str).str.strip()
    kelly_src = df["kelly_probability_source"].astype(str).str.strip()
    counts["selected_blank_probability_source"] = int(((ev_src == "") | (kelly_src == "") | (ev_src == "nan") | (kelly_src == "nan")).sum())
    counts["selected_probability_source_mismatch"] = int((ev_src != kelly_src).sum())

    prob_ev = pd.to_numeric(df["prob_for_ev"], errors="coerce")
    prob_kelly = pd.to_numeric(df["prob_for_kelly"], errors="coerce")
    prob_mismatch = int(((prob_ev - prob_kelly).abs() > PROB_TOLERANCE).sum())
    counts["selected_probability_source_mismatch"] += prob_mismatch

    raw_ev = pd.to_numeric(df["raw_ev"], errors="coerce")
    adjusted_ev = pd.to_numeric(df["adjusted_ev"], errors="coerce")
    counts["selected_adjusted_only_positive"] = int(((raw_ev <= 0) & (adjusted_ev > 0)).sum())

    failures = {k: v for k, v in counts.items() if v > 0}

    if failures:
        raise ValueError(f"{label} failed selected-bet Step 4 validation: {failures}")

    return counts


def write_output_csv(df: pd.DataFrame, path: Path, label: str) -> dict:
    validate_no_duplicate_columns(df, label)
    counts = validate_selected_output(df, label)
    df.to_csv(path, index=False)
    return counts


def row_count_check(slate: str, market_frames: dict, summary: dict) -> None:
    counts = {
        market: len(df)
        for market, df in market_frames.items()
        if df is not None
    }

    if len(counts) <= 1:
        return

    unique_counts = sorted(set(counts.values()))

    if len(unique_counts) > 1:
        summary["row_count_warnings"] += 1
        _log(f"{slate} market row-count mismatch before selection: {counts}", "WARN")
    else:
        _log(f"{slate} market row-count check OK: {counts}")


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


def iv(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def sv(x):
    try:
        if pd.isna(x):
            return None
        return str(x).strip()
    except Exception:
        return None


def in_range(val, ranges):
    if val is None:
        return False
    return any(lo <= val <= hi for lo, hi in ranges)


def rescale_prob(p, k=3.0):
    if p is None:
        return None
    if not (0 < p < 1):
        return p
    logit = math.log(p / (1 - p))
    return 1 / (1 + math.exp(-k * logit))


def violates_exclude_rules(ev, kelly, odds, line, prob, rules):
    for r in rules.get("exclude_rules", []):
        if "ev_min" in r and (ev is None or ev < r["ev_min"]):
            continue
        if "ev_max" in r and (ev is None or ev > r["ev_max"]):
            continue
        if "kelly_min" in r and (kelly is None or kelly < r["kelly_min"]):
            continue
        if "kelly_max" in r and (kelly is None or kelly > r["kelly_max"]):
            continue
        if "odds_min" in r and (odds is None or odds < r["odds_min"]):
            continue
        if "odds_max" in r and (odds is None or odds > r["odds_max"]):
            continue
        if "line_min" in r and (line is None or line < r["line_min"]):
            continue
        if "line_max" in r and (line is None or line > r["line_max"]):
            continue
        if "prob_min" in r and (prob is None or prob < r["prob_min"]):
            continue
        if "prob_max" in r and (prob is None or prob > r["prob_max"]):
            continue

        if "prob_bands" in r:
            if prob is None or not in_range(prob, r["prob_bands"]):
                continue

        return True

    return False


def check_probability_basis(prob_for_ev, prob_for_kelly, ev_source, kelly_source, counters):
    if not ev_source or not kelly_source:
        counters["source_fail"] += 1
        return False

    if str(ev_source).strip() != str(kelly_source).strip():
        counters["source_fail"] += 1
        return False

    if prob_for_ev is None or prob_for_kelly is None:
        counters["source_fail"] += 1
        return False

    if abs(prob_for_ev - prob_for_kelly) > PROB_TOLERANCE:
        counters["source_fail"] += 1
        return False

    return True


def check_rules(ev, kelly, odds, line, prob, rules, counters):
    if ev is None or kelly is None:
        counters["missing"] += 1
        return False

    if kelly <= 0:
        counters["kelly_fail"] += 1
        return False

    if not in_range(ev, rules.get("ev_bands", [])):
        counters["ev_fail"] += 1
        return False

    if not in_range(kelly, rules.get("kelly_bands", [])):
        counters["kelly_fail"] += 1
        return False

    if "odds_bands" in rules:
        if odds is None:
            counters["odds_fail"] += 1
            return False
        if not in_range(odds, rules["odds_bands"]):
            counters["odds_fail"] += 1
            return False

    if "line_bands" in rules:
        if line is None:
            counters["line_fail"] += 1
            return False
        if not in_range(line, rules["line_bands"]):
            counters["line_fail"] += 1
            return False

    if "prob_bands" in rules:
        if prob is None:
            counters["prob_fail"] += 1
            return False
        if not in_range(prob, rules["prob_bands"]):
            counters["prob_fail"] += 1
            return False

    if "prob_min" in rules and (prob is None or prob < rules["prob_min"]):
        counters["prob_fail"] += 1
        return False

    if "prob_max" in rules and (prob is None or prob > rules["prob_max"]):
        counters["prob_fail"] += 1
        return False

    if violates_exclude_rules(ev, kelly, odds, line, prob, rules):
        counters["excluded"] += 1
        return False

    counters["passed"] += 1
    return True


def init_counter():
    return {
        "passed": 0,
        "ev_fail": 0,
        "kelly_fail": 0,
        "odds_fail": 0,
        "line_fail": 0,
        "prob_fail": 0,
        "source_fail": 0,
        "adjusted_only_fail": 0,
        "excluded": 0,
        "missing": 0,
    }


def select_candidate(candidates, preference, market_name=None, game_id=None):
    if not candidates:
        return []
    if preference == "all":
        return candidates
    if preference == "best_prob":
        return [max(candidates, key=lambda x: x["model_prob"] or -999)]
    return [max(candidates, key=lambda x: x["ev"] or -999)]


def get_market_rows(df: pd.DataFrame, game_id: str, away: str, home: str, game_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "game_id" in df.columns:
        matched = df[df["game_id"].astype(str) == str(game_id)]
        if not matched.empty:
            return matched

    return df[
        (df["away_team"] == away) &
        (df["home_team"] == home) &
        (df["game_date"] == game_date)
    ]


def build_base_games(market_frames: dict) -> pd.DataFrame:
    pieces = []

    for market_name in ["run_line", "moneyline", "total"]:
        df = market_frames.get(market_name)

        if df is None or df.empty:
            continue

        cols = [col for col in REQUIRED_BASE_COLUMNS if col in df.columns]
        piece = df[cols].copy()
        piece["source_market"] = market_name
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=REQUIRED_BASE_COLUMNS + ["source_market"])

    base = pd.concat(pieces, ignore_index=True)
    base = base.drop_duplicates(subset=["game_id"], keep="first")

    return base


def adjusted_only_positive(raw_ev, adjusted_ev) -> bool:
    raw = fv(raw_ev)
    adj = fv(adjusted_ev)

    if raw is None or adj is None:
        return False

    return raw <= 0 and adj > 0


# =========================
# CONTEXT FILTERS
# =========================

def rain_excluded(row) -> bool:
    weather_applicable = iv(row.get("weather_applicable"))

    if weather_applicable == 0:
        return False

    will_it_rain = iv(row.get("will_it_rain"))
    symbol_code = sv(row.get("symbol_code"))

    if FILTERS.get("rain_exclude_on_will_it_rain", True) and will_it_rain == 1:
        return True

    rain_terms = FILTERS.get("rain_symbol_terms", [
        "rain",
        "heavyrain",
        "lightrain",
        "sleet",
        "snow",
        "thunder",
    ])

    if symbol_code:
        symbol = symbol_code.lower()
        if any(str(term).lower() in symbol for term in rain_terms):
            return True

    return False


def sp_sample_excluded_for_total(row) -> bool:
    if not FILTERS.get("sp_sample_exclude_totals", True):
        return False

    home_flag = sv(row.get("home_sp_sample_flag"))
    away_flag = sv(row.get("away_sp_sample_flag"))

    return home_flag == "low" or away_flag == "low"


def is_low_confidence(row) -> int:
    warn = FILTERS.get("lineup_low_sample_warn", 3)
    home_low = fv(row.get("home_low_sample_count"))
    away_low = fv(row.get("away_low_sample_count"))

    if home_low is not None and home_low > warn:
        return 1
    if away_low is not None and away_low > warn:
        return 1
    return 0


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(row, counters):
    candidates = []

    for side in ["home", "away"]:
        rules = CONFIG["moneyline"][side]
        side_counter = counters["moneyline"][side]

        if not rules["enabled"]:
            continue

        ev = fv(row.get(f"{side}_ml_ev"))
        kelly = fv(row.get(f"{side}_ml_kelly"))
        odds = fv(row.get(f"{side}_dk_moneyline_american"))
        dec = fv(row.get(f"{side}_dk_decimal_moneyline"))
        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))
        ev_source = sv(row.get(f"{side}_ev_probability_source"))
        kelly_source = sv(row.get(f"{side}_kelly_probability_source"))
        raw_ev = fv(row.get(f"{side}_ml_raw_ev"))
        adjusted_ev = fv(row.get(f"{side}_ml_adjusted_ev"))
        model_prob = prob_for_ev

        if adjusted_only_positive(raw_ev, adjusted_ev):
            side_counter["adjusted_only_fail"] += 1
            continue

        if not check_probability_basis(prob_for_ev, prob_for_kelly, ev_source, kelly_source, side_counter):
            continue

        if not check_rules(ev, kelly, odds, None, model_prob, rules, side_counter):
            continue

        candidates.append({
            "market_type": "moneyline",
            "bet_side": side,
            "market": "moneyline",
            "side": side,
            "line": "",
            "take_bet": f"{side}_moneyline",
            "dk_odds_american": odds,
            "dk_odds_decimal": dec,
            "model_prob": model_prob,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": ev_source,
            "kelly_probability_source": kelly_source,
            "raw_ev": raw_ev,
            "adjusted_ev": adjusted_ev,
            "ev": ev,
            "kelly": kelly,
        })

    return select_candidate(
        candidates,
        CONFIG["moneyline"].get("pick_preference", "best_ev"),
        "moneyline",
        row.get("game_id"),
    )


def process_run_line(row, counters):
    candidates = []

    for side in ["home", "away"]:
        rules = CONFIG["run_line"][side]
        side_counter = counters["run_line"][side]

        if not rules["enabled"]:
            continue

        ev = fv(row.get(f"{side}_rl_ev"))
        kelly = fv(row.get(f"{side}_rl_kelly"))
        odds = fv(row.get(f"{side}_dk_run_line_american"))
        dec = fv(row.get(f"{side}_dk_run_line_decimal"))
        line = fv(row.get(f"{side}_run_line"))
        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))
        ev_source = sv(row.get(f"{side}_ev_probability_source"))
        kelly_source = sv(row.get(f"{side}_kelly_probability_source"))
        raw_ev = fv(row.get(f"{side}_rl_raw_ev"))
        adjusted_ev = fv(row.get(f"{side}_rl_adjusted_ev"))
        model_prob = prob_for_ev

        if adjusted_only_positive(raw_ev, adjusted_ev):
            side_counter["adjusted_only_fail"] += 1
            continue

        if not check_probability_basis(prob_for_ev, prob_for_kelly, ev_source, kelly_source, side_counter):
            continue

        if not check_rules(ev, kelly, odds, line, model_prob, rules, side_counter):
            continue

        candidates.append({
            "market_type": "run_line",
            "bet_side": side,
            "market": "run_line",
            "side": side,
            "line": line,
            "take_bet": f"{side}_run_line",
            "dk_odds_american": odds,
            "dk_odds_decimal": dec,
            "model_prob": model_prob,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": ev_source,
            "kelly_probability_source": kelly_source,
            "raw_ev": raw_ev,
            "adjusted_ev": adjusted_ev,
            "ev": ev,
            "kelly": kelly,
        })

    return select_candidate(
        candidates,
        CONFIG["run_line"].get("pick_preference", "best_ev"),
        "run_line",
        row.get("game_id"),
    )


def process_total(row, counters):
    candidates = []

    for side in ["over", "under"]:
        rules = CONFIG["total"][side]
        side_counter = counters["total"][side]

        if not rules["enabled"]:
            continue

        ev = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        odds = fv(row.get(f"dk_total_{side}_american"))
        dec = fv(row.get(f"dk_total_{side}_decimal"))
        line = fv(row.get("total"))
        prob_for_ev = fv(row.get(f"{side}_prob_for_ev"))
        prob_for_kelly = fv(row.get(f"{side}_prob_for_kelly"))
        ev_source = sv(row.get(f"{side}_ev_probability_source"))
        kelly_source = sv(row.get(f"{side}_kelly_probability_source"))
        raw_ev = fv(row.get(f"{side}_raw_ev"))
        adjusted_ev = fv(row.get(f"{side}_adjusted_ev"))
        model_prob = prob_for_ev

        if adjusted_only_positive(raw_ev, adjusted_ev):
            side_counter["adjusted_only_fail"] += 1
            continue

        if not check_probability_basis(prob_for_ev, prob_for_kelly, ev_source, kelly_source, side_counter):
            continue

        if not check_rules(ev, kelly, odds, line, model_prob, rules, side_counter):
            continue

        candidates.append({
            "market_type": "total",
            "bet_side": side,
            "market": "total",
            "side": side,
            "line": line,
            "take_bet": f"{side}_total",
            "dk_odds_american": odds,
            "dk_odds_decimal": dec,
            "model_prob": model_prob,
            "prob_for_ev": prob_for_ev,
            "prob_for_kelly": prob_for_kelly,
            "ev_probability_source": ev_source,
            "kelly_probability_source": kelly_source,
            "raw_ev": raw_ev,
            "adjusted_ev": adjusted_ev,
            "ev": ev,
            "kelly": kelly,
        })

    return select_candidate(
        candidates,
        CONFIG["total"].get("pick_preference", "best_ev"),
        "total",
        row.get("game_id"),
    )


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== MLB select_bets RUN {_now()} ===\n")

    summary = {
        "slates_found": 0,
        "slates_written": 0,
        "total_bets": 0,
        "moneyline_bets": 0,
        "run_line_bets": 0,
        "total_mkt_bets": 0,
        "skipped_slates": 0,
        "missing_moneyline": 0,
        "missing_run_line": 0,
        "missing_total": 0,
        "row_count_warnings": 0,
        "selected_nonpositive_kelly": 0,
        "selected_blank_probability_source": 0,
        "selected_probability_source_mismatch": 0,
        "selected_adjusted_only_positive": 0,
        "schema_errors": 0,
        "rain_excluded": 0,
        "sp_sample_excluded": 0,
        "low_confidence": 0,
        "errors": 0,
        "counters": {},
    }

    per_slate = []

    for old in OUTPUT_DIR.glob("*.csv"):
        old.unlink()

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    _log(
        f"Rain filter: will_it_rain={FILTERS.get('rain_exclude_on_will_it_rain', True)} "
        f"symbol_terms={FILTERS.get('rain_symbol_terms')} | "
        f"SP sample exclude totals: {FILTERS.get('sp_sample_exclude_totals')} | "
        f"Lineup low sample warn: {FILTERS.get('lineup_low_sample_warn')}"
    )
    _log("Selection requires kelly > 0 and matching EV/Kelly probability source per selected row.")
    _log("Selection rejects rows where adjusted EV is positive but raw EV is zero/negative.")

    try:
        files = sorted(INPUT_DIR.glob("*_mlb_*.csv"))
        _log(f"Files found: {len(files)}")

        if not files:
            _log("No input files found", "WARN")
            _write_summary(summary, per_slate)
            return

        slates = {}

        for fp in files:
            key = fp.name.split("_mlb_")[0]
            slates.setdefault(key, []).append(fp)

        summary["slates_found"] = len(slates)

        global_counters = {
            "moneyline": {
                "home": init_counter(),
                "away": init_counter(),
            },
            "run_line": {
                "home": init_counter(),
                "away": init_counter(),
            },
            "total": {
                "over": init_counter(),
                "under": init_counter(),
            },
        }

        for slate, _ in slates.items():
            ps = {
                "slate": slate,
                "bets": 0,
                "ml": 0,
                "rl": 0,
                "tot": 0,
                "status": "ok",
            }

            _log(f"--- SLATE: {slate}")

            try:
                ml_path = INPUT_DIR / f"{slate}_mlb_moneyline.csv"
                rl_path = INPUT_DIR / f"{slate}_mlb_run_line.csv"
                tt_path = INPUT_DIR / f"{slate}_mlb_total.csv"

                ml_df = None
                rl_df = None
                tt_df = None

                if ml_path.exists():
                    ml_df = read_market_csv(
                        ml_path,
                        REQUIRED_MONEYLINE_COLUMNS,
                        f"{slate} moneyline input",
                    )
                else:
                    summary["missing_moneyline"] += 1
                    _log(f"{slate} missing moneyline file — continuing without moneyline", "WARN")

                if rl_path.exists():
                    rl_df = read_market_csv(
                        rl_path,
                        REQUIRED_RUN_LINE_COLUMNS,
                        f"{slate} run-line input",
                    )
                    validate_forbidden_columns(
                        rl_df,
                        FORBIDDEN_RUN_LINE_COLUMNS,
                        f"{slate} run-line input",
                    )
                else:
                    summary["missing_run_line"] += 1
                    _log(f"{slate} missing run_line file — continuing without run_line", "WARN")

                if tt_path.exists():
                    tt_df = read_market_csv(
                        tt_path,
                        REQUIRED_TOTAL_COLUMNS,
                        f"{slate} total input",
                    )
                else:
                    summary["missing_total"] += 1
                    _log(f"{slate} missing total file — continuing without total", "WARN")

                market_frames = {
                    "moneyline": ml_df,
                    "run_line": rl_df,
                    "total": tt_df,
                }

                if all(df is None or df.empty for df in market_frames.values()):
                    _log(f"{slate} no usable market files — skipping slate", "WARN")
                    ps["status"] = "skipped"
                    summary["skipped_slates"] += 1
                    per_slate.append(ps)
                    continue

                row_count_check(slate, market_frames, summary)

                base_games = build_base_games(market_frames)

                if base_games.empty:
                    _log(f"{slate} no base games after market load — skipping slate", "WARN")
                    ps["status"] = "skipped"
                    summary["skipped_slates"] += 1
                    per_slate.append(ps)
                    continue

                final = []
                seen = set()

                for _, base_row in base_games.iterrows():
                    game_id = base_row["game_id"]
                    sport = base_row.get("sport", "")
                    game_date = base_row["game_date"]
                    game_time = base_row.get("game_time", "")
                    away = base_row["away_team"]
                    home = base_row["home_team"]

                    base = {
                        "game_id": game_id,
                        "sport": sport,
                        "game_date": game_date,
                        "game_time": game_time,
                        "league": LEAGUE_CODE,
                        "away_team": away,
                        "home_team": home,
                    }

                    context_row = None

                    rl_matches = get_market_rows(rl_df, game_id, away, home, game_date)
                    ml_matches = get_market_rows(ml_df, game_id, away, home, game_date)
                    tt_matches = get_market_rows(tt_df, game_id, away, home, game_date)

                    if not rl_matches.empty:
                        context_row = rl_matches.iloc[0]
                    elif not tt_matches.empty:
                        context_row = tt_matches.iloc[0]
                    elif not ml_matches.empty:
                        context_row = ml_matches.iloc[0]

                    low_conf = is_low_confidence(context_row) if context_row is not None else 0

                    game_rain_excluded = rain_excluded(context_row) if context_row is not None else False

                    if game_rain_excluded:
                        summary["rain_excluded"] += 1
                        _log(
                            f"  {game_id} rain excluded "
                            f"(will_it_rain={iv(context_row.get('will_it_rain'))} "
                            f"symbol_code={sv(context_row.get('symbol_code'))})",
                            "WARN",
                        )
                        continue

                    if not rl_matches.empty:
                        for _, rline_row in rl_matches.iterrows():
                            for r in process_run_line(rline_row, global_counters):
                                k = f"{game_id}_{r['market_type']}_{r['bet_side']}_{r['line']}"

                                if k not in seen:
                                    if low_conf:
                                        summary["low_confidence"] += 1

                                    final.append({**base, **r, "low_confidence": low_conf})
                                    seen.add(k)
                                    ps["rl"] += 1

                    if not tt_matches.empty:
                        for _, total_row in tt_matches.iterrows():
                            if sp_sample_excluded_for_total(total_row):
                                summary["sp_sample_excluded"] += 1
                                _log(
                                    f"  {game_id} total SP sample excluded "
                                    f"(home={sv(total_row.get('home_sp_sample_flag'))} "
                                    f"away={sv(total_row.get('away_sp_sample_flag'))})",
                                    "WARN",
                                )
                                continue

                            for r in process_total(total_row, global_counters):
                                k = f"{game_id}_{r['market_type']}_{r['bet_side']}_{r['line']}"

                                if k not in seen:
                                    if low_conf:
                                        summary["low_confidence"] += 1

                                    final.append({**base, **r, "low_confidence": low_conf})
                                    seen.add(k)
                                    ps["tot"] += 1

                    if not ml_matches.empty:
                        for _, ml_row in ml_matches.iterrows():
                            for r in process_moneyline(ml_row, global_counters):
                                k = f"{game_id}_{r['market_type']}_{r['bet_side']}_{r['line']}"

                                if k not in seen:
                                    if low_conf:
                                        summary["low_confidence"] += 1

                                    final.append({**base, **r, "low_confidence": low_conf})
                                    seen.add(k)
                                    ps["ml"] += 1

                ps["bets"] = len(final)

                if final:
                    out = OUTPUT_DIR / f"{slate}_MLB.csv"
                    out_df = pd.DataFrame(final)
                    validation_counts = write_output_csv(out_df, out, f"{slate} selected output")

                    for key, value in validation_counts.items():
                        summary[key] += value

                    summary["slates_written"] += 1
                    summary["total_bets"] += len(final)
                    summary["moneyline_bets"] += ps["ml"]
                    summary["run_line_bets"] += ps["rl"]
                    summary["total_mkt_bets"] += ps["tot"]

                    _log(
                        f"WROTE: {out.name} "
                        f"({len(final)} bets | ml={ps['ml']} rl={ps['rl']} tot={ps['tot']})"
                    )
                else:
                    _log(f"{slate} no bets passed filters", "WARN")
                    ps["status"] = "no_bets"

            except ValueError as e:
                _log(f"{slate} SCHEMA FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                ps["status"] = "schema_error"
                summary["schema_errors"] += 1
                summary["errors"] += 1

            except Exception as e:
                _log(f"{slate} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                ps["status"] = "error"
                summary["errors"] += 1

            per_slate.append(ps)

        summary["counters"] = global_counters

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    _write_summary(summary, per_slate)

    if summary["errors"] > 0 or summary["schema_errors"] > 0:
        print(
            f"baseball select_bets completed with errors. "
            f"errors={summary['errors']} schema_errors={summary['schema_errors']}"
        )
        raise SystemExit(1)

    print(
        f"baseball select_bets complete. "
        f"slates_written={summary['slates_written']} "
        f"total_bets={summary['total_bets']} "
        f"moneyline_bets={summary['moneyline_bets']} "
        f"run_line_bets={summary['run_line_bets']} "
        f"total_mkt_bets={summary['total_mkt_bets']} "
        f"selected_nonpositive_kelly={summary['selected_nonpositive_kelly']} "
        f"selected_probability_source_mismatch={summary['selected_probability_source_mismatch']} "
        f"selected_adjusted_only_positive={summary['selected_adjusted_only_positive']} "
        f"row_count_warnings={summary['row_count_warnings']}"
    )


if __name__ == "__main__":
    main()
