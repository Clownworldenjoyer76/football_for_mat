#!/usr/bin/env python3
"""
UFC reports / grading script.

Repo path: docs/win/mma/ufc/scripts/04_final/ufc_reports.py

Iterates over manual_files/{date}_ufc.csv (results = source of truth),
joins to 03_select/{date}_ufc_select.csv (selected bets), grades each
bet, and writes summary + bucketed breakdown CSVs to 04_final/.

Outputs:
  docs/win/mma/ufc/04_final/ufc_summary_overall.csv
  docs/win/mma/ufc/04_final/reports/ufc_moneyline_by_ev.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_odds.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_implied_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_model_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_dratings_prob.csv
  docs/win/mma/ufc/04_final/reports/ufc_by_date.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from glob import glob
from math import floor

# ---------- Paths ----------
# script lives at docs/win/mma/ufc/scripts/04_final/ufc_reports.py
# go up 6 levels (04_final -> scripts -> ufc -> mma -> win -> docs) to reach repo root
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
    )
)

UFC_BASE = os.path.join(REPO_ROOT, "docs", "win", "mma", "ufc")
SELECT_DIR = os.path.join(UFC_BASE, "03_select")
RESULTS_DIR = os.path.join(UFC_BASE, "manual_files")
OUT_DIR = os.path.join(UFC_BASE, "04_final")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")

LEAGUE = "ufc"
MARKET = "moneyline"

# ---------- Helpers ----------

def american_to_profit(ml: int) -> float:
    """Profit on a 1-unit bet at American odds."""
    if ml >= 0:
        return ml / 100.0
    return 100.0 / -ml


def parse_ml(s: str) -> int:
    s = (s or "").strip().replace(" ", "")
    if not s:
        raise ValueError("empty moneyline")
    return int(s)


def fmt_ml(ml: int) -> str:
    return f"+{ml}" if ml >= 0 else f"{ml}"


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def safe_float(s):
    if s is None:
        return None
    v = str(s).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------- Bucketing ----------

def ev_bucket(ev: float):
    """0.01-wide. e.g. 0.0499 -> (0.04, '0.04 to 0.05')."""
    lo = floor(ev * 100) / 100.0
    hi = lo + 0.01
    return lo, f"{lo:.2f} to {hi:.2f}"


def prob_bucket(p: float):
    """0.10-wide. e.g. 0.4533 -> (0.40, '0.40 to 0.50')."""
    lo = floor(p * 10) / 10.0
    hi = lo + 0.10
    return lo, f"{lo:.2f} to {hi:.2f}"


def odds_bucket(ml: int):
    """50-wide symmetric.
       +112 -> '+100 to +150'
       -125 -> '-150 to -100'
       |ml|<100 (rare) -> '-100 to +100'
    """
    if abs(ml) < 100:
        return 0, "-100 to +100"
    if ml >= 100:
        lo = ((ml - 100) // 50) * 50 + 100
        hi = lo + 50
        return lo, f"+{lo} to +{hi}"
    a = abs(ml)
    lo_abs = ((a - 100) // 50) * 50 + 100
    hi_abs = lo_abs + 50
    # label more-negative on the left: e.g. "-150 to -100"
    return -hi_abs, f"-{hi_abs} to -{lo_abs}"


# ---------- Discovery / IO ----------

DATE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})_ufc\.csv$")


def discover_dates() -> list:
    dates = []
    if not os.path.isdir(RESULTS_DIR):
        return dates
    for path in sorted(glob(os.path.join(RESULTS_DIR, "*_ufc.csv"))):
        m = DATE_RE.search(os.path.basename(path))
        if m:
            dates.append(m.group(1))
    return dates


def read_csv(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- Grading ----------

def grade_result(s: str) -> str:
    """Anything not Win/Loss = Push."""
    v = (s or "").strip().lower()
    if v == "win":
        return "Win"
    if v == "loss":
        return "Loss"
    return "Push"


def build_results_index(rows: list) -> dict:
    """Index by frozenset({fighter_1, fighter_2}) -> row."""
    idx = {}
    for r in rows:
        f1 = normalize_name(r.get("fighter_1", ""))
        f2 = normalize_name(r.get("fighter_2", ""))
        if f1 and f2:
            idx[frozenset((f1, f2))] = r
    return idx


def collect_graded() -> list:
    """Iterate over results files (source of truth), grade each matching select."""
    graded = []
    for date in discover_dates():
        results = read_csv(os.path.join(RESULTS_DIR, f"{date}_ufc.csv"))
        if not results:
            continue
        idx = build_results_index(results)

        select_path = os.path.join(SELECT_DIR, f"{date}_ufc_select.csv")
        selects = read_csv(select_path)
        if not selects:
            continue

        for s in selects:
            fighter = normalize_name(s.get("fighter", ""))
            opp = normalize_name(s.get("opponent", ""))
            if not fighter or not opp:
                continue
            r = idx.get(frozenset((fighter, opp)))
            if not r:
                continue

            r_f1 = normalize_name(r.get("fighter_1", ""))
            outcome_raw = r.get("result_fighter_1", "") if fighter == r_f1 else r.get("result_fighter_2", "")
            outcome = grade_result(outcome_raw)

            try:
                ml = parse_ml(s.get("moneyline", ""))
            except Exception:
                continue

            if outcome == "Win":
                units = american_to_profit(ml)
            elif outcome == "Loss":
                units = -1.0
            else:
                units = 0.0

            graded.append({
                "date": s.get("match_date", date),
                "fighter": s.get("fighter", ""),
                "opponent": s.get("opponent", ""),
                "moneyline": ml,
                "implied_prob": safe_float(s.get("implied_prob")),
                "model_prob": safe_float(s.get("model_prob")),
                "dratings_prob": safe_float(s.get("dratings_prob")),
                "ev": safe_float(s.get("ev")),
                "outcome": outcome,
                "units": units,
            })
    return graded


# ---------- Aggregation ----------

def empty_agg() -> dict:
    return {
        "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
        "units": 0.0,
        "sum_implied": 0.0, "n_implied": 0,
        "sum_model": 0.0, "n_model": 0,
        "sum_dr": 0.0, "n_dr": 0,
        "sum_ml": 0, "n_ml": 0,
    }


def add_to_agg(a: dict, g: dict) -> None:
    a["bets"] += 1
    if g["outcome"] == "Win":
        a["wins"] += 1
    elif g["outcome"] == "Loss":
        a["losses"] += 1
    else:
        a["pushes"] += 1
    a["units"] += g["units"]
    if g["implied_prob"] is not None:
        a["sum_implied"] += g["implied_prob"]; a["n_implied"] += 1
    if g["model_prob"] is not None:
        a["sum_model"] += g["model_prob"]; a["n_model"] += 1
    if g["dratings_prob"] is not None:
        a["sum_dr"] += g["dratings_prob"]; a["n_dr"] += 1
    a["sum_ml"] += g["moneyline"]; a["n_ml"] += 1


def render_row(bucket_dim: str, bucket_label: str, a: dict) -> dict:
    decided = a["wins"] + a["losses"]
    win_pct = (a["wins"] / decided) if decided else ""
    roi = (a["units"] / a["bets"]) if a["bets"] else ""
    return {
        "league": LEAGUE,
        "market_type": MARKET,
        "bucket_dimension": bucket_dim,
        "bucket": bucket_label,
        "bets": a["bets"],
        "wins": a["wins"],
        "losses": a["losses"],
        "pushes": a["pushes"],
        "total": a["bets"],
        "win_pct": f"{win_pct:.4f}" if win_pct != "" else "",
        "units_flat": f"{a['units']:.4f}",
        "roi_flat": f"{roi:.4f}" if roi != "" else "",
        "avg_implied_prob": f"{(a['sum_implied'] / a['n_implied']):.4f}" if a["n_implied"] else "",
        "avg_model_prob": f"{(a['sum_model'] / a['n_model']):.4f}" if a["n_model"] else "",
        "avg_dratings_prob": f"{(a['sum_dr'] / a['n_dr']):.4f}" if a["n_dr"] else "",
        "avg_odds_american": fmt_ml(round(a["sum_ml"] / a["n_ml"])) if a["n_ml"] else "",
    }


REPORT_HEADERS = [
    "league", "market_type", "bucket_dimension", "bucket",
    "bets", "wins", "losses", "pushes", "total",
    "win_pct", "units_flat", "roi_flat",
    "avg_implied_prob", "avg_model_prob", "avg_dratings_prob", "avg_odds_american",
]


def write_report(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_HEADERS)
        w.writeheader()
        w.writerows(rows)


def write_summary(path: str, graded: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = empty_agg()
    for g in graded:
        add_to_agg(a, g)
    decided = a["wins"] + a["losses"]
    win_pct = (a["wins"] / decided) if decided else ""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["league", "market_type", "Win", "Loss", "Push", "Total", "Win_Pct"],
        )
        w.writeheader()
        w.writerow({
            "league": LEAGUE,
            "market_type": MARKET,
            "Win": a["wins"],
            "Loss": a["losses"],
            "Push": a["pushes"],
            "Total": a["bets"],
            "Win_Pct": f"{win_pct:.4f}" if win_pct != "" else "",
        })


def bucketed_rows(graded: list, dim_name: str, key_fn) -> list:
    buckets = defaultdict(empty_agg)
    sort_keys = {}
    for g in graded:
        kv = key_fn(g)
        if kv is None:
            continue
        sort_val, label = kv
        add_to_agg(buckets[label], g)
        sort_keys[label] = sort_val
    return [render_row(dim_name, label, buckets[label])
            for label in sorted(buckets.keys(), key=lambda k: sort_keys[k])]


# ---------- main ----------

def main() -> int:
    graded = collect_graded()
    print(f"Graded {len(graded)} bets across "
          f"{len({g['date'] for g in graded})} dates.")

    write_summary(os.path.join(OUT_DIR, "ufc_summary_overall.csv"), graded)

    write_report(
        os.path.join(REPORTS_DIR, "ufc_moneyline_by_ev.csv"),
        bucketed_rows(graded, "ev",
                      lambda g: ev_bucket(g["ev"]) if g["ev"] is not None else None),
    )
    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_odds.csv"),
        bucketed_rows(graded, "odds",
                      lambda g: odds_bucket(g["moneyline"])),
    )
    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_implied_prob.csv"),
        bucketed_rows(graded, "implied_prob",
                      lambda g: prob_bucket(g["implied_prob"]) if g["implied_prob"] is not None else None),
    )
    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_model_prob.csv"),
        bucketed_rows(graded, "model_prob",
                      lambda g: prob_bucket(g["model_prob"]) if g["model_prob"] is not None else None),
    )
    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_dratings_prob.csv"),
        bucketed_rows(graded, "dratings_prob",
                      lambda g: prob_bucket(g["dratings_prob"]) if g["dratings_prob"] is not None else None),
    )
    write_report(
        os.path.join(REPORTS_DIR, "ufc_by_date.csv"),
        bucketed_rows(graded, "by_date",
                      lambda g: (g["date"], g["date"])),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
