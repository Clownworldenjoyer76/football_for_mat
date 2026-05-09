"""
03_select.py

Filters edge output using rules defined in docs/win/mma/ufc/config/markets.yaml.

Input:
    docs/win/mma/ufc/02_edges/{date}_ufc_edges.csv
    docs/win/mma/ufc/config/markets.yaml

Output:
    docs/win/mma/ufc/03_select/{date}_ufc_select.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

# --- Paths ---
EDGES_DIR = Path("docs/win/mma/ufc/02_edges")
CONFIG_PATH = Path("docs/win/mma/ufc/config/markets.yaml")
OUT_DIR = Path("docs/win/mma/ufc/03_select")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load config ---
with CONFIG_PATH.open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

ml_config = config["ufc"]["moneyline"]
enabled = ml_config.get("enabled", True)
pick_pref = ml_config.get("pick_preference", "best_ev")
odds_bands = ml_config.get("odds_bands", [])
edge_bands = ml_config.get("edge_bands", [])
ev_bands = ml_config.get("ev_bands", [])
kelly_bands = ml_config.get("kelly_bands", [])
model_prob_min = ml_config.get("model_probability_minimum", 0.0)
dratings_prob_min = ml_config.get("dratings_probability_minimum", 0.0)

def in_any_band(value, bands):
    """Returns True if value falls within at least one [min, max] band."""
    if value is None or value == "":
        return False
    try:
        v = float(value)
        return any(lo <= v <= hi for lo, hi in bands)
    except:
        return False

def ml_to_float(ml_str):
    try:
        return float(str(ml_str).replace("+", ""))
    except:
        return None

def safe_float(val):
    try:
        return float(val)
    except:
        return None

def pick_metric(row, fighter_key, pref):
    """Return the metric value used for pick_preference sorting."""
    suffix = "_f1" if fighter_key == "f1" else "_f2"
    mapping = {
        "best_ev": f"ev{suffix}",
        "best_edge": f"edge{suffix}",
        "best_kelly": f"kelly{suffix}",
        "best_model_prob": f"model_prob{suffix}",
        "best_dratings_prob": f"dratings_prob{suffix}",
    }
    col = mapping.get(pref, f"ev{suffix}")
    return safe_float(row.get(col)) or 0.0

def passes_filters(ml, edge, ev, kelly, model_prob, dratings_prob):
    if not enabled:
        return False
    if odds_bands and not in_any_band(ml, odds_bands):
        return False
    if edge_bands and not in_any_band(edge, edge_bands):
        return False
    if ev_bands and not in_any_band(ev, ev_bands):
        return False
    if kelly_bands and not in_any_band(kelly, kelly_bands):
        return False
    if model_prob is not None and model_prob < model_prob_min:
        return False
    if dratings_prob is not None and dratings_prob != "" and safe_float(dratings_prob) is not None:
        if safe_float(dratings_prob) < dratings_prob_min:
            return False
    return True

# --- Process each edges file ---
edges_files = sorted(EDGES_DIR.glob("*_ufc_edges.csv"))
if not edges_files:
    print("No edges files found.")
    raise SystemExit(1)

for edges_file in edges_files:
    date_str = edges_file.stem.replace("_ufc_edges", "")

    with edges_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows in {edges_file.name}, skipping")
        continue

    selected = []
    for row in rows:
        candidates = []

        # Check fighter 1
        ml1 = ml_to_float(row.get("moneyline_f1"))
        edge1 = safe_float(row.get("edge_f1"))
        ev1 = safe_float(row.get("ev_f1"))
        kelly1 = safe_float(row.get("kelly_f1"))
        mp1 = safe_float(row.get("model_prob_f1"))
        dr1 = safe_float(row.get("dratings_prob_f1"))

        if passes_filters(ml1, edge1, ev1, kelly1, mp1, dr1):
            candidates.append({
                "match_date": row["match_date"],
                "fighter": row["fighter_1"],
                "opponent": row["fighter_2"],
                "moneyline": row["moneyline_f1"],
                "implied_prob": row["implied_prob_f1"],
                "model_prob": row["model_prob_f1"],
                "dratings_prob": row["dratings_prob_f1"],
                "edge": row["edge_f1"],
                "ev": row["ev_f1"],
                "kelly": row["kelly_f1"],
                "_sort_val": pick_metric(row, "f1", pick_pref),
            })

        # Check fighter 2
        ml2 = ml_to_float(row.get("moneyline_f2"))
        edge2 = safe_float(row.get("edge_f2"))
        ev2 = safe_float(row.get("ev_f2"))
        kelly2 = safe_float(row.get("kelly_f2"))
        mp2 = safe_float(row.get("model_prob_f2"))
        dr2 = safe_float(row.get("dratings_prob_f2"))

        if passes_filters(ml2, edge2, ev2, kelly2, mp2, dr2):
            candidates.append({
                "match_date": row["match_date"],
                "fighter": row["fighter_2"],
                "opponent": row["fighter_1"],
                "moneyline": row["moneyline_f2"],
                "implied_prob": row["implied_prob_f2"],
                "model_prob": row["model_prob_f2"],
                "dratings_prob": row["dratings_prob_f2"],
                "edge": row["edge_f2"],
                "ev": row["ev_f2"],
                "kelly": row["kelly_f2"],
                "_sort_val": pick_metric(row, "f2", pick_pref),
            })

        # Pick best candidate from this fight per pick_preference
        if candidates:
            best = max(candidates, key=lambda x: x["_sort_val"])
            best.pop("_sort_val")
            selected.append(best)

    # Sort all selected picks by sort metric descending
    selected.sort(key=lambda x: safe_float(x.get("ev")) or 0, reverse=True)

    out_file = OUT_DIR / f"{date_str}_ufc_select.csv"
    if selected:
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
        print(f"WROTE {out_file} ({len(selected)} picks)")
    else:
        print(f"No picks passed filters for {date_str}")
