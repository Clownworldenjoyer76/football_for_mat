#!/usr/bin/env python3
"""
Combine weekly NFL moneyline, ATS/spread, and totals enrichment outputs.

No prediction values are recalculated. No enrichment rules are re-evaluated.
No weights are applied. The exact 118-column combined contract is preserved.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
NFL_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_reporter import PipelineReporter

ENRICHED_ROOT = NFL_ROOT / "00_intake" / "predictions" / "enriched"
MONEYLINE_DIR = ENRICHED_ROOT / "moneyline"
SPREAD_DIR = ENRICHED_ROOT / "spread"
TOTALS_DIR = ENRICHED_ROOT / "totals"
OUTPUT_DIR = ENRICHED_ROOT / "combined"
REPORT_ROOT = NFL_ROOT / "errors"
WEEK_FILE_RE = re.compile(r"^week_(\d+)_NFL_enriched\.csv$")

WEEKLY_COLUMNS = ['season',
 'season_type',
 'week',
 'game_id',
 'odds_provider_game_id',
 'game_date',
 'game_time',
 'commence_time',
 'away_team',
 'home_team',
 'odds_away_team',
 'odds_home_team',
 'neutral_site',
 'stadium',
 'roof',
 'surface',
 'home_timezone',
 'away_timezone',
 'game_timezone',
 'bookmaker',
 'home_moneyline_american',
 'away_moneyline_american',
 'home_spread',
 'away_spread',
 'home_spread_american',
 'away_spread_american',
 'total',
 'over_american',
 'under_american',
 'odds_last_update',
 'odds_available',
 'odds_missing_reason']
SHARED_COLUMNS = ['season',
 'season_type',
 'week',
 'game_id',
 'game_date',
 'game_time',
 'commence_time',
 'away_team',
 'home_team']
MONEYLINE_APPENDED = ['drat_home_prob',
 'drat_away_prob',
 'epred_home_prob_raw',
 'epred_away_prob_raw',
 'epred_home_prob',
 'epred_away_prob',
 'epred_home_rating',
 'epred_away_rating',
 'epred_matchupQuality',
 'market_bookmaker',
 'market_last_update',
 'market_home_moneyline_american',
 'market_away_moneyline_american',
 'market_home_spread',
 'market_away_spread',
 'market_total',
 'market_home_prob_novig',
 'market_away_prob_novig',
 'drat_pick',
 'epred_pick',
 'market_pick',
 'drat_epred_agree',
 'drat_market_agree',
 'epred_market_agree',
 'all_three_agree',
 'epred_rating_gap_home',
 'drat_epred_prob_diff_pp',
 'drat_market_edge_home_pp',
 'epred_market_edge_home_pp',
 'matched_rule_count',
 'matched_positive_rule_count',
 'matched_negative_rule_count',
 'matched_rule_ids',
 'matched_rule_conditions',
 'home_matched_rule_count',
 'home_matched_rule_ids',
 'home_strongest_positive_rule_id',
 'home_strongest_positive_hist_win_rate_pct',
 'home_strongest_positive_lift_pp',
 'home_strongest_positive_games',
 'home_strongest_negative_rule_id',
 'home_strongest_negative_hist_win_rate_pct',
 'home_strongest_negative_lift_pp',
 'home_strongest_negative_games',
 'away_matched_rule_count',
 'away_matched_rule_ids',
 'away_strongest_positive_rule_id',
 'away_strongest_positive_hist_win_rate_pct',
 'away_strongest_positive_lift_pp',
 'away_strongest_positive_games',
 'away_strongest_negative_rule_id',
 'away_strongest_negative_hist_win_rate_pct',
 'away_strongest_negative_lift_pp',
 'away_strongest_negative_games',
 'drat_matched_rule_count',
 'drat_matched_rule_ids',
 'epred_matched_rule_count',
 'epred_matched_rule_ids',
 'market_matched_rule_count',
 'market_matched_rule_ids',
 'drat_epred_consensus_matched_rule_count',
 'drat_epred_consensus_matched_rule_ids',
 'all3_consensus_matched_rule_count',
 'all3_consensus_matched_rule_ids']
SPREAD_APPENDED = ['drat_home_prob',
 'drat_away_prob',
 'epred_home_prob_raw',
 'epred_away_prob_raw',
 'epred_home_prob',
 'epred_away_prob',
 'epred_home_rating',
 'epred_away_rating',
 'epred_matchupQuality',
 'market_bookmaker',
 'market_last_update',
 'market_home_moneyline_american',
 'market_away_moneyline_american',
 'market_home_spread',
 'market_away_spread',
 'market_total',
 'market_home_prob_novig',
 'market_away_prob_novig',
 'drat_pick',
 'epred_pick',
 'market_pick',
 'drat_epred_agree',
 'drat_market_agree',
 'epred_market_agree',
 'all_three_agree',
 'epred_rating_gap_home',
 'drat_epred_prob_diff_pp',
 'drat_market_edge_home_pp',
 'epred_market_edge_home_pp',
 'matched_rule_count',
 'matched_positive_rule_count',
 'matched_negative_rule_count',
 'matched_rule_ids',
 'matched_rule_conditions',
 'home_matched_rule_count',
 'home_matched_rule_ids',
 'home_strongest_positive_rule_id',
 'home_strongest_positive_hist_cover_rate_pct',
 'home_strongest_positive_lift_pp',
 'home_strongest_positive_games',
 'home_strongest_negative_rule_id',
 'home_strongest_negative_hist_cover_rate_pct',
 'home_strongest_negative_lift_pp',
 'home_strongest_negative_games',
 'away_matched_rule_count',
 'away_matched_rule_ids',
 'away_strongest_positive_rule_id',
 'away_strongest_positive_hist_cover_rate_pct',
 'away_strongest_positive_lift_pp',
 'away_strongest_positive_games',
 'away_strongest_negative_rule_id',
 'away_strongest_negative_hist_cover_rate_pct',
 'away_strongest_negative_lift_pp',
 'away_strongest_negative_games',
 'drat_matched_rule_count',
 'drat_matched_rule_ids',
 'epred_matched_rule_count',
 'epred_matched_rule_ids',
 'market_matched_rule_count',
 'market_matched_rule_ids',
 'drat_epred_consensus_matched_rule_count',
 'drat_epred_consensus_matched_rule_ids',
 'all3_consensus_matched_rule_count',
 'all3_consensus_matched_rule_ids']
TOTALS_APPENDED = ['drat_home_prob',
 'drat_away_prob',
 'epred_home_prob_raw',
 'epred_away_prob_raw',
 'epred_home_prob',
 'epred_away_prob',
 'epred_home_rating',
 'epred_away_rating',
 'epred_matchupQuality',
 'market_bookmaker',
 'market_last_update',
 'market_home_moneyline_american',
 'market_away_moneyline_american',
 'market_home_spread',
 'market_away_spread',
 'market_total',
 'market_home_prob_novig',
 'market_away_prob_novig',
 'drat_pick',
 'epred_pick',
 'market_pick',
 'drat_epred_agree',
 'drat_market_agree',
 'epred_market_agree',
 'all_three_agree',
 'epred_rating_gap_home',
 'drat_epred_prob_diff_pp',
 'drat_market_edge_home_pp',
 'epred_market_edge_home_pp',
 'matched_rule_count',
 'matched_positive_rule_count',
 'matched_negative_rule_count',
 'matched_rule_ids',
 'matched_rule_conditions',
 'over_matched_rule_count',
 'over_matched_positive_rule_count',
 'over_matched_negative_rule_count',
 'over_matched_rule_ids',
 'over_strongest_positive_rule_id',
 'over_strongest_positive_hist_hit_rate_pct',
 'over_strongest_positive_lift_pp',
 'over_strongest_positive_games',
 'over_strongest_negative_rule_id',
 'over_strongest_negative_hist_hit_rate_pct',
 'over_strongest_negative_lift_pp',
 'over_strongest_negative_games',
 'under_matched_rule_count',
 'under_matched_positive_rule_count',
 'under_matched_negative_rule_count',
 'under_matched_rule_ids',
 'under_strongest_positive_rule_id',
 'under_strongest_positive_hist_hit_rate_pct',
 'under_strongest_positive_lift_pp',
 'under_strongest_positive_games',
 'under_strongest_negative_rule_id',
 'under_strongest_negative_hist_hit_rate_pct',
 'under_strongest_negative_lift_pp',
 'under_strongest_negative_games',
 'drat_matched_rule_count',
 'drat_matched_rule_ids',
 'epred_matched_rule_count',
 'epred_matched_rule_ids',
 'market_matched_rule_count',
 'market_matched_rule_ids',
 'drat_epred_consensus_matched_rule_count',
 'drat_epred_consensus_matched_rule_ids',
 'all3_consensus_matched_rule_count',
 'all3_consensus_matched_rule_ids']
MONEYLINE_HEADERS = WEEKLY_COLUMNS + MONEYLINE_APPENDED
SPREAD_HEADERS = WEEKLY_COLUMNS + SPREAD_APPENDED
TOTALS_HEADERS = WEEKLY_COLUMNS + TOTALS_APPENDED

MONEYLINE_MAP = {'ml_matched_rule_count': 'matched_rule_count',
 'ml_matched_positive_rule_count': 'matched_positive_rule_count',
 'ml_matched_negative_rule_count': 'matched_negative_rule_count',
 'ml_matched_rule_ids': 'matched_rule_ids',
 'ml_matched_rule_conditions': 'matched_rule_conditions',
 'ml_home_matched_rule_count': 'home_matched_rule_count',
 'ml_home_matched_rule_ids': 'home_matched_rule_ids',
 'ml_home_strongest_positive_rule_id': 'home_strongest_positive_rule_id',
 'ml_home_strongest_positive_hist_win_rate_pct': 'home_strongest_positive_hist_win_rate_pct',
 'ml_home_strongest_positive_lift_pp': 'home_strongest_positive_lift_pp',
 'ml_home_strongest_positive_games': 'home_strongest_positive_games',
 'ml_home_strongest_negative_rule_id': 'home_strongest_negative_rule_id',
 'ml_home_strongest_negative_hist_win_rate_pct': 'home_strongest_negative_hist_win_rate_pct',
 'ml_home_strongest_negative_lift_pp': 'home_strongest_negative_lift_pp',
 'ml_home_strongest_negative_games': 'home_strongest_negative_games',
 'ml_away_matched_rule_count': 'away_matched_rule_count',
 'ml_away_matched_rule_ids': 'away_matched_rule_ids',
 'ml_away_strongest_positive_rule_id': 'away_strongest_positive_rule_id',
 'ml_away_strongest_positive_hist_win_rate_pct': 'away_strongest_positive_hist_win_rate_pct',
 'ml_away_strongest_positive_lift_pp': 'away_strongest_positive_lift_pp',
 'ml_away_strongest_positive_games': 'away_strongest_positive_games',
 'ml_away_strongest_negative_rule_id': 'away_strongest_negative_rule_id',
 'ml_away_strongest_negative_hist_win_rate_pct': 'away_strongest_negative_hist_win_rate_pct',
 'ml_away_strongest_negative_lift_pp': 'away_strongest_negative_lift_pp',
 'ml_away_strongest_negative_games': 'away_strongest_negative_games',
 'ml_drat_matched_rule_count': 'drat_matched_rule_count',
 'ml_drat_matched_rule_ids': 'drat_matched_rule_ids',
 'ml_epred_matched_rule_count': 'epred_matched_rule_count',
 'ml_epred_matched_rule_ids': 'epred_matched_rule_ids',
 'ml_market_matched_rule_count': 'market_matched_rule_count',
 'ml_market_matched_rule_ids': 'market_matched_rule_ids',
 'ml_drat_epred_consensus_matched_rule_count': 'drat_epred_consensus_matched_rule_count',
 'ml_drat_epred_consensus_matched_rule_ids': 'drat_epred_consensus_matched_rule_ids',
 'ml_all3_consensus_matched_rule_count': 'all3_consensus_matched_rule_count',
 'ml_all3_consensus_matched_rule_ids': 'all3_consensus_matched_rule_ids'}
ATS_MAP = {'ats_matched_rule_count': 'matched_rule_count',
 'ats_matched_positive_rule_count': 'matched_positive_rule_count',
 'ats_matched_negative_rule_count': 'matched_negative_rule_count',
 'ats_matched_rule_ids': 'matched_rule_ids',
 'ats_matched_rule_conditions': 'matched_rule_conditions',
 'ats_home_matched_rule_count': 'home_matched_rule_count',
 'ats_home_matched_rule_ids': 'home_matched_rule_ids',
 'ats_home_strongest_positive_rule_id': 'home_strongest_positive_rule_id',
 'ats_home_strongest_positive_hist_cover_rate_pct': 'home_strongest_positive_hist_cover_rate_pct',
 'ats_home_strongest_positive_lift_pp': 'home_strongest_positive_lift_pp',
 'ats_home_strongest_positive_games': 'home_strongest_positive_games',
 'ats_home_strongest_negative_rule_id': 'home_strongest_negative_rule_id',
 'ats_home_strongest_negative_hist_cover_rate_pct': 'home_strongest_negative_hist_cover_rate_pct',
 'ats_home_strongest_negative_lift_pp': 'home_strongest_negative_lift_pp',
 'ats_home_strongest_negative_games': 'home_strongest_negative_games',
 'ats_away_matched_rule_count': 'away_matched_rule_count',
 'ats_away_matched_rule_ids': 'away_matched_rule_ids',
 'ats_away_strongest_positive_rule_id': 'away_strongest_positive_rule_id',
 'ats_away_strongest_positive_hist_cover_rate_pct': 'away_strongest_positive_hist_cover_rate_pct',
 'ats_away_strongest_positive_lift_pp': 'away_strongest_positive_lift_pp',
 'ats_away_strongest_positive_games': 'away_strongest_positive_games',
 'ats_away_strongest_negative_rule_id': 'away_strongest_negative_rule_id',
 'ats_away_strongest_negative_hist_cover_rate_pct': 'away_strongest_negative_hist_cover_rate_pct',
 'ats_away_strongest_negative_lift_pp': 'away_strongest_negative_lift_pp',
 'ats_away_strongest_negative_games': 'away_strongest_negative_games',
 'ats_drat_matched_rule_count': 'drat_matched_rule_count',
 'ats_drat_matched_rule_ids': 'drat_matched_rule_ids',
 'ats_epred_matched_rule_count': 'epred_matched_rule_count',
 'ats_epred_matched_rule_ids': 'epred_matched_rule_ids',
 'ats_market_matched_rule_count': 'market_matched_rule_count',
 'ats_market_matched_rule_ids': 'market_matched_rule_ids',
 'ats_drat_epred_consensus_matched_rule_count': 'drat_epred_consensus_matched_rule_count',
 'ats_drat_epred_consensus_matched_rule_ids': 'drat_epred_consensus_matched_rule_ids',
 'ats_all3_consensus_matched_rule_count': 'all3_consensus_matched_rule_count',
 'ats_all3_consensus_matched_rule_ids': 'all3_consensus_matched_rule_ids'}
TOTALS_MAP = {'totals_matched_rule_count': 'matched_rule_count',
 'totals_matched_positive_rule_count': 'matched_positive_rule_count',
 'totals_matched_negative_rule_count': 'matched_negative_rule_count',
 'totals_matched_rule_ids': 'matched_rule_ids',
 'totals_matched_rule_conditions': 'matched_rule_conditions',
 'totals_over_matched_rule_count': 'over_matched_rule_count',
 'totals_over_matched_positive_rule_count': 'over_matched_positive_rule_count',
 'totals_over_matched_negative_rule_count': 'over_matched_negative_rule_count',
 'totals_over_matched_rule_ids': 'over_matched_rule_ids',
 'totals_over_strongest_positive_rule_id': 'over_strongest_positive_rule_id',
 'totals_over_strongest_positive_hist_hit_rate_pct': 'over_strongest_positive_hist_hit_rate_pct',
 'totals_over_strongest_positive_lift_pp': 'over_strongest_positive_lift_pp',
 'totals_over_strongest_positive_games': 'over_strongest_positive_games',
 'totals_over_strongest_negative_rule_id': 'over_strongest_negative_rule_id',
 'totals_over_strongest_negative_hist_hit_rate_pct': 'over_strongest_negative_hist_hit_rate_pct',
 'totals_over_strongest_negative_lift_pp': 'over_strongest_negative_lift_pp',
 'totals_over_strongest_negative_games': 'over_strongest_negative_games',
 'totals_under_matched_rule_count': 'under_matched_rule_count',
 'totals_under_matched_positive_rule_count': 'under_matched_positive_rule_count',
 'totals_under_matched_negative_rule_count': 'under_matched_negative_rule_count',
 'totals_under_matched_rule_ids': 'under_matched_rule_ids',
 'totals_under_strongest_positive_rule_id': 'under_strongest_positive_rule_id',
 'totals_under_strongest_positive_hist_hit_rate_pct': 'under_strongest_positive_hist_hit_rate_pct',
 'totals_under_strongest_positive_lift_pp': 'under_strongest_positive_lift_pp',
 'totals_under_strongest_positive_games': 'under_strongest_positive_games',
 'totals_under_strongest_negative_rule_id': 'under_strongest_negative_rule_id',
 'totals_under_strongest_negative_hist_hit_rate_pct': 'under_strongest_negative_hist_hit_rate_pct',
 'totals_under_strongest_negative_lift_pp': 'under_strongest_negative_lift_pp',
 'totals_under_strongest_negative_games': 'under_strongest_negative_games',
 'totals_drat_matched_rule_count': 'drat_matched_rule_count',
 'totals_drat_matched_rule_ids': 'drat_matched_rule_ids',
 'totals_epred_matched_rule_count': 'epred_matched_rule_count',
 'totals_epred_matched_rule_ids': 'epred_matched_rule_ids',
 'totals_market_matched_rule_count': 'market_matched_rule_count',
 'totals_market_matched_rule_ids': 'market_matched_rule_ids',
 'totals_drat_epred_consensus_matched_rule_count': 'drat_epred_consensus_matched_rule_count',
 'totals_drat_epred_consensus_matched_rule_ids': 'drat_epred_consensus_matched_rule_ids',
 'totals_all3_consensus_matched_rule_count': 'all3_consensus_matched_rule_count',
 'totals_all3_consensus_matched_rule_ids': 'all3_consensus_matched_rule_ids'}
EXPECTED_OUTPUT_COLUMNS = SHARED_COLUMNS + list(MONEYLINE_MAP) + list(ATS_MAP) + list(TOTALS_MAP)
pd = None

class CombineEnrichmentError(RuntimeError):
    pass

def fail(message: str) -> None:
    raise CombineEnrichmentError(message)

def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine NFL enrichment outputs for one season.")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    if args.season < 2000 or args.season > 2100:
        parser.error("--season must be between 2000 and 2100")
    return args

def import_pandas() -> None:
    global pd
    if pd is not None:
        return
    try:
        import pandas as pandas_module
    except ImportError as exc:
        fail("This script requires pandas. Install the repository requirements before running it.")
    pd = pandas_module

def read_raw_header(path: Path) -> list[str]:
    if not path.is_file():
        fail(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        fail(f"Input file is zero bytes: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except Exception as exc:
        fail(f"Could not read header from {path}: {type(exc).__name__}: {exc}")
    if not header:
        fail(f"CSV has no header: {path}")
    if len(header) != len(set(header)):
        duplicates = sorted({c for c in header if header.count(c) > 1})
        fail(f"CSV contains duplicate columns {duplicates}: {path}")
    return header

def require_exact_header(path: Path, expected: list[str], *, label: str) -> None:
    actual = read_raw_header(path)
    if actual != expected:
        fail(
            f"{label} schema mismatch in {path}; "
            f"expected_columns={len(expected)} actual_columns={len(actual)}"
        )

def read_frame(path: Path, expected_headers: list[str], *, label: str):
    require_exact_header(path, expected_headers, label=label)
    try:
        frame = pd.read_csv(
            path, encoding="utf-8-sig", dtype=str,
            keep_default_na=False, na_filter=False, low_memory=False,
        )
    except Exception as exc:
        fail(f"Could not read {label} {path}: {type(exc).__name__}: {exc}")
    if list(frame.columns) != expected_headers:
        fail(f"{label} columns changed while reading {path}")
    if frame.empty:
        fail(f"{label} contains no data rows: {path}")
    return frame

def parse_integer_text(value: Any, *, label: str) -> int:
    text = clean(value)
    if not text:
        fail(f"{label} is blank")
    try:
        number = float(text)
    except ValueError:
        fail(f"{label} must be an integer; received={text!r}")
    if not number.is_integer():
        fail(f"{label} must be an integer; received={text!r}")
    return int(number)

def managed_files(folder: Path, *, label: str) -> dict[int, Path]:
    if not folder.is_dir():
        fail(f"{label} enrichment folder not found: {folder}")
    found: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        match = WEEK_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        week = int(match.group(1))
        if week < 1 or week > 25:
            fail(f"{label} filename week out of range: {path.name}")
        if week in found:
            fail(f"{label} contains duplicate managed week={week}")
        found[week] = path
    if not found:
        fail(f"No managed weekly {label} enrichment files found in {folder}")
    return found

def validate_identity(frame, *, path: Path, season: int, week: int, label: str) -> str:
    ids = frame["game_id"].map(clean)
    if ids.eq("").any():
        fail(f"{label} contains blank game_id: {path}")
    if ids.duplicated(keep=False).any():
        duplicates = sorted(set(ids[ids.duplicated(keep=False)].tolist()))
        fail(f"{label} duplicate game_id values in {path}: {duplicates}")
    seasons = {parse_integer_text(v, label=f"{path.name} season") for v in frame["season"]}
    weeks = {parse_integer_text(v, label=f"{path.name} week") for v in frame["week"]}
    season_types = {clean(v) for v in frame["season_type"]}
    if seasons != {season}:
        fail(f"{label} {path.name} season mismatch expected={season} actual={sorted(seasons)}")
    if weeks != {week}:
        fail(f"{label} {path.name} week mismatch expected={week} actual={sorted(weeks)}")
    if len(season_types) != 1:
        fail(f"{label} {path.name} must contain exactly one season_type")
    season_type = next(iter(season_types))
    if season_type not in {"pre","reg","post"}:
        fail(f"{label} {path.name} invalid season_type={season_type!r}")
    for column in SHARED_COLUMNS:
        if column == "commence_time":
            continue
        if frame[column].map(clean).eq("").any():
            fail(f"{label} {path.name} contains blank {column}")

    commence_time = frame["commence_time"].map(clean)

    if "odds_available" in frame.columns:
        odds_available = frame["odds_available"].map(clean)
        invalid_odds_available = ~odds_available.isin({"0", "1"})

        if invalid_odds_available.any():
            values = sorted(
                set(
                    odds_available[
                        invalid_odds_available
                    ].tolist()
                )
            )
            fail(
                f"{label} {path.name} has invalid "
                f"odds_available values={values}"
            )

        missing_required_commence = (
            commence_time.eq("")
            & odds_available.eq("1")
        )

        if missing_required_commence.any():
            game_ids = sorted(
                set(
                    frame.loc[
                        missing_required_commence,
                        "game_id",
                    ].map(clean).tolist()
                )
            )
            fail(
                f"{label} {path.name} contains blank "
                "commence_time for odds_available=1 "
                f"game_id values={game_ids}"
            )

    return season_type

def normalized(series):
    return series.astype("string").fillna("").map(clean)

def validate_source_triplet(moneyline, spread, totals, *, filename: str) -> None:
    ml_ids = set(normalized(moneyline["game_id"]))
    ats_ids = set(normalized(spread["game_id"]))
    totals_ids = set(normalized(totals["game_id"]))
    if not (ml_ids == ats_ids == totals_ids):
        fail(f"{filename} game_id universes do not match across moneyline/spread/totals")
    ml_index = moneyline.set_index("game_id", drop=False)
    ats_index = spread.set_index("game_id", drop=False)
    totals_index = totals.set_index("game_id", drop=False)
    for game_id in normalized(moneyline["game_id"]).tolist():
        for column in WEEKLY_COLUMNS:
            values = [
                clean(ml_index.at[game_id, column]),
                clean(ats_index.at[game_id, column]),
                clean(totals_index.at[game_id, column]),
            ]
            if not (values[0] == values[1] == values[2]):
                fail(
                    f"{filename} shared weekly field mismatch game_id={game_id} "
                    f"field={column} moneyline={values[0]!r} spread={values[1]!r} totals={values[2]!r}"
                )

def mapped_frame(frame, mapping: dict[str, str]):
    source_columns = ["game_id"] + list(mapping.values())
    out = frame[source_columns].copy()
    return out.rename(columns={source: combined for combined, source in mapping.items()})

def combine_week(moneyline, spread, totals):
    combined = (
        moneyline[SHARED_COLUMNS].copy()
        .merge(mapped_frame(moneyline, MONEYLINE_MAP), on="game_id", how="left", validate="one_to_one")
        .merge(mapped_frame(spread, ATS_MAP), on="game_id", how="left", validate="one_to_one")
        .merge(mapped_frame(totals, TOTALS_MAP), on="game_id", how="left", validate="one_to_one")
    )
    if list(combined.columns) != EXPECTED_OUTPUT_COLUMNS:
        fail("Combined output columns do not match the required 118-column schema")
    if len(combined) != len(moneyline):
        fail(f"Combined row count changed unexpectedly: moneyline={len(moneyline)} combined={len(combined)}")
    return combined

def validate_combined(combined, *, moneyline, spread, totals, filename: str, season: int, week: int) -> None:
    if list(combined.columns) != EXPECTED_OUTPUT_COLUMNS:
        fail(f"{filename} combined schema is not exact")
    if len(combined) != len(moneyline):
        fail(f"{filename} combined row count mismatch")
    season_type = validate_identity(
        combined, path=Path(filename), season=season, week=week, label="combined output"
    )
    source_types = {
        clean(moneyline["season_type"].iloc[0]),
        clean(spread["season_type"].iloc[0]),
        clean(totals["season_type"].iloc[0]),
    }
    if source_types != {season_type}:
        fail(f"{filename} season_type mismatch across sources/output")
    ml_ids = normalized(moneyline["game_id"]).tolist()
    if normalized(combined["game_id"]).tolist() != ml_ids:
        fail(f"{filename} combined row order does not preserve moneyline order")
    ml_index = moneyline.set_index("game_id", drop=False)
    ats_index = spread.set_index("game_id", drop=False)
    totals_index = totals.set_index("game_id", drop=False)
    out_index = combined.set_index("game_id", drop=False)
    for game_id in ml_ids:
        for column in SHARED_COLUMNS:
            if clean(out_index.at[game_id,column]) != clean(ml_index.at[game_id,column]):
                fail(f"{filename} combined shared value mismatch game_id={game_id} field={column}")
        for out_col, src_col in MONEYLINE_MAP.items():
            if clean(out_index.at[game_id,out_col]) != clean(ml_index.at[game_id,src_col]):
                fail(f"{filename} moneyline mapping mismatch game_id={game_id} field={out_col}")
        for out_col, src_col in ATS_MAP.items():
            if clean(out_index.at[game_id,out_col]) != clean(ats_index.at[game_id,src_col]):
                fail(f"{filename} spread mapping mismatch game_id={game_id} field={out_col}")
        for out_col, src_col in TOTALS_MAP.items():
            if clean(out_index.at[game_id,out_col]) != clean(totals_index.at[game_id,src_col]):
                fail(f"{filename} totals mapping mismatch game_id={game_id} field={out_col}")

def normalized_records(frame) -> list[dict[str,str]]:
    return [
        {column: clean(record.get(column)) for column in frame.columns}
        for record in frame.to_dict(orient="records")
    ]

def write_frame(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")

def build_staged_root(week_outputs: dict[int, tuple[Any,Any,Any,Any]]) -> Path:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".combined_enrichment_stage_", dir=str(OUTPUT_DIR.parent)))
    try:
        if OUTPUT_DIR.exists():
            shutil.copytree(OUTPUT_DIR, stage_root, dirs_exist_ok=True)
        for stale in list(stage_root.iterdir()):
            if stale.is_file() and WEEK_FILE_RE.fullmatch(stale.name):
                stale.unlink()
        expected_names: set[str] = set()
        for week, (combined, moneyline, spread, totals) in sorted(week_outputs.items()):
            filename = f"week_{week}_NFL_enriched.csv"
            expected_names.add(filename)
            path = stage_root / filename
            write_frame(path, combined)
            require_exact_header(path, EXPECTED_OUTPUT_COLUMNS, label="staged combined enrichment")
            staged = pd.read_csv(
                path, encoding="utf-8-sig", dtype=str,
                keep_default_na=False, na_filter=False, low_memory=False,
            )
            validate_combined(
                staged, moneyline=moneyline, spread=spread, totals=totals,
                filename=filename,
                season=parse_integer_text(combined["season"].iloc[0], label=f"{filename} season"),
                week=week,
            )
            if normalized_records(staged) != normalized_records(combined):
                fail(f"Staged combined enrichment differs from validated in-memory output: {path}")
        actual_names = {
            path.name for path in stage_root.iterdir()
            if path.is_file() and WEEK_FILE_RE.fullmatch(path.name)
        }
        if actual_names != expected_names:
            fail(
                "Staged combined managed file set mismatch "
                f"expected={sorted(expected_names)} actual={sorted(actual_names)}"
            )
        return stage_root
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

def publish_staged_root(stage_root: Path, *, reporter: PipelineReporter) -> None:
    backup_root = OUTPUT_DIR.parent / f".{OUTPUT_DIR.name}_backup_{uuid.uuid4().hex}"
    try:
        if OUTPUT_DIR.exists():
            os.replace(OUTPUT_DIR, backup_root)
        os.replace(stage_root, OUTPUT_DIR)
    except Exception:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        if backup_root.exists():
            os.replace(backup_root, OUTPUT_DIR)
        raise
    if backup_root.exists():
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            reporter.warning(
                "Combined enrichment published but temporary backup cleanup failed",
                backup_path=str(backup_root), error_type=type(exc).__name__, error=str(exc),
            )

def run(reporter: PipelineReporter, *, season: int) -> None:
    import_pandas()
    source_sets = {
        "moneyline": managed_files(MONEYLINE_DIR, label="moneyline"),
        "spread": managed_files(SPREAD_DIR, label="spread"),
        "totals": managed_files(TOTALS_DIR, label="totals"),
    }
    week_sets = [set(files) for files in source_sets.values()]
    if not (week_sets[0] == week_sets[1] == week_sets[2]):
        fail("Managed weekly enrichment filenames do not match across moneyline/spread/totals")
    weeks = sorted(week_sets[0])
    week_outputs: dict[int, tuple[Any,Any,Any,Any]] = {}
    rows_by_source = {"moneyline":0,"spread":0,"totals":0}
    combined_rows = 0
    for week in weeks:
        filename = f"week_{week}_NFL_enriched.csv"
        ml_path = source_sets["moneyline"][week]
        ats_path = source_sets["spread"][week]
        tot_path = source_sets["totals"][week]
        moneyline = read_frame(ml_path, MONEYLINE_HEADERS, label="moneyline enrichment")
        spread = read_frame(ats_path, SPREAD_HEADERS, label="spread enrichment")
        totals = read_frame(tot_path, TOTALS_HEADERS, label="totals enrichment")
        reporter.add_input(ml_path)
        reporter.add_input(ats_path)
        reporter.add_input(tot_path)
        ml_type = validate_identity(moneyline,path=ml_path,season=season,week=week,label="moneyline enrichment")
        ats_type = validate_identity(spread,path=ats_path,season=season,week=week,label="spread enrichment")
        tot_type = validate_identity(totals,path=tot_path,season=season,week=week,label="totals enrichment")
        if not (ml_type == ats_type == tot_type):
            fail(f"{filename} season_type mismatch across moneyline/spread/totals")
        validate_source_triplet(moneyline,spread,totals,filename=filename)
        combined = combine_week(moneyline,spread,totals)
        validate_combined(
            combined,moneyline=moneyline,spread=spread,totals=totals,
            filename=filename,season=season,week=week,
        )
        week_outputs[week] = (combined,moneyline,spread,totals)
        rows_by_source["moneyline"] += len(moneyline)
        rows_by_source["spread"] += len(spread)
        rows_by_source["totals"] += len(totals)
        combined_rows += len(combined)

    rows_in = sum(rows_by_source.values())
    reporter.set_rows(rows_in=rows_in, rows_out=0)
    reporter.update_details({
        "season":season,"weeks_combined":len(weeks),"games_combined":combined_rows,
        "source_rows_moneyline":rows_by_source["moneyline"],
        "source_rows_spread":rows_by_source["spread"],
        "source_rows_totals":rows_by_source["totals"],
        "source_schema_columns":{"moneyline":96,"spread":96,"totals":100},
        "output_columns":118,
        "publication_mode":"validated_directory_swap_with_rollback",
        "publication_completed":False,"staged_roundtrip_verified":False,
    })

    stage_root: Path | None = None
    try:
        stage_root = build_staged_root(week_outputs)
        reporter.set_detail("staged_roundtrip_verified", True)
        publish_staged_root(stage_root, reporter=reporter)
        stage_root = None
    finally:
        if stage_root is not None and stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)

    published_names = {
        path.name for path in OUTPUT_DIR.iterdir()
        if path.is_file() and WEEK_FILE_RE.fullmatch(path.name)
    }
    expected_names = {f"week_{week}_NFL_enriched.csv" for week in weeks}
    if published_names != expected_names:
        fail(
            "Published combined managed file set mismatch "
            f"expected={sorted(expected_names)} actual={sorted(published_names)}"
        )
    for week in weeks:
        reporter.add_output(OUTPUT_DIR / f"week_{week}_NFL_enriched.csv")
    reporter.set_rows(rows_in=rows_in, rows_out=combined_rows)
    reporter.update_details({
        "files_published":len(weeks),"rows_published":combined_rows,"publication_completed":True,
    })
    print(
        f"COMPLETE | combined weekly files={len(weeks)} | games={combined_rows} | "
        f"columns={len(EXPECTED_OUTPUT_COLUMNS)} | output={OUTPUT_DIR}",
        flush=True,
    )

def main() -> int:
    args = parse_args()
    try:
        with PipelineReporter(
            script=SCRIPT_PATH, stage="00_intake", report_root=REPORT_ROOT,
            pipeline="NFL", league="NFL", season=args.season,
            extra_context={
                "component":"combined prediction enrichment",
                "refresh_scope":"requested season available weeks",
            },
        ) as reporter:
            run(reporter, season=args.season)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
