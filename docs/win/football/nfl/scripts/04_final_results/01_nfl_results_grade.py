#!/usr/bin/env python3
"""
Grade NFL bets from docs/win/football/nfl/03_picks/selected against final results.

READS
-----
Selected bets:
  docs/win/football/nfl/03_picks/selected/week_*_NFL_select_picks.csv

Final scores:
  docs/win/football/nfl/04_final_results/results/{season}_{season_type}_{week}.csv

WRITES
------
Compatibility weekly graded files:
  docs/win/football/nfl/04_final_results/results/graded/{season}_{season_type}_{week}_graded.csv

Cumulative graded master:
  docs/win/football/nfl/04_final_results/results/graded/NFL_final.csv

Unmatched / pending records:
  docs/win/football/nfl/04_final_results/results/unmatched/*.csv

Audits:
  docs/win/football/nfl/04_final_results/results/audit/*.csv

Behavior
--------
- The grading source is ONLY 03_picks/selected.
- One output row is emitted per selected market.
- Final games grade to Win/Loss/Push and compatibility WIN/LOSS/PUSH.
- Non-final or missing-result bets are retained in unmatched/pending outputs.
- Exact duplicate bets are collapsed; conflicting duplicates fail loudly.
- Final-score duplicate game_ids are allowed only when the duplicate rows are identical.
- Optional selection metrics are preserved when present. If implied probability,
  edge, EV, or full Kelly are absent but enough inputs exist, deterministic derived
  values are calculated and the corresponding *_source field records that fact.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

SELECTED_DIR = NFL_ROOT / "03_picks" / "selected"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
GRADED_DIR = RESULTS_DIR / "graded"
UNMATCHED_DIR = RESULTS_DIR / "unmatched"
AUDIT_DIR = RESULTS_DIR / "audit"
ERROR_DIR = NFL_ROOT / "errors" / "04_final_results"

SELECTED_PATTERN = "week_*_NFL_select_picks.csv"
MASTER_FILE = GRADED_DIR / "NFL_final.csv"
UNMATCHED_FILE = UNMATCHED_DIR / "NFL_unmatched_selected_bets.csv"
NOT_FINAL_FILE = UNMATCHED_DIR / "NFL_not_final_selected_bets.csv"
RECON_FILE = AUDIT_DIR / "selected_vs_graded_reconciliation.csv"
DUPLICATE_FILE = AUDIT_DIR / "grading_duplicate_audit.csv"
VALIDATION_FILE = AUDIT_DIR / "graded_output_validation_audit.csv"
RESULT_COUNTS_FILE = AUDIT_DIR / "grading_result_counts.csv"
SPOT_CHECK_FILE = AUDIT_DIR / "grading_spot_check.csv"
SUMMARY_LOG = ERROR_DIR / "nfl_results_grade_summary.txt"
ERROR_LOG = ERROR_DIR / "nfl_results_grade_errors.txt"

SELECTED_REQUIRED = [
    "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team",
    "ml_selected", "ml_selection", "ml_odds_american",
    "spread_selected", "spread_selection", "spread_line", "spread_odds_american",
    "total_selected", "total_selection", "total_line", "total_odds_american",
]

RESULT_REQUIRED = [
    "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team", "away_score", "home_score", "status",
]

OUTPUT_COLUMNS = [
    # Existing compatibility fields first.
    "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team", "away_score", "home_score", "status",
    "market", "selection", "line", "odds_american", "result",
    # Reporting fields.
    "game_time", "commence_time", "edt_time",
    "market_type", "bet_side", "model_prob", "implied_prob", "edge", "ev",
    "full_kelly", "kelly", "selection_reason",
    "implied_prob_source", "edge_source", "ev_source", "full_kelly_source", "kelly_source",
    "final_total", "bet_result", "bet_units",
    "selected_source_file", "grading_generated_at_utc",
]

UNMATCHED_COLUMNS = [
    "unmatched_reason", "season", "season_type", "week", "game_id", "game_date",
    "away_team", "home_team", "market_type", "bet_side", "line", "odds_american",
    "model_prob", "selected_source_file", "result_file", "status",
]

DUPLICATE_COLUMNS = [
    "duplicate_scope", "season", "season_type", "week", "game_id",
    "market_type", "bet_side", "line", "duplicate_count", "identical_duplicate",
    "action_taken", "source_files",
]

RECON_COLUMNS = [
    "season", "season_type", "week", "selected_game_rows", "selected_bets",
    "graded_bets", "unmatched_bets", "not_final_bets", "missing_result_file_bets",
    "missing_game_id_bets", "missing_game_in_results_bets", "status",
]

BET_KEY = ["season", "season_type", "week", "game_id", "market_type", "bet_side", "line"]
EPSILON = 1e-9


MARKET_SPECS = {
    "moneyline": {
        "prefix": "ml",
        "market": "MONEYLINE",
        "selected": "ml_selected",
        "selection": "ml_selection",
        "line": None,
        "odds": "ml_odds_american",
        "valid_sides": {"HOME", "AWAY"},
    },
    "spread": {
        "prefix": "spread",
        "market": "SPREAD",
        "selected": "spread_selected",
        "selection": "spread_selection",
        "line": "spread_line",
        "odds": "spread_odds_american",
        "valid_sides": {"HOME", "AWAY"},
    },
    "total": {
        "prefix": "total",
        "market": "TOTAL",
        "selected": "total_selected",
        "selection": "total_selection",
        "line": "total_line",
        "odds": "total_odds_american",
        "valid_sides": {"OVER", "UNDER"},
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "nat", "<na>"}:
        return ""
    if text.endswith(".0"):
        # Only remove a trailing .0 for integer-like IDs, not general numeric fields.
        try:
            if float(text).is_integer() and all(ch.isdigit() or ch in ".-+" for ch in text):
                return str(int(float(text)))
        except Exception:
            pass
    return text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "nat", "<na>"} else text


def reset_logs() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG.write_text("", encoding="utf-8")
    ERROR_LOG.write_text("", encoding="utf-8")


def log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_utc()}] {message}\n")


def fail(message: str) -> None:
    log(ERROR_LOG, message)
    raise RuntimeError(message)


def duplicate_header_columns(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in header:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        fail(f"{label}: missing required columns: {missing}")


def read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing file: {path}")
    duplicates = duplicate_header_columns(path)
    if duplicates:
        fail(f"{label}: duplicate header columns: {duplicates}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    require_columns(frame, required, label)
    return frame


def to_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def number_text(value: float | None, precision: int = 12) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{precision}g}"


def selection_flag(value: Any) -> bool:
    number = to_float(value)
    return number == 1.0


def american_implied_probability(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_odds(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def derive_edge(model_prob: float | None, implied_prob: float | None) -> float | None:
    if model_prob is None or implied_prob is None:
        return None
    return model_prob - implied_prob


def derive_ev(model_prob: float | None, odds: float | None) -> float | None:
    dec = decimal_odds(odds)
    if model_prob is None or dec is None:
        return None
    return model_prob * (dec - 1.0) - (1.0 - model_prob)


def derive_full_kelly(model_prob: float | None, odds: float | None) -> float | None:
    dec = decimal_odds(odds)
    if model_prob is None or dec is None:
        return None
    b = dec - 1.0
    if b <= 0:
        return None
    q = 1.0 - model_prob
    return (b * model_prob - q) / b


def metric_value(row: pd.Series, prefix: str, metric: str) -> str:
    return clean_text(row.get(f"{prefix}_{metric}", ""))


def build_metrics(row: pd.Series, prefix: str, odds_text: str) -> dict[str, str]:
    model_text = metric_value(row, prefix, "model_probability")
    model_prob = to_float(model_text)
    odds = to_float(odds_text)

    implied_text = metric_value(row, prefix, "implied_probability")
    implied = to_float(implied_text)
    implied_source = "selected" if implied is not None else ""
    if implied is None:
        implied = american_implied_probability(odds)
        if implied is not None:
            implied_source = "derived_from_odds"

    edge_text = metric_value(row, prefix, "edge")
    edge = to_float(edge_text)
    edge_source = "selected" if edge is not None else ""
    if edge is None:
        edge = derive_edge(model_prob, implied)
        if edge is not None:
            edge_source = "derived_from_model_prob_and_odds"

    ev_text = metric_value(row, prefix, "ev")
    ev = to_float(ev_text)
    ev_source = "selected" if ev is not None else ""
    if ev is None:
        ev = derive_ev(model_prob, odds)
        if ev is not None:
            ev_source = "derived_from_model_prob_and_odds"

    full_kelly_text = metric_value(row, prefix, "full_kelly")
    full_kelly = to_float(full_kelly_text)
    full_kelly_source = "selected" if full_kelly is not None else ""
    if full_kelly is None:
        full_kelly = derive_full_kelly(model_prob, odds)
        if full_kelly is not None:
            full_kelly_source = "derived_from_model_prob_and_odds"

    kelly_text = metric_value(row, prefix, "kelly")
    kelly = to_float(kelly_text)
    kelly_source = "selected" if kelly is not None else ""

    return {
        "model_prob": model_text,
        "implied_prob": implied_text if implied_source == "selected" else number_text(implied),
        "edge": edge_text if edge_source == "selected" else number_text(edge),
        "ev": ev_text if ev_source == "selected" else number_text(ev),
        "full_kelly": full_kelly_text if full_kelly_source == "selected" else number_text(full_kelly),
        "kelly": kelly_text if kelly_source == "selected" else "",
        "implied_prob_source": implied_source,
        "edge_source": edge_source,
        "ev_source": ev_source,
        "full_kelly_source": full_kelly_source,
        "kelly_source": kelly_source,
        "selection_reason": metric_value(row, prefix, "selection_reason"),
    }


def explode_selected_file(path: Path) -> tuple[list[dict[str, str]], int]:
    frame = read_csv(path, SELECTED_REQUIRED, f"selected {path}")
    bets: list[dict[str, str]] = []

    for row_index, row in frame.iterrows():
        base = {
            "season": clean(row.get("season")),
            "season_type": clean_text(row.get("season_type")).lower(),
            "week": clean(row.get("week")),
            "game_id": clean(row.get("game_id")),
            "game_date": clean_text(row.get("game_date")),
            "game_time": clean_text(row.get("game_time")),
            "commence_time": clean_text(row.get("commence_time")),
            "edt_time": clean_text(row.get("edt_time")),
            "away_team": clean_text(row.get("away_team")),
            "home_team": clean_text(row.get("home_team")),
            "selected_source_file": path.name,
            "selected_row_number": str(row_index + 2),
        }

        if not all(base[key] for key in ["season", "season_type", "week", "game_id"]):
            fail(f"{path}: row {row_index + 2} missing season/season_type/week/game_id")

        for market_type, spec in MARKET_SPECS.items():
            if not selection_flag(row.get(spec["selected"], "")):
                continue

            selection = clean_text(row.get(spec["selection"], "")).upper()
            if selection not in spec["valid_sides"]:
                fail(
                    f"{path}: row {row_index + 2} game_id={base['game_id']} "
                    f"invalid {market_type} selection={selection!r}"
                )

            line_text = ""
            if spec["line"] is not None:
                line_text = clean_text(row.get(spec["line"], ""))
                if to_float(line_text) is None:
                    fail(
                        f"{path}: row {row_index + 2} game_id={base['game_id']} "
                        f"selected {market_type} missing/invalid line={line_text!r}"
                    )

            odds_text = clean_text(row.get(spec["odds"], ""))
            if to_float(odds_text) is None:
                fail(
                    f"{path}: row {row_index + 2} game_id={base['game_id']} "
                    f"selected {market_type} missing/invalid odds={odds_text!r}"
                )

            metrics = build_metrics(row, spec["prefix"], odds_text)
            bets.append({
                **base,
                "market": spec["market"],
                "selection": selection,
                "market_type": market_type,
                "bet_side": selection.lower(),
                "line": line_text,
                "odds_american": odds_text,
                **metrics,
            })

    return bets, len(frame)


def duplicate_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(clean_text(row.get(column, "")) for column in BET_KEY)


def collapse_selected_duplicates(
    bets: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for bet in bets:
        groups.setdefault(duplicate_key(bet), []).append(bet)

    output: list[dict[str, str]] = []
    for key, rows in groups.items():
        if len(rows) == 1:
            output.append(rows[0])
            continue

        compare_fields = sorted(set(rows[0]) - {"selected_row_number", "selected_source_file"})
        signatures = {
            tuple(clean_text(row.get(field, "")) for field in compare_fields)
            for row in rows
        }
        identical = len(signatures) == 1
        audit_rows.append({
            "duplicate_scope": "selected_bet",
            "season": key[0], "season_type": key[1], "week": key[2], "game_id": key[3],
            "market_type": key[4], "bet_side": key[5], "line": key[6],
            "duplicate_count": str(len(rows)),
            "identical_duplicate": str(identical),
            "action_taken": "collapsed_identical" if identical else "failed_conflict",
            "source_files": ";".join(sorted({row["selected_source_file"] for row in rows})),
        })
        if not identical:
            fail(f"Conflicting duplicate selected bet key={key}")
        output.append(rows[0])

    return output


def load_result_group(
    season: str,
    season_type: str,
    week: str,
    cache: dict[tuple[str, str, str], dict[str, dict[str, str]] | None],
    duplicate_audit: list[dict[str, str]],
) -> dict[str, dict[str, str]] | None:
    key = (season, season_type, week)
    if key in cache:
        return cache[key]

    path = RESULTS_DIR / f"{season}_{season_type}_{week}.csv"
    if not path.exists():
        cache[key] = None
        return None

    frame = read_csv(path, RESULT_REQUIRED, f"results {path}")
    groups: dict[str, list[dict[str, str]]] = {}
    for _, row in frame.iterrows():
        record = {column: clean_text(row.get(column, "")) for column in frame.columns}
        game_id = clean(record.get("game_id"))
        if not game_id:
            fail(f"{path}: final-results row missing game_id")
        record["game_id"] = game_id
        groups.setdefault(game_id, []).append(record)

    indexed: dict[str, dict[str, str]] = {}
    for game_id, rows in groups.items():
        if len(rows) == 1:
            indexed[game_id] = rows[0]
            continue
        signatures = {tuple(sorted(row.items())) for row in rows}
        identical = len(signatures) == 1
        duplicate_audit.append({
            "duplicate_scope": "final_result_game",
            "season": season, "season_type": season_type, "week": week, "game_id": game_id,
            "market_type": "", "bet_side": "", "line": "",
            "duplicate_count": str(len(rows)),
            "identical_duplicate": str(identical),
            "action_taken": "collapsed_identical" if identical else "failed_conflict",
            "source_files": path.name,
        })
        if not identical:
            fail(f"{path}: conflicting duplicate game_id={game_id}")
        indexed[game_id] = rows[0]

    cache[key] = indexed
    return indexed


def normalize_status(value: Any) -> str:
    return clean_text(value).casefold()


def is_final(value: Any) -> bool:
    status = normalize_status(value)
    return status.startswith("final") or status in {"completed", "complete", "game over"}


def compare(left: float, right: float) -> int:
    difference = left - right
    if abs(difference) <= EPSILON:
        return 0
    return 1 if difference > 0 else -1


def grade_bet(bet: dict[str, str], result: dict[str, str]) -> tuple[str, str]:
    away_score = to_float(result.get("away_score"))
    home_score = to_float(result.get("home_score"))
    if away_score is None or home_score is None:
        fail(f"game_id={bet['game_id']}: final game has missing/invalid scores")

    market_type = bet["market_type"]
    side = bet["bet_side"]

    if market_type == "moneyline":
        outcome = compare(home_score, away_score) if side == "home" else compare(away_score, home_score)
    elif market_type == "spread":
        line = to_float(bet["line"])
        if line is None:
            fail(f"game_id={bet['game_id']}: invalid spread line={bet['line']!r}")
        outcome = (
            compare(home_score + line, away_score)
            if side == "home"
            else compare(away_score + line, home_score)
        )
    elif market_type == "total":
        line = to_float(bet["line"])
        if line is None:
            fail(f"game_id={bet['game_id']}: invalid total line={bet['line']!r}")
        total = home_score + away_score
        raw = compare(total, line)
        outcome = raw if side == "over" else -raw
    else:
        fail(f"Unsupported market_type={market_type!r}")
        raise AssertionError

    if outcome > 0:
        return "Win", "WIN"
    if outcome < 0:
        return "Loss", "LOSS"
    return "Push", "PUSH"


def units_won(odds_text: str, bet_result: str) -> float:
    if bet_result == "Push":
        return 0.0
    if bet_result == "Loss":
        return -1.0
    odds = to_float(odds_text)
    if odds is None or odds == 0:
        fail(f"Cannot calculate units from odds={odds_text!r}")
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def build_graded_row(bet: dict[str, str], result: dict[str, str], generated_at: str) -> dict[str, str]:
    bet_result, compat_result = grade_bet(bet, result)
    away_score = clean_text(result.get("away_score"))
    home_score = clean_text(result.get("home_score"))
    total = (to_float(away_score) or 0.0) + (to_float(home_score) or 0.0)

    row = {
        "season": bet["season"],
        "season_type": bet["season_type"],
        "week": bet["week"],
        "game_id": bet["game_id"],
        "game_date": clean_text(result.get("game_date")) or bet["game_date"],
        "away_team": clean_text(result.get("away_team")) or bet["away_team"],
        "home_team": clean_text(result.get("home_team")) or bet["home_team"],
        "away_score": away_score,
        "home_score": home_score,
        "status": clean_text(result.get("status")),
        "market": bet["market"],
        "selection": bet["selection"],
        "line": bet["line"],
        "odds_american": bet["odds_american"],
        "result": compat_result,
        "game_time": bet["game_time"],
        "commence_time": bet["commence_time"],
        "edt_time": bet["edt_time"],
        "market_type": bet["market_type"],
        "bet_side": bet["bet_side"],
        "model_prob": bet["model_prob"],
        "implied_prob": bet["implied_prob"],
        "edge": bet["edge"],
        "ev": bet["ev"],
        "full_kelly": bet["full_kelly"],
        "kelly": bet["kelly"],
        "selection_reason": bet["selection_reason"],
        "implied_prob_source": bet["implied_prob_source"],
        "edge_source": bet["edge_source"],
        "ev_source": bet["ev_source"],
        "full_kelly_source": bet["full_kelly_source"],
        "kelly_source": bet["kelly_source"],
        "final_total": number_text(total),
        "bet_result": bet_result,
        "bet_units": number_text(units_won(bet["odds_american"], bet_result)),
        "selected_source_file": bet["selected_source_file"],
        "grading_generated_at_utc": generated_at,
    }
    return {column: clean_text(row.get(column, "")) for column in OUTPUT_COLUMNS}


def build_unmatched_row(
    bet: dict[str, str], reason: str, result_file: str = "", status: str = ""
) -> dict[str, str]:
    row = {
        "unmatched_reason": reason,
        **{column: bet.get(column, "") for column in [
            "season", "season_type", "week", "game_id", "game_date",
            "away_team", "home_team", "market_type", "bet_side", "line",
            "odds_american", "model_prob", "selected_source_file",
        ]},
        "result_file": result_file,
        "status": status,
    }
    return {column: clean_text(row.get(column, "")) for column in UNMATCHED_COLUMNS}


def write_csv(rows: list[dict[str, str]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, lineterminator="\n")


def write_weekly_outputs(graded: list[dict[str, str]], all_group_keys: set[tuple[str, str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in graded:
        key = (row["season"], row["season_type"], row["week"])
        grouped.setdefault(key, []).append(row)

    for key in sorted(all_group_keys):
        season, season_type, week = key
        rows = grouped.get(key, [])
        path = GRADED_DIR / f"{season}_{season_type}_{week}_graded.csv"
        write_csv(rows, OUTPUT_COLUMNS, path)


def reconciliation_rows(
    bets: list[dict[str, str]],
    graded: list[dict[str, str]],
    unmatched: list[dict[str, str]],
    selected_game_counts: dict[tuple[str, str, str], int],
) -> list[dict[str, str]]:
    keys = sorted({(b["season"], b["season_type"], b["week"]) for b in bets})
    output: list[dict[str, str]] = []

    for key in keys:
        season, season_type, week = key
        group_bets = [b for b in bets if (b["season"], b["season_type"], b["week"]) == key]
        group_graded = [r for r in graded if (r["season"], r["season_type"], r["week"]) == key]
        group_unmatched = [r for r in unmatched if (r["season"], r["season_type"], r["week"]) == key]

        reason_count = lambda reason: sum(r["unmatched_reason"] == reason for r in group_unmatched)
        balanced = len(group_bets) == len(group_graded) + len(group_unmatched)
        output.append({
            "season": season,
            "season_type": season_type,
            "week": week,
            "selected_game_rows": str(selected_game_counts.get(key, 0)),
            "selected_bets": str(len(group_bets)),
            "graded_bets": str(len(group_graded)),
            "unmatched_bets": str(len(group_unmatched)),
            "not_final_bets": str(reason_count("game_not_final")),
            "missing_result_file_bets": str(reason_count("missing_result_file")),
            "missing_game_id_bets": str(reason_count("missing_game_id")),
            "missing_game_in_results_bets": str(reason_count("missing_game_in_results")),
            "status": "OK" if balanced else "ERROR_UNBALANCED",
        })
    return output


def validation_audit(graded: list[dict[str, str]]) -> list[dict[str, str]]:
    frame = pd.DataFrame(graded, columns=OUTPUT_COLUMNS)
    checks: list[tuple[str, int, str]] = []

    if frame.empty:
        checks.append(("graded_rows_present", 0, "No final selected bets were graded"))
    else:
        checks.append(("graded_rows_present", len(frame), ""))

        invalid_results = int((~frame["bet_result"].isin(["Win", "Loss", "Push"])).sum())
        checks.append(("invalid_bet_result_rows", invalid_results, "Expected Win/Loss/Push"))

        blank_game_ids = int(frame["game_id"].astype(str).str.strip().eq("").sum())
        checks.append(("blank_game_id_rows", blank_game_ids, ""))

        blank_odds = int(frame["odds_american"].astype(str).str.strip().eq("").sum())
        checks.append(("blank_odds_rows", blank_odds, ""))

        needs_line = frame["market_type"].isin(["spread", "total"])
        blank_line = int((needs_line & frame["line"].astype(str).str.strip().eq("")).sum())
        checks.append(("spread_total_blank_line_rows", blank_line, ""))

        dupes = int(frame.duplicated(BET_KEY, keep=False).sum())
        checks.append(("duplicate_graded_key_rows", dupes, ""))

        non_final = int((~frame["status"].map(is_final)).sum())
        checks.append(("graded_non_final_rows", non_final, ""))

    output = []
    for name, count, detail in checks:
        # graded_rows_present is informational; every other count must be zero.
        status = "OK" if (name == "graded_rows_present" or count == 0) else "ERROR"
        output.append({"check": name, "count": str(count), "status": status, "detail": detail})
    return output


def result_counts(graded: list[dict[str, str]]) -> pd.DataFrame:
    columns = ["season", "season_type", "week", "market_type", "bet_result", "count"]
    if not graded:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(graded)
    out = (
        frame.groupby(["season", "season_type", "week", "market_type", "bet_result"], dropna=False)
        .size().reset_index(name="count")
        .sort_values(["season", "season_type", "week", "market_type", "bet_result"])
    )
    return out[columns]


def spot_check_rows(graded: list[dict[str, str]], limit: int = 50) -> list[dict[str, str]]:
    rows = []
    for row in sorted(graded, key=lambda r: (r["season"], r["week"], r["game_id"], r["market_type"]))[:limit]:
        away = to_float(row["away_score"])
        home = to_float(row["home_score"])
        line = to_float(row["line"])
        calculation = ""
        if away is not None and home is not None:
            if row["market_type"] == "moneyline":
                calculation = f"home={home:g} away={away:g} side={row['bet_side']}"
            elif row["market_type"] == "spread" and line is not None:
                selected_score = home if row["bet_side"] == "home" else away
                opponent_score = away if row["bet_side"] == "home" else home
                calculation = f"selected_score={selected_score:g} + line={line:g} vs opponent={opponent_score:g}"
            elif row["market_type"] == "total" and line is not None:
                calculation = f"final_total={home + away:g} vs line={line:g} side={row['bet_side']}"
        rows.append({
            "season": row["season"], "season_type": row["season_type"], "week": row["week"],
            "game_id": row["game_id"], "market_type": row["market_type"], "bet_side": row["bet_side"],
            "line": row["line"], "away_score": row["away_score"], "home_score": row["home_score"],
            "calculation": calculation, "bet_result": row["bet_result"],
        })
    return rows


def main() -> None:
    reset_logs()
    for directory in [GRADED_DIR, UNMATCHED_DIR, AUDIT_DIR, ERROR_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    selected_files = sorted(SELECTED_DIR.glob(SELECTED_PATTERN))
    if not selected_files:
        fail(f"No selected pick files found in {SELECTED_DIR}")

    all_bets: list[dict[str, str]] = []
    duplicate_audit: list[dict[str, str]] = []
    selected_game_counts: dict[tuple[str, str, str], int] = {}

    for path in selected_files:
        bets, game_rows = explode_selected_file(path)
        all_bets.extend(bets)
        file_keys = {(b["season"], b["season_type"], b["week"]) for b in bets}
        if len(file_keys) == 1:
            key = next(iter(file_keys))
            selected_game_counts[key] = selected_game_counts.get(key, 0) + game_rows
        elif not bets:
            # Empty selected file is valid but cannot infer season/week from selected bets.
            log(SUMMARY_LOG, f"selected file has no selected markets: {path}")
        else:
            fail(f"{path}: selected bets span multiple season/type/week groups: {sorted(file_keys)}")

    all_bets = collapse_selected_duplicates(all_bets, duplicate_audit)
    generated_at = now_utc()
    result_cache: dict[tuple[str, str, str], dict[str, dict[str, str]] | None] = {}
    graded: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []

    for bet in all_bets:
        group = (bet["season"], bet["season_type"], bet["week"])
        result_file = f"{group[0]}_{group[1]}_{group[2]}.csv"
        results = load_result_group(*group, result_cache, duplicate_audit)

        if results is None:
            unmatched.append(build_unmatched_row(bet, "missing_result_file", result_file=result_file))
            continue

        game_id = bet["game_id"]
        if not game_id:
            unmatched.append(build_unmatched_row(bet, "missing_game_id", result_file=result_file))
            continue

        result = results.get(game_id)
        if result is None:
            unmatched.append(build_unmatched_row(bet, "missing_game_in_results", result_file=result_file))
            continue

        status = clean_text(result.get("status"))
        if not is_final(status):
            unmatched.append(build_unmatched_row(bet, "game_not_final", result_file=result_file, status=status))
            continue

        graded.append(build_graded_row(bet, result, generated_at))

    all_group_keys = {(b["season"], b["season_type"], b["week"]) for b in all_bets}
    write_weekly_outputs(graded, all_group_keys)
    write_csv(graded, OUTPUT_COLUMNS, MASTER_FILE)
    write_csv(unmatched, UNMATCHED_COLUMNS, UNMATCHED_FILE)
    write_csv([row for row in unmatched if row["unmatched_reason"] == "game_not_final"], UNMATCHED_COLUMNS, NOT_FINAL_FILE)
    write_csv(duplicate_audit, DUPLICATE_COLUMNS, DUPLICATE_FILE)

    recon = reconciliation_rows(all_bets, graded, unmatched, selected_game_counts)
    write_csv(recon, RECON_COLUMNS, RECON_FILE)

    validation = validation_audit(graded)
    write_csv(validation, ["check", "count", "status", "detail"], VALIDATION_FILE)
    result_counts(graded).to_csv(RESULT_COUNTS_FILE, index=False, lineterminator="\n")

    spot_columns = [
        "season", "season_type", "week", "game_id", "market_type", "bet_side", "line",
        "away_score", "home_score", "calculation", "bet_result",
    ]
    write_csv(spot_check_rows(graded), spot_columns, SPOT_CHECK_FILE)

    if any(row["status"] != "OK" for row in recon):
        fail("Selected-vs-graded reconciliation is unbalanced; see audit output")
    validation_errors = [row for row in validation if row["status"] == "ERROR"]
    if validation_errors:
        fail(f"Graded output validation failed: {validation_errors}")

    summary = (
        f"selected_files={len(selected_files)} selected_bets={len(all_bets)} "
        f"graded_bets={len(graded)} unmatched_bets={len(unmatched)} "
        f"duplicate_audit_rows={len(duplicate_audit)} master={MASTER_FILE}"
    )
    print(summary)
    log(SUMMARY_LOG, summary)


if __name__ == "__main__":
    main()
