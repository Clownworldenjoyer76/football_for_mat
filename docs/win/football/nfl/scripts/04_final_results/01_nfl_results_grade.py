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
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
NFL_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = NFL_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


pd = None

SELECTED_DIR = NFL_ROOT / "03_picks" / "selected"
RESULTS_DIR = NFL_ROOT / "04_final_results" / "results"
GRADED_DIR = RESULTS_DIR / "graded"
UNMATCHED_DIR = RESULTS_DIR / "unmatched"
AUDIT_DIR = RESULTS_DIR / "audit"
ERROR_DIR = NFL_ROOT / "errors" / "04_final_results"

SELECTED_PATTERN = "week_*_NFL_select_picks.csv"
SELECTED_FILE_RE = re.compile(
    r"^week_(\d+)_NFL_select_picks\.csv$"
)

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


def load_runtime_dependencies(
    reporter: PipelineReporter,
) -> None:
    global pd

    try:
        import pandas as pd_module
    except Exception:
        reporter.set_detail(
            "dependency_imports_ok",
            False,
        )
        raise

    pd = pd_module
    reporter.set_detail(
        "dependency_imports_ok",
        True,
    )


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


def validate_csv_header(
    path: Path,
    *,
    label: str,
) -> list[str]:
    if not path.is_file():
        fail(f"Missing file: {path}")

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            header = next(csv.reader(handle), [])
    except UnicodeDecodeError as exc:
        fail(
            f"{label}: invalid UTF-8 CSV: "
            f"{path}: {exc}"
        )

    if not header:
        fail(
            f"{label}: missing CSV header: "
            f"{path}"
        )

    normalized = [
        clean_text(column)
        for column in header
    ]

    if any(not column for column in normalized):
        fail(
            f"{label}: blank CSV header "
            f"column: {path}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []

    for column in normalized:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)

    if duplicates:
        fail(
            f"{label}: duplicate header "
            f"columns: {duplicates}"
        )

    return header


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        fail(
            f"{label}: missing required "
            f"columns: {missing}"
        )


def read_csv(
    path: Path,
    required: Iterable[str],
    label: str,
) -> pd.DataFrame:
    validate_csv_header(
        path,
        label=label,
    )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )

    require_columns(
        frame,
        required,
        label,
    )

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


def parse_positive_int(
    value: Any,
    *,
    label: str,
) -> int:
    number = to_float(value)

    if (
        number is None
        or not number.is_integer()
        or number <= 0
    ):
        fail(
            f"{label} must be a positive "
            f"integer; found {value!r}"
        )

    return int(number)


def validate_score(
    value: Any,
    *,
    label: str,
) -> float:
    number = to_float(value)

    if (
        number is None
        or number < 0
        or not number.is_integer()
    ):
        fail(
            f"{label} must be a nonnegative "
            f"integer score; found {value!r}"
        )

    return number


def number_text(value: float | None, precision: int = 12) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{precision}g}"


def strict_selection_flag(
    value: Any,
    *,
    label: str,
) -> int:
    number = to_float(value)

    if number not in {0.0, 1.0}:
        fail(
            f"{label} must be exactly 0 or 1; "
            f"found {value!r}"
        )

    return int(number)


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


def explode_selected_file(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    int,
    tuple[str, str, str] | None,
]:
    frame = read_csv(
        path,
        SELECTED_REQUIRED,
        f"selected {path}",
    )

    filename_match = SELECTED_FILE_RE.fullmatch(path.name)

    if filename_match is None:
        fail(
            f"Selected filename does not match "
            f"expected pattern: {path.name}"
        )

    filename_week = parse_positive_int(
        filename_match.group(1),
        label=f"{path.name} filename week",
    )

    if frame.empty:
        return [], 0, None

    bets: list[dict[str, str]] = []
    seasons: set[str] = set()
    season_types: set[str] = set()
    weeks: set[str] = set()
    seen_game_ids: set[str] = set()

    for row_index, row in frame.iterrows():
        row_number = row_index + 2

        season = str(
            parse_positive_int(
                row.get("season"),
                label=(
                    f"{path}: row {row_number} "
                    "season"
                ),
            )
        )

        week_int = parse_positive_int(
            row.get("week"),
            label=(
                f"{path}: row {row_number} week"
            ),
        )

        if week_int != filename_week:
            fail(
                f"{path}: row {row_number} "
                f"week={week_int} does not "
                f"match filename week="
                f"{filename_week}"
            )

        week = str(week_int)
        season_type = clean_text(
            row.get("season_type")
        ).lower()
        game_id = clean(row.get("game_id"))
        game_date = clean_text(row.get("game_date"))
        away_team = clean_text(row.get("away_team"))
        home_team = clean_text(row.get("home_team"))

        if not season_type:
            fail(
                f"{path}: row {row_number} "
                "season_type is blank"
            )

        if not game_id:
            fail(
                f"{path}: row {row_number} "
                "game_id is blank"
            )

        if game_id in seen_game_ids:
            fail(
                f"{path}: duplicate selected "
                f"game_id={game_id}"
            )

        seen_game_ids.add(game_id)

        for column, value in (
            ("game_date", game_date),
            ("away_team", away_team),
            ("home_team", home_team),
        ):
            if not value:
                fail(
                    f"{path}: row {row_number} "
                    f"{column} is blank"
                )

        if away_team == home_team:
            fail(
                f"{path}: row {row_number} "
                "away_team and home_team "
                "must differ"
            )

        base = {
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "game_date": game_date,
            "game_time": clean_text(row.get("game_time")),
            "commence_time": clean_text(row.get("commence_time")),
            "edt_time": clean_text(row.get("edt_time")),
            "away_team": away_team,
            "home_team": home_team,
            "selected_source_file": path.name,
            "selected_row_number": str(row_number),
        }

        seasons.add(season)
        season_types.add(season_type)
        weeks.add(week)

        selected_count = 0

        for market_type, spec in MARKET_SPECS.items():
            selected = strict_selection_flag(
                row.get(spec["selected"], ""),
                label=(
                    f"{path}: row {row_number} "
                    f"game_id={game_id} "
                    f"{spec['selected']}"
                ),
            )

            if selected == 0:
                continue

            selected_count += 1

            selection = clean_text(
                row.get(spec["selection"], "")
            ).upper()

            if selection not in spec["valid_sides"]:
                fail(
                    f"{path}: row {row_number} "
                    f"game_id={game_id} "
                    f"invalid {market_type} "
                    f"selection={selection!r}"
                )

            line_text = ""
            if spec["line"] is not None:
                line_text = clean_text(
                    row.get(spec["line"], "")
                )
                if to_float(line_text) is None:
                    fail(
                        f"{path}: row {row_number} "
                        f"game_id={game_id} "
                        f"selected {market_type} "
                        "missing/invalid "
                        f"line={line_text!r}"
                    )

            odds_text = clean_text(
                row.get(spec["odds"], "")
            )
            odds = to_float(odds_text)

            if odds is None or odds == 0:
                fail(
                    f"{path}: row {row_number} "
                    f"game_id={game_id} "
                    f"selected {market_type} "
                    "missing/invalid "
                    f"odds={odds_text!r}"
                )

            metrics = build_metrics(
                row,
                spec["prefix"],
                odds_text,
            )

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

        if selected_count < 1:
            fail(
                f"{path}: row {row_number} "
                f"game_id={game_id} has no "
                "selected market"
            )

    if len(seasons) != 1:
        fail(
            f"{path}: selected file spans "
            f"multiple seasons: "
            f"{sorted(seasons)}"
        )

    if len(season_types) != 1:
        fail(
            f"{path}: selected file spans "
            "multiple season_type values: "
            f"{sorted(season_types)}"
        )

    if weeks != {str(filename_week)}:
        fail(
            f"{path}: selected file week "
            f"integrity failed: "
            f"{sorted(weeks)}"
        )

    group = (
        next(iter(seasons)),
        next(iter(season_types)),
        str(filename_week),
    )

    return bets, len(frame), group


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
    cache: dict[
        tuple[str, str, str],
        dict[str, dict[str, str]] | None,
    ],
    duplicate_audit: list[dict[str, str]],
) -> dict[str, dict[str, str]] | None:
    key = (season, season_type, week)

    if key in cache:
        return cache[key]

    path = RESULTS_DIR / f"{season}_{season_type}_{week}.csv"

    if not path.exists():
        cache[key] = None
        return None

    frame = read_csv(
        path,
        RESULT_REQUIRED,
        f"results {path}",
    )

    if frame.empty:
        fail(
            f"{path}: result file contains "
            "no rows"
        )

    expected_season = parse_positive_int(
        season,
        label="requested result season",
    )
    expected_week = parse_positive_int(
        week,
        label="requested result week",
    )
    expected_type = clean_text(season_type).lower()

    groups: dict[str, list[dict[str, str]]] = {}

    for row_index, row in frame.iterrows():
        row_number = row_index + 2

        record = {
            column: clean_text(row.get(column, ""))
            for column in frame.columns
        }

        row_season = parse_positive_int(
            record.get("season"),
            label=(
                f"{path}: row {row_number} season"
            ),
        )
        row_week = parse_positive_int(
            record.get("week"),
            label=(
                f"{path}: row {row_number} week"
            ),
        )
        row_type = clean_text(
            record.get("season_type")
        ).lower()

        if (
            row_season != expected_season
            or row_week != expected_week
            or row_type != expected_type
        ):
            fail(
                f"{path}: row {row_number} "
                "season/type/week does not "
                "match requested result "
                f"group={key}"
            )

        game_id = clean(record.get("game_id"))
        if not game_id:
            fail(
                f"{path}: row {row_number} "
                "final-results row missing "
                "game_id"
            )

        game_date = clean_text(record.get("game_date"))
        away_team = clean_text(record.get("away_team"))
        home_team = clean_text(record.get("home_team"))
        status = clean_text(record.get("status"))

        for column, value in (
            ("game_date", game_date),
            ("away_team", away_team),
            ("home_team", home_team),
            ("status", status),
        ):
            if not value:
                fail(
                    f"{path}: row {row_number} "
                    f"game_id={game_id} "
                    f"{column} is blank"
                )

        if away_team == home_team:
            fail(
                f"{path}: row {row_number} "
                f"game_id={game_id} "
                "away_team and home_team "
                "must differ"
            )

        validate_score(
            record.get("away_score"),
            label=(
                f"{path}: row {row_number} "
                f"game_id={game_id} away_score"
            ),
        )
        validate_score(
            record.get("home_score"),
            label=(
                f"{path}: row {row_number} "
                f"game_id={game_id} home_score"
            ),
        )

        record["season"] = str(row_season)
        record["season_type"] = row_type
        record["week"] = str(row_week)
        record["game_id"] = game_id
        record["game_date"] = game_date
        record["away_team"] = away_team
        record["home_team"] = home_team
        record["status"] = status

        groups.setdefault(game_id, []).append(record)

    indexed: dict[str, dict[str, str]] = {}

    for game_id, rows in groups.items():
        if len(rows) == 1:
            indexed[game_id] = rows[0]
            continue

        signatures = {
            tuple(sorted(row.items()))
            for row in rows
        }
        identical = len(signatures) == 1

        duplicate_audit.append({
            "duplicate_scope": "final_result_game",
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "market_type": "",
            "bet_side": "",
            "line": "",
            "duplicate_count": str(len(rows)),
            "identical_duplicate": str(identical),
            "action_taken": (
                "collapsed_identical"
                if identical
                else "failed_conflict"
            ),
            "source_files": path.name,
        })

        if not identical:
            fail(
                f"{path}: conflicting "
                f"duplicate game_id={game_id}"
            )

        indexed[game_id] = rows[0]

    cache[key] = indexed
    return indexed


def normalize_status(value: Any) -> str:
    return clean_text(value).casefold()


def is_final(value: Any) -> bool:
    status = normalize_status(value)
    return status.startswith("final") or status in {"completed", "complete", "game over"}


def validate_result_identity(
    bet: dict[str, str],
    result: dict[str, str],
) -> None:
    game_id = bet["game_id"]

    for column in (
        "game_date",
        "away_team",
        "home_team",
    ):
        selected_value = clean_text(
            bet.get(column, "")
        )
        result_value = clean_text(
            result.get(column, "")
        )

        if selected_value != result_value:
            fail(
                f"game_id={game_id}: "
                f"result {column}="
                f"{result_value!r} does not "
                "match selected value="
                f"{selected_value!r}"
            )


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


def normalize_output_row(
    row: dict[str, Any],
    columns: list[str],
) -> dict[str, str]:
    return {
        column: clean_text(row.get(column, ""))
        for column in columns
    }


def make_output_spec(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "path": path.resolve(),
        "columns": list(columns),
        "rows": [
            normalize_output_row(row, columns)
            for row in rows
        ],
    }


def build_weekly_output_specs(
    graded: list[dict[str, str]],
    all_group_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}

    for row in graded:
        key = (
            row["season"],
            row["season_type"],
            row["week"],
        )
        grouped.setdefault(key, []).append(row)

    specs: list[dict[str, Any]] = []

    for key in sorted(
        all_group_keys,
        key=lambda value: (
            value[0],
            value[1],
            int(value[2]),
        ),
    ):
        season, season_type, week = key
        specs.append(
            make_output_spec(
                GRADED_DIR
                / (
                    f"{season}_{season_type}_"
                    f"{week}_graded.csv"
                ),
                OUTPUT_COLUMNS,
                grouped.get(key, []),
            )
        )

    return specs


def result_count_rows(
    graded: list[dict[str, str]],
) -> list[dict[str, str]]:
    frame = result_counts(graded)

    return [
        {
            column: clean_text(row.get(column, ""))
            for column in frame.columns
        }
        for _, row in frame.iterrows()
    ]


def validate_graded_rows(
    graded: list[dict[str, str]],
) -> None:
    seen_keys: set[tuple[str, ...]] = set()

    for row_index, row in enumerate(
        graded,
        start=2,
    ):
        if set(row) != set(OUTPUT_COLUMNS):
            fail(
                f"graded row {row_index}: "
                "output column contract failed"
            )

        for column in (
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "status",
            "market",
            "selection",
            "odds_american",
            "market_type",
            "bet_side",
            "bet_result",
            "result",
            "bet_units",
            "selected_source_file",
            "grading_generated_at_utc",
        ):
            if not clean_text(row.get(column, "")):
                fail(
                    f"graded row {row_index}: "
                    f"{column} is blank"
                )

        parse_positive_int(
            row["season"],
            label=f"graded row {row_index} season",
        )
        parse_positive_int(
            row["week"],
            label=f"graded row {row_index} week",
        )

        away = validate_score(
            row["away_score"],
            label=f"graded row {row_index} away_score",
        )
        home = validate_score(
            row["home_score"],
            label=f"graded row {row_index} home_score",
        )

        if not is_final(row["status"]):
            fail(
                f"graded row {row_index}: "
                "status is not final"
            )

        market_type = clean_text(
            row["market_type"]
        ).lower()

        if market_type not in MARKET_SPECS:
            fail(
                f"graded row {row_index}: "
                f"unsupported market_type="
                f"{market_type!r}"
            )

        spec = MARKET_SPECS[market_type]
        selection = clean_text(
            row["selection"]
        ).upper()
        bet_side = clean_text(
            row["bet_side"]
        ).lower()

        if (
            selection not in spec["valid_sides"]
            or bet_side != selection.lower()
            or clean_text(row["market"]) != spec["market"]
        ):
            fail(
                f"graded row {row_index}: "
                "market/selection identity "
                "contract failed"
            )

        odds = to_float(row["odds_american"])
        if odds is None or odds == 0:
            fail(
                f"graded row {row_index}: "
                "invalid odds_american"
            )

        if (
            spec["line"] is not None
            and to_float(row["line"]) is None
        ):
            fail(
                f"graded row {row_index}: "
                "spread/total line is invalid"
            )

        final_total = to_float(row["final_total"])
        if (
            final_total is None
            or abs(final_total - (away + home)) > EPSILON
        ):
            fail(
                f"graded row {row_index}: "
                "final_total does not match "
                "scores"
            )

        bet_result = clean_text(row["bet_result"])
        compat_expected = {
            "Win": "WIN",
            "Loss": "LOSS",
            "Push": "PUSH",
        }.get(bet_result)

        if (
            compat_expected is None
            or clean_text(row["result"]) != compat_expected
        ):
            fail(
                f"graded row {row_index}: "
                "result compatibility contract "
                "failed"
            )

        expected_units = units_won(
            row["odds_american"],
            bet_result,
        )
        actual_units = to_float(row["bet_units"])

        if (
            actual_units is None
            or abs(actual_units - expected_units) > 1e-9
        ):
            fail(
                f"graded row {row_index}: "
                "bet_units contract failed"
            )

        key = tuple(
            clean_text(row.get(column, ""))
            for column in BET_KEY
        )

        if key in seen_keys:
            fail(
                f"graded row {row_index}: "
                f"duplicate graded key={key}"
            )

        seen_keys.add(key)


def validate_unmatched_rows(
    unmatched: list[dict[str, str]],
) -> None:
    allowed_reasons = {
        "missing_result_file",
        "missing_game_id",
        "missing_game_in_results",
        "game_not_final",
    }

    for row_index, row in enumerate(
        unmatched,
        start=2,
    ):
        if set(row) != set(UNMATCHED_COLUMNS):
            fail(
                f"unmatched row {row_index}: "
                "column contract failed"
            )

        reason = clean_text(row["unmatched_reason"])
        if reason not in allowed_reasons:
            fail(
                f"unmatched row {row_index}: "
                f"invalid reason={reason!r}"
            )

        for column in (
            "season",
            "season_type",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "market_type",
            "bet_side",
            "odds_american",
            "selected_source_file",
        ):
            if not clean_text(row.get(column, "")):
                fail(
                    f"unmatched row {row_index}: "
                    f"{column} is blank"
                )

        if (
            reason != "missing_game_id"
            and not clean(row.get("game_id"))
        ):
            fail(
                f"unmatched row {row_index}: "
                "game_id is blank"
            )

        market_type = clean_text(
            row["market_type"]
        ).lower()

        if market_type not in MARKET_SPECS:
            fail(
                f"unmatched row {row_index}: "
                "invalid market_type"
            )

        spec = MARKET_SPECS[market_type]

        if (
            clean_text(row["bet_side"]).upper()
            not in spec["valid_sides"]
        ):
            fail(
                f"unmatched row {row_index}: "
                "invalid bet_side"
            )

        odds = to_float(row["odds_american"])
        if odds is None or odds == 0:
            fail(
                f"unmatched row {row_index}: "
                "invalid odds"
            )

        if (
            spec["line"] is not None
            and to_float(row["line"]) is None
        ):
            fail(
                f"unmatched row {row_index}: "
                "invalid line"
            )

        if reason == "game_not_final":
            if (
                not clean_text(row["result_file"])
                or not clean_text(row["status"])
                or is_final(row["status"])
            ):
                fail(
                    f"unmatched row {row_index}: "
                    "game_not_final contract "
                    "failed"
                )

        if (
            reason in {
                "missing_result_file",
                "missing_game_in_results",
            }
            and not clean_text(row["result_file"])
        ):
            fail(
                f"unmatched row {row_index}: "
                "result_file is blank"
            )


def validate_audit_rows(
    duplicate_audit: list[dict[str, str]],
    recon: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> None:
    for row_index, row in enumerate(
        duplicate_audit,
        start=2,
    ):
        if set(row) != set(DUPLICATE_COLUMNS):
            fail(
                f"duplicate audit row "
                f"{row_index}: column "
                "contract failed"
            )

    for row_index, row in enumerate(
        recon,
        start=2,
    ):
        if set(row) != set(RECON_COLUMNS):
            fail(
                f"reconciliation row "
                f"{row_index}: column "
                "contract failed"
            )

        if clean_text(row["status"]) != "OK":
            fail(
                "Selected-vs-graded "
                "reconciliation is "
                "unbalanced; see audit "
                "output"
            )

        selected_bets = int(clean_text(row["selected_bets"]))
        graded_bets = int(clean_text(row["graded_bets"]))
        unmatched_bets = int(clean_text(row["unmatched_bets"]))

        if (
            selected_bets < 0
            or graded_bets < 0
            or unmatched_bets < 0
            or selected_bets
            != graded_bets + unmatched_bets
        ):
            fail(
                f"reconciliation row "
                f"{row_index}: count "
                "contract failed"
            )

    for row in validation:
        if set(row) != {
            "check",
            "count",
            "status",
            "detail",
        }:
            fail(
                "validation audit column "
                "contract failed"
            )

        if clean_text(row["status"]) != "OK":
            fail(
                "Graded output validation "
                f"failed: {row}"
            )


def validate_serialized_output(
    path: Path,
    spec: dict[str, Any],
) -> None:
    header = validate_csv_header(
        path,
        label=f"output {path}",
    )
    columns = spec["columns"]

    if header != columns:
        fail(
            f"{path}: output header "
            "contract failed"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                fail(
                    f"{path}: row "
                    f"{row_number} contains "
                    "unexpected extra fields"
                )

            rows.append({
                column: clean_text(row.get(column, ""))
                for column in columns
            })

    if rows != spec["rows"]:
        fail(
            f"{path}: staged/published "
            "CSV differs from expected "
            "in-memory output"
        )


def stage_output_spec(
    spec: dict[str, Any],
) -> Path:
    live_path: Path = spec["path"]
    live_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{live_path.name}.stage.",
        suffix=".csv",
        dir=str(live_path.parent),
    )
    os.close(descriptor)
    stage_path = Path(raw_path)

    try:
        pd.DataFrame(
            spec["rows"],
            columns=spec["columns"],
        ).to_csv(
            stage_path,
            index=False,
            lineterminator="\n",
        )

        validate_serialized_output(
            stage_path,
            spec,
        )
        return stage_path
    except Exception:
        stage_path.unlink(missing_ok=True)
        raise


def stale_weekly_graded_paths(
    expected_paths: set[Path],
) -> list[Path]:
    if not GRADED_DIR.exists():
        return []

    pattern = re.compile(
        r"^\d+_[A-Za-z0-9-]+_"
        r"\d+_graded\.csv$"
    )

    stale: list[Path] = []

    for path in GRADED_DIR.iterdir():
        if (
            not path.is_file()
            or not pattern.fullmatch(path.name)
        ):
            continue

        resolved = path.resolve()
        if resolved not in expected_paths:
            stale.append(resolved)

    return sorted(stale)


def publish_outputs(
    specs: list[dict[str, Any]],
    staged: dict[Path, Path],
    stale_paths: list[Path],
    *,
    reporter: PipelineReporter,
) -> None:
    spec_by_path = {
        spec["path"]: spec
        for spec in specs
    }
    managed_paths = [
        *spec_by_path.keys(),
        *stale_paths,
    ]

    backup_root = Path(
        tempfile.mkdtemp(
            prefix="nfl_grade_backups_"
        )
    )
    backups: dict[Path, Path] = {}
    live_modified = False
    rollback_failed = False

    reporter.update_details({
        "publication_mode": (
            "transactional_multi_file_"
            "atomic_replace_with_rollback"
        ),
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    try:
        for live_path in managed_paths:
            if not live_path.exists():
                continue

            backup_path = (
                backup_root
                / f"{uuid.uuid4().hex}_{live_path.name}"
            )
            shutil.copy2(
                live_path,
                backup_path,
            )
            backups[live_path] = backup_path

        for live_path, stage_path in staged.items():
            os.replace(
                stage_path,
                live_path,
            )
            live_modified = True

        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
                live_modified = True

        for live_path, spec in spec_by_path.items():
            validate_serialized_output(
                live_path,
                spec,
            )

        remaining_stale = [
            str(path)
            for path in stale_paths
            if path.exists()
        ]
        if remaining_stale:
            fail(
                "Stale weekly graded files "
                "remain after publication: "
                f"{remaining_stale}"
            )

        reporter.update_details({
            "publication_completed": True,
            "post_publish_validation": True,
        })
    except Exception as publish_exc:
        if live_modified:
            try:
                for live_path in managed_paths:
                    backup_path = backups.get(live_path)

                    if (
                        backup_path is not None
                        and backup_path.exists()
                    ):
                        restore_descriptor, restore_raw = (
                            tempfile.mkstemp(
                                prefix=(
                                    f".{live_path.name}."
                                    "restore."
                                ),
                                suffix=".tmp",
                                dir=str(live_path.parent),
                            )
                        )
                        os.close(restore_descriptor)
                        restore_path = Path(restore_raw)

                        try:
                            shutil.copy2(
                                backup_path,
                                restore_path,
                            )
                            os.replace(
                                restore_path,
                                live_path,
                            )
                        finally:
                            restore_path.unlink(
                                missing_ok=True
                            )
                    elif live_path.exists():
                        live_path.unlink()

                reporter.update_details({
                    "publication_completed": False,
                    "post_publish_validation": False,
                    "rollback_performed": True,
                })
            except Exception as rollback_exc:
                rollback_failed = True
                reporter.update_details({
                    "publication_completed": False,
                    "post_publish_validation": False,
                    "rollback_performed": False,
                    "rollback_error_type": (
                        type(rollback_exc).__name__
                    ),
                    "rollback_error": str(
                        rollback_exc
                    ),
                })
                raise RuntimeError(
                    "NFL grader publication "
                    "failed and rollback also "
                    "failed: "
                    f"publication_error="
                    f"{publish_exc}; "
                    f"rollback_error="
                    f"{rollback_exc}"
                ) from rollback_exc

        raise
    finally:
        for stage_path in staged.values():
            stage_path.unlink(missing_ok=True)

        try:
            shutil.rmtree(
                backup_root,
                ignore_errors=False,
            )
        except Exception as cleanup_exc:
            reporter.warning(
                "Temporary NFL grader "
                "backup cleanup failed",
                backup_root=str(backup_root),
                error_type=(
                    type(cleanup_exc).__name__
                ),
                error=str(cleanup_exc),
                rollback_failed=rollback_failed,
            )


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


def run(
    reporter: PipelineReporter,
) -> None:
    reset_logs()
    reporter.add_output(SUMMARY_LOG)
    reporter.add_output(ERROR_LOG)

    load_runtime_dependencies(reporter)

    for directory in [
        GRADED_DIR,
        UNMATCHED_DIR,
        AUDIT_DIR,
        ERROR_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    selected_files = sorted(
        SELECTED_DIR.glob(SELECTED_PATTERN)
    )

    if not selected_files:
        fail(
            "No selected pick files found "
            f"in {SELECTED_DIR}"
        )

    all_bets: list[dict[str, str]] = []
    duplicate_audit: list[dict[str, str]] = []
    selected_game_counts: dict[
        tuple[str, str, str],
        int,
    ] = {}
    selected_groups: dict[
        tuple[str, str, str],
        Path,
    ] = {}
    selected_game_rows_total = 0

    for path in selected_files:
        reporter.add_input(path)

        bets, game_rows, file_group = (
            explode_selected_file(path)
        )

        selected_game_rows_total += game_rows
        all_bets.extend(bets)

        if file_group is None:
            log(
                SUMMARY_LOG,
                (
                    "selected file has no "
                    "selected rows: "
                    f"{path}"
                ),
            )
            continue

        if file_group in selected_groups:
            fail(
                "Multiple selected files map "
                f"to group={file_group}: "
                f"{selected_groups[file_group]} "
                f"and {path}"
            )

        selected_groups[file_group] = path
        selected_game_counts[file_group] = game_rows

    all_bets = collapse_selected_duplicates(
        all_bets,
        duplicate_audit,
    )

    bet_groups = {
        (
            bet["season"],
            bet["season_type"],
            bet["week"],
        )
        for bet in all_bets
    }

    seasons = {
        group[0]
        for group in bet_groups
    }
    weeks = {
        group[2]
        for group in bet_groups
    }

    if len(seasons) == 1:
        reporter.season = next(iter(seasons))

    if len(weeks) == 1:
        reporter.week = next(iter(weeks))

    for group in sorted(
        bet_groups,
        key=lambda value: (
            value[0],
            value[1],
            int(value[2]),
        ),
    ):
        reporter.add_input(
            RESULTS_DIR
            / (
                f"{group[0]}_"
                f"{group[1]}_"
                f"{group[2]}.csv"
            )
        )

    generated_at = now_utc()
    result_cache: dict[
        tuple[str, str, str],
        dict[str, dict[str, str]] | None,
    ] = {}
    graded: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []

    for bet in all_bets:
        group = (
            bet["season"],
            bet["season_type"],
            bet["week"],
        )
        result_file = (
            f"{group[0]}_"
            f"{group[1]}_"
            f"{group[2]}.csv"
        )

        results = load_result_group(
            *group,
            result_cache,
            duplicate_audit,
        )

        if results is None:
            unmatched.append(
                build_unmatched_row(
                    bet,
                    "missing_result_file",
                    result_file=result_file,
                )
            )
            continue

        game_id = bet["game_id"]

        if not game_id:
            unmatched.append(
                build_unmatched_row(
                    bet,
                    "missing_game_id",
                    result_file=result_file,
                )
            )
            continue

        result = results.get(game_id)

        if result is None:
            unmatched.append(
                build_unmatched_row(
                    bet,
                    "missing_game_in_results",
                    result_file=result_file,
                )
            )
            continue

        validate_result_identity(
            bet,
            result,
        )

        status = clean_text(
            result.get("status")
        )

        if not is_final(status):
            unmatched.append(
                build_unmatched_row(
                    bet,
                    "game_not_final",
                    result_file=result_file,
                    status=status,
                )
            )
            continue

        graded.append(
            build_graded_row(
                bet,
                result,
                generated_at,
            )
        )

    all_group_keys = {
        (
            bet["season"],
            bet["season_type"],
            bet["week"],
        )
        for bet in all_bets
    }

    validate_graded_rows(graded)
    validate_unmatched_rows(unmatched)

    recon = reconciliation_rows(
        all_bets,
        graded,
        unmatched,
        selected_game_counts,
    )
    validation = validation_audit(graded)

    validate_audit_rows(
        duplicate_audit,
        recon,
        validation,
    )

    spot_columns = [
        "season",
        "season_type",
        "week",
        "game_id",
        "market_type",
        "bet_side",
        "line",
        "away_score",
        "home_score",
        "calculation",
        "bet_result",
    ]

    specs = (
        build_weekly_output_specs(
            graded,
            all_group_keys,
        )
        + [
            make_output_spec(
                MASTER_FILE,
                OUTPUT_COLUMNS,
                graded,
            ),
            make_output_spec(
                UNMATCHED_FILE,
                UNMATCHED_COLUMNS,
                unmatched,
            ),
            make_output_spec(
                NOT_FINAL_FILE,
                UNMATCHED_COLUMNS,
                [
                    row
                    for row in unmatched
                    if (
                        row["unmatched_reason"]
                        == "game_not_final"
                    )
                ],
            ),
            make_output_spec(
                DUPLICATE_FILE,
                DUPLICATE_COLUMNS,
                duplicate_audit,
            ),
            make_output_spec(
                RECON_FILE,
                RECON_COLUMNS,
                recon,
            ),
            make_output_spec(
                VALIDATION_FILE,
                [
                    "check",
                    "count",
                    "status",
                    "detail",
                ],
                validation,
            ),
            make_output_spec(
                RESULT_COUNTS_FILE,
                [
                    "season",
                    "season_type",
                    "week",
                    "market_type",
                    "bet_result",
                    "count",
                ],
                result_count_rows(graded),
            ),
            make_output_spec(
                SPOT_CHECK_FILE,
                spot_columns,
                spot_check_rows(graded),
            ),
        ]
    )

    output_paths = [
        spec["path"]
        for spec in specs
    ]

    if len(output_paths) != len(set(output_paths)):
        fail(
            "Duplicate grader output path "
            "detected"
        )

    for output_path in output_paths:
        reporter.add_output(output_path)

    expected_weekly_paths = {
        spec["path"]
        for spec in specs
        if (
            spec["path"].parent
            == GRADED_DIR.resolve()
            and spec["path"]
            != MASTER_FILE.resolve()
        )
    }

    stale_paths = stale_weekly_graded_paths(
        expected_weekly_paths
    )

    reporter.set_rows(
        rows_in=selected_game_rows_total,
        rows_out=len(graded),
    )

    reporter.update_details({
        "selected_files": len(selected_files),
        "selected_game_rows": (
            selected_game_rows_total
        ),
        "selected_bets": len(all_bets),
        "graded_bets": len(graded),
        "unmatched_bets": len(unmatched),
        "duplicate_audit_rows": len(
            duplicate_audit
        ),
        "result_groups_requested": len(
            bet_groups
        ),
        "managed_output_files": len(specs),
        "stale_weekly_graded_files": [
            str(path)
            for path in stale_paths
        ],
        "stale_weekly_graded_count": len(
            stale_paths
        ),
        "staged_roundtrip_verified": False,
        "publication_completed": False,
        "post_publish_validation": False,
        "rollback_performed": False,
    })

    staged: dict[Path, Path] = {}

    try:
        for spec in specs:
            live_path = spec["path"]
            staged[live_path] = (
                stage_output_spec(spec)
            )

        reporter.set_detail(
            "staged_roundtrip_verified",
            True,
        )

        publish_outputs(
            specs,
            staged,
            stale_paths,
            reporter=reporter,
        )
    finally:
        for stage_path in staged.values():
            stage_path.unlink(
                missing_ok=True
            )

    summary = (
        f"selected_files={len(selected_files)} "
        f"selected_bets={len(all_bets)} "
        f"graded_bets={len(graded)} "
        f"unmatched_bets={len(unmatched)} "
        f"duplicate_audit_rows="
        f"{len(duplicate_audit)} "
        f"master={MASTER_FILE}"
    )

    print(summary)
    log(
        SUMMARY_LOG,
        summary,
    )


def main() -> int:
    try:
        with PipelineReporter(
            script=SCRIPT_PATH,
            stage="04_final_results",
            report_root=REPORT_ROOT,
            pipeline="NFL",
            league="NFL",
            extra_context={
                "component": (
                    "selected-bet results "
                    "grader"
                ),
            },
        ) as reporter:
            run(reporter)

        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
