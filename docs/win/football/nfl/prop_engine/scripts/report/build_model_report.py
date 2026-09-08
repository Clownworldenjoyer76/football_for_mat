#!/usr/bin/env python3
"""Build out-of-sample model performance reports for the NFL Prop Engine.

Issue 39 reporting policy:
- Use only the frozen untouched test split from Issue 25 model selection.
- Resolve each target's selected architecture from models/{target}/selected_model.json.
- Join only pregame historical context from the canonical Issue 17 feature table.
- Do not retrain, rescore, recalibrate, or consume market-derived inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common

GRAIN = ["season", "week", "game_id", "player_id"]
AUDIT_REL = Path("docs/win/football/nfl/prop_engine/evaluation/model_selection_predictions.parquet")
REPORT_LOG_REL = Path("docs/win/football/nfl/prop_engine/logs/model_performance_report.json")

METRIC_HEADERS = [
    "target",
    "sample_size",
    "mae",
    "rmse",
    "median_absolute_error",
    "r2",
    "poisson_deviance",
    "brier_1plus",
    "logloss_1plus",
    "mean_actual",
    "mean_projection",
    "bias",
]

OUTPUTS = {
    "target": "metrics_by_target.csv",
    "position": "metrics_by_position.csv",
    "usage": "metrics_by_usage.csv",
    "week": "metrics_by_week.csv",
    "role": "metrics_by_role.csv",
    "calibration": "calibration_by_projection_range.csv",
}

SELECTED_PROJECTION_COLUMNS = {
    "baseline": "baseline_projection",
    "direct": "direct_projection",
    "component": "component_projection",
    "direct_component_blend": "blend_projection",
}

# Exact reporting usage context follows the accepted Issue 26 calibration policy.
USAGE_CANDIDATES = {
    "passing_yards": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "passing_tds": [
        "player_pass_attempts_roll3_mean",
        "player_pass_attempts_roll5_mean",
        "player_pass_attempts_ewm5",
        "player_pass_attempts_career_prior",
    ],
    "rushing_yards": [
        "player_carries_roll3_mean",
        "player_carries_roll5_mean",
        "player_carries_ewm5",
        "player_carries_career_prior",
    ],
    "rushing_tds": [
        "player_goal_line_carries_roll3_mean",
        "player_goal_line_carries_roll5_mean",
        "player_carries_roll3_mean",
        "player_carries_career_prior",
    ],
    "receiving_yards": [
        "player_targets_roll3_mean",
        "player_targets_roll5_mean",
        "player_targets_ewm5",
        "player_targets_career_prior",
    ],
    "receiving_tds": [
        "player_red_zone_targets_roll3_mean",
        "player_red_zone_targets_roll5_mean",
        "player_targets_roll3_mean",
        "player_targets_career_prior",
    ],
    "kicking_points": [
        "player_field_goal_attempts_roll3_mean",
        "player_field_goal_attempts_roll5_mean",
        "player_field_goal_attempts_career_prior",
    ],
    "tackles": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
    "sacks": [
        "player_defense_participation_roll3_mean",
        "role_participation_roll3",
        "player_defense_participation_career_prior",
    ],
}
KICKING_USAGE_COMPONENTS = [
    "player_field_goal_attempts_roll3_mean",
    "player_extra_point_attempts_roll3_mean",
]

BASE_CONTEXT_REQUIRED = [
    *GRAIN,
    "position",
    "role_depth_starter_flag_pregame",
    "history_no_nfl_history_flag",
    "history_new_team_flag",
]
INJURY_STATUS_CANDIDATES = [
    "role_injury_status_pregame",
    "injury_game_status",
    "depth_injury",
]
INJURY_FLAG_COLUMNS = [
    "role_injury_out_flag",
    "role_injury_doubtful_flag",
    "role_injury_questionable_flag",
]

CALIBRATION_HEADERS = [
    "target",
    "projection_decile",
    "sample_size",
    "projection_min",
    "projection_max",
    "mean_projection",
    "mean_actual",
    "bias",
    "mean_probability_1plus",
    "actual_rate_1plus",
    "brier_1plus",
    "logloss_1plus",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build historical model performance reports.")
    p.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Reporting split. Default is the untouched test split; validation is diagnostic only.",
    )
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    root = common.prop_root().resolve()
    dest = path.resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Issue 39 write outside Prop Engine: {dest}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=f".{dest.name}.", suffix=".tmp",
        dir=dest.parent, delete=False,
    )
    tmp = Path(h.name)
    try:
        with h:
            frame.to_csv(h, index=False, lineterminator="\n")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    tmp = Path(h.name)
    try:
        with h:
            json.dump(payload, h, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            h.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_market_preflight() -> dict[str, Any]:
    script = common.prop_root() / "scripts" / "validate" / "audit_market_exclusion.py"
    if not script.is_file():
        raise FileNotFoundError(f"Market audit script missing: {script}")
    cp = subprocess.run(
        [sys.executable, str(script)], cwd=common.repo_root(), capture_output=True, text=True, check=False
    )
    if cp.returncode != 0 or "MARKET EXCLUSION AUDIT: PASS" not in cp.stdout:
        raise RuntimeError(
            "Market-exclusion preflight failed before model reporting. "
            f"stdout={cp.stdout[-2000:]!r} stderr={cp.stderr[-2000:]!r}"
        )
    return {"passed": True, "validator": str(script.relative_to(common.repo_root()))}


def target_is_count(config: dict[str, Any], target: str) -> bool:
    # Match Issue 25 exactly. derived_count (kicking_points) is not evaluated as Poisson.
    return str(config["targets"][target].get("type", "")) == "count_nonnegative"


def metric_values(frame: pd.DataFrame, *, count_target: bool) -> dict[str, Any]:
    y = numeric(frame["actual"]).to_numpy(dtype="float64")
    p = numeric(frame["projection"]).to_numpy(dtype="float64")
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    n = len(y)
    if n == 0:
        raise ValueError("Metric slice has zero finite rows")

    err = p - y
    abs_err = np.abs(err)
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = None if ss_tot <= 0.0 else float(1.0 - np.sum(np.square(err)) / ss_tot)

    poisson = brier = logloss = None
    if count_target:
        if np.any(y < 0.0):
            raise ValueError("Negative actual found in count target")
        lam = np.maximum(p, 1e-12)
        terms = np.empty_like(y)
        zero = y <= 0.0
        terms[zero] = lam[zero]
        nz = ~zero
        terms[nz] = y[nz] * np.log(y[nz] / lam[nz]) - (y[nz] - lam[nz])
        poisson = float(2.0 * np.mean(terms))
        prob = np.clip(1.0 - np.exp(-np.maximum(p, 0.0)), 1e-12, 1.0 - 1e-12)
        event = (y >= 1.0).astype("float64")
        brier = float(np.mean(np.square(prob - event)))
        logloss = float(-np.mean(event * np.log(prob) + (1.0 - event) * np.log(1.0 - prob)))

    return {
        "sample_size": int(n),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "median_absolute_error": float(np.median(abs_err)),
        "r2": r2,
        "poisson_deviance": poisson,
        "brier_1plus": brier,
        "logloss_1plus": logloss,
        "mean_actual": float(np.mean(y)),
        "mean_projection": float(np.mean(p)),
        "bias": float(np.mean(p) - np.mean(y)),
    }


def metric_row(target: str, frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return {"target": target, **metric_values(frame, count_target=target_is_count(config, target))}


def rank_bucket(series: pd.Series, buckets: int, prefix: str) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    values = numeric(series)
    finite = values.notna()
    if finite.any():
        ranked = values.loc[finite].rank(method="first", pct=True)
        labels = np.minimum(np.ceil(ranked.to_numpy(dtype=float) * buckets).astype(int), buckets)
        labels = np.maximum(labels, 1)
        out.loc[finite] = [f"{prefix}{x}" for x in labels]
    return out


def coalesce_usage(frame: pd.DataFrame, target: str, available: set[str]) -> tuple[pd.Series, str]:
    if target == "kicking_points" and all(c in available for c in KICKING_USAGE_COMPONENTS):
        a = numeric(frame[KICKING_USAGE_COMPONENTS[0]])
        b = numeric(frame[KICKING_USAGE_COMPONENTS[1]])
        value = a + b
        value = value.where(a.notna() | b.notna())
        return value, "+".join(KICKING_USAGE_COMPONENTS)

    candidates = [c for c in USAGE_CANDIDATES[target] if c in available]
    if not candidates:
        raise ValueError(f"No canonical pregame usage feature available for {target}")
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in candidates:
        candidate = numeric(frame[column])
        out = out.where(out.notna(), candidate)
    return out, "|".join(candidates)


def normalize_injury(frame: pd.DataFrame, columns: set[str]) -> tuple[pd.Series, str]:
    for column in INJURY_STATUS_CANDIDATES:
        if column in columns:
            value = frame[column].fillna("").astype(str).str.strip().str.casefold()
            value = value.replace({"": "none", "nan": "none", "<na>": "none"})
            return value.astype("string"), column

    if all(c in columns for c in INJURY_FLAG_COLUMNS):
        out = pd.Series("none", index=frame.index, dtype="string")
        out.loc[numeric(frame["role_injury_questionable_flag"]).fillna(0).gt(0)] = "questionable"
        out.loc[numeric(frame["role_injury_doubtful_flag"]).fillna(0).gt(0)] = "doubtful"
        out.loc[numeric(frame["role_injury_out_flag"]).fillna(0).gt(0)] = "out"
        return out, "derived_injury_flags"
    raise ValueError("Historical feature table has no usable pregame injury status context")


def selected_rows(config: dict[str, Any], audit: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, str]]:
    required = [
        "split", *GRAIN, "target", "actual", "baseline_projection", "direct_projection",
        "component_projection", "blend_projection",
    ]
    common.require_columns(audit, required, "Issue 25 model selection audit")
    common.ensure_unique(audit, [*GRAIN, "target", "split"], "Issue 25 model selection audit")

    chunks: list[pd.DataFrame] = []
    architectures: dict[str, str] = {}
    for target in config["targets"]:
        selected_path = common.prop_root() / "models" / target / "selected_model.json"
        selected = read_json(selected_path)
        architecture = clean(selected.get("selected_architecture") or selected.get("selected_candidate"))
        if architecture not in SELECTED_PROJECTION_COLUMNS:
            raise ValueError(f"{target}: invalid selected architecture {architecture!r}")
        if bool(selected.get("test_used_for_selection", False)):
            raise ValueError(f"{target}: selected model says test was used for selection")
        architectures[target] = architecture
        source_col = SELECTED_PROJECTION_COLUMNS[architecture]
        sub = audit.loc[
            audit["split"].astype(str).eq(split) & audit["target"].astype(str).eq(target),
            [*GRAIN, "target", "actual", source_col],
        ].copy()
        if sub.empty:
            raise ValueError(f"{target}: no rows for reporting split {split}")
        sub = sub.rename(columns={source_col: "projection"})
        sub["actual"] = numeric(sub["actual"])
        sub["projection"] = numeric(sub["projection"])
        sub = sub.loc[sub["actual"].notna() & sub["projection"].notna()].copy()
        if sub.empty:
            raise ValueError(f"{target}: no finite selected predictions on {split}")
        chunks.append(sub)
    out = pd.concat(chunks, ignore_index=True)
    common.ensure_unique(out, [*GRAIN, "target"], "Issue 39 selected reporting predictions")
    return out, architectures


def load_context(config: dict[str, Any], selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = common.repo_root() / str(config["paths"]["historical_features"])
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.with_name("feature_manifest.json")
    manifest = read_json(manifest_path)
    schema = set(manifest.get("leading_columns", []))
    families = manifest.get("column_families", {})
    if isinstance(families, dict):
        for values in families.values():
            if isinstance(values, list):
                schema.update(str(v) for v in values)
    if not schema:
        raise ValueError(f"Historical feature manifest has no declared schema: {manifest_path}")
    missing = [c for c in BASE_CONTEXT_REQUIRED if c not in schema]
    if missing:
        raise ValueError(f"Historical feature table missing reporting context: {missing}")

    optional = set(INJURY_STATUS_CANDIDATES + INJURY_FLAG_COLUMNS + KICKING_USAGE_COMPONENTS)
    for candidates in USAGE_CANDIDATES.values():
        optional.update(candidates)
    columns = list(dict.fromkeys(BASE_CONTEXT_REQUIRED + sorted(optional & schema)))
    context = pd.read_parquet(path, columns=columns)
    common.ensure_unique(context, GRAIN, "Issue 39 historical reporting context")

    # Restrict before merge to the reporting seasons/keys to keep the join light.
    seasons = set(pd.to_numeric(selected["season"], errors="raise").astype(int).unique().tolist())
    context = context.loc[pd.to_numeric(context["season"], errors="coerce").isin(seasons)].copy()
    enriched = selected.merge(context, on=GRAIN, how="left", validate="many_to_one")
    if enriched["position"].isna().any():
        raise ValueError("Selected reporting predictions failed historical context join")

    columns_set = set(enriched.columns)
    injury, injury_source = normalize_injury(enriched, columns_set)
    enriched["injury_status"] = injury
    enriched["starter_backup"] = np.where(
        numeric(enriched["role_depth_starter_flag_pregame"]).fillna(0.0).gt(0.0),
        "starter", "backup",
    )
    enriched["history_status"] = np.where(
        numeric(enriched["history_no_nfl_history_flag"]).fillna(0.0).gt(0.0),
        "no_history", "has_history",
    )
    enriched["team_status"] = np.where(
        numeric(enriched["history_new_team_flag"]).fillna(0.0).gt(0.0),
        "new_team", "returning_team",
    )
    week_num = pd.to_numeric(enriched["week"], errors="raise").astype(int)
    enriched["week_phase"] = np.where(week_num.le(4), "week_1_4", "week_5_plus")
    enriched["week_exact"] = [f"week_{int(x):02d}" for x in week_num]

    usage_sources: dict[str, str] = {}
    enriched["usage_score"] = np.nan
    for target in config["targets"]:
        mask = enriched["target"].eq(target)
        usage, source = coalesce_usage(enriched.loc[mask], target, columns_set)
        enriched.loc[mask, "usage_score"] = usage.to_numpy()
        usage_sources[target] = source

    enriched["usage_quartile"] = pd.NA
    enriched["projection_decile"] = pd.NA
    for target, idx in enriched.groupby("target", sort=False).groups.items():
        idx = list(idx)
        uq = rank_bucket(enriched.loc[idx, "usage_score"], 4, "Q")
        uq = uq.fillna("unknown")
        pdq = rank_bucket(enriched.loc[idx, "projection"], 10, "D")
        if pdq.isna().any():
            raise ValueError(f"{target}: projection decile could not be assigned")
        enriched.loc[idx, "usage_quartile"] = uq.to_numpy()
        enriched.loc[idx, "projection_decile"] = pdq.to_numpy()

    enriched["usage_quartile"] = enriched["usage_quartile"].astype("string")
    enriched["projection_decile"] = enriched["projection_decile"].astype("string")
    return enriched, {"injury_source": injury_source, "usage_sources": usage_sources, "context_columns": len(columns)}


def grouped_metrics(
    enriched: pd.DataFrame,
    config: dict[str, Any],
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in enriched.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(group_columns, key))
        target = str(values["target"])
        rows.append({**values, **metric_values(group, count_target=target_is_count(config, target))})
    return pd.DataFrame(rows)


def calibration_rows(enriched: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, decile), group in enriched.groupby(["target", "projection_decile"], sort=True):
        y = numeric(group["actual"]).to_numpy(dtype="float64")
        p = numeric(group["projection"]).to_numpy(dtype="float64")
        if len(y) == 0:
            continue
        mean_prob = actual_rate = brier = logloss = None
        if target_is_count(config, str(target)):
            prob = np.clip(1.0 - np.exp(-np.maximum(p, 0.0)), 1e-12, 1.0 - 1e-12)
            event = (y >= 1.0).astype("float64")
            mean_prob = float(np.mean(prob))
            actual_rate = float(np.mean(event))
            brier = float(np.mean(np.square(prob - event)))
            logloss = float(-np.mean(event * np.log(prob) + (1.0 - event) * np.log(1.0 - prob)))
        rows.append({
            "target": str(target),
            "projection_decile": str(decile),
            "sample_size": int(len(group)),
            "projection_min": float(np.min(p)),
            "projection_max": float(np.max(p)),
            "mean_projection": float(np.mean(p)),
            "mean_actual": float(np.mean(y)),
            "bias": float(np.mean(p) - np.mean(y)),
            "mean_probability_1plus": mean_prob,
            "actual_rate_1plus": actual_rate,
            "brier_1plus": brier,
            "logloss_1plus": logloss,
        })
    return pd.DataFrame(rows, columns=CALIBRATION_HEADERS)


def main() -> int:
    args = parse_args()
    config = common.load_config()
    prop = common.prop_root()
    evaluation = prop / "evaluation"

    market = run_market_preflight()
    audit_path = common.repo_root() / AUDIT_REL
    audit = pd.read_parquet(audit_path)
    selected, architectures = selected_rows(config, audit, args.split)
    # _CONFIG_ENFORCED_REPORTING_SPLIT
    training = config["training"]
    split_season = {
        "validation": int(
            training["development_validation_season"]
        ),
        "test": int(
            training["untouched_test_season"]
        ),
    }.get(str(args.split))
    if split_season is None:
        raise ValueError(
            f"Unsupported reporting split: {args.split!r}"
        )
    observed_seasons = set(
        pd.to_numeric(
            selected["season"],
            errors="raise",
        ).astype(int)
    )
    if observed_seasons != {split_season}:
        raise ValueError(
            f"{args.split} report rows must use configured "
            f"season {split_season}; "
            f"observed={sorted(observed_seasons)}"
        )
    enriched, context_meta = load_context(config, selected)

    # Required reports.
    target_rows = [metric_row(target, group, config) for target, group in enriched.groupby("target", sort=True)]
    by_target = pd.DataFrame(target_rows, columns=METRIC_HEADERS)

    by_position = grouped_metrics(enriched, config, ["target", "position"])
    by_position = by_position[["target", "position", *METRIC_HEADERS[1:]]]

    usage_q = grouped_metrics(enriched, config, ["target", "usage_quartile"])
    usage_q.insert(1, "slice_type", "usage_quartile")
    usage_q = usage_q.rename(columns={"usage_quartile": "slice_value"})
    proj_d = grouped_metrics(enriched, config, ["target", "projection_decile"])
    proj_d.insert(1, "slice_type", "projection_decile")
    proj_d = proj_d.rename(columns={"projection_decile": "slice_value"})
    by_usage = pd.concat([usage_q, proj_d], ignore_index=True)
    by_usage = by_usage[["target", "slice_type", "slice_value", *METRIC_HEADERS[1:]]]

    week_exact = grouped_metrics(enriched, config, ["target", "week_exact"])
    week_exact.insert(1, "slice_type", "week")
    week_exact = week_exact.rename(columns={"week_exact": "slice_value"})
    week_phase = grouped_metrics(enriched, config, ["target", "week_phase"])
    week_phase.insert(1, "slice_type", "week_phase")
    week_phase = week_phase.rename(columns={"week_phase": "slice_value"})
    by_week = pd.concat([week_exact, week_phase], ignore_index=True)
    by_week = by_week[["target", "slice_type", "slice_value", *METRIC_HEADERS[1:]]]

    role_frames = []
    for column, slice_type in [
        ("starter_backup", "starter_backup"),
        ("history_status", "rookie_no_history"),
        ("team_status", "new_team"),
        ("injury_status", "injury_status"),
    ]:
        piece = grouped_metrics(enriched, config, ["target", column])
        piece.insert(1, "slice_type", slice_type)
        piece = piece.rename(columns={column: "slice_value"})
        role_frames.append(piece)
    by_role = pd.concat(role_frames, ignore_index=True)
    by_role = by_role[["target", "slice_type", "slice_value", *METRIC_HEADERS[1:]]]

    calibration = calibration_rows(enriched, config)

    frames = {
        "target": by_target,
        "position": by_position,
        "usage": by_usage,
        "week": by_week,
        "role": by_role,
        "calibration": calibration,
    }
    for key, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Issue 39 generated empty report: {key}")
        write_csv_atomic(frame, evaluation / OUTPUTS[key])

    payload = {
        "status": "passed",
        "script": Path(__file__).name,
        "reporting_split": args.split,
        "reporting_policy": "untouched test split by default; validation split diagnostic only",
        "rows_evaluated": int(len(enriched)),
        "targets": int(enriched["target"].nunique()),
        "test_seasons": sorted(pd.to_numeric(enriched["season"], errors="raise").astype(int).unique().tolist()),
        "selected_architectures": architectures,
        "usage_sources": context_meta["usage_sources"],
        "injury_status_source": context_meta["injury_source"],
        "context_columns_loaded": int(context_meta["context_columns"]),
        "outputs": {key: str((evaluation / name).relative_to(common.repo_root())) for key, name in OUTPUTS.items()},
        "output_rows": {key: int(len(frame)) for key, frame in frames.items()},
        "metric_headers": METRIC_HEADERS,
        "market_exclusion_passed": bool(market["passed"]),
        "market_features_used": False,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    log_path = common.repo_root() / REPORT_LOG_REL
    write_json_atomic(payload, log_path)
    try:
        common.log_run(Path(__file__).name, payload)
    except Exception:
        pass

    print(json.dumps({"script": Path(__file__).name, "payload": payload}, sort_keys=True, default=str))
    print("MODEL PERFORMANCE REPORTS BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
