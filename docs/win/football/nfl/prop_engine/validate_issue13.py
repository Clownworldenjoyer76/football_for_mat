from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

NFL = Path("docs/win/football/nfl")

TEAM_SOURCE = NFL / "prop_engine/data/historical/opportunity/team_week_opportunity.parquet"
OPP_SOURCE = NFL / "prop_engine/data/historical/opportunity/opponent_week_opportunity.parquet"
TEAM_OUT = NFL / "prop_engine/data/historical/features/team_form.parquet"
OPP_OUT = NFL / "prop_engine/data/historical/features/opponent_form.parquet"

GRAIN = ["season", "week", "team"]

TEAM_METRICS = [
    "offensive_plays",
    "drives",
    "dropbacks",
    "pass_attempts",
    "rush_attempts",
    "pass_rate",
    "rush_rate",
    "points_per_drive",
    "red_zone_drives",
    "red_zone_pass_attempts",
    "red_zone_rush_attempts",
    "goal_line_rush_attempts",
    "field_goal_attempts",
    "extra_point_attempts",
    "off_epa_per_play",
    "off_success_rate",
    "yards_per_play",
    "red_zone_td_rate",
    "early_down_epa",
    "third_down_conversion_rate",
]

OPP_METRICS = [
    "defensive_plays",
    "opponent_dropbacks",
    "opponent_pass_attempts",
    "opponent_rush_attempts",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "passing_tds_allowed",
    "rushing_tds_allowed",
    "sacks",
    "qb_hits",
    "red_zone_pass_attempts_allowed",
    "red_zone_rush_attempts_allowed",
    "goal_line_rush_attempts_allowed",
    "def_epa_per_play",
    "def_success_rate",
    "yards_per_play_allowed",
    "points_per_drive_allowed",
    "red_zone_td_rate_allowed",
]

SUFFIXES = [
    "lag1",
    "roll3_mean",
    "roll5_mean",
    "roll8_mean",
    "ewm3",
    "ewm5",
    "season_to_date",
]

ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
    "JAC": "JAX",
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LAR",
}


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def team(value: Any) -> str:
    value = clean(value).upper()
    return ALIASES.get(value, value)


def columns(metrics: list[str]) -> list[str]:
    return GRAIN + [
        f"{metric}_{suffix}"
        for metric in metrics
        for suffix in SUFFIXES
    ]


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def ewm(previous: float | None, value: float, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def same(actual, expected, label: str, rtol=1e-10, atol=1e-10) -> None:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = np.asarray(expected, dtype="float64")
    ok = (
        (np.isnan(a) & np.isnan(e))
        |
        (
            np.isfinite(a)
            & np.isfinite(e)
            & np.isclose(a, e, rtol=rtol, atol=atol)
        )
    )
    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        sample = [
            {
                "row": int(i),
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in bad[:10]
        ]
        fail(f"{label}: {len(bad):,} mismatches; sample={sample}")


def normalize_source(df: pd.DataFrame, metrics: list[str], label: str) -> pd.DataFrame:
    missing = [c for c in GRAIN + metrics if c not in df.columns]
    if missing:
        fail(f"{label} source missing columns: {missing}")

    x = df[GRAIN + metrics].copy()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(team)

    for metric in metrics:
        x[metric] = (
            pd.to_numeric(x[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float64")
        )

    if x.duplicated(GRAIN).any():
        fail(f"{label} source has duplicate canonical grain")

    return (
        x.sort_values(
            ["season", "week", "team"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def validate_one(
    source_path: Path,
    output_path: Path,
    metrics: list[str],
    label: str,
) -> None:
    source = normalize_source(
        pd.read_parquet(source_path),
        metrics,
        label,
    )

    out = pd.read_parquet(output_path)
    expected_cols = columns(metrics)

    if list(out.columns) != expected_cols:
        missing = [c for c in expected_cols if c not in out.columns]
        extra = [c for c in out.columns if c not in expected_cols]
        fail(
            f"{label} exact header/order mismatch; "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    out = out.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    out["team"] = out["team"].map(team)

    out = (
        out.sort_values(
            ["season", "week", "team"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(out) != len(source):
        fail(f"{label} row count {len(out):,} != source {len(source):,}")

    if out.duplicated(GRAIN).any():
        fail(f"{label} output duplicate canonical grain")

    for c in GRAIN:
        if not out[c].astype(str).eq(source[c].astype(str)).all():
            fail(f"{label} output grain differs from source: {c}")

    print(
        f"PASS: {label} exact contract, row count, "
        f"grain, and source key set."
    )

    expected = {
        f"{metric}_{suffix}": np.full(len(source), np.nan, dtype="float64")
        for metric in metrics
        for suffix in SUFFIXES
    }

    by_team = source.groupby("team", sort=False).indices
    seasons = source["season"].to_numpy(dtype="int64")

    for metric_index, metric in enumerate(metrics, start=1):
        print(
            f"VALIDATE {label} metric "
            f"{metric_index:02d}/{len(metrics):02d}: {metric}"
        )

        values = source[metric].to_numpy(dtype="float64")

        for _, positions in by_team.items():
            history: list[float] = []
            season_history: list[float] = []
            current_season: int | None = None
            ewm3_state: float | None = None
            ewm5_state: float | None = None

            for pos in positions:
                season_value = int(seasons[pos])

                if current_season != season_value:
                    current_season = season_value
                    season_history = []

                if history:
                    expected[f"{metric}_lag1"][pos] = history[-1]
                    expected[f"{metric}_roll3_mean"][pos] = float(
                        np.mean(history[-3:])
                    )
                    expected[f"{metric}_roll5_mean"][pos] = float(
                        np.mean(history[-5:])
                    )
                    expected[f"{metric}_roll8_mean"][pos] = float(
                        np.mean(history[-8:])
                    )

                if ewm3_state is not None:
                    expected[f"{metric}_ewm3"][pos] = ewm3_state

                if ewm5_state is not None:
                    expected[f"{metric}_ewm5"][pos] = ewm5_state

                if season_history:
                    expected[f"{metric}_season_to_date"][pos] = float(
                        np.mean(season_history)
                    )

                current = number(values[pos])
                if current is None:
                    continue

                history.append(current)
                season_history.append(current)
                ewm3_state = ewm(ewm3_state, current, 3)
                ewm5_state = ewm(ewm5_state, current, 5)

        for suffix in SUFFIXES:
            name = f"{metric}_{suffix}"
            same(out[name], expected[name], f"{label}:{name}")

    print(
        f"PASS: All {len(metrics) * len(SUFFIXES)} "
        f"{label} form features exactly reconstruct "
        "from prior observed games."
    )

    # Explicit source-boundary checks.
    rich_metrics = (
        [
            "drives",
            "points_per_drive",
            "red_zone_drives",
            "red_zone_pass_attempts",
            "red_zone_rush_attempts",
            "goal_line_rush_attempts",
            "off_epa_per_play",
            "off_success_rate",
            "yards_per_play",
            "red_zone_td_rate",
            "early_down_epa",
            "third_down_conversion_rate",
        ]
        if label == "team"
        else [
            "red_zone_pass_attempts_allowed",
            "red_zone_rush_attempts_allowed",
            "goal_line_rush_attempts_allowed",
            "def_epa_per_play",
            "def_success_rate",
            "yards_per_play_allowed",
            "points_per_drive_allowed",
            "red_zone_td_rate_allowed",
        ]
    )

    pre_2021 = out["season"].lt(2021)

    for metric in rich_metrics:
        for suffix in SUFFIXES:
            c = f"{metric}_{suffix}"
            if out.loc[pre_2021, c].notna().any():
                fail(f"{label}:{c} populated before 2021 source availability")

    # Week 1 season-to-date must be null for every metric because there is
    # no prior observed game in the current season.
    week1 = out["week"].eq(1)

    for metric in metrics:
        c = f"{metric}_season_to_date"
        if out.loc[week1, c].notna().any():
            fail(f"{label}:{c} should be null in Week 1")

    # Cross-season continuity: for core metrics, Week 1 after 2012 should
    # have lag1 whenever the franchise had a prior observed value.
    core_probe = metrics[0]
    probe = out.loc[
        out["week"].eq(1) & out["season"].gt(2012),
        ["season", "team", f"{core_probe}_lag1"],
    ]
    if probe[f"{core_probe}_lag1"].isna().any():
        bad = probe.loc[probe[f"{core_probe}_lag1"].isna()].head(10)
        fail(
            f"{label}:{core_probe}_lag1 lost cross-season franchise history; "
            f"sample={bad.to_dict(orient='records')}"
        )

    feature_cols = [
        c for c in out.columns if c not in GRAIN
    ]

    matrix = (
        out[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64")
    )

    if np.isinf(matrix).any():
        fail(f"{label} output contains infinity")

    print(
        f"PASS: {label} source boundaries, Week-1 season reset, "
        "cross-season continuity, and finite-value checks."
    )


print("=" * 100)
print("ISSUE 13 ACCEPTANCE VALIDATION")
print("=" * 100)

validate_one(
    TEAM_SOURCE,
    TEAM_OUT,
    TEAM_METRICS,
    "team",
)

validate_one(
    OPP_SOURCE,
    OPP_OUT,
    OPP_METRICS,
    "opponent",
)

team_out = pd.read_parquet(TEAM_OUT)
opp_out = pd.read_parquet(OPP_OUT)

if len(team_out.columns) != 143:
    fail(f"team_form expected 143 total columns; got {len(team_out.columns)}")

if len(opp_out.columns) != 129:
    fail(f"opponent_form expected 129 total columns; got {len(opp_out.columns)}")

if len(team_out) != 7326 or len(opp_out) != 7326:
    fail(
        "expected 7,326 rows in each output; "
        f"team={len(team_out):,} opponent={len(opp_out):,}"
    )

print("=" * 100)
print("ISSUE 13 ACCEPTANCE: PASS")
print("=" * 100)
