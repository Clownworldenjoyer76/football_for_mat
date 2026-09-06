#!/usr/bin/env python3
"""
Independent acceptance validator for Prop Engine Issue 21.

Validates:
- exact output headers
- unique OOF grain
- all 13 chronological folds and all 9 targets
- exact validation/test-window membership
- exact position eligibility
- exact realized targets
- independent reconstruction of all nine baseline formulas
- fold-training-only shrinkage priors
- no 2025 development/tuning contamination
- no infinities / invalid negative count projections
- deterministic sorted output contract
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(r"C:\Users\Mat\Documents\GitHub\football_for_mat")
PROP = ROOT / "docs/win/football/nfl/prop_engine"

FEATURE_PATH = PROP / "data/historical/features/player_game_features.parquet"
FOLD_PATH = PROP / "evaluation/backtest_folds.parquet"
OUTPUT_PATH = PROP / "evaluation/baseline_oof_predictions.parquet"
ELIGIBILITY_PATH = PROP / "config/target_eligibility.yaml"
TRAINER_PATH = PROP / "scripts/train/train_baselines.py"

OUTPUT_COLUMNS = [
    "fold_id",
    "season",
    "week",
    "game_id",
    "player_id",
    "target",
    "actual",
    "baseline_projection",
]

TARGETS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
]

TARGET_COL = {t: f"target_{t}" for t in TARGETS}

FEATURE_COLUMNS = [
    "season", "week", "game_id", "kickoff_timestamp", "player_id", "position",

    "player_pass_attempts_lag1",
    "player_pass_attempts_roll3_mean",
    "player_pass_attempts_roll5_mean",
    "player_pass_attempts_season_to_date",
    "player_pass_attempts_career_prior",
    "player_yards_per_attempt_lag1",
    "player_yards_per_attempt_roll3_mean",
    "player_yards_per_attempt_roll5_mean",
    "player_yards_per_attempt_season_to_date",
    "player_yards_per_attempt_career_prior",
    "player_passing_td_rate_lag1",
    "player_passing_td_rate_roll3_mean",
    "player_passing_td_rate_roll5_mean",
    "player_passing_td_rate_season_to_date",
    "player_passing_td_rate_career_prior",

    "player_carries_lag1",
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carries_season_to_date",
    "player_carries_career_prior",
    "player_yards_per_carry_lag1",
    "player_yards_per_carry_roll3_mean",
    "player_yards_per_carry_roll5_mean",
    "player_yards_per_carry_season_to_date",
    "player_yards_per_carry_career_prior",
    "player_goal_line_carries_lag1",
    "player_goal_line_carries_roll3_mean",
    "player_goal_line_carries_roll5_mean",
    "player_goal_line_carries_season_to_date",
    "player_goal_line_carries_career_prior",
    "player_rushing_tds_roll5_mean",

    "team_pass_attempts_lag1",
    "team_pass_attempts_roll3_mean",
    "team_pass_attempts_roll5_mean",
    "team_pass_attempts_season_to_date",
    "player_target_share_lag1",
    "player_target_share_roll3_mean",
    "player_target_share_roll5_mean",
    "player_target_share_season_to_date",
    "player_target_share_career_prior",
    "player_yards_per_target_lag1",
    "player_yards_per_target_roll3_mean",
    "player_yards_per_target_roll5_mean",
    "player_yards_per_target_season_to_date",
    "player_yards_per_target_career_prior",
    "player_red_zone_targets_lag1",
    "player_red_zone_targets_roll3_mean",
    "player_red_zone_targets_roll5_mean",
    "player_red_zone_targets_season_to_date",
    "player_red_zone_targets_career_prior",
    "player_receiving_tds_roll5_mean",

    "player_kicking_fg_attempts_lag1",
    "player_kicking_fg_attempts_roll3",
    "player_kicking_fg_attempts_roll5",
    "player_kicking_fg_make_pct_career_prior",
    "player_kicking_fg_make_pct_season_prior",
    "player_kicking_pat_attempts_roll3",
    "player_kicking_pat_make_pct_career_prior",

    "matchup_expected_opponent_plays",
    "matchup_expected_opponent_dropbacks",
    "player_defensive_def_snap_pct_lag1",
    "player_defensive_def_snap_pct_roll3",
    "player_defensive_def_participation_lag1",
    "player_defensive_def_participation_roll3",
    "player_defensive_tackles_roll3",
    "player_defensive_sacks_roll5",
    "player_defensive_opponent_plays_roll3",
    "player_defensive_opponent_dropbacks_roll3",

    *[TARGET_COL[t] for t in TARGETS],
]
FEATURE_COLUMNS = list(dict.fromkeys(FEATURE_COLUMNS))


def num(s):
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def coalesce(df, names):
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for name in names:
        candidate = num(df[name])
        result = result.where(result.notna(), candidate)
    return result


def nonneg(s):
    return num(s).clip(lower=0.0)


def prob(s):
    return num(s).clip(lower=0.0, upper=1.0)


def divide(a, b):
    a = num(a)
    b = num(b)
    result = pd.Series(np.nan, index=a.index, dtype="float64")
    valid = a.notna() & b.notna() & b.gt(0.0)
    result.loc[valid] = a.loc[valid] / b.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)


def window_mask(df, ss, sw, es, ew):
    season = df["season"].astype(int)
    week = df["week"].astype(int)
    return (
        ((season > ss) | ((season == ss) & (week >= sw)))
        & ((season < es) | ((season == es) & (week <= ew)))
    )


def pos_mask(df, positions):
    p = df["position"].fillna("").astype(str).str.strip().str.upper()
    return p.isin({str(x).strip().upper() for x in positions})


def prior(raw, exposure):
    r = prob(raw)
    e = nonneg(exposure)
    valid = r.notna() & e.notna() & e.gt(0.0)
    if not valid.any():
        return (float("nan"), float("nan"))
    ev = e.loc[valid]
    rv = r.loc[valid]
    total = float(ev.sum())
    if not math.isfinite(total) or total <= 0:
        return (float("nan"), float("nan"))
    pr = float((rv * ev).sum() / total)
    pe = float(ev.median())
    if not math.isfinite(pr):
        pr = float("nan")
    if not math.isfinite(pe) or pe <= 0:
        pe = float("nan")
    return pr, pe


def shrink(raw, exposure, pr, pe):
    r = prob(raw)
    e = nonneg(exposure)
    out = pd.Series(np.nan, index=r.index, dtype="float64")
    player_ok = r.notna() & e.notna() & e.gt(0.0)
    prior_ok = math.isfinite(pr) and math.isfinite(pe) and pe > 0.0
    if prior_ok:
        out.loc[player_ok] = (
            r.loc[player_ok] * e.loc[player_ok] + pr * pe
        ) / (e.loc[player_ok] + pe)
        out.loc[~player_ok] = pr
    else:
        out.loc[player_ok] = r.loc[player_ok]
    return prob(out)


print("CHECK 01: files and exact headers")
assert TRAINER_PATH.is_file(), f"Missing trainer: {TRAINER_PATH}"
assert OUTPUT_PATH.is_file(), f"Missing output: {OUTPUT_PATH}"

oof = pd.read_parquet(OUTPUT_PATH)
assert list(oof.columns) == OUTPUT_COLUMNS, "Output header/order mismatch"

print("CHECK 02: canonical OOF uniqueness and target/fold coverage")
assert not oof.duplicated(
    ["fold_id", "season", "week", "game_id", "player_id", "target"]
).any()
assert not oof.duplicated(
    ["season", "week", "game_id", "player_id", "target"]
).any()
assert set(oof["target"].unique()) == set(TARGETS)

folds = pd.read_parquet(FOLD_PATH)
assert set(oof["fold_id"].astype(str)) == set(folds["fold_id"].astype(str))
assert oof["fold_id"].nunique() == 13
assert len(folds[folds["test_flag"] == 1]) == 1

print("CHECK 03: load canonical pregame sources and eligibility")
with ELIGIBILITY_PATH.open("r", encoding="utf-8-sig") as f:
    eligibility = yaml.safe_load(f)

assert list(eligibility.keys()) == TARGETS

features = pd.read_parquet(FEATURE_PATH, columns=FEATURE_COLUMNS)
features["season"] = features["season"].astype(int)
features["week"] = features["week"].astype(int)
features["kickoff_timestamp"] = pd.to_datetime(
    features["kickoff_timestamp"], errors="raise", utc=True
)

assert not features.duplicated(
    ["season", "week", "game_id", "player_id"]
).any()

print("CHECK 04: reconstruct deterministic pregame baseline components")
c = {}

c["pa"] = nonneg(coalesce(features, [
    "player_pass_attempts_roll3_mean",
    "player_pass_attempts_roll5_mean",
    "player_pass_attempts_season_to_date",
    "player_pass_attempts_career_prior",
    "player_pass_attempts_lag1",
]))
c["ypa"] = coalesce(features, [
    "player_yards_per_attempt_roll5_mean",
    "player_yards_per_attempt_roll3_mean",
    "player_yards_per_attempt_season_to_date",
    "player_yards_per_attempt_career_prior",
    "player_yards_per_attempt_lag1",
])
c["pass_td_rate"] = prob(coalesce(features, [
    "player_passing_td_rate_roll5_mean",
    "player_passing_td_rate_roll3_mean",
    "player_passing_td_rate_season_to_date",
    "player_passing_td_rate_career_prior",
    "player_passing_td_rate_lag1",
]))

c["carries"] = nonneg(coalesce(features, [
    "player_carries_roll3_mean",
    "player_carries_roll5_mean",
    "player_carries_season_to_date",
    "player_carries_career_prior",
    "player_carries_lag1",
]))
c["ypc"] = coalesce(features, [
    "player_yards_per_carry_roll5_mean",
    "player_yards_per_carry_roll3_mean",
    "player_yards_per_carry_season_to_date",
    "player_yards_per_carry_career_prior",
    "player_yards_per_carry_lag1",
])
c["gl_carries"] = nonneg(coalesce(features, [
    "player_goal_line_carries_roll3_mean",
    "player_goal_line_carries_roll5_mean",
    "player_goal_line_carries_season_to_date",
    "player_goal_line_carries_career_prior",
    "player_goal_line_carries_lag1",
]))
c["gl_exp"] = nonneg(features["player_goal_line_carries_roll5_mean"]) * 5.0
c["gl_success"] = nonneg(features["player_rushing_tds_roll5_mean"]) * 5.0
c["gl_rate"] = prob(divide(c["gl_success"], c["gl_exp"]))

c["team_pa"] = nonneg(coalesce(features, [
    "team_pass_attempts_roll3_mean",
    "team_pass_attempts_roll5_mean",
    "team_pass_attempts_season_to_date",
    "team_pass_attempts_lag1",
]))
c["target_share"] = prob(coalesce(features, [
    "player_target_share_roll3_mean",
    "player_target_share_roll5_mean",
    "player_target_share_season_to_date",
    "player_target_share_career_prior",
    "player_target_share_lag1",
]))
c["ypt"] = coalesce(features, [
    "player_yards_per_target_roll5_mean",
    "player_yards_per_target_roll3_mean",
    "player_yards_per_target_season_to_date",
    "player_yards_per_target_career_prior",
    "player_yards_per_target_lag1",
])
c["rz_targets"] = nonneg(coalesce(features, [
    "player_red_zone_targets_roll3_mean",
    "player_red_zone_targets_roll5_mean",
    "player_red_zone_targets_season_to_date",
    "player_red_zone_targets_career_prior",
    "player_red_zone_targets_lag1",
]))
c["rec_exp"] = nonneg(features["player_red_zone_targets_roll5_mean"]) * 5.0
c["rec_success"] = nonneg(features["player_receiving_tds_roll5_mean"]) * 5.0
c["rec_rate"] = prob(divide(c["rec_success"], c["rec_exp"]))

c["fg_att"] = nonneg(coalesce(features, [
    "player_kicking_fg_attempts_roll3",
    "player_kicking_fg_attempts_roll5",
    "player_kicking_fg_attempts_lag1",
]))
c["fg_pct"] = prob(coalesce(features, [
    "player_kicking_fg_make_pct_season_prior",
    "player_kicking_fg_make_pct_career_prior",
]))
c["pat_att"] = nonneg(features["player_kicking_pat_attempts_roll3"])
c["pat_pct"] = prob(features["player_kicking_pat_make_pct_career_prior"])

c["def_part"] = prob(coalesce(features, [
    "player_defensive_def_participation_roll3",
    "player_defensive_def_participation_lag1",
    "player_defensive_def_snap_pct_roll3",
    "player_defensive_def_snap_pct_lag1",
]))
c["opp_plays"] = nonneg(coalesce(features, [
    "matchup_expected_opponent_plays",
    "player_defensive_opponent_plays_roll3",
]))
c["opp_db"] = nonneg(coalesce(features, [
    "matchup_expected_opponent_dropbacks",
    "player_defensive_opponent_dropbacks_roll3",
]))
c["tackle_exp"] = (
    nonneg(features["player_defensive_opponent_plays_roll3"])
    * c["def_part"]
)
c["tackle_rate"] = prob(divide(
    nonneg(features["player_defensive_tackles_roll3"]),
    c["tackle_exp"],
))
c["sack_exp"] = (
    nonneg(features["player_defensive_opponent_dropbacks_roll3"])
    * c["def_part"]
    * 5.0
)
c["sack_success"] = nonneg(features["player_defensive_sacks_roll5"]) * 5.0
c["sack_rate"] = prob(divide(c["sack_success"], c["sack_exp"]))

print("CHECK 05: exact fold membership, target reconciliation, and formulas")
all_expected = []
max_abs_error = {t: 0.0 for t in TARGETS}
nan_pattern_mismatches = {t: 0 for t in TARGETS}

for fold in folds.itertuples(index=False):
    train = window_mask(
        features,
        int(fold.train_start_season),
        int(fold.train_start_week),
        int(fold.train_end_season),
        int(fold.train_end_week),
    )
    val = window_mask(
        features,
        int(fold.validation_start_season),
        int(fold.validation_start_week),
        int(fold.validation_end_season),
        int(fold.validation_end_week),
    )

    assert features.loc[train, "kickoff_timestamp"].max() < (
        features.loc[val, "kickoff_timestamp"].min()
    ), f"Kickoff leakage in {fold.fold_id}"

    gl_prior = prior(c["gl_rate"].loc[train], c["gl_exp"].loc[train])
    rec_prior = prior(c["rec_rate"].loc[train], c["rec_exp"].loc[train])
    sack_prior = prior(c["sack_rate"].loc[train], c["sack_exp"].loc[train])

    gl_shrunk = shrink(c["gl_rate"], c["gl_exp"], *gl_prior)
    rec_shrunk = shrink(c["rec_rate"], c["rec_exp"], *rec_prior)
    sack_shrunk = shrink(c["sack_rate"], c["sack_exp"], *sack_prior)

    projections = {
        "passing_yards": num(c["pa"] * c["ypa"]),
        "passing_tds": nonneg(c["pa"] * c["pass_td_rate"]),
        "rushing_yards": num(c["carries"] * c["ypc"]),
        "rushing_tds": nonneg(c["gl_carries"] * gl_shrunk),
        "receiving_yards": num(c["team_pa"] * c["target_share"] * c["ypt"]),
        "receiving_tds": nonneg(c["rz_targets"] * rec_shrunk),
        "kicking_points": nonneg(3.0 * c["fg_att"] * c["fg_pct"] + c["pat_att"] * c["pat_pct"]),
        "tackles": nonneg(c["opp_plays"] * c["def_part"] * c["tackle_rate"]),
        "sacks": nonneg(c["opp_db"] * c["def_part"] * sack_shrunk),
    }

    for target in TARGETS:
        actual = num(features[TARGET_COL[target]])
        expected_mask = (
            val
            & pos_mask(features, eligibility[target]["eligible_positions"])
            & actual.notna()
        )

        expected = features.loc[
            expected_mask,
            ["season", "week", "game_id", "player_id"],
        ].copy()
        expected.insert(0, "fold_id", str(fold.fold_id))
        expected["target"] = target
        expected["actual_expected"] = actual.loc[expected_mask].to_numpy()
        expected["projection_expected"] = projections[target].loc[expected_mask].to_numpy()
        all_expected.append(expected)

expected = pd.concat(all_expected, ignore_index=True)

keys = ["fold_id", "season", "week", "game_id", "player_id", "target"]

assert len(expected) == len(oof), (
    f"OOF row count mismatch: expected={len(expected)}, actual={len(oof)}"
)

check = expected.merge(
    oof,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert check["_merge"].eq("both").all(), (
    "Missing or extra OOF rows detected"
)

actual_ok = np.isclose(
    check["actual_expected"].to_numpy(dtype=float),
    check["actual"].to_numpy(dtype=float),
    rtol=0.0,
    atol=0.0,
    equal_nan=True,
)
assert actual_ok.all(), "Actual target reconciliation failed"

for target in TARGETS:
    sub = check[check["target"] == target]

    exp = sub["projection_expected"]
    got = sub["baseline_projection"]

    same_nan = exp.isna().to_numpy() == got.isna().to_numpy()
    nan_pattern_mismatches[target] = int((~same_nan).sum())
    assert same_nan.all(), (
        f"{target}: projection missingness does not match independent formula"
    )

    finite = exp.notna() & got.notna()
    if finite.any():
        errors = np.abs(
            exp.loc[finite].to_numpy(dtype=float)
            - got.loc[finite].to_numpy(dtype=float)
        )
        max_abs_error[target] = float(errors.max())
        assert np.allclose(
            exp.loc[finite].to_numpy(dtype=float),
            got.loc[finite].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ), f"{target}: formula reconstruction mismatch"

print("CHECK 06: 2025 isolation and no tuning leakage")
test_fold = folds[folds["test_flag"] == 1]
assert len(test_fold) == 1
test_id = str(test_fold.iloc[0]["fold_id"])
assert int(test_fold.iloc[0]["train_end_season"]) == 2024
assert int(test_fold.iloc[0]["validation_start_season"]) == 2025

rows_2025 = oof[oof["season"] == 2025]
assert not rows_2025.empty
assert set(rows_2025["fold_id"].astype(str)) == {test_id}

dev_rows = oof[oof["fold_id"].isin(
    folds.loc[folds["test_flag"] == 0, "fold_id"]
)]
assert not dev_rows["season"].eq(2025).any()

print("CHECK 07: finite/nonnegative constraints and missing-data semantics")
assert not np.isinf(num(oof["baseline_projection"]).dropna()).any()
assert oof["actual"].notna().all()

nonnegative_targets = {
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "kicking_points",
    "tackles",
    "sacks",
}
bad_negative = (
    oof["target"].isin(nonnegative_targets)
    & num(oof["baseline_projection"]).lt(0.0)
)
assert not bad_negative.any()

# Missing formula inputs are allowed to yield missing projections; they are
# not silently zero-filled.
assert oof["baseline_projection"].isna().any(), (
    "Expected source-unavailable baseline rows to remain missing"
)

print("CHECK 08: required policy markers in trainer source")
source = TRAINER_PATH.read_text(encoding="utf-8")
for text in [
    "random_split_used",
    "target_columns_used_in_projection",
    "test_tuning_used",
    "untouched_test_season",
]:
    assert text in source, f"Missing policy marker in trainer: {text}"

print("CHECK 09: output sort is deterministic")
sorted_copy = oof.sort_values(
    ["season", "week", "game_id", "player_id"],
    kind="mergesort",
    na_position="last",
).reset_index(drop=True)
assert oof.reset_index(drop=True).equals(sorted_copy), (
    "Output does not follow deterministic canonical sort"
)

print("CHECK 10: summarize independent formula error")
for target in TARGETS:
    print(
        f"{target}: max_abs_error={max_abs_error[target]:.3e}, "
        f"nan_pattern_mismatches={nan_pattern_mismatches[target]}"
    )

print(f"rows={len(oof)}")
print(f"folds={oof['fold_id'].nunique()}")
print(f"targets={oof['target'].nunique()}")
print("ISSUE 21 ACCEPTANCE: PASS")
