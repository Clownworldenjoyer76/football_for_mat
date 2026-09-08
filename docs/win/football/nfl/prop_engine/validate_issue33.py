#!/usr/bin/env python3
"""Independent acceptance validator for expanded Issue 33 component projections."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Issue 33 validation requires LightGBM."
    ) from exc

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PROJECT = SCRIPTS / "project"
TRAIN = SCRIPTS / "train"
for path in (SCRIPTS, PROJECT, TRAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common
import train_opportunity_models as opportunity
import train_efficiency_models as efficiency

GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_GRAIN = ["season", "week", "game_id", "team"]

HEADERS = [
    "season", "week", "game_id", "player_id", "team", "opponent", "position",
    "projected_team_pass_attempts", "projected_qb_pass_attempts",
    "projected_team_rush_attempts", "projected_player_carries",
    "projected_target_share", "projected_targets",
    "projected_yards_per_attempt", "projected_yards_per_carry",
    "projected_yards_per_target", "projected_red_zone_targets",
    "projected_goal_line_carries", "projected_fg_attempts",
    "projected_fg_make_probability", "projected_pat_attempts",
    "projected_pat_make_probability",
    "projected_opponent_plays", "projected_opponent_dropbacks",
    "projected_defensive_participation",
    "projected_tackle_rate", "projected_sack_rate",
    "component_passing_yards", "component_passing_tds",
    "component_rushing_yards", "component_rushing_tds",
    "component_receiving_yards", "component_receiving_tds",
    "component_kicking_points", "component_tackles", "component_sacks",
]

OPP = [
    "qb_pass_attempts", "team_pass_attempts", "team_rush_attempts",
    "player_carry_share", "player_target_share",
    "player_red_zone_target_share", "player_goal_line_carry_share",
    "field_goal_attempts", "extra_point_attempts",
    "opponent_offensive_plays", "opponent_dropbacks",
    "player_defensive_participation",
]
EFF = [
    "passing_yards_per_attempt", "passing_td_rate",
    "rushing_yards_per_carry", "rushing_td_per_goal_line_carry",
    "receiving_yards_per_target", "receiving_td_per_red_zone_target",
    "field_goal_conversion", "extra_point_conversion",
    "tackle_rate_per_defensive_play", "sack_rate_per_defensive_play",
]
RZ = [
    "team_red_zone_pass_attempts_roll3_mean",
    "team_red_zone_pass_attempts_roll5_mean",
    "team_red_zone_pass_attempts_ewm5",
    "team_red_zone_pass_attempts_season_to_date",
]
GL = [
    "team_goal_line_rush_attempts_roll3_mean",
    "team_goal_line_rush_attempts_roll5_mean",
    "team_goal_line_rush_attempts_ewm5",
    "team_goal_line_rush_attempts_season_to_date",
]

TEAM_OPP_SOURCE = {
    "player_defensive_opponent_plays_roll3":
        "team_offensive_plays_roll3_mean",
    "player_defensive_opponent_dropbacks_roll3":
        "team_dropbacks_roll3_mean",
    "player_defensive_opponent_rush_rate_roll3":
        "team_rush_rate_roll3_mean",
    "player_defensive_opponent_pass_rate_roll3":
        "team_pass_rate_roll3_mean",
}
TEAM_DEF_RATE = "player_defensive_team_def_sack_rate_roll3"

TARGET_COMPONENTS = [
    "component_passing_yards",
    "component_passing_tds",
    "component_rushing_yards",
    "component_rushing_tds",
    "component_receiving_yards",
    "component_receiving_tds",
    "component_kicking_points",
    "component_tackles",
    "component_sacks",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    return parser.parse_args()


def num(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def assert_close(
    actual: pd.Series,
    expected: pd.Series,
    label: str,
    atol: float = 1e-8,
) -> None:
    a = num(actual)
    e = num(expected)
    if not a.isna().eq(e.isna()).all():
        raise AssertionError(f"{label}: null-mask mismatch")
    ok = a.isna() | np.isclose(
        a,
        e,
        rtol=1e-9,
        atol=atol,
        equal_nan=True,
    )
    if not ok.all():
        idx = ok.index[~ok][:10].tolist()
        raise AssertionError(
            f"{label}: numeric mismatch at rows {idx}"
        )


def coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        out = out.where(out.notna(), num(frame[column]))
    return out


def zero_safe_product(*series: pd.Series) -> pd.Series:
    values = [num(item) for item in series]
    result = values[0].copy()
    for item in values[1:]:
        result = result * item
    zero = pd.Series(False, index=result.index)
    for item in values:
        zero |= item.eq(0.0)
    result.loc[zero] = 0.0
    return result


def booster(
    root: Path,
    family: str,
    name: str,
) -> tuple[lgb.Booster, dict[str, Any], list[str]]:
    directory = root / "models" / family / name
    manifest = load_json(directory / "feature_manifest.json")
    model = lgb.Booster(model_file=str(directory / "model.txt"))
    names = (
        list(manifest.get("numeric_features", []))
        + list(manifest.get("categorical_features", []))
    )
    if list(model.feature_name()) != names:
        raise AssertionError(
            f"{family}/{name}: persisted feature order mismatch"
        )
    return model, manifest, names


def strict_team_def_rate(
    config: dict[str, Any],
    repo: Path,
    season: int,
) -> pd.DataFrame:
    path = repo / str(config["paths"]["opponent_opportunity"])
    raw = common.read_parquet_required(
        path,
        ["season", "week", "team", "sacks", "opponent_dropbacks"],
    ).copy()
    raw["season"] = pd.to_numeric(
        raw["season"],
        errors="raise",
    ).astype(int)
    raw["week"] = pd.to_numeric(
        raw["week"],
        errors="raise",
    ).astype(int)
    raw = raw.loc[raw["season"].lt(season)].copy()
    raw["_team_key"] = raw["team"].map(opportunity.canonical_team)
    sacks = num(raw["sacks"])
    drops = num(raw["opponent_dropbacks"])
    raw["_rate"] = np.where(
        sacks.notna() & drops.notna() & drops.ne(0.0),
        sacks / drops,
        np.nan,
    )
    raw = raw.sort_values(
        ["_team_key", "season", "week"],
        kind="mergesort",
    )
    records = []
    for team, frame in raw.groupby("_team_key", sort=False):
        values = frame["_rate"].dropna().to_numpy(dtype="float64")
        records.append(
            {
                "_team_key": team,
                TEAM_DEF_RATE: (
                    float(np.mean(values[-3:]))
                    if len(values)
                    else np.nan
                ),
            }
        )
    out = pd.DataFrame(records)
    common.ensure_unique(
        out,
        ["_team_key"],
        "Issue33 validator strict team defensive rate",
    )
    return out


def team_opp_rows(
    features: pd.DataFrame,
    names: list[str],
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    reconstructed = set(TEAM_OPP_SOURCE) | {TEAM_DEF_RATE}
    passthrough = [name for name in names if name not in reconstructed]
    common.require_columns(
        features,
        [
            *TEAM_GRAIN,
            "opponent",
            *passthrough,
            *TEAM_OPP_SOURCE.values(),
        ],
        "Issue33 validator team-opponent context",
    )
    opportunity.check_team_feature_invariance(features, passthrough)
    rows = opportunity.team_rows_from_features(features, passthrough)
    source_cols = list(TEAM_OPP_SOURCE.values())
    opportunity.check_team_feature_invariance(features, source_cols)
    ctx = opportunity.team_rows_from_features(
        features,
        source_cols,
    ).rename(
        columns={
            "team": "_context_team",
            **{
                source: target
                for target, source in TEAM_OPP_SOURCE.items()
            },
        }
    )
    rows = rows.merge(
        ctx[
            [
                "season",
                "week",
                "game_id",
                "_context_team",
                *TEAM_OPP_SOURCE.keys(),
            ]
        ],
        left_on=["season", "week", "game_id", "opponent"],
        right_on=["season", "week", "game_id", "_context_team"],
        how="left",
        validate="one_to_one",
    )
    rows["_team_key"] = rows["team"].map(opportunity.canonical_team)
    rows = rows.merge(
        team_def_rate,
        on="_team_key",
        how="left",
        validate="many_to_one",
    )
    common.ensure_unique(
        rows,
        TEAM_GRAIN,
        "Issue33 validator team-opponent rows",
    )
    return rows


def opp_rows(
    features: pd.DataFrame,
    name: str,
    eligibility: dict[str, Any],
    team_def_rate: pd.DataFrame,
) -> pd.DataFrame:
    spec = opportunity.COMPONENTS[name]
    feature_names = list(spec["features"])
    scope = str(spec["scope"])
    if scope == "team":
        opportunity.check_team_feature_invariance(
            features,
            feature_names,
        )
        return opportunity.team_rows_from_features(
            features,
            feature_names,
        )
    if scope == "team_opponent":
        return team_opp_rows(
            features,
            feature_names,
            team_def_rate,
        )
    if scope == "player":
        missing = [
            column
            for column in feature_names
            if column not in features.columns
        ]
        if missing:
            raise AssertionError(
                f"{name}: missing features {missing[:20]}"
            )
        rule = str(spec["eligible_rule"])
        positions = {
            str(value).strip().upper()
            for value in eligibility[rule]["eligible_positions"]
        }
        pos = (
            features["position"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        return features.loc[pos.isin(positions)].copy()
    raise AssertionError(
        f"{name}: unsupported opportunity scope {scope!r}"
    )


def score_opp(
    root: Path,
    features: pd.DataFrame,
    eligibility: dict[str, Any],
    config: dict[str, Any],
    repo: Path,
    season: int,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    team_def_rate = strict_team_def_rate(config, repo, season)
    for name in OPP:
        model, _manifest, names = booster(
            root,
            "components",
            name,
        )
        rows = opp_rows(
            features,
            name,
            eligibility,
            team_def_rate,
        )
        prediction = opportunity.transform_prediction(
            model.predict(
                opportunity.numeric_frame(rows, names)
            ),
            name,
        )
        if not np.isfinite(prediction).all():
            raise AssertionError(
                f"{name}: nonfinite opportunity prediction"
            )
        scope = str(opportunity.COMPONENTS[name]["scope"])
        key = GRAIN if scope == "player" else TEAM_GRAIN
        frame = rows[key].copy()
        frame[name] = prediction
        common.ensure_unique(
            frame,
            key,
            f"Issue33 validator {name}",
        )
        out[name] = frame
    return out


def eff_cols() -> list[str]:
    columns = [
        *GRAIN,
        "kickoff_timestamp",
        "position",
        "position_group",
    ]
    for name in EFF:
        for feature in efficiency.FEATURES[name]:
            if (
                feature not in efficiency.DERIVED_FEATURES
                and feature not in columns
            ):
                columns.append(feature)
    return columns


def eff_inference(
    current: pd.DataFrame,
    raw: pd.DataFrame,
    name: str,
    eligibility: dict[str, Any],
) -> pd.DataFrame:
    rule = efficiency.ELIGIBILITY_RULE[name]
    positions = {
        str(value).strip().upper()
        for value in eligibility[rule]["eligible_positions"]
    }
    pos = (
        current["position"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    target = current.loc[pos.isin(positions)].copy()
    required = [
        *GRAIN,
        "kickoff_timestamp",
        "position",
        "position_group",
        "_prior_position_group",
        "_numerator",
        "_exposure",
        "_label",
    ]
    history = raw[required].copy()
    history["_mark"] = 0

    placeholder = target[
        [
            *GRAIN,
            "kickoff_timestamp",
            "position",
            "position_group",
        ]
    ].copy()
    placeholder["_prior_position_group"] = (
        efficiency.normalize_position_group(
            placeholder["position"],
            placeholder["position_group"],
        )
    )
    placeholder["_numerator"] = np.nan
    placeholder["_exposure"] = np.nan
    placeholder["_label"] = np.nan
    placeholder["_mark"] = 1

    enriched = efficiency.add_strict_prior_features(
        pd.concat(
            [history, placeholder],
            ignore_index=True,
            sort=False,
        ),
        name,
    )
    enriched = enriched.loc[enriched["_mark"].eq(1)].copy()

    canonical = [
        feature
        for feature in efficiency.FEATURES[name]
        if feature not in efficiency.DERIVED_FEATURES
    ]
    return enriched.merge(
        target[[*GRAIN, *canonical]],
        on=GRAIN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_canonical"),
    )


def score_eff(
    root: Path,
    current: pd.DataFrame,
    history: pd.DataFrame,
    eligibility: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    label_base = efficiency.prepare_label_base(config, history)
    out: dict[str, pd.DataFrame] = {}
    for name in EFF:
        raw = efficiency.build_component_label(
            label_base,
            name,
        )
        raw = efficiency.apply_eligibility(
            raw,
            name,
            eligibility,
        )
        rows = eff_inference(
            current,
            raw,
            name,
            eligibility,
        )
        model, _manifest, names = booster(
            root,
            "efficiency",
            name,
        )
        if names != list(efficiency.FEATURES[name]):
            raise AssertionError(
                f"{name}: manifest/trainer order mismatch"
            )
        prediction = efficiency.transform_prediction(
            model.predict(
                efficiency.feature_matrix(rows, name)
            ),
            name,
        )
        if not np.isfinite(prediction).all():
            raise AssertionError(
                f"{name}: nonfinite efficiency prediction"
            )
        frame = rows[GRAIN].copy()
        frame[name] = prediction
        common.ensure_unique(
            frame,
            GRAIN,
            f"Issue33 validator {name}",
        )
        out[name] = frame
    return out


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(
                    f"Unable to inspect {path}:{name}"
                )
            return segment
    raise AssertionError(f"Missing function {path}:{name}")


def audit_downstream_formula_ownership(
    prop: Path,
) -> None:
    allocation_path = (
        prop
        / "scripts"
        / "project"
        / "allocate_team_opportunity.py"
    )
    project_week_path = (
        prop
        / "scripts"
        / "project"
        / "project_week.py"
    )

    allocation_main = function_source(
        allocation_path,
        "main",
    )
    if "score_component(" in allocation_main:
        raise AssertionError(
            "Issue 34 still rescoring component models instead of consuming "
            "Issue 33 outputs"
        )
    for marker in [
        "projected_player_carries",
        "projected_team_rush_attempts",
        "projected_defensive_participation",
    ]:
        if marker not in allocation_main:
            raise AssertionError(
                f"Issue 34 missing canonical Item 33 input: {marker}"
            )

    final_builder = function_source(
        project_week_path,
        "build_component_points",
    )
    if "pc.final_component_points(" not in final_builder:
        raise AssertionError(
            "Issue 36 does not consume canonical Item 33 component outputs"
        )
    for forbidden in [
        "score_opportunity_component(",
        "score_efficiency_model(",
        "passing_td_rate",
        "rushing_td_per_goal_line_carry",
        "receiving_td_per_red_zone_target",
        "extra_point_conversion",
        "tackle_rate_per_defensive_play",
        "sack_rate_per_defensive_play",
    ]:
        if forbidden in final_builder:
            raise AssertionError(
                f"Issue 36 duplicates component model/formula logic: {forbidden}"
            )


def main() -> int:
    parsed = args()
    config = common.load_config()
    season = (
        int(parsed.season)
        if parsed.season is not None
        else int(config["seasons"]["current"])
    )
    week = int(parsed.week)
    repo = common.repo_root()
    prop = common.prop_root()

    builder = (
        prop
        / "scripts"
        / "project"
        / "project_components.py"
    )
    features_path = (
        prop
        / "data"
        / "current"
        / "features"
        / f"{season}_week_{week}_features.parquet"
    )
    roles_path = (
        prop
        / "data"
        / "current"
        / f"{season}_week_{week}_roles.parquet"
    )
    output_path = (
        prop
        / "data"
        / "current"
        / f"{season}_week_{week}_component_projections.parquet"
    )
    log_path = (
        prop
        / "logs"
        / f"component_projections_{season}_week_{week}.json"
    )
    eligibility_path = (
        prop
        / "config"
        / "target_eligibility.yaml"
    )
    historical_path = repo / config["paths"]["historical_features"]

    print(
        "CHECK 01: exact expanded Item 33 headers and canonical player grain"
    )
    required = [
        builder,
        features_path,
        roles_path,
        output_path,
        log_path,
        eligibility_path,
        historical_path,
    ]
    if week == 1:
        required.append(
            prop
            / "data"
            / "current"
            / f"{season}_week_1_priors.parquet"
        )
    for path in required:
        if not path.is_file():
            raise AssertionError(
                f"Missing Issue 33 artifact/input: {path}"
            )

    out = pd.read_parquet(output_path)
    if list(out.columns) != HEADERS:
        raise AssertionError(
            f"Issue 33 headers mismatch: {list(out.columns)}"
        )
    common.ensure_unique(
        out,
        GRAIN,
        "Issue33 validator output",
    )

    features = pd.read_parquet(features_path)
    roles = pd.read_parquet(roles_path)
    common.ensure_unique(
        features,
        GRAIN,
        "Issue33 validator features",
    )
    features = features.copy()
    features["kickoff_timestamp"] = pd.to_datetime(
        features["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )

    grain_check = features[
        [*GRAIN, "team", "opponent", "position"]
    ].merge(
        out,
        on=[
            *GRAIN,
            "team",
            "opponent",
            "position",
        ],
        how="outer",
        indicator=True,
    )
    if not grain_check["_merge"].eq("both").all():
        raise AssertionError(
            "Issue 33 output does not exactly match current feature player grain"
        )

    print(
        "CHECK 02: independently rescore all 12 Issue 22 opportunity models"
    )
    eligibility = load_yaml(eligibility_path)
    opp = score_opp(
        prop,
        features,
        eligibility,
        config,
        repo,
        season,
    )

    print(
        "CHECK 03: independently rescore all 10 Issue 23 efficiency models"
    )
    history = pd.read_parquet(
        historical_path,
        columns=eff_cols(),
    )
    history["season"] = pd.to_numeric(
        history["season"],
        errors="raise",
    ).astype(int)
    history["week"] = pd.to_numeric(
        history["week"],
        errors="raise",
    ).astype(int)
    history["kickoff_timestamp"] = pd.to_datetime(
        history["kickoff_timestamp"],
        errors="raise",
        utc=True,
    )
    history = history.loc[history["season"].lt(season)].copy()
    eff = score_eff(
        prop,
        features,
        history,
        eligibility,
        config,
    )

    print(
        "CHECK 04: independently recompute every projected component input"
    )
    expected = features[
        [
            *GRAIN,
            "team",
            "opponent",
            "position",
            *RZ,
            *GL,
        ]
    ].copy()

    for name, output_column in [
        ("team_pass_attempts", "projected_team_pass_attempts"),
        ("team_rush_attempts", "projected_team_rush_attempts"),
        ("field_goal_attempts", "_team_fg_attempts"),
        ("extra_point_attempts", "_team_pat_attempts"),
        ("opponent_offensive_plays", "projected_opponent_plays"),
        ("opponent_dropbacks", "projected_opponent_dropbacks"),
    ]:
        expected = expected.merge(
            opp[name].rename(
                columns={name: output_column}
            ),
            on=TEAM_GRAIN,
            how="left",
            validate="many_to_one",
        )

    for name, output_column in [
        ("qb_pass_attempts", "_raw_qb"),
        ("player_carry_share", "_raw_carry_share"),
        ("player_target_share", "projected_target_share"),
        (
            "player_red_zone_target_share",
            "_raw_rz_share",
        ),
        (
            "player_goal_line_carry_share",
            "_raw_gl_share",
        ),
        (
            "player_defensive_participation",
            "_raw_def_part",
        ),
    ]:
        expected = expected.merge(
            opp[name].rename(
                columns={name: output_column}
            ),
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )

    for name, output_column in [
        (
            "passing_yards_per_attempt",
            "projected_yards_per_attempt",
        ),
        ("passing_td_rate", "_pass_td_rate"),
        (
            "rushing_yards_per_carry",
            "projected_yards_per_carry",
        ),
        (
            "rushing_td_per_goal_line_carry",
            "_rush_td_rate",
        ),
        (
            "receiving_yards_per_target",
            "projected_yards_per_target",
        ),
        (
            "receiving_td_per_red_zone_target",
            "_rec_td_rate",
        ),
        (
            "field_goal_conversion",
            "_raw_fg_probability",
        ),
        (
            "extra_point_conversion",
            "_raw_pat_probability",
        ),
        (
            "tackle_rate_per_defensive_play",
            "projected_tackle_rate",
        ),
        (
            "sack_rate_per_defensive_play",
            "projected_sack_rate",
        ),
    ]:
        expected = expected.merge(
            eff[name].rename(
                columns={name: output_column}
            ),
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )

    role = roles[
        [
            *GRAIN,
            "primary_qb_flag",
            "primary_kicker_flag",
        ]
    ].copy()
    expected = expected.merge(
        role,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    qb = num(expected["primary_qb_flag"]).fillna(0).gt(0)
    kicker = (
        num(expected["primary_kicker_flag"]).fillna(0).gt(0)
    )

    expected["projected_qb_pass_attempts"] = 0.0
    expected.loc[
        qb,
        "projected_qb_pass_attempts",
    ] = num(expected.loc[qb, "_raw_qb"]).to_numpy()

    expected["_raw_carry_share"] = (
        num(expected["_raw_carry_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    expected["projected_target_share"] = (
        num(expected["projected_target_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    expected["_raw_rz_share"] = (
        num(expected["_raw_rz_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    expected["_raw_gl_share"] = (
        num(expected["_raw_gl_share"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    expected["projected_player_carries"] = (
        num(expected["projected_team_rush_attempts"]).clip(lower=0.0)
        * expected["_raw_carry_share"]
    )
    expected["projected_targets"] = (
        num(expected["projected_team_pass_attempts"]).clip(lower=0.0)
        * expected["projected_target_share"]
    )
    rz_volume = coalesce(expected, RZ).clip(lower=0.0)
    gl_volume = coalesce(expected, GL).clip(lower=0.0)
    expected["projected_red_zone_targets"] = (
        rz_volume * expected["_raw_rz_share"]
    )
    expected["projected_goal_line_carries"] = (
        gl_volume * expected["_raw_gl_share"]
    )

    expected["projected_fg_attempts"] = 0.0
    expected.loc[
        kicker,
        "projected_fg_attempts",
    ] = num(expected.loc[kicker, "_team_fg_attempts"]).to_numpy()
    expected["projected_pat_attempts"] = 0.0
    expected.loc[
        kicker,
        "projected_pat_attempts",
    ] = num(expected.loc[kicker, "_team_pat_attempts"]).to_numpy()

    expected["projected_fg_make_probability"] = np.nan
    expected.loc[
        kicker,
        "projected_fg_make_probability",
    ] = num(
        expected.loc[kicker, "_raw_fg_probability"]
    ).clip(0.0, 1.0).to_numpy()
    expected["projected_pat_make_probability"] = np.nan
    expected.loc[
        kicker,
        "projected_pat_make_probability",
    ] = num(
        expected.loc[kicker, "_raw_pat_probability"]
    ).clip(0.0, 1.0).to_numpy()

    expected["projected_defensive_participation"] = (
        num(expected["_raw_def_part"])
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    expected["projected_tackle_rate"] = num(
        expected["projected_tackle_rate"]
    ).clip(0.0, 1.0)
    expected["projected_sack_rate"] = num(
        expected["projected_sack_rate"]
    ).clip(0.0, 1.0)

    expected["_pass_td_rate"] = num(
        expected["_pass_td_rate"]
    ).clip(0.0, 1.0)
    expected["_rush_td_rate"] = num(
        expected["_rush_td_rate"]
    ).clip(0.0, 1.0)
    expected["_rec_td_rate"] = num(
        expected["_rec_td_rate"]
    ).clip(0.0, 1.0)

    for column in [
        "projected_team_pass_attempts",
        "projected_qb_pass_attempts",
        "projected_team_rush_attempts",
        "projected_player_carries",
        "projected_target_share",
        "projected_targets",
        "projected_yards_per_attempt",
        "projected_yards_per_carry",
        "projected_yards_per_target",
        "projected_red_zone_targets",
        "projected_goal_line_carries",
        "projected_fg_attempts",
        "projected_fg_make_probability",
        "projected_pat_attempts",
        "projected_pat_make_probability",
        "projected_opponent_plays",
        "projected_opponent_dropbacks",
        "projected_defensive_participation",
        "projected_tackle_rate",
        "projected_sack_rate",
    ]:
        assert_close(
            out[column],
            expected[column],
            column,
        )

    print(
        "CHECK 05: independently recompute all nine target component projections"
    )
    expected["component_passing_yards"] = zero_safe_product(
        expected["projected_qb_pass_attempts"],
        expected["projected_yards_per_attempt"],
    )
    expected["component_passing_tds"] = zero_safe_product(
        expected["projected_qb_pass_attempts"],
        expected["_pass_td_rate"],
    ).clip(lower=0.0)
    expected["component_rushing_yards"] = zero_safe_product(
        expected["projected_player_carries"],
        expected["projected_yards_per_carry"],
    )
    expected["component_rushing_tds"] = zero_safe_product(
        expected["projected_goal_line_carries"],
        expected["_rush_td_rate"],
    ).clip(lower=0.0)
    expected["component_receiving_yards"] = zero_safe_product(
        expected["projected_targets"],
        expected["projected_yards_per_target"],
    )
    expected["component_receiving_tds"] = zero_safe_product(
        expected["projected_red_zone_targets"],
        expected["_rec_td_rate"],
    ).clip(lower=0.0)
    expected["component_kicking_points"] = (
        3.0
        * zero_safe_product(
            expected["projected_fg_attempts"],
            expected["projected_fg_make_probability"],
        )
        + zero_safe_product(
            expected["projected_pat_attempts"],
            expected["projected_pat_make_probability"],
        )
    ).clip(lower=0.0)
    expected["component_tackles"] = zero_safe_product(
        expected["projected_opponent_plays"],
        expected["projected_defensive_participation"],
        expected["projected_tackle_rate"],
    ).clip(lower=0.0)
    expected["component_sacks"] = zero_safe_product(
        expected["projected_opponent_plays"],
        expected["projected_defensive_participation"],
        expected["projected_sack_rate"],
    ).clip(lower=0.0)

    for column in TARGET_COMPONENTS:
        assert_close(
            out[column],
            expected[column],
            column,
        )
        if num(out[column]).isna().any():
            raise AssertionError(
                f"{column}: nonfinite target component projection"
            )

    print(
        "CHECK 06: bounds, PAT/defensive semantics, and sacks denominator"
    )
    for column in [
        "projected_team_pass_attempts",
        "projected_qb_pass_attempts",
        "projected_team_rush_attempts",
        "projected_player_carries",
        "projected_target_share",
        "projected_targets",
        "projected_red_zone_targets",
        "projected_goal_line_carries",
        "projected_fg_attempts",
        "projected_pat_attempts",
        "projected_opponent_plays",
        "projected_opponent_dropbacks",
        "projected_defensive_participation",
    ]:
        values = num(out[column])
        if values.isna().any() or values.lt(0.0).any():
            raise AssertionError(
                f"Invalid nonnegative Issue 33 output: {column}"
            )

    if num(out["projected_target_share"]).gt(1.0).any():
        raise AssertionError("Target share outside [0,1]")
    if not num(
        out.loc[kicker, "projected_fg_make_probability"]
    ).between(0.0, 1.0).all():
        raise AssertionError("Primary kicker FG probability invalid")
    if not num(
        out.loc[kicker, "projected_pat_make_probability"]
    ).between(0.0, 1.0).all():
        raise AssertionError("Primary kicker PAT probability invalid")
    if out.loc[
        ~kicker,
        [
            "projected_fg_make_probability",
            "projected_pat_make_probability",
        ],
    ].notna().any().any():
        raise AssertionError(
            "Non-primary kicker received kicking efficiency probability"
        )

    builder_source = builder.read_text(encoding="utf-8-sig")
    sacks_block = function_source(
        builder,
        "add_raw_target_components",
    )
    if (
        '"projected_opponent_plays"' not in sacks_block
        or '"projected_sack_rate"' not in sacks_block
        or '"projected_defensive_participation"' not in sacks_block
    ):
        raise AssertionError(
            "Canonical sacks component formula is incomplete"
        )
    if '"projected_opponent_dropbacks"' in sacks_block:
        raise AssertionError(
            "Canonical sacks component formula uses opponent dropbacks"
        )

    print(
        "CHECK 07: downstream allocation/final assembly consume Item 33 outputs"
    )
    audit_downstream_formula_ownership(prop)

    print(
        "CHECK 08: market exclusion and Issue 33 log policy"
    )
    common.reject_forbidden_feature_columns(
        out.columns,
        config,
    )
    market = (
        SCRIPTS
        / "validate"
        / "audit_market_exclusion.py"
    )
    completed = subprocess.run(
        [sys.executable, str(market)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        completed.returncode != 0
        or "MARKET EXCLUSION AUDIT: PASS" not in completed.stdout
    ):
        raise AssertionError("Market exclusion audit failed")

    log = load_json(log_path)
    if log.get("shares_reconciled") is not False:
        raise AssertionError(
            "Issue 33 prematurely reconciled shares"
        )
    if (
        log.get("share_reconciliation_stage")
        != "current_week_allocation"
    ):
        raise AssertionError(
            "Issue 33 share allocation boundary is incorrect"
        )
    if int(log.get("opportunity_models_scored", -1)) != len(OPP):
        raise AssertionError(
            "Issue 33 did not score all opportunity models"
        )
    if int(log.get("efficiency_models_scored", -1)) != len(EFF):
        raise AssertionError(
            "Issue 33 did not score all efficiency models"
        )
    if int(log.get("target_components_projected", -1)) != 9:
        raise AssertionError(
            "Issue 33 did not log all nine target components"
        )
    if log.get("market_features_used") is not False:
        raise AssertionError(
            "Issue 33 log says market features were used"
        )

    print(f"season={season}")
    print(f"week={week}")
    print(f"games={out['game_id'].nunique()}")
    print(f"teams={out['team'].nunique()}")
    print(f"rows={len(out)}")
    print(f"columns={len(out.columns)}")
    print(f"opportunity_models_rescored={len(OPP)}")
    print(f"efficiency_models_rescored={len(EFF)}")
    print("target_component_projections_recomputed=9")
    print("projected_pat_make_probability=validated")
    print("defensive_opportunity_outputs=validated")
    print("defensive_efficiency_outputs=validated")
    print("sacks_denominator=projected_opponent_plays")
    print("duplicate_component_model_scoring_downstream=false")
    print("duplicate_component_formula_implementation_downstream=false")
    print("shares_reconciled=false")
    print("market_features_used=false")
    print("ISSUE 33 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
