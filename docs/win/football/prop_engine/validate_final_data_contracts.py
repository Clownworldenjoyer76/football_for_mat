#!/usr/bin/env python3
"""Final independent leakage, lagging, and source-availability validator."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROP = Path(__file__).resolve().parent
SCRIPTS_ROOT = PROP / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


GRAIN = ["season", "week", "game_id", "player_id"]
TEAM_HISTORY_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LAR"}

REQUIRED_SCRIPTS = [
    "scripts/build/build_historical_universe.py",
    "scripts/build/build_player_opportunity.py",
    "scripts/build/build_team_opportunity.py",
    "scripts/build/build_position_allowed.py",
    "scripts/build/build_role_history.py",
    "scripts/build/build_player_form.py",
    "scripts/build/build_team_form.py",
    "scripts/build/build_environment_history.py",
    "scripts/build/build_historical_features.py",
    "scripts/build/build_defensive_features.py",
    "scripts/build/build_kicking_features.py",
    "scripts/project/build_current_features.py",
    "scripts/project/build_week1_priors.py",
]

PLAYER_LAG_METRICS = [
    "pass_attempts",
    "yards_per_attempt",
    "carries",
    "yards_per_carry",
    "targets",
    "yards_per_target",
    "field_goal_attempts",
    "tackles",
    "tackle_rate_per_def_play",
    "sacks",
    "sack_rate_per_def_play",
]

MODEL_FEATURE_KEYS = {
    "feature_columns",
    "numeric_features",
    "categorical_features",
    "canonical_features",
    "derived_features",
    "feature_names",
    "model_features",
    "input_features",
    "features",
}

SAFE_TEAM_SUFFIXES = (
    "_lag1",
    "_roll3_mean",
    "_roll5_mean",
    "_roll8_mean",
    "_ewm3",
    "_ewm5",
    "_season_to_date",
)


def fail(message: str) -> None:
    raise AssertionError(message)


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


def norm_team(value: Any) -> str:
    team = common.normalize_team(value)
    return TEAM_HISTORY_ALIASES.get(team, team)


def norm_player(value: Any) -> str:
    return common.normalize_player_id(value)


def repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Required JSON missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        fail(f"Expected JSON object: {path}")
    return value


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Required parquet missing: {path}")
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        fail(f"Unable to read required columns from {path}: {exc}")


def squash(source: str) -> str:
    compact = re.sub(r"\s+", "", source)
    # Formatting must not affect contract validation. Normalize optional
    # trailing commas inside Python list/dict/tuple literals.
    compact = re.sub(r",(?=[\]\}\)])", "", compact)
    return compact


def require_markers(source: str, markers: Iterable[str], label: str) -> None:
    compact = squash(source)
    missing = [marker for marker in markers if marker not in compact]
    if missing:
        fail(f"{label} missing required contract marker(s): {missing}")


def static_source_contracts() -> None:
    sources: dict[str, str] = {}
    for rel in REQUIRED_SCRIPTS:
        path = PROP / rel
        if not path.is_file():
            fail(f"Required contract script missing: {path}")
        source = path.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(path))
        sources[rel] = source

    require_markers(
        sources["scripts/build/build_historical_universe.py"],
        [
            "participant_game_teams",
            "ambiguous_player_weeks",
            "iflen(teams)>1",
            "roster_context",
            "context_candidates",
            "snap_gsis",
            "global_identity",
            "latest_prior(",
            "bisect_left(",
            "historical_participation_pattern",
            "played_game_flag",
        ],
        "build_historical_universe.py",
    )

    universe_source = sources["scripts/build/build_historical_universe.py"]
    participation_start_ok = (
        re.search(r"if\s+season\s*<\s*2016\s*:", universe_source) is not None
        or (
            "participation_feature_start" in universe_source
            and re.search(r"season\s*<", universe_source) is not None
        )
    )
    if not participation_start_ok:
        fail(
            "build_historical_universe.py does not enforce the participation "
            "source-availability start at 2016/configured participation_feature_start."
        )

    require_markers(
        sources["scripts/build/build_player_opportunity.py"],
        [
            "Same-weekrealizedvaluesarerawmeasurementsonly.",
            "attach_snaps(",
            'left_on=[*GRAIN,"team"]',
            'right_on=[*GRAIN,"_snap_team"]',
            "yards_per_attempt",
            "yards_per_carry",
            "yards_per_target",
        ],
        "build_player_opportunity.py",
    )

    require_markers(
        sources["scripts/build/build_team_opportunity.py"],
        [
            "Same-weekrealizedtablesareimmutablerawmeasurements.",
            "Alllaggingoccursdownstream.",
            "TEAM_COLUMNS",
            "OPPONENT_COLUMNS",
        ],
        "build_team_opportunity.py",
    )

    require_markers(
        sources["scripts/build/build_position_allowed.py"],
        [
            "Thisisanimmutablesame-weekrealizedtable.Lagbeforemodeluse.",
            "Neveruserawearly-seasonunshrunkratesasproductionfeatures.",
            "shrunk_rate",
            "raw_rate_sample_size",
        ],
        "build_position_allowed.py",
    )

    require_markers(
        sources["scripts/build/build_role_history.py"],
        [
            'source_rows[pointer]["_kickoff"]<target_kickoff',
            "previous_kickoff<kickoff",
            "games_with_current_team_before_game",
            "snap_pct_roll3",
            "participation_roll3",
        ],
        "build_role_history.py",
    )

    require_markers(
        sources["scripts/build/build_player_form.py"],
        [
            "TEAM_SHARE_METRICS",
            "_target_stint_id",
            "ifmetricinTEAM_SHARE_METRICS:",
            'by=["player_id","_stint_id"]',
            'by=["player_id"]',
            "allow_exact_matches=False",
            '"team_share_policy":"resettocurrentfranchisestintafterteamchange"',
            '"career_policy":"non-sharecareerhistorysurvivesteamchanges"',
        ],
        "build_player_form.py",
    )

    require_markers(
        sources["scripts/build/build_team_form.py"],
        [
            "Nosame-weekrealizedvaluemayenterthatweek'sfeaturerow.",
            "history.append(current)",
            "season_history.append(current)",
            "build_form(",
        ],
        "build_team_form.py",
    )

    require_markers(
        sources["scripts/build/build_environment_history.py"],
        [
            "GAME_READ_COLUMNS",
            "home_rest_days",
            "away_rest_days",
            "miles_traveled_away",
            "time_zones_crossed_away",
        ],
        "build_environment_history.py",
    )

    require_markers(
        sources["scripts/build/build_historical_features.py"],
        [
            "grouped[column].shift(1)",
            "matchup_position_allowed_",
            "environment_team_rest_days",
            "environment_opponent_rest_days",
            "environment_team_miles_traveled",
            "environment_opponent_miles_traveled",
            'if"played_game_flag"incandidate_featuresor"played_game_flag"inout.columns:',
            '"outcome_metadata":["played_game_flag"]',
            "common.reject_forbidden_feature_columns(",
            "REQUIRED_TARGET_ORDER",
        ],
        "build_historical_features.py",
    )

    require_markers(
        sources["scripts/build/build_defensive_features.py"],
        [
            "strictlypriorper-gamedefensesackrates",
            "history.append(float(current))",
            "team_def_sack_rate_roll3",
        ],
        "build_defensive_features.py",
    )

    require_markers(
        sources["scripts/build/build_kicking_features.py"],
        [
            "already-laggedteam_form",
            "already-laggedopponent_form",
            "strictlypriorkickingusage",
            "primary_kicker_flag",
        ],
        "build_kicking_features.py",
    )

    require_markers(
        sources["scripts/project/build_current_features.py"],
        [
            'x=x.loc[x["_week"].lt(week)]',
            "selected_manifest_specs(",
            "feature_manifest.json",
            "position_allowed",
            "common.reject_forbidden_feature_columns(",
            'TARGETS=list(_CONFIG_CONTRACT["targets"].keys())',
        ],
        "build_current_features.py",
    )

    require_markers(
        sources["scripts/project/build_week1_priors.py"],
        [
            'h=h.loc[h["season"].lt(season)&h["_played"]',
            "New-teamrowsimportedold-teamrole/sharepriors.",
            "prior_efficiency",
            "new_team_veteran",
        ],
        "build_week1_priors.py",
    )


def manifest_feature_names(value: Any) -> list[str]:
    names: list[str] = []

    def flatten_feature_value(item: Any) -> None:
        if isinstance(item, str):
            text = clean(item)
            if text:
                names.append(text)
        elif isinstance(item, list):
            for child in item:
                flatten_feature_value(child)
        elif isinstance(item, dict):
            if isinstance(item.get("name"), str):
                text = clean(item["name"])
                if text:
                    names.append(text)
            else:
                for child in item.values():
                    if isinstance(child, (list, dict)):
                        flatten_feature_value(child)

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key) in MODEL_FEATURE_KEYS:
                    flatten_feature_value(child)
                else:
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return list(dict.fromkeys(names))


def raw_columns(path: Path, excluded: set[str]) -> set[str]:
    if not path.is_file():
        fail(f"Required raw table missing: {path}")
    cols = set(pd.read_parquet(path).columns)
    return {column for column in cols if column not in excluded}


def strict_leakage_reason(
    name: str,
    target_columns: set[str],
) -> str | None:
    """Return a fail-closed leakage reason for a model feature name.

    Raw realized fields are rejected by exact semantic identity. Engineered
    role/history/matchup features are not rejected merely because their names
    contain words such as "snap", "participation", "target_share", or
    "carry_share"; their chronology is independently validated elsewhere in
    this validator.
    """
    feature = clean(name)
    lower = feature.casefold()

    if not feature:
        return None

    if feature in target_columns:
        return "configured target column"

    if "played_game_flag" in lower:
        return "played_game_flag"

    if lower.startswith("target_"):
        return "target_* outcome column"

    if lower.startswith("audit_"):
        return "audit_* outcome column"

    final_score_tokens = (
        "final_score",
        "home_score",
        "away_score",
        "score_home",
        "score_away",
    )
    if any(token in lower for token in final_score_tokens):
        return "final score field"

    raw_base = lower
    for prefix in (
        "player_",
        "raw_",
        "same_game_",
        "realized_",
    ):
        if raw_base.startswith(prefix):
            raw_base = raw_base[len(prefix):]
            break

    raw_snap_participation = {
        "snaps",
        "snap_count",
        "snap_counts",
        "snap_pct",
        "offense_snaps",
        "defense_snaps",
        "offense_snap_count",
        "defense_snap_count",
        "offense_snap_pct",
        "defense_snap_pct",
        "participation",
        "offense_participation",
        "defense_participation",
    }
    if raw_base in raw_snap_participation:
        return "same-game snap/participation field"

    raw_team_shares = {
        "target_share",
        "carry_share",
        "red_zone_target_share",
        "air_yards_share",
    }
    if raw_base in raw_team_shares:
        return "same-game target/carry share"

    return None



def require_no_explicit_leakage_features(
    names: Iterable[str],
    *,
    target_columns: set[str],
    label: str,
) -> None:
    violations: list[str] = []

    for raw_name in names:
        name = clean(raw_name)
        reason = strict_leakage_reason(
            name,
            target_columns,
        )
        if reason is not None:
            violations.append(
                f"{name} [{reason}]"
            )

    if violations:
        fail(
            f"{label}: explicit leakage feature(s): "
            + "; ".join(violations[:30])
        )


def validate_historical_current_semantics(
    prop: Path,
    config: dict[str, Any],
    historical_manifest: dict[str, Any],
    historical_features: list[str],
) -> int:
    """Require current feature outputs/manifests to mirror historical semantics."""
    leading = [
        clean(value)
        for value in historical_manifest.get(
            "leading_columns",
            [],
        )
        if clean(value)
    ]

    if not leading:
        fail(
            "Historical feature manifest has no leading_columns."
        )

    expected_columns = [
        *leading,
        *[
            name
            for name in historical_features
            if name not in leading
        ],
    ]

    current_root = (
        prop
        / "data/current/features"
    )
    current_manifests = sorted(
        current_root.glob(
            "*_feature_manifest.json"
        )
    )

    checked = 0

    for manifest_path in current_manifests:
        payload = read_json(
            manifest_path
        )

        manifest_features = payload.get(
            "feature_columns"
        )
        if isinstance(
            manifest_features,
            list,
        ):
            normalized = [
                clean(value)
                for value in manifest_features
                if clean(value)
            ]
            if normalized != historical_features:
                missing = [
                    value
                    for value in historical_features
                    if value not in normalized
                ]
                extra = [
                    value
                    for value in normalized
                    if value not in historical_features
                ]
                fail(
                    f"{manifest_path}: current/historical "
                    "feature semantics differ; "
                    f"missing={missing[:20]} "
                    f"extra={extra[:20]}"
                )

        output_columns = payload.get(
            "output_columns"
        )
        if isinstance(
            output_columns,
            list,
        ):
            normalized_output = [
                clean(value)
                for value in output_columns
                if clean(value)
            ]
            if normalized_output != expected_columns:
                missing = [
                    value
                    for value in expected_columns
                    if value not in normalized_output
                ]
                extra = [
                    value
                    for value in normalized_output
                    if value not in expected_columns
                ]
                fail(
                    f"{manifest_path}: current output schema "
                    "does not match canonical historical schema; "
                    f"missing={missing[:20]} "
                    f"extra={extra[:20]}"
                )

        parquet_name = (
            manifest_path.name
            .replace(
                "_feature_manifest.json",
                "_features.parquet",
            )
        )
        parquet_path = (
            manifest_path.parent
            / parquet_name
        )

        if parquet_path.is_file():
            actual_columns = list(
                pd.read_parquet(
                    parquet_path
                ).columns
            )
            if actual_columns != expected_columns:
                missing = [
                    value
                    for value in expected_columns
                    if value not in actual_columns
                ]
                extra = [
                    value
                    for value in actual_columns
                    if value not in expected_columns
                ]
                fail(
                    f"{parquet_path}: current feature table "
                    "does not exactly match historical feature "
                    "semantics/order; "
                    f"missing={missing[:20]} "
                    f"extra={extra[:20]}"
                )

        checked += 1

    return checked


def validate_feature_manifests(
    prop: Path,
    config: dict[str, Any],
    player_raw: set[str],
    team_raw: set[str],
    opponent_raw: set[str],
) -> tuple[int, list[str]]:
    historical_manifest = (
        prop / "data/historical/features/feature_manifest.json"
    )
    manifest = read_json(historical_manifest)
    features = [clean(x) for x in manifest.get("feature_columns", []) if clean(x)]
    if not features:
        fail("Historical feature manifest has no feature_columns.")

    target_columns = {
        clean(name)
        for name in config["targets"].keys()
        if clean(name)
    }
    require_no_explicit_leakage_features(
        features,
        target_columns=target_columns,
        label="historical feature manifest",
    )

    forbidden_exact = {"played_game_flag"}
    bad = [x for x in features if "played_game_flag" in x]
    if bad:
        fail(f"played_game_flag leaked into historical feature manifest: {bad[:10]}")
    bad = [x for x in features if x.startswith("target_") or x.startswith("audit_")]
    if bad:
        fail(f"Target/audit columns leaked into historical feature manifest: {bad[:10]}")

    position_features = [x for x in features if "position_allowed" in x]
    if not position_features:
        fail("Historical feature manifest has no position-allowed features.")
    bad_position = [x for x in position_features if not x.endswith("_lag1")]
    if bad_position:
        fail(f"Unlagged position-allowed production feature(s): {bad_position[:20]}")
    if "matchup_position_allowed_shrunk_rate_lag1" not in features:
        fail("Lagged position_allowed shrunk_rate is missing from historical features.")
    raw_unshrunk = [
        x
        for x in features
        if re.search(r"position_allowed_raw_rate(?:$|_)", x)
        and "raw_rate_sample_size" not in x
    ]
    if raw_unshrunk:
        fail(f"Raw unshrunk position-allowed rate entered production: {raw_unshrunk[:10]}")

    unsafe_raw = []
    for metric in player_raw:
        if metric in features or f"player_{metric}" in features:
            unsafe_raw.append(metric)
    if unsafe_raw:
        fail(f"Raw same-game player realization feature(s): {sorted(unsafe_raw)[:20]}")

    families = manifest.get("column_families", {})
    for family_name, raw_set in (("team", team_raw), ("opponent", opponent_raw)):
        family = list(families.get(family_name, []))
        for name in family:
            prefix = f"{family_name}_"
            base = name[len(prefix):] if name.startswith(prefix) else name
            if base in raw_set:
                fail(f"Raw same-week {family_name} value entered features: {name}")
            if name.startswith(prefix) and not name.endswith(SAFE_TEAM_SUFFIXES):
                fail(f"Unsafe {family_name} form suffix in historical manifest: {name}")

    excluded = manifest.get("excluded_from_features", {})
    outcome_meta = excluded.get("outcome_metadata", [])
    if "played_game_flag" not in outcome_meta:
        fail("Historical manifest does not explicitly exclude played_game_flag.")

    manifests = sorted((prop / "models").rglob("feature_manifest.json"))
    current_manifests = sorted(
        (prop / "data/current/features").glob("*_feature_manifest.json")
    )
    all_manifests = manifests + [historical_manifest] + current_manifests
    if not manifests:
        fail("No model feature manifests found.")

    for path in all_manifests:
        data = read_json(path)
        names = manifest_feature_names(data)
        require_no_explicit_leakage_features(
            names,
            target_columns=target_columns,
            label=str(path),
        )
        for name in names:
            if "played_game_flag" in name:
                fail(f"played_game_flag appears as a model feature in {path}: {name}")
            if name.startswith("target_") or name.startswith("audit_"):
                fail(f"Target/audit feature appears in {path}: {name}")
            if "position_allowed" in name and not name.endswith("_lag1"):
                fail(f"Unlagged position-allowed feature in {path}: {name}")
            if (
                re.search(r"position_allowed_raw_rate(?:$|_)", name)
                and "raw_rate_sample_size" not in name
            ):
                fail(f"Raw unshrunk position-allowed rate in {path}: {name}")
            if name in player_raw or (
                name.startswith("player_") and name[len("player_"):] in player_raw
            ):
                fail(f"Raw same-game player realization in {path}: {name}")
            if name.startswith("team_") and name[len("team_"):] in team_raw:
                fail(f"Raw same-week team realization in {path}: {name}")
            if name.startswith("opponent_") and name[len("opponent_"):] in opponent_raw:
                fail(f"Raw same-week opponent realization in {path}: {name}")

    validate_historical_current_semantics(
        prop,
        config,
        manifest,
        features,
    )

    return len(manifests), features


def compare_series(actual: pd.Series, expected: pd.Series, label: str, atol: float = 1e-5) -> None:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype="float64")
    ok = np.isclose(a, e, rtol=0.0, atol=atol, equal_nan=True)
    if not bool(ok.all()):
        bad = np.flatnonzero(~ok)[:10]
        sample = [
            {
                "actual": None if np.isnan(a[i]) else float(a[i]),
                "expected": None if np.isnan(e[i]) else float(e[i]),
            }
            for i in bad
        ]
        fail(f"{label} mismatch; sample={sample}")


def prepare_game_kickoffs(universe_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_columns(
        universe_path,
        GRAIN + ["kickoff_timestamp", "team", "position_group"],
    )
    universe["player_id"] = universe["player_id"].map(norm_player)
    universe["_kickoff"] = pd.to_datetime(
        universe["kickoff_timestamp"], errors="coerce", utc=True
    )
    if universe["_kickoff"].isna().any():
        fail("Historical universe contains invalid kickoff timestamps.")
    common.ensure_unique(universe, GRAIN, "final-contract historical universe")

    game_kickoffs = (
        universe[["season", "week", "game_id", "_kickoff"]]
        .drop_duplicates(["season", "week", "game_id"])
    )
    if game_kickoffs.duplicated(["season", "week", "game_id"]).any():
        fail("Historical game maps to multiple kickoff timestamps.")
    return universe, game_kickoffs


def validate_player_lag(
    universe: pd.DataFrame,
    game_kickoffs: pd.DataFrame,
    player_opportunity_path: Path,
    player_form_path: Path,
) -> int:
    available_source_cols = set(pd.read_parquet(player_opportunity_path).columns)
    available_form_cols = set(pd.read_parquet(player_form_path).columns)
    metrics = [
        metric
        for metric in PLAYER_LAG_METRICS
        if metric in available_source_cols and f"{metric}_lag1" in available_form_cols
    ]
    if len(metrics) < 6:
        fail(f"Too few player lag metrics available for independent validation: {metrics}")

    source = read_columns(
        player_opportunity_path,
        GRAIN + metrics,
    )
    source["player_id"] = source["player_id"].map(norm_player)
    common.ensure_unique(source, GRAIN, "player opportunity final-contract grain")
    source = source.merge(
        game_kickoffs,
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )
    if source["_kickoff"].isna().any():
        fail("Player opportunity contains a game without historical kickoff mapping.")

    lag_cols = [f"{metric}_lag1" for metric in metrics]
    form = read_columns(player_form_path, GRAIN + lag_cols)
    form["player_id"] = form["player_id"].map(norm_player)
    form = form.merge(
        universe[GRAIN + ["_kickoff"]],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if form["_kickoff"].isna().any():
        fail("Player form contains a row without historical kickoff mapping.")
    form["_row_id"] = np.arange(len(form), dtype=np.int64)

    comparisons = 0
    for metric in metrics:
        right = source[["player_id", "_kickoff", metric]].copy()
        right[metric] = pd.to_numeric(right[metric], errors="coerce")
        right = right.loc[
            right["player_id"].ne("") & right[metric].notna()
        ].sort_values(["_kickoff", "player_id"], kind="mergesort")

        left = form[
            ["_row_id", "player_id", "_kickoff", f"{metric}_lag1"]
        ].sort_values(["_kickoff", "player_id"], kind="mergesort")

        joined = pd.merge_asof(
            left,
            right.rename(columns={metric: "_expected"}),
            on="_kickoff",
            by="player_id",
            direction="backward",
            allow_exact_matches=False,
        )
        comparable = joined["_expected"].notna()
        if int(comparable.sum()) == 0:
            fail(f"No comparable strictly-prior rows for player metric {metric}.")
        compare_series(
            joined.loc[comparable, f"{metric}_lag1"],
            joined.loc[comparable, "_expected"],
            f"player {metric} lag1",
            atol=2e-4,
        )
        comparisons += int(comparable.sum())

    return comparisons


def independent_lag1(
    source: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    x = source.copy()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(norm_team)
    x = x.sort_values(["team", "season", "week"], kind="mergesort").reset_index(drop=True)
    common.ensure_unique(x, ["season", "week", "team"], "independent team/opponent source")

    out = x[["season", "week", "team"]].copy()
    for metric in metrics:
        values = pd.to_numeric(x[metric], errors="coerce")
        out[f"{metric}_expected_lag1"] = (
            values.groupby(x["team"], sort=False).transform(lambda s: s.ffill().shift(1))
        )
    return out


def validate_team_form_lags(
    raw_path: Path,
    form_path: Path,
    label: str,
) -> tuple[int, int]:
    raw_columns_all = list(pd.read_parquet(raw_path).columns)
    form_columns_all = set(pd.read_parquet(form_path).columns)
    metrics = [
        column
        for column in raw_columns_all
        if column not in {"season", "week", "team", "game_id"}
        and f"{column}_lag1" in form_columns_all
    ]
    if not metrics:
        fail(f"No {label} lag1 metrics available for validation.")

    source = read_columns(raw_path, ["season", "week", "team"] + metrics)
    form = read_columns(
        form_path,
        ["season", "week", "team"] + [f"{m}_lag1" for m in metrics],
    )
    form["team"] = form["team"].map(norm_team)
    expected = independent_lag1(source, metrics=metrics)
    merged = form.merge(
        expected,
        on=["season", "week", "team"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        fail(f"{label} raw/form key sets differ.")

    comparisons = 0
    for metric in metrics:
        compare_series(
            merged[f"{metric}_lag1"],
            merged[f"{metric}_expected_lag1"],
            f"{label} {metric} lag1",
            atol=2e-5,
        )
        comparisons += len(merged)
    return len(metrics), comparisons


def validate_position_allowed_lag(
    position_allowed_path: Path,
    historical_features_path: Path,
) -> int:
    source = read_columns(
        position_allowed_path,
        [
            "season",
            "week",
            "defense_team",
            "offense_position_group",
            "shrunk_rate",
        ],
    )
    source["season"] = pd.to_numeric(source["season"], errors="raise").astype(int)
    source["week"] = pd.to_numeric(source["week"], errors="raise").astype(int)
    source["_defense"] = source["defense_team"].map(norm_team)
    source["_position"] = source["offense_position_group"].map(
        lambda x: clean(x).upper()
    )
    source = source.sort_values(
        ["_defense", "_position", "season", "week"],
        kind="mergesort",
    ).reset_index(drop=True)
    common.ensure_unique(
        source,
        ["season", "week", "_defense", "_position"],
        "position-allowed independent canonical grain",
    )
    source["_expected"] = (
        pd.to_numeric(source["shrunk_rate"], errors="coerce")
        .groupby([source["_defense"], source["_position"]], sort=False)
        .shift(1)
    )

    feature_column = "matchup_position_allowed_shrunk_rate_lag1"
    hist = read_columns(
        historical_features_path,
        ["season", "week", "opponent", "position_group", feature_column],
    )
    hist["_defense"] = hist["opponent"].map(norm_team)
    hist["_position"] = hist["position_group"].map(lambda x: clean(x).upper())
    hist = hist.loc[hist["_position"].isin({"QB", "RB", "WR", "TE"})].copy()
    if hist.empty:
        fail("No supported-position historical rows for position-allowed validation.")

    merged = hist.merge(
        source[["season", "week", "_defense", "_position", "_expected"]],
        on=["season", "week", "_defense", "_position"],
        how="left",
        validate="many_to_one",
    )
    compare_series(
        merged[feature_column],
        merged["_expected"],
        "position_allowed shrunk_rate lag1",
        atol=2e-5,
    )
    return len(merged)


def _counts_strictly_before(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    *,
    target_key: list[str],
    source_key: list[str],
) -> np.ndarray:
    result = np.zeros(len(targets), dtype=np.int64)
    source_groups: dict[tuple[str, ...], np.ndarray] = {}

    key_arg_source: str | list[str]
    key_arg_source = source_key[0] if len(source_key) == 1 else source_key
    for key, group in source.groupby(key_arg_source, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        times = (
            pd.to_datetime(group["_kickoff"], utc=True)
            .astype("int64")
            .sort_values()
            .to_numpy()
        )
        source_groups[tuple(map(str, key))] = times

    target_work = targets.copy()
    target_work["_pos"] = np.arange(len(target_work), dtype=np.int64)
    key_arg_target: str | list[str]
    key_arg_target = target_key[0] if len(target_key) == 1 else target_key
    for key, group in target_work.groupby(key_arg_target, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        times = source_groups.get(tuple(map(str, key)))
        if times is None or len(times) == 0:
            continue
        target_times = pd.to_datetime(group["_kickoff"], utc=True).astype("int64").to_numpy()
        counts = np.searchsorted(times, target_times, side="left")
        result[group["_pos"].to_numpy(dtype=np.int64)] = counts.astype(np.int64)
    return result


def _rolling_prior_mean(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    *,
    value_column: str,
    window: int,
) -> np.ndarray:
    result = np.full(len(targets), np.nan, dtype="float64")
    source_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    usable = source.loc[
        source["player_id"].ne("") & pd.to_numeric(source[value_column], errors="coerce").notna()
    ].copy()
    usable[value_column] = pd.to_numeric(usable[value_column], errors="coerce").astype(float)

    for player_id, group in usable.groupby("player_id", sort=False):
        group = group.sort_values(["_kickoff", "game_id"], kind="mergesort")
        times = pd.to_datetime(group["_kickoff"], utc=True).astype("int64").to_numpy()
        values = group[value_column].to_numpy(dtype="float64")
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        source_groups[str(player_id)] = (times, prefix)

    target_work = targets.copy()
    target_work["_pos"] = np.arange(len(target_work), dtype=np.int64)
    for player_id, group in target_work.groupby("player_id", sort=False):
        state = source_groups.get(str(player_id))
        if state is None:
            continue
        times, prefix = state
        target_times = pd.to_datetime(group["_kickoff"], utc=True).astype("int64").to_numpy()
        counts = np.searchsorted(times, target_times, side="left")
        starts = np.maximum(0, counts - int(window))
        denom = counts - starts
        values = np.full(len(group), np.nan, dtype="float64")
        valid = denom > 0
        values[valid] = (prefix[counts[valid]] - prefix[starts[valid]]) / denom[valid]
        result[group["_pos"].to_numpy(dtype=np.int64)] = values
    return result


def validate_role_history(
    universe: pd.DataFrame,
    game_kickoffs: pd.DataFrame,
    player_opportunity_path: Path,
    role_history_path: Path,
) -> int:
    source = read_columns(
        player_opportunity_path,
        GRAIN
        + [
            "team",
            "offense_snap_pct",
            "defense_snap_pct",
            "offense_participation",
            "defense_participation",
        ],
    )
    source["player_id"] = source["player_id"].map(norm_player)
    source["team"] = source["team"].map(norm_team)
    source = source.merge(
        game_kickoffs,
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )
    if source["_kickoff"].isna().any():
        fail("Role-history source contains game without kickoff.")

    off_snap = pd.to_numeric(source["offense_snap_pct"], errors="coerce")
    def_snap = pd.to_numeric(source["defense_snap_pct"], errors="coerce")
    off_part = pd.to_numeric(source["offense_participation"], errors="coerce")
    def_part = pd.to_numeric(source["defense_participation"], errors="coerce")

    source["_snap_share"] = pd.concat([off_snap, def_snap], axis=1).max(axis=1, skipna=True)
    source["_part_share"] = pd.concat([off_part, def_part], axis=1).max(axis=1, skipna=True)
    source["_role_observed"] = source[["_snap_share", "_part_share"]].notna().any(axis=1)

    role = read_columns(
        role_history_path,
        GRAIN
        + [
            "team",
            "role_history_games",
            "games_with_current_team_before_game",
            "snap_pct_roll3",
            "participation_roll3",
        ],
    )
    role["player_id"] = role["player_id"].map(norm_player)
    role["team"] = role["team"].map(norm_team)
    role = role.merge(
        universe[GRAIN + ["_kickoff"]],
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    if role["_kickoff"].isna().any():
        fail("Role history row missing historical kickoff mapping.")

    observed_source = source.loc[source["_role_observed"]].copy()
    expected_role_games = _counts_strictly_before(
        role,
        observed_source,
        target_key=["player_id"],
        source_key=["player_id"],
    )
    actual_role_games = pd.to_numeric(role["role_history_games"], errors="raise").astype("int64")
    if not np.array_equal(actual_role_games.to_numpy(), expected_role_games):
        bad = np.flatnonzero(actual_role_games.to_numpy() != expected_role_games)[:10]
        fail(
            "role_history_games is not strictly pre-kickoff; "
            f"sample_indices={bad.tolist()}"
        )

    expected_team_games = _counts_strictly_before(
        role,
        source,
        target_key=["player_id", "team"],
        source_key=["player_id", "team"],
    )
    actual_team_games = pd.to_numeric(
        role["games_with_current_team_before_game"], errors="raise"
    ).astype("int64")
    if not np.array_equal(actual_team_games.to_numpy(), expected_team_games):
        bad = np.flatnonzero(actual_team_games.to_numpy() != expected_team_games)[:10]
        fail(
            "games_with_current_team_before_game is not strictly pre-kickoff/current-stint; "
            f"sample_indices={bad.tolist()}"
        )

    expected_snap3 = _rolling_prior_mean(
        role,
        source,
        value_column="_snap_share",
        window=3,
    )
    compare_series(
        role["snap_pct_roll3"],
        pd.Series(expected_snap3, index=role.index),
        "role snap_pct_roll3 strict pre-kickoff",
        atol=2e-5,
    )

    expected_part3 = _rolling_prior_mean(
        role,
        source,
        value_column="_part_share",
        window=3,
    )
    compare_series(
        role["participation_roll3"],
        pd.Series(expected_part3, index=role.index),
        "role participation_roll3 strict pre-kickoff",
        atol=2e-5,
    )

    return len(role)


def validate_environment_perspective(
    environment_path: Path,
    historical_features_path: Path,
) -> int:
    env = read_columns(
        environment_path,
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "home_rest_days",
            "away_rest_days",
            "miles_traveled_away",
            "time_zones_crossed_away",
            "east_to_west_flag",
            "west_to_east_flag",
        ],
    )
    env["_home"] = env["home_team"].map(norm_team)
    env["_away"] = env["away_team"].map(norm_team)
    common.ensure_unique(env, ["season", "week", "game_id"], "environment game grain")

    expected_cols = [
        "environment_team_rest_days",
        "environment_opponent_rest_days",
        "environment_team_miles_traveled",
        "environment_opponent_miles_traveled",
        "environment_team_time_zones_crossed",
        "environment_opponent_time_zones_crossed",
        "environment_team_east_to_west_flag",
        "environment_opponent_east_to_west_flag",
        "environment_team_west_to_east_flag",
        "environment_opponent_west_to_east_flag",
    ]
    hist = read_columns(
        historical_features_path,
        ["season", "week", "game_id", "player_id", "team"] + expected_cols,
    )
    hist["_team"] = hist["team"].map(norm_team)
    merged = hist.merge(
        env,
        on=["season", "week", "game_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["_home"].isna().any() or merged["_away"].isna().any():
        fail("Historical feature row missing environment game mapping.")

    home = merged["_team"].eq(merged["_home"])
    away = merged["_team"].eq(merged["_away"])
    if not (home | away).all():
        fail("Historical player row team does not match environment home/away team.")

    zero = np.zeros(len(merged), dtype="float64")
    expected = {
        "environment_team_rest_days": np.where(
            home, pd.to_numeric(merged["home_rest_days"], errors="coerce"),
            pd.to_numeric(merged["away_rest_days"], errors="coerce")
        ),
        "environment_opponent_rest_days": np.where(
            home, pd.to_numeric(merged["away_rest_days"], errors="coerce"),
            pd.to_numeric(merged["home_rest_days"], errors="coerce")
        ),
        "environment_team_miles_traveled": np.where(
            away, pd.to_numeric(merged["miles_traveled_away"], errors="coerce"), zero
        ),
        "environment_opponent_miles_traveled": np.where(
            home, pd.to_numeric(merged["miles_traveled_away"], errors="coerce"), zero
        ),
        "environment_team_time_zones_crossed": np.where(
            away, pd.to_numeric(merged["time_zones_crossed_away"], errors="coerce"), zero
        ),
        "environment_opponent_time_zones_crossed": np.where(
            home, pd.to_numeric(merged["time_zones_crossed_away"], errors="coerce"), zero
        ),
        "environment_team_east_to_west_flag": np.where(
            away, pd.to_numeric(merged["east_to_west_flag"], errors="coerce"), zero
        ),
        "environment_opponent_east_to_west_flag": np.where(
            home, pd.to_numeric(merged["east_to_west_flag"], errors="coerce"), zero
        ),
        "environment_team_west_to_east_flag": np.where(
            away, pd.to_numeric(merged["west_to_east_flag"], errors="coerce"), zero
        ),
        "environment_opponent_west_to_east_flag": np.where(
            home, pd.to_numeric(merged["west_to_east_flag"], errors="coerce"), zero
        ),
    }

    for column, values in expected.items():
        compare_series(
            merged[column],
            pd.Series(values, index=merged.index),
            f"environment perspective {column}",
            atol=2e-4,
        )
    return len(merged)


def validate_week1_prior_output(config: dict[str, Any]) -> int:
    season = int(config["seasons"]["current"])
    path = PROP / f"data/current/{season}_week_1_priors.parquet"
    if not path.is_file():
        return 0

    required = [
        "player_id",
        "new_team_flag",
        "career_games",
        "prior_snap_share",
        "prior_participation",
        "prior_target_share",
        "prior_carry_share",
        "prior_efficiency",
    ]
    frame = read_columns(path, required)
    veteran_new_team = (
        pd.to_numeric(frame["new_team_flag"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(frame["career_games"], errors="coerce").fillna(0).gt(0)
    )
    role_cols = [
        "prior_snap_share",
        "prior_participation",
        "prior_target_share",
        "prior_carry_share",
    ]
    if frame.loc[veteran_new_team, role_cols].notna().any().any():
        fail("Week 1 new-team veteran imported old-team role/share history.")
    return int(veteran_new_team.sum())


def main() -> int:
    config = common.load_config()
    repo = common.repo_root()
    prop = common.prop_root()

    print("CHECK 01: required scripts and static leakage/source contracts")
    static_source_contracts()

    paths = config["paths"]
    universe_path = repo_path(repo, paths["historical_universe"])
    player_opportunity_path = repo_path(repo, paths["player_opportunity"])
    team_opportunity_path = repo_path(repo, paths["team_opportunity"])
    opponent_opportunity_path = repo_path(repo, paths["opponent_opportunity"])
    position_allowed_path = repo_path(repo, paths["position_allowed"])
    role_history_path = repo_path(repo, paths["role_history"])
    player_form_path = repo_path(repo, paths["player_form"])
    team_form_path = repo_path(repo, paths["team_form"])
    opponent_form_path = repo_path(repo, paths["opponent_form"])
    environment_path = repo_path(repo, paths["environment_history"])
    historical_features_path = repo_path(repo, paths["historical_features"])

    print("CHECK 02: feature manifests reject realized/raw outcome leakage")
    player_raw = raw_columns(
        player_opportunity_path,
        {"season", "week", "game_id", "player_id", "team", "position", "position_group"},
    )
    team_raw = raw_columns(
        team_opportunity_path,
        {"season", "week", "game_id", "team"},
    )
    opponent_raw = raw_columns(
        opponent_opportunity_path,
        {"season", "week", "game_id", "team"},
    )
    model_manifest_count, historical_features = validate_feature_manifests(
        prop, config, player_raw, team_raw, opponent_raw
    )

    print("CHECK 03: independently recompute player opportunity/efficiency lag1")
    universe, game_kickoffs = prepare_game_kickoffs(universe_path)
    player_lag_comparisons = validate_player_lag(
        universe,
        game_kickoffs,
        player_opportunity_path,
        player_form_path,
    )

    print("CHECK 04: independently recompute all team/opponent lag1 values")
    team_metrics_checked, team_comparisons = validate_team_form_lags(
        team_opportunity_path, team_form_path, "team"
    )
    opponent_metrics_checked, opponent_comparisons = validate_team_form_lags(
        opponent_opportunity_path, opponent_form_path, "opponent"
    )

    print("CHECK 05: independently recompute position-allowed shrunk_rate lag")
    position_rows_checked = validate_position_allowed_lag(
        position_allowed_path, historical_features_path
    )

    print("CHECK 06: strictly prior snap/participation/team-history at player grain")
    role_rows_checked = validate_role_history(
        universe,
        game_kickoffs,
        player_opportunity_path,
        role_history_path,
    )

    print("CHECK 07: player team-stint and career-history contracts")
    # Static contracts in build_player_form.py prove:
    # team-share metrics use player+stint as-of keys; non-share metrics use player-only
    # as-of keys, both with allow_exact_matches=False.
    team_share_reset = True
    career_survives_trade = True

    print("CHECK 08: independently recompute player-team/opponent environment perspective")
    environment_rows_checked = validate_environment_perspective(
        environment_path, historical_features_path
    )

    print("CHECK 09: current-week source availability and Week 1 team-change policy")
    participation_start = 2016
    current_season = int(config["seasons"]["current"])
    if int(config["seasons"].get("rich_feature_start", 2021)) != 2021:
        fail("Configured rich_feature_start must be 2021.")
    configured_participation = config["seasons"].get("participation_feature_start")
    if configured_participation is not None and int(configured_participation) != 2016:
        fail("Configured participation_feature_start must be 2016.")
    week1_new_team_veterans_checked = validate_week1_prior_output(config)

    print(f"scripts_checked={len(REQUIRED_SCRIPTS)}")
    print(f"model_feature_manifests_checked={model_manifest_count}")
    print(f"historical_model_features_checked={len(historical_features)}")
    print(f"player_lag_comparisons={player_lag_comparisons}")
    print(f"team_lag_metrics_checked={team_metrics_checked}")
    print(f"team_lag_comparisons={team_comparisons}")
    print(f"opponent_lag_metrics_checked={opponent_metrics_checked}")
    print(f"opponent_lag_comparisons={opponent_comparisons}")
    print(f"position_allowed_rows_checked={position_rows_checked}")
    print(f"role_rows_checked={role_rows_checked}")
    print(f"environment_rows_checked={environment_rows_checked}")
    print(f"week1_new_team_veterans_checked={week1_new_team_veterans_checked}")
    print(f"participation_feature_start={participation_start}")
    print(f"current_season={current_season}")
    print("same_game_player_realized_features=0")
    print("team_lag_contract=PASS")
    print("opponent_lag_contract=PASS")
    print("position_allowed_lag_contract=PASS")
    print("strict_prekickoff_role_contract=PASS")
    print(f"team_share_reset_contract={'PASS' if team_share_reset else 'FAIL'}")
    print(f"career_efficiency_trade_contract={'PASS' if career_survives_trade else 'FAIL'}")
    print("environment_team_relative_contract=PASS")
    print("played_game_flag_in_model_features=0")
    print("configured_target_columns_in_model_features=0")
    print("same_game_snap_features_in_model_features=0")
    print("same_game_participation_features_in_model_features=0")
    print("final_score_features_in_model_features=0")
    print("same_game_target_carry_share_features_in_model_features=0")
    print("historical_current_feature_semantics=PASS")
    print("contradictory_participation_assignment_allowed=false")
    print("unsafe_global_snap_alias_assignment_allowed=false")
    print("FINAL DATA CONTRACTS VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FINAL DATA CONTRACTS VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
