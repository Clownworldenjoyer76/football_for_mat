#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROP = Path(__file__).resolve().parents[1]
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common


def load_module(name: str, relative: str):
    path = PROP / "scripts" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


current_features = load_module(
    "issue47_e2e_current_features",
    "project/build_current_features.py",
)
project_direct = load_module(
    "issue47_e2e_project_direct",
    "project/project_direct.py",
)


class EndToEndHistoricalAsCurrentTests(unittest.TestCase):
    """Treat one 2025 historical player-week as a current as-of state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = common.load_config()
        cls.repo = common.repo_root()
        cls.historical_path = cls.repo / cls.config["paths"]["historical_features"]
        cls.season = 2025

    def choose_historical_qb_state(self) -> tuple[pd.Series, pd.DataFrame]:
        columns = [
            "season", "week", "game_id", "player_id", "team", "position",
            "player_pass_attempts_lag1",
            "player_pass_attempts_season_to_date",
        ]
        frame = pd.read_parquet(
            self.historical_path,
            columns=columns,
            filters=[("season", "=", self.season)],
        )
        candidates = frame.loc[
            frame["position"].astype(str).str.upper().eq("QB")
            & frame["week"].gt(1)
            & pd.to_numeric(
                frame["player_pass_attempts_lag1"], errors="coerce"
            ).notna()
            & pd.to_numeric(
                frame["player_pass_attempts_season_to_date"], errors="coerce"
            ).notna()
        ].sort_values(["week", "game_id", "player_id"], kind="mergesort")
        self.assertGreater(len(candidates), 0)

        # The historical player-form builder consumes canonical player_opportunity.
        # Use that same football-only source as the simulated current player-stat
        # feed so this test compares equivalent as-of states rather than relying
        # on a raw-source identity representation that may require the historical
        # identity builder first.
        opportunity = pd.read_parquet(
            self.repo / self.config["paths"]["player_opportunity"],
            columns=["season", "week", "player_id", "pass_attempts"],
        )
        opportunity = opportunity.loc[
            pd.to_numeric(opportunity["season"], errors="coerce").eq(self.season)
        ].copy()
        opportunity["player_id"] = opportunity["player_id"].map(
            common.normalize_player_id
        )
        opportunity["week"] = pd.to_numeric(
            opportunity["week"], errors="coerce"
        )
        opportunity["pass_attempts"] = pd.to_numeric(
            opportunity["pass_attempts"], errors="coerce"
        )

        for _, candidate in candidates.iterrows():
            player_id = common.normalize_player_id(candidate["player_id"])
            target_week = int(candidate["week"])
            prior = opportunity.loc[
                opportunity["player_id"].eq(player_id)
                & opportunity["week"].lt(target_week)
                & opportunity["pass_attempts"].notna()
            ].sort_values("week", kind="mergesort")
            if prior.empty:
                continue

            expected_lag = float(prior.iloc[-1]["pass_attempts"])
            expected_std = float(prior["pass_attempts"].mean())
            historical_lag = float(candidate["player_pass_attempts_lag1"])
            historical_std = float(
                candidate["player_pass_attempts_season_to_date"]
            )
            if (
                np.isclose(expected_lag, historical_lag, rtol=1e-6, atol=1e-6)
                and np.isclose(
                    expected_std, historical_std, rtol=1e-6, atol=1e-6
                )
            ):
                return candidate, opportunity

        self.fail(
            "No 2025 QB fixture had equivalent canonical historical and "
            "strict-prior current-style pass-attempt state."
        )

    def test_historical_week_as_current_feature_projection_validation(self) -> None:
        chosen, stats = self.choose_historical_qb_state()
        week = int(chosen["week"])
        player_id = common.normalize_player_id(chosen["player_id"])

        # Build current-style as-of features from the same canonical historical
        # opportunity state. overlay_current_player_stats itself enforces
        # source week < projection week.
        shadow_current = pd.DataFrame(
            {
                "player_id": [player_id],
                "player_pass_attempts_lag1": [np.nan],
                "player_pass_attempts_season_to_date": [np.nan],
            }
        )
        audit = current_features.overlay_current_player_stats(
            shadow_current,
            stats,
            week,
        )
        self.assertGreater(audit["rows_used"], 0)

        historical_lag = float(chosen["player_pass_attempts_lag1"])
        historical_std = float(chosen["player_pass_attempts_season_to_date"])
        current_lag = float(shadow_current.loc[0, "player_pass_attempts_lag1"])
        current_std = float(
            shadow_current.loc[0, "player_pass_attempts_season_to_date"]
        )
        self.assertAlmostEqual(current_lag, historical_lag, places=5)
        self.assertAlmostEqual(current_std, historical_std, places=5)

        # Project the historical-as-current row through the persisted direct
        # passing-yards model. That model is trained only through 2024.
        target = "passing_yards"
        model_dir = PROP / "models" / target
        manifest = json.loads(
            (model_dir / "feature_manifest.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        numeric = list(manifest.get("numeric_features", []))
        categorical = list(manifest.get("categorical_features", []))
        feature_columns = [*numeric, *categorical]
        self.assertGreater(len(feature_columns), 0)

        row = pd.read_parquet(
            self.historical_path,
            columns=[
                "season", "week", "game_id", "player_id", "team",
                *feature_columns,
            ],
            filters=[("season", "=", self.season), ("week", "=", week)],
        )
        row = row.loc[
            row["player_id"].map(common.normalize_player_id).eq(player_id)
        ].head(1)
        self.assertEqual(len(row), 1)
        historical_model_row = row.copy()

        # Rebuild the representative current/as-of fields on the exact row that
        # will be scored. This is the historical week acting as a current week.
        for column in [
            "player_pass_attempts_lag1",
            "player_pass_attempts_season_to_date",
        ]:
            if column in row.columns:
                row.loc[:, column] = np.nan
        current_features.overlay_current_player_stats(row, stats, week)
        self.assertAlmostEqual(
            float(row.iloc[0]["player_pass_attempts_lag1"]),
            float(historical_model_row.iloc[0]["player_pass_attempts_lag1"]),
            places=5,
        )
        self.assertAlmostEqual(
            float(row.iloc[0]["player_pass_attempts_season_to_date"]),
            float(historical_model_row.iloc[0]["player_pass_attempts_season_to_date"]),
            places=5,
        )

        # Rebuild additional current-style context when those fields participate
        # in the selected direct model. These comparisons are conditional only
        # on the model schema; no future rows are ever eligible.
        team_stats_rel = self.config["paths"]["team_stats_pattern"].format(
            season=self.season
        )
        team_stats_path = self.repo / team_stats_rel
        if team_stats_path.is_file():
            team_stats = pd.read_csv(team_stats_path, low_memory=False)
            current_features.overlay_current_team_stats(row, team_stats, week)
            for column in [
                "team_pass_attempts_lag1",
                "team_pass_attempts_season_to_date",
            ]:
                if column in row.columns:
                    a = pd.to_numeric(row[column], errors="coerce").iloc[0]
                    h = pd.to_numeric(
                        historical_model_row[column], errors="coerce"
                    ).iloc[0]
                    if pd.notna(a) and pd.notna(h):
                        self.assertAlmostEqual(float(a), float(h), places=5)

        snaps_rel = self.config["paths"]["historical_snaps_pattern"].format(
            season=self.season
        )
        snaps_path = self.repo / snaps_rel
        if snaps_path.is_file() and "role_prior_offense_snap_pct" in row.columns:
            snaps = pd.read_parquet(snaps_path)
            current_features.overlay_current_snaps(row, snaps, week)
            a = pd.to_numeric(
                row["role_prior_offense_snap_pct"], errors="coerce"
            ).iloc[0]
            h = pd.to_numeric(
                historical_model_row["role_prior_offense_snap_pct"],
                errors="coerce",
            ).iloc[0]
            if pd.notna(a) and pd.notna(h):
                self.assertAlmostEqual(float(a), float(h), places=5)

        common.reject_forbidden_feature_columns(feature_columns, self.config)
        levels = manifest.get("categorical_levels_final_through_2024", {})
        self.assertIsInstance(levels, dict)
        self.assertEqual(
            set(categorical),
            set(levels),
            msg="Persisted categorical levels do not cover the direct-model schema",
        )
        matrix = project_direct.model_matrix(row, numeric, categorical, levels)
        booster = lgb.Booster(model_file=str(model_dir / "direct_model.txt"))
        self.assertEqual(list(booster.feature_name()), feature_columns)
        prediction = float(np.asarray(booster.predict(matrix), dtype="float64")[0])
        projection = max(0.0, prediction)

        # Validation for this isolated historical-as-current projection.
        self.assertTrue(np.isfinite(projection))
        self.assertGreaterEqual(projection, 0.0)
        self.assertEqual(
            project_direct.feature_hash(numeric, categorical),
            manifest["feature_hash"],
        )


if __name__ == "__main__":
    unittest.main()
