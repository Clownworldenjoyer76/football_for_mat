#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

PROP = Path(__file__).resolve().parents[1]
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, relative: str):
    path = PROP / "scripts" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


player_form = load_module("issue47_player_form", "build/build_player_form.py")
role_history = load_module("issue47_role_history", "build/build_role_history.py")
team_form = load_module("issue47_team_form", "build/build_team_form.py")
universe = load_module("issue47_universe", "build/build_historical_universe.py")


class LeakageTests(unittest.TestCase):
    def test_week_n_player_rolling_feature_excludes_week_n(self) -> None:
        t1 = pd.Timestamp("2024-09-01T17:00:00Z")
        t2 = pd.Timestamp("2024-09-08T17:00:00Z")
        t3 = pd.Timestamp("2024-09-15T17:00:00Z")
        source = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "player_id": ["p1", "p1", "p1"],
                "position_group": ["QB", "QB", "QB"],
                "_kickoff": [t1, t2, t3],
                "_stint_id": [1, 1, 1],
                "pass_attempts": [10.0, 20.0, 999.0],
            }
        )
        target = pd.DataFrame(
            {
                "_row_id": [0],
                "season": [2024],
                "player_id": ["p1"],
                "position_group": ["QB"],
                "_kickoff": [t3],
            }
        )
        block = player_form.metric_features_for_targets(target, source, "pass_attempts")
        names = [f"pass_attempts_{suffix}" for suffix in player_form.FEATURE_SUFFIXES]
        values = dict(zip(names, block[0]))
        self.assertEqual(values["pass_attempts_lag1"], 20.0)
        self.assertEqual(values["pass_attempts_roll3_mean"], 15.0)
        self.assertNotEqual(values["pass_attempts_roll3_mean"], (10 + 20 + 999) / 3)

    def _role_fixture(self):
        target_time = pd.Timestamp("2024-09-15T17:00:00Z")
        base = pd.DataFrame(
            {
                "player_id": ["p1"],
                "game_id": ["g3"],
                "team": ["BUF"],
                "_kickoff": [target_time],
            }
        )
        history = pd.DataFrame(
            {
                "player_id": ["p1", "p1"],
                "game_id": ["g2", "g3"],
                "team": ["BUF", "BUF"],
                "_kickoff": [pd.Timestamp("2024-09-08T17:00:00Z"), target_time],
                "_role_observed": [True, True],
                "_snap_share": [0.40, 1.00],
                "_participation_share": [0.50, 1.00],
            }
        )
        return base, history

    def test_week_n_snap_excludes_week_n_snap(self) -> None:
        base, history = self._role_fixture()
        out = role_history.add_strict_prior_history(base, history)
        self.assertAlmostEqual(float(out.loc[0, "snap_pct_roll3"]), 0.40)
        self.assertNotEqual(float(out.loc[0, "snap_pct_roll3"]), 1.00)

    def test_week_n_participation_excludes_week_n_participation(self) -> None:
        base, history = self._role_fixture()
        out = role_history.add_strict_prior_history(base, history)
        self.assertAlmostEqual(float(out.loc[0, "participation_roll3"]), 0.50)
        self.assertNotEqual(float(out.loc[0, "participation_roll3"]), 1.00)

    def test_week_n_team_form_excludes_week_n_result(self) -> None:
        source = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [1, 2, 3],
                "team": ["BUF", "BUF", "BUF"],
                "offensive_plays": [60.0, 70.0, 999.0],
            }
        )
        out = team_form.build_form(source, ["offensive_plays"], label="unit-team")
        row = out.loc[out["week"].eq(3)].iloc[0]
        self.assertEqual(float(row["offensive_plays_lag1"]), 70.0)
        self.assertEqual(float(row["offensive_plays_roll3_mean"]), 65.0)

    def test_depth_snapshot_precedes_kickoff(self) -> None:
        game = {"gameday": "2024-09-15"}
        cutoff = universe.conservative_depth_cutoff(game)
        prior = {"p1": {"depth_rank": 1}}
        at_cutoff = {"p1": {"depth_rank": 99}}
        timestamped = {
            "BUF": [
                (cutoff - pd.Timedelta(hours=1), prior),
                (cutoff, at_cutoff),
            ]
        }
        selected = universe.depth_snapshot_for_game(
            mode="timestamp",
            weekly={},
            timestamped=timestamped,
            team="BUF",
            week=3,
            game=game,
        )
        self.assertEqual(selected, prior)
        kickoff = pd.Timestamp("2024-09-15T17:00:00Z")
        self.assertLess(cutoff - pd.Timedelta(hours=1), kickoff)

    def test_injury_snapshot_precedes_kickoff(self) -> None:
        source_weeks = {2024: [1, 3]}
        selected_week = role_history.resolve_injury_source_week(
            season=2024,
            target_week=2,
            source_weeks=source_weeks,
        )
        self.assertEqual(selected_week, 1)
        self.assertLess(selected_week, 2)
        injury_snapshot = role_history.parse_modified("2024-09-06T18:00:00Z")
        kickoff = pd.Timestamp("2024-09-08T17:00:00Z")
        self.assertIsNotNone(injury_snapshot)
        self.assertLess(injury_snapshot, kickoff)


if __name__ == "__main__":
    unittest.main()
