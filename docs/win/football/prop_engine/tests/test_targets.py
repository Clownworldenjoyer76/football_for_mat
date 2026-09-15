#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROP = Path(__file__).resolve().parents[1]
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = common.load_config()
        repo = common.repo_root()
        cls.grain = ["season", "week", "game_id", "player_id"]
        cls.universe = pd.read_parquet(
            repo / config["paths"]["historical_universe"],
            columns=[*cls.grain, "played_game_flag"],
        )
        cls.targets = pd.read_parquet(repo / config["paths"]["historical_targets"])
        common.ensure_unique(cls.universe, cls.grain, "historical universe")
        common.ensure_unique(cls.targets, cls.grain, "historical targets")

    def test_kicking_points_formula_exact(self) -> None:
        expected = (
            3.0 * pd.to_numeric(self.targets["field_goals_made"], errors="coerce")
            + pd.to_numeric(self.targets["extra_points_made"], errors="coerce")
        )
        actual = pd.to_numeric(self.targets["kicking_points"], errors="coerce")
        self.assertTrue(
            np.allclose(
                actual.to_numpy(dtype="float64"),
                expected.to_numpy(dtype="float64"),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

    def test_tackles_formula_exact(self) -> None:
        expected = (
            pd.to_numeric(self.targets["solo_tackles"], errors="coerce")
            + pd.to_numeric(self.targets["assisted_tackles"], errors="coerce")
        )
        actual = pd.to_numeric(self.targets["tackles"], errors="coerce")
        self.assertTrue(
            np.allclose(
                actual.to_numpy(dtype="float64"),
                expected.to_numpy(dtype="float64"),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

    def joined(self) -> pd.DataFrame:
        return self.universe.merge(
            self.targets,
            on=self.grain,
            how="inner",
            validate="one_to_one",
        )

    def test_zero_stat_participant_retained(self) -> None:
        joined = self.joined()
        mask = (
            pd.to_numeric(joined["played_game_flag"], errors="coerce").eq(1)
            & pd.to_numeric(joined["target_source_present"], errors="coerce").eq(0)
        )
        self.assertGreater(int(mask.sum()), 0)
        target_values = [
            "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds", "field_goals_made",
            "extra_points_made", "kicking_points", "solo_tackles",
            "assisted_tackles", "tackles", "sacks",
        ]
        values = joined.loc[mask, target_values].apply(pd.to_numeric, errors="coerce")
        self.assertTrue(values.notna().all().all())
        self.assertTrue(values.eq(0.0).all().all())

    def test_nonparticipant_not_converted_into_false_zero(self) -> None:
        joined = self.joined()
        mask = (
            pd.to_numeric(joined["played_game_flag"], errors="coerce").eq(0)
            & pd.to_numeric(joined["target_source_present"], errors="coerce").eq(0)
        )
        self.assertGreater(int(mask.sum()), 0)
        target_values = [
            "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds", "field_goals_made",
            "extra_points_made", "kicking_points", "solo_tackles",
            "assisted_tackles", "tackles", "sacks",
        ]
        values = joined.loc[mask, target_values].apply(pd.to_numeric, errors="coerce")
        self.assertTrue(values.isna().all().all())


if __name__ == "__main__":
    unittest.main()
