#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROP = Path(__file__).resolve().parents[1]
SCRIPTS = PROP / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common


class ProjectionConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = common.load_config()
        season = int(config["seasons"]["current"])
        output = PROP / "output" / str(season) / "week_1_player_projections.csv"
        active = PROP / "output" / str(season) / "week_1_active_player_projections.csv"
        allocation = PROP / "data" / "current" / f"{season}_week_1_allocated_opportunity.parquet"
        if not output.is_file() or not active.is_file() or not allocation.is_file():
            raise AssertionError(
                "Accepted Week 1 projection/allocation artifacts are required "
                "for Issue 47 projection-constraint tests."
            )
        cls.long = pd.read_csv(output, low_memory=False)
        cls.active = pd.read_csv(active, low_memory=False)
        cls.allocation = pd.read_parquet(allocation)

    def test_no_duplicate_player_game_target(self) -> None:
        keys = ["season", "week", "game_id", "player_id", "target"]
        self.assertFalse(self.long.duplicated(keys).any())

    def test_projection_interval_constraints(self) -> None:
        tol = 1e-9
        projection = pd.to_numeric(self.long["projection"], errors="coerce")
        low = pd.to_numeric(self.long["low"], errors="coerce")
        high = pd.to_numeric(self.long["high"], errors="coerce")
        self.assertFalse(projection.isna().any())
        self.assertTrue(projection.ge(-tol).all())

        # Match the accepted Issue 38 contract: interval bounds may both be
        # unavailable for targets without an interval, but one-sided/null-
        # asymmetric intervals are invalid. When present, bounds must contain
        # the projection.
        asymmetric = low.isna() ^ high.isna()
        self.assertFalse(bool(asymmetric.any()))
        bounded = low.notna() & high.notna()
        self.assertTrue(
            (low.loc[bounded] <= projection.loc[bounded] + tol).all()
        )
        self.assertTrue(
            (projection.loc[bounded] <= high.loc[bounded] + tol).all()
        )

    def test_probability_constraints(self) -> None:
        for column in ["probability_1_plus", "probability_2_plus"]:
            values = pd.to_numeric(self.long[column], errors="coerce").dropna()
            self.assertTrue(values.between(0.0, 1.0).all())

    def test_active_output_contains_only_eligible_rows(self) -> None:
        status = self.active["eligibility_status"].fillna("").astype(str).str.casefold()
        self.assertTrue(status.eq("eligible").all())

    def test_allocated_team_shares_reconcile(self) -> None:
        for column in ["allocated_target_share", "allocated_carry_share"]:
            values = pd.to_numeric(self.allocation[column], errors="coerce").fillna(0.0)
            self.assertTrue(values.between(0.0, 1.0).all())
            work = self.allocation.assign(_share=values)
            sums = work.groupby(
                ["season", "week", "game_id", "team"], dropna=False
            )["_share"].sum()
            positive = sums.gt(0.0)
            self.assertTrue(
                (sums.loc[positive].sub(1.0).abs() <= 1e-8).all(),
                msg=f"{column} does not reconcile to 1.0",
            )


if __name__ == "__main__":
    unittest.main()
