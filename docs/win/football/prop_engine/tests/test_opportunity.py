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


class OpportunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = common.load_config()
        repo = common.repo_root()
        cls.player = pd.read_parquet(repo / config["paths"]["player_opportunity"])
        cls.team = pd.read_parquet(repo / config["paths"]["team_opportunity"])

    def test_player_opportunity_grain_unique(self) -> None:
        common.ensure_unique(
            self.player,
            ["season", "week", "game_id", "player_id"],
            "player opportunity",
        )

    def test_share_features_bounded(self) -> None:
        share_columns = [
            "carry_share", "target_share", "red_zone_target_share",
            "goal_line_carry_share", "offense_snap_pct", "defense_snap_pct",
            "offense_participation", "defense_participation",
        ]
        present = [c for c in share_columns if c in self.player.columns]
        self.assertGreater(len(present), 0)
        for column in present:
            values = pd.to_numeric(self.player[column], errors="coerce")
            invalid = values.notna() & ~values.between(0.0, 1.0)
            self.assertFalse(bool(invalid.any()), msg=f"{column} outside [0,1]")

    def test_opportunity_volumes_nonnegative(self) -> None:
        player_counts = [
            "pass_attempts", "dropbacks", "carries", "targets",
            "red_zone_targets", "red_zone_carries", "goal_line_carries",
            "field_goal_attempts", "extra_point_attempts",
        ]
        team_counts = [
            "offensive_plays", "dropbacks", "pass_attempts", "rush_attempts",
            "field_goal_attempts", "extra_point_attempts",
        ]
        for frame, columns in ((self.player, player_counts), (self.team, team_counts)):
            for column in [c for c in columns if c in frame.columns]:
                values = pd.to_numeric(frame[column], errors="coerce")
                self.assertFalse(
                    bool((values.notna() & values.lt(0.0)).any()),
                    msg=f"{column} contains negative opportunity",
                )

    def test_no_infinite_opportunity_values(self) -> None:
        numeric = self.player.select_dtypes(include=["number"])
        self.assertFalse(
            np.isinf(numeric.to_numpy(dtype="float64", copy=False)).any()
        )


if __name__ == "__main__":
    unittest.main()
