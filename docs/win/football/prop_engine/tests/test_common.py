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


class CommonTests(unittest.TestCase):
    def test_team_aliases(self) -> None:
        self.assertEqual(common.normalize_team("WAS"), "WSH")
        self.assertEqual(common.normalize_team("LA"), "LAR")
        self.assertEqual(common.normalize_team("JAC"), "JAX")
        self.assertEqual(common.normalize_team("BUF"), "BUF")

    def test_player_id_normalization(self) -> None:
        self.assertEqual(common.normalize_player_id(123.0), "123")
        self.assertEqual(common.normalize_player_id("00-0031234"), "00-0031234")
        self.assertEqual(common.normalize_player_id(None), "")

    def test_name_normalization(self) -> None:
        self.assertEqual(common.normalize_name("José  Núñez Jr."), "jose nunez jr")

    def test_game_id_normalization(self) -> None:
        self.assertEqual(
            common.parse_game_id("2024_1_JAC_WAS"),
            "2024_01_JAX_WSH",
        )

    def test_duplicate_grain_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [1, 1],
                "game_id": ["2024_01_BUF_MIA", "2024_01_BUF_MIA"],
                "player_id": ["00-0000001", "00-0000001"],
            }
        )
        with self.assertRaises(ValueError):
            common.ensure_unique(
                frame,
                ["season", "week", "game_id", "player_id"],
                "unit-test grain",
            )


if __name__ == "__main__":
    unittest.main()
