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


roles = load_module("issue47_select_roles", "project/select_roles.py")


def qb_frame(qb1_status: str = "eligible") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "BUF", "player_id": "qb1", "player_name": "QB One",
                "position": "QB", "depth_rank": 1, "depth_starter_flag": 1,
                "eligibility_status": qb1_status, "injury_game_status": "",
                "depth_injury": "",
            },
            {
                "team": "BUF", "player_id": "qb2", "player_name": "QB Two",
                "position": "QB", "depth_rank": 2, "depth_starter_flag": 0,
                "eligibility_status": "eligible", "injury_game_status": "",
                "depth_injury": "",
            },
        ]
    )


class RoleSelectionTests(unittest.TestCase):
    def test_healthy_qb1_selected(self) -> None:
        player_id, confidence, reason, _ = roles.select_primary_qb(qb_frame())
        self.assertEqual(player_id, "qb1")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(reason, "healthy_depth_rank_1_qb")

    def test_backup_promoted_when_qb1_ineligible(self) -> None:
        player_id, confidence, reason, _ = roles.select_primary_qb(qb_frame("ineligible"))
        self.assertEqual(player_id, "qb2")
        self.assertGreaterEqual(confidence, 0.80)
        self.assertEqual(reason, "qb_promoted_after_higher_depth_qb_ineligible")

    def test_qb_depth_ambiguity_hard_fails(self) -> None:
        frame = qb_frame()
        frame.loc[:, "depth_rank"] = 1
        frame.loc[:, "depth_starter_flag"] = 0
        with self.assertRaises(RuntimeError):
            roles.select_primary_qb(frame)


if __name__ == "__main__":
    unittest.main()
