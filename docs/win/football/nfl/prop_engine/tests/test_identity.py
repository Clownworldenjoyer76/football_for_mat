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


universe = load_module(
    "issue47_build_historical_universe",
    "build/build_historical_universe.py",
)


class IdentityTests(unittest.TestCase):
    def crosswalk(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "player_id": "00-0000001",
                    "gsis_id": "00-0000001",
                    "espn_id": "101",
                    "pfr_id": "AlphaA00",
                    "display_name": "Alpha Player",
                    "normalized_name": "alpha player",
                    "position": "QB",
                    "position_group": "OFF",
                    "resolution_status": "resolved",
                },
                {
                    "player_id": "00-0000002",
                    "gsis_id": "00-0000002",
                    "espn_id": "202",
                    "pfr_id": "BetaB00",
                    "display_name": "Shared Name",
                    "normalized_name": "shared name",
                    "position": "WR",
                    "position_group": "OFF",
                    "resolution_status": "resolved",
                },
                {
                    "player_id": "00-0000003",
                    "gsis_id": "00-0000003",
                    "espn_id": "303",
                    "pfr_id": "GammaG00",
                    "display_name": "Shared Name",
                    "normalized_name": "shared name",
                    "position": "WR",
                    "position_group": "OFF",
                    "resolution_status": "resolved",
                },
            ]
        )

    def test_gsis_is_authoritative(self) -> None:
        resolver = universe.IdentityResolver(self.crosswalk())
        row = resolver.resolve(
            gsis_id="00-0000001",
            espn_id="202",
            name="Shared Name",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["player_id"], "00-0000001")

    def test_unique_alias_resolution(self) -> None:
        resolver = universe.IdentityResolver(self.crosswalk())
        self.assertEqual(resolver.resolve(espn_id="101")["player_id"], "00-0000001")
        self.assertEqual(resolver.resolve(pfr_id="AlphaA00")["player_id"], "00-0000001")

    def test_ambiguous_name_is_not_guessed(self) -> None:
        resolver = universe.IdentityResolver(self.crosswalk())
        self.assertIsNone(resolver.resolve(name="Shared Name"))

    def test_extract_gsis_ids(self) -> None:
        found = universe.extract_gsis_ids(
            "participants: 00-0000001, 00-0000002"
        )
        self.assertEqual(found, {"00-0000001", "00-0000002"})


if __name__ == "__main__":
    unittest.main()
