#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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


audit = load_module(
    "issue47_market_audit",
    "validate/audit_market_exclusion.py",
)


class MarketExclusionTests(unittest.TestCase):
    def assert_forbidden_feature(self, token: str) -> None:
        with self.assertRaises(ValueError):
            common.reject_forbidden_feature_columns(
                [f"feature_{token}"], common.load_config()
            )

    def assert_forbidden_source_root(self, forbidden_root: str) -> None:
        config = common.load_config()
        deny = audit.forbidden_source_references(config)
        self.assertIn(forbidden_root, deny)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "bad_source_reference.txt"
            bad.write_text(
                f"source={forbidden_root}artifact.parquet\n",
                encoding="utf-8",
            )
            payload = audit.audit_paths(
                [root],
                config=config,
                repo=None,
            )
            self.assertFalse(payload["passed"])
            self.assertTrue(payload["market_features_used"])
            self.assertEqual(payload["forbidden_feature_hit_count"], 0)
            self.assertGreaterEqual(
                payload["forbidden_source_reference_count"],
                1,
            )
            self.assertTrue(
                any(
                    forbidden_root in hit
                    for hit in payload["forbidden_source_references"]
                )
            )

    def test_reject_odds(self) -> None:
        self.assert_forbidden_feature("odds")

    def test_reject_spread(self) -> None:
        self.assert_forbidden_feature("spread")

    def test_reject_moneyline(self) -> None:
        self.assert_forbidden_feature("moneyline")

    def test_reject_drat(self) -> None:
        self.assert_forbidden_feature("drat")

    def test_reject_epred(self) -> None:
        self.assert_forbidden_feature("epred")

    def test_reject_forbidden_source_paths(self) -> None:
        self.assert_forbidden_source_root(
            "docs/win/football/nfl/data/historic_data/odds/"
        )

    def test_reject_training_source_root(self) -> None:
        self.assert_forbidden_source_root(
            "docs/win/football/nfl/training/"
        )

    def test_reject_legacy_01_merge_source_root(self) -> None:
        self.assert_forbidden_source_root(
            "docs/win/football/nfl/01_merge/"
        )

    def test_reject_scripts_01_merge_source_root(self) -> None:
        self.assert_forbidden_source_root(
            "docs/win/football/nfl/scripts/01_merge/"
        )

    def test_clean_audit_declares_market_features_unused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean = root / "clean.txt"
            clean.write_text(
                "source=docs/win/football/nfl/00_intake/pbp/2026.csv.gz\n",
                encoding="utf-8",
            )
            payload = audit.audit_paths(
                [root],
                config=common.load_config(),
                repo=None,
            )
            self.assertTrue(payload["passed"])
            self.assertFalse(payload["market_features_used"])
            self.assertEqual(payload["forbidden_source_reference_count"], 0)
            self.assertEqual(payload["forbidden_feature_hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
