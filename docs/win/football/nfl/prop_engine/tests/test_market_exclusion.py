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
        deny = audit.forbidden_source_references()
        forbidden_path = "docs/win/football/nfl/data/historic_data/odds/"
        self.assertIn(forbidden_path, deny)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "bad.txt"
            bad.write_text(f"source={forbidden_path}market.csv\n", encoding="utf-8")
            payload = audit.audit_paths(
                [root], config=common.load_config(), repo=None
            )
            self.assertFalse(payload["passed"])
            self.assertGreater(len(payload["forbidden_source_references"]), 0)


if __name__ == "__main__":
    unittest.main()
