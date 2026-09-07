#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pyarrow.parquet as pq

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
    "issue47_current_features",
    "project/build_current_features.py",
)


class FeatureSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = common.load_config()
        repo = common.repo_root()
        cls.feature_path = repo / config["paths"]["historical_features"]
        cls.manifest_path = cls.feature_path.with_name("feature_manifest.json")
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.schema = pq.read_schema(cls.feature_path)

    def test_manifest_feature_partition_exact(self) -> None:
        features = list(self.manifest["feature_columns"])
        numeric = list(self.manifest["numeric_features"])
        categorical = list(self.manifest["categorical_features"])
        self.assertEqual(len(features), len(set(features)))
        self.assertFalse(set(numeric) & set(categorical))
        self.assertEqual(set(features), set(numeric) | set(categorical))

    def test_no_target_or_audit_feature_leakage(self) -> None:
        features = list(self.manifest["feature_columns"])
        self.assertFalse(any(c.startswith("target_") for c in features))
        self.assertFalse(any(c.startswith("audit_") for c in features))
        common.reject_forbidden_feature_columns(features, common.load_config())

    def test_manifest_columns_exist_in_parquet(self) -> None:
        missing = set(self.manifest["feature_columns"]) - set(self.schema.names)
        self.assertEqual(missing, set())

    def test_direct_model_feature_hashes_match(self) -> None:
        for target in common.load_config()["targets"]:
            model_dir = PROP / "models" / target
            manifest_path = model_dir / "feature_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            numeric = list(manifest.get("numeric_features", []))
            categorical = list(manifest.get("categorical_features", []))
            calculated = current_features.manifest_feature_hash(numeric, categorical)
            stored = manifest.get("feature_hash")
            if stored is not None:
                self.assertEqual(calculated, stored, msg=f"{target} feature hash mismatch")


if __name__ == "__main__":
    unittest.main()
