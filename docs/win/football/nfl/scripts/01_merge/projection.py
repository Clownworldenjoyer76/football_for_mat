#!/usr/bin/env python3
"""Production NFL Week 2+ projection using market-independent v4 models."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import projection_feature_builder_legacy_inseason as helper
from v4_production import apply_v4_production_models

SEASON = 2026
SCRIPT_VERSION = "2026-09-17-settings-week"


def load_compatibility_schema(root: Path) -> dict:
    path = root / "models/archive/legacy_260_feature_model/step11_feature_schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature-builder compatibility schema: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_configured_week(root: Path, season: int) -> int:
    path = root / "config/settings.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing NFL settings file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    configured_season = settings.get("season")
    configured_week = settings.get("week")

    if configured_season in (None, ""):
        raise ValueError(f"{path}: missing season")
    if configured_week in (None, ""):
        raise ValueError(f"{path}: missing week")

    configured_season = int(configured_season)
    configured_week = int(configured_week)

    if configured_season != season:
        raise ValueError(
            f"{path}: configured season={configured_season} does not match projection season={season}"
        )
    if configured_week < 1:
        raise ValueError(f"{path}: week must be >= 1; found {configured_week}")

    return configured_week


def run_projection(*, season: int, week1_mode: bool = False) -> list[Path]:
    root = helper.nfl_root()
    week = 1 if week1_mode else load_configured_week(root, season)
    compatibility_schema = load_compatibility_schema(root)

    original, full_features = helper.prepare_week(
        root,
        season,
        week,
        week1_mode,
        compatibility_schema,
    )
    projected = apply_v4_production_models(root, original, full_features)

    output_dir = root / "01_merge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"week_{week}_NFL_enriched.csv"
    projected.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"WROTE {output_path} | games={len(projected)} | "
        f"model=v4_market_independent_outcomes | columns={len(projected.columns)}"
    )
    return [output_path]


def main() -> None:
    print(f"projection.py version={SCRIPT_VERSION}")
    run_projection(season=SEASON, week1_mode=False)


if __name__ == "__main__":
    main()
