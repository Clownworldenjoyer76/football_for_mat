#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

OLD = "matchup_player_sack_rate_x_opp_dropbacks"
NEW = "matchup_player_sack_rate_x_opp_plays"
OLD_DEN = "matchup_expected_opponent_dropbacks"
NEW_DEN = "matchup_expected_opponent_plays"

PROP = Path(__file__).resolve().parent

SOURCE_FILES = [
    PROP / "scripts/build/build_historical_features.py",
    PROP / "scripts/project/build_current_features.py",
    PROP / "validate_issue17.py",
    PROP / "config/features/sacks.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-rebuild", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def feature_hash(numeric: list[str], categorical: list[str]) -> str:
    payload = [
        *[{"name": x, "type": "numeric"} for x in numeric],
        *[{"name": x, "type": "categorical"} for x in categorical],
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_checks() -> None:
    for path in SOURCE_FILES:
        if not path.is_file():
            raise AssertionError(f"Missing source: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        if OLD in text:
            raise AssertionError(f"{path.name}: stale feature name remains")
        if NEW not in text:
            raise AssertionError(f"{path.name}: corrected feature name missing")

    hist = (PROP / "scripts/build/build_historical_features.py").read_text(encoding="utf-8")
    current = (PROP / "scripts/project/build_current_features.py").read_text(encoding="utf-8")
    issue17 = (PROP / "validate_issue17.py").read_text(encoding="utf-8")

    hist_anchor = hist.find(f'out["{NEW}"]')
    hist_chunk = hist[hist_anchor:hist_anchor + 500] if hist_anchor >= 0 else ""
    if NEW_DEN not in hist_chunk or OLD_DEN in hist_chunk:
        raise AssertionError("historical builder still uses opponent dropbacks")

    current_anchor = current.find(f'if exists("{NEW}"')
    current_chunk = current[current_anchor:current_anchor + 650] if current_anchor >= 0 else ""
    if NEW_DEN not in current_chunk or OLD_DEN in current_chunk:
        raise AssertionError("current builder still uses opponent dropbacks")

    issue17_anchor = issue17.find(f'expected["{NEW}"]')
    issue17_chunk = issue17[issue17_anchor:issue17_anchor + 500] if issue17_anchor >= 0 else ""
    if NEW_DEN not in issue17_chunk or OLD_DEN in issue17_chunk:
        raise AssertionError("Issue 17 validator still uses opponent dropbacks")

    config = read_json(PROP / "config/features/sacks.json")
    numeric = list(config.get("numeric_features", []))
    required = list(config.get("required_features", []))
    optional = list(config.get("optional_features", []))
    if NEW not in numeric:
        raise AssertionError("sacks feature config missing corrected numeric feature")
    if NEW not in required and NEW not in optional:
        raise AssertionError("sacks feature config does not classify corrected feature")
    if OLD in numeric + required + optional:
        raise AssertionError("sacks feature config still contains stale feature")

    print("PASS: source contracts use per-defensive-play sack rate x expected opponent plays")


def post_rebuild_checks() -> None:
    historical_manifest_path = PROP / "data/historical/features/feature_manifest.json"
    sacks_manifest_path = PROP / "models/sacks/feature_manifest.json"
    metadata_path = PROP / "models/sacks/metadata.json"
    selected_path = PROP / "models/sacks/selected_model.json"
    model_path = PROP / "models/sacks/direct_model.txt"

    for path in [
        historical_manifest_path,
        sacks_manifest_path,
        metadata_path,
        selected_path,
        model_path,
    ]:
        if not path.is_file():
            raise AssertionError(f"Required rebuilt artifact missing: {path}")

    historical = read_json(historical_manifest_path)
    hist_features = list(historical.get("feature_columns", []))
    if NEW not in hist_features or OLD in hist_features:
        raise AssertionError("historical feature manifest is stale")

    sacks_manifest = read_json(sacks_manifest_path)
    numeric = list(sacks_manifest.get("numeric_features", []))
    categorical = list(sacks_manifest.get("categorical_features", []))
    selected_features = list(sacks_manifest.get("selected_features", numeric + categorical))
    if NEW not in selected_features or OLD in selected_features:
        raise AssertionError("sacks direct-model feature manifest is stale")

    expected_hash = feature_hash(numeric, categorical)
    if sacks_manifest.get("feature_hash") != expected_hash:
        raise AssertionError("sacks feature_manifest feature_hash mismatch")

    metadata = read_json(metadata_path)
    if metadata.get("feature_hash") != expected_hash:
        raise AssertionError("sacks metadata feature_hash mismatch")
    if metadata.get("market_features_used") is not False:
        raise AssertionError("sacks metadata market_features_used must be false")

    model_text = model_path.read_text(encoding="utf-8", errors="replace")
    feature_line = next((line for line in model_text.splitlines() if line.startswith("feature_names=")), "")
    if not feature_line:
        raise AssertionError("direct_model.txt feature_names line missing")
    model_features = feature_line.split("=", 1)[1].split()
    if NEW not in model_features or OLD in model_features:
        raise AssertionError("sacks direct model was not retrained on corrected feature")

    selected = read_json(selected_path)
    if selected.get("test_used_for_selection") is not False:
        raise AssertionError("selected_model test_used_for_selection must remain false")
    if selected.get("selection_frozen_before_test_reporting") is not True:
        raise AssertionError("selected_model selection_frozen_before_test_reporting must remain true")
    if selected.get("market_features_used") is not False:
        raise AssertionError("selected_model market_features_used must remain false")

    primary = metadata.get("primary", {})
    metrics = primary.get("validation_2024_metrics", {})
    print(
        "PASS: rebuilt sacks direct model uses corrected feature "
        f"(best_iteration={primary.get('best_iteration_selected_on_2024')}, "
        f"validation_2024_mae={metrics.get('mae')})"
    )
    print(
        "PASS: frozen architecture selection preserved "
        f"(selected={selected.get('selected_architecture')}, "
        f"validation_season={selected.get('validation_season')}, "
        f"test_season={selected.get('test_season')})"
    )


def main() -> int:
    args = parse_args()
    source_checks()
    if args.post_rebuild:
        post_rebuild_checks()
    print("ISSUE 56 SACKS DIRECT FEATURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
