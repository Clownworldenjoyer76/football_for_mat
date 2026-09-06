#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 41."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

RELATIVE_CONFIG = Path(
    "docs/win/football/nfl/prop_engine/config/fallback_rules.yaml"
)

EXPECTED_HEADERS = [
    "rookie",
    "new_team",
    "backup_promotion",
    "missing_snap_history",
    "missing_participation",
    "missing_weather",
    "missing_depth",
    "missing_injury",
    "missing_player_stats",
    "defensive_low_volume",
    "kicker_change",
]

EFFICIENCY_PRIORITY = [
    "Same-season player prior",
    "Recent prior-season player prior",
    "Career player prior",
    "Position-group league prior",
]

ROLE_SHARE_PRIORITY = [
    "Current-team recent role",
    "Current depth rank + career role",
    "Position/depth prior",
]

FORBIDDEN_TOKENS = (
    "sportsbook",
    "moneyline",
    "spread",
    "prop_line",
    "betting",
    "drat",
    "epred",
)


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "docs/win/football/nfl/prop_engine").is_dir():
            return candidate
    raise AssertionError(
        "Unable to resolve repository root containing "
        "docs/win/football/nfl/prop_engine."
    )


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a YAML mapping.")
    return value


def require_rule(section: dict[str, Any], key: str, expected: Any, label: str) -> None:
    rules = require_mapping(section.get("rules"), f"{label}.rules")
    actual = rules.get(key)
    if actual != expected:
        raise AssertionError(
            f"{label}.rules.{key} must be {expected!r}; got {actual!r}."
        )


def require_uncertainty_widen(section: dict[str, Any], label: str) -> None:
    uncertainty = require_mapping(
        section.get("uncertainty"), f"{label}.uncertainty"
    )
    if uncertainty.get("widen") is not True:
        raise AssertionError(
            f"{label}.uncertainty.widen must be true."
        )


def main() -> int:
    repo = find_repo_root()
    path = repo / RELATIVE_CONFIG
    if not path.is_file():
        raise AssertionError(f"Missing required Issue 41 file: {path}")

    raw = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    data = require_mapping(loaded, "fallback_rules.yaml")

    actual_headers = list(data.keys())
    if actual_headers != EXPECTED_HEADERS:
        raise AssertionError(
            "Top-level fallback headers/order mismatch.\n"
            f"Expected: {EXPECTED_HEADERS}\n"
            f"Actual:   {actual_headers}"
        )

    for header in EXPECTED_HEADERS:
        section = require_mapping(data[header], header)

        actual_eff = section.get("efficiency_priority")
        if actual_eff != EFFICIENCY_PRIORITY:
            raise AssertionError(
                f"{header}.efficiency_priority mismatch.\n"
                f"Expected: {EFFICIENCY_PRIORITY}\n"
                f"Actual:   {actual_eff}"
            )

        actual_role = section.get("role_share_priority")
        if actual_role != ROLE_SHARE_PRIORITY:
            raise AssertionError(
                f"{header}.role_share_priority mismatch.\n"
                f"Expected: {ROLE_SHARE_PRIORITY}\n"
                f"Actual:   {actual_role}"
            )

        if not isinstance(section.get("trigger"), str) or not section["trigger"].strip():
            raise AssertionError(f"{header}.trigger must be a non-empty string.")

        require_mapping(section.get("rules"), f"{header}.rules")
        require_mapping(section.get("uncertainty"), f"{header}.uncertainty")

    # Exact special-case contract.
    require_rule(
        data["backup_promotion"],
        "efficiency",
        "retain_player_efficiency",
        "backup_promotion",
    )
    require_rule(
        data["backup_promotion"],
        "workload",
        "replace_with_promoted_role_expectation",
        "backup_promotion",
    )

    require_rule(
        data["new_team"],
        "efficiency",
        "retain_player_efficiency",
        "new_team",
    )
    require_rule(
        data["new_team"],
        "team_share",
        "reset",
        "new_team",
    )

    require_rule(
        data["missing_weather"],
        "environment",
        "use_roof_surface_and_missing_flags",
        "missing_weather",
    )
    require_rule(
        data["missing_weather"],
        "set_missing_flag",
        True,
        "missing_weather",
    )

    for header in ("missing_snap_history", "missing_participation"):
        require_rule(
            data[header],
            "workload",
            "use_depth_plus_statistical_usage",
            header,
        )
        require_rule(data[header], "set_missing_flag", True, header)
        require_uncertainty_widen(data[header], header)

    # Required uncertainty widening for the role-change/no-history cases.
    for header in (
        "rookie",
        "new_team",
        "backup_promotion",
        "missing_depth",
        "missing_injury",
        "missing_player_stats",
        "defensive_low_volume",
        "kicker_change",
    ):
        require_uncertainty_widen(data[header], header)

    # Rookie rule must never invent NFL production.
    require_rule(
        data["rookie"],
        "nfl_production_imputation",
        "forbidden",
        "rookie",
    )

    # Missing injury data must not silently become "healthy".
    require_rule(
        data["missing_injury"],
        "injury_status",
        "do_not_assume_healthy",
        "missing_injury",
    )

    # Kicker changes retain player efficiency only when it exists, but role volume
    # comes from the new primary-kicker expectation.
    require_rule(
        data["kicker_change"],
        "efficiency",
        "retain_player_efficiency_when_available",
        "kicker_change",
    )
    require_rule(
        data["kicker_change"],
        "workload",
        "replace_with_primary_kicker_role_expectation",
        "kicker_change",
    )

    lower = raw.casefold()
    found_forbidden = [token for token in FORBIDDEN_TOKENS if token in lower]
    if found_forbidden:
        raise AssertionError(
            "Forbidden betting/market-derived token(s) found in fallback rules: "
            + ", ".join(found_forbidden)
        )

    print(f"config={RELATIVE_CONFIG.as_posix()}")
    print(f"headers={len(EXPECTED_HEADERS)}")
    print(f"efficiency_priority_levels={len(EFFICIENCY_PRIORITY)}")
    print(f"role_share_priority_levels={len(ROLE_SHARE_PRIORITY)}")
    print("backup_promotion_retains_efficiency=true")
    print("backup_promotion_replaces_workload=true")
    print("trade_retains_efficiency=true")
    print("trade_resets_team_share=true")
    print("missing_weather_uses_roof_surface_flags=true")
    print("missing_snap_participation_uses_depth_stat_usage=true")
    print("missing_snap_participation_widens_uncertainty=true")
    print("market_features_used=false")
    print("FALLBACK RULES VALIDATION: PASS")
    print("ISSUE 41 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FALLBACK RULES VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
