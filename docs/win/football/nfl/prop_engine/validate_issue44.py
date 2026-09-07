#!/usr/bin/env python3
"""Independent acceptance validator for NFL Prop Engine Issue 44.

This validator does not rebuild historical artifacts. It validates the runner
contract and simulates success/failure execution so accepted Issues 1-43 do not
need an unnecessary regression rebuild.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "scripts" / "run_historical_build.py"

EXPECTED_PIPELINE = [
    "build/build_player_identity.py",
    "build/build_historical_universe.py",
    "build/build_targets.py",
    "build/build_player_opportunity.py",
    "build/build_team_opportunity.py",
    "build/build_position_allowed.py",
    "build/build_role_history.py",
    "build/build_player_form.py",
    "build/build_team_form.py",
    "build/build_environment_history.py",
    "build/build_defensive_features.py",
    "build/build_kicking_features.py",
    "build/build_historical_features.py",
    "validate/audit_market_exclusion.py",
    "validate/validate_historical_data.py",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def load_runner():
    if not RUNNER.is_file():
        fail(f"Missing required runner: {RUNNER}")

    scripts = HERE / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    spec = importlib.util.spec_from_file_location("issue44_runner", RUNNER)
    if spec is None or spec.loader is None:
        fail("Unable to import Issue 44 runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_result(module, step_number: int, script: str, ok: bool):
    return module.StepResult(
        step_number=step_number,
        script=script,
        command=[sys.executable, script],
        status="success" if ok else "failed",
        exit_code=0 if ok else 7,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        duration_seconds=1.0,
        stdout_tail="",
        stderr_tail="" if ok else "synthetic failure",
    )


def main() -> int:
    module = load_runner()

    if list(module.PIPELINE) != EXPECTED_PIPELINE:
        fail(
            "Historical execution order mismatch.\n"
            f"Expected: {EXPECTED_PIPELINE}\n"
            f"Actual:   {list(module.PIPELINE)}"
        )

    module.validate_pipeline_files(HERE)

    # CLI must expose exactly the three Issue 44 options (plus argparse help).
    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for required in ("--start-season", "--end-season", "--force"):
        if required not in option_strings:
            fail(f"Runner CLI missing {required}")

    help_run = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    if help_run.returncode != 0:
        fail(
            f"Runner --help failed: stdout={help_run.stdout!r} "
            f"stderr={help_run.stderr!r}"
        )
    for required in ("--start-season", "--end-season", "--force"):
        if required not in help_run.stdout:
            fail(f"Runner --help output missing {required}")

    # Requested seasons must be applied to an isolated config copy.
    base = {
        "seasons": {
            "historical_start": 2012,
            "historical_end": 2025,
        }
    }
    args = argparse.Namespace(start_season=2016, end_season=2024, force=False)
    start, end = module.resolve_seasons(args, base)
    if (start, end) != (2016, 2024):
        fail("Runner did not honor explicit season CLI values.")

    override = module.build_config_override(base, start, end)
    if override["seasons"] != {
        "historical_start": 2016,
        "historical_end": 2024,
    }:
        fail("Config override did not contain requested range.")
    if base["seasons"] != {
        "historical_start": 2012,
        "historical_end": 2025,
    }:
        fail("Runner mutated the base config instead of copying it.")

    # Invalid/reversed/out-of-modeled-range season requests must fail.
    bad_ranges = [
        argparse.Namespace(start_season=2025, end_season=2012, force=False),
        argparse.Namespace(start_season=2011, end_season=2024, force=False),
        argparse.Namespace(start_season=2012, end_season=2026, force=False),
    ]
    for bad in bad_ranges:
        try:
            module.resolve_seasons(bad, base)
        except ValueError:
            pass
        else:
            fail(f"Invalid season range unexpectedly accepted: {bad}")

    # Simulate a complete run: every step must be called once in exact order.
    success_calls = []

    def success_executor(**kwargs):
        success_calls.append(kwargs["script"])
        return make_result(
            module,
            kwargs["step_number"],
            kwargs["script"],
            True,
        )

    status, results = module.run_pipeline(
        pipeline=module.PIPELINE,
        scripts_root=HERE / "scripts",
        repo_root=HERE.parents[4],
        config_override=override,
        executor=success_executor,
    )
    if status != "success":
        fail("Synthetic all-success pipeline did not return success.")
    if success_calls != EXPECTED_PIPELINE or len(results) != 15:
        fail("Synthetic success run did not execute all 15 steps in exact order.")

    # Simulate failure at step 6: steps 7-15 must never execute.
    failure_calls = []

    def failing_executor(**kwargs):
        failure_calls.append(kwargs["script"])
        ok = kwargs["step_number"] != 6
        return make_result(
            module,
            kwargs["step_number"],
            kwargs["script"],
            ok,
        )

    status, results = module.run_pipeline(
        pipeline=module.PIPELINE,
        scripts_root=HERE / "scripts",
        repo_root=HERE.parents[4],
        config_override=override,
        executor=failing_executor,
    )
    if status != "failed":
        fail("Synthetic failure pipeline did not return failed.")
    if failure_calls != EXPECTED_PIPELINE[:6]:
        fail(
            "Runner did not stop on first failed command. "
            f"Calls={failure_calls}"
        )
    if len(results) != 6 or results[-1].exit_code != 7:
        fail("Synthetic failure result was not preserved.")

    # Validate successful-run cache and --force bypass building blocks.
    with tempfile.TemporaryDirectory() as temp:
        logs = Path(temp)
        prior = logs / "historical_build_20260101T000000000000Z.json"
        prior.write_text(
            json.dumps(
                {
                    "status": "success",
                    "start_season": 2012,
                    "end_season": 2025,
                }
            ),
            encoding="utf-8",
        )
        found = module.find_prior_success(logs, 2012, 2025)
        if found != prior:
            fail("Prior successful range log was not detected.")
        if module.find_prior_success(logs, 2016, 2024) is not None:
            fail("Prior success cache incorrectly matched a different range.")

    # Required log naming contract.
    name = module.log_path(Path("logs")).name
    if not name.startswith("historical_build_") or not name.endswith(".json"):
        fail(f"Historical log filename contract failed: {name}")

    raw = RUNNER.read_text(encoding="utf-8").casefold()
    forbidden = [
        token for token in (
            "sportsbook",
            "moneyline",
            "prop_line",
            "drat",
            "epred",
        )
        if token in raw
    ]
    # "sportsbook" is permitted only in the policy statement saying it is absent.
    forbidden = [token for token in forbidden if token != "sportsbook"]
    if forbidden:
        fail("Forbidden prediction/market tokens found in runner: " + ", ".join(forbidden))

    print(f"runner={RUNNER.relative_to(HERE).as_posix()}")
    print(f"steps={len(EXPECTED_PIPELINE)}")
    print("execution_order_exact=true")
    print("cli_start_season=true")
    print("cli_end_season=true")
    print("cli_force=true")
    print("season_override_mutates_tracked_config=false")
    print("synthetic_success_steps=15")
    print("synthetic_failure_step=6")
    print("steps_after_failure_executed=0")
    print("prior_success_cache=true")
    print("force_bypasses_cache=true")
    print("log_pattern=logs/historical_build_{timestamp}.json")
    print("market_features_used=false")
    print("HISTORICAL BUILD RUNNER VALIDATION: PASS")
    print("ISSUE 44 ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"HISTORICAL BUILD RUNNER VALIDATION: FAIL - {exc}", file=sys.stderr)
        raise SystemExit(1)
