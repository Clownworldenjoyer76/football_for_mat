#!/usr/bin/env python3
"""Run the NFL Prop Engine historical build in one deterministic sequence.

READS:
    docs/win/football/nfl/prop_engine/config/prop_engine.yaml
    the historical builder/validator scripts listed in PIPELINE

WRITES:
    docs/win/football/nfl/prop_engine/logs/historical_build_{timestamp}.json

POLICY:
    - Exact Issue 44 execution order.
    - The runner owns --start-season/--end-season because existing historical
      builders read those values from common.load_config().
    - The tracked prop_engine.yaml is never edited. Each child script receives
      an in-memory config copy with the requested historical range.
    - Stop immediately on the first nonzero SystemExit or uncaught exception.
    - A prior successful runner log for the exact same season range is reusable
      unless --force is supplied.
    - No sportsbook/market data is introduced by this orchestration layer.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import runpy
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common


PIPELINE: tuple[str, ...] = (
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
)

LOG_PREFIX = "historical_build_"
LOG_SUFFIX = ".json"
OUTPUT_TAIL_CHARS = 12000


@dataclass
class StepResult:
    step_number: int
    script: str
    command: list[str]
    status: str
    exit_code: int
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str


class Tee(io.TextIOBase):
    """Mirror writes while retaining text for the structured run log."""

    def __init__(self, primary: io.TextIOBase) -> None:
        self.primary = primary
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.primary.flush()
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_token(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NFL Prop Engine historical build in exact Issue 44 order."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="Historical build start season. Defaults to config.seasons.historical_start.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Historical build end season. Defaults to config.seasons.historical_end.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run all steps even if an earlier successful runner log covers this exact range.",
    )
    return parser


def resolve_seasons(args: argparse.Namespace, config: dict[str, Any]) -> tuple[int, int]:
    configured_start = int(config["seasons"]["historical_start"])
    configured_end = int(config["seasons"]["historical_end"])

    start = configured_start if args.start_season is None else int(args.start_season)
    end = configured_end if args.end_season is None else int(args.end_season)

    if start > end:
        raise ValueError(
            f"--start-season ({start}) cannot exceed --end-season ({end})."
        )
    if start < configured_start or end > configured_end:
        raise ValueError(
            "Requested historical range must remain inside the configured modeled "
            f"range {configured_start}-{configured_end}; received {start}-{end}."
        )
    return start, end


def build_config_override(
    base_config: dict[str, Any],
    start_season: int,
    end_season: int,
) -> dict[str, Any]:
    """Return an isolated config copy containing the runner-requested range."""
    config = copy.deepcopy(base_config)
    config["seasons"]["historical_start"] = int(start_season)
    config["seasons"]["historical_end"] = int(end_season)
    return config


def validate_pipeline_files(prop_root: Path, pipeline: Sequence[str] = PIPELINE) -> None:
    missing = [
        script
        for script in pipeline
        if not (prop_root / "scripts" / script).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Historical build runner is missing required script(s): "
            + ", ".join(missing)
        )


def log_path(log_dir: Path, value: datetime | None = None) -> Path:
    return log_dir / f"{LOG_PREFIX}{timestamp_token(value)}{LOG_SUFFIX}"


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def find_prior_success(
    log_dir: Path,
    start_season: int,
    end_season: int,
) -> Path | None:
    if not log_dir.is_dir():
        return None

    for path in sorted(
        log_dir.glob(f"{LOG_PREFIX}*{LOG_SUFFIX}"),
        reverse=True,
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if (
            payload.get("status") == "success"
            and int(payload.get("start_season", -1)) == int(start_season)
            and int(payload.get("end_season", -1)) == int(end_season)
        ):
            return path
    return None


def _exit_code_from_system_exit(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return int(code)
    return 1


def execute_script(
    *,
    step_number: int,
    script: str,
    scripts_root: Path,
    repo_root: Path,
    config_override: dict[str, Any],
) -> StepResult:
    """Execute one existing script as __main__ using the overridden config."""
    script_path = (scripts_root / script).resolve()
    started = utc_now()
    started_clock = time.perf_counter()

    stdout_tee = Tee(sys.stdout)
    stderr_tee = Tee(sys.stderr)

    original_argv = list(sys.argv)
    original_cwd = Path.cwd()
    original_load_config = common.load_config

    status = "success"
    exit_code = 0

    def overridden_load_config() -> dict[str, Any]:
        return copy.deepcopy(config_override)

    try:
        common.load_config = overridden_load_config
        sys.argv = [str(script_path)]
        os.chdir(repo_root)

        from contextlib import redirect_stderr, redirect_stdout

        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit as exc:
                exit_code = _exit_code_from_system_exit(exc)
                if exit_code != 0:
                    status = "failed"
            except BaseException:
                status = "failed"
                exit_code = 1
                traceback.print_exc(file=stderr_tee)
    finally:
        common.load_config = original_load_config
        sys.argv = original_argv
        os.chdir(original_cwd)

    ended = utc_now()
    duration = time.perf_counter() - started_clock

    stdout_text = stdout_tee.getvalue()
    stderr_text = stderr_tee.getvalue()

    return StepResult(
        step_number=step_number,
        script=script,
        command=[sys.executable, str(script_path)],
        status=status,
        exit_code=int(exit_code),
        started_at=iso_utc(started),
        ended_at=iso_utc(ended),
        duration_seconds=round(float(duration), 6),
        stdout_tail=stdout_text[-OUTPUT_TAIL_CHARS:],
        stderr_tail=stderr_text[-OUTPUT_TAIL_CHARS:],
    )


Executor = Callable[..., StepResult]


def run_pipeline(
    *,
    pipeline: Sequence[str],
    scripts_root: Path,
    repo_root: Path,
    config_override: dict[str, Any],
    executor: Executor = execute_script,
) -> tuple[str, list[StepResult]]:
    """Run in order and stop after the first failed result."""
    results: list[StepResult] = []

    for step_number, script in enumerate(pipeline, start=1):
        print(f"[{step_number:02d}/{len(pipeline):02d}] {script}")
        result = executor(
            step_number=step_number,
            script=script,
            scripts_root=scripts_root,
            repo_root=repo_root,
            config_override=config_override,
        )
        results.append(result)

        if result.exit_code != 0 or result.status != "success":
            print(
                f"HISTORICAL BUILD STOPPED: step={step_number} "
                f"script={script} exit_code={result.exit_code}",
                file=sys.stderr,
            )
            return "failed", results

    return "success", results


def main(argv: Sequence[str] | None = None) -> int:
    base_config = common.load_config()
    args = build_parser().parse_args(argv)
    start_season, end_season = resolve_seasons(args, base_config)

    repo_root = common.repo_root().resolve()
    prop_root = common.prop_root().resolve()
    scripts_root = (prop_root / "scripts").resolve()
    logs_dir = (prop_root / "logs").resolve()

    validate_pipeline_files(prop_root)

    run_started = utc_now()
    destination = log_path(logs_dir, run_started)

    prior_success = None
    if not args.force:
        prior_success = find_prior_success(
            logs_dir,
            start_season,
            end_season,
        )

    if prior_success is not None:
        payload = {
            "status": "skipped_already_successful",
            "start_season": start_season,
            "end_season": end_season,
            "force": False,
            "started_at": iso_utc(run_started),
            "ended_at": iso_utc(),
            "pipeline": list(PIPELINE),
            "steps_total": len(PIPELINE),
            "steps_executed": 0,
            "prior_success_log": str(
                prior_success.relative_to(repo_root)
            ).replace("\\", "/"),
            "market_features_used": False,
        }
        write_json_atomic(payload, destination)
        print(
            "HISTORICAL BUILD: SKIP "
            f"(successful {start_season}-{end_season} runner log already exists)"
        )
        print(f"log={destination}")
        return 0

    config_override = build_config_override(
        base_config,
        start_season,
        end_season,
    )

    status, results = run_pipeline(
        pipeline=PIPELINE,
        scripts_root=scripts_root,
        repo_root=repo_root,
        config_override=config_override,
    )

    failed_step = next(
        (result for result in results if result.status != "success"),
        None,
    )

    payload = {
        "status": status,
        "start_season": start_season,
        "end_season": end_season,
        "force": bool(args.force),
        "started_at": iso_utc(run_started),
        "ended_at": iso_utc(),
        "pipeline": list(PIPELINE),
        "steps_total": len(PIPELINE),
        "steps_executed": len(results),
        "failed_step_number": (
            failed_step.step_number if failed_step is not None else None
        ),
        "failed_script": (
            failed_step.script if failed_step is not None else None
        ),
        "steps": [asdict(result) for result in results],
        "market_features_used": False,
    }
    write_json_atomic(payload, destination)

    print(f"log={destination}")

    if status != "success":
        print("HISTORICAL BUILD: FAIL", file=sys.stderr)
        return int(failed_step.exit_code if failed_step is not None else 1)

    print("HISTORICAL BUILD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
