#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Protocol


class WarningReporter(Protocol):
    def warning(self, message: str, **details: object) -> None:
        ...


def publish_staged_directory(
    stage_root: Path,
    *,
    output_dir: Path,
    reporter: WarningReporter,
    cleanup_warning: str,
) -> None:
    backup_root = (
        output_dir.parent
        / f".{output_dir.name}_backup_{uuid.uuid4().hex}"
    )

    try:
        if output_dir.exists():
            os.replace(output_dir, backup_root)

        os.replace(stage_root, output_dir)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        if backup_root.exists():
            os.replace(backup_root, output_dir)

        raise

    if backup_root.exists():
        try:
            shutil.rmtree(backup_root)
        except Exception as exc:
            reporter.warning(
                cleanup_warning,
                backup_path=str(backup_root),
                error_type=type(exc).__name__,
                error=str(exc),
            )
