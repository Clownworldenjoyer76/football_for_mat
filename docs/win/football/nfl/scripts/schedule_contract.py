#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def schedule_target_key(
    row: Mapping[str, Any],
    *,
    clean: Callable[[Any], str],
) -> tuple[str, str, str]:
    return (
        clean(row.get("season")),
        clean(row.get("season_type")),
        clean(row.get("week")),
    )
