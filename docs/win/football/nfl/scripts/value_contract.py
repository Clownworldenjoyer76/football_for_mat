#!/usr/bin/env python3
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Never


def finite_decimal_text(
    raw: Any,
    *,
    label: str,
    clean: Callable[[Any], str],
    fail: Callable[[str], Never],
) -> tuple[str, Decimal]:
    text = clean(raw)
    if not text:
        fail(f"{label} is blank")

    try:
        number = Decimal(text)
    except InvalidOperation:
        fail(f"{label} must be numeric; received={text!r}")
        raise AssertionError("unreachable")

    if not number.is_finite():
        fail(f"{label} must be finite; received={text!r}")

    return text, number


def bind_optional_numeric_parsers(
    clean: Callable[[Any], str],
) -> tuple[
    Callable[[Any], float | None],
    Callable[[Any], int | None],
]:
    """Bind shared optional finite-float and integral-number parsers."""

    def parse_float(value: Any) -> float | None:
        text = clean(value)
        if not text:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def parse_int(value: Any) -> int | None:
        number = parse_float(value)
        if number is None or not float(number).is_integer():
            return None
        return int(number)

    return parse_float, parse_int
