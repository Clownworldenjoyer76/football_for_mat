#!/usr/bin/env python3
from __future__ import annotations

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
