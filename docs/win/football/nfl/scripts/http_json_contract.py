#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def fetch_json_object(
    request: urllib.request.Request,
    *,
    timeout: int,
    attempts: int,
    retryable_http_codes: set[int],
) -> tuple[dict[str, Any] | None, int, str]:
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read()

            try:
                payload = json.loads(raw.decode("utf-8"))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = (
                    "invalid UTF-8/JSON response: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if isinstance(payload, dict):
                    return payload, attempt, ""
                last_error = "response root was not a JSON object"

        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in retryable_http_codes:
                return None, attempt, last_error

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            return None, attempt, last_error

        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))

    return None, attempts, last_error
