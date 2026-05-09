from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


URL = "https://www.dratings.com/predictor/ufc-mma-predictions/"
OUTDIR = Path("docs/win/mma/ufc/00_intake/predictions")

PLAYWRIGHT_TIMEZONE = "America/New_York"

NOISE_LINES = {
    "Sports Ratings, Prediction, & Analysis",
    "Ratings",
    "Predictions",
    "Tools",
    "Sportsbooks",
    "About",
    "Blog",
    "★ PREMIUM",
    "Odds Feed",
    "Offshore Odds",
    "Vegas Odds",
    "UFC MMA Predictions",
    "Upcoming",
    "Completed",
    "Season",
    "Methodology",
    "Related",
    "Time Fighters Win Best",
    "ML Bet",
    "Value",
    "More Details",
}

STOP_PATTERNS = [
    r"^Fights for ",
    r"^Completed Fights$",
    r"^Load More Fights$",
    r"^Season Prediction Results$",
]


def ensure_pkg(pkg: str, import_name: str | None = None) -> None:
    module_name = import_name or pkg
    try:
        __import__(module_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_pkg("playwright")

try:
    from playwright.sync_api import sync_playwright
except Exception:
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.sync_api import sync_playwright

try:
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
except Exception:
    pass


def clean_line(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_date_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", value))


def is_time_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}\s+(AM|PM)", value, re.I))


def is_percent_line(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}\.\d+%", value))


def is_moneyline(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]\d{2,5}", value))


def normalize_date(value: str) -> str:
    dt = datetime.strptime(value, "%m/%d/%Y")
    return dt.strftime("%Y_%m_%d")


def pct_to_decimal_str(value: str) -> str:
    num = float(value.replace("%", "")) / 100.0
    return f"{num:.3f}"


def split_fighter_and_prob(value: str) -> tuple[str | None, str | None]:
    """
    Example:
      'Mike Malott 25.4%' -> ('Mike Malott', '0.254')
    """
    match = re.fullmatch(r"(.+?)\s+(\d{1,3}\.\d+%)", value)
    if not match:
        return None, None

    fighter = match.group(1).strip()
    prob = pct_to_decimal_str(match.group(2))
    return fighter, prob


def scrape_body_text() -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        context = browser.new_context(
            viewport={"width": 1600, "height": 4000},
            timezone_id=PLAYWRIGHT_TIMEZONE,
            locale="en-US",
        )

        page = context.new_page()

        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(6000)

        for _ in range(10):
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(600)

        body_text = page.locator("body").inner_text(timeout=30000)

        browser.close()
        return body_text


def parse_rows(body_text: str) -> list[dict[str, str]]:
    lines = [clean_line(line) for line in body_text.splitlines()]
    lines = [line for line in lines if line and line not in NOISE_LINES]

    rows: list[dict[str, str]] = []
    in_upcoming = False
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        if line.startswith("Upcoming Fights for "):
            in_upcoming = True
            idx += 1
            continue

        if in_upcoming and any(re.search(pattern, line, re.I) for pattern in STOP_PATTERNS):
            break

        if not in_upcoming:
            idx += 1
            continue

        # Expected upcoming block:
        # 04/18/2026
        # 05:00 PM
        # Gilbert Burns
        # Mike Malott 25.4%
        # 74.6%
        # +290
        # -330
        if idx + 6 < len(lines) and is_date_line(lines[idx]) and is_time_line(lines[idx + 1]):
            raw_date = lines[idx]
            fighter_1 = lines[idx + 2]
            fighter_2_line = lines[idx + 3]
            fighter_2_prob_line = lines[idx + 4]
            moneyline_1 = lines[idx + 5]
            moneyline_2 = lines[idx + 6]

            fighter_2, fighter_1_prob = split_fighter_and_prob(fighter_2_line)

            if (
                fighter_2
                and fighter_1
                and is_percent_line(fighter_2_prob_line)
                and is_moneyline(moneyline_1)
                and is_moneyline(moneyline_2)
            ):
                rows.append(
                    {
                        "match_date": normalize_date(raw_date),
                        "fighter_1": fighter_1,
                        "fighter_2": fighter_2,
                        "fighter_1_win_prob": fighter_1_prob,
                        "fighter_2_win_prob": pct_to_decimal_str(fighter_2_prob_line),
                    }
                )
                idx += 7
                continue

        idx += 1

    return rows


def write_output_files(rows: list[dict[str, str]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_date[row["match_date"]].append(row)

    fieldnames = [
        "match_date",
        "fighter_1",
        "fighter_2",
        "fighter_1_win_prob",
        "fighter_2_win_prob",
    ]

    for match_date, date_rows in sorted(rows_by_date.items()):
        outfile = OUTDIR / f"{match_date}_ufc_predictions.csv"
        with outfile.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(date_rows)

        print(f"WROTE {outfile} ({len(date_rows)} rows)")


def main() -> int:
    try:
        body_text = scrape_body_text()
        rows = parse_rows(body_text)

        if not rows:
            print("No rows parsed from page.")
            return 1

        write_output_files(rows)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
