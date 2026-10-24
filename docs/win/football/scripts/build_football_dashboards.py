#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, UTC
import html

FRONTEND = Path("frontend")

LEAGUES = [
    ("nfl", "NFL"),
    ("ncaaf", "College Football"),
    ("cfl", "CFL"),
]

MASTER_OUTPUT = FRONTEND / "football_dashboard.html"

LEAGUE_OUTPUTS = {
    "nfl": FRONTEND / "nfl_dashboard.html",
    "ncaaf": FRONTEND / "ncaaf_dashboard.html",
    "cfl": FRONTEND / "cfl_dashboard.html",
}


def shell(title: str, subtitle: str, body: str) -> str:
    built = datetime.now(UTC).isoformat(timespec="seconds")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="assets/css/matstheme.css">
<style>
.dashboard-wrap {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 28px 20px 48px;
}}
.dashboard-head {{
  border-bottom: 1px solid var(--border-soft);
  padding-bottom: 14px;
  margin-bottom: 18px;
}}
.dashboard-head h1 {{
  margin: 0 0 6px;
  font-size: 24px;
  letter-spacing: 2px;
  text-transform: uppercase;
}}
.dashboard-sub {{
  color: var(--text-muted);
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
}}
.league-master-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
}}
.league-master-card {{
  display: block;
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 4px;
  padding: 18px;
  color: var(--text-main);
  text-decoration: none;
}}
.league-master-card:hover {{
  border-color: var(--accent-green);
}}
.league-master-card strong {{
  color: var(--accent-green);
  font-size: 15px;
  letter-spacing: 1px;
  text-transform: uppercase;
}}
.empty-panel {{
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 4px;
  padding: 24px;
  color: var(--text-muted);
  font-size: 12px;
}}
.build-ts {{
  margin-top: 18px;
  color: var(--text-muted);
  font-size: 10px;
}}
</style>
</head>
<body>
<div id="nav-placeholder"></div>
<main class="dashboard-wrap">
  <div class="dashboard-head">
    <h1>{html.escape(title)}</h1>
    <div class="dashboard-sub">{html.escape(subtitle)}</div>
  </div>
  {body}
  <div class="build-ts">Built {html.escape(built)} UTC</div>
</main>
<script src="assets/js/shared/nav.js"></script>
</body>
</html>
"""


def build_master() -> str:
    cards = "\n".join(
        f'<a class="league-master-card" href="{LEAGUE_OUTPUTS[key].name}">'
        f'<strong>{html.escape(label)}</strong></a>'
        for key, label in LEAGUES
    )
    return shell(
        "Football Dashboard",
        "Master sport dashboard",
        f'<div class="league-master-grid">{cards}</div>',
    )


def build_league(label: str) -> str:
    return shell(
        f"{label} Dashboard",
        "Football league dashboard",
        '<section class="empty-panel">'
        f'{html.escape(label)} dashboard structure is created. '
        'Analytics sections are intentionally not invented until the league data/report requirements are specified.'
        '</section>',
    )


def main() -> None:
    FRONTEND.mkdir(parents=True, exist_ok=True)
    MASTER_OUTPUT.write_text(build_master(), encoding="utf-8")
    for key, label in LEAGUES:
        LEAGUE_OUTPUTS[key].write_text(build_league(label), encoding="utf-8")

    print(f"Football master dashboard generated: {MASTER_OUTPUT}")
    for key, _ in LEAGUES:
        print(f"Football league dashboard generated: {LEAGUE_OUTPUTS[key]}")


if __name__ == "__main__":
    main()
