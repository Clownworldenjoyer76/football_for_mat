#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/04_basketball_results_dashboard.py
#
# Builds a static HTML dashboard per league from the report CSVs produced by
# 03_basketball_results_reports.py. No server required: open the HTML files
# in a browser.
#
# Inputs (per league nba, ncaam, wnba):
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/reports/{league}/{moneyline,spread,total,overview}/*.csv
#
# Outputs:
#   docs/win/basketball/05_final_scores/dashboard/{league}/index.html
#   docs/win/basketball/05_final_scores/dashboard/index.html  (top-level league chooser)

from datetime import datetime, UTC
from pathlib import Path
import html
import json

import pandas as pd

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

BASE          = Path("docs/win/basketball/05_final_scores")
REPORT_DIR    = BASE / "reports"
DASHBOARD_DIR = BASE / "dashboard"


# =========================
# CSS / JS (inlined)
# =========================

CSS = """
:root {
  --bg: #0e1117;
  --panel: #161b22;
  --panel-2: #1c232c;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --good: #3fb950;
  --bad:  #f85149;
  --warn: #d29922;
  --border: #30363d;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; }
header { padding: 18px 24px; border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; align-items:baseline; }
header h1 { margin: 0; font-size: 20px; }
header .ts { color: var(--muted); font-size: 12px; }
nav.leagues { padding: 8px 24px; border-bottom: 1px solid var(--border); }
nav.leagues a { color: var(--accent); margin-right: 16px; text-decoration: none; }
main { padding: 18px 24px; max-width: 1400px; margin: 0 auto; }
h2 { font-size: 16px; margin: 24px 0 8px 0; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
h3 { font-size: 13px; margin: 14px 0 6px 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0 4px 0; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 22px; margin-top: 4px; font-weight: 600; }
.kpi .value.good { color: var(--good); }
.kpi .value.bad  { color: var(--bad); }
.tabs { display:flex; gap: 4px; margin: 16px 0 0 0; flex-wrap: wrap; }
.tab { background: var(--panel); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px 6px 0 0; cursor: pointer; color: var(--muted); font-size: 13px; }
.tab.active { background: var(--panel-2); color: var(--text); border-bottom-color: var(--panel-2); }
.tab-body { background: var(--panel-2); border: 1px solid var(--border); border-top: none; padding: 14px; border-radius: 0 6px 6px 6px; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
.controls label { color: var(--muted); font-size: 12px; }
.controls select { background: var(--panel); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--panel-2); }
th .arrow { opacity: 0.4; margin-left: 4px; }
th.sorted .arrow { opacity: 1; color: var(--accent); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: var(--good); }
.neg { color: var(--bad); }
.muted { color: var(--muted); }
.section { margin-top: 18px; }
.scroll { max-height: 60vh; overflow: auto; border: 1px solid var(--border); border-radius: 6px; }
footer { color: var(--muted); font-size: 11px; padding: 16px 24px; }
"""

JS = """
function fmtPct(v) { if (v == null || isNaN(v)) return ''; return (v*100).toFixed(2) + '%'; }
function fmtNum(v, d) { if (v == null || isNaN(v)) return ''; return Number(v).toFixed(d); }
function fmtInt(v) { if (v == null || isNaN(v)) return ''; return Number(v).toLocaleString(); }
function classForSigned(v) { if (v == null || isNaN(v)) return ''; return v > 0 ? 'pos' : (v < 0 ? 'neg' : ''); }
function showTab(host, key) {
  host.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.key === key));
  host.querySelectorAll('.tab-panel').forEach(el => el.style.display = (el.dataset.key === key ? '' : 'none'));
}
function renderTable(data, columns, container) {
  if (!data || data.length === 0) { container.innerHTML = '<div class="muted">No rows.</div>'; return; }
  const wrap = document.createElement('div'); wrap.className = 'scroll';
  const tbl = document.createElement('table');
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  let sortCol = null, sortDir = 'desc';
  columns.forEach((c, idx) => {
    const th = document.createElement('th');
    th.dataset.idx = idx;
    th.innerHTML = c.label + ' <span class="arrow">▾</span>';
    th.onclick = () => {
      if (sortCol === c.key) sortDir = (sortDir === 'asc' ? 'desc' : 'asc');
      else { sortCol = c.key; sortDir = 'desc'; }
      sortAndRender();
      thead.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
      th.classList.add('sorted');
    };
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement('tbody');
  function sortAndRender() {
    const sorted = data.slice();
    if (sortCol) {
      sorted.sort((a,b) => {
        const av = a[sortCol], bv = b[sortCol];
        if (av == null) return 1; if (bv == null) return -1;
        if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av-bv : bv-av;
        return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
    }
    tbody.innerHTML = '';
    sorted.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(c => {
        const td = document.createElement('td');
        let v = row[c.key];
        let cls = '';
        if (c.fmt === 'int')      { td.classList.add('num'); td.textContent = fmtInt(v); }
        else if (c.fmt === 'pct') { td.classList.add('num'); td.textContent = fmtPct(v); cls = c.color ? classForSigned(v) : ''; }
        else if (c.fmt === 'num') { td.classList.add('num'); td.textContent = fmtNum(v, c.decimals != null ? c.decimals : 2); cls = c.color ? classForSigned(v) : ''; }
        else if (c.fmt === 'roi') { td.classList.add('num'); td.textContent = fmtPct(v); cls = classForSigned(v); }
        else                      { td.textContent = (v == null ? '' : String(v)); }
        if (cls) td.classList.add(cls);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
  sortAndRender();
  tbl.appendChild(thead); tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  container.innerHTML = ''; container.appendChild(wrap);
}
const STD_COLUMNS = [
  { key: 'bucket', label: 'Bucket' },
  { key: 'bets', label: 'Bets', fmt: 'int' },
  { key: 'wins', label: 'W', fmt: 'int' },
  { key: 'losses', label: 'L', fmt: 'int' },
  { key: 'pushes', label: 'P', fmt: 'int' },
  { key: 'win_pct', label: 'Win %', fmt: 'pct' },
  { key: 'units_flat', label: 'Units (flat)', fmt: 'num', decimals: 2, color: true },
  { key: 'roi_flat', label: 'ROI flat', fmt: 'roi' },
  { key: 'units_kelly', label: 'Units (Kelly)', fmt: 'num', decimals: 4, color: true },
  { key: 'roi_kelly', label: 'ROI Kelly', fmt: 'roi' },
  { key: 'avg_ev', label: 'Avg EV', fmt: 'pct' },
  { key: 'avg_edge_vs_market_pp', label: 'Avg edge (pp)', fmt: 'num', decimals: 2 },
  { key: 'avg_kelly_pct', label: 'Avg Kelly', fmt: 'pct' },
  { key: 'avg_model_prob', label: 'Avg model p', fmt: 'pct' },
  { key: 'avg_odds_american', label: 'Avg odds', fmt: 'num', decimals: 0 },
];
const STD_COLUMNS_WITH_SIDE = [
  { key: 'side_group', label: 'Side' },
  ...STD_COLUMNS,
];
"""


# =========================
# DATA LOADING
# =========================

def df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    df = df.copy()
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_row_dict(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    rec = df.iloc[0].to_dict()
    return {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}


def collect_league_data(league: str) -> dict:
    """Bundle every report CSV for one league into a dict keyed for the JS frontend."""
    data = {
        "league": league.upper(),
        "grand_total": first_row_dict(safe_read(BASE / f"{league}_summary_grand_total.csv")),
        "by_market_summary": df_to_records(safe_read(BASE / f"{league}_summary_overall.csv")),
        "overview": {
            "by_market":     df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_market.csv")),
            "by_side_group": df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_side_group.csv")),
            "by_date":       df_to_records(safe_read(REPORT_DIR / league / "overview" / f"{league}_summary_by_date.csv")),
        },
        "markets": {},
    }

    for mt in MARKETS:
        mt_dir = REPORT_DIR / league / mt
        market_data = {"by": {}, "by_side": {}}
        for label in ("ev", "kelly", "odds", "win_prob", "edge_vs_market", "dow", "month"):
            no_side = mt_dir / f"{league}_{mt}_by_{label}.csv"
            with_side_sfx = "over_under" if mt == "total" else "home_away"
            with_side = mt_dir / f"{league}_{mt}_by_{label}_{with_side_sfx}_summary.csv"
            market_data["by"][label]      = df_to_records(safe_read(no_side))
            market_data["by_side"][label] = df_to_records(safe_read(with_side))
        # spread/total side and total range
        if mt in ("spread", "total"):
            with_side_sfx = "over_under" if mt == "total" else "home_away"
            market_data["by"]["side"]      = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_side.csv"))
            market_data["by_side"]["side"] = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_side_{with_side_sfx}_summary.csv"))
        if mt == "total":
            market_data["by"]["total_range"]      = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_total_range.csv"))
            market_data["by_side"]["total_range"] = df_to_records(safe_read(mt_dir / f"{league}_{mt}_by_total_range_over_under_summary.csv"))

        data["markets"][mt] = market_data

    return data


# =========================
# HTML BUILDERS
# =========================

def build_index_page() -> str:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    links = "".join(f'<li><a href="{lg}/index.html">{lg.upper()}</a></li>' for lg in LEAGUES)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>Basketball Dashboard</title>
<style>{CSS}</style></head>
<body>
<header><h1>Basketball Dashboard</h1><span class="ts">Built {html.escape(ts)} UTC</span></header>
<main>
<h2>Leagues</h2>
<ul>{links}</ul>
</main></body></html>"""


def build_league_page(league: str, data: dict) -> str:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    league_label = league.upper()

    # KPIs from grand total
    gt = data.get("grand_total", {}) or {}
    def kpi(label: str, value, fmt: str = "raw") -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            disp, cls = "—", ""
        elif fmt == "pct":
            disp = f"{value*100:.2f}%" if value is not None else "—"
            cls = "good" if (value or 0) > 0 else ("bad" if (value or 0) < 0 else "")
        elif fmt == "int":
            disp = f"{int(value):,}"
            cls = ""
        elif fmt == "signed":
            disp = f"{value:+,.2f}"
            cls = "good" if (value or 0) > 0 else ("bad" if (value or 0) < 0 else "")
        else:
            disp = str(value)
            cls = ""
        return f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value {cls}">{html.escape(disp)}</div></div>'

    kpi_html = "".join([
        kpi("Bets",               gt.get("bets"),         "int"),
        kpi("Wins",                gt.get("wins"),         "int"),
        kpi("Losses",              gt.get("losses"),       "int"),
        kpi("Pushes",              gt.get("pushes"),       "int"),
        kpi("Win %",               gt.get("win_pct"),      "pct"),
        kpi("Units (flat)",        gt.get("units_flat"),   "signed"),
        kpi("ROI flat",            gt.get("roi_flat"),     "pct"),
        kpi("Units (Kelly)",       gt.get("units_kelly"),  "signed"),
        kpi("ROI Kelly",           gt.get("roi_kelly"),    "pct"),
    ])

    nav_links = "".join(f'<a href="../{lg}/index.html">{lg.upper()}</a>' for lg in LEAGUES)
    payload = json.dumps(data, default=str)

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>{league_label} Dashboard</title>
<style>{CSS}</style></head>
<body>
<header><h1>{league_label} Dashboard</h1><span class="ts">Built {html.escape(ts)} UTC</span></header>
<nav class="leagues">{nav_links}<a href="../index.html">All</a></nav>
<main>

<h2>Headline</h2>
<div class="kpis">{kpi_html}</div>

<h2>By Market</h2>
<div id="by-market-summary"></div>

<h2>Per Market Drilldown</h2>
<div id="market-area">
  <div class="tabs" id="market-tabs">
    <div class="tab active" data-key="moneyline">Moneyline</div>
    <div class="tab" data-key="spread">Spread</div>
    <div class="tab" data-key="total">Total</div>
  </div>
  <div class="tab-body">
    <div class="tab-panel" data-key="moneyline" id="panel-moneyline"></div>
    <div class="tab-panel" data-key="spread" style="display:none" id="panel-spread"></div>
    <div class="tab-panel" data-key="total" style="display:none" id="panel-total"></div>
  </div>
</div>

<h2>Overview</h2>
<div id="overview-area">
  <div class="tabs" id="overview-tabs">
    <div class="tab active" data-key="market">By market</div>
    <div class="tab" data-key="side">By side group</div>
    <div class="tab" data-key="date">By date</div>
  </div>
  <div class="tab-body">
    <div class="tab-panel" data-key="market" id="overview-market"></div>
    <div class="tab-panel" data-key="side" style="display:none" id="overview-side"></div>
    <div class="tab-panel" data-key="date" style="display:none" id="overview-date"></div>
  </div>
</div>

</main>
<footer>Built from CSVs in <code>{html.escape(str(REPORT_DIR / league))}</code> · click column headers to sort</footer>

<script>
{JS}
const DATA = {payload};

// By-market headline table
(function() {{
  const cols = [
    {{ key: 'market_type', label: 'Market' }},
    {{ key: 'Win', label: 'W', fmt: 'int' }},
    {{ key: 'Loss', label: 'L', fmt: 'int' }},
    {{ key: 'Push', label: 'P', fmt: 'int' }},
    {{ key: 'Total', label: 'Total', fmt: 'int' }},
    {{ key: 'Win_Pct', label: 'Win %', fmt: 'pct' }},
  ];
  renderTable(DATA.by_market_summary, cols, document.getElementById('by-market-summary'));
}})();

// Per-market drilldown
function buildMarketPanel(mt) {{
  const wrap = document.getElementById('panel-' + mt);
  const md = DATA.markets[mt] || {{by:{{}}, by_side:{{}}}};
  const dimensions = Object.keys(md.by);
  const ctrlId = 'ctrl-' + mt;
  const sideId = 'side-' + mt;
  const tableId = 'table-' + mt;

  wrap.innerHTML = `
    <div class="controls">
      <label>Dimension:
        <select id="${{ctrlId}}">
          ${{dimensions.map(d => `<option value="${{d}}">${{d}}</option>`).join('')}}
        </select>
      </label>
      <label>View:
        <select id="${{sideId}}">
          <option value="overall">Overall</option>
          <option value="side">Split by side</option>
        </select>
      </label>
    </div>
    <div id="${{tableId}}"></div>
  `;
  function refresh() {{
    const dim = document.getElementById(ctrlId).value;
    const view = document.getElementById(sideId).value;
    const data = view === 'side' ? (md.by_side[dim] || []) : (md.by[dim] || []);
    const cols = view === 'side' ? STD_COLUMNS_WITH_SIDE : STD_COLUMNS;
    renderTable(data, cols, document.getElementById(tableId));
  }}
  document.getElementById(ctrlId).onchange = refresh;
  document.getElementById(sideId).onchange = refresh;
  refresh();
}}
['moneyline','spread','total'].forEach(buildMarketPanel);

// Tab wiring (markets)
document.querySelectorAll('#market-tabs .tab').forEach(t => {{
  t.onclick = () => showTab(document.getElementById('market-area'), t.dataset.key);
}});

// Overview
(function() {{
  renderTable(DATA.overview.by_market, [
    {{ key: 'market_type', label: 'Market' }},
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], document.getElementById('overview-market'));
  renderTable(DATA.overview.by_side_group, [
    {{ key: 'side_group', label: 'Side' }},
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], document.getElementById('overview-side'));
  renderTable(DATA.overview.by_date, [
    {{ key: 'bucket', label: 'Date' }},
    ...STD_COLUMNS.filter(c => c.key !== 'bucket'),
  ], document.getElementById('overview-date'));
}})();
document.querySelectorAll('#overview-tabs .tab').forEach(t => {{
  t.onclick = () => showTab(document.getElementById('overview-area'), t.dataset.key);
}});
</script>
</body></html>"""
    return page


# =========================
# MAIN
# =========================

def run():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Top-level chooser
    (DASHBOARD_DIR / "index.html").write_text(build_index_page(), encoding="utf-8")

    for league in LEAGUES:
        out_dir = DASHBOARD_DIR / league
        out_dir.mkdir(parents=True, exist_ok=True)
        data = collect_league_data(league)
        (out_dir / "index.html").write_text(build_league_page(league, data), encoding="utf-8")
        print(f"[{league}] dashboard -> {out_dir / 'index.html'}")

    print("Dashboard build complete.")


if __name__ == "__main__":
    run()
