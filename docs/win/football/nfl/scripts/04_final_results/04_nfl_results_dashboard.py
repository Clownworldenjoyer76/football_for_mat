#!/usr/bin/env python3
"""Build a self-contained NFL graded-bets dashboard from generated report CSVs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import html
import json
import traceback

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
BASE = NFL_ROOT / "04_final_results"
REPORTS = BASE / "reports"
OVERVIEW = REPORTS / "overview"
OUTPUT_FILE = Path("frontend/nfl_dashboard.html")
ERROR_DIR = NFL_ROOT / "errors" / "04_final_results"
LOG_FILE = ERROR_DIR / "04_nfl_results_dashboard.txt"

MARKETS = {
    "moneyline": {
        "label": "Moneyline", "directory": "moneyline", "file_key": "moneyline",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "week"],
    },
    "spread": {
        "label": "Spread", "directory": "spread", "file_key": "spread",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "spread_range", "line", "week", "side"],
    },
    "total": {
        "label": "Total", "directory": "totals", "file_key": "total",
        "dimensions": ["ev", "odds", "kelly", "win_prob", "total_range", "line", "week", "side"],
    },
}

ERROR_DIR.mkdir(parents=True, exist_ok=True)
RUN_STARTED = datetime.now(timezone.utc)
WARNING_COUNT = 0
INPUT_FILE_COUNT = 0
INPUT_ROW_COUNT = 0


def reset_log() -> None:
    LOG_FILE.write_text(
        "=== 04_nfl_results_dashboard ===\n"
        f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n",
        encoding="utf-8",
    )


def log(level: str, message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} | {level} | {message}\n")


def warn(message: str) -> None:
    global WARNING_COUNT
    WARNING_COUNT += 1
    log("WARNING", message)


def safe_read(path: Path, required: bool = False) -> pd.DataFrame:
    global INPUT_FILE_COUNT, INPUT_ROW_COUNT
    INPUT_FILE_COUNT += 1
    if not path.exists():
        if required:
            warn(f"Required dashboard input missing: {path}")
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        INPUT_ROW_COUNT += len(frame)
        log("INFO", f"INPUT | file={path} | rows={len(frame)}")
        return frame
    except Exception as exc:
        warn(f"Unable to read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def first_row(frame: pd.DataFrame) -> dict:
    rows = records(frame.head(1))
    return rows[0] if rows else {}


def collect_data() -> dict:
    data = {
        "headline": first_row(safe_read(OVERVIEW / "nfl_summary_overall.csv", required=True)),
        "by_market_summary": records(safe_read(BASE / "nfl_summary_overall.csv", required=True)),
        "overview": {
            "by_market": records(safe_read(OVERVIEW / "nfl_summary_by_market.csv")),
            "by_side_group": records(safe_read(OVERVIEW / "nfl_summary_by_side_group.csv")),
            "by_week": records(safe_read(OVERVIEW / "nfl_summary_by_week.csv")),
            "by_date": records(safe_read(OVERVIEW / "nfl_summary_by_date.csv")),
            "by_season_type": records(safe_read(OVERVIEW / "nfl_summary_by_season_type.csv")),
            "by_day_night": records(safe_read(OVERVIEW / "nfl_summary_by_day_night.csv")),
            "bet_log": records(safe_read(OVERVIEW / "nfl_bet_log.csv")),
        },
        "markets": {},
    }

    for market, cfg in MARKETS.items():
        market_dir = REPORTS / cfg["directory"]
        market_data = {"by": {}, "by_side": {}}
        for dimension in cfg["dimensions"]:
            base_name = f"nfl_{cfg['file_key']}_by_{dimension}"
            market_data["by"][dimension] = records(safe_read(market_dir / f"{base_name}.csv"))
            market_data["by_side"][dimension] = records(safe_read(market_dir / f"{base_name}_side_summary.csv"))
        data["markets"][market] = market_data
    return data


CSS = r"""
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1f2630;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--good:#3fb950;--bad:#f85149;--accent:#58a6ff;--head:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}
header{padding:20px 24px;border-bottom:1px solid var(--border);background:#010409;display:flex;align-items:baseline;justify-content:space-between;gap:20px;flex-wrap:wrap}h1{margin:0;font-size:24px}h2{margin:26px 0 12px;font-size:18px}h3{margin:0 0 12px;font-size:15px}.ts{color:var(--muted);font-size:12px}main{max-width:1600px;margin:0 auto;padding:20px 24px 48px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:13px}.kpi .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.kpi .value{font-size:21px;font-weight:650;margin-top:4px}.good{color:var(--good)!important}.bad{color:var(--bad)!important}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.tab{padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--muted);cursor:pointer;user-select:none}.tab.active{background:#1f6feb;color:white;border-color:#1f6feb}.tab-body,.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}select{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:6px 8px}label{color:var(--muted)}
.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--border);border-radius:6px}table{border-collapse:collapse;width:100%;min-width:840px;background:var(--panel)}th,td{padding:8px 10px;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left}th{position:sticky;top:0;background:var(--head);z-index:1;color:#c9d1d9;font-size:12px;cursor:pointer}td.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:hover{background:#1b222c}.empty{color:var(--muted);padding:18px;text-align:center}.section-note{color:var(--muted);font-size:12px;margin-top:-6px;margin-bottom:12px}
@media(max-width:700px){header,main{padding-left:12px;padding-right:12px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

JS = r"""
function n(v){const x=Number(v);return Number.isFinite(x)?x:null}
function fmtInt(v){const x=n(v);return x===null?'—':Math.round(x).toLocaleString()}
function fmtNum(v,d=2){const x=n(v);return x===null?'—':x.toFixed(d)}
function fmtPct(v){const x=n(v);return x===null?'—':(x*100).toFixed(1)+'%'}
function signedClass(v){const x=n(v);if(x===null||x===0)return'';return x>0?'good':'bad'}
function showTab(root,key){root.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.classList.toggle('active',t.dataset.key===key));root.querySelectorAll(':scope > .tab-body > .tab-panel').forEach(p=>p.style.display=p.dataset.key===key?'block':'none')}
function valueText(v,fmt,decimals){if(v===null||v===undefined||v==='')return'—';if(fmt==='int')return fmtInt(v);if(fmt==='pct')return fmtPct(v);if(fmt==='num')return fmtNum(v,decimals==null?2:decimals);return String(v)}
function renderTable(rows,columns,container){
  if(!rows||rows.length===0){container.innerHTML='<div class="empty">No data</div>';return}
  const wrap=document.createElement('div');wrap.className='table-wrap';const table=document.createElement('table');const thead=document.createElement('thead');const hr=document.createElement('tr');
  let sortKey=null,sortAsc=true;
  columns.forEach(c=>{const th=document.createElement('th');th.textContent=c.label;th.onclick=()=>{if(sortKey===c.key)sortAsc=!sortAsc;else{sortKey=c.key;sortAsc=true}draw()};hr.appendChild(th)});thead.appendChild(hr);const tbody=document.createElement('tbody');
  function draw(){tbody.innerHTML='';let data=[...rows];if(sortKey){data.sort((a,b)=>{const av=a[sortKey],bv=b[sortKey],an=n(av),bn=n(bv);let cmp;if(an!==null&&bn!==null)cmp=an-bn;else cmp=String(av??'').localeCompare(String(bv??''),undefined,{numeric:true});return sortAsc?cmp:-cmp})}data.forEach(row=>{const tr=document.createElement('tr');columns.forEach(c=>{const td=document.createElement('td');if(c.fmt)td.classList.add('num');td.textContent=valueText(row[c.key],c.fmt,c.decimals);if(c.color){const cls=signedClass(row[c.key]);if(cls)td.classList.add(cls)}tr.appendChild(td)});tbody.appendChild(tr)})}
  draw();table.appendChild(thead);table.appendChild(tbody);wrap.appendChild(table);container.innerHTML='';container.appendChild(wrap)
}
const METRICS=[
 {key:'variable',label:'Bucket'}, {key:'Win',label:'W',fmt:'int'}, {key:'Loss',label:'L',fmt:'int'}, {key:'Push',label:'P',fmt:'int'}, {key:'Total',label:'Total',fmt:'int'},
 {key:'Win_Pct',label:'Win %',fmt:'pct'}, {key:'units',label:'Units',fmt:'num',decimals:2,color:true}, {key:'ROI_Excluding_Pushes',label:'ROI excl',fmt:'pct',color:true},
 {key:'ROI_Including_Pushes',label:'ROI incl',fmt:'pct',color:true}, {key:'avg_ev',label:'Avg EV',fmt:'pct'}, {key:'avg_odds',label:'Avg odds',fmt:'num',decimals:0}, {key:'avg_model_prob',label:'Avg model',fmt:'pct'}
];
const SIDE_METRICS=[{key:'side_group',label:'Side'},...METRICS];
const SUMMARY_METRICS=[
 {key:'Win',label:'W',fmt:'int'}, {key:'Loss',label:'L',fmt:'int'}, {key:'Push',label:'P',fmt:'int'}, {key:'Total',label:'Total',fmt:'int'}, {key:'Win_Pct',label:'Win %',fmt:'pct'},
 {key:'units',label:'Units',fmt:'num',decimals:2,color:true}, {key:'ROI_Excluding_Pushes',label:'ROI excl',fmt:'pct',color:true}, {key:'ROI_Including_Pushes',label:'ROI incl',fmt:'pct',color:true},
 {key:'avg_ev',label:'Avg EV',fmt:'pct'}, {key:'avg_odds',label:'Avg odds',fmt:'num',decimals:0}, {key:'avg_model_prob',label:'Avg model',fmt:'pct'}
];
function kpi(label,value,fmt,color=false){let cls=color?signedClass(value):'';return '<div class="kpi"><div class="label">'+label+'</div><div class="value '+cls+'">'+valueText(value,fmt,2)+'</div></div>'}
function build(data){
 const h=data.headline||{};document.querySelector('.kpis').innerHTML=[kpi('Bets',h.Total,'int'),kpi('Wins',h.Win,'int'),kpi('Losses',h.Loss,'int'),kpi('Pushes',h.Push,'int'),kpi('Win %',h.Win_Pct,'pct'),kpi('Units',h.units,'num',true),kpi('ROI excl pushes',h.ROI_Excluding_Pushes,'pct',true),kpi('ROI incl pushes',h.ROI_Including_Pushes,'pct',true),kpi('Avg EV',h.avg_ev,'pct'),kpi('Avg odds',h.avg_odds,'num')].join('');
 renderTable(data.by_market_summary||[],[{key:'market_type',label:'Market'},...SUMMARY_METRICS],document.querySelector('.by-market-summary'));
 ['moneyline','spread','total'].forEach(m=>{const panel=document.querySelector('.panel-'+m);const md=(data.markets||{})[m]||{by:{},by_side:{}};const dims=Object.keys(md.by||{});panel.innerHTML='<div class="controls"><label>Dimension <select class="dim">'+dims.map(d=>'<option value="'+d+'">'+d.replaceAll('_',' ')+'</option>').join('')+'</select></label><label>View <select class="view"><option value="overall">Overall</option><option value="side">Split by side</option></select></label></div><div class="target"></div>';const dim=panel.querySelector('.dim'),view=panel.querySelector('.view'),target=panel.querySelector('.target');function refresh(){const side=view.value==='side',key=dim.value;renderTable(side?(md.by_side[key]||[]):(md.by[key]||[]),side?SIDE_METRICS:METRICS,target)}dim.onchange=refresh;view.onchange=refresh;refresh()});
 const marketArea=document.querySelector('.market-area');marketArea.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.onclick=()=>showTab(marketArea,t.dataset.key));
 const ov=data.overview||{};renderTable(ov.by_market||[],[{key:'variable',label:'Market'},...SUMMARY_METRICS],document.querySelector('.ov-market'));renderTable(ov.by_side_group||[],[{key:'variable',label:'Side'},...SUMMARY_METRICS],document.querySelector('.ov-side'));renderTable(ov.by_week||[],[{key:'variable',label:'Week'},...SUMMARY_METRICS,{key:'cumulative_units',label:'Cum units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-week'));renderTable(ov.by_date||[],[{key:'variable',label:'Date'},...SUMMARY_METRICS,{key:'cumulative_units',label:'Cum units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-date'));renderTable(ov.by_season_type||[],[{key:'variable',label:'Season type'},...SUMMARY_METRICS],document.querySelector('.ov-season'));renderTable(ov.by_day_night||[],[{key:'variable',label:'Day/Night'},...SUMMARY_METRICS],document.querySelector('.ov-daynight'));
 renderTable(ov.bet_log||[],[{key:'season',label:'Season'},{key:'week',label:'Week'},{key:'game_date',label:'Date'},{key:'away_team',label:'Away'},{key:'home_team',label:'Home'},{key:'market_type',label:'Market'},{key:'bet_side',label:'Side'},{key:'line',label:'Line',fmt:'num',decimals:1},{key:'odds_american',label:'Odds',fmt:'num',decimals:0},{key:'model_prob',label:'Model',fmt:'pct'},{key:'ev',label:'EV',fmt:'pct'},{key:'bet_result',label:'Result'},{key:'bet_units',label:'Units',fmt:'num',decimals:2,color:true}],document.querySelector('.ov-log'));
 const overview=document.querySelector('.overview-area');overview.querySelectorAll(':scope > .tabs .tab').forEach(t=>t.onclick=()=>showTab(overview,t.dataset.key));
}
"""


def build_html(data: dict) -> str:
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Graded Bets Dashboard</title><style>{CSS}</style></head>
<body><header><h1>NFL Graded Bets Dashboard</h1><span class="ts">Built {html.escape(built_at)} UTC</span></header><main>
<div class="kpis"></div>
<h2>By Market</h2><div class="by-market-summary panel"></div>
<h2>Per-Market Drilldown</h2><div class="market-area"><div class="tabs"><div class="tab active" data-key="moneyline">Moneyline</div><div class="tab" data-key="spread">Spread</div><div class="tab" data-key="total">Total</div></div><div class="tab-body"><div class="tab-panel panel-moneyline" data-key="moneyline"></div><div class="tab-panel panel-spread" data-key="spread" style="display:none"></div><div class="tab-panel panel-total" data-key="total" style="display:none"></div></div></div>
<h2>Overview</h2><div class="section-note">Win percentage and ROI excluding pushes use wins + losses as the denominator.</div><div class="overview-area"><div class="tabs"><div class="tab active" data-key="market">Market</div><div class="tab" data-key="side">Side</div><div class="tab" data-key="week">Week</div><div class="tab" data-key="date">Date</div><div class="tab" data-key="season">Season type</div><div class="tab" data-key="daynight">Day/Night</div><div class="tab" data-key="log">Bet log</div></div><div class="tab-body"><div class="tab-panel ov-market" data-key="market"></div><div class="tab-panel ov-side" data-key="side" style="display:none"></div><div class="tab-panel ov-week" data-key="week" style="display:none"></div><div class="tab-panel ov-date" data-key="date" style="display:none"></div><div class="tab-panel ov-season" data-key="season" style="display:none"></div><div class="tab-panel ov-daynight" data-key="daynight" style="display:none"></div><div class="tab-panel ov-log" data-key="log" style="display:none"></div></div></div>
</main><script>{JS}\nconst DATA={payload};document.addEventListener('DOMContentLoaded',()=>build(DATA));</script></body></html>"""


def main() -> None:
    reset_log()
    status = "FAILED"
    try:
        data = collect_data()
        page = build_html(data)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(page, encoding="utf-8")
        log("INFO", f"OUTPUT | file={OUTPUT_FILE} | bytes={len(page.encode('utf-8'))}")
        status = "SUCCESS"
        print(f"NFL dashboard complete. output={OUTPUT_FILE}")
    except Exception as exc:
        log("ERROR", f"Unhandled exception: {type(exc).__name__}: {exc}")
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
    finally:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"INPUT_SUMMARY | files={INPUT_FILE_COUNT} | rows={INPUT_ROW_COUNT}\n")
            handle.write(f"WARNING_COUNT: {WARNING_COUNT}\n")
            handle.write(f"STATUS: {status}\n")


if __name__ == "__main__":
    main()
