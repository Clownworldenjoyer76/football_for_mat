#!/usr/bin/env python3
from pathlib import Path
import time, json, gzip, re
import requests
import pandas as pd

OUT = Path("data/odds"); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "_dk_summary.txt"
LOG = OUT / "_dk_errors.log"

EVENTGROUP_IDS = [88808, 88670846, 42648, 84240]
MARKETS = {
    "qb_passing_yards": ["player passing yards","passing yards"],
    "rb_rushing_yards": ["player rushing yards","rushing yards"],
    "wr_rec_yards":     ["player receiving yards","receiving yards"],
    "wrte_receptions":  ["player receptions","receptions"],
}
HOSTS = [
    "https://sportsbook.draftkings.com/sites/US-SB",    # v5/v4
    "https://sportsbook.draftkings.com/sites"           # fallback
]
UA = {"User-Agent": "Mozilla/5.0 (OddsPuller/1.4)"}

def log(msg):
    ts = pd.Timestamp.utcnow().isoformat()
    with LOG.open("a", encoding="utf-8") as f: f.write(f"{ts} {msg}\n")
    print(msg, flush=True)

def wr_summary(lines):
    with SUMMARY.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def get_json(url):
    try:
        r = requests.get(url, headers=UA, timeout=25)
        code = r.status_code
        try:
            j = r.json()
        except Exception:
            j = None
        return code, j
    except Exception as e:
        log(f"[err] GET {url} failed: {e}")
        return None, None

def dump_gz(obj, path):
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f: json.dump(obj, f)
    except Exception as e:
        log(f"[err] dump_gz {path.name} failed: {e}")

def text_match(s, needles):
    s = (s or "").lower()
    return any(n in s for n in needles)

def nearest_half(x):
    try: return round(float(x)*2)/2
    except: return None

def american_to_float(x):
    try: return float(x)
    except: return None

def normalize_name(s):
    if not isinstance(s, str): return ""
    return s.replace(" Jr.", " Jr").replace(".", "").strip()

def yield_offers_v5(j):
    eg = (j or {}).get("eventGroup") or {}
    for cat in eg.get("offerCategories", []) or []:
        cname = (cat.get("name") or "")
        for sub in cat.get("offerSubcategoryDescriptors", []) or []:
            sname = (sub.get("name") or "")
            offers = (sub.get("offerSubcategory") or {}).get("offers", []) or []
            for offer_list in offers:
                yield cname, sname, offer_list

def yield_offers_v4(cat_json):
    cname = (cat_json.get("name") or "")
    for sub in cat_json.get("subcategories", []) or []:
        sname = (sub.get("name") or "")
        for offer_list in sub.get("offers", []) or []:
            yield cname, sname, offer_list

def parse_offer_list(offer_list):
    player = team = None; line = over = under = None
    for o in offer_list:
        lbl = (o.get("label") or "")
        player = player or o.get("participant") or lbl
        team = team or o.get("teamAbbreviation") or o.get("team")
        line = line or o.get("line") or o.get("handicap")
        if line is None and lbl:
            m = re.search(r"(\d+(?:\.\d+)?)", lbl)
            if m: line = float(m.group(1))
    for o in offer_list:
        lbl = (o.get("label") or "").lower()
        odds = o.get("oddsAmerican")
        if "over" in lbl:  over  = american_to_float(odds)
        if "under" in lbl: under = american_to_float(odds)
    return normalize_name(player), team, nearest_half(line), over, under

def collect_rows(offers_iter, market, needles):
    rows = []
    scans = 0
    for cname, sname, offer_list in offers_iter:
        scans += 1
        ok = text_match(cname, needles) or text_match(sname, needles)
        if not ok and offer_list:
            ok = text_match((offer_list[0].get("label") or ""), needles)
        if not ok: continue
        player, team, line, over, under = parse_offer_list(offer_list)
        if not player or line is None or (over is None and under is None): continue
        rows.append({
            "player_name": player, "team": team, "prop": market,
            "line": line, "over_odds": over, "under_odds": under,
            "book": "DK", "season": pd.NA, "week": pd.NA
        })
    df = pd.DataFrame(rows)
    return df, scans

def pull_once(egid):
    lines = [f"== EG {egid} =="]
    # v5
    v5_code = v5_count = 0
    for host in HOSTS:
        url5 = f"{host}/api/v5/eventgroups/{egid}?format=json"
        code, j = get_json(url5)
        v5_code = code or 0
        lines.append(f"v5 GET {url5} -> {code}")
        if code == 200 and j:
            dump_gz(j, OUT / f"_dk_eventgroup_{egid}.json.gz")
            offer_names = [ (c or "", s or "") for c,s,_ in yield_offers_v5(j) ]
            v5_count = len(offer_names)
            lines.append(f"v5 offers nodes: {v5_count}")
            if v5_count:
                lines += [f"  - {c} / {s}" for c,s in offer_names[:10]]
            break
        time.sleep(0.5)

    # v4 per-category
    v4_found = 0
    if v5_code == 200 and j:
        cat_ids = []
        for cat in (j.get("eventGroup") or {}).get("offerCategories", []) or []:
            cid = cat.get("offerCategoryId")
            if isinstance(cid, int): cat_ids.append(cid)
        for cid in sorted(set(cat_ids)):
            for host in HOSTS:
                url4 = f"{host}/api/v4/eventgroups/{egid}/categories/{cid}?format=json"
                code4, j4 = get_json(url4)
                lines.append(f"v4 GET cat {cid} -> {code4}")
                if code4 == 200 and j4:
                    dump_gz(j4, OUT / f"_dk_category_{egid}_{cid}.json.gz")
                    v4_found += 1
                    break
            time.sleep(0.2)
    return lines

def main():
    overall = []
    any_rows = False

    for egid in EVENTGROUP_IDS:
        overall += pull_once(egid)

        # load combined offers (from latest v5 + all v4 we just dumped)
        offers = []
        ev = OUT / f"_dk_eventgroup_{egid}.json.gz"
        if ev.exists():
            with gzip.open(ev, "rt", encoding="utf-8") as f:
                j = json.load(f)
            offers += list(yield_offers_v5(j))
        for gz in sorted(OUT.glob(f"_dk_category_{egid}_*.json.gz")):
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                j4 = json.load(f)
            offers += list(yield_offers_v4(j4))

        overall.append(f"combined offers lists: {len(offers)}")

        for market, needles in MARKETS.items():
            df, scans = collect_rows(offers, market, needles)
            out = OUT / f"dk_{market}.csv"
            df.to_csv(out, index=False)
            overall.append(f"{market}: scans={scans} rows={len(df)} -> {out.name}")
            if len(df): any_rows = True

        # if we got any rows, stop trying other EGs
        if any_rows: break
        time.sleep(0.8)

    wr_summary(overall)
    if not any_rows:
        log("[warn] No DK player-prop rows found. See data/odds/_dk_summary.txt and _dk_errors.log")

if __name__ == "__main__":
    main()
