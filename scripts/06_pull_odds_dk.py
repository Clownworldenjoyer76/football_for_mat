#!/usr/bin/env python3
from pathlib import Path
import time, json, gzip, re
import requests
import pandas as pd

OUT_DIR = Path("data/odds"); OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTGROUP_IDS = [88808, 88670846, 42648, 84240]
MARKETS = {
    "qb_passing_yards": ["player passing yards","passing yards"],
    "rb_rushing_yards": ["player rushing yards","rushing yards"],
    "wr_rec_yards":     ["player receiving yards","receiving yards"],
    "wrte_receptions":  ["player receptions","receptions"],
}
HOSTS = [
    "https://sportsbook.draftkings.com",      # v5/v4 documented in the wild
    "https://sportsbook.draftkings.com/sites" # fallback (older patterns)
]
UA = {"User-Agent": "Mozilla/5.0 (OddsPuller/1.2)"}
LOG = OUT_DIR / "_dk_errors.log"

def log(msg: str):
    ts = pd.Timestamp.utcnow().isoformat()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")
    print(msg, flush=True)

def fetch_json(url: str):
    try:
        r = requests.get(url, headers=UA, timeout=25)
        if r.status_code != 200:
            log(f"[warn] GET {url} -> {r.status_code}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"[err] GET {url} failed: {e}")
        return None

def dump_gz(obj, path: Path):
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception as e:
        log(f"[err] dump_gz {path.name} failed: {e}")

def text_match(s: str, needles):
    s = (s or "").lower()
    return any(n in s for n in needles)

def yield_offers_v5(j):
    eg = (j or {}).get("eventGroup") or {}
    for cat in eg.get("offerCategories", []) or []:
        cname = (cat.get("name") or "")
        for sub in cat.get("offerSubcategoryDescriptors", []) or []:
            sname = (sub.get("name") or "")
            offers = (sub.get("offerSubcategory") or {}).get("offers", []) or []
