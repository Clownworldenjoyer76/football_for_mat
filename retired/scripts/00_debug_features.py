#!/usr/bin/env python3
# /scripts/00_debug_features.py
# Prints feature columns and suggests best column-name matches for desired targets.
import sys, re, json
from pathlib import Path
import pandas as pd
import yaml

FEATS = Path("data/features/weekly_clean.csv.gz")
CONFIG = Path("config/models.yml")
OUT_TXT = Path("output/logs/features_columns.txt")
OUT_SUG = Path("output/logs/targets_suggestions.json")
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def tokens(s: str) -> set[str]:
    s = s.lower()
    parts = re.split(r"[^a-z0-9]+", s)
    return set([t for t in parts if t])

def score(want: str, cand: str) -> float:
    wn, cn = norm(want), norm(cand)
    if not cn: return 0.0
    base = 0.0
    if wn == cn: base += 3.0
    if wn in cn or cn in wn: base += 2.0
    tw, tc = tokens(want), tokens(cand)
    inter = len(tw & tc)
    union = max(1, len(tw | tc))
    base += 3.0 * (inter / union)
    if ("yard" in cand or "yd" in cand): base += 0.5
    role_map = {
        "qb_passing_yards": ["qb","pass","yard","yd"],
        "rb_rushing_yards": ["rb","rush","yard","yd"],
        "wr_rec_yards": ["wr","receiv","rec","yard","yd"],
        "wrte_receptions": ["wr","te","rec","recept","catch"],
    }
    for k, cues in role_map.items():
        if norm(k) == wn and any(cue in cand.lower() for cue in cues):
            base += 0.5
    return base

def main():
    if not FEATS.exists():
        print(f"[FATAL] Features file not found: {FEATS}", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
        desired = list(cfg.get("targets", []))
    except Exception:
        desired = []

    df = pd.read_csv(FEATS, nrows=100)
    cols = list(df.columns)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        for c in cols:
            f.write(c + "\n")
    print(f"[OK] Wrote {OUT_TXT} ({len(cols)} columns)")

    sugg = {}
    for want in desired:
        scored = sorted(cols, key=lambda c: score(want, c), reverse=True)
        sugg[want] = [{"column": c, "score": round(score(want, c), 3)} for c in scored[:10]]
    OUT_SUG.write_text(json.dumps(sugg, indent=2))
    print(f"[OK] Suggested top-10 matches per target -> {OUT_SUG}")

    for want in desired:
        top = ", ".join([s['column'] for s in sugg[want][:5]])
        print(f"[SUGGEST] {want}: {top}")

if __name__ == "__main__":
    main()
