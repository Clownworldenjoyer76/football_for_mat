#!/usr/bin/env python3
# /scripts/00_debug_features.py
import sys
from pathlib import Path
import pandas as pd

FEATS = Path("data/features/weekly_clean.csv.gz")
OUT = Path("output/logs/features_columns.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

if not FEATS.exists():
    print(f"[FATAL] Features file not found: {FEATS}", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(FEATS, nrows=5)
cols = list(df.columns)
with OUT.open("w", encoding="utf-8") as f:
    for c in cols:
        f.write(c + "\n")

print(f"[OK] Wrote {OUT} ({len(cols)} columns)")
TARGETS = {"qb_passing_yards","rb_rushing_yards","wr_rec_yards","wrte_receptions"}
present = TARGETS.intersection(cols)
missing = TARGETS.difference(cols)
print(f"[DEBUG] Present targets: {sorted(present)}")
print(f"[DEBUG] Missing targets: {sorted(missing)}")
