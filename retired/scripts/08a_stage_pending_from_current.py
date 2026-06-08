#!/usr/bin/env python3
"""
Create data/props/history_pending.csv from data/props/props_current.csv.

It picks the minimal columns finalize needs:
  season, week, market, player_id (or player_name), line, prob_over, run_ts
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

SRC = Path("data/props/props_current.csv")
OUT = Path("data/props/history_pending.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise SystemExit(f"ERROR: {SRC} not found. Run your props generation first.")

df = pd.read_csv(SRC)

# market column: prefer explicit 'market'; else copy from 'prop'
if "market" not in df.columns:
    if "prop" in df.columns:
        df["market"] = df["prop"].astype(str)
    else:
        raise SystemExit("ERROR: Neither 'market' nor 'prop' present in props_current.csv")

# IDs: prefer player_id if present; else keep player_name (finalize can fall back)
id_cols = []
if "player_id" in df.columns:
    id_cols.append("player_id")
if "player_name" in df.columns:
    id_cols.append("player_name")

if not id_cols:
    raise SystemExit("ERROR: Need at least 'player_id' or 'player_name' in props_current.csv")

need = ["season","week","market","line","prob_over"] + id_cols
missing = [c for c in ["season","week","line","prob_over"] if c not in df.columns]
if missing:
    raise SystemExit(f"ERROR: Missing required columns in props_current.csv: {missing}")

out = df[need].copy()
out["run_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
out.to_csv(OUT, index=False)
print(f"✓ Wrote {OUT} with {len(out)} rows")
