#!/usr/bin/env python3
import os
import pandas as pd

URL = "https://www.espn.com/nfl/injuries"

def main():
    tables = pd.read_html(URL)  # needs lxml
    rows = sum(len(t) for t in tables)
    print(f"[probe] ESPN injuries: {len(tables)} tables, {rows} rows")

    os.makedirs("data/raw/injuries", exist_ok=True)
    out = pd.concat(tables, ignore_index=True, sort=False)
    out.to_csv("data/raw/injuries/_probe_espn_injuries.csv", index=False)
    print("[probe] wrote data/raw/injuries/_probe_espn_injuries.csv")

if __name__ == "__main__":
    main()
