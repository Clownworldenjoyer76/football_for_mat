import os
import pandas as pd

url = "https://www.espn.com/nfl/injuries"

tables = pd.read_html(url)  # requires lxml
rows = sum(len(t) for t in tables)
print(f"[probe] ESPN injuries: {len(tables)} tables, {rows} rows")

os.makedirs("data/raw/injuries", exist_ok=True)
out = pd.concat(tables, ignore_index=True, sort=False)
out.to_csv("data/raw/injuries/_probe_espn_injuries.csv", index=False)
print("[probe] wrote data/raw/injuries/_probe_espn_injuries.csv")
