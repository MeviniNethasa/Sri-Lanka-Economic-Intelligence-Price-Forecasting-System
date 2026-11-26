"""Download LKR/USD exchange rates from FRED (Federal Reserve Board H.10 series)."""
import io
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SERIES = {
    "EXSLUS": "fred_lkr_usd_monthly.csv",   # monthly average, LKR per USD
    "DEXSLUS": "fred_lkr_usd_daily.csv",    # daily noon buying rate, LKR per USD
}

for sid, fname in SERIES.items():
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "lkr_per_usd"]
    df["lkr_per_usd"] = pd.to_numeric(df["lkr_per_usd"], errors="coerce")
    df.to_csv(RAW / fname, index=False)
    print(sid, df["date"].min(), df["date"].max(), len(df))
