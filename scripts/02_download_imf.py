"""Download monthly Sri Lanka series from the IMF SDMX API.

Datasets:
  CPI     - Sri Lanka consumer price index (headline + COICOP divisions)
  MFS_IR  - Sri Lanka interest rates (policy, T-bill, lending, deposit ...)
  ER      - LKR exchange rates (USD, EUR, SDR)
  PCPS    - World primary commodity prices (rice, sugar, crude oil, coconut oil ...)
"""
import io
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = "https://api.imf.org/external/sdmx/2.1/data"
HEADERS = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

# Metadata columns repeated on every row; dropped to keep files small.
DROP = {
    "FULL_DESCRIPTION", "METHODOLOGY", "METHODOLOGY_NOTES", "FULL_SOURCE_CITATION",
    "SUGGESTED_CITATION", "LICENSE", "KEYWORDS", "KEYWORDS_DATASET", "SHORT_SOURCE_CITATION",
    "SOURCE", "AUTHOR", "PUBLISHER", "DEPARTMENT", "CONTACT_POINT", "TOPIC", "TOPIC_DATASET",
    "LANGUAGE", "PUBLICATION_DATE", "UPDATE_DATE", "ACCESS_SHARING_LEVEL", "ACCESS_SHARING_NOTES",
    "SECURITY_CLASSIFICATION", "DOI", "KEY_INDICATOR",
}

JOBS = [
    ("IMF.STA,CPI", "LKA", "imf_cpi_lka.csv"),
    ("IMF.STA,MFS_IR", "LKA", "imf_interest_rates_lka.csv"),
    ("IMF.STA,ER", "LKA", "imf_exchange_rates_lka.csv"),
    ("IMF.RES,PCPS", "G001", "imf_world_commodity_prices.csv"),
]


def fetch(flow: str, first: str) -> pd.DataFrame:
    """Wildcard every dimension after the first; find the key length by trial."""
    for n in range(2, 9):
        key = ".".join([first] + [""] * (n - 1))
        r = requests.get(f"{BASE}/{flow}/{key}", params={"startPeriod": "2000"},
                         headers=HEADERS, timeout=300)
        if r.ok and len(r.text) > 200:
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            return df[[c for c in df.columns if c not in DROP]]
    raise RuntimeError(f"no data for {flow}")


def main() -> None:
    for flow, first, fname in JOBS:
        df = fetch(flow, first)
        if "FREQUENCY" in df.columns:  # keep monthly only
            df = df[df["FREQUENCY"] == "M"]
        df.to_csv(RAW / fname, index=False)
        print(f"{flow:16s} -> {fname} {df.shape}")


if __name__ == "__main__":
    main()
