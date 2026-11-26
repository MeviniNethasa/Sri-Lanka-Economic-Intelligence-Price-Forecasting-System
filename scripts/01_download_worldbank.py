"""Download annual Sri Lanka macro indicators from the World Bank WDI API."""
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "FP.CPI.TOTL": "cpi_index_2010_100",
    "NY.GDP.DEFL.KD.ZG": "inflation_gdp_deflator_pct",
    "PA.NUS.FCRF": "exchange_rate_lkr_per_usd",
    "FR.INR.LEND": "lending_rate_pct",
    "FR.INR.DPST": "deposit_rate_pct",
    "FR.INR.RINR": "real_interest_rate_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "NV.AGR.TOTL.ZS": "agriculture_share_gdp_pct",
    "AG.PRD.FOOD.XD": "food_production_index",
    "AG.PRD.CREL.MT": "cereal_production_tonnes",
    "AG.YLD.CREL.KG": "cereal_yield_kg_per_ha",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "FI.RES.TOTL.CD": "total_reserves_usd",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "BX.TRF.PWKR.CD.DT": "remittances_usd",
    "GC.DOD.TOTL.GD.ZS": "central_govt_debt_pct_gdp",
    "FM.LBL.BMNY.ZG": "broad_money_growth_pct",
    "SP.POP.TOTL": "population",
}


def fetch(code: str) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/LKA/indicator/{code}"
    r = requests.get(url, params={"format": "json", "per_page": 200, "date": "1990:2026"}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", code])
    rows = [{"year": int(x["date"]), code: x["value"]} for x in payload[1]]
    return pd.DataFrame(rows)


def main() -> None:
    out = None
    for code, name in INDICATORS.items():
        df = fetch(code).rename(columns={code: name})
        out = df if out is None else out.merge(df, on="year", how="outer")
        print(f"{code:22s} {name:32s} non-null={df[name].notna().sum()}")
    out = out.sort_values("year")
    out.to_csv(RAW / "worldbank_wdi_lka_annual.csv", index=False)
    print("saved", len(out), "rows")


if __name__ == "__main__":
    main()
