"""Clean raw downloads and build analysis-ready datasets in data/processed/.

Outputs
  commodity_prices_market_monthly.csv  WFP prices, one row per market x commodity x month
  commodity_prices_national_monthly.csv  median across markets, wide-ish long format
  cbsl_daily_prices.csv                CBSL daily wholesale/retail prices (cleaned)
  cbsl_prices_monthly.csv              CBSL daily prices averaged to month
  macro_monthly.csv                    CPI (reconciled with DCS CCPI), exchange rate, interest rates, world prices
  cpi_reconciliation.csv               raw IMF CPI vs official DCS CCPI, month by month
  macro_annual.csv                     World Bank WDI indicators
  analysis_panel_monthly.csv           key national commodity prices joined to macro_monthly
"""
from pathlib import Path

import pandas as pd

from cpi_fix import load_dcs_ccpi, reconcile, reconciliation_table

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def month(s: pd.Series) -> pd.Series:
    """Normalise '2024-03-15', '2024-M03', '2024-03-01' -> Timestamp month start."""
    s = s.astype(str).str.replace(r"-M(\d\d)$", r"-\1", regex=True)
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


# ---------------------------------------------------------------- WFP prices
def build_wfp() -> pd.DataFrame:
    w = pd.read_csv(RAW / "wfp_food_prices_lka.csv", skiprows=[1])  # row 1 = HXL tags
    w = w[w["priceflag"].isin(["actual", "aggregate"]) | w["priceflag"].isna()]
    w = w.rename(columns={"admin1": "province", "admin2": "district",
                          "price": "price_lkr", "usdprice": "price_usd"})
    w["date"] = month(w["date"])
    w = w[w["market"] != "National Average"]
    cols = ["date", "province", "district", "market", "latitude", "longitude",
            "category", "commodity", "unit", "pricetype", "price_lkr", "price_usd"]
    w = (w[cols].groupby(cols[:10], dropna=False, as_index=False)[["price_lkr", "price_usd"]].mean())
    w.to_csv(OUT / "commodity_prices_market_monthly.csv", index=False)

    nat = (w.groupby(["date", "category", "commodity", "unit", "pricetype"], as_index=False)
             .agg(price_lkr_median=("price_lkr", "median"), price_lkr_mean=("price_lkr", "mean"),
                  price_lkr_min=("price_lkr", "min"), price_lkr_max=("price_lkr", "max"),
                  n_markets=("market", "nunique")))
    nat.to_csv(OUT / "commodity_prices_national_monthly.csv", index=False)
    print(f"WFP market-monthly: {len(w):,} rows, {w['market'].nunique()} markets, "
          f"{w['commodity'].nunique()} commodities, {w['date'].min():%Y-%m}..{w['date'].max():%Y-%m}")
    return nat


# --------------------------------------------------------------- CBSL daily
def build_cbsl() -> pd.DataFrame | None:
    f = RAW / "cbsl_daily_prices.csv"
    if not f.exists() or f.stat().st_size == 0:
        print("CBSL daily prices not found - run 03_scrape_cbsl_daily_prices.py")
        return None
    c = pd.read_csv(f, parse_dates=["date"])
    c["item"] = c["item"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Drop obvious parse errors: a price more than 10x or under 1/10 of the item-market median.
    med = c.groupby(["item", "pricetype", "market"])["price_lkr"].transform("median")
    bad = (c["price_lkr"] > 10 * med) | (c["price_lkr"] < med / 10) | (c["price_lkr"] <= 0)
    c = c[~bad].drop_duplicates(["date", "item", "pricetype", "market"])
    c.to_csv(OUT / "cbsl_daily_prices.csv", index=False)
    m = (c.assign(date=c["date"].dt.to_period("M").dt.to_timestamp())
           .groupby(["date", "item", "unit", "pricetype", "market"], as_index=False)
           .agg(price_lkr=("price_lkr", "mean"), n_days=("price_lkr", "size")))
    m.to_csv(OUT / "cbsl_prices_monthly.csv", index=False)
    print(f"CBSL daily: {len(c):,} rows ({bad.sum()} outliers dropped), {c['item'].nunique()} items, "
          f"{c['date'].min():%Y-%m-%d}..{c['date'].max():%Y-%m-%d}")
    return m


# --------------------------------------------------------------- macro monthly
def imf_series(fname: str, filt: dict, name: str) -> pd.Series:
    d = pd.read_csv(RAW / fname, low_memory=False)
    for k, v in filt.items():
        d = d[d[k] == v]
    d = d.dropna(subset=["OBS_VALUE"])
    return pd.Series(d["OBS_VALUE"].astype(float).values, index=month(d["TIME_PERIOD"]), name=name).groupby(level=0).mean()


def build_macro_monthly() -> pd.DataFrame:
    # Raw IMF CPI. It switches from NCPI to CCPI in 2024-09; cpi_fix.reconcile()
    # rebuilds the headline columns from official DCS CCPI (see PHASE1_NOTES.md).
    cpi = "imf_cpi_lka.csv"
    imf = pd.concat([
        imf_series(cpi, {"COICOP_1999": "_T", "TYPE_OF_TRANSFORMATION": "SRP_IX"}, "imf_cpi_index_raw"),
        imf_series(cpi, {"COICOP_1999": "_T", "TYPE_OF_TRANSFORMATION": "IX"}, "imf_cpi_index_2021_raw"),
        imf_series(cpi, {"COICOP_1999": "_T", "TYPE_OF_TRANSFORMATION": "SRP_YOY_PCH_PA_PT"}, "imf_inflation_yoy_raw"),
        imf_series(cpi, {"COICOP_1999": "_T", "TYPE_OF_TRANSFORMATION": "SRP_POP_PCH_PA_PT"}, "imf_inflation_mom_raw"),
        imf_series(cpi, {"COICOP_1999": "CP01", "TYPE_OF_TRANSFORMATION": "IX"}, "imf_cpi_food_index_raw"),
        imf_series(cpi, {"COICOP_1999": "CP01", "TYPE_OF_TRANSFORMATION": "YOY_PCH_PA_PT"}, "imf_food_inflation_yoy_raw"),
    ], axis=1)
    dcs = load_dcs_ccpi()
    rec = reconciliation_table(imf, dcs)
    if len(rec):
        rec.to_csv(OUT / "cpi_reconciliation.csv")
    cpi_df = reconcile(imf, dcs)
    lead = ["cpi_headline_index", "inflation_yoy_pct", "inflation_mom_pct", "inflation_source",
            "cpi_break_flag", "cpi_food_index", "food_inflation_yoy_pct"]
    s = [cpi_df[lead + [c for c in cpi_df.columns if c not in lead]]]

    fx = pd.read_csv(RAW / "fred_lkr_usd_monthly.csv")
    s.append(pd.Series(fx["lkr_per_usd"].values, index=month(fx["date"]), name="lkr_per_usd"))
    s.append(imf_series("imf_exchange_rates_lka.csv", {"INDICATOR": "XDC_EUR", "TYPE_OF_TRANSFORMATION": "PA_RT"}, "lkr_per_eur"))

    ir = "imf_interest_rates_lka.csv"
    for code, name in {"DISR_RT_PT_A_PT": "policy_discount_rate_pct",
                       "GSTBILY_RT_PT_A_PT": "tbill_rate_pct",
                       "MMRT_RT_PT_A_PT": "money_market_rate_pct",
                       "MFS162_RT_PT_A_PT": "deposit_rate_pct",
                       "MFS172_RT_PT_A_PT": "lending_rate_pct_to2014",
                       "MFS174_RT_PT_A_PT": "lending_rate_pct"}.items():
        s.append(imf_series(ir, {"INDICATOR": code}, name))

    pc = "imf_world_commodity_prices.csv"
    for code, name in {"PRICENPQ": "world_rice_usd_mt", "PWHEAMT": "world_wheat_usd_mt",
                       "PSUGAISA": "world_sugar_usd_lb", "POILBRE": "world_brent_usd_bbl",
                       "PCOIL": "world_coconut_oil_usd_mt", "PPOIL": "world_palm_oil_usd_mt",
                       "PSOIL": "world_soybean_oil_usd_mt", "PTEASL": "world_tea_colombo_usd_kg",
                       "PUREA": "world_urea_usd_mt"}.items():
        s.append(imf_series(pc, {"INDICATOR": code, "DATA_TRANSFORMATION": "USD"}, name))
    s.append(imf_series(pc, {"INDICATOR": "PFOOD", "DATA_TRANSFORMATION": "INDEX"}, "world_food_price_index"))

    m = pd.concat(s, axis=1).sort_index()
    m = m[m.index >= "2000-01-01"]
    m["lkr_usd_mom_pct"] = m["lkr_per_usd"].pct_change(fill_method=None) * 100
    m.index.name = "date"
    m.to_csv(OUT / "macro_monthly.csv")
    print(f"macro monthly: {m.shape}, {m.index.min():%Y-%m}..{m.index.max():%Y-%m}")
    return m


def build_macro_annual() -> None:
    a = pd.read_csv(RAW / "worldbank_wdi_lka_annual.csv")
    a.to_csv(OUT / "macro_annual.csv", index=False)
    print(f"macro annual: {a.shape}, {a['year'].min()}..{a['year'].max()}")


# --------------------------------------------------------------- panel
KEY = {  # WFP commodity -> panel column (retail, LKR/kg unless noted)
    "Rice (white)": "rice_white", "Rice (red nadu)": "rice_red_nadu", "Rice (medium grain)": "rice_medium",
    "Wheat flour": "wheat_flour", "Sugar": "sugar", "Coconut": "coconut_per_nut", "Lentils": "lentils",
    "Onions (red, local)": "red_onion_local", "Onions (imported)": "big_onion_imported",
    "Potatoes (local)": "potato_local", "Potatoes (imported)": "potato_imported",
    "Tomatoes": "tomato", "Carrots": "carrot", "Beans": "beans", "Cabbage": "cabbage",
    "Eggs": "egg_each", "Meat (chicken, broiler)": "chicken",
    "Chili (red, dry raw)": "dried_chilli", "Fish (dry, sprats)": "dried_sprats",
}


def build_panel(nat: pd.DataFrame, macro: pd.DataFrame) -> None:
    n = nat[(nat["pricetype"] == "Retail") & nat["commodity"].isin(KEY)]
    wide = n.pivot_table(index="date", columns="commodity", values="price_lkr_median").rename(columns=KEY)
    wide.columns = [f"price_{c}" for c in wide.columns]
    panel = macro.join(wide, how="left")
    panel = panel[panel.index >= "2004-01-01"]
    panel["crisis_2022"] = ((panel.index >= "2022-01-01") & (panel.index <= "2023-06-01")).astype(int)
    panel["covid_2020"] = ((panel.index >= "2020-03-01") & (panel.index <= "2021-12-01")).astype(int)
    panel["post_float_2022"] = (panel.index >= "2022-03-01").astype(int)
    panel.to_csv(OUT / "analysis_panel_monthly.csv")
    print(f"analysis panel: {panel.shape}")


def coverage_report() -> None:
    lines = ["file,rows,cols,start,end"]
    for f in sorted(OUT.glob("*.csv")):
        if f.name == "coverage_report.csv":
            continue
        d = pd.read_csv(f, low_memory=False)
        key = next((k for k in ("date", "year", "effective_date") if k in d), None)
        lo, hi = (d[key].min(), d[key].max()) if key else ("", "")
        lines.append(f"{f.name},{len(d)},{d.shape[1]},{lo},{hi}")
    (OUT / "coverage_report.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    nat = build_wfp()
    build_cbsl()
    macro = build_macro_monthly()
    build_macro_annual()
    build_panel(nat, macro)
    coverage_report()
    print((OUT / "coverage_report.csv").read_text())
