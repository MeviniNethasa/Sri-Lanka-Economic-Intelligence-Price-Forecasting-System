"""Build a monthly retail fuel price history (LKR per litre / per 12.5 kg LPG cylinder).

Sources, in order of preference for a given month and product:
  1. dcs_ccpi    - DCS CCPI release Table 05: monthly AVERAGE Colombo retail price
                   (petrol & diesel 'CPC & IOC', kerosene, LPG). 2022 onwards.
  2. cpc_revision- data/raw/fuel/cpc_price_revisions.csv (hand-maintained, sourced):
                   step series from revision dates, day-weighted to a monthly average.
                   Fills months the DCS release doesn't cover yet (e.g. the latest month).
  3. wfp         - WFP/HDX diesel & petrol, 2013-2018 (Colombo City / national average).

Outputs:
  data/processed/fuel_prices_monthly.csv   date, product, price_lkr, source, flag
      flag = 'possible_missing_revision' where the CPC step series spans a revision
      that is not yet in the log (price is carried forward and may be stale)
  data/processed/fuel_revision_check.csv   DCS monthly average vs CPC revision step series
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

DCS_MAP = {  # DCS Table 05 item -> product
    "Petrol (CPC & IOC)": "petrol_92",
    "Diesel (CPC & IOC)": "auto_diesel",
    "Kerosene Oil": "kerosene",
    "Gas - Avg.": "lpg_12_5kg",
}
WFP_MAP = {"Fuel (diesel)": "auto_diesel", "Fuel (petrol-gasoline)": "petrol_92"}


def from_dcs() -> pd.DataFrame:
    f = RAW / "dcs_ccpi_retail_prices.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "product", "price_lkr", "source"])
    d = pd.read_csv(f, parse_dates=["date"])
    d = d[d["item"].isin(DCS_MAP)]
    cur = d.assign(product=d["item"].map(DCS_MAP))[["date", "product", "price_lkr"]]
    # 'same month last year' column back-fills the 12 months before the first release
    ly = d.assign(product=d["item"].map(DCS_MAP), date=d["date"] - pd.DateOffset(years=1),
                  price_lkr=d["price_same_month_last_year"])[["date", "product", "price_lkr"]]
    out = pd.concat([cur, ly]).dropna().drop_duplicates(["date", "product"], keep="first")
    return out.assign(source="dcs_ccpi")


def uncertain_intervals(all_rows: pd.DataFrame, product: str) -> list[tuple]:
    """Date ranges where the step series may be wrong because a revision is missing:
    (a) a dated 'todo_manual' row (price unknown) until the next sourced revision;
    (b) a sourced revision whose previous_price differs from the price carried
        forward from the revision before it (so something happened in between)."""
    g = all_rows[(all_rows["product"] == product) & all_rows["effective_date"].notna()]
    g = g.sort_values("effective_date")
    known = g[g["price_lkr_per_litre"].notna()]
    out = []
    for _, t in g[g["price_lkr_per_litre"].isna()].iterrows():
        nxt = known[known["effective_date"] > t["effective_date"]]["effective_date"]
        out.append((t["effective_date"], nxt.min() if len(nxt) else pd.Timestamp.max))
    for (_, a), (_, b) in zip(known.iloc[:-1].iterrows(), known.iloc[1:].iterrows()):
        if pd.notna(b["previous_price_lkr"]) and b["previous_price_lkr"] != a["price_lkr_per_litre"]:
            out.append((a["effective_date"], b["effective_date"]))
    return out


def from_revisions() -> pd.DataFrame:
    allr = pd.read_csv(RAW / "fuel" / "cpc_price_revisions.csv", parse_dates=["effective_date"])
    r = allr.dropna(subset=["effective_date", "price_lkr_per_litre"]).sort_values("effective_date")
    rows = []
    end = pd.Timestamp.today().normalize()
    for product, g in r.groupby("product"):
        steps = g.set_index("effective_date")["price_lkr_per_litre"].astype(float)
        # Seed the pre-revision price from the 1st of the first revision's month so
        # that month's average isn't computed from a partial month.
        first = g.iloc[0]
        month_start = first["effective_date"].replace(day=1)
        if pd.notna(first["previous_price_lkr"]) and month_start < first["effective_date"]:
            steps = pd.concat([pd.Series([float(first["previous_price_lkr"])], index=[month_start]), steps])
        daily = steps.reindex(pd.date_range(steps.index.min(), end, freq="D")).ffill()
        monthly = daily.resample("MS").mean()
        bad = uncertain_intervals(allr, product)
        for date, v in monthly.items():
            m_end = date + pd.offsets.MonthEnd(0)
            flag = "possible_missing_revision" if any(a <= m_end and date < b for a, b in bad) else ""
            rows.append({"date": date, "product": product, "price_lkr": round(v, 2),
                         "source": "cpc_revision", "flag": flag})
    return pd.DataFrame(rows)


def from_wfp() -> pd.DataFrame:
    w = pd.read_csv(PROC / "commodity_prices_market_monthly.csv", parse_dates=["date"])
    w = w[w["commodity"].isin(WFP_MAP)]
    w = w.groupby(["date", "commodity"], as_index=False)["price_lkr"].median()
    return w.assign(product=w["commodity"].map(WFP_MAP), source="wfp")[["date", "product", "price_lkr", "source"]]


def main() -> None:
    dcs, rev, wfp = from_dcs(), from_revisions(), from_wfp()
    allp = pd.concat([dcs, rev, wfp])
    rank = {"dcs_ccpi": 0, "cpc_revision": 1, "wfp": 2}
    allp["_r"] = allp["source"].map(rank)
    out = (allp.sort_values(["date", "product", "_r"]).drop_duplicates(["date", "product"])
               .drop(columns="_r").sort_values(["product", "date"]))
    out["flag"] = out["flag"].fillna("")
    out.to_csv(PROC / "fuel_prices_monthly.csv", index=False)

    chk = dcs.merge(rev, on=["date", "product"], suffixes=("_dcs", "_cpc_revision"))
    if len(chk):
        chk["diff_pct"] = ((chk["price_lkr_dcs"] / chk["price_lkr_cpc_revision"] - 1) * 100).round(1)
    chk.drop(columns=[c for c in chk.columns if c.startswith("source")]).to_csv(
        PROC / "fuel_revision_check.csv", index=False)

    print(out.groupby(["product", "source"])["date"].agg(["min", "max", "count"]).to_string())
    if len(chk):
        print("\nDCS monthly average vs CPC revision step series (overlap):")
        print(chk[["date", "product", "price_lkr_dcs", "price_lkr_cpc_revision", "diff_pct"]].to_string(index=False))
    todo = pd.read_csv(RAW / "fuel" / "cpc_price_revisions.csv")
    todo = todo[todo["status"] == "todo_manual"]
    print(f"\n{len(todo)} revision rows still marked todo_manual in data/raw/fuel/cpc_price_revisions.csv")


if __name__ == "__main__":
    main()
