"""Compare WFP and CBSL retail prices over their overlap and build continuous series.

For each WFP commodity with a CBSL counterpart (series_utils.ITEM_MAP):
  * WFP national median retail price vs the CBSL retail mean (Pettah/Dambulla/
    Narahenpita/Negombo, whichever report the item), month by month
  * WFP Colombo City vs CBSL Narahenpita retail (both are Colombo prices)
Metrics: overlapping months, median WFP/CBSL ratio, MAPE after level adjustment,
correlation of levels and of month-on-month % changes.

Continuous series: WFP national median up to its last month, then the CBSL
retail mean multiplied by the median WFP/CBSL ratio of the last 12 overlapping
months ('cbsl_scaled'). Only items with good agreement are extended: either
month-on-month changes track (corr >= 0.5 and MAPE <= 15%) or, for sticky
prices, levels agree within 5% MAPE.

Outputs (data/processed/):
  wfp_cbsl_overlap_monthly.csv       side-by-side monthly values
  wfp_cbsl_agreement.csv             one row per item with the metrics above
  commodity_prices_continuous_monthly.csv
"""
import numpy as np
import pandas as pd

from series_utils import ITEM_MAP, PROC, cbsl_retail_mean, load_cbsl_monthly, load_wfp_national

MIN_OVERLAP = 6          # months needed to judge agreement
MIN_CHANGE_CORR = 0.5    # MoM %-change correlation needed to extend a series
MAX_MAPE = 15.0          # level-adjusted MAPE (%) allowed to extend a series
MAX_MAPE_STICKY = 5.0    # ...or extend when levels agree this closely (sticky prices)


def main() -> None:
    wfp = load_wfp_national()
    cbsl = load_cbsl_monthly()
    wfp_mkt = pd.read_csv(PROC / "commodity_prices_market_monthly.csv", parse_dates=["date"])
    wfp_col = wfp_mkt[(wfp_mkt["market"] == "Colombo City") & (wfp_mkt["pricetype"] == "Retail")]
    nar = cbsl[(cbsl["market"] == "Narahenpita") & (cbsl["pricetype"] == "Retail")]

    overlap_rows, metrics, continuous = [], [], []
    for wname, (cname, conf) in ITEM_MAP.items():
        w = wfp[wfp["commodity"] == wname].set_index("date")["price_lkr_median"]
        c = cbsl_retail_mean(cbsl, cname)
        both = pd.concat({"wfp_national": w, "cbsl_retail_mean": c}, axis=1).dropna()
        wc = wfp_col[wfp_col["commodity"] == wname].set_index("date")["price_lkr"]
        cn = nar[nar["item"] == cname].set_index("date")["price_lkr"]
        col = pd.concat({"wfp_colombo": wc, "cbsl_narahenpita": cn}, axis=1).dropna()

        side = pd.concat({"wfp_national": w, "cbsl_retail_mean": c, "wfp_colombo": wc,
                          "cbsl_narahenpita": cn}, axis=1)
        side = side[side.index >= "2023-01-01"].reset_index()
        side.insert(1, "commodity", wname)
        side.insert(2, "cbsl_item", cname)
        overlap_rows.append(side)

        m = {"commodity": wname, "cbsl_item": cname, "match_confidence": conf,
             "wfp_last_month": w.index.max().strftime("%Y-%m") if len(w) else None,
             "overlap_months": len(both)}
        if len(both) >= MIN_OVERLAP:
            ratio = both["wfp_national"] / both["cbsl_retail_mean"]
            adj = both["cbsl_retail_mean"] * ratio.median()
            m.update({
                "median_ratio_wfp_to_cbsl": round(ratio.median(), 3),
                "mape_level_adjusted_pct": round((abs(adj - both["wfp_national"]) / both["wfp_national"]).mean() * 100, 1),
                "corr_levels": round(both.corr().iloc[0, 1], 3),
                "corr_mom_changes": round(both.pct_change().dropna().corr().iloc[0, 1], 3),
            })
        if len(col) >= MIN_OVERLAP:
            m["colombo_overlap_months"] = len(col)
            m["colombo_median_ratio"] = round((col["wfp_colombo"] / col["cbsl_narahenpita"]).median(), 3)
            m["colombo_corr_levels"] = round(col.corr().iloc[0, 1], 3)
        mape = m.get("mape_level_adjusted_pct", 999)
        tracks = m.get("corr_mom_changes", -1) >= MIN_CHANGE_CORR and mape <= MAX_MAPE
        # Sticky / administered prices (rice, dhal) barely move, so their change
        # correlation is noise; accept them when levels agree closely instead.
        close_levels = mape <= MAX_MAPE_STICKY
        ok = tracks or close_levels
        m["extend_with_cbsl"] = bool(ok)
        m["extend_reason"] = ("changes_track" if tracks else
                              "levels_within_5pct" if close_levels else "not_extended")
        metrics.append(m)

        # continuous series
        base = w.to_frame("price_lkr").assign(source="wfp")
        if ok:
            k = (both["wfp_national"] / both["cbsl_retail_mean"]).tail(12).median()
            ext = (c[c.index > w.index.max()] * k).to_frame("price_lkr").assign(source="cbsl_scaled")
            base = pd.concat([base, ext])
        base = base.reset_index().rename(columns={"index": "date"})
        base.insert(1, "commodity", wname)
        continuous.append(base)

    pd.concat(overlap_rows).to_csv(PROC / "wfp_cbsl_overlap_monthly.csv", index=False)
    met = pd.DataFrame(metrics)
    met.to_csv(PROC / "wfp_cbsl_agreement.csv", index=False)
    cont = pd.concat(continuous).sort_values(["commodity", "date"])
    cont["price_lkr"] = cont["price_lkr"].round(2)
    cont.to_csv(PROC / "commodity_prices_continuous_monthly.csv", index=False)

    pd.set_option("display.width", 200)
    print(met.drop(columns=["wfp_last_month"]).to_string(index=False))
    ext = cont[cont["source"] == "cbsl_scaled"]
    print(f"\ncontinuous series: {cont['commodity'].nunique()} commodities, "
          f"{len(ext)} CBSL-extended rows up to {ext['date'].max():%Y-%m}" if len(ext) else "no extension")


if __name__ == "__main__":
    main()
