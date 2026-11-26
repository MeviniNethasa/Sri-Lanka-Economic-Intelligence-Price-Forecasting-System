"""Reconcile the IMF CPI series with official DCS CCPI (base 2021=100).

Problem
-------
The IMF 'CPI' series for Sri Lanka (IMF.STA:CPI, COICOP _T) is not one index:
  * up to 2024-08 its values/YoY match the NCPI (2021=100)
    (e.g. YoY 2.5% Jul-2024, 1.1% Aug-2024 = published NCPI inflation),
  * from 2024-09 it is the CCPI (2021=100)
    (190.9 Sep-2024, 189.9 Oct-2024, 193.8 Oct-2025 = DCS CCPI releases).
NCPI sits ~6.5% above CCPI in level, so the switch shows up as a fake -6.5%
month-on-month drop in Sep-2024 and pushes every YoY rate from 2024-09 to
2025-08 about 6 points too low (e.g. -6.2% instead of the official -0.5% for
Sep-2024, -10.4% instead of -4.0% for Jan-2025).

Fix
---
Headline columns used downstream are rebuilt as:
  cpi_headline_index   official CCPI level where DCS data exist; months before/after
                       are chain-linked with IMF month-on-month growth, so the
                       level has no break
  inflation_yoy_pct    official DCS CCPI YoY where available, else IMF YoY
  inflation_mom_pct    official DCS CCPI MoM where available, else IMF MoM
  inflation_source     'dcs_ccpi', 'imf_ccpi' (IMF from 2024-09, identical to CCPI)
                       or 'imf_ncpi' (IMF before 2024-09, NCPI-based)
  cpi_break_flag       1 where the raw IMF YoY/MoM is distorted by the index switch
The raw IMF numbers are kept as imf_cpi_index_raw / imf_inflation_*_raw.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SWITCH_MONTH = pd.Timestamp("2024-09-01")
BREAK_WINDOW = (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-08-01"))


def load_dcs_ccpi() -> pd.DataFrame | None:
    f = RAW / "dcs_ccpi_monthly.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    return d


def reconcile(imf: pd.DataFrame, dcs: pd.DataFrame | None) -> pd.DataFrame:
    """imf: index=month, columns imf_cpi_index_raw, imf_inflation_yoy_raw,
    imf_inflation_mom_raw, imf_cpi_food_index_raw, imf_food_inflation_yoy_raw."""
    out = imf.copy()
    in_break = (out.index >= BREAK_WINDOW[0]) & (out.index <= BREAK_WINDOW[1])
    out["cpi_break_flag"] = 0
    out.loc[in_break, "cpi_break_flag"] = 1

    if dcs is None or dcs.empty:
        # No official data: keep IMF values but blank the distorted window.
        out["cpi_headline_index"] = out["imf_cpi_index_raw"]
        out["inflation_yoy_pct"] = out["imf_inflation_yoy_raw"].mask(in_break)
        out["inflation_mom_pct"] = out["imf_inflation_mom_raw"].mask(out.index == SWITCH_MONTH)
        out["inflation_source"] = "imf"
        return out

    cols = {"ccpi_index": "ccpi_index", "ccpi_yoy_pct": "ccpi_yoy_pct", "ccpi_mom_pct": "ccpi_mom_pct",
            "ccpi_food_index": "ccpi_food_index", "ccpi_food_yoy_pct": "ccpi_food_yoy_pct",
            "ccpi_nonfood_index": "ccpi_nonfood_index", "ccpi_nonfood_yoy_pct": "ccpi_nonfood_yoy_pct"}
    out = out.join(dcs[list(cols)], how="outer")
    out["cpi_break_flag"] = out["cpi_break_flag"].fillna(0).astype(int)
    has = out["ccpi_index"].notna()

    # Chain-link the level on official CCPI: back-cast before the first DCS month and
    # forward-cast after the last one with IMF month-on-month growth.
    first, last = out.index[has][0], out.index[has][-1]
    level = out["ccpi_index"].copy()
    growth = out["imf_cpi_index_raw"] / out["imf_cpi_index_raw"].shift(1)
    idx = list(out.index)
    for t in reversed([t for t in idx if t < first]):
        nxt = idx[idx.index(t) + 1]
        g = growth.loc[nxt]
        level.loc[t] = level.loc[nxt] / g if pd.notna(g) and pd.notna(level.loc[nxt]) else None
    for t in [t for t in idx if t > last]:
        prv = idx[idx.index(t) - 1]
        g = growth.loc[t]
        level.loc[t] = level.loc[prv] * g if pd.notna(g) and pd.notna(level.loc[prv]) else None
    out["cpi_headline_index"] = level.round(2)

    out["inflation_yoy_pct"] = out["ccpi_yoy_pct"].where(has, out["imf_inflation_yoy_raw"])
    out["inflation_mom_pct"] = out["ccpi_mom_pct"].where(has, out["imf_inflation_mom_raw"])
    # Which index each month's inflation rate comes from:
    #   dcs_ccpi  - official DCS CCPI release
    #   imf_ccpi  - IMF series from 2024-09 on (verified identical to CCPI)
    #   imf_ncpi  - IMF series before 2024-09 (NCPI-based; not comparable 1:1 with CCPI)
    src = pd.Series("imf_ncpi", index=out.index)
    src[out.index >= SWITCH_MONTH] = "imf_ccpi"
    src[has] = "dcs_ccpi"
    src[out["inflation_yoy_pct"].isna()] = None
    out["inflation_source"] = src
    # Any IMF month still inside the break window without DCS cover stays blank.
    leftover = (out["cpi_break_flag"] == 1) & ~has
    out.loc[leftover, ["inflation_yoy_pct", "inflation_mom_pct"]] = None

    out["cpi_food_index"] = out["ccpi_food_index"].where(has, out.get("imf_cpi_food_index_raw"))
    out["food_inflation_yoy_pct"] = out["ccpi_food_yoy_pct"].where(has, out.get("imf_food_inflation_yoy_raw"))
    return out


def reconciliation_table(imf: pd.DataFrame, dcs: pd.DataFrame | None) -> pd.DataFrame:
    """Month-by-month comparison of raw IMF vs official DCS CCPI (overlap only)."""
    if dcs is None:
        return pd.DataFrame()
    t = imf[["imf_cpi_index_2021_raw", "imf_inflation_yoy_raw", "imf_inflation_mom_raw"]].join(
        dcs[["ccpi_index", "ccpi_yoy_pct", "ccpi_mom_pct"]], how="inner")
    t["index_ratio_imf_to_ccpi"] = (t["imf_cpi_index_2021_raw"] / t["ccpi_index"]).round(4)
    t["yoy_gap_pp"] = (t["imf_inflation_yoy_raw"] - t["ccpi_yoy_pct"]).round(2)
    t["imf_segment"] = ["ccpi" if abs(r - 1) < 0.005 else "not_ccpi (ncpi)" for r in t["index_ratio_imf_to_ccpi"]]
    t.index.name = "date"
    return t
