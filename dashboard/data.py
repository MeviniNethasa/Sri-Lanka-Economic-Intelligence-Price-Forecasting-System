"""Data loaders for the LankaEcon dashboard.

Everything here reads files that the pipeline has already written
(data/processed/*.csv and reports/tables/phase3-5_*.csv). Nothing is re-estimated
or re-trained in the app. Loaders are wrapped in st.cache_data by app.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "reports" / "tables"

CPI_SOURCE_LABEL = {
    "imf_ncpi": "IMF (NCPI basis)",
    "dcs_ccpi": "DCS official CCPI",
    "imf_ccpi": "IMF (CCPI basis)",
}

# CBSL item used for wholesale→retail margins, keyed by the continuous-series name
CBSL_ITEM = {
    "Beans": "Beans", "Cabbage": "Cabbage", "Carrots": "Carrot", "Tomatoes": "Tomato",
    "Eggplants": "Brinjal", "Pumpkin": "Pumpkin", "Snake gourd": "Snake gourd",
    "Coconut": "Coconut (Avg.)", "Potatoes (local)": "Potato (Local)", "Potatoes (imported)": "Potato (Imp)",
    "Onions (imported)": "Big Onion (Imp)", "Onions (red, local)": "Red Onion (Local)",
    "Lentils": "Red Dhal", "Rice (white)": "Nadu", "Rice (medium grain)": "Samba",
    "Papaya": "Papaw", "Pineapples": "Pineapple", "Bananas": "Banana (Sour)",
    "Chili (red, dry raw)": "Dried Chilli (Imp)", "Eggs": "Egg (White)",
}


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


# ------------------------------------------------------------------ macro
def macro() -> pd.DataFrame:
    m = _read(PROC / "macro_monthly.csv", parse_dates=["date"])
    if m.empty:
        return m
    m["cpi_source_label"] = m["inflation_source"].map(CPI_SOURCE_LABEL).fillna("n/a")
    m["cpi_break_flag"] = m["cpi_break_flag"].fillna(0).astype(int)
    return m


# ------------------------------------------------------------------ commodities
def continuous() -> pd.DataFrame:
    return _read(PROC / "commodity_prices_continuous_monthly.csv", parse_dates=["date"])


def agreement() -> pd.DataFrame:
    return _read(PROC / "wfp_cbsl_agreement.csv")


def cbsl_monthly() -> pd.DataFrame:
    c = _read(PROC / "cbsl_prices_monthly.csv", parse_dates=["date"])
    if not c.empty:
        c["item"] = c["item"].str.replace("(lmp)", "(Imp)", regex=False)
    return c


def markets() -> pd.DataFrame:
    return _read(PROC / "commodity_prices_market_monthly.csv", parse_dates=["date"])


def fuel() -> pd.DataFrame:
    f = _read(PROC / "fuel_prices_monthly.csv", parse_dates=["date"], keep_default_na=False)
    if not f.empty:
        f["price_lkr"] = pd.to_numeric(f["price_lkr"], errors="coerce")
    return f


def seasonal_profile(s: pd.Series) -> pd.Series:
    """Average % deviation from a centred 13-month mean, by calendar month (needs ≥ 24 months)."""
    s = s.dropna()
    if len(s) < 24:
        return pd.Series(dtype=float)
    s = s.asfreq("MS")
    lv = np.log(s)
    dev = 100 * (lv - lv.rolling(13, center=True, min_periods=9).mean())
    prof = dev.groupby(dev.index.month).mean()
    return prof - prof.mean()


def margins(cb: pd.DataFrame, item: str) -> pd.DataFrame:
    """Monthly wholesale (Pettah / Dambulla / Marandagahamula) vs retail (all markets) for one CBSL item."""
    g = cb[cb["item"] == item]
    if g.empty:
        return pd.DataFrame()
    piv = g.pivot_table(index="date", columns=["pricetype", "market"], values="price_lkr")
    out = pd.DataFrame(index=piv.index)
    if "Wholesale" in piv:
        for mk in piv["Wholesale"].columns:
            out[f"Wholesale – {mk}"] = piv[("Wholesale", mk)]
    if "Retail" in piv:
        for mk in piv["Retail"].columns:
            out[f"Retail – {mk}"] = piv[("Retail", mk)]
        out["Retail (mean)"] = piv["Retail"].mean(axis=1)
    if "Wholesale" in piv and "Retail" in piv:
        out["markup_pct"] = 100 * (out["Retail (mean)"] / piv["Wholesale"].mean(axis=1) - 1)
    return out


# ------------------------------------------------------------------ forecasts (Phase 4)
def forecast_history() -> dict[str, pd.Series]:
    """History of each Phase 4 series, built the same way as src/forecasting.build_specs."""
    m, c, cb = macro(), continuous(), cbsl_monthly()
    out: dict[str, pd.Series] = {}
    if not m.empty:
        mi = m.set_index("date")
        out["CPI (cleaned)"] = mi["cpi_headline_index"].dropna()
        out["LKR/USD"] = mi["lkr_per_usd"].dropna()
    if not c.empty:
        for name, g in c.groupby("commodity"):
            out[name] = g.set_index("date")["price_lkr"].sort_index()
    if not cb.empty:
        s = cb[(cb["item"] == "Sugar (White)") & (cb["pricetype"] == "Retail")]
        out["Sugar (CBSL)"] = s.groupby("date")["price_lkr"].mean().sort_index()
    return out


def phase4() -> dict[str, pd.DataFrame]:
    return {
        "selected": _read(TAB / "phase4_selected_forecasts.csv", parse_dates=["target_date"]),
        "all": _read(TAB / "phase4_forecasts.csv", parse_dates=["origin", "target_date"]),
        "scores": _read(TAB / "phase4_backtest_scores.csv"),
        "best": _read(TAB / "phase4_best_models.csv"),
        "specs": _read(TAB / "phase4_model_specs.csv"),
    }


# ------------------------------------------------------------------ anomalies (Phase 5)
def phase5() -> dict[str, pd.DataFrame]:
    return {
        "scores": _read(TAB / "phase5_anomaly_scores.csv", parse_dates=["date"]),
        "flagged": _read(TAB / "phase5_anomalies.csv", parse_dates=["date"]),
        "episodes": _read(TAB / "phase5_anomaly_episodes.csv", parse_dates=["start", "end", "peak_month"]),
        "segments": _read(TAB / "phase5_change_points.csv", parse_dates=["start", "end"]),
        "events": _read(TAB / "phase5_event_catalogue.csv"),
        "validation": _read(TAB / "phase5_event_validation.csv"),
    }


# ------------------------------------------------------------------ relationships (Phase 3)
def phase3() -> dict[str, pd.DataFrame]:
    return {
        "passthrough": _read(TAB / "phase3_passthrough_summary.csv"),
        "irf": _read(TAB / "phase3_irf_summary.csv"),
        "granger_var": _read(TAB / "phase3_granger_var.csv"),
        "granger_recent": _read(TAB / "phase3_granger_recent.csv"),
    }


def freshness() -> pd.DataFrame:
    """Last observation date of each input, for the sidebar."""
    rows = []
    m = macro()
    if not m.empty:
        rows.append(("CPI", m.dropna(subset=["cpi_headline_index"])["date"].max()))
        rows.append(("LKR/USD", m.dropna(subset=["lkr_per_usd"])["date"].max()))
    for label, df in [("CBSL daily prices", _read(PROC / "cbsl_daily_prices.csv", usecols=["date"], parse_dates=["date"])),
                      ("WFP market prices", _read(PROC / "commodity_prices_market_monthly.csv", usecols=["date"], parse_dates=["date"])),
                      ("Fuel prices", fuel())]:
        if not df.empty:
            rows.append((label, df["date"].max()))
    return pd.DataFrame(rows, columns=["dataset", "last"])
