"""Phase 3 – economic relationship analysis helpers.

Builds one monthly dataset from the processed files and provides:
  * coverage reporting (sample length, gaps, truncation)
  * lead/lag cross-correlations with pairwise sample sizes
  * unit-root tests (ADF + KPSS)
  * VAR estimation with lag selection, diagnostics, Granger tests and IRFs
  * distributed-lag pass-through regressions with HAC standard errors
  * Engle-Granger long-run (levels) pass-through

Conventions
-----------
* Growth rates are 100 x log-differences (approximately % change), so
  regression coefficients are elasticities.
* Inflation uses the cleaned CPI from Phase 1 (`inflation_mom_pct`,
  `cpi_headline_index`), never the raw IMF series; `ccpi_basis` (1 when the
  month comes from CCPI rather than NCPI) is available as a control.
* Fuel months flagged `possible_missing_revision` are dropped (NaN).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
# Import submodules directly: `statsmodels.api` pulls in statsmodels.distributions,
# which breaks on statsmodels < 0.14.5 with SciPy >= 1.16 (removed `_lazywhere`).
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.var_model import VAR

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

COVID = ("2020-03-01", "2021-12-01")
CRISIS = ("2022-01-01", "2023-06-01")
FLOAT = "2022-03-01"

CBSL_ITEMS = {  # CBSL daily-report item -> column (retail mean across markets)
    "Sugar (White)": "cbsl_sugar",
    "Red Dhal": "cbsl_dhal",
    "Big Onion (Imp)": "cbsl_big_onion_imp",
    "Potato (Imp)": "cbsl_potato_imp",
    "Coconut (Avg.)": "cbsl_coconut",
    "Coconut oil": "cbsl_coconut_oil",
    "Nadu": "cbsl_rice_nadu",
    "Samba": "cbsl_rice_samba",
}
WFP_COLS = {  # analysis_panel column -> column
    "price_rice_white": "wfp_rice_white",
    "price_wheat_flour": "wfp_wheat_flour",
    "price_sugar": "wfp_sugar",
    "price_lentils": "wfp_lentils",
    "price_big_onion_imported": "wfp_big_onion_imp",
    "price_coconut_per_nut": "wfp_coconut",
}
MACRO_COLS = ["lkr_per_usd", "cpi_headline_index", "inflation_mom_pct", "inflation_yoy_pct",
              "inflation_source", "cpi_break_flag", "world_rice_usd_mt", "world_wheat_usd_mt",
              "world_sugar_usd_lb", "world_brent_usd_bbl", "world_coconut_oil_usd_mt",
              "world_palm_oil_usd_mt", "world_food_price_index"]


# --------------------------------------------------------------------- data
def load_monthly() -> pd.DataFrame:
    """One row per month (2000-01 onwards) with macro, world, WFP, CBSL and fuel series."""
    macro = pd.read_csv(PROC / "macro_monthly.csv", parse_dates=["date"], index_col="date")
    df = macro[MACRO_COLS].copy()

    panel = pd.read_csv(PROC / "analysis_panel_monthly.csv", parse_dates=["date"], index_col="date")
    df = df.join(panel[list(WFP_COLS)].rename(columns=WFP_COLS), how="outer")

    cb = pd.read_csv(PROC / "cbsl_prices_monthly.csv", parse_dates=["date"])
    cb["item"] = cb["item"].str.replace("(lmp)", "(Imp)", regex=False)
    cb = cb[(cb["pricetype"] == "Retail") & cb["item"].isin(CBSL_ITEMS)]
    df = df.join(cb.pivot_table(index="date", columns="item", values="price_lkr").rename(columns=CBSL_ITEMS),
                 how="outer")

    fuel = pd.read_csv(PROC / "fuel_prices_monthly.csv", parse_dates=["date"], keep_default_na=False)
    fuel["price_lkr"] = pd.to_numeric(fuel["price_lkr"], errors="coerce")
    fuel = fuel[fuel["flag"] == ""]  # drop possible_missing_revision months
    fp = fuel.pivot_table(index="date", columns="product", values="price_lkr")
    df = df.join(fp.rename(columns=lambda c: f"fuel_{c}"), how="outer")

    df = df.sort_index()
    df = df[df.index >= "2000-01-01"]
    # Derived series
    df["world_brent_lkr"] = df["world_brent_usd_bbl"] * df["lkr_per_usd"]
    df["world_rice_lkr"] = df["world_rice_usd_mt"] * df["lkr_per_usd"]
    df["world_wheat_lkr"] = df["world_wheat_usd_mt"] * df["lkr_per_usd"]
    df["world_sugar_lkr"] = df["world_sugar_usd_lb"] * df["lkr_per_usd"]
    df["world_coconut_oil_lkr"] = df["world_coconut_oil_usd_mt"] * df["lkr_per_usd"]
    # Regime / data-basis controls
    idx = df.index
    df["covid"] = ((idx >= COVID[0]) & (idx <= COVID[1])).astype(int)
    df["crisis"] = ((idx >= CRISIS[0]) & (idx <= CRISIS[1])).astype(int)
    df["post_float"] = (idx >= FLOAT).astype(int)
    df["ccpi_basis"] = df["inflation_source"].isin(["dcs_ccpi", "imf_ccpi"]).astype(int)
    df.index.name = "date"
    return df


def dlog(s: pd.Series) -> pd.Series:
    """100 x month-on-month log change; NaN unless both months are observed."""
    s = s.asfreq("MS") if isinstance(s.index, pd.DatetimeIndex) else s
    return 100 * np.log(s).diff()


def coverage(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            rows.append({"series": c, "n_months": 0})
            continue
        full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        ok = s.reindex(full).notna()
        run_id = (ok != ok.shift()).cumsum()
        runs = ok.groupby(run_id).agg(["first", "size"])
        best = runs[runs["first"]]["size"].max()
        start = ok[ok].groupby(run_id[ok]).apply(lambda g: g.index[0])
        size = ok[ok].groupby(run_id[ok]).size()
        b = size.idxmax()
        rows.append({"series": c, "first": f"{s.index.min():%Y-%m}", "last": f"{s.index.max():%Y-%m}",
                     "n_months": len(s), "gaps": int((~ok).sum()),
                     "longest_unbroken": f"{start[b]:%Y-%m}..{start[b] + pd.DateOffset(months=int(best) - 1):%Y-%m}",
                     "longest_len": int(best)})
    return pd.DataFrame(rows).set_index("series")


# --------------------------------------------------------------- lead / lag
def ccf(x: pd.Series, y: pd.Series, lags=range(0, 7)) -> pd.DataFrame:
    """corr(x_{t-k}, y_t) for each k: positive k means x LEADS y by k months.
    Uses all overlapping non-missing pairs and reports N for each lag."""
    rows = []
    for k in lags:
        pair = pd.concat([x.shift(k), y], axis=1).dropna()
        n = len(pair)
        r = pair.corr().iloc[0, 1] if n > 3 else np.nan
        rows.append({"lag": k, "corr": r, "n": n, "crit_95": 1.96 / np.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows).set_index("lag")


def ccf_grid(drivers: dict[str, pd.Series], targets: dict[str, pd.Series], lags=range(0, 7)):
    """Return (corr matrix, N matrix, best-lag table) for every driver x target pair."""
    corr, nobs, best = {}, {}, []
    for dn, x in drivers.items():
        for tn, y in targets.items():
            t = ccf(x, y, lags)
            corr[(dn, tn)] = t["corr"]
            nobs[(dn, tn)] = t["n"]
            if t["corr"].notna().any():
                k = t["corr"].abs().idxmax()
                best.append({"driver": dn, "target": tn, "best_lag": k, "corr": t.loc[k, "corr"],
                             "n": t.loc[k, "n"], "significant_95": abs(t.loc[k, "corr"]) > t.loc[k, "crit_95"]})
    return pd.DataFrame(corr).T, pd.DataFrame(nobs).T, pd.DataFrame(best)


# --------------------------------------------------------------- unit roots
def unit_root_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) for each series."""
    import warnings
    rows = []
    for name, s in series.items():
        s = s.dropna()
        if len(s) < 20:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_p = adfuller(s, autolag="AIC")[1]
            kpss_p = kpss(s, regression="c", nlags="auto")[1]
        verdict = ("stationary" if adf_p < 0.05 and kpss_p > 0.05 else
                   "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "mixed")
        rows.append({"series": name, "n": len(s), "adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


# --------------------------------------------------------------- VAR
@dataclass
class VarResult:
    data: pd.DataFrame
    results: object
    lag_order: pd.DataFrame
    chosen_lag: int
    diagnostics: pd.Series
    granger: pd.DataFrame


def fit_var(data: pd.DataFrame, exog: pd.DataFrame | None = None, maxlags: int = 6,
            ic: str = "bic") -> VarResult:
    """Fit a VAR on already-stationary columns; pick the lag by `ic`, report diagnostics
    and pairwise Granger-causality tests (Wald, H0: column i does not cause column j)."""
    data = data.dropna()
    if exog is not None:
        exog = exog.loc[data.index]
    model = VAR(data, exog=exog)
    sel = model.select_order(maxlags=maxlags)
    lag_tab = pd.DataFrame({k: v for k, v in sel.ics.items()}, index=range(len(sel.ics["aic"])))
    p = max(1, int(sel.selected_orders[ic]))
    res = model.fit(p)
    white = res.test_whiteness(nlags=max(p + 6, 12))
    norm = res.test_normality()
    diag = pd.Series({
        "n_obs": res.nobs, "lag_order": p, "lag_rule": ic.upper(),
        "stable (all roots outside unit circle)": bool(res.is_stable()),
        "Portmanteau whiteness p": white.pvalue,
        "Jarque-Bera normality p": norm.pvalue,
    })
    rows = []
    for cause in data.columns:
        for effect in data.columns:
            if cause == effect:
                continue
            t = res.test_causality(effect, [cause], kind="f")
            rows.append({"cause": cause, "effect": effect, "F": t.test_statistic, "p": t.pvalue,
                         "significant_5%": t.pvalue < 0.05})
    return VarResult(data, res, lag_tab, p, diag, pd.DataFrame(rows))


def bivariate_granger(x: pd.Series, y: pd.Series, maxlag: int = 3) -> pd.DataFrame:
    """Does x Granger-cause y? SSR F-test for each lag 1..maxlag (short samples)."""
    import warnings
    pair = pd.concat([y, x], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = grangercausalitytests(pair.values, maxlag=maxlag, verbose=False)
    return pd.DataFrame([{"lag": k, "F": v[0]["ssr_ftest"][0], "p": v[0]["ssr_ftest"][1], "n": len(pair)}
                         for k, v in out.items()]).set_index("lag")


# --------------------------------------------------------------- pass-through
@dataclass
class PassThrough:
    name: str
    model: object
    n: int
    sample: str
    cumulative: pd.DataFrame


def passthrough(y: pd.Series, drivers: dict[str, pd.Series], lags: int, controls: pd.DataFrame | None = None,
                month_fe: bool = False, hac_lags: int | None = None, name: str = "",
                ar_lag: bool = False) -> PassThrough:
    """Distributed-lag regression  y_t = a + sum_k b_k x_{t-k} + controls + e_t  (HAC SEs).

    The cumulative pass-through of each driver after `lags` months is sum_k b_k,
    reported with its HAC standard error and p-value."""
    X = {}
    for dn, x in drivers.items():
        for k in range(lags + 1):
            X[f"{dn}_L{k}"] = x.shift(k)
    X = pd.DataFrame(X)
    if ar_lag:
        X["y_L1"] = y.shift(1)
    if controls is not None:
        X = X.join(controls)
    if month_fe:
        X = X.join(pd.get_dummies(X.index.month, prefix="m", drop_first=True, dtype=float).set_index(X.index))
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    d = d.loc[:, (d != d.iloc[0]).any() | (d.columns == "y")]  # drop constant controls in this sample
    Xc = add_constant(d.drop(columns="y"))
    hac = hac_lags if hac_lags is not None else max(1, int(round(4 * (len(d) / 100) ** (2 / 9))))
    m = OLS(d["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    rows = []
    for dn in drivers:
        terms = [f"{dn}_L{k}" for k in range(lags + 1) if f"{dn}_L{k}" in Xc.columns]
        r = np.zeros(len(Xc.columns))
        for t in terms:
            r[list(Xc.columns).index(t)] = 1
        t = m.t_test(r)
        cum = float(np.squeeze(t.effect))
        if ar_lag and "y_L1" in Xc.columns:  # long-run multiplier with a lagged dependent variable
            cum_lr = cum / (1 - m.params["y_L1"])
        else:
            cum_lr = np.nan
        rows.append({"driver": dn, "impact (lag 0)": m.params.get(f"{dn}_L0", np.nan),
                     f"cumulative 0..{lags}": cum, "se": float(np.squeeze(t.sd)),
                     "p": float(np.squeeze(t.pvalue)), "long-run (with AR term)": cum_lr})
    return PassThrough(name, m, len(d), f"{d.index.min():%Y-%m}..{d.index.max():%Y-%m}",
                       pd.DataFrame(rows).set_index("driver"))


def long_run_levels(domestic: pd.Series, import_parity: pd.Series, controls: pd.DataFrame | None = None) -> pd.Series:
    """Engle-Granger: log domestic = a + b log(world price in LKR) (+ controls); b = long-run
    pass-through elasticity, plus the cointegration test p-value (H0: no cointegration)."""
    d = pd.concat([np.log(domestic).rename("y"), np.log(import_parity).rename("x")], axis=1).dropna()
    X = add_constant(d[["x"]] if controls is None else d[["x"]].join(controls))
    m = OLS(d["y"], X).fit()
    p_coint = coint(d["y"], d["x"])[1]
    return pd.Series({"n": len(d), "sample": f"{d.index.min():%Y-%m}..{d.index.max():%Y-%m}",
                      "long_run_elasticity": m.params["x"], "r2": m.rsquared,
                      "engle_granger_p": p_coint, "cointegrated_5%": p_coint < 0.05})
