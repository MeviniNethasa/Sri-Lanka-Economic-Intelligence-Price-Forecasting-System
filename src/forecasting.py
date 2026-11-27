"""Phase 4 – price forecasting with rolling-origin backtests.

Models
------
naive           last observed value
seasonal_naive  value 12 months before the target month (falls back to naive)
sarima          SARIMA on log price, order chosen by AIC on the initial window only
sarimax         SARIMA + lagged external drivers validated in Phase 3 (log levels, lag 1)
lightgbm        direct multi-horizon gradient boosting on lagged growth, calendar,
                lagged driver and regime features (pooled across items for short series)

Leakage guardrails
------------------
* A forecast made at origin t uses only observations dated <= t.
* Drivers enter with a lag of >= 1 month. For future months whose driver value
  is not yet known at the information date, the last observed value is carried
  forward (random-walk expectation). In backtests the information date is the
  origin itself; for the final forecast it is the last month each driver is
  actually observed (e.g. LKR/USD for Sep–Oct 2026 is known when CPI ends in Aug).
* SARIMA orders are chosen once, on the initial training window, never on test data.
* Fuel months flagged `possible_missing_revision` are not used anywhere; fuel is
  not a model input (Phase 3 found Brent does not pass directly to CPI).
* Gaps in a target series stay missing (no imputation): the state-space SARIMA
  handles them, LightGBM drops rows whose target is missing, and no forecast is
  scored against a missing actual.

Evaluation
----------
Forecast errors are split by *target month*: targets inside the 2022 crisis
window (2022-01 → 2023-06) form the crisis stress test and are excluded from the
routine accuracy scores.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
# Direct submodule imports (statsmodels.api breaks on statsmodels < 0.14.5 + SciPy >= 1.16)
from statsmodels.tsa.statespace.sarimax import SARIMAX

CRISIS = (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-01"))
H = 6  # forecast horizon in months
MODELS = ["naive", "seasonal_naive", "sarima", "sarimax", "lightgbm"]


# ------------------------------------------------------------------ specs
@dataclass
class SeriesSpec:
    name: str
    y: pd.Series                       # monthly price level (NaN = missing)
    group: str                         # "long" or "recent"
    drivers: dict = field(default_factory=dict)   # name -> monthly level series (validated in Phase 3)
    lgbm_exog: dict = field(default_factory=dict)  # name -> monthly level series for LightGBM features
    regime_exog: bool = False          # add crisis dummy to SARIMAX
    forecastable: bool = True          # False if the series has ended (backtest only)
    seasonal: bool = True
    note: str = ""


def build_specs(df: pd.DataFrame, cont: pd.DataFrame) -> list[SeriesSpec]:
    """Select forecast targets. `df` = relationships.load_monthly(), `cont` = continuous prices."""
    def cs(name, start=None):
        s = cont[cont["commodity"] == name].set_index("date")["price_lkr"].asfreq("MS")
        return s.loc[start:] if start else s

    fx, wfood = df["lkr_per_usd"], df["world_food_price_index"]
    specs = [
        SeriesSpec("CPI (cleaned)", df["cpi_headline_index"].dropna().asfreq("MS"), "long",
                   drivers={"lkr_per_usd": fx, "world_food_index": wfood},
                   lgbm_exog={"lkr_per_usd": fx, "world_food_index": wfood}, regime_exog=True,
                   note="drivers: FX (0.27→0.58 pass-through), world food (0.29)"),
        SeriesSpec("LKR/USD", fx.dropna().asfreq("MS"), "long", seasonal=False,
                   note="macro driver; no external drivers"),
        SeriesSpec("Rice (white)", cs("Rice (white)"), "long",
                   lgbm_exog={"lkr_per_usd": fx, "world_rice_usd": df["world_rice_usd_mt"]},
                   note="gap 2022-07..2023-05 kept missing; Phase 3: no FX/world pass-through, so no SARIMAX"),
        SeriesSpec("Wheat flour", df["wfp_wheat_flour"].loc["2006-01":"2022-06"].asfreq("MS"), "long",
                   drivers={"lkr_per_usd": fx, "world_wheat_usd": df["world_wheat_usd_mt"]},
                   lgbm_exog={"lkr_per_usd": fx, "world_wheat_usd": df["world_wheat_usd_mt"]},
                   regime_exog=True, forecastable=False,
                   note="ends 2022-06: backtest/stress test only"),
    ]
    recent = {
        "Coconut": ({}, "domestic crop; no validated driver"),
        "Beans": ({}, "seasonal vegetable"),
        "Carrots": ({}, "seasonal vegetable"),
        "Tomatoes": ({}, "seasonal vegetable"),
        "Cabbage": ({}, "seasonal vegetable"),
        "Onions (imported)": ({"lkr_per_usd": fx}, "driver: FX (≈3.0, p=0.08)"),
        "Lentils": ({"lkr_per_usd": fx}, "driver: FX (dhal ≈1.15, p=0.04)"),
        "Potatoes (local)": ({}, "domestic crop"),
    }
    for name, (drv, note) in recent.items():
        specs.append(SeriesSpec(name, cs(name, "2022-07-01"), "recent", drivers=drv,
                                lgbm_exog={"lkr_per_usd": fx}, note=note))
    specs.append(SeriesSpec("Sugar (CBSL)", df["cbsl_sugar"].dropna().asfreq("MS"), "recent",
                            drivers={"lkr_per_usd": fx, "world_sugar_usd": df["world_sugar_usd_lb"]},
                            lgbm_exog={"lkr_per_usd": fx, "world_sugar_usd": df["world_sugar_usd_lb"]},
                            note="CBSL retail mean from 2023-01; drivers: world sugar (0.37), FX (0.57)"))
    return specs


# ------------------------------------------------------------------ helpers
def last_obs(s: pd.Series, t: pd.Timestamp):
    s = s.loc[:t].dropna()
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, np.nan)


def future_index(t: pd.Timestamp, h: int = H) -> pd.DatetimeIndex:
    return pd.date_range(t + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def exog_matrix(spec: SeriesSpec, index: pd.DatetimeIndex, info_date: pd.Timestamp) -> pd.DataFrame | None:
    """log(driver) lagged one month; values after `info_date` are replaced by the last value
    observed on or before `info_date` (random-walk expectation)."""
    if not spec.drivers and not spec.regime_exog:
        return None
    cols = {}
    for name, s in spec.drivers.items():
        known = np.log(s.loc[:info_date].dropna())
        lagged = known.reindex(known.index.union(index)).ffill().shift(1)  # lag 1
        # months whose lag-1 value lies beyond the info date: hold last known value
        lagged = lagged.reindex(index)
        lagged = lagged.fillna(known.iloc[-1]) if len(known) else lagged
        cols[f"log_{name}_L1"] = lagged.values
    if spec.regime_exog:
        crisis_at_info = int(CRISIS[0] <= info_date <= CRISIS[1])
        cols["crisis"] = [int(CRISIS[0] <= d <= CRISIS[1]) if d <= info_date else crisis_at_info for d in index]
    return pd.DataFrame(cols, index=index)


# ------------------------------------------------------------------ baselines
def f_naive(y: pd.Series, t: pd.Timestamp, h: int = H) -> np.ndarray:
    return np.repeat(last_obs(y, t)[1], h)


def f_seasonal_naive(y: pd.Series, t: pd.Timestamp, h: int = H) -> np.ndarray:
    base = last_obs(y, t)[1]
    out = []
    for d in future_index(t, h):
        v = y.get(d - pd.DateOffset(years=1), np.nan)
        out.append(v if pd.notna(v) and d - pd.DateOffset(years=1) <= t else base)
    return np.array(out)


# ------------------------------------------------------------------ SARIMA / SARIMAX
def select_order(y: pd.Series, exog: pd.DataFrame | None, seasonal: bool) -> tuple:
    """AIC grid over (p,1,q)(P,0,Q,12), p,q,P,Q in {0,1}; called on the initial window only."""
    best = (np.inf, (0, 1, 1), (0, 0, 0, 0))
    grid_s = [(P, 0, Q, 12) for P in (0, 1) for Q in (0, 1)] if seasonal else [(0, 0, 0, 0)]
    for p, q in itertools.product((0, 1), (0, 1)):
        for so in grid_s:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = SARIMAX(np.log(y), exog=exog, order=(p, 1, q), seasonal_order=so, trend="t" if exog is None else "n",
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
                if r.aic < best[0]:
                    best = (r.aic, (p, 1, q), so)
            except Exception:
                continue
    return best[1], best[2]


def f_sarima(spec: SeriesSpec, t: pd.Timestamp, order, sorder, use_exog: bool, h: int = H,
             info_date: pd.Timestamp | None = None, return_ci: bool = False):
    y = spec.y.loc[:t]
    y = y.loc[y.first_valid_index():]
    info = info_date or t
    exog_tr = exog_matrix(spec, y.index, t) if use_exog else None
    if exog_tr is not None:
        keep = exog_tr.columns[exog_tr.nunique() > 1]          # drop constant columns (e.g. crisis=0 pre-2022)
        exog_tr = exog_tr[keep] if len(keep) else None
    fidx = future_index(t, h)
    exog_f = exog_matrix(spec, fidx, info)[exog_tr.columns] if exog_tr is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = SARIMAX(np.log(y), exog=exog_tr, order=order, seasonal_order=sorder,
                    trend="t" if exog_tr is None else "n",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        fc = r.get_forecast(h, exog=exog_f)
    mean = np.exp(fc.predicted_mean.values)
    if not return_ci:
        return mean
    ci80, ci95 = np.exp(fc.conf_int(alpha=0.2).values), np.exp(fc.conf_int(alpha=0.05).values)
    return mean, ci80, ci95


# ------------------------------------------------------------------ LightGBM (direct)
def make_features(spec: SeriesSpec, item_id: int = 0) -> pd.DataFrame:
    """One row per origin t; features use data <= t only; targets tgt_h = log y_{t+h} - log y_t."""
    y = spec.y
    ly = np.log(y)
    dl = 100 * ly.diff()
    f = pd.DataFrame(index=y.index)
    for k in (0, 1, 2, 3, 5, 11):
        f[f"dl_L{k}"] = dl.shift(k)
    f["mom3"] = dl.rolling(3, min_periods=2).mean()
    f["mom12"] = dl.rolling(12, min_periods=6).mean()
    f["dev12"] = 100 * (ly - ly.rolling(12, min_periods=6).mean())
    f["vol6"] = dl.rolling(6, min_periods=4).std()
    for name, s in spec.lgbm_exog.items():
        ds = 100 * np.log(s).diff().reindex(y.index)
        for k in (0, 1, 2):
            f[f"{name}_dl_L{k}"] = ds.shift(k)
    idx = f.index
    f["crisis"] = ((idx >= CRISIS[0]) & (idx <= CRISIS[1])).astype(int)
    f["post_float"] = (idx >= "2022-03-01").astype(int)
    f["item"] = item_id
    f["origin_month"] = idx.month
    for h in range(1, H + 1):
        f[f"tgt_{h}"] = 100 * (ly.shift(-h) - ly)
    f["origin_ok"] = y.notna()
    f["series"] = spec.name
    return f


LGB_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                  n_jobs=1)  # tiny data: threading overhead dominates


def lgbm_forecasts(feats: pd.DataFrame, specs_by_name: dict, origins: dict, seed: int = 0) -> pd.DataFrame:
    """Direct LightGBM forecasts for every (series, origin, h). `feats` may pool several series
    (item feature distinguishes them). Training rows for origin t / horizon h are rows with
    origin <= t - h (so the target month is <= t, i.e. already observed)."""
    import lightgbm as lgb
    xcols = [c for c in feats.columns if not c.startswith("tgt_") and c not in ("origin_ok", "series")]
    feats = feats.copy()
    feats["origin"] = feats.index
    out = []
    all_origins = sorted({o for v in origins.values() for o in v})
    for t in all_origins:
        for h in range(1, H + 1):
            train = feats[(feats["origin"] <= t - pd.DateOffset(months=h)) & feats[f"tgt_{h}"].notna()
                          & feats["origin_ok"]]
            if len(train) < 20:
                continue
            Xtr = train[xcols].copy()
            Xtr["target_month"] = ((train["origin_month"] - 1 + h) % 12) + 1
            m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xtr, train[f"tgt_{h}"], categorical_feature=["item"])
            test = feats[(feats["origin"] == t) & feats["origin_ok"]]
            test = test[test["series"].map(lambda s: t in origins.get(s, ()))]
            if test.empty:
                continue
            Xte = test[xcols].copy()
            Xte["target_month"] = ((test["origin_month"] - 1 + h) % 12) + 1
            pred = m.predict(Xte)
            for (_, row), p in zip(test.iterrows(), pred):
                base = specs_by_name[row["series"]].y.loc[t]
                out.append({"series": row["series"], "origin": t, "horizon": h,
                            "target_date": t + pd.DateOffset(months=h), "forecast": base * np.exp(p / 100),
                            "model": "lightgbm"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ backtest
def origins_for(spec: SeriesSpec, start: str | None = None, min_train: int = 60) -> list[pd.Timestamp]:
    y = spec.y
    first = y.first_valid_index()
    earliest = first + pd.DateOffset(months=min_train - 1)
    if start:
        earliest = max(earliest, pd.Timestamp(start))
    last = y.last_valid_index() - pd.DateOffset(months=1)
    return [t for t in pd.date_range(earliest, last, freq="MS") if pd.notna(y.get(t))]


def backtest_stat(spec: SeriesSpec, origins: list[pd.Timestamp]) -> tuple[pd.DataFrame, dict]:
    """naive, seasonal naive, SARIMA, SARIMAX forecasts at each origin (orders fixed from the
    initial window ending at the first origin)."""
    init = spec.y.loc[:origins[0]]
    init = init.loc[init.first_valid_index():]
    order, sorder = select_order(init, None, spec.seasonal)
    info = {"sarima_order": order, "sarima_seasonal": sorder}
    use_x = bool(spec.drivers)
    if use_x:
        ex = exog_matrix(spec, init.index, origins[0])
        ex = ex[ex.columns[ex.nunique() > 1]]
        xo, xs = select_order(init, ex if len(ex.columns) else None, spec.seasonal)
        info.update({"sarimax_order": xo, "sarimax_seasonal": xs, "sarimax_exog": list(spec.drivers)})
    rows = []
    for t in origins:
        fidx = future_index(t)
        preds = {"naive": f_naive(spec.y, t), "seasonal_naive": f_seasonal_naive(spec.y, t)}
        try:
            preds["sarima"] = f_sarima(spec, t, order, sorder, use_exog=False)
        except Exception:
            pass
        if use_x:
            try:
                preds["sarimax"] = f_sarima(spec, t, xo, xs, use_exog=True)
            except Exception:
                pass
        for model, p in preds.items():
            for h, (d, v) in enumerate(zip(fidx, p), start=1):
                rows.append({"series": spec.name, "model": model, "origin": t, "horizon": h,
                             "target_date": d, "forecast": v})
    return pd.DataFrame(rows), info


def attach_actuals(bt: pd.DataFrame, specs_by_name: dict) -> pd.DataFrame:
    bt = bt.copy()
    bt["actual"] = [specs_by_name[s].y.get(d, np.nan) for s, d in zip(bt["series"], bt["target_date"])]
    bt = bt.dropna(subset=["actual", "forecast"])
    bt["regime"] = np.where((bt["target_date"] >= CRISIS[0]) & (bt["target_date"] <= CRISIS[1]),
                            "crisis_2022", "routine")
    bt["error"] = bt["forecast"] - bt["actual"]
    return bt


def score(bt: pd.DataFrame, by=("series", "model", "regime", "horizon"), eps: float = 1e-8) -> pd.DataFrame:
    """MAE, RMSE (price units) and MAPE (%). MAPE skips actuals with |actual| < eps (prices here
    are strictly positive, so nothing is dropped in practice; the guard keeps it safe)."""
    def agg(g):
        e = g["error"]
        ok = g["actual"].abs() > eps
        return pd.Series({"n": len(g), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                          "MAPE_%": (e[ok].abs() / g.loc[ok, "actual"].abs()).mean() * 100 if ok.any() else np.nan})
    s = bt.groupby(list(by)).apply(agg).reset_index()
    # relative MAE vs the naive model on the same cells (<1 = better than naive)
    key = [c for c in by if c != "model"]
    naive = s[s["model"] == "naive"].set_index(key)["MAE"]
    s["MAE_vs_naive"] = s.apply(lambda r: r["MAE"] / naive.get(tuple(r[k] for k in key), np.nan), axis=1)
    return s


# ------------------------------------------------------------------ orchestration
def backtest_origins(spec: SeriesSpec) -> list[pd.Timestamp]:
    """Long series: monthly origins from 2010-01 (>= 60 training months).
    Recent series: monthly origins after a 24-month initial window."""
    if spec.group == "long":
        return origins_for(spec, start="2010-01-01", min_train=60)
    return origins_for(spec, min_train=24)


def run_backtests(specs: list[SeriesSpec], verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    S = {s.name: s for s in specs}
    origins = {s.name: backtest_origins(s) for s in specs}
    parts, infos = [], []
    for s in specs:
        bt, info = backtest_stat(s, origins[s.name])
        parts.append(bt)
        infos.append({"series": s.name, "group": s.group, "n_origins": len(origins[s.name]),
                      "first_origin": origins[s.name][0].strftime("%Y-%m"),
                      "last_origin": origins[s.name][-1].strftime("%Y-%m"), "note": s.note, **info})
        if verbose:
            print(f"stat models done: {s.name} ({len(origins[s.name])} origins)")
    for s in [s for s in specs if s.group == "long"]:
        parts.append(lgbm_forecasts(make_features(s), S, {s.name: origins[s.name]}))
        if verbose:
            print(f"lightgbm done: {s.name}")
    recent = [s for s in specs if s.group == "recent"]
    pooled = pd.concat([make_features(s, i) for i, s in enumerate(recent)])
    parts.append(lgbm_forecasts(pooled, S, {s.name: origins[s.name] for s in recent}))
    if verbose:
        print("lightgbm done: recent items (pooled)")
    bt = attach_actuals(pd.concat(parts, ignore_index=True), S)
    return bt, pd.DataFrame(infos)


def best_models(scores: pd.DataFrame) -> pd.DataFrame:
    """Best model per series = lowest mean routine MAE over horizons 1..6."""
    r = scores[scores["regime"] == "routine"].groupby(["series", "model"])["MAE"].mean().reset_index()
    best = r.loc[r.groupby("series")["MAE"].idxmin()].rename(columns={"model": "best_model", "MAE": "mean_MAE_h1_6"})
    naive = r[r["model"] == "naive"].set_index("series")["MAE"]
    best["vs_naive"] = best["mean_MAE_h1_6"] / best["series"].map(naive)
    return best.reset_index(drop=True)


def final_forecasts(specs: list[SeriesSpec], infos: pd.DataFrame, info_date: pd.Timestamp) -> pd.DataFrame:
    """6-month-ahead forecasts from each forecastable series' last observation, all models.
    Drivers are used up to `info_date` (their real last observation), then held constant."""
    S = {s.name: s for s in specs}
    rows = []
    for s in [s for s in specs if s.forecastable]:
        t = s.y.last_valid_index()
        fidx = future_index(t)
        inf = infos.set_index("series").loc[s.name]
        preds = {"naive": (f_naive(s.y, t), None, None), "seasonal_naive": (f_seasonal_naive(s.y, t), None, None)}
        preds["sarima"] = f_sarima(s, t, inf["sarima_order"], inf["sarima_seasonal"], False, return_ci=True)
        if s.drivers:
            preds["sarimax"] = f_sarima(s, t, inf["sarimax_order"], inf["sarimax_seasonal"], True,
                                        info_date=info_date, return_ci=True)
        for model, (mean, c80, c95) in preds.items():
            for h, d in enumerate(fidx):
                rows.append({"series": s.name, "model": model, "origin": t, "horizon": h + 1, "target_date": d,
                             "forecast": mean[h],
                             "lo80": c80[h, 0] if c80 is not None else np.nan, "hi80": c80[h, 1] if c80 is not None else np.nan,
                             "lo95": c95[h, 0] if c95 is not None else np.nan, "hi95": c95[h, 1] if c95 is not None else np.nan})
    long_ = [s for s in specs if s.group == "long" and s.forecastable]
    for s in long_:
        lg = lgbm_forecasts(make_features(s), S, {s.name: [s.y.last_valid_index()]})
        rows += lg.assign(lo80=np.nan, hi80=np.nan, lo95=np.nan, hi95=np.nan).to_dict("records")
    recent = [s for s in specs if s.group == "recent"]
    pooled = pd.concat([make_features(s, i) for i, s in enumerate(recent)])
    lg = lgbm_forecasts(pooled, S, {s.name: [s.y.last_valid_index()] for s in recent if s.forecastable})
    rows += lg.assign(lo80=np.nan, hi80=np.nan, lo95=np.nan, hi95=np.nan).to_dict("records")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ selection & intervals
def best_by_horizon(scores: pd.DataFrame) -> pd.DataFrame:
    """Lowest routine MAE model for every (series, horizon)."""
    r = scores[scores["regime"] == "routine"]
    b = r.loc[r.groupby(["series", "horizon"])["MAE"].idxmin(), ["series", "horizon", "model", "MAE", "MAE_vs_naive"]]
    return b.rename(columns={"model": "best_model"}).reset_index(drop=True)


def empirical_intervals(bt: pd.DataFrame, fc: pd.DataFrame, levels=(0.8, 0.95)) -> pd.DataFrame:
    """Model-agnostic prediction intervals: apply the quantiles of each (series, model, horizon)'s
    routine backtest log-errors log(actual/forecast) to the final forecast."""
    r = bt[bt["regime"] == "routine"].copy()
    r["logerr"] = np.log(r["actual"] / r["forecast"])
    out = fc.copy()
    for lv in levels:
        lo_q, hi_q = (1 - lv) / 2, 1 - (1 - lv) / 2
        q = r.groupby(["series", "model", "horizon"])["logerr"].quantile([lo_q, hi_q]).unstack()
        q.columns = ["qlo", "qhi"]
        m = out.merge(q.reset_index(), on=["series", "model", "horizon"], how="left")
        tag = int(lv * 100)
        out[f"emp_lo{tag}"] = (m["forecast"] * np.exp(m["qlo"])).values
        out[f"emp_hi{tag}"] = (m["forecast"] * np.exp(m["qhi"])).values
    return out


def selected_forecasts(fc: pd.DataFrame, best_h: pd.DataFrame) -> pd.DataFrame:
    """Forecast path that uses, at each horizon, the model with the best routine backtest MAE."""
    fc = fc.drop(columns=[c for c in ("best_model", "is_best_model") if c in fc.columns])
    sel = fc.merge(best_h[["series", "horizon", "best_model"]], on=["series", "horizon"])
    return sel[sel["model"] == sel["best_model"]].drop(columns="best_model").sort_values(["series", "horizon"])
