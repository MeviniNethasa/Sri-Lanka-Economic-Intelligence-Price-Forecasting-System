"""Phase 5 – anomaly detection for macro, commodity and fuel series.

Three complementary detectors:
  1. STL residual robust z-scores. Seasonal-trend decomposition (period 12) where a
     series has enough history and is seasonal; otherwise a centred rolling-median
     trend; for administered fuel prices, the month-on-month revision itself. Residuals are scored with a median/MAD z-score. This is a *retrospective*
     detector: the decomposition is two-sided and uses months on both sides.
  2. Isolation Forest on causal features. Log returns, lagged return, deviation from
     the trailing 12-month median and a volatility-scaled shock use only data dated
     <= t. Each group's model is refitted every January on rows dated before that
     January, and the flag threshold is a quantile of the training scores, so a
     month's score never uses later data.
  3. Change-point detection with PELT. It uses `ruptures` when installed, otherwise
     an equivalent built-in PELT with the same costs: "normal" (mean + variance
     shift) on monthly log changes for market prices, and "l2" (mean shift) on the
     log level for administered fuel prices.

Data rules (from Phases 1–4):
  * Missing months stay missing. STL fits over short interpolated gaps, but
    interpolated months are never scored. Isolation Forest drops incomplete
    feature rows and never imputes them. Change points use only observed changes.
  * CPI uses the cleaned, chain-linked series. The raw IMF series is scored too,
    only as a diagnostic, to show that the Sep-2024 NCPI→CCPI switch is an artifact.
  * Fuel months flagged `possible_missing_revision` are dropped.

Detected anomalies are then classified against a catalogue of known events and
data quirks (`EVENTS`), plus cross-source (WFP vs CBSL) confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
# Direct submodule import (statsmodels.api breaks on statsmodels < 0.14.5 + SciPy >= 1.16)
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

Z_THRESH = 3.5        # Iglewicz–Hoaglin robust z cut-off
IF_QUANTILE = 0.975   # Isolation Forest flag threshold (quantile of training scores)
IF_MIN_TRAIN = 36     # minimum pooled training rows before a group is scored
MIN_STL = 36          # minimum observed months for a seasonal STL decomposition


# ---------------------------------------------------------------- series specs
@dataclass
class AnomalySpec:
    name: str
    y: pd.Series                  # monthly level, MS frequency, NaN = missing
    group: str                    # macro | staple | vegetable | fuel
    seasonal: bool = True         # use STL (period 12) when long enough
    cpd: str = "dlog"             # change-point signal: "dlog" (normal cost) or "level" (l2 cost)
    train: bool = True            # include in Isolation Forest training (False for diagnostics)
    scale_floor: float = 0.25     # floor on the robust scale (100×log units)
    source: pd.Series | None = None   # source label per month (splice detection)
    alt: pd.Series | None = None      # second source for cross-checking (CBSL)
    if_pool: str | None = None    # Isolation Forest model key (default: the group; own name for long series)
    cpd_min_size: int = 6         # minimum change-point segment length (months)
    note: str = ""


def _ms(s: pd.Series) -> pd.Series:
    s = s.dropna()
    s = s[s > 0]
    return s.asfreq("MS") if len(s) else s


def build_specs() -> list[AnomalySpec]:
    import relationships as R  # local import: src/ is on sys.path for scripts and notebooks
    df = R.load_monthly()
    macro = pd.read_csv(PROC / "macro_monthly.csv", parse_dates=["date"], index_col="date")
    cont = pd.read_csv(PROC / "commodity_prices_continuous_monthly.csv", parse_dates=["date"])
    ov = pd.read_csv(PROC / "wfp_cbsl_overlap_monthly.csv", parse_dates=["date"])

    def cs(name):
        g = cont[cont["commodity"] == name].set_index("date").sort_index()
        return _ms(g["price_lkr"]), g["source"].reindex(_ms(g["price_lkr"]).index)

    def alt(name):
        g = ov[ov["commodity"] == name].set_index("date")["cbsl_retail_mean"]
        return _ms(g) if len(g) else None

    specs = [
        AnomalySpec("CPI (cleaned)", _ms(df["cpi_headline_index"]), "macro", scale_floor=0.3,
                    if_pool="CPI", cpd_min_size=12, note="chain-linked NCPI→CCPI (Phase 1)"),
        AnomalySpec("CPI (raw IMF, diagnostic)", _ms(macro["imf_cpi_index_raw"]), "macro", train=False,
                    scale_floor=0.3, if_pool="CPI", cpd_min_size=12,
                    note="uncleaned IMF series; contains the Sep-2024 index switch; scored with the "
                         "model trained on cleaned CPI"),
        AnomalySpec("LKR/USD", _ms(df["lkr_per_usd"]), "macro", seasonal=False, scale_floor=0.5,
                    if_pool="LKR/USD", cpd_min_size=12),
        AnomalySpec("Brent (USD/bbl)", _ms(df["world_brent_usd_bbl"]), "macro", seasonal=False,
                    scale_floor=1.0, if_pool="Brent", cpd_min_size=12,
                    note="external driver of fuel revisions"),
    ]
    for name, seasonal in [("Rice (white)", True), ("Coconut", True), ("Lentils", False),
                           ("Onions (imported)", False)]:
        y, src = cs(name)
        long_ = name == "Rice (white)"
        specs.append(AnomalySpec(name, y, "staple", seasonal=seasonal, source=src, alt=alt(name),
                                 if_pool=name if long_ else None, cpd_min_size=12 if long_ else 6))
    specs.append(AnomalySpec("Wheat flour", _ms(df["wfp_wheat_flour"].loc[:"2022-06"]), "staple",
                             seasonal=False, if_pool="Wheat flour", cpd_min_size=12, scale_floor=1.0,
                             note="WFP series ends 2022-06; importer-set step prices"))
    specs.append(AnomalySpec("Sugar (CBSL)", _ms(df["cbsl_sugar"]), "staple", seasonal=False,
                             note="CBSL retail mean, from 2023-01"))
    for name in ["Beans", "Cabbage", "Carrots", "Tomatoes", "Potatoes (local)"]:
        y, src = cs(name)
        specs.append(AnomalySpec(name, y, "vegetable", source=src, alt=alt(name),
                                 note="WFP national median to 2025-09, level-adjusted CBSL after"))
    for prod, label in [("auto_diesel", "Diesel"), ("petrol_92", "Petrol 92"),
                        ("kerosene", "Kerosene"), ("lpg_12_5kg", "LPG 12.5 kg")]:
        col = f"fuel_{prod}"
        if col in df:
            specs.append(AnomalySpec(label, _ms(df[col]), "fuel", seasonal=False, cpd="level",
                                     scale_floor=2.0,
                                     note="administered price; possible_missing_revision months dropped"))
    return specs


# ------------------------------------------------------------------ helpers
def robust_z(x: pd.Series, floor: float = 0.0) -> pd.Series:
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return (x - med) / max(mad, floor, 1e-12)


def lp(s: pd.Series) -> pd.Series:
    """100 × log level."""
    return 100 * np.log(s)


# ------------------------------------------------------ 1. STL residual z-scores
def stl_scores(spec: AnomalySpec) -> pd.DataFrame:
    y = lp(spec.y)
    obs = y.notna()
    # interpolate internal gaps for fitting only
    yi = y.interpolate(limit_area="inside")
    use_stl = spec.seasonal and obs.sum() >= MIN_STL
    if spec.cpd == "level":
        # administered step prices: score the size of each revision (month-on-month jump)
        trend = y.shift(1)
        seas = pd.Series(0.0, index=y.index)
        resid = y.diff()
        method = "jump"
    elif use_stl:
        res = STL(yi, period=12, seasonal=13, robust=True).fit()
        trend, seas, resid = res.trend, res.seasonal, res.resid
        method = "stl"
    else:
        trend = yi.rolling(7, center=True, min_periods=4).median()
        seas = pd.Series(0.0, index=yi.index)
        resid = yi - trend
        method = "rolling_median"
    resid = resid.where(obs)
    z = robust_z(resid, spec.scale_floor)
    pos = obs.cumsum().where(obs)
    edge = (pos <= 3) | (pos > obs.sum() - 3)   # two-sided smoothers are least reliable here
    return pd.DataFrame({"log_level": y, "trend": trend, "seasonal": seas, "resid": resid,
                         "stl_z": z, "stl_flag": z.abs() >= Z_THRESH, "stl_method": method,
                         "edge": edge.fillna(False).astype(bool)})


# -------------------------------------------- 2. Isolation Forest (causal features)
FEATURES = ["r1", "r1_lag1", "r3", "dev12", "shock"]


def causal_features(spec: AnomalySpec) -> pd.DataFrame:
    """Features at month t use only observations dated <= t."""
    y = lp(spec.y)
    r1 = y.diff(1)
    vol = r1.shift(1).rolling(6, min_periods=4).std()
    f = pd.DataFrame({
        "r1": r1,
        "r1_lag1": r1.shift(1),
        "r3": y.diff(3),
        "dev12": y - y.shift(1).rolling(12, min_periods=6).median(),
        "shock": r1 / np.maximum(vol, spec.scale_floor * 4),
    })
    pool = spec.if_pool or ("recent_commodity" if spec.group in ("staple", "vegetable") else spec.group)
    return f.assign(series=spec.name, group=spec.group, train=spec.train, pool=pool)


def isolation_forest_scores(specs: list[AnomalySpec], seed: int = 0) -> pd.DataFrame:
    """Expanding-window Isolation Forest, refitted each January. Long series get their own
    model; short series are pooled within their group (features are scale-free)."""
    from sklearn.ensemble import IsolationForest
    feats = pd.concat([causal_features(s) for s in specs])
    feats.index.name = "date"
    feats = feats.reset_index()
    complete = feats[FEATURES].notna().all(axis=1)
    out = []
    for pool, g in feats[complete].groupby("pool"):
        years = sorted(g["date"].dt.year.unique())
        for yr in years:
            cut = pd.Timestamp(f"{yr}-01-01")
            tr = g[(g["date"] < cut) & g["train"]]
            te = g[(g["date"] >= cut) & (g["date"] < cut + pd.DateOffset(years=1))]
            if len(tr) < IF_MIN_TRAIN or te.empty:
                continue
            m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(tr[FEATURES])
            tr_s = -m.score_samples(tr[FEATURES])
            te_s = -m.score_samples(te[FEATURES])
            thr = np.quantile(tr_s, IF_QUANTILE)
            out.append(te[["date", "series"]].assign(
                if_score=te_s, if_threshold=thr, if_flag=te_s > thr,
                if_pctile=[(tr_s < v).mean() for v in te_s], if_train_rows=len(tr)))
    sc = pd.concat(out) if out else pd.DataFrame(columns=["date", "series"])
    dropped = (~complete).groupby(feats["series"]).sum().rename("if_rows_incomplete")
    return sc, dropped


# --------------------------------------------------------- 3. change points (PELT)
def _seg_cost_fn(x: np.ndarray, model: str, var_floor: float):
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def cost(a: int, b: int) -> float:
        m = b - a
        mu = (s1[b] - s1[a]) / m
        ss = (s2[b] - s2[a]) - m * mu * mu
        if model == "l2":
            return ss
        return m * np.log(max(ss / m, var_floor))
    return cost


def pelt(x: np.ndarray, model: str = "normal", pen: float = 10.0, min_size: int = 6,
         var_floor: float = 1e-2) -> list[int]:
    """Exact penalised segmentation by PELT. Returns segment end indices (last = len(x)),
    the same convention as `ruptures.Pelt(...).predict(pen)`."""
    n = len(x)
    cost = _seg_cost_fn(np.asarray(x, float), model, var_floor)
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    R = [0]
    for t in range(1, n + 1):
        cands = [s for s in R if t - s >= min_size]
        if cands:
            vals = np.array([F[s] + cost(s, t) + pen for s in cands])
            k = int(vals.argmin())
            F[t], last[t] = vals[k], cands[k]
            keep = [s for s, v in zip(cands, vals) if v - pen <= F[t]]
        else:
            keep = []
        R = keep + [s for s in R if t - s < min_size] + [t]
    bk, t = [], n
    while t > 0:
        bk.append(t)
        t = last[t]
    return sorted(bk)


def optimal_partition(x, model="normal", pen=10.0, min_size=6, var_floor=1e-2) -> list[int]:
    """Unpruned O(n²) dynamic programme; used only to test `pelt`."""
    n = len(x)
    cost = _seg_cost_fn(np.asarray(x, float), model, var_floor)
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    for t in range(min_size, n + 1):
        for s in range(0, t - min_size + 1):
            if np.isfinite(F[s]):
                v = F[s] + cost(s, t) + pen
                if v < F[t]:
                    F[t], last[t] = v, s
    bk, t = [], n
    while t > 0:
        bk.append(t)
        t = last[t]
    return sorted(bk)


def detect_breaks(x: np.ndarray, model: str, pen: float, min_size: int) -> tuple[list[int], str]:
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(np.asarray(x, float).reshape(-1, 1))
        return algo.predict(pen=pen), "ruptures"
    except ImportError:
        return pelt(x, model=model, pen=pen, min_size=min_size), "builtin_pelt"


def change_points(spec: AnomalySpec, beta: float = 3.0, min_size: int | None = None) -> pd.DataFrame:
    """Segments of a series. "dlog": normal cost on observed monthly log changes (changes
    in trend growth and volatility). "level": l2 cost on the observed log level (step changes)."""
    min_size = min_size or spec.cpd_min_size
    y = lp(spec.y)
    sig = (y.diff() if spec.cpd == "dlog" else y).dropna()
    n = len(sig)
    if n < 2 * min_size:
        return pd.DataFrame()
    x = sig.values
    if spec.cpd == "dlog":
        model, pen = "normal", beta * np.log(n)
    else:
        sigma2 = max((1.4826 * np.median(np.abs(np.diff(x)))) ** 2 / 2, spec.scale_floor ** 2)
        model, pen = "l2", beta * np.log(n) * sigma2
    bk, engine = detect_breaks(x, model, pen, min_size)
    rows, a = [], 0
    for b in bk:
        seg = sig.iloc[a:b]
        prev = sig.index[a - 1] if a > 0 else None
        gap = 0 if prev is None else (seg.index[0].year - prev.year) * 12 + seg.index[0].month - prev.month - 1
        rows.append({"series": spec.name, "group": spec.group, "signal": spec.cpd, "engine": engine,
                     "start": seg.index[0], "end": seg.index[-1], "n_obs": len(seg),
                     "mean": seg.mean(), "sd": seg.std(ddof=0), "missing_months_before": gap})
        a = b
    out = pd.DataFrame(rows)
    out["is_break"] = out.index > 0
    return out


# ---------------------------------------------------------------- run all
def score_all(specs: list[AnomalySpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (monthly scores, change-point segments, IF incomplete-row counts)."""
    parts = []
    for s in specs:
        d = stl_scores(s)
        d["series"], d["group"], d["train"] = s.name, s.group, s.train
        d["price"] = s.y
        d["source"] = s.source.reindex(d.index) if s.source is not None else None
        parts.append(d.rename_axis("date").reset_index())
    sc = pd.concat(parts, ignore_index=True)
    sc = sc[sc["price"].notna()]
    iso, dropped = isolation_forest_scores(specs)
    sc = sc.merge(iso, on=["date", "series"], how="left")
    sc["if_flag"] = sc["if_flag"].astype("boolean").fillna(False).astype(bool)
    sc["if_scored"] = sc["if_score"].notna()

    segs = pd.concat([change_points(s) for s in specs], ignore_index=True)
    breaks = segs[segs["is_break"]][["series", "start"]].rename(columns={"start": "date"})
    sc["cp_near"] = False
    for _, b in breaks.iterrows():
        m = (sc["series"] == b["series"]) & ((sc["date"] - b["date"]).abs() <= pd.Timedelta(days=62))
        sc.loc[m, "cp_near"] = True
    sc["r1"] = sc.groupby("series")["log_level"].diff().where(
        sc.groupby("series")["date"].diff() <= pd.Timedelta(days=31))
    sc["n_methods"] = sc[["stl_flag", "if_flag"]].sum(axis=1) + sc["cp_near"].astype(int)
    sc["anomaly"] = sc["stl_flag"] | sc["if_flag"]
    return sc, segs, dropped


# ---------------------------------------------------------- known events
@dataclass
class Event:
    id: str
    name: str
    start: str
    end: str
    kind: str                 # economic_shock | policy | weather_shock | seasonal | data_artifact
    series: list = field(default_factory=list)   # series names or group names ("*" = all)
    expect: str = "detect"    # detect | not_detect (artifact must be absent from cleaned data) | check
    note: str = ""
    source: str = ""


EVENTS = [
    # source: "project" = shown in this project's data/notes; "cited" = checked in a news/agency
    # source during Phase 5 (see PHASE5_NOTES.md); "background" = well-documented event, not
    # separately verified here.
    Event("tsunami_2004", "Indian Ocean tsunami", "2004-12-01", "2005-02-01", "economic_shock",
          ["CPI (cleaned)", "CPI (raw IMF, diagnostic)"], note="26 Dec 2004", source="background"),
    Event("global_2008", "Global food and oil price spike", "2007-10-01", "2008-10-01", "economic_shock",
          ["macro", "Rice (white)", "Wheat flour"], note="world grain and oil peak mid-2008",
          source="background"),
    Event("floods_2011", "Floods hit the Maha paddy harvest", "2011-01-01", "2011-06-01", "weather_shock",
          ["Rice (white)"], note="Jan–Feb 2011 floods in the east", source="background"),
    Event("lkr_2012", "LKR allowed to depreciate", "2012-02-01", "2012-04-01", "policy",
          ["LKR/USD"], note="CBSL stopped defending the rate in Feb 2012", source="background"),
    Event("price_cuts_2015", "Post-election price cuts", "2015-01-01", "2015-02-01", "policy",
          ["CPI (cleaned)", "CPI (raw IMF, diagnostic)", "fuel"],
          note="Jan-2015 fuel price cuts and interim-budget cuts to essential goods", source="background"),
    Event("drought_2016_17", "2016–17 drought", "2016-10-01", "2017-06-01", "weather_shock",
          ["Rice (white)", "CPI (cleaned)"], note="failed Yala/Maha seasons; rice imports in 2017",
          source="background"),
    Event("oil_2014_16", "Oil price collapse", "2014-11-01", "2016-02-01", "economic_shock",
          ["Brent (USD/bbl)"], note="OPEC decision Nov 2014; Brent bottomed Jan 2016", source="background"),
    Event("political_2018", "Oct–Dec 2018 constitutional crisis", "2018-10-01", "2018-12-01", "policy",
          ["LKR/USD"], note="rupee sell-off", source="background"),
    Event("covid", "COVID-19 lockdowns", "2020-03-01", "2020-06-01", "economic_shock",
          ["*"], note="supply disruption; Phase 1 covid dummy 2020-03..2021-12", source="project"),
    Event("fertiliser_ban", "Chemical fertiliser import ban", "2021-05-01", "2021-12-01", "policy",
          ["Rice (white)", "vegetable"], note="ban from Apr/May 2021, eased Nov 2021", source="background"),
    Event("crisis_2022", "2022 economic crisis", "2021-09-01", "2023-06-01", "economic_shock",
          ["*"], note="FX, fuel and food shortages; Phase 1 window 2022-01..2023-06, here from 2021-09 "
                      "when inflation and staple prices started to accelerate", source="project"),
    Event("lkr_float", "LKR float and collapse", "2022-03-01", "2022-05-01", "policy",
          ["LKR/USD", "CPI (cleaned)"], note="peg abandoned Mar 2022", source="project"),
    Event("cabbage_wfp_2023_02", "WFP cabbage 48 LKR/kg", "2023-02-01", "2023-02-01", "data_artifact",
          ["Cabbage"], note="single WFP market (Peliyagoda); CBSL retail fell only ~23% (132→102)",
          source="project"),
    Event("wfp_fuel_swap_2017_08", "WFP diesel/petrol labels swapped", "2017-08-01", "2017-09-01",
          "data_artifact", ["Diesel", "Petrol 92"],
          note="Aug-2017 diesel 117 / petrol 95 – exactly the reverse of Jul and Sep (95 / 117) in "
               "every WFP market; found by the detectors, confirmed by inspection", source="project"),
    Event("vat_2024", "VAT raised to 18%", "2024-01-01", "2024-02-01", "policy",
          ["CPI (cleaned)", "CPI (raw IMF, diagnostic)"], note="from 1 Jan 2024", source="background"),
    Event("coconut_2024_25", "Coconut shortage", "2024-10-01", "2025-06-01", "economic_shock",
          ["Coconut"], note="low yields after dry weather; record nut prices in early 2025",
          source="background"),
    Event("cpi_switch_2024_09", "NCPI→CCPI index switch in IMF CPI", "2024-09-01", "2024-12-01",
          "data_artifact", ["CPI (raw IMF, diagnostic)"],
          note="raw IMF index drops 6.5% in Sep-2024 (Phase 1); the level shift also inflates the "
               "STL residuals of the next months", source="project"),
    Event("cpi_switch_clean", "Index switch removed by the Phase 1 fix", "2024-09-01", "2024-12-01",
          "data_artifact", ["CPI (cleaned)"], expect="not_detect",
          note="cleaned CPI changes −0.1% in Sep-2024", source="project"),
    Event("splice_2025_10", "WFP→CBSL splice", "2025-10-01", "2025-10-01", "data_artifact",
          ["vegetable", "Coconut", "Rice (white)", "Lentils", "Onions (imported)"], expect="check",
          note="continuous series switch to level-adjusted CBSL; a flag here may be a splice effect",
          source="project"),
    Event("ditwah", "Cyclone Ditwah floods", "2025-12-01", "2026-01-01", "weather_shock",
          ["vegetable"], note="landfall 27–28 Nov 2025; vegetable supply disruption",
          source="cited"),
    Event("fuel_2026", "2026 global oil shock and fuel hikes", "2026-03-01", "2026-06-01", "economic_shock",
          ["fuel", "CPI (cleaned)", "LKR/USD", "Brent (USD/bbl)"],
          note="Middle East conflict; fuel up ~8% (9 Mar) and ~25% (21–22 Mar 2026)", source="cited"),
    Event("veg_nov_2026", "Oct–Nov 2026 vegetable spike", "2026-10-01", "2026-11-01", "seasonal",
          ["vegetable"], expect="check",
          note="cause not verified; coincides with the usual Nov–Jan monsoon/lean-season peak; "
               "Nov-2026 covers CBSL reports to 25 Nov only", source="project"),
]


def _in_scope(ev: Event, spec_name: str, group: str) -> bool:
    return "*" in ev.series or spec_name in ev.series or group in ev.series


def classify(sc: pd.DataFrame, events: list[Event] = EVENTS, breadth_share: float = 0.25,
             min_observed: int = 8, min_flagged: int = 4) -> pd.DataFrame:
    """Label every flagged series-month. Order: known data artifact > splice month >
    cross-source disagreement > isolated spike-and-revert in a smooth series > known event >
    broad co-movement > unexplained."""
    a = sc[sc["anomaly"]].copy()
    trained = sc[sc["train"]]
    n_series = trained.groupby("date")["series"].nunique()
    n_flag = trained[trained["anomaly"]].groupby("date")["series"].nunique()
    share = (n_flag / n_series).where((n_series >= min_observed) & (n_flag >= min_flagged))
    breadth = share.reindex(a["date"]).fillna(0).values
    a["breadth"] = breadth
    nxt = sc.sort_values("date").groupby("series")["r1"].shift(-1)
    a["next_r1"] = nxt.reindex(a.index)
    labels, why = [], []
    for (_, r), br in zip(a.iterrows(), breadth):
        lab, w = None, ""
        for ev in events:
            if ev.kind == "data_artifact" and ev.expect != "check" and _in_scope(ev, r["series"], r["group"]) \
                    and pd.Timestamp(ev.start) <= r["date"] <= pd.Timestamp(ev.end):
                lab, w = "data_artifact", ev.id
                break
        if lab is None and isinstance(r.get("source"), str) and r.get("splice"):
            lab, w = "possible_splice_artifact", "first month of a new source"
        if lab is None and pd.notna(r.get("alt_r1")) and pd.notna(r.get("r1")) and abs(r["r1"]) >= 15:
            if np.sign(r["alt_r1"]) != np.sign(r["r1"]) or abs(r["alt_r1"]) < abs(r["r1"]) / 3:
                lab, w = "possible_data_artifact", f"CBSL moved {r['alt_r1']:+.0f} vs {r['r1']:+.0f} (100×log)"
        if lab is None and r["group"] in ("macro", "staple") and pd.notna(r.get("r1")) \
                and pd.notna(r.get("next_r1")) and abs(r["r1"]) >= 30 and -r["next_r1"] / r["r1"] >= 2 / 3:
            lab, w = "possible_data_artifact", (f"isolated spike {r['r1']:+.0f} then {r['next_r1']:+.0f} "
                                                "(100×log) in a normally smooth series")
        if lab is None:
            for ev in events:
                if ev.kind != "data_artifact" and _in_scope(ev, r["series"], r["group"]) \
                        and pd.Timestamp(ev.start) - pd.DateOffset(months=1) <= r["date"] <= pd.Timestamp(ev.end) + pd.DateOffset(months=1):
                    lab = "seasonal_or_unverified" if ev.kind == "seasonal" else "economic_shock"
                    w = ev.id
                    break
        if lab is None and br >= breadth_share:
            lab, w = "economic_shock", f"broad: {br:.0%} of series flagged"
        if lab is None:
            lab = "unexplained"
            if pd.notna(r.get("alt_r1")) and pd.notna(r.get("r1")) and np.sign(r["alt_r1"]) == np.sign(r["r1"]):
                w = "same-direction move in CBSL"
        labels.append(lab)
        why.append(w)
    a["classification"], a["reason"] = labels, why
    return a


def add_cross_source(sc: pd.DataFrame, specs: list[AnomalySpec]) -> pd.DataFrame:
    """Attach the CBSL month-on-month change (alt_r1) and a splice marker."""
    alts, spl = [], []
    for s in specs:
        if s.alt is not None:
            r = (100 * np.log(s.alt)).diff()
            if s.source is not None:   # only a cross-check while the primary month is WFP
                r = r.where(s.source.reindex(r.index) == "wfp")
            alts.append(pd.DataFrame({"date": r.index, "series": s.name, "alt_r1": r.values}))
        if s.source is not None:
            src = s.source.dropna()
            ch = src != src.shift()
            ch.iloc[0] = False
            spl.append(pd.DataFrame({"date": src.index, "series": s.name, "splice": ch.values}))
    sc = sc.drop(columns=[c for c in ["alt_r1", "splice"] if c in sc])
    if alts:
        sc = sc.merge(pd.concat(alts), on=["date", "series"], how="left")
    if spl:
        sc = sc.merge(pd.concat(spl), on=["date", "series"], how="left")
    sc["splice"] = sc.get("splice", pd.Series(False, index=sc.index)).astype("boolean").fillna(False).astype(bool)
    return sc


def validate_events(sc: pd.DataFrame, segs: pd.DataFrame, specs: list[AnomalySpec],
                    events: list[Event] = EVENTS) -> pd.DataFrame:
    """For each event × in-scope series: was it flagged by STL / IF / a nearby change point?"""
    rows = []
    br = segs[segs["is_break"]]
    for ev in events:
        lo = pd.Timestamp(ev.start) - pd.DateOffset(months=1)
        hi = pd.Timestamp(ev.end) + pd.DateOffset(months=1)
        if ev.kind == "data_artifact":      # artifacts are dated exactly
            lo, hi = pd.Timestamp(ev.start), pd.Timestamp(ev.end)
        for s in specs:
            if not _in_scope(ev, s.name, s.group):
                continue
            w = sc[(sc["series"] == s.name) & sc["date"].between(lo, hi)]
            if w.empty:
                continue
            b = br[(br["series"] == s.name) & br["start"].between(lo - pd.DateOffset(months=1),
                                                                 hi + pd.DateOffset(months=1))]
            stl_hit, if_hit, cp_hit = bool(w["stl_flag"].any()), bool(w["if_flag"].any()), len(b) > 0
            hit = stl_hit or if_hit
            if ev.expect == "detect":
                outcome = "detected" if (hit or cp_hit) else "missed"
            elif ev.expect == "not_detect":
                outcome = "correctly_absent" if not hit else "false_alarm"
            else:
                outcome = "flagged" if hit else "not_flagged"
            top = w.loc[w["stl_z"].abs().idxmax()] if w["stl_z"].notna().any() else None
            rows.append({"event": ev.id, "event_name": ev.name, "kind": ev.kind, "series": s.name,
                         "window": f"{lo:%Y-%m}..{hi:%Y-%m}", "months_observed": len(w),
                         "stl_hit": stl_hit, "if_hit": if_hit, "if_scored": bool(w["if_scored"].any()),
                         "cp_hit": cp_hit, "max_abs_stl_z": None if top is None else abs(top["stl_z"]),
                         "peak_month": None if top is None else f"{top['date']:%Y-%m}",
                         "expect": ev.expect, "outcome": outcome})
    return pd.DataFrame(rows)


def seasonal_profile(sc: pd.DataFrame, group: str = "vegetable") -> pd.DataFrame:
    """Average STL seasonal factor by calendar month (≈ % above/below trend)."""
    g = sc[(sc["group"] == group) & (sc["stl_method"] == "stl")]
    return (g.assign(month=g["date"].dt.month).groupby(["series", "month"])["seasonal"].mean()
              .unstack("month").round(1))


def episodes(a: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """Merge flagged months of one series into episodes (gaps of <= max_gap months bridged).
    An episode takes the classification of its peak month (largest |STL z|)."""
    rows = []
    for s, g in a.sort_values("date").groupby("series"):
        ep, prev = 0, None
        ids = []
        for d in g["date"]:
            if prev is not None and (d.year - prev.year) * 12 + d.month - prev.month > max_gap + 1:
                ep += 1
            ids.append(ep)
            prev = d
        g = g.assign(ep=ids)
        for _, e in g.groupby("ep"):
            pk = e.loc[e["stl_z"].abs().idxmax()] if e["stl_z"].notna().any() else e.iloc[0]
            rows.append({"series": s, "group": e["group"].iloc[0], "start": e["date"].min(),
                         "end": e["date"].max(), "months_flagged": len(e),
                         "peak_month": pk["date"], "peak_stl_z": pk["stl_z"],
                         "if_months": int(e["if_flag"].sum()), "cp_near": bool(e["cp_near"].any()),
                         "classification": pk["classification"], "reason": pk["reason"],
                         "edge_of_series": bool(e["edge"].any())})
    return pd.DataFrame(rows).sort_values(["group", "series", "start"])


def events_table(events: list[Event] = EVENTS) -> pd.DataFrame:
    return pd.DataFrame([{"id": e.id, "name": e.name, "start": e.start[:7], "end": e.end[:7], "kind": e.kind,
                          "series": ", ".join(e.series), "expect": e.expect, "source": e.source,
                          "note": e.note} for e in events])
