"""LankaEcon – Sri Lanka Economic Intelligence & Price Forecasting dashboard.

Run locally:   streamlit run dashboard/app.py
Reads the pipeline's processed data and Phase 3–5 tables; nothing is re-trained in the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

st.set_page_config(page_title="LankaEcon – Sri Lanka prices", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- styling helpers
COL = {"shock": "#d62728", "artifact": "#1f77b4", "possible": "#9467bd", "unexplained": "#7f7f7f",
       "seasonal": "#ff7f0e", "ok": "#2ca02c", "warn": "#e6a700", "info": "#4c78a8"}
CLS_COLOR = {"economic_shock": COL["shock"], "data_artifact": COL["artifact"],
             "possible_data_artifact": COL["possible"], "possible_splice_artifact": COL["possible"],
             "unexplained": COL["unexplained"], "seasonal_or_unverified": COL["seasonal"]}
CLS_LABEL = {"economic_shock": "Economic shock", "data_artifact": "Data artifact",
             "possible_data_artifact": "Possible data artifact", "possible_splice_artifact": "Possible splice artifact",
             "unexplained": "Unexplained", "seasonal_or_unverified": "Seasonal / unverified"}

st.markdown("""
<style>
.badge {display:inline-block;padding:2px 9px;margin:2px 4px 2px 0;border-radius:11px;font-size:0.78rem;
        font-weight:600;color:white;white-space:nowrap}
.block-container {padding-top:1.6rem}
</style>""", unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def badges(*items: tuple[str, str]) -> None:
    st.markdown(" ".join(badge(t, c) for t, c in items), unsafe_allow_html=True)


def fig_layout(fig: go.Figure, h: int = 380, legend_top: bool = True) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      title=dict(font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, x=0, font=dict(size=11))
                      if legend_top else None)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- cached loaders
cache = st.cache_data(show_spinner=False, ttl=6 * 3600)
macro = cache(D.macro)
continuous = cache(D.continuous)
agreement = cache(D.agreement)
cbsl = cache(D.cbsl_monthly)
markets = cache(D.markets)
fuel = cache(D.fuel)
history = cache(D.forecast_history)
p3, p4, p5 = cache(D.phase3), cache(D.phase4), cache(D.phase5)
freshness = cache(D.freshness)

# ---------------------------------------------------------------- header + sidebar
st.title("LankaEcon – Sri Lanka Economic Intelligence & Price Forecasting")
st.caption("Monthly inflation, exchange rate, food and fuel prices for Sri Lanka, with forecasts, anomaly alerts "
           "and pass-through estimates. Data: IMF, DCS, CBSL, WFP/HDX, World Bank, FRED.")

with st.sidebar:
    st.header("Data freshness")
    fr = freshness()
    if not fr.empty:
        st.dataframe(fr.assign(last=fr["last"].dt.strftime("%Y-%m-%d")), hide_index=True, use_container_width=True)
    st.markdown(
        "- Tables are refreshed weekly by a GitHub Action (`weekly_refresh.yml`).\n"
        "- Forecasts and anomaly scores are **precomputed** by `scripts/09` and `scripts/11`.\n"
        "- Prices in LKR; changes are 100×log differences (≈ %).")
    st.divider()
    st.markdown("**Notes:** "
                "[Phase 1](https://github.com/MeviniNethasa/Sri-Lanka-Economic-Intelligence-Price-Forecasting-System/blob/main/PHASE1_NOTES.md) · "
                "[Phase 3](https://github.com/MeviniNethasa/Sri-Lanka-Economic-Intelligence-Price-Forecasting-System/blob/main/PHASE3_NOTES.md) · "
                "[Phase 4](https://github.com/MeviniNethasa/Sri-Lanka-Economic-Intelligence-Price-Forecasting-System/blob/main/PHASE4_NOTES.md) · "
                "[Phase 5](https://github.com/MeviniNethasa/Sri-Lanka-Economic-Intelligence-Price-Forecasting-System/blob/main/PHASE5_NOTES.md)")

tabs = st.tabs(["📊 Macro overview", "🥕 Commodity explorer", "🗺️ Regional map", "🔮 Forecasts",
                "🚨 Anomaly alerts", "🔗 Relationships"])

# ================================================================ 1. MACRO
with tabs[0]:
    m = macro()
    if m.empty:
        st.warning("macro_monthly.csv not found – run the data pipeline first.")
    else:
        cpi = m.dropna(subset=["cpi_headline_index"])
        last = cpi.iloc[-1]
        fx = m.dropna(subset=["lkr_per_usd"])
        k = st.columns(4)
        k[0].metric(f"Headline inflation, {last['date']:%b %Y}", f"{last['inflation_yoy_pct']:.1f}%",
                    f"{last['inflation_yoy_pct'] - cpi.iloc[-2]['inflation_yoy_pct']:+.1f} pp vs prior month",
                    delta_color="inverse")
        k[1].metric("CPI month-on-month", f"{last['inflation_mom_pct']:+.2f}%")
        k[2].metric(f"LKR per USD, {fx.iloc[-1]['date']:%b %Y}", f"{fx.iloc[-1]['lkr_per_usd']:.1f}",
                    f"{100 * (fx.iloc[-1]['lkr_per_usd'] / fx.iloc[-13]['lkr_per_usd'] - 1):+.1f}% y/y",
                    delta_color="inverse")
        peak = cpi.loc[cpi["inflation_yoy_pct"].idxmax()]
        k[3].metric("Crisis peak inflation", f"{peak['inflation_yoy_pct']:.1f}%", f"{peak['date']:%b %Y}",
                    delta_color="off")
        badges((f"Latest CPI source: {last['cpi_source_label']}", COL["info"]),
               ("Sep-2024 NCPI→CCPI break fixed (Phase 1)", COL["ok"]),
               ("YoY flagged Sep-2024 → Aug-2025 (spans the index switch)", COL["warn"]))

        start = st.slider("From year", 2000, int(cpi["date"].dt.year.max()), 2015, key="macro_from")
        cp = cpi[cpi["date"].dt.year >= start]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for src, g in cp.groupby("cpi_source_label"):
                fig.add_bar(x=g["date"], y=g["inflation_yoy_pct"], name=src)
            br = cp[cp["cpi_break_flag"] == 1]
            if len(br):
                fig.add_scatter(x=br["date"], y=br["inflation_yoy_pct"], mode="markers", name="YoY spans index switch",
                                marker=dict(symbol="x", color="black", size=7))
            fig.update_layout(title="Headline inflation, % y/y (coloured by source)", barmode="overlay")
            show(fig_layout(fig))
        with c2:
            raw = st.toggle("Compare with raw IMF index (shows the artifact)", value=False)
            fig = go.Figure()
            base = cp["cpi_headline_index"].iloc[0]
            fig.add_scatter(x=cp["date"], y=100 * cp["cpi_headline_index"] / base, name="Cleaned CPI")
            if raw:
                rr = m[m["date"].isin(cp["date"])]
                fig.add_scatter(x=rr["date"], y=100 * rr["imf_cpi_index_raw"] / rr["imf_cpi_index_raw"].iloc[0],
                                name="Raw IMF CPI", line=dict(dash="dot"))
                fig.add_vline(x=pd.Timestamp("2024-09-01"), line_dash="dash", line_color="grey")
            fig.update_layout(title=f"CPI index ({start} = 100)")
            show(fig_layout(fig))
        c3, c4 = st.columns(2)
        with c3:
            f = fx[fx["date"].dt.year >= start]
            fig = px.line(f, x="date", y="lkr_per_usd", title="LKR per USD")
            fig.add_vrect(x0="2022-01-01", x1="2023-06-30", fillcolor="red", opacity=0.08, line_width=0,
                          annotation_text="2022 crisis", annotation_position="top left")
            show(fig_layout(fig))
        with c4:
            w = m[m["date"].dt.year >= start].set_index("date")[["world_food_price_index", "world_brent_usd_bbl"]].dropna(how="all")
            fig = go.Figure()
            for c, n in [("world_food_price_index", "World food price index"), ("world_brent_usd_bbl", "Brent, USD/bbl")]:
                s = w[c].dropna()
                if len(s):
                    fig.add_scatter(x=s.index, y=100 * s / s.iloc[0], name=n)
            fig.update_layout(title=f"World prices ({start} = 100)")
            show(fig_layout(fig))
        with st.expander("Caveats for this tab"):
            st.markdown(
                "- **CPI basis:** IMF NCPI to 2023-11, official DCS CCPI from 2023-12 to 2025-10, then IMF growth chained onto "
                "CCPI. The raw IMF index drops 6.5% in Sep-2024 when its basis switches; the cleaned series removes that.\n"
                "- **YoY flag:** y/y rates for Sep-2024 → Aug-2025 compare months on different index bases "
                "(`cpi_break_flag`), so treat them as approximate.\n"
                "- **Real-time vs revised:** the latest IMF months can be revised.\n"
                "- **Relationships (Phase 3):** FX leads inflation by 1–2 months; Brent does not pass through to CPI directly.")

# ================================================================ 2. COMMODITIES
with tabs[1]:
    c = continuous()
    if c.empty:
        st.warning("commodity_prices_continuous_monthly.csv not found.")
    else:
        names = sorted(c["commodity"].unique())
        cc = st.columns([2, 1, 1])
        item = cc[0].selectbox("Commodity", names, index=names.index("Cabbage") if "Cabbage" in names else 0,
                               key="com_item")
        real = cc[1].toggle("Real prices (CPI-deflated)", value=False)
        log_y = cc[2].toggle("Log scale", value=False)
        g = c[c["commodity"] == item].sort_index().set_index("date")
        ag = agreement()
        a = ag[ag["commodity"] == item]
        b = [("Source: WFP national median" + (" → CBSL (level-adjusted)" if (g["source"] != "wfp").any() else ""), COL["info"])]
        if len(a):
            r = a.iloc[0]
            b.append((f"CBSL match: {r['cbsl_item']} ({r['match_confidence']})",
                      COL["ok"] if r["match_confidence"] == "high" else COL["warn"]))
            b.append(("Extended with CBSL: " + r["extend_reason"].replace("_", " ") if r["extend_with_cbsl"]
                      else "Not extended (WFP ends 2025-09)", COL["ok"] if r["extend_with_cbsl"] else COL["warn"]))
            if pd.notna(r.get("mape_level_adjusted_pct")):
                b.append((f"WFP vs CBSL MAPE {r['mape_level_adjusted_pct']:.0f}%", COL["unexplained"]))
        if g["price_lkr"].isna().sum() or len(pd.date_range(g.index.min(), g.index.max(), freq="MS")) > len(g):
            b.append(("Gaps kept missing", COL["warn"]))
        if (g["source"] != "wfp").any() and g.index.max() >= pd.Timestamp("2026-11-01"):
            b.append(("Latest month may be partial (CBSL to report date)", COL["warn"]))
        fl = p5()["flagged"]
        nfl = fl[fl["series"] == item] if not fl.empty else fl
        if len(nfl):
            b.append((f"{len(nfl)} anomaly months (Phase 5)", COL["shock"]))
        badges(*b)

        s = g["price_lkr"].asfreq("MS")
        y, ylab = s, "LKR"
        if real:
            mm = macro().set_index("date")["cpi_headline_index"]
            defl = mm.reindex(s.index).ffill()
            y, ylab = 100 * s / defl * defl.dropna().iloc[-1] / 100, "LKR at latest CPI prices"
        src = g["source"].reindex(s.index)
        fig = go.Figure()
        for sname, color in [("wfp", "#1f77b4"), ("cbsl_scaled", "#ff7f0e")]:
            yy = y.where(src == sname)
            if yy.notna().any():
                fig.add_scatter(x=yy.index, y=yy, name={"wfp": "WFP", "cbsl_scaled": "CBSL (level-adjusted)"}[sname],
                                line=dict(color=color), connectgaps=False)
        if len(nfl):
            fig.add_scatter(x=nfl["date"], y=y.reindex(nfl["date"]), mode="markers", name="Anomaly (Phase 5)",
                            marker=dict(color=[CLS_COLOR.get(k, "grey") for k in nfl["classification"]], size=9,
                                        symbol="circle-open", line=dict(width=2)),
                            text=[CLS_LABEL.get(k, k) for k in nfl["classification"]], hovertemplate="%{text}")
        fig.update_layout(title=f"{item}, {ylab}/unit", yaxis_type="log" if log_y else "linear")
        show(fig_layout(fig, 420))

        c1, c2 = st.columns(2)
        with c1:
            prof = D.seasonal_profile(s)
            if len(prof):
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                fig = go.Figure(go.Bar(x=[months[i - 1] for i in prof.index], y=prof.values,
                                       marker_color=np.where(prof.values > 0, "#d62728", "#1f77b4")))
                fig.add_vrect(x0=0.5, x1=2.5, fillcolor="green", opacity=0.07, line_width=0,
                              annotation_text="Maha harvest", annotation_position="top left")
                fig.add_vrect(x0=6.5, x1=8.5, fillcolor="green", opacity=0.07, line_width=0,
                              annotation_text="Yala harvest", annotation_position="top left")
                fig.update_layout(title="Typical seasonal pattern, % vs 13-month centred mean", showlegend=False)
                show(fig_layout(fig, 340))
            else:
                st.info("Too little history for a seasonal profile.")
        with c2:
            cbi = D.CBSL_ITEM.get(item)
            mg = D.margins(cbsl(), cbi) if cbi else pd.DataFrame()
            if len(mg) and "markup_pct" in mg:
                fig = go.Figure()
                for col in [k for k in mg.columns if k.startswith("Wholesale")] + ["Retail (mean)"]:
                    fig.add_scatter(x=mg.index, y=mg[col], name=col)
                fig.add_scatter(x=mg.index, y=mg["markup_pct"], name="Retail markup %", yaxis="y2",
                                line=dict(dash="dot", color="black"))
                fig.update_layout(title=f"Wholesale vs retail – CBSL '{cbi}'",
                                  yaxis2=dict(overlaying="y", side="right", title="markup %", showgrid=False))
                show(fig_layout(fig, 340))
                st.caption(f"Median retail markup over wholesale: **{mg['markup_pct'].median():.0f}%**. "
                           "Retail prices are sticky: markups shrink when wholesale spikes and widen when it falls (Phase 2).")
            else:
                st.info("No CBSL wholesale/retail pair for this item.")
        with st.expander("Fuel prices (administered)"):
            fu = fuel()
            if len(fu):
                prods = sorted(fu["product"].unique())
                sel = st.multiselect("Products", prods, default=[p for p in ["auto_diesel", "petrol_92", "kerosene"] if p in prods])
                ff = fu[fu["product"].isin(sel) & (fu["date"] >= "2022-01-01")]
                fig = go.Figure()
                for p, gg in ff.groupby("product"):
                    gg = gg.set_index("date")["price_lkr"].asfreq("MS")
                    fig.add_scatter(x=gg.index, y=gg, name=p, mode="lines+markers", connectgaps=False)
                fl2 = ff[ff["flag"] == "possible_missing_revision"]
                if len(fl2):
                    fig.add_scatter(x=fl2["date"], y=fl2["price_lkr"], mode="markers", name="possible missing revision",
                                    marker=dict(symbol="x", color="red", size=8))
                fig.update_layout(title="Retail fuel, LKR per litre (LPG per 12.5 kg)")
                show(fig_layout(fig, 340))
                st.caption("Red ✕ = month where a gazetted revision may be missing from the log; "
                           "these months are excluded from all models.")

# ================================================================ 3. REGIONAL MAP
with tabs[2]:
    mk = markets()
    if mk.empty:
        st.warning("commodity_prices_market_monthly.csv not found.")
    else:
        mk = mk[mk["pricetype"] == "Retail"]
        st.caption("WFP market prices (43 markets, 25 districts). WFP collection ends 2025-09; "
                   "CBSL reports cover Colombo-area and Dambulla markets only, so the map stops at 2025-09.")
        cols = st.columns([2, 2, 1])
        com = sorted(mk["commodity"].unique())
        item = cols[0].selectbox("Commodity", com, index=com.index("Rice (white)") if "Rice (white)" in com else 0, key="map_item")
        g = mk[mk["commodity"] == item]
        months_avail = sorted(g["date"].dt.to_period("M").unique())
        period = cols[1].select_slider("Month", options=[str(p) for p in months_avail], value=str(months_avail[-1]))
        win = cols[2].selectbox("Average over", [1, 3, 6, 12], index=2, format_func=lambda v: f"{v} month(s)")
        end = pd.Period(period).to_timestamp()
        sel = g[(g["date"] > end - pd.DateOffset(months=win)) & (g["date"] <= end)]
        agg = (sel.groupby(["market", "district", "province", "latitude", "longitude"], as_index=False)
                  .agg(price=("price_lkr", "mean"), months=("date", "nunique")))
        if agg.empty:
            st.info("No market data for this selection.")
        else:
            med = agg["price"].median()
            agg["vs_median_pct"] = 100 * (agg["price"] / med - 1)
            k = st.columns(4)
            k[0].metric("Markets reporting", len(agg))
            k[1].metric("National median", f"{med:,.0f} LKR")
            k[2].metric("Spread (max/min)", f"{agg['price'].max() / agg['price'].min():.2f}×")
            k[3].metric("Coefficient of variation", f"{100 * agg['price'].std() / agg['price'].mean():.1f}%")
            c1, c2 = st.columns([3, 2])
            with c1:
                lim = max(5, float(agg["vs_median_pct"].abs().quantile(0.95)))
                fig = px.scatter_mapbox(agg, lat="latitude", lon="longitude", color="vs_median_pct", size="price",
                                        hover_name="market", hover_data={"district": True, "price": ":.0f",
                                                                         "vs_median_pct": ":+.1f", "latitude": False,
                                                                         "longitude": False, "months": True},
                                        color_continuous_scale="RdBu_r", range_color=(-lim, lim), size_max=18,
                                        zoom=6.3, center={"lat": 7.85, "lon": 80.7}, mapbox_style="open-street-map",
                                        labels={"vs_median_pct": "% vs median"})
                fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
                show(fig)
            with c2:
                prov = (agg.groupby("province")["price"].mean() / med * 100 - 100).sort_values()
                fig = go.Figure(go.Bar(x=prov.values, y=prov.index, orientation="h",
                                       marker_color=np.where(prov.values > 0, "#d62728", "#1f77b4")))
                fig.update_layout(title="Province average, % vs national median")
                show(fig_layout(fig, 300, legend_top=False))
                st.dataframe(agg.sort_values("price", ascending=False)[["market", "district", "price", "vs_median_pct"]]
                             .rename(columns={"price": "LKR", "vs_median_pct": "% vs median"}).round(1),
                             hide_index=True, use_container_width=True, height=240)
            disp = (g.groupby("date")["price_lkr"].agg(["min", "median", "max", "count"]))
            disp = disp[disp["count"] >= 5]
            if len(disp):
                fig = go.Figure()
                fig.add_scatter(x=disp.index, y=disp["max"], line=dict(width=0), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=disp.index, y=disp["min"], fill="tonexty", line=dict(width=0), name="min–max across markets",
                                fillcolor="rgba(31,119,180,0.18)")
                fig.add_scatter(x=disp.index, y=disp["median"], name="median", line=dict(color="#1f77b4"))
                fig.add_scatter(x=disp.index, y=100 * (disp["max"] / disp["min"] - 1), name="max/min spread %",
                                yaxis="y2", line=dict(dash="dot", color="black"))
                fig.update_layout(title=f"{item}: price dispersion across markets (months with ≥ 5 markets)",
                                  yaxis2=dict(overlaying="y", side="right", showgrid=False, title="spread %"))
                show(fig_layout(fig, 340))
        st.caption("Phase 2: Western Province is the most expensive region; Jaffna and up-country producing areas the cheapest. "
                   "Rice prices are nationally integrated; vegetables vary much more between markets.")

# ================================================================ 4. FORECASTS
with tabs[3]:
    P4 = p4()
    sel4 = P4["selected"]
    if sel4.empty:
        st.warning("phase4_selected_forecasts.csv not found – run scripts/09_forecast_backtest.py.")
    else:
        hist = history()
        names = list(sel4["series"].unique())
        c = st.columns([2, 1, 1])
        name = c[0].selectbox("Series", names, index=names.index("CPI (cleaned)") if "CPI (cleaned)" in names else 0,
                             key="fc_series")
        show_all = c[1].toggle("Show every model", value=False)
        band95 = c[2].toggle("Show 95% band", value=False)
        f = sel4[sel4["series"] == name].sort_values("horizon")
        h = hist.get(name, pd.Series(dtype=float))
        h = h[h.index >= h.index.max() - pd.DateOffset(months=30)] if len(h) else h
        fig = go.Figure()
        if len(h):
            fig.add_scatter(x=h.index, y=h.values, name="Actual", line=dict(color="#333"))
        if band95:
            fig.add_scatter(x=f["target_date"], y=f["emp_hi95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=f["target_date"], y=f["emp_lo95"], fill="tonexty", line=dict(width=0),
                            fillcolor="rgba(214,39,40,0.08)", name="95% band")
        fig.add_scatter(x=f["target_date"], y=f["emp_hi80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f["target_date"], y=f["emp_lo80"], fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(214,39,40,0.18)", name="80% band")
        fig.add_scatter(x=f["target_date"], y=f["forecast"], mode="lines", line=dict(color="#d62728", width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip")
        MCOL = {"naive": "#d62728", "seasonal_naive": "#ff7f0e", "sarima": "#9467bd", "sarimax": "#8c564b",
                "lightgbm": "#2ca02c"}
        for mdl, g in f.groupby("model", sort=False):
            fig.add_scatter(x=g["target_date"], y=g["forecast"], mode="markers", name=f"Forecast: {mdl}",
                            marker=dict(color=MCOL.get(mdl, "#d62728"), size=10, line=dict(width=1, color="white")),
                            customdata=g["horizon"], hovertemplate="h%{customdata} " + mdl + ": %{y:,.1f}<extra></extra>")
        if show_all and not P4["all"].empty:
            al = P4["all"][P4["all"]["series"] == name]
            for mdl, g in al.groupby("model"):
                fig.add_scatter(x=g["target_date"], y=g["forecast"], mode="lines", name=mdl, line=dict(dash="dot", width=1))
        fig.update_layout(title=f"{name}: 6-month forecast with empirical bands")
        show(fig_layout(fig, 440))
        models_used = f["model"].unique()
        b = [("Bands from routine backtest errors (understate crisis risk)", COL["warn"])]
        if len(models_used) > 1:
            b.insert(0, (f"Model switches across horizons: {' → '.join(dict.fromkeys(f['model']))}", COL["possible"]))
        best = P4["best"]
        if len(best) and name in set(best["series"]):
            r = best[best["series"] == name].iloc[0]
            b.append((f"Avg MAE vs naive (h1–6): {r['vs_naive']:.2f}", COL["ok"] if r["vs_naive"] < 0.97 else COL["unexplained"]))
        badges(*b)
        c1, c2 = st.columns([3, 2])
        with c1:
            sc = P4["scores"]
            if len(sc):
                s = sc[(sc["series"] == name)]
                piv = s.pivot_table(index="model", columns=["regime", "horizon"], values="MAPE_%")
                st.markdown("**Backtest MAPE % by horizon – routine months vs 2022 crisis stress test**")
                st.dataframe(piv.round(1), use_container_width=True)
        with c2:
            st.dataframe(f[["target_date", "horizon", "model", "forecast", "emp_lo80", "emp_hi80"]]
                         .assign(target_date=f["target_date"].dt.strftime("%b %Y"))
                         .rename(columns={"emp_lo80": "80% low", "emp_hi80": "80% high"}).round(1),
                         hide_index=True, use_container_width=True)
        with st.expander("How to read these forecasts"):
            st.markdown(
                "- **Model per horizon:** the point at each horizon comes from the model with the lowest routine backtest error at "
                "that horizon, so the path can *switch* models (e.g. CPI: LightGBM to h3, naive from h4). A step between "
                "horizons is a model switch, not a predicted change.\n"
                "- **Naive is hard to beat:** for FX, rice, coconut and imported onion the best forecast is 'no change'.\n"
                "- **Crises:** in the 2022 stress test, errors were roughly **3–12×** routine levels. The 80% bands come from "
                "normal-times errors and would be far too narrow in a crisis.\n"
                "- **SARIMAX drivers** are held at their last observed value beyond the information date.\n"
                "- Forecasts are refreshed weekly by the GitHub Action, not re-trained in the app.")

# ================================================================ 5. ANOMALIES
with tabs[4]:
    P5 = p5()
    sc5, fl5, ep5 = P5["scores"], P5["flagged"], P5["episodes"]
    if sc5.empty:
        st.warning("phase5 tables not found – run scripts/11_detect_anomalies.py.")
    else:
        latest = sc5.groupby("series")["date"].max()
        recent = fl5.merge(latest.rename("last"), left_on="series", right_index=True)
        recent = recent[recent["date"] >= recent["last"] - pd.DateOffset(months=2)]
        st.subheader("Latest alerts (last 3 observed months of each series)")
        if recent.empty:
            st.success("No series is flagged in its latest three months.")
        else:
            cols = st.columns(min(4, len(recent)))
            for i, (_, r) in enumerate(recent.sort_values("date", ascending=False).head(8).iterrows()):
                with cols[i % len(cols)]:
                    st.markdown(badge(CLS_LABEL.get(r["classification"], r["classification"]),
                                      CLS_COLOR.get(r["classification"], "grey")), unsafe_allow_html=True)
                    move = f"{r['r1']:+.0f}% m/m" if pd.notna(r["r1"]) else ""
                    st.markdown(f"**{r['series']}** – {r['date']:%b %Y}  \n{move} · STL z {r['stl_z']:+.1f}"
                                + (" · IF" if r["if_flag"] else "") + (" · ⚠ series edge" if r["edge"] else "")
                                + (f"  \n_{r['reason']}_" if isinstance(r["reason"], str) and r["reason"] else ""))
        st.divider()
        c = st.columns([2, 2])
        series = sorted(sc5["series"].unique())
        name = c[0].selectbox("Series", series, index=series.index("CPI (cleaned)") if "CPI (cleaned)" in series else 0, key="an_series")
        classes = c[1].multiselect("Classification", list(CLS_LABEL), default=list(CLS_LABEL),
                                   format_func=lambda k: CLS_LABEL[k])
        d = sc5[sc5["series"] == name].sort_values("date")
        ff = fl5[(fl5["series"] == name) & fl5["classification"].isin(classes)]
        fig = go.Figure()
        fig.add_scatter(x=d["date"], y=d["price"], name=name, line=dict(color="#555"))
        for k, g in ff.groupby("classification"):
            fig.add_scatter(x=g["date"], y=g["price"], mode="markers", name=CLS_LABEL.get(k, k),
                            marker=dict(color=CLS_COLOR.get(k, "grey"), size=9),
                            text=g["reason"].fillna(""), hovertemplate="%{x|%b %Y}: %{y:,.1f}<br>%{text}")
        seg = P5["segments"]
        for _, s in seg[(seg["series"] == name) & seg["is_break"]].iterrows():
            fig.add_vline(x=s["start"], line_dash="dot", line_color="#d62728", opacity=0.6)
        fig.update_layout(title=f"{name}: flagged months and change points (dotted lines)")
        show(fig_layout(fig, 400))
        sg = seg[seg["series"] == name]
        if len(sg):
            unit = "mean monthly change, %" if sg["signal"].iloc[0] == "dlog" else "mean log level ×100"
            st.markdown(f"**Regimes (PELT change points, {unit})**")
            st.dataframe(sg.assign(start=sg["start"].dt.strftime("%Y-%m"), end=sg["end"].dt.strftime("%Y-%m"))
                         [["start", "end", "n_obs", "mean", "sd", "missing_months_before"]].round(2),
                         hide_index=True, use_container_width=True)
        st.markdown("**All anomaly episodes**")
        e = ep5[ep5["classification"].isin(classes)].copy()
        grp = st.multiselect("Groups", sorted(e["group"].unique()), default=sorted(e["group"].unique()))
        e = e[e["group"].isin(grp)].sort_values("start", ascending=False)
        st.dataframe(e.assign(start=e["start"].dt.strftime("%Y-%m"), end=e["end"].dt.strftime("%Y-%m"),
                              classification=e["classification"].map(CLS_LABEL))
                     [["series", "start", "end", "months_flagged", "peak_stl_z", "if_months", "classification", "reason",
                       "edge_of_series"]].round(1), hide_index=True, use_container_width=True, height=320)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Known events and data quirks**")
            st.dataframe(P5["events"][["name", "start", "end", "kind", "source", "note"]], hide_index=True,
                         use_container_width=True, height=300)
        with c2:
            v = P5["validation"]
            if len(v):
                tab = v.groupby("event")["outcome"].value_counts().unstack(fill_value=0)
                st.markdown("**Validation: did the detectors pick up each event?**")
                st.dataframe(tab, use_container_width=True, height=300)
        with st.expander("Methods and caveats"):
            st.markdown(
                "- **STL residual z** (|z| ≥ 3.5): retrospective, two-sided; least reliable in a series' last 3 months (⚠ edge).\n"
                "- **Isolation Forest (IF):** causal – refitted each January on earlier data only; the 2020+ commodities are scored from 2024.\n"
                "- **Change points:** PELT on monthly changes (normal cost) or fuel price levels (l2 cost).\n"
                "- **Artifacts:** the Sep-2024 CPI index switch (raw IMF series only), the Feb-2023 WFP cabbage price, and "
                "the Aug-2017 WFP diesel/petrol swap are data problems, not price shocks.\n"
                "- **Known blind spot:** shocks that coincide with the seasonal peak (Cyclone Ditwah, Dec-2025) are under-detected.")

# ================================================================ 6. RELATIONSHIPS
with tabs[5]:
    P3 = p3()
    pt = P3["passthrough"]
    if pt.empty:
        st.warning("phase3 tables not found.")
    else:
        st.markdown("**Pass-through elasticities** – cumulative % change in the local price per 1% change in the driver "
                    "(distributed-lag regressions with HAC standard errors).")
        groups = st.multiselect("Groups", list(pt["group"].unique()), default=list(pt["group"].unique()))
        q = pt[pt["group"].isin(groups)].copy()
        q["lo"], q["hi"] = q["elasticity"] - 1.96 * q["se"], q["elasticity"] + 1.96 * q["se"]
        q["sig"] = np.where(q["p"] < 0.05, "p < 0.05", np.where(q["p"] < 0.10, "p < 0.10", "not significant"))
        q = q.sort_values("elasticity")
        fig = px.scatter(q, x="elasticity", y="label", color="sig", error_x=q["hi"] - q["elasticity"],
                         error_x_minus=q["elasticity"] - q["lo"],
                         color_discrete_map={"p < 0.05": "#2ca02c", "p < 0.10": "#e6a700", "not significant": "#9e9e9e"},
                         hover_data={"n": True, "sample": True, "horizon": True, "p": ":.3f"})
        fig.add_vline(x=0, line_color="grey"); fig.add_vline(x=1, line_color="grey", line_dash="dot")
        fig.update_layout(height=120 + 32 * len(q), margin=dict(l=10, r=10, t=30, b=10), yaxis_title="",
                          legend=dict(orientation="h", y=1.04))
        show(fig)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**VAR impulse responses (cumulative, 12 months)**")
            st.dataframe(P3["irf"].round(3), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Granger causality (VAR, 5% level)**")
            gv = P3["granger_var"]
            st.dataframe(gv[gv["significant_5%"]][["model", "cause", "effect", "F", "p"]].round(4),
                         hide_index=True, use_container_width=True)
            st.markdown("**Granger causality – recent CBSL items (lags 1–3)**")
            st.dataframe(P3["granger_recent"].round(3), hide_index=True, use_container_width=True)
        st.markdown(
            "**Key lags and insights**\n"
            "- FX leads CPI inflation by **1–2 months**; pass-through roughly doubled after the 2022 float (**0.27 → 0.58** within 6 months).\n"
            "- World food prices pass through to CPI (**≈0.29**); Brent does **not** reach CPI directly – it reaches consumers "
            "through retail fuel revisions (diesel ← Brent in LKR **0.41**, about one month later).\n"
            "- Wheat flour and sugar follow world prices; red dhal and imported onion follow the exchange rate; rice and "
            "coconut are insulated from both.")
        with st.expander("Caveats"):
            st.markdown(
                "- **Granger ≠ causation:** it tests whether past values help predict, not structural causality.\n"
                "- **Structural breaks:** the 2022 float changes the FX regime; estimates pooled across it are averages. "
                "The post-float interaction captures the main shift.\n"
                "- **Short samples:** recent CBSL items have ~45 months, so their elasticities have wide intervals.\n"
                "- **Orthogonalised IRFs** depend on the variable ordering; growth rates are 100×log differences.\n"
                "- **Fuel:** months flagged `possible_missing_revision` are excluded.")

st.caption("LankaEcon · built with Streamlit · data refreshed weekly · not financial advice")
