# Phase 6 – Interactive Dashboard & Weekly Automation

Branch: `feat/phase6-dashboard`, created from `main` at `f28046c` (the merge of Phase 5, PR #5).

## What was added

| Path | What it is |
|---|---|
| `dashboard/app.py` | Streamlit app with six tabs (below) |
| `dashboard/data.py` | Loaders that only **read** processed data and Phase 3–5 tables (cached with `st.cache_data`, 6 h TTL) |
| `.streamlit/config.toml` | Light theme, headless server, usage stats off |
| `.github/workflows/weekly_refresh.yml` | Weekly refresh: download, rebuild, re-score, check, commit |
| `scripts/12_check_refresh.py` | Sanity checks that gate the weekly commit |
| `requirements.txt` / `requirements-dev.txt` | Runtime (pipeline + app, what Streamlit Cloud installs) vs notebook extras |
| `scripts/03`, `scripts/06` | Fix: "no report" is cached only for periods safely in the past (see §3) |

Run locally:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 1. Dashboard

| Tab | Contents | Caveats shown |
|---|---|---|
| **Macro overview** | KPIs (latest inflation, CPI m/m, LKR/USD, crisis peak); y/y inflation coloured by `inflation_source`; cleaned vs raw IMF CPI toggle; LKR/USD; world food index and Brent | NCPI → DCS CCPI → IMF CCPI basis; `cpi_break_flag` months marked ✕; FX leads CPI; Brent doesn't pass through directly |
| **Commodity explorer** | Any of the 23 continuous WFP/CBSL series: nominal or CPI-deflated, log scale, WFP vs CBSL segments, Phase 5 anomaly markers; seasonal profile with Maha/Yala harvest shading; CBSL wholesale vs retail with median markup; fuel prices | Badges for source and splice, CBSL match confidence, extension rule, WFP-vs-CBSL MAPE, gaps, partial latest month, anomaly count; ✕ for `possible_missing_revision` fuel months |
| **Regional map** | 43 WFP retail markets on an OpenStreetMap basemap (no API key), coloured by % vs national median; 1/3/6/12-month averages; province bars; market table; dispersion band and max/min spread over time | WFP collection ends 2025-09; CBSL covers only Colombo-area and Dambulla markets |
| **Forecasts** | 12 series: actuals, best-model forecast per horizon (coloured by model), 80% and optional 95% empirical bands, optional all-model overlay; backtest MAPE by horizon (routine vs crisis); forecast table | Badge listing the model switches across horizons; "bands understate crisis risk"; naive is hard to beat; crisis errors 3–12×; SARIMAX drivers held flat |
| **Anomaly alerts** | Alert cards for flags in each series' last 3 months; per-series chart with flagged months by class and change-point lines; regime table; all 107 episodes (filterable); event catalogue; validation outcomes | Shock vs artifact labels; series-edge warning; IF is causal but STL is two-sided; the Ditwah blind spot |
| **Relationships** | Pass-through elasticities with 95% CIs, coloured by significance; VAR IRF table; significant VAR Granger pairs; recent CBSL Granger tests; key lags | Granger ≠ causation; the 2022 break; short CBSL samples; IRF ordering |

- **Performance:** every input is a CSV the pipeline has already written. The largest, 6.5 MB of CBSL daily prices, is read only for its date column. No model is fitted in the app, so tab switches are instant once the cache is warm.
- **Testing:**
  - Streamlit's `AppTest` ran the app headlessly with every toggle on, through **all 96 dropdown options** (23 commodities, 12 forecast series, 42 map commodities, 19 anomaly series), with no exceptions.
  - I also checked the Macro, Map, Forecast and Anomaly tabs in a browser at 1440 px.
- **Map rendering:** `px.scatter_mapbox` with the keyless `open-street-map` style. `carto-positron` showed an "API key required" watermark, and the newer `scatter_map` (MapLibre) doesn't render with Streamlit 1.37's bundled plotly.js. `plotly<6` is pinned for that reason.

## 2. Weekly refresh (`.github/workflows/weekly_refresh.yml`)
- **Triggers:**
  - `schedule`: Mondays 02:30 UTC (08:00 in Sri Lanka).
  - `workflow_dispatch`, with inputs `run_forecasts` (re-run the ~5-minute backtest) and `dry_run` (run everything, commit nothing).
- **Steps:**
  1. Install `requirements.txt` on Python 3.12 (pip cache).
  2. **Downloads** (World Bank, IMF, FRED, CBSL daily reports, DCS CCPI). Each runs with `continue-on-error`, so a source that's down keeps last week's committed file. If a failed download left a raw file empty, it's restored from git.
  3. **Build** (`05`, `07`, `08`), **forecasts** (`09`) and **anomalies** (`11`). These must succeed.
  4. **`12_check_refresh.py`** fails the run, before any commit, if a key file is missing or empty, loses more than 2% of its rows, has its latest date move backwards, or any data file exceeds 50 MB.
  5. **Commit** only `data/raw`, `data/processed` and `reports/tables`, and only if something changed: `Weekly data refresh YYYY-MM-DD`, with no attribution trailer. It then rebases on the branch and pushes.
  6. A job summary lists each download's outcome.
- **Not refreshed:**
  - Notebooks are not re-executed, to avoid large output diffs.
  - The WFP/HDX files are static (collection ended 2025-09).
- **Commit identity:** defaults to `MeviniNethasa <mevinimunaweera@gmail.com>`, so all commits stay under your name. To use a bot identity instead, set the repository variables `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL`.
- **Permissions:** `contents: write` via the built-in `GITHUB_TOKEN`. If `main` gets branch protection that blocks direct pushes, switch the last step to open a PR instead.
- **Tested locally:** the build, scoring and check steps (`05`, `07`, `08`, `11`, `12`) reproduce the committed tables byte for byte. The patched scrapers fetched only what was new.
- **Not tested:** the workflow itself hasn't run on GitHub yet. It runs on the next Monday after this branch is merged, or on demand from the Actions tab. Whether CBSL and DCS accept requests from GitHub's runner IPs is unknown; if not, those steps fail softly and the rest of the refresh still runs.

## 3. Bug fixed along the way: permanent "no report" cache
- **Problem:** `03_scrape_cbsl_daily_prices.py` and `06_download_dcs_ccpi.py` cached a "no report" marker (an empty file) **whenever** a request found nothing. That included:
  - reports not yet published: the Nov-2026 CCPI release had been cached as missing on 2026-11-26;
  - temporary blocks or server errors.

  Such periods were then never retried, which would quietly freeze a weekly refresh.
- **Fix:** a negative result is cached only if every request got a clean 404 **and** the period is safely in the past (14 days for daily CBSL reports, 10 days after the end of the search window for DCS releases). Future dates are no longer requested.
- **Cleanup:** I removed the three stale markers (CCPI 2026-11; CBSL 2026-11-18 and 2026-11-24) so they're retried.
- **Check:** a live re-run fetched the new 26 Nov CBSL report and left the still-missing ones uncached. That fetched data was **not** committed on this branch; the weekly Action will commit it.

## 4. Deploying to Streamlit Community Cloud
1. Go to <https://share.streamlit.io>, sign in with GitHub, then **Create app → Deploy a public app from GitHub**.
2. Repository `MeviniNethasa/Sri-Lanka-Economic-Intelligence-Price-Forecasting-System`, branch `main`, main file `dashboard/app.py`.
3. **Advanced settings → Python 3.12**. No secrets are needed.
4. Deploy. Then replace `YOUR-APP-URL` in the README badge and the "Live dashboard" line.

The app redeploys on every push to `main`, including the weekly refresh commits.

## 5. Caveats
- **Coverage report lag:** `coverage_report.csv` is written by `05`, which runs before `08`, so its fuel rows describe the previous build. It was refreshed here (fuel: 195 rows), and the weekly run keeps it at most one week behind.
- **Prices shown are monthly averages:** fuel revisions show up in the month after a late-month hike (e.g. the 21 Mar 2026 increase appears in April).
- **The app shows the committed tables:** forecasts and anomaly flags are only as fresh as the last successful weekly run. The sidebar shows each dataset's last date.
