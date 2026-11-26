# LankaEcon – Sri Lanka Economic Intelligence & Price Forecasting System

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline

Run the scripts in order from the project folder:

```bash
python3 scripts/01_download_worldbank.py        # annual macro indicators
python3 scripts/02_download_imf.py              # monthly CPI, interest rates, FX, world commodity prices
python3 scripts/03_scrape_cbsl_daily_prices.py 2023-01-01   # CBSL daily market prices (slow, cached)
python3 scripts/04_download_fred.py             # LKR/USD monthly + daily
python3 scripts/06_download_dcs_ccpi.py         # official DCS CCPI + Colombo retail/fuel prices (slow, cached)
python3 scripts/05_build_datasets.py            # clean + merge -> data/processed/ (applies the CPI fix)
python3 scripts/07_compare_wfp_cbsl.py          # WFP vs CBSL agreement + continuous price series
python3 scripts/08_build_fuel_prices.py         # monthly fuel price history
```

`06` must run before `05`, and `07`/`08` after it. Shared helpers are in `scripts/series_utils.py` (item mapping, `continuous_price()`) and `scripts/cpi_fix.py` (CPI reconciliation). See [PHASE1_NOTES.md](PHASE1_NOTES.md) for the Phase 1 data-integrity findings.

The WFP files were downloaded directly from HDX:
`wfp_food_prices_lka.csv`, `wfp_markets_lka.csv`
(https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka).

## Analysis

| Notebook | Contents | Notes |
|---|---|---|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Inflation & FX, commodity trends, Maha/Yala seasonality, 2022 crisis, regional comparison across 38 markets, wholesale→retail margins | [PHASE2_NOTES.md](PHASE2_NOTES.md) |

Figures are saved to `reports/figures/`. Early findings:
- **Inflation** peaked at 72.5% (Sep-2022), bottomed at −4.2% (Feb-2025) and was 8.0% by Aug-2026. **LKR/USD** is 331, 9% off its Jan-2023 peak of 364.
- **Rice** is stable and nationally integrated. **Vegetables** are highly seasonal (Dec–Jan peak, Sep–Oct trough) and volatile. **Coconut** is still +40% in real terms vs 2023.
- **Western Province** is the most expensive region (Colombo City +7.7%); **Jaffna** (−9.7%) and up-country producing areas are the cheapest.
- **Dambulla wholesale → Colombo retail** markups are ~70% for vegetables, and retail prices are sticky when wholesale prices fall.

## Processed datasets (`data/processed/`)

| File | Grain | Coverage | Use for |
|---|---|---|---|
| `commodity_prices_market_monthly.csv` | market × commodity × month | 2004-01 → 2025-09, 43 markets in 9 provinces / 25 districts, 42 commodities | Regional comparison, forecasting, anomalies |
| `commodity_prices_national_monthly.csv` | commodity × month (median/mean/min/max across markets) | 2004-01 → 2025-09 | National trends, forecasting |
| `cbsl_daily_prices.csv` | item × market × wholesale/retail × day | 2023-01 → 2026-11 | Recent prices, daily anomalies, wholesale–retail margins |
| `cbsl_prices_monthly.csv` | the above averaged to month | 2023-01 → 2026-11 | Extending WFP series past 2025-09 |
| `macro_monthly.csv` | month | 2000-01 → 2026-10 | CPI and inflation, LKR/USD and LKR/EUR, interest rates, world commodity prices |
| `macro_annual.csv` | year | 1990 → 2025 | GDP, agriculture, reserves, remittances, debt, cereal production |
| `commodity_prices_continuous_monthly.csv` | commodity × month | 2006 → 2026-11 | WFP national median extended with level-adjusted CBSL prices (`source` column) |
| `wfp_cbsl_agreement.csv`, `wfp_cbsl_overlap_monthly.csv` | item / item × month | 2023 → 2026 | How well WFP and CBSL prices agree |
| `cpi_reconciliation.csv` | month | 2023-12 → 2025-10 | Raw IMF CPI vs official DCS CCPI |
| `fuel_prices_monthly.csv` | product × month | 2015 → 2026-11 | Petrol, diesel, kerosene, LPG (`source`, `flag` columns) |
| `analysis_panel_monthly.csv` | month | 2004-01 → 2026-10 | **Main modelling table**: key national retail prices + macro + event dummies |
| `coverage_report.csv` | — | — | Row counts and date ranges of each file |

### Key columns in `analysis_panel_monthly.csv`
- Macro: `cpi_headline_index`, `inflation_yoy_pct`, `inflation_mom_pct`, `inflation_source`, `cpi_break_flag`, `ccpi_*` (official CCPI), `cpi_food_index`, `food_inflation_yoy_pct`, `lkr_per_usd`, `lkr_per_eur`, `lkr_usd_mom_pct`, `tbill_rate_pct`, `policy_discount_rate_pct`, `money_market_rate_pct`, `deposit_rate_pct`, `lending_rate_pct`
- World prices (USD): `world_rice_usd_mt`, `world_wheat_usd_mt`, `world_sugar_usd_lb`, `world_brent_usd_bbl`, `world_coconut_oil_usd_mt`, `world_palm_oil_usd_mt`, `world_tea_colombo_usd_kg`, `world_urea_usd_mt`, `world_food_price_index`
- Local retail prices (LKR/kg unless noted, median across WFP markets): `price_rice_white`, `price_rice_red_nadu`, `price_wheat_flour`, `price_sugar`, `price_coconut_per_nut`, `price_lentils`, `price_red_onion_local`, `price_big_onion_imported`, `price_potato_local`, `price_tomato`, `price_carrot`, `price_beans`, `price_egg_each`, `price_chicken`, `price_dried_chilli`, …
- Event dummies: `crisis_2022` (2022-01 → 2023-06), `covid_2020` (2020-03 → 2021-12), `post_float_2022` (from 2022-03)

## Sources

| Source | Raw file(s) | Notes |
|---|---|---|
| WFP VAM via HDX | `wfp_food_prices_lka.csv`, `wfp_markets_lka.csv` | Compiled from DCS, HARTI, FAO and CEYPETCO data |
| IMF SDMX API (`api.imf.org`) | `imf_cpi_lka.csv`, `imf_interest_rates_lka.csv`, `imf_exchange_rates_lka.csv`, `imf_world_commodity_prices.csv` | CPI base 2021=100. `SRP_*` = IMF spliced long series |
| FRED (Fed H.10) | `fred_lkr_usd_monthly.csv`, `fred_lkr_usd_daily.csv` | LKR per USD |
| World Bank WDI API | `worldbank_wdi_lka_annual.csv` | 22 indicators |
| DCS CCPI releases (PDF) | `dcs_ccpi_monthly.csv`, `dcs_ccpi_retail_prices.csv`, `dcs_ccpi_text/` | Official CCPI 2021=100 (Table 03) and Colombo retail prices incl. fuel (Table 05) |
| CPC fuel price revisions | `fuel/cpc_price_revisions.csv` | Hand-maintained; every row has a source and status |
| CBSL Daily Price Reports (PDF) | `cbsl_daily_prices.csv`, `cbsl_daily_text/` (cached text) | Pettah, Dambulla, Marandagahamula (rice wholesale), Narahenpita, Peliyagoda, Negombo |

## Known gaps and caveats

1. **WFP coverage is uneven.** It is dense for 2015–2017, 2020 and 2023–2025 (about 29k of 34k rows are 2023–2025) and sparse otherwise. For long-horizon forecasting use `price_rice_white` and `price_wheat_flour` (about 220 months each). Other commodities have about 50 months.
2. **Fuel prices**: `fuel_prices_monthly.csv` uses DCS Colombo averages for 2023-01 → 2026-04 and the CPC revision log after that. Six revision entries are still `todo_manual`, and months that span a missing revision are flagged `possible_missing_revision`. See PHASE1_NOTES.md.
3. **Interest rates** from the IMF stop in 2020-05. Recent policy rates, AWPR and T-bill yields have to be taken from the CBSL website / Economic Data Library by hand. The CBSL HTML pages block automated requests. The PDFs do not.
4. **CPI break in Sep 2024 – fixed.** The IMF series switches from NCPI to CCPI in 2024-09, which created a fake −6.5% month. Headline CPI and inflation are now rebuilt from official DCS CCPI (2023-12 → 2025-10) and chain-linked, and the raw IMF values are kept as `imf_*_raw`. Before 2023-12 the rates are NCPI-based (`inflation_source = imf_ncpi`). See PHASE1_NOTES.md.
5. **Food CPI**: official CCPI food index for 2023-12 → 2025-10, and IMF COICOP 01 for 2014-01 → 2024-08.
6. **CBSL scraper**: 818 daily reports were parsed into 125,853 rows across 41 items (2023-01-02 → 2026-11-25). There are fewer reports for 2023 (174) and 2024 (191) than for 2025–26 (about 230 each). The missing days are holidays or reports that weren't published under the usual filename. Rows with 3 or 2 price pairs (sugar, dhal, coconut oil, eggs, dried fish, imported fruit) use an inferred market layout. See `layout_pairs` and treat them as lower confidence. Imported-rice rows with a missing market are skipped.
7. **Paddy and agricultural production by district/season** is not yet included. Source: DCS Paddy Statistics (statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics). Annual national cereal production is in `macro_annual.csv`.
8. **World Bank Real-Time Food Prices** (43 markets to 2026-08, microdata.worldbank.org/catalog/6169) needs a free account to download, so it is not included.
