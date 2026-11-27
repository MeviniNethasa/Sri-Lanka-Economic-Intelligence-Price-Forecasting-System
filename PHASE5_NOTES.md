# Phase 5 – Anomaly Detection

Branch: `feat/phase5-anomalies`, created from `main` at `487d251` (the merge of Phase 4, PR #4).

## What was added

| Path | What it is |
|---|---|
| `src/anomalies.py` | Series specs, STL residual robust z-scores, causal Isolation Forest, PELT change points (`ruptures` or built-in), known-event catalogue, classification, event validation, episodes, seasonal profiles |
| `scripts/11_detect_anomalies.py` | Runs everything (~20 s) and writes `reports/tables/phase5_*.csv` |
| `notebooks/04_anomalies.ipynb` | Executed notebook: coverage, overview, CPI artifact, WFP/CBSL cross-check, change points, seasonality, Ditwah and Nov-2026 vegetables, 2026 oil shock, event validation, Isolation Forest scores |
| `reports/figures/24–32_*.png` | Anomaly timeline, CPI index switch, WFP vs CBSL, change points, vegetable seasonality, 2025–26 vegetables, 2026 oil shock, event validation, Isolation Forest scores |
| `reports/tables/phase5_*.csv` | Scores for every series-month, flagged months with classification, episodes, change-point segments, event catalogue, event validation, series specs, seasonal profile |

Rerun:

```bash
python3 scripts/11_detect_anomalies.py
jupyter nbconvert --to notebook --execute --inplace notebooks/04_anomalies.ipynb
```

**`ruptures` is not installed** in this environment, and no packages were installed for this phase. `src/anomalies.py` uses `ruptures.Pelt` automatically when it is available. Otherwise it runs a built-in PELT with the same costs (`normal`, `l2`) and the same breakpoint convention. The notebook checks that the built-in version matches an exact, unpruned optimal-partitioning solution on 80 random signals, with 0 mismatches. `ruptures` is listed in `requirements.txt` as optional.

## 1. Series (19)

| Group | Series | Coverage | Residual method | Isolation Forest model |
|---|---|---|---|---|
| macro | CPI (cleaned), **CPI (raw IMF, diagnostic)**, LKR/USD, Brent | 2000-01 → 2026-08/10 | STL (CPI), rolling median (FX, Brent) | own model each (raw CPI is scored by the cleaned-CPI model, never trained on) |
| staple | White rice, wheat flour (to 2022-06) | 2004/06 → | STL (rice), rolling median | own model |
| staple | Coconut, lentils, imported onion, CBSL sugar | 2020/2023 → 2026-11 | STL (coconut), rolling median | pooled `recent_commodity` |
| vegetable | Beans, cabbage, carrots, tomatoes, local potato | 2020-01 → 2026-11 | STL | pooled `recent_commodity` |
| fuel | Diesel, petrol 92, kerosene, LPG | 2015–18 WFP, 2023+ DCS/CPC | revision size | too sparse; not scored |

- **Missing months** stay missing. STL fits over internal gaps, but interpolated months are never scored. The Isolation Forest drops incomplete feature rows; how many each series loses is in `phase5_series_specs.csv`.
- **Fuel months** flagged `possible_missing_revision` are dropped. The WFP→CBSL splice month (2025-10) is tracked.

## 2. Methods
- **STL residual z.**
  - Decomposition: STL (period 12, robust) on 100×log price for seasonal series with ≥ 36 observed months; a centred 7-month rolling median otherwise.
  - For administered fuel prices, the score is the month-on-month revision itself.
  - Robust z = (residual − median) / (1.4826·MAD), with a per-series floor on the scale. A month is flagged at **|z| ≥ 3.5**.
  - The smoother is **two-sided**, so this detector is retrospective. Its results are least reliable in a series' first and last 3 months (`edge` column).
- **Isolation Forest (causal).**
  - Features: 1- and 3-month log change, last month's change, deviation from the trailing 12-month median, and the change divided by trailing 6-month volatility. All use data ≤ t.
  - The model is refitted each January on rows dated before that January. A month is flagged if its score exceeds the **97.5th percentile of the training scores**.
  - A model needs ≥ 36 training rows, so the pooled 2020+ model scores **2024 onwards** only (WFP gaps in 2021–22).
- **Change points.**
  - PELT with the BIC-style penalty 3·log(n).
  - "normal" cost (shifts in mean *and* variance) on monthly log changes for market series; "l2" cost on the log level for fuel. The fuel penalty is scaled by a robust noise variance.
  - Minimum segment length: 12 months for long series, 6 for short. Breaks that follow a data gap are marked (`missing_months_before`).
- **Classification of each flagged month**, first match wins:
  1. **known data artifact:** in the event catalogue;
  2. **splice month:** the first month of a new source;
  3. **cross-source disagreement:** while the primary series is WFP, the CBSL series moves the other way or by less than a third as much (|move| ≥ 15);
  4. **isolated spike-and-revert** (≥ 30 then reversed by ≥ 2/3) in a normally smooth macro or staple series;
  5. **known economic, policy or weather event** (±1 month);
  6. **broad co-movement:** ≥ 25% of ≥ 8 observed series flagged, and at least 4;
  7. **unexplained.**

  Flagged months are then merged into **episodes**. Each episode takes the class of its peak month.

## 3. Results

| | Count |
|---|---|
| Series-months scored | 2,461 |
| Flagged months (STL or IF) | 282 |
| Episodes | 107: 60 economic shock, 36 unexplained, 5 possible data artifact, 4 data artifact, 2 seasonal/unverified |
| Event validation, expected-to-detect pairs | **56 of 71 detected**; index-switch absence check passed |

### Real economic shocks (detected)
- **2022 crisis (15 of 19 series).**
  - CPI is flagged by the Isolation Forest from **Dec-2021**, in real time, through Sep-2022. STL z peaks at 18.6 in Aug-2022.
  - The CPI change point puts the crisis regime at **2021-10 → 2022-09**, averaging +4.5% a month (sd 3.1), against 0.4% in 2008–21 and 0.2% since 2022-10.
  - **LKR/USD:** a regime break in **2022-03** (the float: +2.7% a month, sd 8.5) and a post-crisis regime from 2023-08. The Isolation Forest flags Mar–Jul 2022 and the 2023 appreciation.
  - **Rice:** regime **2021-05 → 2022-06** (+5.7% a month). It begins with the fertiliser ban, before the crisis proper.
  - **Wheat flour** from 2021-07 (+9.3% a month). **Lentils** peak in Jul-2022.
  - **Missed:** Brent (a world price), tomatoes, kerosene and LPG (z 1.5–3.4).
- **Earlier shocks.**
  - 2008 global food/oil spike: CPI, rice, wheat flour, Brent.
  - 2015 post-election price cuts: CPI.
  - 2018 constitutional crisis: LKR/USD.
  - COVID 2020: FX, Brent, most foods. CPI and rice barely moved.
  - 2024 VAT rise to 18%: CPI, +2.9% in Jan-2024.
  - 2024–25 coconut shortage.
- **2026 oil shock.**
  - Fuel hikes of ~8% (9–10 Mar) and ~25% (21–22 Mar 2026), tied to the Middle East conflict, show up as **April** jumps in monthly averages:
    - diesel: z 13.2, plus a level change point;
    - petrol 92: z 13.0;
    - kerosene: z 4.6;
    - LPG: +14%, z 2.9, just under the threshold.
  - **CPI:** +3.0% in Apr-2026 and +2.1% in Jun-2026. From 2026-03, sugar enters a rising regime (+1.8% a month) and lentils a high-volatility one (sd 6.7). LKR/USD depreciated ~7% from Mar to Jun 2026, too gradually to be flagged.

### Data artifacts (separated from real shocks)
| Episode | Evidence | Status |
|---|---|---|
| Raw IMF CPI, Sep–Dec 2024 | NCPI→CCPI switch: −6.5% in one month, STL z −5.8 and IF flag | **Cleaned CPI is not flagged** (max \|z\| 0.5), which confirms the Phase 1 fix |
| WFP cabbage, Feb-2023 (48 LKR/kg) | WFP −71% (168→48) vs CBSL −23% (132→102); WFP had one market (Peliyagoda) | real dip, exaggerated by a single-market value |
| WFP carrots, Feb–Mar 2023; WFP lentils, imported onion, local potato, May-2023 | CBSL moves the other way or much less | flagged automatically by the cross-source rule |
| **WFP diesel/petrol, Aug-2017 (new finding)** | diesel 117 / petrol 95, exactly the reverse of Jul and Sep (95 / 117) in every WFP market; also produces a spurious diesel change point | swap the labels back or drop the month in a later data fix |
| WFP wheat flour, Dec-2004 (68 LKR/kg) | +54 then −66 (100×log) in an importer-set price | likely a data error |
| WFP→CBSL splice, 2025-10 | **no series flagged** at the splice month | the Phase 2 level adjustment works |

### Seasonal and weather effects
- **Seasonal factors (STL, ≈ % vs trend):**
  - December peak for beans (+37), tomatoes (+41) and cabbage (+17).
  - Oct–Nov trough for cabbage and carrots (−26 to −39).
  - Rice lowest in Apr–Jun, after the Maha harvest.

  Swings of this size are treated as normal and are not flagged.
- **Cyclone Ditwah (landfall 27–28 Nov 2025):**
  - Dec-2025 was the largest December rise on record for beans (+74) and tomatoes (+98).
  - Because it landed on the seasonal December peak, **STL did not flag it** (|z| ≤ 1.9). The Isolation Forest ranked beans and tomatoes at the 95th percentile, just under the 97.5% cut-off.
  - Carrots are flagged in Feb-2026, when prices fell back faster than usual.
  - **This is a known blind spot:** a weather shock that coincides with the seasonal peak is under-detected.
- **Oct–Nov 2026 vegetable spike:**
  - Cabbage (z 6.8 → 9.0) and carrots (6.0 → 9.1) are flagged strongly, because they rose during their usual seasonal **trough**.
  - The cause is **not verified**. These are the last months of the series, where two-sided STL is least reliable, and Nov-2026 covers CBSL reports only to 25 Nov.
  - Classified `seasonal_or_unverified`.

## 4. Caveats
- **Smoother side effect in CPI:** the two-sided STL trend starts rising before the April-2026 CPI jump, which makes Feb–Mar 2026 residuals *negative* (z −3.9, −5.5). These two flags are a side effect of the smoother, not a price fall. For real-time use, rely on the causal Isolation Forest and the change-point regimes.
- **Background events:** some catalogue entries (tsunami, 2008 spike, 2011 floods, 2012 FX, 2015 price cuts, 2016–17 drought, 2018 crisis, fertiliser ban, VAT, coconut shortage) are well-documented events that weren't re-verified here (`source = background`). Ditwah and the 2026 fuel hikes were checked in news and agency reports this phase (`source = cited`).
- **Unexplained episodes:** 36 are left unexplained, mostly one-month moves in Brent, early WFP wheat and rice, and 2024–25 onion and vegetable moves that CBSL confirms in direction. They are not forced onto an event.
- **Pooled Isolation Forest:** it scores the 2020+ commodities only from 2024. Their crisis-period anomalies come from STL and change points.

Sources for the events checked this phase:
- Cyclone Ditwah:
  - [ReliefWeb](https://reliefweb.int/report/sri-lanka/sri-lanka-cyclone-ditwah-urgent-call-assistance)
  - [IPS via Ada Derana Biz](https://bizenglish.adaderana.lk/sri-lankas-food-security-after-cyclone-ditwah-risk-recovery-and-resilience/)
  - [FAO GIEWS update, 16 Dec 2025](https://knowledge4policy.ec.europa.eu/publication/giews-update-democratic-socialist-republic-sri-lanka-16-december-2025_en)
- March 2026 fuel hikes:
  - [Business Today, 10 Mar 2026](https://www.businesstoday.in/world/story/sri-lanka-hikes-fuel-prices-by-8-to-curb-panic-buying-amid-rising-global-oil-prices-519839-2026-03-10)
  - [The Star, 22 Mar 2026](https://www.thestar.com.my/aseanplus/aseanplus-news/2026/03/22/sri-lanka-raises-fuel-prices-by-25-as-conflict-bites)
  - [Ada Derana](https://www.adaderana.lk/news/119537/fuel-prices-increased-to-manage-stocks-and-stabilize-market-officials-say)

## 5. Next (Phase 6 – dashboard)
- **Show `phase5_anomaly_scores.csv`** as an "unusual this month" panel, using the causal Isolation Forest flag for current months.
- **Data fixes:** apply the Aug-2017 WFP fuel swap fix and the Feb-2023 WFP vegetable overrides in the processed data.
- **Ditwah blind spot:** add a seasonal-difference detector (this month's change vs the same month in earlier years).
