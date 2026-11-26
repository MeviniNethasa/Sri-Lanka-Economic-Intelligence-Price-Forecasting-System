# Phase 1 – Data Integrity & Series Alignment

Branch: `feat/phase1-cpi-and-prices`

## 1. The September 2024 CPI drop

### What was wrong
`macro_monthly.csv` used the IMF CPI series for Sri Lanka (`IMF.STA:CPI`, all items). It showed:

| Month | IMF index (2021=100) | IMF MoM | IMF YoY | **Official DCS CCPI** | **CCPI YoY** |
|---|---|---|---|---|---|
| 2024-08 | 204.1 | −1.4% | 1.1% | **191.1** | **0.5%** |
| 2024-09 | 190.9 | **−6.5%** | **−6.2%** | **190.9** | **−0.5%** |
| 2025-01 | 192.6 | 0.5% | **−10.4%** | **192.6** | **−4.0%** |

The IMF series is two different indices joined together:
- **Up to 2024-08 it is the NCPI (2021=100).** Its YoY rates match published NCPI inflation (2.5% Jul-2024, 1.1% Aug-2024). Its level is a steady 6.3–7.1% above CCPI (`cpi_reconciliation.csv`, column `index_ratio_imf_to_ccpi`).
- **From 2024-09 it is the CCPI (2021=100).** Every month matches the DCS release exactly (ratio 1.0000).

The jump between the two indices shows up as a fake −6.5% month and makes every YoY rate from 2024-09 to 2025-08 roughly 6 points too negative. The IMF reported −10.4% for Jan-2025; the official figure is −4.0%. Once the base month falls out of the window (2025-09), the IMF and DCS agree again.

### Fix
- `scripts/06_download_dcs_ccpi.py` downloads the DCS CCPI monthly releases (statistics.gov.lk) and parses Table 03. That table gives 13 months of headline, food and non-food index, MoM, YoY and moving-average inflation. Result: official CCPI for **2023-12 → 2025-10** (23 months, all of the affected window). Many other releases are scanned images with no text layer and could not be parsed without OCR.
- `scripts/cpi_fix.py` rebuilds the headline columns, and `05_build_datasets.py` applies it:

| Column | Meaning |
|---|---|
| `cpi_headline_index` | Official CCPI level where DCS data exist. Earlier and later months are chain-linked with IMF month-on-month growth, so the level has **no break** |
| `inflation_yoy_pct`, `inflation_mom_pct` | Official CCPI where available, otherwise IMF |
| `inflation_source` | `dcs_ccpi` (official), `imf_ccpi` (IMF ≥ 2024-09, identical to CCPI), `imf_ncpi` (IMF < 2024-09, NCPI-based) |
| `cpi_break_flag` | 1 for 2024-09 … 2025-08, where the **raw** IMF YoY is distorted |
| `imf_*_raw` | The original IMF numbers, kept for transparency |
| `ccpi_*` | Official CCPI headline / food / non-food index and inflation |

- `data/processed/cpi_reconciliation.csv` shows raw IMF vs official CCPI month by month.
- **Cross-check:** the rebuilt Apr-2026 level (201.6) matches the CCPI figure published on the DCS homepage.

**For analysis:** use `inflation_yoy_pct` and `cpi_headline_index`. When modelling across 2024, keep `inflation_source` in mind: before Dec-2023 the rates come from NCPI, not CCPI.

## 2. WFP vs CBSL commodity prices (2023–2025 overlap)

`scripts/07_compare_wfp_cbsl.py` matches 23 WFP commodities to CBSL daily-report items (`series_utils.ITEM_MAP`, each with a match-confidence label). It compares the WFP national median retail price with the CBSL retail mean across Pettah, Dambulla, Narahenpita and Negombo. It also compares WFP Colombo City with CBSL Narahenpita, since both are Colombo prices.

| Item | Months | WFP/CBSL level | MAPE after level adj. | Corr. of MoM changes | Extended? |
|---|---|---|---|---|---|
| Carrots | 33 | 1.25 | 9.2% | 0.86 | yes |
| Eggplants (brinjal) | 33 | 1.00 | 6.1% | 0.95 | yes |
| Tomatoes | 33 | 1.22 | 9.5% | 0.83 | yes |
| Big onion (imported) | 33 | 1.07 | 5.7% | 0.81 | yes |
| Coconut | 33 | 0.78 | 8.3% | 0.68 | yes |
| Eggs | 33 | 1.06 | 4.2% | 0.77 | yes |
| Rice (white) ↔ Nadu | 28 | 1.10 | 3.9% | 0.06 | yes (levels) |
| Lentils ↔ Red dhal | 33 | 1.02 | 2.3% | −0.10 | yes (levels) |
| Red onion (local) | 31 | 1.23 | 24.9% | 0.35 | **no** |
| Dried fish (katta, sprats) | 22–33 | 0.94 | 5.5–9.1% | 0.05–0.27 | **no** |

Full table: `data/processed/wfp_cbsl_agreement.csv`.

**Findings**
- The sources **agree well**. 13 items have month-on-month changes that track (corr ≥ 0.5, MAPE ≤ 15%), and 5 sticky-price items (rice, dhal, imported potato, dried chilli) have levels within 5%.
- **The level gap is systematic.** WFP's national median is 10–25% above the CBSL mean for vegetables. This is likely because the CBSL mean includes Dambulla, a wholesale hub with lower retail prices. For Colombo-to-Colombo comparisons the ratio is close to 1 (`colombo_median_ratio`).
- WFP's coconut price is ~22% *below* CBSL. This is plausibly because WFP's figure is a national median and coconuts are cheaper outside Colombo. Not verified.
- **Sugar has no overlap.** WFP sugar ends 2022-06. Use the CBSL `Sugar (White)` series directly.

**Output:** `data/processed/commodity_prices_continuous_monthly.csv`. Each item uses the WFP national median up to 2025-09, then the CBSL retail mean × the median WFP/CBSL ratio of the last 12 overlapping months (`source = cbsl_scaled`). That extends 18 of 23 items to 2026-11. Load one item with `series_utils.continuous_price("Coconut")`.

## 3. Fuel prices

What existed before: only WFP diesel and petrol for 2015–2018, in `commodity_prices_market_monthly.csv`.

New structure:
- `data/raw/fuel/cpc_price_revisions.csv` is the **hand-maintained** CPC revision log. Every row has a `source`, `source_url` and `status`:
  - `gazette_text`: figure quoted from the Extraordinary Gazette
  - `news_report` / `cpc_website`: secondary source, worth confirming against the gazette
  - `todo_manual`: still to be filled in
- `scripts/08_build_fuel_prices.py` builds `data/processed/fuel_prices_monthly.csv` from, in order of preference:
  1. DCS CCPI Table 05: Colombo monthly **average** retail petrol, diesel, kerosene and LPG, 2023-01 → 2026-04
  2. The CPC revision log, as a day-weighted step series
  3. WFP 2015–2018
- `fuel_revision_check.csv` compares the two sources. **Apr-2026 matches exactly**: petrol 398, diesel 382, kerosene 255. This confirms the March-2026 surge figures.

2026 revisions in the log:

| Effective | Change | Status |
|---|---|---|
| 2026-03-09 | Petrol 92 293→317, Petrol 95 340→365, Kerosene 182→195 | gazette 2479/27 |
| 2026-03-21 | **Auto diesel 303→382 (+79)**, Super diesel →443, Kerosene 195→255 | gazette 2480/49 + Ada Derana |
| 2026-04-01 | Petrol 92 = 398 | CPC site; DCS Apr-2026 average agrees |
| 2026-05-02 | Auto diesel 382→392 | gazette 2487/06 |
| 2026-11-01 | **Petrol 92 414→424, Petrol 95 475→490, Auto diesel 392→397, Super diesel 528→543, Kerosene 285 (unchanged)** | news reports |

**Still manual (`todo_manual`, 6 rows):**
- Petrol 92 and 95 prices from the 2026-03-21 gazette
- Auto-diesel price on 2026-03-09
- The revision(s) between May and Oct 2026 that moved petrol 92 to 414, petrol 95 to 475, super diesel to 528 and kerosene to 285. The Nov-2026 "previous price" values show these happened, but the dates aren't known yet.

Months where the step series spans a missing revision are marked `flag = possible_missing_revision`. **Don't use those values as real prices.** documents.gov.lk does not serve the gazette PDFs to scripts, so these have to be entered by hand.

## Run order

```bash
python3 scripts/06_download_dcs_ccpi.py      # official CCPI + Colombo retail prices (slow, cached)
python3 scripts/05_build_datasets.py         # rebuild processed data with the CPI fix
python3 scripts/07_compare_wfp_cbsl.py       # WFP vs CBSL agreement + continuous series
python3 scripts/08_build_fuel_prices.py      # monthly fuel price history
```
