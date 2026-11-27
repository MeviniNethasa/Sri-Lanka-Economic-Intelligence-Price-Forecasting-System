"""Download DCS Colombo Consumer Price Index (CCPI, base 2021=100) monthly releases.

Each release (statistics.gov.lk/qlink/PressRelease/CCPI_<YYYYMMDD>E, published on the
last working day of the month) contains:
  Table 03 - the last 13 months of CCPI all-items / food / non-food index, MoM %,
             YoY % and 12-month moving-average inflation
  Table 05 - monthly average Colombo retail prices of ~40 items incl. petrol,
             diesel, kerosene and LPG (same month last year, previous month, current month)

Outputs (data/raw/):
  dcs_ccpi_monthly.csv          one row per month (latest release wins on revisions)
  dcs_ccpi_retail_prices.csv    item x month retail prices from Table 05
  dcs_ccpi_text/<YYYYMM>.txt    cached extracted text (re-runs don't re-download)

Usage: python scripts/06_download_dcs_ccpi.py [START YYYY-MM] [END YYYY-MM]
"""
import calendar
import io
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "dcs_ccpi_text"
URL = "https://www.statistics.gov.lk/qlink/PressRelease/CCPI_{d}E"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-data-collector"}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_RE = re.compile(r"(" + "|".join(r"\s*".join(m) for m in MONTHS) + r")", re.I)
NUM = re.compile(r"-?\d[\d,]*\.\d+|-?\d+(?=\s|$)")


def fetch_release(year: int, month: int) -> str | None:
    """Try the last 10 days of the month and the first 5 of the next (late releases)."""
    cache = CACHE / f"{year}{month:02d}.txt"
    if cache.exists():
        return cache.read_text() or None
    last = calendar.monthrange(year, month)[1]
    nxt = date(year + month // 12, month % 12 + 1, 1)
    candidates = [date(year, month, d) for d in range(last, last - 10, -1)]
    candidates += [date(nxt.year, nxt.month, d) for d in range(1, 6)]
    definitive = True
    for d in candidates:
        if d.weekday() >= 5 or d > date.today():
            continue
        try:
            r = requests.get(URL.format(d=f"{d:%Y%m%d}"), headers=UA, timeout=60)
        except requests.RequestException:
            definitive = False
            continue
        if not r.ok and r.status_code != 404:
            definitive = False
        time.sleep(0.3)
        # A release dated in the last days of month M (or the first days of M+1)
        # is the CCPI for month M. Some releases have image-only first pages, so
        # the month is taken from the release date, not the PDF header.
        if r.ok and r.content[:4] == b"%PDF":
            txt = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
            cache.write_text(f"#SOURCE {URL.format(d=f'{d:%Y%m%d}')}\n" + txt)
            return txt
    # Remember "no release" only once the whole search window is >= 10 days in the past
    # and every request got a clean answer; otherwise retry on the next run.
    if definitive and (date.today() - candidates[-1]).days >= 10:
        cache.write_text("")
    return None


def parse_table03(txt: str, year: int, month: int) -> list[dict]:
    """Rows of Table 03 are the 13 months ending at the release month, in order."""
    start = txt.find("Table 03")
    end = txt.find("Note", start)
    if start < 0:
        return []
    rows = []
    for line in txt[start:end].splitlines():
        mm = MONTH_RE.search(line)
        if not mm:
            continue
        nums = [float(x.replace(",", "")) for x in NUM.findall(line[mm.end():])]
        if len(nums) >= 11:
            rows.append(nums[:12] + [None] * (12 - len(nums[:12])))
    rows = rows[-13:]
    out = []
    for i, v in enumerate(rows):
        k = len(rows) - 1 - i  # months back from the release month
        y, m = year, month - k
        while m <= 0:
            m += 12
            y -= 1
        out.append({
            "date": f"{y}-{m:02d}-01",
            "ccpi_index": v[0], "ccpi_mom_pct": v[1], "ccpi_yoy_pct": v[2], "ccpi_12m_avg_pct": v[3],
            "ccpi_food_index": v[4], "ccpi_food_mom_pct": v[5], "ccpi_food_yoy_pct": v[6],
            "ccpi_food_12m_avg_pct": v[7], "ccpi_nonfood_index": v[8], "ccpi_nonfood_mom_pct": v[9],
            "ccpi_nonfood_yoy_pct": v[10], "ccpi_nonfood_12m_avg_pct": v[11],
            "release": f"{year}-{month:02d}",
        })
    return out


ITEM_RE = re.compile(r"^\s*\d+\s*\.\s*(?P<item>[A-Za-z][A-Za-z .()&/-]*?)\s+(?P<unit>Kg|Lt|Each|750ml|400g|12\.5kg|[\d.]+\s?kg)\s+(?P<rest>.*)$", re.I)


def parse_table05(txt: str, year: int, month: int) -> list[dict]:
    start = txt.find("Table 05")
    if start < 0:
        return []
    rows = []
    for line in txt[start:].splitlines():
        m = ITEM_RE.match(line)
        if not m:
            continue
        nums = re.findall(r"-?\d[\d,]*\.\d+", m["rest"])
        if len(nums) < 3:
            continue
        cur, prev_month, last_year = (float(nums[i].replace(",", "")) for i in (2, 1, 0))
        rows.append({"date": f"{year}-{month:02d}-01", "item": " ".join(m["item"].split()),
                     "unit": m["unit"], "price_lkr": cur, "price_prev_month": prev_month,
                     "price_same_month_last_year": last_year})
    return rows


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-02"
    end = sys.argv[2] if len(sys.argv) > 2 else f"{date.today():%Y-%m}"
    CACHE.mkdir(parents=True, exist_ok=True)
    idx, retail, found = [], [], 0
    for p in pd.period_range(start, end, freq="M"):
        txt = fetch_release(p.year, p.month)
        if not txt:
            print(f"{p}: no release found")
            continue
        found += 1
        t3 = parse_table03(txt, p.year, p.month)
        t5 = parse_table05(txt, p.year, p.month)
        idx += t3
        retail += t5
        print(f"{p}: table03 rows={len(t3)} table05 rows={len(t5)}")
    ccpi = (pd.DataFrame(idx).sort_values(["date", "release"])
              .drop_duplicates("date", keep="last").sort_values("date"))
    ccpi.to_csv(RAW / "dcs_ccpi_monthly.csv", index=False)
    pd.DataFrame(retail).to_csv(RAW / "dcs_ccpi_retail_prices.csv", index=False)
    print(f"releases: {found}, CCPI months: {len(ccpi)} ({ccpi['date'].min()}..{ccpi['date'].max()}), "
          f"retail rows: {len(retail)}")


if __name__ == "__main__":
    main()
