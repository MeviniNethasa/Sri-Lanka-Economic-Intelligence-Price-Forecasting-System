"""Scrape CBSL 'Daily Price Report' PDFs into a tidy CSV.

Each report's page 2 holds a table: item, unit, then (yesterday, today) price
pairs per market. The number of pairs tells us which markets are present:

  5 pairs (vegetables, fruits, rice, coconut):
      WS Pettah | WS Dambulla* | RT Pettah | RT Dambulla | RT Narahenpita
      (*rice wholesale is Marandagahamula, not Dambulla)
  4 pairs (fish): WS Peliyagoda | WS Negombo | RT Negombo | RT Narahenpita
  3 pairs (sugar, dhal, coconut oil, eggs, dried fish):
      WS Pettah | RT Pettah | RT Narahenpita        (lower confidence)
  2 pairs (imported apple/orange): RT Pettah | RT Narahenpita (lower confidence)

Only the 'today' value of each pair is kept. Extracted text is cached in
data/raw/cbsl_daily_text/ so re-runs don't re-download.

Usage: python scripts/03_scrape_cbsl_daily_prices.py [START YYYY-MM-DD] [END YYYY-MM-DD]
"""
import io
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "cbsl_daily_text"
OUT = ROOT / "data" / "raw" / "cbsl_daily_prices.csv"
URLS = [
    "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{d}_e.pdf",
    "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{d}e.pdf",
]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-data-collector"}

RICE = {"Samba", "Nadu", "Kekulu (White)", "Kekulu (Red)", "Ponni Samba (Imp)", "Nadu (Imp)",
        "Kekulu (White) (Imp)", "Kekulu (Red) (Imp)", "Keeri Samba"}
LAYOUT = {
    5: [("Wholesale", "Pettah"), ("Wholesale", "Dambulla"), ("Retail", "Pettah"),
        ("Retail", "Dambulla"), ("Retail", "Narahenpita")],
    4: [("Wholesale", "Peliyagoda"), ("Wholesale", "Negombo"), ("Retail", "Negombo"),
        ("Retail", "Narahenpita")],
    3: [("Wholesale", "Pettah"), ("Retail", "Pettah"), ("Retail", "Narahenpita")],
    2: [("Retail", "Pettah"), ("Retail", "Narahenpita")],
}
ROW = re.compile(r"^(?P<item>[A-Za-z][A-Za-z ().'/-]*?)\s+(?P<unit>Rs\./\s?\w+)\s+(?P<rest>.+)$")
VAL = re.compile(r"n\.a\.|[\d,]+\.\d+")


def get_text(d: date) -> str | None:
    cache = CACHE / f"{d:%Y%m%d}.txt"
    if cache.exists():
        txt = cache.read_text()
        return txt or None
    for tmpl in URLS:
        try:
            r = requests.get(tmpl.format(d=f"{d:%Y%m%d}"), headers=UA, timeout=60)
        except requests.RequestException:
            continue
        if r.ok and r.content[:4] == b"%PDF":
            reader = PdfReader(io.BytesIO(r.content))
            txt = "\n".join(p.extract_text() or "" for p in reader.pages[1:])
            cache.write_text(txt)
            return txt
    cache.write_text("")  # remember "no report that day" (weekend/holiday)
    return None


def parse(txt: str, d: date) -> list[dict]:
    rows = []
    for line in txt.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        item = m["item"].strip()
        vals = VAL.findall(m["rest"])
        if len(vals) % 2 or len(vals) // 2 not in LAYOUT:
            continue
        if item in RICE and len(vals) // 2 != 5:
            continue  # rice row with a missing market column: ambiguous, skip
        today = vals[1::2]
        for (ptype, market), v in zip(LAYOUT[len(vals) // 2], today):
            if v == "n.a.":
                continue
            if item in RICE and ptype == "Wholesale" and market == "Dambulla":
                market = "Marandagahamula"
            rows.append({
                "date": d.isoformat(), "item": item, "unit": m["unit"].replace(" ", ""),
                "pricetype": ptype, "market": market, "price_lkr": float(v.replace(",", "")),
                "layout_pairs": len(vals) // 2,
            })
    return rows


def main() -> None:
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2023, 1, 1)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, found, d = [], 0, start
    while d <= end:
        if d.weekday() < 5:
            fresh = not (CACHE / f"{d:%Y%m%d}.txt").exists()
            txt = get_text(d)
            if txt:
                found += 1
                rows.extend(parse(txt, d))
            if fresh:
                time.sleep(0.4)  # be polite to the CBSL server
        d += timedelta(days=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"reports found: {found}, rows: {len(df)}, items: {df['item'].nunique() if len(df) else 0}")


if __name__ == "__main__":
    main()
