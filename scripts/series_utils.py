"""Shared helpers for loading and joining LankaEcon price series.

Example
-------
>>> from series_utils import continuous_price, ITEM_MAP
>>> s = continuous_price("Coconut")          # WFP to 2025-09, CBSL-extended after
>>> s.tail()
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# WFP commodity -> (CBSL daily-report item, confidence of the match)
# "high": same product and unit; "medium": same product, variety/grade may differ.
ITEM_MAP = {
    "Beans": ("Beans", "high"),
    "Cabbage": ("Cabbage", "high"),
    "Carrots": ("Carrot", "high"),
    "Tomatoes": ("Tomato", "high"),
    "Eggplants": ("Brinjal", "high"),
    "Pumpkin": ("Pumpkin", "high"),
    "Snake gourd": ("Snake gourd", "high"),
    "Coconut": ("Coconut (Avg.)", "high"),
    "Eggs": ("Egg (White)", "medium"),
    "Lentils": ("Red Dhal", "high"),
    "Onions (imported)": ("Big Onion (Imp)", "high"),
    "Onions (red, local)": ("Red Onion (Local)", "high"),
    "Onions (red, imported)": ("Red Onion (Imp)", "high"),
    "Potatoes (local)": ("Potato (Local)", "high"),
    "Potatoes (imported)": ("Potato (Imp)", "high"),
    "Chili (red, dry raw)": ("Dried Chilli (Imp)", "medium"),
    "Fish (dry, katta)": ("Katta (Imp)", "medium"),
    "Fish (dry, sprats)": ("Sprat (Imp)", "medium"),
    "Papaya": ("Papaw", "high"),
    "Pineapples": ("Pineapple", "medium"),
    "Bananas": ("Banana (Sour)", "medium"),
    "Rice (white)": ("Nadu", "medium"),
    "Rice (medium grain)": ("Samba", "medium"),
}


def normalise_cbsl_item(s: pd.Series) -> pd.Series:
    """Fix OCR-style typos in CBSL item names (e.g. 'Red Onion (lmp)')."""
    return s.str.replace("(lmp)", "(Imp)", regex=False).str.strip()


def load_wfp_national() -> pd.DataFrame:
    w = pd.read_csv(PROC / "commodity_prices_national_monthly.csv", parse_dates=["date"])
    return w[w["pricetype"] == "Retail"]


def load_cbsl_monthly() -> pd.DataFrame:
    c = pd.read_csv(PROC / "cbsl_prices_monthly.csv", parse_dates=["date"])
    c["item"] = normalise_cbsl_item(c["item"])
    return c


def cbsl_retail_mean(c: pd.DataFrame, item: str) -> pd.Series:
    """Mean CBSL retail price across the markets that report the item, by month."""
    x = c[(c["item"] == item) & (c["pricetype"] == "Retail")]
    return x.groupby("date")["price_lkr"].mean()


def continuous_price(wfp_commodity: str) -> pd.DataFrame:
    """Read the precomputed continuous series for one WFP commodity."""
    d = pd.read_csv(PROC / "commodity_prices_continuous_monthly.csv", parse_dates=["date"])
    return d[d["commodity"] == wfp_commodity].set_index("date")
