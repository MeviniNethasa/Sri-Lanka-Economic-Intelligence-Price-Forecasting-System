"""Sanity checks after a data refresh, before anything is committed.

Compares each key output with the version in the last commit (git HEAD) and
fails (exit code 1) if a refresh looks broken:
  * a required file is missing or empty
  * its row count fell by more than 2% (e.g. a download returned partial data)
  * its latest date moved backwards
  * any tracked data/report file is larger than 50 MB

Usage: python scripts/12_check_refresh.py      (run from the repository root)
"""
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAX_MB = 50
MAX_ROW_DROP = 0.02

CHECKS = {  # file -> date column (None = no date check)
    "data/raw/cbsl_daily_prices.csv": "date",
    "data/raw/dcs_ccpi_monthly.csv": "date",
    "data/raw/imf_cpi_lka.csv": None,
    "data/raw/fred_lkr_usd_monthly.csv": None,
    "data/processed/macro_monthly.csv": "date",
    "data/processed/analysis_panel_monthly.csv": "date",
    "data/processed/cbsl_prices_monthly.csv": "date",
    "data/processed/commodity_prices_continuous_monthly.csv": "date",
    "data/processed/fuel_prices_monthly.csv": "date",
    "reports/tables/phase4_selected_forecasts.csv": None,
    "reports/tables/phase5_anomaly_scores.csv": "date",
}


def head_version(rel: str) -> pd.DataFrame | None:
    try:
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None  # new file
    return pd.read_csv(io.BytesIO(blob))


def main() -> int:
    problems = []
    for rel, dcol in CHECKS.items():
        path = ROOT / rel
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"{rel}: missing or empty")
            continue
        new = pd.read_csv(path)
        old = head_version(rel)
        msg = f"{rel}: {len(new):,} rows"
        if old is not None:
            if len(new) < (1 - MAX_ROW_DROP) * len(old):
                problems.append(f"{rel}: rows fell {len(old):,} -> {len(new):,}")
            if dcol and dcol in new and dcol in old and len(old):
                n_max, o_max = pd.to_datetime(new[dcol]).max(), pd.to_datetime(old[dcol]).max()
                msg += f", last {n_max:%Y-%m-%d} (was {o_max:%Y-%m-%d})"
                if n_max < o_max:
                    problems.append(f"{rel}: latest date went backwards {o_max:%Y-%m-%d} -> {n_max:%Y-%m-%d}")
        print("  " + msg)
    for folder in ["data", "reports"]:
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and p.stat().st_size > MAX_MB * 1024 ** 2:
                problems.append(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1024 ** 2:.0f} MB > {MAX_MB} MB")
    if problems:
        print("\nREFRESH CHECK FAILED:\n" + "\n".join(f"  - {p}" for p in problems))
        return 1
    print("\nrefresh check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
