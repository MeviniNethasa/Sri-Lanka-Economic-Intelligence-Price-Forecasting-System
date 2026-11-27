"""Run Phase 4 rolling-origin backtests and final forecasts; write tables to reports/tables/.

Usage: python scripts/09_forecast_backtest.py        (takes a few minutes)
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecasting as F  # noqa: E402
import relationships as R  # noqa: E402

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = R.load_monthly()
    cont = pd.read_csv(ROOT / "data" / "processed" / "commodity_prices_continuous_monthly.csv", parse_dates=["date"])
    specs = F.build_specs(df, cont)

    bt, infos = F.run_backtests(specs)
    scores = F.score(bt)
    scores_all_h = F.score(bt, by=("series", "model", "regime"))
    best = F.best_models(scores)
    fc = F.final_forecasts(specs, infos, info_date=df.index.max())
    fc = fc.merge(best[["series", "best_model"]], on="series", how="left")
    fc["is_best_model"] = fc["model"] == fc["best_model"]

    r4 = lambda d: d.round(4)
    r4(bt).to_csv(TAB / "phase4_backtest_forecasts.csv", index=False)
    r4(scores).to_csv(TAB / "phase4_backtest_scores.csv", index=False)
    r4(scores_all_h).to_csv(TAB / "phase4_backtest_scores_all_horizons.csv", index=False)
    r4(best).to_csv(TAB / "phase4_best_models.csv", index=False)
    infos.astype(str).to_csv(TAB / "phase4_model_specs.csv", index=False)
    r4(fc).to_csv(TAB / "phase4_forecasts.csv", index=False)
    print(best.round(3).to_string(index=False))
    print(f"backtest rows: {len(bt):,}; final forecast rows: {len(fc)}")


if __name__ == "__main__":
    main()
