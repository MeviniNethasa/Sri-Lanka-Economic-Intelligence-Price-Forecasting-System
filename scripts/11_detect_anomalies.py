"""Run Phase 5 anomaly detection and write tables to reports/tables/.

Usage: python scripts/11_detect_anomalies.py      (about 15 seconds)

Outputs (reports/tables/):
  phase5_series_specs.csv        series, group, coverage, detector settings
  phase5_anomaly_scores.csv      every observed series-month: STL residual z, Isolation Forest score, change-point proximity
  phase5_anomalies.csv           flagged series-months with classification (economic shock / data artifact / ...)
  phase5_anomaly_episodes.csv    flagged months merged into episodes
  phase5_change_points.csv       PELT segments (start, end, mean, sd) per series
  phase5_event_catalogue.csv     known events and data quirks used for validation
  phase5_event_validation.csv    did each detector pick up each event?
  phase5_seasonal_profile.csv    STL seasonal factors by calendar month (vegetables, rice, coconut)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anomalies as A  # noqa: E402

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def main() -> None:
    specs = A.build_specs()
    sc, segs, dropped = A.score_all(specs)
    sc = A.add_cross_source(sc, specs)
    flagged = A.classify(sc)
    eps = A.episodes(flagged)
    val = A.validate_events(sc, segs, specs)

    spec_rows = []
    for s in specs:
        y = s.y
        d = sc[sc["series"] == s.name]
        spec_rows.append({
            "series": s.name, "group": s.group, "first": f"{y.first_valid_index():%Y-%m}",
            "last": f"{y.last_valid_index():%Y-%m}", "observed_months": int(y.notna().sum()),
            "missing_months": int(y.isna().sum()), "residual_method": d["stl_method"].iloc[0],
            "scale_floor": s.scale_floor, "if_model": s.if_pool or ("recent_commodity" if s.group in
                                                                    ("staple", "vegetable") else s.group),
            "if_months_scored": int(d["if_scored"].sum()),
            "if_rows_incomplete": int(dropped.get(s.name, 0)),
            "cpd_signal": "monthly log change (normal cost)" if s.cpd == "dlog" else "log level (l2 cost)",
            "cpd_min_size": s.cpd_min_size, "in_if_training": s.train, "note": s.note})
    pd.DataFrame(spec_rows).to_csv(TAB / "phase5_series_specs.csv", index=False)

    cols = ["date", "series", "group", "price", "source", "log_level", "trend", "seasonal", "resid",
            "stl_z", "stl_method", "stl_flag", "edge", "if_score", "if_threshold", "if_pctile", "if_flag",
            "if_scored", "cp_near", "r1", "alt_r1", "splice", "n_methods", "anomaly"]
    r = lambda d: d.round(4)
    r(sc[cols]).to_csv(TAB / "phase5_anomaly_scores.csv", index=False)
    keep = ["date", "series", "group", "price", "source", "r1", "next_r1", "alt_r1", "stl_z", "stl_method",
            "stl_flag", "if_flag", "if_pctile", "cp_near", "n_methods", "edge", "breadth",
            "classification", "reason"]
    r(flagged[keep].sort_values(["group", "series", "date"])).to_csv(TAB / "phase5_anomalies.csv", index=False)
    r(eps).to_csv(TAB / "phase5_anomaly_episodes.csv", index=False)
    r(segs).to_csv(TAB / "phase5_change_points.csv", index=False)
    A.events_table().to_csv(TAB / "phase5_event_catalogue.csv", index=False)
    r(val).to_csv(TAB / "phase5_event_validation.csv", index=False)
    prof = pd.concat([A.seasonal_profile(sc, "vegetable"),
                      A.seasonal_profile(sc[sc["series"].isin(["Rice (white)", "Coconut"])], "staple")])
    prof.to_csv(TAB / "phase5_seasonal_profile.csv")

    engine = segs["engine"].iloc[0] if len(segs) else "n/a"
    print(f"series: {len(specs)}   series-months scored: {len(sc):,}   flagged: {len(flagged):,} "
          f"in {len(eps)} episodes   change-point engine: {engine}")
    print("\nEpisodes by classification:\n" + eps["classification"].value_counts().to_string())
    print("\nEvent validation:\n" + val["outcome"].value_counts().to_string())


if __name__ == "__main__":
    np.random.seed(0)
    main()
