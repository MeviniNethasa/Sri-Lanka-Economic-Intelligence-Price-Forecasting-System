"""Optional LSTM check (Phase 4): does a small LSTM beat the simpler models?

Trains a compact PyTorch LSTM (direct 6-month output on the last 24 monthly log
changes) for the two longest series, CPI and white rice, with an expanding
window refitted every 12 months. It is scored on the same *routine* forecast
origins as the main backtest (targets outside the 2022 crisis window) and
compared with the main models' MAE on exactly those forecasts.

The LSTM is NOT part of the main model set; it is kept only if it clearly wins.

Usage: python scripts/10_lstm_experiment.py   (needs torch; reads phase4_backtest_forecasts.csv)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecasting as F  # noqa: E402
import relationships as R  # noqa: E402

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables"
WIN, H, SEEDS = 24, 6, (0, 1, 2)


def windows(dl: pd.Series, upto: pd.Timestamp):
    """Training pairs (24 past changes -> next 6 changes) with every target month <= upto."""
    X, Y = [], []
    v = dl.loc[:upto]
    for i in range(WIN, len(v) - H + 1):
        x, y = v.iloc[i - WIN:i].values, v.iloc[i:i + H].values
        if np.isfinite(x).all() and np.isfinite(y).all():
            X.append(x), Y.append(y)
    return np.array(X), np.array(Y)


def train_lstm(X, Y, seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = X.mean(), X.std() + 1e-8

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 16, batch_first=True)
            self.head = torch.nn.Linear(16, H)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor((Y - mu) / sd, dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    return net, mu, sd


def main() -> None:
    import torch
    torch.set_num_threads(1)
    df = R.load_monthly()
    cont = pd.read_csv(ROOT / "data" / "processed" / "commodity_prices_continuous_monthly.csv", parse_dates=["date"])
    S = {s.name: s for s in F.build_specs(df, cont)}
    bt = pd.read_csv(TAB / "phase4_backtest_forecasts.csv", parse_dates=["origin", "target_date"])
    rows = []
    for name in ["CPI (cleaned)", "Rice (white)"]:
        y = S[name].y
        dl = 100 * np.log(y).diff()
        origins = sorted(bt.loc[(bt["series"] == name) & (bt["regime"] == "routine"), "origin"].unique())
        model, fitted_at = None, None
        for t in origins:
            t = pd.Timestamp(t)
            if fitted_at is None or (t - fitted_at).days > 360:      # refit every ~12 months
                X, Y = windows(dl, t)
                model = [train_lstm(X, Y, s) for s in SEEDS]           # small seed ensemble
                fitted_at = t
            x = dl.loc[:t].iloc[-WIN:].values
            if not np.isfinite(x).all() or pd.isna(y.get(t)):
                continue
            preds = []
            for net, mu, sd in model:
                with torch.no_grad():
                    p = net(torch.tensor(((x - mu) / sd)[None, :, None], dtype=torch.float32)).numpy()[0] * sd + mu
                preds.append(p)
            path = y.loc[t] * np.exp(np.cumsum(np.mean(preds, axis=0)) / 100)
            for h in range(H):
                rows.append({"series": name, "model": "lstm", "origin": t, "horizon": h + 1,
                             "target_date": t + pd.DateOffset(months=h + 1), "forecast": path[h]})
        print(f"lstm done: {name}")
    ls = F.attach_actuals(pd.DataFrame(rows), S)
    ls = ls[ls["regime"] == "routine"]
    # Compare on identical (series, origin, horizon) cells
    keys = ["series", "origin", "horizon"]
    base = bt[(bt["regime"] == "routine") & bt["series"].isin(ls["series"].unique())]
    common = ls[keys].merge(base[keys].drop_duplicates(), on=keys)
    both = pd.concat([ls, base]).merge(common, on=keys)
    cmp = (both.assign(ae=both["error"].abs()).groupby(["series", "model"])["ae"].mean().unstack(0))
    cmp.loc["n_forecasts"] = both[both["model"] == "lstm"].groupby("series").size()
    cmp.round(3).to_csv(TAB / "phase4_lstm_check.csv")
    print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
