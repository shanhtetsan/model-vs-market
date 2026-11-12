"""Walk-forward backtest: refit before every match date, predict that day's games.

No look-ahead: the model only ever sees matches played before the prediction date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import result_label
from .model import DixonColes

OUTCOMES = ["H", "D", "A"]


def brier(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Multi-class Brier per match: sum over H/D/A of (p - y)^2. 0 = perfect, lower = better."""
    y = np.stack([(outcome == o).astype(float) for o in OUTCOMES], axis=1)
    return ((p - y) ** 2).sum(axis=1)


def log_loss(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    i = np.array([OUTCOMES.index(o) for o in outcome])
    return -np.log(np.clip(p[np.arange(len(p)), i], 1e-12, 1))


def walk_forward(df: pd.DataFrame, test_season: str, train_years: int = 3, **model_kw) -> pd.DataFrame:
    test = df[df["season"] == test_season]
    rows = []
    for day, games in test.groupby(test["date"].dt.normalize()):
        hist = df[(df["date"] < day) & (df["date"] >= day - pd.DateOffset(years=train_years))]
        m = DixonColes(**model_kw).fit(hist, as_of=day)
        for g in games.itertuples():
            p = m.predict(g.home, g.away)
            row = {"date": g.date, "home": g.home, "away": g.away, "hg": g.hg, "ag": g.ag,
                   "result": result_label(g.hg, g.ag), "pH": p["H"], "pD": p["D"], "pA": p["A"]}
            if hasattr(g, "psh") and pd.notna(getattr(g, "psh", np.nan)):
                inv = np.array([1 / g.psh, 1 / g.psd, 1 / g.psa])
                row.update(dict(zip(["mH", "mD", "mA"], inv / inv.sum())))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, train_df: pd.DataFrame | None = None) -> pd.DataFrame:
    out = bt["result"].to_numpy()
    scores = {}
    model_p = bt[["pH", "pD", "pA"]].to_numpy()
    scores["Model (Dixon-Coles)"] = model_p

    # Baseline 1: everyone gets 1/3.
    scores["Coin flip (1/3 each)"] = np.full_like(model_p, 1 / 3)
    # Baseline 2: league-wide historical H/D/A frequencies (from training data, no look-ahead).
    if train_df is not None:
        hist = train_df.apply(lambda r: result_label(r.hg, r.ag), axis=1).value_counts(normalize=True)
        scores["Base rates (home/draw/away %)"] = np.tile([hist["H"], hist["D"], hist["A"]], (len(bt), 1))
    if {"mH", "mD", "mA"} <= set(bt.columns) and bt["mH"].notna().all():
        scores["Market (Pinnacle close)"] = bt[["mH", "mD", "mA"]].to_numpy()

    return pd.DataFrame({
        name: {"Brier": brier(p, out).mean(), "LogLoss": log_loss(p, out).mean(),
               "Accuracy": (np.array(["H", "D", "A"])[p.argmax(1)] == out).mean()}
        for name, p in scores.items()
    }).T.sort_values("Brier")


def calibration(bt: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Pool all H/D/A predictions; compare predicted prob vs observed frequency per bin."""
    p = bt[["pH", "pD", "pA"]].to_numpy().ravel()
    y = np.stack([(bt["result"] == o) for o in OUTCOMES], axis=1).astype(float).ravel()
    b = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    return pd.DataFrame({"pred": p, "obs": y, "bin": b}).groupby("bin", observed=True).agg(
        predicted=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size"))
