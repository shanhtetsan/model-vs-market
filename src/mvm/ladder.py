"""Ladders: turn a set of 'above X' (or 'before date') contracts into the market's full forecast.

Kalshi lists many outcomes as ladders, e.g. GTA 6 Metacritic >= 85 / >= 90 / >= 95. Each price is
P(outcome >= threshold). Adjacent differences give the probability of each bucket, which yields an
expected value. Ladders must be non-increasing (P(>=95) can't exceed P(>=90)). When they aren't,
the contracts disagree with each other, which is a market-health flag.
"""
from __future__ import annotations

import pandas as pd


def parse_ladder(spec: str) -> pd.DataFrame:
    """'85=95,90=78,95=41' -> thresholds with P(>= threshold), sorted ascending."""
    rows = []
    for part in spec.split(","):
        k, v = part.split("=")
        rows.append({"threshold": float(k), "p_above": float(v) / 100})
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def buckets(ladder: pd.DataFrame, floor: float | None = None, ceil: float | None = None,
            integer: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Bucket probabilities + consistency issues.

    floor / ceil: plausible min / max of the quantity (used for the tail buckets' midpoints).
    """
    t = ladder.threshold.tolist(); p = ladder.p_above.tolist()
    issues = [f"P(≥{t[i+1]:g}) = {p[i+1]:.0%} is above P(≥{t[i]:g}) = {p[i]:.0%}"
              for i in range(len(t) - 1) if p[i + 1] > p[i] + 1e-9]
    # Repair for the forecast (keep issues for reporting): enforce non-increasing via running min.
    fixed, m = [], 1.0
    for x in p:
        m = min(m, x); fixed.append(m)
    step = (t[-1] - t[0]) / max(len(t) - 1, 1) if len(t) > 1 else 5
    lo = floor if floor is not None else t[0] - step
    hi = ceil if ceil is not None else t[-1] + step
    if integer:  # whole-number outcomes (review scores): label exact values, e.g. "96", "95 or lower"
        rows = [{"label": f"{t[0] - 1:g} or lower", "p": 1 - fixed[0], "mid": (lo + t[0] - 1) / 2}]
        for i in range(len(t) - 1):
            lab = f"{t[i]:g}" if t[i + 1] - t[i] == 1 else f"{t[i]:g}–{t[i+1] - 1:g}"
            rows.append({"label": lab, "p": fixed[i] - fixed[i + 1], "mid": (t[i] + t[i + 1] - 1) / 2})
        rows.append({"label": f"{t[-1]:g}+", "p": fixed[-1], "mid": (t[-1] + hi) / 2})
    else:
        rows = [{"label": f"Below {t[0]:g}", "p": 1 - fixed[0], "mid": (lo + t[0]) / 2}]
        for i in range(len(t) - 1):
            rows.append({"label": f"{t[i]:g}–{t[i+1]:g}", "p": fixed[i] - fixed[i + 1],
                         "mid": (t[i] + t[i + 1]) / 2})
        rows.append({"label": f"{t[-1]:g}+", "p": fixed[-1], "mid": (t[-1] + hi) / 2})
    b = pd.DataFrame(rows)
    b["p"] = b["p"].clip(lower=0)
    b["p"] = b["p"] / b["p"].sum()
    return b, issues


def expected(b: pd.DataFrame) -> float:
    return float((b.p * b.mid).sum())
