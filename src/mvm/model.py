"""Dixon-Coles (1997) goals model with exponential time decay.

Goals for home team i vs away team j:
    home_goals ~ Poisson(exp(attack_i + defence_j + home_adv))
    away_goals ~ Poisson(exp(attack_j + defence_i))
with the Dixon-Coles tau correction for low scores (0-0, 1-0, 0-1, 1-1),
and each match weighted by exp(-xi * days_ago) so recent form counts more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    xi: float = 0.0019          # time decay per day (half-life ~ 1 year)
    l2: float = 0.01            # small ridge penalty on everyone
    shrink: float = 20.0        # extra pull toward a prior for teams with few matches
    promoted_prior: tuple = (-0.3, 0.3)   # (attack, defence) for teams new to the data window

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        if as_of is None:
            as_of = df["date"].max() + pd.Timedelta(days=1)
        df = df[df["date"] < as_of]
        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        x = df["hg"].to_numpy()
        y = df["ag"].to_numpy()
        days = (as_of - df["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * days)

        # Bayesian-style shrinkage: few matches -> trust the prior more.
        # Teams with no matches older than a year are treated as promoted (weak prior).
        n_games = np.bincount(np.concatenate([h, a]), minlength=n)
        first = pd.concat([df[["date", "home"]].rename(columns={"home": "t"}),
                           df[["date", "away"]].rename(columns={"away": "t"})]).groupby("t")["date"].min()
        new = np.array([(as_of - first[t]).days < 365 for t in self.teams_])
        pa = np.where(new, self.promoted_prior[0], 0.0)
        pd_ = np.where(new, self.promoted_prior[1], 0.0)
        k = self.l2 + self.shrink / (n_games + 1)

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            att = att - att.mean()                      # identifiability
            home, rho = p[2 * n], p[2 * n + 1]
            lam = np.exp(att[h] + dfn[a] + home)
            mu = np.exp(att[a] + dfn[h])
            tau = np.clip(_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
            pen = (k * ((att - pa) ** 2 + (dfn - pd_) ** 2)).sum()
            return -(w * ll).sum() + pen

        p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n) + [(None, None), (-0.2, 0.2)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        att = res.x[:n] - res.x[:n].mean()
        self.attack_ = dict(zip(self.teams_, att))
        self.defence_ = dict(zip(self.teams_, res.x[n:2 * n]))
        self.home_adv_, self.rho_ = res.x[2 * n], res.x[2 * n + 1]
        return self

    def _params(self, team: str) -> tuple[float, float]:
        """Unseen team (e.g. newly promoted) -> average of the 3 weakest known teams."""
        if team in self.attack_:
            return self.attack_[team], self.defence_[team]
        weakest = self.ratings().tail(3)
        return float(weakest["attack"].mean()), float(weakest["defence"].mean())

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        (ah, dh), (aa, da) = self._params(home), self._params(away)
        lam = np.exp(ah + da + self.home_adv_)
        mu = np.exp(aa + dh)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        X, Y = np.meshgrid(g, g, indexing="ij")
        m *= _tau(X, Y, np.full_like(m, lam), np.full_like(m, mu), self.rho_)
        return m / m.sum()

    def predict(self, home: str, away: str) -> dict:
        m = self.score_matrix(home, away)
        return {
            "H": float(np.tril(m, -1).sum()),
            "D": float(np.trace(m)),
            "A": float(np.triu(m, 1).sum()),
        }

    def ratings(self) -> pd.DataFrame:
        r = pd.DataFrame({"attack": self.attack_, "defence": self.defence_})
        r["strength"] = r["attack"] - r["defence"]
        return r.sort_values("strength", ascending=False)
