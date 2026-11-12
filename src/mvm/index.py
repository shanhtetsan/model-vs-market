"""Kalshi Premier League Index: what the market thinks every team is worth.

Idea: Kalshi prices every match (home / draw / away). Each price is the market's
view of *two* teams at once. Fit one Dixon-Coles rating per team so that the model
reproduces Kalshi's prices as closely as possible across recent matches. The result
is a market-implied rating for all 20 teams, which:

  * updates every time prices move,
  * puts every team on one scale (expected points vs an average PL side),
  * feeds a season simulation (title / top 4 / relegation odds).

Few matches per team -> fitting to prices alone is noisy, so ratings are shrunk
toward the results-based model (`mvm.model.DixonColes`). More market data = less shrinkage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .model import DixonColes

G = np.arange(11)  # goals 0..10


def outcome_probs(lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized Dixon-Coles H/D/A probabilities. lam, mu: (n,) -> (n, 3)."""
    ph = poisson.pmf(G[None, :], lam[:, None])           # (n, 11)
    pa = poisson.pmf(G[None, :], mu[:, None])
    m = ph[:, :, None] * pa[:, None, :]                  # (n, 11, 11)
    m[:, 0, 0] *= 1 - lam * mu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + mu * rho
    m[:, 1, 1] *= 1 - rho
    m /= m.sum(axis=(1, 2), keepdims=True)
    h = np.tril(np.ones((11, 11)), -1)[None]             # home goals > away goals
    return np.stack([(m * h).sum((1, 2)), np.trace(m, axis1=1, axis2=2),
                     (m * h.transpose(0, 2, 1)).sum((1, 2))], axis=1)


@dataclass
class MarketImplied:
    """Fit team ratings to market prices, shrunk toward a results-based prior."""
    market_weight: float = 60.0      # a price is far more informative than one result
    prior_strength: float = 1.0      # pull toward the results model (stabilizes teams with few markets)
    half_life_days: float = 45.0     # older prices count less
    rating_: pd.DataFrame = field(init=False, repr=False)

    def fit(self, markets: pd.DataFrame, prior: DixonColes, as_of: pd.Timestamp) -> "MarketImplied":
        """markets: date, home, away, mH, mD, mA (normalized Kalshi probs, pre-kickoff)."""
        self.prior_ = prior
        teams = sorted(set(markets.home) | set(markets.away))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = markets.home.map(idx).to_numpy()
        a = markets.away.map(idx).to_numpy()
        target = markets[["mH", "mD", "mA"]].to_numpy()
        age = (as_of - pd.to_datetime(markets.date)).dt.days.clip(lower=0).to_numpy()
        w = 0.5 ** (age / self.half_life_days)

        p_att = np.array([prior._params(t)[0] for t in teams])
        p_def = np.array([prior._params(t)[1] for t in teams])
        x0 = np.concatenate([p_att, p_def, [prior.home_adv_]])
        rho = prior.rho_

        def loss(x):
            att, dfn, home = x[:n], x[n:2 * n], x[2 * n]
            lam = np.exp(att[h] + dfn[a] + home)
            mu = np.exp(att[a] + dfn[h])
            p = np.clip(outcome_probs(lam, mu, rho), 1e-9, 1)
            ce = -self.market_weight * (w[:, None] * target * np.log(p)).sum()   # cross-entropy to market
            pen = self.prior_strength * (((att - p_att) ** 2).sum() + ((dfn - p_def) ** 2).sum())
            return ce + pen

        res = minimize(loss, x0, method="L-BFGS-B")
        att, dfn = res.x[:n], res.x[n:2 * n]
        shift = att.mean()                     # identifiability: mean attack = 0
        att, dfn = att - shift, dfn + shift
        self.home_adv_, self.rho_ = res.x[2 * n], rho
        self.attack_ = dict(zip(teams, att))
        self.defence_ = dict(zip(teams, dfn))
        self.fit_error_ = float(np.abs(self._predict(h, a, att, dfn) - target).mean())
        return self

    def _predict(self, h, a, att, dfn):
        lam = np.exp(att[h] + dfn[a] + self.home_adv_)
        mu = np.exp(att[a] + dfn[h])
        return outcome_probs(lam, mu, self.rho_)

    def predict(self, home: str, away: str) -> dict:
        lam = np.exp(self.attack_[home] + self.defence_[away] + self.home_adv_)
        mu = np.exp(self.attack_[away] + self.defence_[home])
        p = outcome_probs(np.array([lam]), np.array([mu]), self.rho_)[0]
        return dict(zip("HDA", p))

    def ratings(self) -> pd.DataFrame:
        """Index value = expected points per game vs an average PL team (half home, half away)."""
        rows = []
        avg_def = float(np.mean(list(self.defence_.values())))
        for t in self.teams_:
            at, df_ = self.attack_[t], self.defence_[t]
            pts = []
            for venue in (+1, -1):
                ha = self.home_adv_ if venue > 0 else 0.0
                hb = 0.0 if venue > 0 else self.home_adv_
                lam = np.exp(at + avg_def + ha)       # vs a league-average opponent
                mu = np.exp(0.0 + df_ + hb)           # mean attack is 0 after centring
                p = outcome_probs(np.array([lam]), np.array([mu]), self.rho_)[0]
                pts.append(3 * p[0] + p[1])
            rows.append({"team": t, "index": float(np.mean(pts)),
                         "attack": at, "defence": df_})
        r = pd.DataFrame(rows).sort_values("index", ascending=False).reset_index(drop=True)
        r.index += 1
        return r


def simulate_season(model, played: pd.DataFrame, fixtures: pd.DataFrame,
                    n_sims: int = 20000, seed: int = 0, strength_sd: float = 0.15) -> pd.DataFrame:
    """Monte Carlo the rest of the season.

    Team strength is NOT fixed: each simulated season draws a shock per team
    (sd = strength_sd on the log-goal scale, applied to attack and defence) to represent
    injuries, form and transfers over the remaining months. Without it the simulation is
    overconfident: favourites too high, outsiders near zero.

    played:   home, away, hg, ag  (results so far this season)
    fixtures: home, away          (remaining matches)
    -> per team: expected points, P(title), P(top 4), P(relegation)
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(played.home) | set(played.away) | set(fixtures.home) | set(fixtures.away))
    idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)

    pts0, gd0, gf0 = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for r in played.itertuples():
        i, j = idx[r.home], idx[r.away]
        gf0[i] += r.hg; gf0[j] += r.ag
        gd0[i] += r.hg - r.ag; gd0[j] += r.ag - r.hg
        if r.hg > r.ag: pts0[i] += 3
        elif r.hg < r.ag: pts0[j] += 3
        else: pts0[i] += 1; pts0[j] += 1

    pts = np.tile(pts0, (n_sims, 1)); gd = np.tile(gd0, (n_sims, 1)); gf = np.tile(gf0, (n_sims, 1))
    shock = rng.normal(0, strength_sd, (n_sims, nt))   # + better attack, - fewer conceded
    for r in fixtures.itertuples():
        i, j = idx[r.home], idx[r.away]
        a_h, d_h = model._params(r.home) if hasattr(model, "_params") else (model.attack_[r.home], model.defence_[r.home])
        a_a, d_a = model._params(r.away) if hasattr(model, "_params") else (model.attack_[r.away], model.defence_[r.away])
        sh, sa = shock[:, i], shock[:, j]
        hg = rng.poisson(np.exp(a_h + sh + d_a - sa + model.home_adv_))
        ag = rng.poisson(np.exp(a_a + sa + d_h - sh))
        pts[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd[:, i] += hg - ag; gd[:, j] += ag - hg
        gf[:, i] += hg; gf[:, j] += ag

    # Rank by points, then goal difference, then goals scored (tiny jitter breaks exact ties).
    key = pts * 1e6 + gd * 1e3 + gf + rng.random(pts.shape) * 1e-3
    rank = (-key).argsort(axis=1).argsort(axis=1) + 1          # 1 = top
    out = pd.DataFrame({
        "team": teams, "points_now": pts0, "exp_points": pts.mean(0),
        "p_title": (rank == 1).mean(0), "p_top4": (rank <= 4).mean(0),
        "p_relegation": (rank > nt - 3).mean(0),
    })
    return out.sort_values("exp_points", ascending=False).reset_index(drop=True)
