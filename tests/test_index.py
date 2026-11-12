import numpy as np
import pandas as pd

from mvm import kalshi
from mvm.data import load_results
from mvm.index import MarketImplied, outcome_probs, simulate_season
from mvm.model import DixonColes


def test_outcome_probs_match_scalar_model():
    m = DixonColes()
    m.attack_, m.defence_, m.home_adv_, m.rho_ = {"A": .3, "B": -.1}, {"A": -.2, "B": .1}, .25, -.05
    s = m.predict("A", "B")
    v = outcome_probs(np.array([np.exp(.3 + .1 + .25)]), np.array([np.exp(-.1 - .2)]), -.05)[0]
    assert np.allclose(v, [s["H"], s["D"], s["A"]], atol=1e-9)


def test_index_recovers_market_view():
    """If the market rates Hull much higher than the results model, the index should move Hull up."""
    df = load_results(["2024-25", "2025-26", "2026-27"])
    prior = DixonColes().fit(df)
    truth = DixonColes(); truth.__dict__.update({k: (dict(v) if isinstance(v, dict) else v)
                                                for k, v in prior.__dict__.items()})
    truth.attack_["Hull"] += 0.35
    games = df[df.season == "2026-27"][["date", "home", "away"]]
    rng = np.random.default_rng(0)
    rows = []
    for g in games.itertuples():
        p = np.array(list(truth.predict(g.home, g.away).values())) + rng.normal(0, .01, 3)
        p = np.clip(p, .01, None); p /= p.sum()
        rows.append({"date": g.date, "home": g.home, "away": g.away, "mH": p[0], "mD": p[1], "mA": p[2]})
    mkts = pd.DataFrame(rows)
    mi = MarketImplied().fit(mkts, prior, as_of=games.date.max())
    assert mi.attack_["Hull"] - prior._params("Hull")[0] > 0.15
    assert mi.fit_error_ < 0.03


def test_simulate_season_probabilities_valid():
    df = load_results(["2025-26", "2026-27"])
    m = DixonColes().fit(df)
    played = df[df.season == "2026-27"]
    teams = sorted(set(played.home))
    fixtures = pd.DataFrame([(h, a) for h in teams for a in teams if h != a][:60], columns=["home", "away"])
    out = simulate_season(m, played, fixtures, n_sims=2000)
    assert abs(out.p_title.sum() - 1) < 1e-9
    assert abs(out.p_top4.sum() - 4) < 1e-9
    assert abs(out.p_relegation.sum() - 3) < 1e-9


def test_prekickoff_prices_with_mocked_api(monkeypatch, tmp_path):
    monkeypatch.setattr(kalshi, "CACHE", tmp_path)
    ko = "2026-11-21T15:00:00Z"
    ev = {"event_ticker": "KXEPLGAME-26NOV21ARSLEE", "title": "Arsenal vs Leeds", "markets": [
        {"ticker": "KXEPLGAME-26NOV21ARSLEE-ARS", "yes_sub_title": "Arsenal", "occurrence_datetime": ko},
        {"ticker": "KXEPLGAME-26NOV21ARSLEE-TIE", "yes_sub_title": "Tie", "occurrence_datetime": ko},
        {"ticker": "KXEPLGAME-26NOV21ARSLEE-LEE", "yes_sub_title": "Leeds", "occurrence_datetime": ko}]}
    ts = int(pd.Timestamp("2026-11-21T13:00:00Z").timestamp())
    price = {"ARS": (68, 70), "TIE": (19, 21), "LEE": (11, 13)}

    def fake_get(path, **q):
        b, a = price[path.split("/")[-2].rsplit("-", 1)[-1]]
        return {"candlesticks": [{"end_period_ts": ts, "yes_bid": {"close": b}, "yes_ask": {"close": a}}]}
    monkeypatch.setattr(kalshi, "_get", fake_get)
    df = kalshi.prekickoff_prices([ev])
    assert len(df) == 1 and df.loc[0, "home"] == "Arsenal"
    assert abs(df.loc[0, ["mH", "mD", "mA"]].sum() - 1) < 1e-9 and df.loc[0, "mH"] > .6
