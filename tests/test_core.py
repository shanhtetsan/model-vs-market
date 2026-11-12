import numpy as np
import pandas as pd

from mvm import kalshi
from mvm.backtest import brier
from mvm.model import DixonColes
from mvm.teams import normalize


def test_normalize_across_sources():
    assert normalize("Brighton & Hove Albion FC") == "Brighton"
    assert normalize("Man United") == normalize("Manchester United FC") == "Man United"
    assert normalize("Nott'm Forest") == normalize("Nottingham Forest FC")
    assert normalize("Wolverhampton Wanderers FC") == "Wolves"


def test_kalshi_parse_event_cents_and_dollars():
    ev = {"event_ticker": "KXEPLGAME-26NOV21ARSLEE", "title": "Arsenal vs Leeds United",
          "markets": [
              {"ticker": "KXEPLGAME-26NOV21ARSLEE-ARS", "yes_sub_title": "Arsenal", "yes_bid": 68, "yes_ask": 70},
              {"ticker": "KXEPLGAME-26NOV21ARSLEE-TIE", "yes_sub_title": "Tie", "yes_bid_dollars": "0.19", "yes_ask_dollars": "0.21"},
              {"ticker": "KXEPLGAME-26NOV21ARSLEE-LEE", "yes_sub_title": "Leeds United", "yes_bid": 11, "yes_ask": 13},
          ]}
    p = kalshi.parse_event(ev)
    assert (p["home"], p["away"]) == ("Arsenal", "Leeds")
    assert abs(p["H"] + p["D"] + p["A"] - 1) < 1e-9
    assert p["H"] > 0.6 and p["A"] < 0.15


def test_brier_perfect_and_uniform():
    o = np.array(["H"])
    assert brier(np.array([[1.0, 0, 0]]), o)[0] == 0
    assert abs(brier(np.full((1, 3), 1 / 3), o)[0] - 2 / 3) < 1e-9


def test_model_learns_strong_team():
    rng = np.random.default_rng(0)
    rows, teams = [], ["Arsenal", "Chelsea", "Everton", "Fulham"]
    for d in range(200):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=d), "home": h, "away": a,
                     "hg": rng.poisson(2.5 if h == "Arsenal" else 1.2),
                     "ag": rng.poisson(2.2 if a == "Arsenal" else 1.0)})
    m = DixonColes().fit(pd.DataFrame(rows))
    p = m.predict("Arsenal", "Everton")
    assert p["H"] > 0.55 and abs(sum(p.values()) - 1) < 1e-9
    assert m.ratings().index[0] == "Arsenal"


def test_kalshi_price_formats():
    from mvm.kalshi import _to_prob
    assert _to_prob(43) == 0.43 and _to_prob(1) == 0.01 and _to_prob(0.43) == 0.43
    assert _to_prob("0.4300") == 0.43
    assert _to_prob("43") == 0.43
    assert _to_prob("1.0000") == 1.0
    assert _to_prob(0) is None and _to_prob("") is None and _to_prob(None) is None


def test_ticker_date():
    from mvm.kalshi import ticker_date
    d = ticker_date("KXEPLGAME-26APR10WHUWOL")
    assert (d.year, d.month, d.day) == (2026, 4, 10)
    assert ticker_date("garbage") is None
