"""CLI.

  python -m mvm download              # fetch results (openfootball; add --source footballdata for fresher data + odds)
  python -m mvm backtest              # walk-forward test on last season -> output/backtest_*
  python -m mvm preview               # Thursday: model vs live Kalshi prices -> chart + thread draft
  python -m mvm results               # Monday: score last week's logged predictions -> thread draft
  python -m mvm index                 # Kalshi Premier League Index: market-implied ratings + season odds
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import kalshi, report
from .data import download, load_results, result_label
from .model import DixonColes

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "data" / "predictions_log.csv"
OUT = ROOT / "output"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT = SEASONS[-1]


def cmd_download(a):
    download(SEASONS, source=a.source)


def cmd_backtest(a):
    df = load_results(SEASONS[:-1] if a.season != CURRENT else SEASONS)
    res = bt.walk_forward(df, a.season)
    train = df[df["season"] < a.season]
    summary = bt.summarize(res, train)
    cal = bt.calibration(res)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / f"backtest_{a.season}.csv", index=False)
    summary.to_csv(OUT / f"backtest_{a.season}_summary.csv")
    report.calibration_chart(cal, f"Is my model honest? EPL {a.season}", OUT / f"calibration_{a.season}.png")
    print(f"\nWalk-forward backtest, EPL {a.season} ({len(res)} matches)\n")
    print(summary.round(4).to_string())
    print("\nCalibration:\n", cal.round(3).to_string())


def cmd_preview(a):
    df = load_results(SEASONS)
    model = DixonColes().fit(df, as_of=pd.Timestamp.now().normalize())
    prices = kalshi.epl_prices()
    if not prices:
        raise SystemExit("No open KXEPLGAME events on Kalshi right now.")
    rows = []
    for p in prices:
        m = model.predict(p["home"], p["away"])
        rows.append({"home": p["home"], "away": p["away"], "ticker": p["ticker"],
                     "close_time": p["close_time"],
                     "p_H": m["H"], "p_D": m["D"], "p_A": m["A"],
                     "k_H": p["H"], "k_D": p["D"], "k_A": p["A"]})
    df_w = pd.DataFrame(rows)
    df_w["edge"] = df_w["p_H"] - df_w["k_H"]
    df_w["edge_abs"] = df_w["edge"].abs()
    df_w["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Append to the permanent log (this is your public track record; never edit past rows).
    LOG.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    pd.concat([old, df_w], ignore_index=True).drop_duplicates("ticker", keep="first").to_csv(LOG, index=False)

    tag = a.matchweek or datetime.now().strftime("Week of %b %d")
    chart = report.preview_chart(df_w, tag, OUT / "preview.png", score=a.score)
    thread = report.preview_thread(df_w, tag)
    (OUT / "preview_thread.txt").write_text(thread)
    print(thread, f"\n\nChart: {chart}")


def cmd_results(a):
    if not LOG.exists():
        raise SystemExit("No predictions logged yet. Run `preview` first.")
    log = pd.read_csv(LOG)
    res = load_results([CURRENT])
    merged = log.merge(res, on=["home", "away"], how="inner")
    if merged.empty:
        raise SystemExit("No logged matches have results yet. Re-download data and retry.")
    merged["result"] = [result_label(h, g) for h, g in zip(merged.hg, merged.ag)]
    o = merged["result"].to_numpy()
    merged["brier_model"] = bt.brier(merged[["p_H", "p_D", "p_A"]].to_numpy(), o)
    merged["brier_kalshi"] = bt.brier(merged[["k_H", "k_D", "k_A"]].to_numpy(), o)
    merged["date"] = pd.to_datetime(merged["date"])
    week = merged[merged["date"] >= merged["date"].max() - pd.Timedelta(days=6)]
    tag = a.matchweek or week["date"].max().strftime("Week of %b %d")
    thread = report.results_thread(week, merged, tag)
    OUT.mkdir(exist_ok=True)
    (OUT / "results_thread.txt").write_text(thread)
    merged.to_csv(OUT / "scored_predictions.csv", index=False)
    print(thread)


def _remaining_fixtures(played: pd.DataFrame) -> pd.DataFrame:
    """Double round-robin: every team hosts every other team once. Remaining = all pairs minus played.

    Needs no fixture file, so it works with either data source.
    """
    teams = sorted(set(played.home) | set(played.away))
    done = set(zip(played.home, played.away))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    return pd.DataFrame(rows, columns=["home", "away"])


def _demo_markets(prior: DixonColes, played: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Fake 'Kalshi' prices for offline testing: the results model plus a few team-level opinions."""
    import numpy as np
    rng = np.random.default_rng(42)
    views = {t: rng.normal(0, .08) for t in set(played.home) | set(played.away)}
    rows = []
    games = list(played[["date", "home", "away"]].itertuples(index=False))
    games += [(pd.Timestamp.now().normalize(), h, a) for h, a in upcoming.head(10).itertuples(index=False)]
    for d, h, a in games:
        lam = np.exp(prior._params(h)[0] + views[h] + prior._params(a)[1] - views[a] + prior.home_adv_)
        mu = np.exp(prior._params(a)[0] + views[a] + prior._params(h)[1] - views[h])
        from .index import outcome_probs
        p = outcome_probs(np.array([lam]), np.array([mu]), prior.rho_)[0]
        rows.append({"date": d, "home": h, "away": a, "mH": p[0], "mD": p[1], "mA": p[2]})
    return pd.DataFrame(rows)


def cmd_index(a):
    from .index import MarketImplied, simulate_season
    df = load_results(SEASONS)
    today = pd.Timestamp.now().normalize()
    prior = DixonColes().fit(df, as_of=today)
    played = df[df["season"] == CURRENT]
    remaining = _remaining_fixtures(played)

    if a.demo:
        mk = _demo_markets(prior, played, remaining)
    else:
        season_start = pd.Timestamp(played["date"].min()).tz_localize("UTC").to_pydatetime()
        ko = {}
        if "kickoff_utc" in played:
            ko = {(h, a_): k for h, a_, k in zip(played.home, played.away, played.kickoff_utc)}
        mk = kalshi.prekickoff_prices(kalshi.fetch_settled_events(min_close=season_start), kickoffs=ko)
        live = pd.DataFrame(kalshi.epl_prices())
        if not live.empty:
            live = live.rename(columns={"H": "mH", "D": "mD", "A": "mA"})
            live["date"] = today
            mk = pd.concat([mk, live[["date", "home", "away", "mH", "mD", "mA"]]], ignore_index=True)
    if mk.empty:
        raise SystemExit("No Kalshi prices found.")
    mk["date"] = pd.to_datetime(mk["date"])

    now = MarketImplied().fit(mk, prior, as_of=today)
    week_ago = today - pd.Timedelta(days=7)
    prev_mk = mk[mk["date"] < week_ago]
    prev = MarketImplied().fit(prev_mk, prior, as_of=week_ago).ratings() if len(prev_mk) > 20 else None

    r = now.ratings()
    sim = simulate_season(now, played, remaining, n_sims=a.sims)
    t = r.merge(sim, on="team")
    t["delta"] = t["team"].map(prev.set_index("team")["index"]) if prev is not None else float("nan")
    t["delta"] = t["index"] - t["delta"]

    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "index.csv", index=False)
    leader = t.sort_values("p_title", ascending=False).iloc[0]
    title = a.title or f"{leader.team} lead the title race"
    sub = (f"{leader.team} win the title in {leader.p_title:.0%} of simulated seasons."
           + (" (DEMO: synthetic prices)" if a.demo else ""))
    path = report.index_chart(t, title, sub, OUT / ("index_demo.png" if a.demo else "index.png"))
    print(t[["team", "index", "delta", "exp_points", "p_title", "p_top4", "p_relegation"]].round(3).to_string())
    print(f"\nFit error vs Kalshi (mean abs, prob points): {now.fit_error_:.3f} over {len(mk)} matches")
    if now.fit_error_ > 0.06:
        print("WARNING: fit error is high. Kalshi prices may not be pre-match. Don't post this chart.")
    print(f"Chart: {path}")


def cmd_titlegap(a):
    """Compare Kalshi's title market with the index's match-implied title odds.

    --market "Arsenal=71,Man City=12,Brighton=8,Liverpool=5,Chelsea=4"  (percent, from Kalshi)
    """
    from .teams import normalize
    idx = pd.read_csv(OUT / "index.csv").set_index("team")
    rows = []
    for part in a.market.split(","):
        name, pct = part.rsplit("=", 1)
        t = normalize(name.strip())
        rows.append({"team": t, "market": float(pct) / 100, "implied": float(idx.loc[t, "p_title"])})
    df = pd.DataFrame(rows)
    big = df.assign(g=(df.implied - df.market).abs()).sort_values("g").iloc[-1]
    title = a.title or f"{big.team}: the gap between two Kalshi prices"
    sub = a.subtitle or (f"Match prices say {big.implied:.0%}. The title market says {big.market:.0%}.")
    path = report.title_gap_chart(df, OUT / "title_gap.png", title, sub, a.note)
    print(df.round(3).to_string(index=False), f"\nChart: {path}")


def cmd_ladder(a):
    """Any Kalshi ladder -> full forecast chart.

    python -m mvm ladder --above "80=97,85=90,90=72,95=40" --title "GTA 6 Metacritic" --unit "score" --ceil 100
    """
    from .ladder import buckets, expected, parse_ladder
    spec = a.above
    if a.strictly_above:  # Kalshi "Above 95" on whole-number scores means 96 or higher
        spec = ",".join(f"{float(k) + 1:g}={v}" for k, v in (x.split("=") for x in spec.split(",")))
    b, issues = buckets(parse_ladder(spec), a.floor, a.ceil, integer=a.integer or a.strictly_above)
    ev = expected(b)
    top = b.loc[b.p.idxmax()]
    sub = a.subtitle or f"Most likely: {top.label} ({top.p:.0%}). Market-implied average: {ev:.1f}."
    path = report.ladder_chart(b, OUT / "ladder.png", a.kicker, a.title, sub,
                               hero=f"{ev:.1f}", hero_label=f"Market-implied expected {a.unit}",
                               issues=issues, note=a.note)
    print(b.assign(p=b.p.round(3)).to_string(index=False))
    print(f"\nExpected {a.unit}: {ev:.2f}")
    print("Consistency: " + ("; ".join(issues) if issues else "passed"))
    print(f"Chart: {path}")


def main():
    p = argparse.ArgumentParser(prog="mvm")
    sub = p.add_subparsers(required=True)
    d = sub.add_parser("download")
    d.add_argument("--source", choices=["openfootball", "footballdata"], default="openfootball")
    d.set_defaults(f=cmd_download)
    b = sub.add_parser("backtest")
    b.add_argument("--season", default="2025-26")
    b.set_defaults(f=cmd_backtest)
    for name, fn in [("preview", cmd_preview), ("results", cmd_results)]:
        s = sub.add_parser(name)
        s.add_argument("--matchweek", help='Label for posts, e.g. "Matchweek 12"')
        s.add_argument("--score", help='Season scoreboard for the chart, e.g. "Model 14 – 11 Kalshi"')
        s.set_defaults(f=fn)
    ix = sub.add_parser("index")
    ix.add_argument("--demo", action="store_true", help="synthetic prices, no API (for testing)")
    ix.add_argument("--sims", type=int, default=20000)
    ix.add_argument("--title", help="override chart headline")
    ix.set_defaults(f=cmd_index)
    tg = sub.add_parser("titlegap")
    tg.add_argument("--market", required=True, help='Kalshi title prices, e.g. "Arsenal=71,Man City=12"')
    tg.add_argument("--title"); tg.add_argument("--subtitle"); tg.add_argument("--note")
    tg.set_defaults(f=cmd_titlegap)
    ld = sub.add_parser("ladder")
    ld.add_argument("--above", required=True, help='"threshold=percent,...", e.g. "85=90,90=72,95=40"')
    ld.add_argument("--title", required=True); ld.add_argument("--subtitle")
    ld.add_argument("--kicker", default="Kalshi forecast"); ld.add_argument("--unit", default="value")
    ld.add_argument("--floor", type=float); ld.add_argument("--ceil", type=float); ld.add_argument("--note")
    ld.add_argument("--integer", action="store_true", help="whole-number outcomes: label exact values")
    ld.add_argument("--strictly-above", action="store_true",
                    help='thresholds are Kalshi "Above X" on whole numbers (X+1 or higher); implies --integer')
    ld.set_defaults(f=cmd_ladder)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
