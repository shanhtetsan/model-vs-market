"""Read-only Kalshi client for EPL match markets (series KXEPLGAME).

Public market data needs no API key. Each match is an event with three
yes/no markets: home win, away win, tie. We take the bid/ask midpoint of
each and normalize the three to sum to 1 (removes the overround).
"""
from __future__ import annotations

import pandas as pd
import requests

from .teams import normalize

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXEPLGAME"


def _to_prob(v) -> float | None:
    """Kalshi prices arrive as cent ints (43), dollar strings ("0.4300") or cent strings ("43").

    Normalize all of them to a probability in [0, 1]. Zero / missing -> None.
    """
    if v in (None, ""):
        return None
    if isinstance(v, str):
        try:
            x = float(v)
        except ValueError:
            return None
        x = x if "." in v else x / 100
    else:
        x = float(v) / 100 if float(v) > 1 else float(v)
    return x if 0 < x <= 1 else None


def _price(m: dict, field: str) -> float | None:
    """Return price in [0, 1] from a market dict (prefers the *_dollars field when present)."""
    d = _to_prob(m.get(f"{field}_dollars"))
    return d if d is not None else _to_prob(m.get(field))


def _mid(m: dict) -> float | None:
    bid, ask = _price(m, "yes_bid"), _price(m, "yes_ask")
    if bid and ask:
        return (bid + ask) / 2
    return _price(m, "last_price")


def _get(path: str, retries: int = 6, **params) -> dict:
    """GET with retry: backs off on 429 (rate limit) and 5xx, honouring Retry-After."""
    import time as _time
    for attempt in range(retries):
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After") or 2 ** attempt)
            _time.sleep(min(max(wait, 1.0), 30))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def fetch_open_events() -> list[dict]:
    events, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "open",
                  "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get("/events", **params)
        events += data.get("events", [])
        cursor = data.get("cursor")
        if not cursor:
            return events


def parse_teams(ev: dict) -> tuple[str, str] | None:
    """Event title 'Arsenal vs Leeds United' -> ('Arsenal', 'Leeds'). Home team is listed first."""
    title = ev.get("title", "")
    sep = " vs " if " vs " in title else " vs. " if " vs. " in title else None
    if not sep:
        return None
    try:
        return tuple(normalize(x.strip()) for x in title.split(sep, 1))
    except KeyError:
        return None


def parse_event(ev: dict) -> dict | None:
    """-> {home, away, H, D, A, ticker, close_time} with normalized probabilities."""
    teams = parse_teams(ev)
    if not teams:
        return None
    home, away = teams
    probs, close = {}, None
    for m in ev.get("markets", []):
        sub = (m.get("yes_sub_title") or "").strip()
        suffix = m.get("ticker", "").rsplit("-", 1)[-1].upper()
        if sub.lower() in ("tie", "draw") or suffix == "TIE":
            key = "D"
        else:
            try:
                key = "H" if normalize(sub or suffix) == home else "A"
            except KeyError:
                continue
        p = _mid(m)
        if p is not None:
            probs[key] = p
        close = close or m.get("close_time") or m.get("expected_expiration_time")
    if set(probs) != {"H", "D", "A"}:
        return None
    total = sum(probs.values())
    return {"home": home, "away": away, **{k: v / total for k, v in probs.items()},
            "overround": total - 1, "ticker": ev.get("event_ticker"), "close_time": close}


def epl_prices() -> list[dict]:
    return [p for ev in fetch_open_events() if (p := parse_event(ev))]


# ---------------------------------------------------------------------------
# Settled markets: pre-kickoff prices for every past EPL match (for the index).
# Kalshi splits data into live + historical tiers (see docs: getting_started/historical_data).
# Events are always available via /events; settled markets past the cutoff move to /historical/*.
# NOTE: written against the public docs; verify field names on first live run.
# ---------------------------------------------------------------------------
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "data" / "kalshi_cache"


def _cached(name: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    data = fn()
    f.write_text(json.dumps(data))
    time.sleep(0.3)  # stay well under Kalshi's public rate limit
    return data


def _iso(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def ticker_date(ticker: str) -> datetime | None:
    """KXEPLGAME-26APR10WHUWOL -> 2026-04-10 (UTC)."""
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return datetime(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)), tzinfo=timezone.utc)


def fetch_settled_events(min_close: datetime | None = None) -> list[dict]:
    events, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "settled", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get("/events", **params)
        events += data.get("events", [])
        cursor = data.get("cursor")
        if not cursor:
            break
    if min_close:  # keep only this season's matches (date is encoded in the ticker)
        events = [e for e in events if (ticker_date(e.get("event_ticker")) or min_close) >= min_close]
    return events


def _event_markets(ev: dict) -> list[dict]:
    """Nested markets disappear once archived; fall back to the historical endpoint."""
    if ev.get("markets"):
        return ev["markets"]
    tk = ev["event_ticker"]
    data = _cached(f"hist_mkts_{tk}", lambda: _get("/historical/markets", event_ticker=tk, limit=10))
    return data.get("markets", [])


def _kickoff(m: dict) -> datetime | None:
    for f in ("occurrence_datetime", "expected_occurrence_time"):
        if m.get(f):
            return _iso(m[f])
    exp = _iso(m.get("expected_expiration_time") or m.get("close_time"))
    return exp - timedelta(hours=2, minutes=30) if exp else None   # ~kickoff if expiry ≈ full time


def _candles(ticker: str, start: datetime, end: datetime) -> list[dict]:
    q = {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()), "period_interval": 60}
    def live():
        return _get(f"/series/{SERIES}/markets/{ticker}/candlesticks", **q)
    def hist():
        return _get(f"/historical/markets/{ticker}/candlesticks", **q)
    def fetch():
        try:
            d = live()
            if d.get("candlesticks"):
                return d
        except requests.HTTPError:
            pass
        return hist()
    return _cached(f"candles_{ticker}", fetch).get("candlesticks", [])


def _candle_mid(c: dict) -> float | None:
    def close(k):
        v = c.get(k) or {}
        d = _to_prob(v.get("close_dollars"))
        return d if d is not None else _to_prob(v.get("close"))
    bid, ask = close("yes_bid"), close("yes_ask")
    if bid and ask:
        return (bid + ask) / 2
    return close("price")


def prekickoff_prices(events: list[dict], hours_before: int = 1) -> pd.DataFrame:
    """One row per match: Kalshi H/D/A probabilities ~1h before kickoff, normalized."""
    rows, skipped = [], 0
    print(f"Pulling pre-kickoff Kalshi prices for {len(events)} matches (cached after the first run)...")
    for n, ev in enumerate(events, 1):
        if n % 25 == 0:
            print(f"  {n}/{len(events)}")
        try:
            row = _prekickoff_row(ev, hours_before)
        except Exception as e:  # one bad market should never kill the whole run
            skipped += 1
            print(f"  skipped {ev.get('event_ticker')}: {type(e).__name__}")
            continue
        if row:
            rows.append(row)
    print(f"Got prices for {len(rows)} matches" + (f", skipped {skipped}" if skipped else ""))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _prekickoff_row(ev: dict, hours_before: int) -> dict | None:
    teams = parse_teams(ev)
    if not teams:
        return None
    parsed = {"home": teams[0], "away": teams[1]}
    probs, ko = {}, None
    for m in _event_markets(ev):
        ko = ko or _kickoff(m)
        if not ko:
            continue
        cs = [c for c in _candles(m["ticker"], ko - timedelta(days=3), ko)
              if c.get("end_period_ts", 0) <= (ko - timedelta(hours=hours_before)).timestamp()]
        if not cs:
            continue
        mid = _candle_mid(cs[-1])
        sub = (m.get("yes_sub_title") or "").lower()
        key = "D" if sub in ("tie", "draw") or m["ticker"].upper().endswith("-TIE") else (
            "H" if normalize(m.get("yes_sub_title") or m["ticker"].rsplit("-", 1)[-1]) == parsed["home"] else "A")
        if mid:
            probs[key] = mid
    if set(probs) != {"H", "D", "A"}:
        return None
    tot = sum(probs.values())
    return {"date": ko.astimezone(timezone.utc).replace(tzinfo=None), "home": parsed["home"],
            "away": parsed["away"], "ticker": ev["event_ticker"],
            **{f"m{k}": v / tot for k, v in probs.items()}}
