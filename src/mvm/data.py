"""Load EPL results.

Two sources:
  * openfootball/football.json  (free, on GitHub; can lag a few weeks)
  * football-data.co.uk CSVs    (updated after every matchday; also has closing bookmaker odds)

Both normalize to one DataFrame:
  date, home, away, hg, ag  [+ optional closing odds psh, psd, psa]
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from .teams import normalize

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json"
FOOTBALLDATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def season_code(season: str) -> str:
    """'2025-26' -> '2526' (football-data.co.uk folder name)."""
    a, b = season.split("-")
    return a[-2:] + b[-2:]


def download(seasons: list[str], source: str = "openfootball") -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        if source == "openfootball":
            url, out = OPENFOOTBALL_URL.format(season=s), RAW / f"en1_{s}.json"
        else:
            url, out = FOOTBALLDATA_URL.format(code=season_code(s)), RAW / f"E0_{s}.csv"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"saved {out.name}")


def _ft(score) -> tuple[int, int] | None:
    if not score:
        return None
    if isinstance(score, dict):
        ft = score.get("ft")
        return tuple(ft) if ft else None
    return tuple(score)


def load_openfootball(season: str, include_fixtures: bool = False) -> pd.DataFrame:
    matches = json.loads((RAW / f"en1_{season}.json").read_text())["matches"]
    rows = []
    for m in matches:
        ft = _ft(m.get("score"))
        if ft is None and not include_fixtures:
            continue
        rows.append({
            "date": pd.Timestamp(m["date"]),
            "round": m.get("round"),
            "home": normalize(m["team1"]),
            "away": normalize(m["team2"]),
            "hg": ft[0] if ft else pd.NA,
            "ag": ft[1] if ft else pd.NA,
            "season": season,
        })
    return pd.DataFrame(rows)


def load_footballdata(season: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"E0_{season}.csv")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True),
        "home": df["HomeTeam"].map(normalize),
        "away": df["AwayTeam"].map(normalize),
        "hg": df["FTHG"].astype(int),
        "ag": df["FTAG"].astype(int),
        "season": season,
    })
    # Pinnacle closing odds = best free proxy for a sharp market price.
    for col, src in [("psh", "PSCH"), ("psd", "PSCD"), ("psa", "PSCA")]:
        if src in df:
            out[col] = df[src].values
    return out


def load_results(seasons: list[str]) -> pd.DataFrame:
    """Prefer football-data.co.uk CSV for a season if present, else openfootball JSON."""
    frames = []
    for s in seasons:
        if (RAW / f"E0_{s}.csv").exists():
            frames.append(load_footballdata(s))
        elif (RAW / f"en1_{s}.json").exists():
            frames.append(load_openfootball(s))
        else:
            raise FileNotFoundError(f"No data for {s}; run `python -m mvm download`")
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    return df


def result_label(hg: int, ag: int) -> str:
    return "H" if hg > ag else "A" if ag > hg else "D"
