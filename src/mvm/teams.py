"""Team-name normalization across data sources (openfootball, football-data.co.uk, Kalshi)."""
from __future__ import annotations

import difflib
import re

# Canonical short names -> known aliases (lowercase, punctuation-stripped).
CANONICAL = {
    "Arsenal": ["arsenal", "ars"],
    "Aston Villa": ["aston villa", "villa", "avl"],
    "Bournemouth": ["bournemouth", "afc bournemouth", "bou"],
    "Brentford": ["brentford", "bre"],
    "Brighton": ["brighton", "brighton hove albion", "brighton and hove albion", "bha", "bri"],
    "Burnley": ["burnley", "bur"],
    "Chelsea": ["chelsea", "che"],
    "Coventry": ["coventry", "coventry city", "cov"],
    "Crystal Palace": ["crystal palace", "palace", "cry"],
    "Everton": ["everton", "eve"],
    "Fulham": ["fulham", "ful"],
    "Hull": ["hull", "hull city", "hul"],
    "Ipswich": ["ipswich", "ipswich town", "ips"],
    "Leeds": ["leeds", "leeds united", "lee"],
    "Leicester": ["leicester", "leicester city", "lei"],
    "Liverpool": ["liverpool", "liv"],
    "Luton": ["luton", "luton town", "lut"],
    "Man City": ["man city", "manchester city", "mci"],
    "Man United": ["man united", "man utd", "manchester united", "manchester utd", "mun"],
    "Newcastle": ["newcastle", "newcastle united", "new"],
    "Norwich": ["norwich", "norwich city", "nor"],
    "Nott'm Forest": ["nottm forest", "nottingham forest", "forest", "nfo", "not"],
    "Sheffield United": ["sheffield united", "sheffield utd", "shu"],
    "Southampton": ["southampton", "sou"],
    "Sunderland": ["sunderland", "sun"],
    "Tottenham": ["tottenham", "tottenham hotspur", "spurs", "tot"],
    "Watford": ["watford", "wat"],
    "West Ham": ["west ham", "west ham united", "whu"],
    "Wolves": ["wolves", "wolverhampton", "wolverhampton wanderers", "wol"],
}

_ALIAS = {alias: canon for canon, aliases in CANONICAL.items() for alias in aliases}


def _clean(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(afc|fc)\b", "", s)
    s = s.replace(" and ", " ")
    return re.sub(r"\s+", " ", s).strip()


def normalize(name: str) -> str:
    """Map any source's team name to the canonical short name."""
    c = _clean(name)
    if c in _ALIAS:
        return _ALIAS[c]
    match = difflib.get_close_matches(c, list(_ALIAS), n=1, cutoff=0.75)
    if match:
        return _ALIAS[match[0]]
    raise KeyError(f"Unknown team name: {name!r} (add it to teams.CANONICAL)")
