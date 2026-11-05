"""Link Wyscout players to Transfermarkt players (milestone 1 identity matching).

The two sources use different player ids, so players are matched by identity:
- Pass 1: same date of birth and a name similarity >= 80 (rapidfuzz
  token_set_ratio), checked against several Wyscout name variants, so
  "Neymar da Silva Santos Júnior" still finds "Neymar".
- Pass 2, for Wyscout players left over: same birth year, a stricter full-name
  match >= 90 (token_sort_ratio), and the Transfermarkt player must have
  played in 2017/18.
Both passes need a clear winner (a 5-point gap to the runner-up), and every
Transfermarkt id is used at most once.

Audit: ``club_agreement`` checks whether both sources agree on each matched
player's main 2017/18 club. Disagreements are January movers (fine) or wrong
matches.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from src.data.wyscout import fix_text

SEASON_1718 = ("2017-07-01", "2018-07-01")
PASS1_MIN, PASS2_MIN, MARGIN = 80, 90, 5


def norm_name(s) -> str:
    s = unidecode(fix_text(s)).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_variants(r) -> list[str]:
    """Full name, full name with middle name, short name without the initial, and last name."""
    v = {norm_name(f"{r.firstName} {r.lastName}"),
         norm_name(f"{r.firstName} {r.middleName} {r.lastName}"),
         norm_name(re.sub(r"^\w\.\s*", "", r.shortName or "")),  # "L. Messi" -> "messi"
         norm_name(r.lastName)}
    return [x for x in v if x]


def best_unique(c: pd.DataFrame, margin: float = MARGIN) -> pd.DataFrame:
    """Keep each Wyscout player's best candidate if it beats the runner-up by ``margin``;
    then use each Transfermarkt id once (highest score wins)."""
    c = c.sort_values(["wy_id", "score"], ascending=[True, False]).copy()
    c["rk"] = c.groupby("wy_id").cumcount()
    best = c[c.rk == 0].set_index("wy_id")
    second = c[c.rk == 1].set_index("wy_id")["score"].reindex(best.index).fillna(0)
    best = best[(best.score - second) >= margin].reset_index()
    return best.sort_values("score", ascending=False).drop_duplicates("player_id")[["wy_id", "player_id", "score"]]


def match_players(wy: pd.DataFrame, tm_players: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Link table: wy_id, player_id, score, match_pass.

    ``wy`` needs wy_id, dob and the Wyscout name columns; ``tm_players`` needs player_id, dob and
    name; ``apps`` (Transfermarkt appearances) defines who was active in 2017/18 for pass 2.
    """
    wy = wy.assign(variants=[name_variants(r) for r in wy.itertuples()])
    tm = tm_players.assign(nm=tm_players["name"].map(norm_name))

    c1 = (wy[["wy_id", "dob", "variants"]].dropna(subset=["dob"])
          .merge(tm[["player_id", "dob", "nm"]].dropna(subset=["dob"]), on="dob"))
    c1["score"] = [max((fuzz.token_set_ratio(v, n) for v in vs), default=0) for vs, n in zip(c1.variants, c1.nm)]
    m1 = best_unique(c1[c1.score >= PASS1_MIN]).assign(match_pass="dob+name")

    season = apps[(apps.date >= SEASON_1718[0]) & (apps.date < SEASON_1718[1])]
    active = tm[tm.player_id.isin(season.player_id) & ~tm.player_id.isin(m1.player_id)]
    active = active.assign(yr=active.dob.dt.year)[["player_id", "yr", "nm"]].dropna(subset=["yr"])
    left = wy[~wy.wy_id.isin(m1.wy_id)][["wy_id", "dob", "variants"]]
    left = left.assign(yr=left.dob.dt.year).drop(columns="dob").dropna(subset=["yr"])
    c2 = left.merge(active, on="yr")
    c2["score"] = [max((fuzz.token_sort_ratio(v, n) for v in vs if " " in v), default=0)
                   for vs, n in zip(c2.variants, c2.nm)]
    m2 = best_unique(c2[c2.score >= PASS2_MIN]).assign(match_pass="year+name+active")

    link = pd.concat([m1, m2], ignore_index=True)
    link["player_id"] = link.player_id.astype(int)
    return link


def club_agreement(wy: pd.DataFrame, link: pd.DataFrame, tm_players: pd.DataFrame,
                   apps: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    """Matched players with both sources' main 2017/18 club and ``club_agree``
    (1 if the club names match >= 70, NaN if Transfermarkt has no 2017/18 club)."""
    season = apps[(apps.date >= SEASON_1718[0]) & (apps.date < SEASON_1718[1])]
    club_name = dict(zip(clubs["club_id"], clubs["name"]))
    tm_club = (season.groupby(["player_id", "player_club_id"])["minutes_played"].sum().reset_index()
               .sort_values("minutes_played", ascending=False).drop_duplicates("player_id")
               .set_index("player_id")["player_club_id"].map(club_name))
    mt = wy.merge(link, on="wy_id")
    mt["tm_name"] = mt.player_id.map(tm_players.set_index("player_id")["name"])
    mt["tm_club"] = mt.player_id.map(tm_club)
    mt["club_agree"] = [float(fuzz.token_set_ratio(norm_name(a), norm_name(b)) >= 70) if isinstance(b, str) else np.nan
                        for a, b in zip(mt.wy_team, mt.tm_club)]
    return mt
