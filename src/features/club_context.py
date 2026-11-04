"""Club-context features for transfer examples: destination manager and team strength (Elo).

Time windows, relative to the transfer date t:
- Manager features use Transfermarkt games strictly before t.
- Elo uses the latest ClubElo snapshot on or before t - 1 day (snapshots are on
  the 1st and 15th of each month, so a rating is at most ~15 days old).
- ``dest_mgr_change_in_window`` and ``dest_mgr_differs_after`` use games on or
  after t. They are post-transfer diagnostics for error analysis and must never
  be used as model features.

Identity assumptions:
- Managers are identified by their Transfermarkt name (lower-cased, accents
  removed). Interruptions of <= 2 games between the same manager are treated
  as blips (cup games with a youth coach, licensed coach for Europe); longer
  caretaker spells count as manager changes.
- Transfermarkt clubs are linked to ClubElo names by fuzzy matching within the
  same country, plus the hand-checked ``ELO_OVERRIDES``. Clubs with no known
  country (not in clubs.csv), reserve teams, and leagues ClubElo doesn't cover
  get no Elo rather than a guessed one.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

# Transfermarkt domestic_competition_id -> ClubElo country code (only countries ClubElo covers)
COMP_TO_ELO_COUNTRY = {
    "GB1": "ENG", "ES1": "ESP", "IT1": "ITA", "L1": "GER", "FR1": "FRA", "PO1": "POR",
    "NL1": "NED", "BE1": "BEL", "TR1": "TUR", "SC1": "SCO", "GR1": "GRE", "RU1": "RUS",
    "A1": "AUT", "DK1": "DEN", "SE1": "SWE", "NO1": "NOR", "PL1": "POL", "RO1": "ROM",
}

# Hand-checked links for names fuzzy matching gets wrong (abbreviations, nicknames).
# Keys are Transfermarkt club_id, values are ClubElo club names.
ELO_OVERRIDES: dict[int, str] = {
    13: "Ath Madrid",     # Atlético de Madrid
    18: "MGladbach",      # Borussia Mönchengladbach
    131: "Barcelona",     # FC Barcelona (runner-up: Barcelona B)
    150: "Betis",         # Real Betis Balompié
    273: "Rennes",        # Stade Rennais FC
    276: "Verona",        # Hellas Verona
    336: "Sp Lisbon",     # Sporting CP
    368: "Sevilla",       # Sevilla FC (runner-up: Sevilla B)
    512: "Stoke",         # Stoke City
    621: "Ath Bilbao",    # Athletic Bilbao
    862: "Chievo",        # Chievo Verona
    940: "Celta",         # Celta de Vigo
    985: "Man United",    # Manchester United
    1160: "Nimes",        # Nîmes Olympique
    2441: "AEK",          # AEK Athens
}

# Generic tokens that carry no identity ("FC", "Calcio", ...); removed before comparing names.
_NAME_STOPWORDS = {
    "fc", "cf", "ac", "as", "ss", "sc", "afc", "cd", "ud", "rc", "rcd", "sd", "ca", "club", "de", "del",
    "la", "calcio", "football", "futbol", "sv", "spa", "sad", "s", "a", "p", "d", "the", "and", "us",
    "acf", "ssc", "bc", "vfb", "vfl", "tsg", "fk", "sk", "if", "bk", "kv", "krc", "rsc", "ogc", "osc",
}


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z ]", " ", unidecode(str(s)).lower())
    return " ".join(t for t in s.split() if t not in _NAME_STOPWORDS)


def _name_score(a: str, b: str) -> float:
    """Average of 'shares all words' and 'whole string similar'.

    token_set_ratio alone scores subsets as 100 ("bayern munich" vs "munich"),
    so it is balanced with the plain ratio.
    """
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return 0.5 * fuzz.token_set_ratio(a, b) + 0.5 * fuzz.ratio(a, b)


# ---------------------------------------------------------------- managers

def _absorb_short_interruptions(cg: pd.DataFrame, max_games: int) -> pd.Series:
    """Relabel runs of <= max_games games that sit between two runs of the same manager.

    These are one-off blips, not real changes: a youth coach for a cup game
    (Klopp -> Critchley -> Klopp) or a licensed coach named only for European
    games. Longer interruptions (e.g. Sevilla 2018/19) are kept.
    """
    key = cg["club_id"].astype(str) + "|" + cg["manager"]
    run = key.ne(key.shift()).cumsum()
    runs = cg.groupby(run).agg(club=("club_id", "first"), mgr=("manager", "first"), n=("manager", "size"))
    prev = runs.groupby("club")["mgr"].shift()
    nxt = runs.groupby("club")["mgr"].shift(-1)
    blip = (runs["n"] <= max_games) & prev.notna() & prev.eq(nxt)
    return run.map(runs["mgr"].where(~blip, prev))


def club_manager_games(games: pd.DataFrame, max_blip_games: int = 2) -> pd.DataFrame:
    """One row per (club, game) with the club's manager, sorted by club and date.

    Adds ``spell_start``: the date of the first game in the manager's current
    unbroken run at that club, and ``club_first_game`` to detect spells that
    began before the data starts (tenure is then a lower bound).
    """
    cols = ["date", "club_id", "manager"]
    cg = pd.concat([
        games[["date", "home_club_id", "home_club_manager_name"]].set_axis(cols, axis=1),
        games[["date", "away_club_id", "away_club_manager_name"]].set_axis(cols, axis=1),
    ]).dropna(subset=["date", "club_id", "manager"])
    cg["club_id"] = cg["club_id"].astype(int)
    cg["manager"] = cg["manager"].map(lambda m: unidecode(str(m)).strip().lower())
    cg = cg.sort_values(["club_id", "date"]).reset_index(drop=True)
    cg["manager"] = _absorb_short_interruptions(cg, max_blip_games)
    new_spell = cg["manager"].ne(cg.groupby("club_id")["manager"].shift())
    cg["spell_start"] = cg["date"].where(new_spell).groupby(cg["club_id"]).ffill()
    cg["club_first_game"] = cg.groupby("club_id")["date"].transform("min")
    return cg


def manager_features(t: pd.DataFrame, cg: pd.DataFrame, label_days: int = 365,
                     new_mgr_days: int = 180) -> pd.DataFrame:
    """Destination-manager features for each transfer row (needs transfer_date, to_club_id).

    Pre-transfer (model features):
      dest_mgr_tenure_days    days since the manager of the last game before t took over
      dest_new_mgr_180d       1 if that tenure is under ``new_mgr_days``
      dest_mgr_tenure_censored 1 if the spell started before the data does (tenure is a minimum)
    Post-transfer (diagnostics only, never model inputs):
      dest_mgr_differs_after  first game on/after t has a different manager than the last game before t
      dest_mgr_change_in_window any game in [t, t + label_days) has a manager other than the pre-transfer one
    """
    by_club = {c: g for c, g in cg.groupby("club_id")}
    out = []
    for club, date in zip(t["to_club_id"], t["transfer_date"]):
        g = by_club.get(int(club)) if pd.notna(club) else None
        row = dict(dest_mgr=np.nan, dest_mgr_tenure_days=np.nan, dest_new_mgr_180d=np.nan,
                   dest_mgr_tenure_censored=np.nan, dest_mgr_differs_after=np.nan,
                   dest_mgr_change_in_window=np.nan)
        if g is not None:
            before = g[g["date"] < date]
            if len(before):
                last = before.iloc[-1]
                tenure = (date - last["spell_start"]).days
                row.update(dest_mgr=last["manager"], dest_mgr_tenure_days=tenure,
                           dest_new_mgr_180d=float(tenure < new_mgr_days),
                           dest_mgr_tenure_censored=float(last["spell_start"] == last["club_first_game"]))
                after = g[(g["date"] >= date) & (g["date"] < date + pd.Timedelta(days=label_days))]
                if len(after):
                    row["dest_mgr_differs_after"] = float(after["manager"].iloc[0] != last["manager"])
                    row["dest_mgr_change_in_window"] = float((after["manager"] != last["manager"]).any())
        out.append(row)
    return pd.DataFrame(out, index=t.index)


# ---------------------------------------------------------------- Elo

def load_elo(path) -> pd.DataFrame:
    elo = pd.read_csv(path, parse_dates=["date"])
    return elo.dropna(subset=["date", "club", "elo"]).sort_values("date").reset_index(drop=True)


def club_name_variants(club_ids, clubs: pd.DataFrame, transfers: pd.DataFrame) -> dict[int, set[str]]:
    """All names Transfermarkt uses for each club: the full name in clubs.csv and
    the shorter names in transfers.csv (e.g. 'Bor. M'gladbach')."""
    names = {int(c): set() for c in club_ids if pd.notna(c)}
    for col_id, col_name, df in [("club_id", "name", clubs),
                                 ("from_club_id", "from_club_name", transfers),
                                 ("to_club_id", "to_club_name", transfers)]:
        sub = df[df[col_id].isin(names)][[col_id, col_name]].dropna().drop_duplicates()
        for cid, name in zip(sub[col_id], sub[col_name]):
            names[int(cid)].add(name)
    return names


def match_clubs_to_elo(club_ids, clubs: pd.DataFrame, transfers: pd.DataFrame, elo: pd.DataFrame,
                       active_between: tuple[str, str], min_score: float = 85, margin: float = 8,
                       overrides: dict[int, str] | None = None) -> pd.DataFrame:
    """Link Transfermarkt club_ids to ClubElo names within the same country.

    Candidates are limited to ClubElo clubs with a rating inside
    ``active_between`` (start, end); otherwise long-gone clubs with similar
    names (e.g. from the 2000s) crowd out clear matches. A fuzzy match is
    accepted only if its score is >= min_score and beats the runner-up by
    >= margin. Returns one row per club with the best candidate, its score, and
    ``method`` in {override, fuzzy, rejected, no_country}.
    """
    overrides = ELO_OVERRIDES if overrides is None else overrides
    country = clubs.set_index("club_id")["domestic_competition_id"].map(COMP_TO_ELO_COUNTRY)
    active = elo[elo["date"].between(*active_between)]
    elo_by_country = active.drop_duplicates("club").groupby("country")["club"].apply(list).to_dict()
    names = club_name_variants(club_ids, clubs, transfers)
    rows = []
    for cid, variants in names.items():
        tm_name = sorted(variants, key=len)[-1] if variants else ""
        c = country.get(cid)
        if cid in overrides:
            rows.append((cid, tm_name, c, overrides[cid], 100.0, np.nan, "override"))
            continue
        if not isinstance(c, str):
            rows.append((cid, tm_name, c, None, np.nan, np.nan, "no_country"))
            continue
        scored = sorted(((max(_name_score(v, k) for v in variants), k) for k in elo_by_country.get(c, [])),
                        reverse=True)
        best, second = (scored + [(0.0, None), (0.0, None)])[:2]
        ok = best[0] >= min_score and best[0] - second[0] >= margin
        rows.append((cid, tm_name, c, best[1], best[0], second[0], "fuzzy" if ok else "rejected"))
    return pd.DataFrame(rows, columns=["club_id", "tm_name", "country", "elo_club", "score",
                                       "runner_up_score", "method"])


def elo_asof(club_ids: pd.Series, dates: pd.Series, elo: pd.DataFrame, link: pd.DataFrame) -> pd.Series:
    """Elo of each club from the latest snapshot on or before date - 1 day (NaN if unlinked)."""
    ok = link[link["method"].isin(["fuzzy", "override"])]
    club_to_elo = dict(zip(ok["club_id"], ok["elo_club"]))
    q = pd.DataFrame({"elo_club": club_ids.map(lambda c: club_to_elo.get(int(c)) if pd.notna(c) else None),
                      "date": dates - pd.Timedelta(days=1)}, index=club_ids.index)
    q = q.reset_index().sort_values("date")
    has = q["elo_club"].notna()
    merged = pd.merge_asof(q[has], elo[["date", "club", "elo"]].rename(columns={"club": "elo_club"}),
                           on="date", by="elo_club", direction="backward")
    return merged.set_index("index")["elo"].reindex(club_ids.index)


def elo_features(t: pd.DataFrame, elo: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """from_elo, to_elo and elo_step (= to - from; positive means a step up) at the transfer date."""
    out = pd.DataFrame(index=t.index)
    out["from_elo"] = elo_asof(t["from_club_id"], t["transfer_date"], elo, link)
    out["to_elo"] = elo_asof(t["to_club_id"], t["transfer_date"], elo, link)
    out["elo_step"] = out["to_elo"] - out["from_elo"]
    return out
