"""The playing-time label and the transfer flags it depends on.

Label (decision D2): y = the player's league minutes for a club / (90 x the
club's league games) in a time window, capped at 1. League = Transfermarkt
``competition_type == "domestic_league"``, so cup and European games a player
may not be registered for don't count against him. The same function gives:
- ``y``: [transfer date, transfer date + 365 days) at the new club
- ``prev_share_tm``: [transfer date - 365 days, transfer date) at the old club
Clubs with no recorded league games in the window get NaN, never a fake 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EMPTY = (np.array([], dtype="datetime64[ns]"), np.array([]))


class PlayingTime:
    """Index of club game dates and player minutes, for fast ``share`` look-ups."""

    def __init__(self, games: pd.DataFrame, apps: pd.DataFrame, player_ids, league_only: bool = True):
        if league_only:
            leagues = set(games.loc[games.competition_type == "domestic_league", "competition_id"])
            games = games[games.competition_id.isin(leagues)]
            apps = apps[apps.competition_id.isin(leagues)]
        cg = pd.concat([games[["home_club_id", "date"]].set_axis(["club_id", "date"], axis=1),
                        games[["away_club_id", "date"]].set_axis(["club_id", "date"], axis=1)]).dropna()
        self.club_dates = {int(c): np.sort(g["date"].values) for c, g in cg.groupby("club_id")}
        a = apps[apps.player_id.isin(set(player_ids))].dropna(subset=["date"])
        self.apps = {(int(p), int(c)): (g["date"].values, g["minutes_played"].fillna(0).values)
                     for (p, c), g in a.groupby(["player_id", "player_club_id"])}

    def share(self, pid, club, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, int]:
        """(playing-time share, number of club games) in [start, end); (NaN, 0) without games."""
        if pd.isna(club):
            return np.nan, 0
        d = self.club_dates.get(int(club))
        if d is None:
            return np.nan, 0
        s, e = start.to_datetime64(), end.to_datetime64()
        n = int(np.searchsorted(d, e) - np.searchsorted(d, s))
        if n == 0:
            return np.nan, 0
        dates, mins = self.apps.get((int(pid), int(club)), EMPTY)
        return min(mins[(dates >= s) & (dates < e)].sum() / (90 * n), 1.0), n


def flag_loan_returns(transfers: pd.DataFrame, max_days: int = 400) -> pd.DataFrame:
    """Transfers sorted by player and date, with ``loan_return``: the player goes back to the club he
    left in his previous move within ``max_days`` (~13 months). Rows without a date are dropped."""
    tr = transfers.dropna(subset=["transfer_date"]).sort_values(["player_id", "transfer_date"]).copy()
    g = tr.groupby("player_id")
    tr["loan_return"] = ((tr.to_club_id == g.from_club_id.shift())
                         & ((tr.transfer_date - g.transfer_date.shift()).dt.days < max_days))
    return tr
