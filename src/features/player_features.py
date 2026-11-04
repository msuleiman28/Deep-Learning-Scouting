"""Per-player Wyscout features for the 2017/18 league season.

Time window: every league match of the 2017/18 season in the five Wyscout
leagues. The season ended before the milestone 1 transfer window opens
(15 May 2018), so all features are pre-transfer.

Output (see ``build_player_features``): one row per player with league minutes
> 0, keyed by ``wy_id``. Minutes are kept so each consumer applies its own
threshold (e.g. >= 450 for the transfer dataset and the role clustering).

Feature definitions:
- ``wy_minutes``: league minutes from line-ups and substitutions (red cards
  ignored, so sent-off players are credited until the end of the match).
- ``wy_share``: wy_minutes / (90 x matches of the player's main team), capped at 1.
- ``<stat>_p90``: event count per 90 minutes; ``pass_acc``: accurate passes / passes.

Location features. Wyscout positions are % of the pitch from the acting team's
side: x = 0 own goal line, x = 100 opponent's; y runs between the touchlines.
- ``avg_x``: mean start x of the player's events (how far up the pitch he plays).
- ``avg_y``: mean start y (which side); ``avg_width``: mean |y - 50| (wide vs
  central, the same for left and right players, so better for roles).
- ``final_third_share``: share of his events starting at x >= 66.7.
- ``crosses``: passes with sub-event "Cross".
- ``dribbles``: ground attacking duels tagged as a take-on (tags 503/504;
  501/502 are "free space" runs and are not counted).
- ``progressive_passes``: accurate passes whose end is at least 25% closer to
  the opponent's goal centre than their start, measured in metres on a
  105 x 68 m pitch (x and y percentages are different lengths).
Location sums are added per league and averaged at the end, because players
who changed league mid-season appear in two leagues' event files. Events whose
start is a placeholder (0, 0) or (100, 100) are left out of the location
averages (goalkeeper events such as goal kicks and saves); corners keep their
real (100, 100) spot.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.wyscout import LEAGUES, load_events, load_matches, load_players, load_team_names

EVENT_COLUMNS = ["playerId", "eventName", "subEventName", "tags", "positions"]

# Wyscout tag ids used below
TAG_GOAL, TAG_ASSIST, TAG_KEY_PASS = 101, 301, 302
TAG_TAKE_ON_LEFT, TAG_TAKE_ON_RIGHT = 503, 504
TAG_WON, TAG_INTERCEPTION, TAG_ACCURATE = 703, 1401, 1801

PITCH_LENGTH_M, PITCH_WIDTH_M = 105, 68
FINAL_THIRD_X = 200 / 3          # x >= 66.7% of the pitch
PROGRESSIVE_RATIO = 0.75         # end distance to goal <= 75% of start distance
LOCATION_SUMS = ["n_located", "x_sum", "y_sum", "width_sum", "final_third_n"]  # turned into averages


# ---------------------------------------------------------------- minutes

def match_minutes(m: dict) -> list[tuple[int, int, int, int]]:
    """(player_id, team_id, match_id, minutes) for one match, from the line-up and substitutions."""
    rows = []
    for tid, td in m["teamsData"].items():
        form = td.get("formation") or {}
        start = {p["playerId"]: 0 for p in (form.get("lineup") or [])}
        end = {p: 90 for p in start}
        subs = form.get("substitutions")
        if isinstance(subs, list):  # Wyscout stores the string "null" when there were none
            for s in subs:
                minute = min(int(s["minute"]), 90)
                if s["playerOut"] in end:
                    end[s["playerOut"]] = minute
                start[s["playerIn"]] = minute
                end.setdefault(s["playerIn"], 90)
        for p in start:
            rows.append((p, int(tid), m["wyId"], max(end[p] - start[p], 0)))
    return rows


def league_minutes(wy_dir: str | Path, leagues: list[str] = LEAGUES) -> pd.DataFrame:
    """One row per (player, team, match) with minutes played and league."""
    rows = []
    for lg in leagues:
        for m in load_matches(wy_dir, lg):
            rows += [r + (lg,) for r in match_minutes(m)]
    mins = pd.DataFrame(rows, columns=["wy_id", "team_id", "match_id", "minutes", "league"])
    return mins[mins.wy_id != 0]  # playerId 0 = unknown player in some substitution records


def player_minutes(mins: pd.DataFrame) -> pd.DataFrame:
    """Per player: total minutes, main team (most minutes), its league, and share of that team's minutes."""
    team_matches = mins.groupby("team_id")["match_id"].nunique()
    main_team = (mins.groupby(["wy_id", "team_id", "league"])["minutes"].sum().reset_index()
                 .sort_values("minutes", ascending=False).drop_duplicates("wy_id").set_index("wy_id"))
    out = mins.groupby("wy_id")["minutes"].sum().rename("wy_minutes").to_frame()
    out = out.join(main_team[["team_id", "league"]])
    out = out[out.wy_minutes > 0].reset_index()
    out["wy_share"] = (out.wy_minutes / (90 * out.team_id.map(team_matches))).clip(upper=1)
    return out


# ---------------------------------------------------------------- events

def event_tags(ev: pd.DataFrame) -> pd.Series:
    """Set of tag ids per event."""
    return ev["tags"].map(lambda t: {d["id"] for d in t})


def event_counts(ev: pd.DataFrame, tags: pd.Series) -> pd.DataFrame:
    """Per-player counts of passes, shots, duels, etc. from one league's events (playerId 0 removed)."""
    has = lambda tag: tags.map(lambda s: tag in s)
    is_pass = ev.eventName.eq("Pass")
    is_shot = ev.eventName.eq("Shot") | ev.subEventName.isin(["Penalty", "Free kick shot"])
    is_duel = ev.eventName.eq("Duel")
    return pd.DataFrame({
        "wy_id": ev.playerId.values,
        "passes": is_pass.values,
        "passes_acc": (is_pass & has(TAG_ACCURATE)).values,
        "key_passes": has(TAG_KEY_PASS).values,
        "assists": has(TAG_ASSIST).values,
        "shots": is_shot.values,
        "goals": (is_shot & has(TAG_GOAL)).values,
        "duels": is_duel.values,
        "duels_won": (is_duel & has(TAG_WON)).values,
        "interceptions": has(TAG_INTERCEPTION).values,
        "fouls": ev.eventName.eq("Foul").values,
    }).groupby("wy_id").sum()


def _goal_distance_m(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance in metres from a point (pitch %) to the centre of the opponent's goal (100, 50)."""
    return np.hypot((100 - x) * PITCH_LENGTH_M / 100, (50 - y) * PITCH_WIDTH_M / 100)


def location_sums(ev: pd.DataFrame, tags: pd.Series) -> pd.DataFrame:
    """Per-player location sums and location-based counts from one league's events.

    Sums (not averages) so leagues can be added together; see ``add_location_averages``.
    """
    pos = ev["positions"]
    x0 = pos.map(lambda p: p[0]["x"]).to_numpy(float)
    y0 = pos.map(lambda p: p[0]["y"]).to_numpy(float)
    x1 = pos.map(lambda p: p[-1]["x"]).to_numpy(float)  # single-position events: end = start
    y1 = pos.map(lambda p: p[-1]["y"]).to_numpy(float)
    is_pass = ev.eventName.eq("Pass").to_numpy()
    accurate = tags.map(lambda s: TAG_ACCURATE in s).to_numpy()
    take_on = tags.map(lambda s: TAG_TAKE_ON_LEFT in s or TAG_TAKE_ON_RIGHT in s).to_numpy()
    progressive = (is_pass & accurate
                   & (_goal_distance_m(x1, y1) <= PROGRESSIVE_RATIO * _goal_distance_m(x0, y0)))
    # Goal kicks, reflex saves, save attempts and "goalkeeper leaving line" carry a placeholder start of
    # (0, 0) or (100, 100) instead of a real position; keep them out of the location averages.
    # Corners also start at (100, 100), but that is their true spot (the corner flag).
    placeholder = ((x0 == 0) & (y0 == 0)) | ((x0 == 100) & (y0 == 100))
    located = ~placeholder | ev.subEventName.eq("Corner").to_numpy()
    return pd.DataFrame({
        "wy_id": ev.playerId.to_numpy(),
        "n_located": located,
        "x_sum": np.where(located, x0, 0),
        "y_sum": np.where(located, y0, 0),
        "width_sum": np.where(located, np.abs(y0 - 50), 0),
        "final_third_n": located & (x0 >= FINAL_THIRD_X),
        "crosses": (is_pass & ev.subEventName.eq("Cross").to_numpy()),
        "dribbles": (ev.subEventName.eq("Ground attacking duel").to_numpy() & take_on),
        "progressive_passes": progressive,
    }).groupby("wy_id").sum()


def add_location_averages(df: pd.DataFrame) -> None:
    """Turn the location sums into avg_x, avg_y, avg_width and final_third_share, then drop the sums."""
    n = df["n_located"].replace(0, np.nan)
    df["avg_x"] = df["x_sum"] / n
    df["avg_y"] = df["y_sum"] / n
    df["avg_width"] = df["width_sum"] / n
    df["final_third_share"] = df["final_third_n"] / n
    df.drop(columns=LOCATION_SUMS, inplace=True)


def player_event_features(wy_dir: str | Path, leagues: list[str] = LEAGUES) -> pd.DataFrame:
    """Event counts and location sums per player, summed over leagues. Each league's events are
    loaded once and freed, so every per-event feature is computed in this one loop."""
    parts = []
    for lg in leagues:
        ev = load_events(wy_dir, lg, EVENT_COLUMNS)
        ev = ev[ev.playerId != 0].reset_index(drop=True)
        tags = event_tags(ev)
        parts.append(event_counts(ev, tags).join(location_sums(ev, tags), how="outer"))
        del ev, tags
        gc.collect()
    return pd.concat(parts).groupby(level=0).sum()


def add_rates(df: pd.DataFrame, count_cols: list[str]) -> list[str]:
    """Add <count>_p90 for every count except passes_acc, plus pass_acc. Returns the per-90 column names."""
    p90 = []
    for c in count_cols:
        if c != "passes_acc":
            df[c + "_p90"] = df[c] / df.wy_minutes * 90
            p90.append(c + "_p90")
    df["pass_acc"] = df.passes_acc / df.passes.replace(0, np.nan)
    return p90


# ---------------------------------------------------------------- table

def build_player_features(wy_dir: str | Path, leagues: list[str] = LEAGUES) -> pd.DataFrame:
    """One row per player with 2017/18 league minutes > 0: identity, minutes, counts, per-90 rates."""
    pm = player_minutes(league_minutes(wy_dir, leagues))
    pm["wy_team"] = pm.team_id.map(load_team_names(wy_dir))
    df = pm.merge(load_players(wy_dir), on="wy_id", how="left")
    events = player_event_features(wy_dir, leagues)
    df = df.merge(events, left_on="wy_id", right_index=True, how="left")
    df[list(events.columns)] = df[list(events.columns)].fillna(0)
    count_cols = [c for c in events.columns if c not in LOCATION_SUMS]
    add_rates(df, count_cols)
    add_location_averages(df)
    df.insert(1, "season", "2017/18")
    return df
