"""Build the milestone 1 transfer dataset: one row per eligible transfer, with the label.

Run from the repository root:

    python -m src.pipelines.build_transfers                # writes transfers_joined.csv, wyscout_tm_link.csv,
                                                           # transfer_funnel.csv
    python -m src.pipelines.build_transfers --write-split  # also (re)writes split.csv: only after a decision
                                                           # changes the rows (Plan.md section 4)

Steps (the same logic notebook 01 used to run inline):
1. Wyscout players with 2017/18 league minutes (``player_features.csv``, built
   if missing) are matched to Transfermarkt (``src/data/player_matching.py``).
2. Funnel: linked players' transfers dated WINDOW_START to WINDOW_END ->
   club-to-club moves -> no loan returns -> first move per player ->
   destination club with >= MIN_DEST_GAMES league games in the label window
   -> >= MIN_WY_MINUTES Wyscout minutes.
3. Label y and prev_share_tm (``src/features/labels.py``), plus baseline
   features known at the transfer date: age, position, log market value
   (latest valuation on or before the transfer), destination competition.

Time windows: Wyscout features are from 2017/18, which ended before
WINDOW_START; the label covers the 365 days after the transfer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.player_matching import match_players
from src.data.transfermarkt import load_tables
from src.features.labels import PlayingTime, flag_loan_returns

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"

WINDOW_START = "2018-05-15"   # after the Wyscout season ends, so features are strictly pre-transfer
WINDOW_END = "2019-02-15"     # summer 2018 + winter 2019 transfer windows
LABEL_DAYS = 365
LEAGUE_ONLY = True            # D2: label and prev_share_tm count league games only
MIN_DEST_GAMES = 10           # destination club needs >= 10 recorded league games in the label window
MIN_WY_MINUTES = 450          # >= 5 full matches of Wyscout events, so per-90 stats aren't noise
SEED, TEST_SIZE = 42, 0.2
BAD_DEST = r"retired|without club|unknown|career break"
SPLIT_KEY = ["player_id", "wy_id", "transfer_date", "to_club_id"]

# Wyscout columns carried into transfers_joined.csv, exactly as notebook 01 had them. Newer
# player_features.csv columns (location features) are joined by wy_id where needed, so adding a
# feature there never silently changes the baselines that use this file.
NB01_WY_COLUMNS = [
    "wy_id", "wy_minutes", "team_id", "league", "wy_share", "wy_team",
    "firstName", "middleName", "lastName", "shortName", "dob", "wy_role",
    "passes", "passes_acc", "key_passes", "assists", "shots", "goals", "duels", "duels_won",
    "interceptions", "fouls",
    "passes_p90", "key_passes_p90", "assists_p90", "shots_p90", "goals_p90", "duels_p90",
    "duels_won_p90", "interceptions_p90", "fouls_p90", "pass_acc",
]
NAME_COLUMNS = ["firstName", "middleName", "lastName", "shortName"]


def load_player_features() -> pd.DataFrame:
    """player_features.csv, built first if missing. Empty names come back from CSV as NaN, so they
    are restored to '' (NaN would put 'nan' into name variants and break matching)."""
    path = OUT / "player_features.csv"
    if not path.exists():
        from src.pipelines.build_player_features import main as build_player_features
        build_player_features()
    pf = pd.read_csv(path, parse_dates=["dob"])
    pf[NAME_COLUMNS] = pf[NAME_COLUMNS].fillna("")
    return pf


def load_inputs() -> dict:
    """Everything build_dataset needs: Transfermarkt tables, Wyscout features, the player link,
    the playing-time index and loan-return flags. Takes ~1 minute."""
    tm = load_tables(RAW / "transfermarkt")
    wy = load_player_features()[NB01_WY_COLUMNS]
    link = match_players(wy, tm["players"], tm["appearances"])
    return {
        "tm": tm, "wy": wy, "link": link,
        "pt": PlayingTime(tm["games"], tm["appearances"], link.player_id, LEAGUE_ONLY),
        "tr": flag_loan_returns(tm["transfers"]),
        "vals": tm["player_valuations"].dropna(subset=["date", "market_value_in_eur"]).sort_values("date"),
    }


def build_dataset(inp: dict, window_start: str = WINDOW_START, window_end: str = WINDOW_END,
                  min_dest_games: int = MIN_DEST_GAMES, min_wy_minutes: int = MIN_WY_MINUTES,
                  drop_loan_returns: bool = True) -> tuple[pd.DataFrame, dict[str, int]]:
    """Eligible transfers with label and baseline features, plus the funnel {step: count}."""
    wy, link, pt, tr = inp["wy"], inp["link"][["wy_id", "player_id"]], inp["pt"], inp["tr"]
    players = inp["tm"]["players"].set_index("player_id")
    clubs = inp["tm"]["clubs"]
    f = {}
    f["Wyscout players with 2017/18 league minutes"] = len(wy)
    f["  linked to a Transfermarkt player"] = len(link)
    t = tr.merge(link, on="player_id")
    f["transfers of linked players (all years)"] = len(t)
    t = t[(t.transfer_date >= window_start) & (t.transfer_date < window_end)]
    f[f"  dated {window_start} to {window_end}"] = len(t)
    t = t[t.to_club_id.notna() & (t.to_club_id != t.from_club_id)
          & ~t.to_club_name.fillna("").str.contains(BAD_DEST, case=False)]
    f["  club-to-club moves"] = len(t)
    if drop_loan_returns:
        t = t[~t.loan_return]
        f["  excluding loan returns"] = len(t)
    t = t.sort_values("transfer_date").drop_duplicates("player_id").reset_index(drop=True)
    f["  first move per player"] = len(t)
    lab = [pt.share(r.player_id, r.to_club_id, r.transfer_date, r.transfer_date + pd.Timedelta(days=LABEL_DAYS))
           for r in t.itertuples()]
    t["y"] = [x[0] for x in lab]
    t["dest_games"] = [x[1] for x in lab]
    t = t[t.dest_games >= min_dest_games].reset_index(drop=True)
    f[f"  destination club covered (>= {min_dest_games} games)"] = len(t)
    t["prev_share_tm"] = [pt.share(r.player_id, r.from_club_id, r.transfer_date - pd.Timedelta(days=365),
                                   r.transfer_date)[0] for r in t.itertuples()]
    t = t.merge(wy, on="wy_id")
    t = t[t.wy_minutes >= min_wy_minutes].reset_index(drop=True)
    f[f"  >= {min_wy_minutes} Wyscout minutes  ->  FINAL"] = len(t)

    # baseline features known at transfer time
    t["age"] = (t.transfer_date - t.player_id.map(players["dob"])).dt.days / 365.25
    t["position"] = t.player_id.map(players["position"]).fillna("Unknown")
    mv = pd.merge_asof(t[["player_id", "transfer_date"]].reset_index().sort_values("transfer_date"),
                       inp["vals"][["player_id", "date", "market_value_in_eur"]].rename(columns={"date": "transfer_date"}),
                       on="transfer_date", by="player_id", direction="backward").set_index("index")
    t["log_mv"] = np.log1p(mv["market_value_in_eur"].reindex(t.index))
    t["dest_comp"] = t.to_club_id.map(dict(zip(clubs["club_id"], clubs.get("domestic_competition_id"))))
    return t, f


def make_split(data: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Random train/test split of the transfers, keyed by SPLIT_KEY. Each player appears once
    (first move per player), so no player can be on both sides."""
    _, test = train_test_split(data, test_size=test_size, random_state=seed)
    return data.assign(split=np.where(data.index.isin(test.index), "test", "train"))[SPLIT_KEY + ["split"]]


def show_funnel(f: dict[str, int]) -> None:
    w = max(len(k) for k in f)
    for k, v in f.items():
        print(f"{k:<{w}}  {v:>7,}")


def main(out_dir: Path = OUT, write_split: bool = False) -> pd.DataFrame:
    inp = load_inputs()
    data, funnel = build_dataset(inp)
    show_funnel(funnel)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "transfers_joined.csv", index=False)
    inp["link"].to_csv(out_dir / "wyscout_tm_link.csv", index=False)
    pd.DataFrame(list(funnel.items()), columns=["step", "n"]).to_csv(out_dir / "transfer_funnel.csv", index=False)
    if write_split:
        make_split(data).to_csv(out_dir / "split.csv", index=False)
    print(f"Saved transfers_joined.csv, wyscout_tm_link.csv, transfer_funnel.csv" + (", split.csv" if write_split else "")
          + f" to {out_dir}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the milestone 1 transfer dataset.")
    parser.add_argument("--write-split", action="store_true",
                        help="also (re)write split.csv; only after a decision changes the rows")
    parser.add_argument("--out-dir", type=Path, default=OUT, help="output folder (default data/processed)")
    a = parser.parse_args()
    main(a.out_dir, a.write_split)
