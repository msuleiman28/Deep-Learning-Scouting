"""Export the small datasets the milestone 1 hand-in notebook reads.

Run from the repository root:

    python -m src.pipelines.export_milestone1

Writes handin/milestone1/data/ (a few hundred KB, git-ignored). Zip the
handin/milestone1/ folder (notebook + data) to send it; the notebook then needs
only numpy, pandas, matplotlib and scikit-learn.

Files:
- transfers.csv: the 276 transfers with the label y, the baseline features and
  the fixed train/test split (same rows and order as transfers_joined.csv).
- player_features.csv: Wyscout 2017/18 players with the role-clustering features.
- funnel.csv: how the transfers were selected, including the match counts.
- market_value_one_year_later.csv: ONLY for the leakage demo; never a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.download_data import download_all
from src.features.player_roles import ROLE_FEATURES
from src.pipelines import build_transfers as bt

HANDIN = bt.ROOT / "handin/milestone1/data"

TRANSFER_COLUMNS = [
    "player_id", "wy_id", "player_name", "from_club_id", "from_club_name", "to_club_id", "to_club_name",
    "transfer_date", "position", "age", "log_mv", "prev_share_tm", "wy_share", "pass_acc",
    "passes_p90", "key_passes_p90", "assists_p90", "shots_p90", "goals_p90", "duels_p90",
    "duels_won_p90", "interceptions_p90", "fouls_p90", "y",
]
PLAYER_COLUMNS = ["wy_id", "shortName", "wy_team", "wy_role", "wy_minutes"] + ROLE_FEATURES


def market_value_after(data: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    """log market value on or before transfer_date + days. Post-transfer information: leakage demo only."""
    vals = pd.read_csv(bt.RAW / "transfermarkt/player_valuations.csv.gz", parse_dates=["date"])
    vals = vals.dropna(subset=["market_value_in_eur"]).sort_values("date")
    q = data[["player_id"]].assign(date=data.transfer_date + pd.Timedelta(days=days)).reset_index().sort_values("date")
    mv = pd.merge_asof(q, vals[["player_id", "date", "market_value_in_eur"]], on="date", by="player_id",
                       direction="backward").set_index("index").reindex(data.index)
    return pd.DataFrame({"player_id": data.player_id, "log_mv_1y_after": np.log1p(mv.market_value_in_eur)})


def main() -> None:
    download_all()
    if not all((bt.OUT / f).exists() for f in ["transfers_joined.csv", "transfer_funnel.csv"]):
        bt.main()
    data = pd.read_csv(bt.OUT / "transfers_joined.csv", parse_dates=["transfer_date"])
    split = pd.read_csv(bt.OUT / "split.csv", parse_dates=["transfer_date"])
    data = data.merge(split, on=bt.SPLIT_KEY, how="left", validate="one_to_one")  # keeps row order
    if data.split.isna().any():
        raise ValueError("some transfers are not in split.csv")

    HANDIN.mkdir(parents=True, exist_ok=True)
    data[TRANSFER_COLUMNS + ["split"]].to_csv(HANDIN / "transfers.csv", index=False)
    bt.load_player_features()[PLAYER_COLUMNS].to_csv(HANDIN / "player_features.csv", index=False)
    pd.read_csv(bt.OUT / "transfer_funnel.csv").to_csv(HANDIN / "funnel.csv", index=False)
    market_value_after(data).to_csv(HANDIN / "market_value_one_year_later.csv", index=False)

    sizes = {p.name: p.stat().st_size for p in sorted(HANDIN.glob("*.csv"))}
    print(f"Wrote {len(sizes)} files to {HANDIN} ({sum(sizes.values()) / 1e3:.0f} KB):")
    for name, size in sizes.items():
        print(f"  {name:36s} {size / 1e3:6.0f} KB")


if __name__ == "__main__":
    main()
