"""Load the Transfermarkt tables (dcaribou/transfermarkt-datasets, pinned snapshot).

Files live under ``data/raw/transfermarkt/`` as ``<table>.csv`` or
``<table>.csv.gz`` and are fetched by ``src/data/download_data.py``.
Raw files are never modified here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = ["players", "transfers", "appearances", "games", "clubs", "player_valuations"]


def read_tm(tm_dir: str | Path, name: str) -> pd.DataFrame:
    """One Transfermarkt table, found anywhere under tm_dir (.csv or .csv.gz)."""
    for ext in (".csv", ".csv.gz"):
        hits = sorted(Path(tm_dir).rglob(name + ext))
        if hits:
            return pd.read_csv(hits[0], low_memory=False)
    raise FileNotFoundError(f"{name}.csv(.gz) not found under {tm_dir}")


def load_tables(tm_dir: str | Path) -> dict[str, pd.DataFrame]:
    """All tables, with date columns parsed and a normalised date of birth on players."""
    t = {name: read_tm(tm_dir, name) for name in TABLES}
    t["players"]["dob"] = pd.to_datetime(t["players"]["date_of_birth"], errors="coerce").dt.normalize()
    for name in ("appearances", "games", "player_valuations"):
        t[name]["date"] = pd.to_datetime(t[name]["date"], errors="coerce")
    t["transfers"]["transfer_date"] = pd.to_datetime(t["transfers"]["transfer_date"], errors="coerce")
    return t
