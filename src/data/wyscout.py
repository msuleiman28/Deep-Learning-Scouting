"""Load the public Wyscout dataset (2017/18 season, five leagues).

Files live under ``data/raw/wyscout/`` (subfolders are fine) and are fetched by
``src/data/download_data.py``. Raw files are never modified here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LEAGUES = ["England", "France", "Germany", "Italy", "Spain"]  # 2017/18 club seasons; World Cup / Euro skipped


def find_wy(wy_dir: str | Path, fname: str) -> Path:
    """Path of a Wyscout file anywhere under wy_dir."""
    hits = sorted(Path(wy_dir).rglob(fname))
    if not hits:
        raise FileNotFoundError(f"{fname} not found under {wy_dir}")
    return hits[0]


def load_json(wy_dir: str | Path, fname: str):
    with open(find_wy(wy_dir, fname), encoding="utf-8") as fh:
        return json.load(fh)


def fix_text(s) -> str:
    """Some Wyscout names arrive with literal '\\u00e9' escapes; decode them. Non-strings become ''."""
    if not isinstance(s, str):
        return ""
    if "\\u" in s:
        try:
            s = s.encode("latin-1", "backslashreplace").decode("unicode_escape")
        except Exception:
            pass
    return s


def load_players(wy_dir: str | Path) -> pd.DataFrame:
    """One row per player in players.json: wy_id, names, date of birth and role code (GK/DF/MD/FW)."""
    p = pd.DataFrame(load_json(wy_dir, "players.json"))
    for c in ["firstName", "middleName", "lastName", "shortName"]:
        p[c] = p[c].map(fix_text) if c in p else ""
    p["dob"] = pd.to_datetime(p["birthDate"], errors="coerce").dt.normalize()
    p["wy_role"] = p["role"].map(lambda r: r.get("code2") if isinstance(r, dict) else None)
    return p.rename(columns={"wyId": "wy_id"})[
        ["wy_id", "firstName", "middleName", "lastName", "shortName", "dob", "wy_role"]]


def load_team_names(wy_dir: str | Path) -> dict[int, str]:
    """Wyscout team_id -> team name."""
    t = pd.DataFrame(load_json(wy_dir, "teams.json"))
    return dict(zip(t["wyId"], t["name"].map(fix_text)))


def load_matches(wy_dir: str | Path, league: str) -> list[dict]:
    return load_json(wy_dir, f"matches_{league}.json")


def load_events(wy_dir: str | Path, league: str, columns: list[str]) -> pd.DataFrame:
    """Events of one league with only the requested columns (the full files are ~300 MB each)."""
    with open(find_wy(wy_dir, f"events_{league}.json"), encoding="utf-8") as fh:
        return pd.DataFrame(json.load(fh))[columns]
