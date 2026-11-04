"""Build data/processed/player_features.csv from the raw Wyscout files.

Run from the repository root:

    python -m src.pipelines.build_player_features
"""

from __future__ import annotations

from pathlib import Path

from src.features.player_features import build_player_features

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    df = build_player_features(ROOT / "data/raw/wyscout")
    out = ROOT / "data/processed/player_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} players to {out} "
          f"({(df.wy_minutes >= 450).sum():,} with >= 450 minutes)")
    print(df.groupby("wy_role").size().rename("players").to_string())


if __name__ == "__main__":
    main()
