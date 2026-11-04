# Football Player–Team Fit Scouting Model

Team: **Arab Favella**

This project develops an interpretable scouting model that ranks transfer
candidates by how well their strengths match a destination team's tactical
style and weaknesses.

The initial prediction target is a player's normalized playing-time share
during the first 12 months after a transfer. Player-role and team-style
clusters will support interpretation of the recommendations.

## Data sources

- [Wyscout Soccer Match Event Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000)
- [Transfermarkt datasets](https://github.com/dcaribou/transfermarkt-datasets)
- [ClubElo](https://clubelo.com/) ratings, via the
  [Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data) archive
- [Understat](https://understat.com/) player and shot data, via the published
  [scraping-understat-dataset](https://github.com/douglasbc/scraping-understat-dataset)
  copy (2014/15–2020/21), approved by the professor for this course project.
  We never scrape Understat ourselves or redistribute the files; see decision
  D8 in [Plan.md](Plan.md)
- [EA Sports FC 24 complete player dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
  (Kaggle, CC0): EA's player ratings for every game update from FIFA 15 to
  EA FC 24. These are expert ratings, not observed events. Used lightly in
  milestone 1 (decision D12) and a player-feature source for the multi-season
  dataset, approved by the professor (decision D11)

The data files are not committed to this repository. Keep downloaded or
generated data under the local `data/` directories described in the project
structure.

## Project structure

```text
.
├── data/
│   ├── raw/          # Original downloaded datasets
│   ├── interim/      # Temporary matching and cleaning outputs
│   └── processed/    # Model-ready datasets and feature tables
├── docs/
│   ├── architecture.md
│   └── coding-standards.md
├── notebooks/        # Exploratory analysis and experiments
├── src/
│   ├── data/         # Loading, cleaning, and cross-dataset matching
│   ├── features/     # Player, team, role, style, and weakness features
│   ├── models/       # Baselines and neural fit model
│   ├── pipelines/    # End-to-end training and scoring workflows
│   └── visualization/ # Clusters, diagnostics, and scouting outputs
└── AGENTS.md
```

## Plan and status

The full plan, status, open decisions, and evaluation rules are in
[Plan.md](Plan.md). Current state:

1. ✅ Wyscout ↔ Transfermarkt player matching: 98.5% of players matched.
2. ✅ Post-transfer 12-month playing-time label (league games only):
   **276** eligible transfers.
3. ✅ Constant-mean and linear-regression baselines on a fixed split
   (`data/processed/split.csv`). The constant mean is currently the bar to beat
   (cross-validation MSE 0.0815; test MSE 0.0973).
4. ⬜ Player-role clustering (completes milestone 1), with an optional EA FC
   check of the clusters and optional EA FC baseline features. Milestone 1
   doesn't depend on EA FC.
5. ⬜ Later milestones:
   - a multi-season dataset: Transfermarkt plus Understat (2014/15–2020/21,
     ~1,950 transfers) and EA FC (2015/16–2022/23, ~2,800 transfers
     from big-5 selling clubs)
   - team style features and clustering
   - team weakness features
   - a neural player–team fit model
   - evaluation and case studies

Milestone 1 has fewer than ~300 examples, so its model stays small; the
multi-season dataset removes that limit for later milestones.

## Notebooks

- `01_join_and_baselines.ipynb`: player matching, label, transfer funnel,
  baselines.
- `02_club_context_features.ipynb`: ClubElo team strength and destination
  manager features.

## Building feature tables

From the repository root, after downloading the data:

```bash
python -m src.pipelines.build_player_features   # data/processed/player_features.csv (~1 min)
```

One row per Wyscout player with 2017/18 league minutes: minutes, per-90 event
stats and location features (average position, width, final-third share,
crosses, dribbles, progressive passes). Definitions are in
`src/features/player_features.py`.

## Downloading the data

From the repository root, on Windows, macOS, or Linux:

```bash
python src/data/download_data.py
```

This fetches the Wyscout files (`players.json`, `teams.json`, and
`matches_*.json` / `events_*.json` for England, France, Germany, Italy, and
Spain) into `data/raw/wyscout/`, and the six Transfermarkt tables into
`data/raw/transfermarkt/`, ClubElo team-strength ratings into
`data/raw/clubelo/`, and Understat player-season and shot files (big 5,
2014/15–2020/21) into `data/raw/understat/`. It needs only the Python standard library.

- Files already downloaded are skipped, so re-run it to retry any failures.
- Use `--force` to re-download everything.
- EA FC isn't in the script yet. When it's added, it will need a Kaggle API
  token, with a manual download from the Kaggle page as the fallback (see
  [Plan.md](Plan.md) §4).
- Transfermarkt is pinned to the snapshot published 2026-09-05 (data current
  to 2026-07-06). Every file is checked against a stored MD5, so everyone
  works on identical data. If a check fails, upstream has changed the file:
  talk to the team before updating `TM_MD5` in the script.
