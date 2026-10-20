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

## Initial modelling plan

1. Prepare Wyscout event and player features.
2. Prepare Transfermarkt transfer and appearance outcomes.
3. Match players and create one historical transfer example per row.
4. Calculate post-transfer playing-time share over the following 12 months.
5. Establish a mean predictor and linear-regression baseline.
6. Add role/style clusters, team weakness vectors, and the neural fit model.

The first data milestone is to report the cross-dataset match rate and the
number of eligible transfers. If fewer than approximately 300 reliable
examples remain, model capacity and feature count will be reduced.

## Downloading the data

From the repository root, on Windows, macOS, or Linux:

```bash
python src/data/download_data.py
```

This fetches the Wyscout files (`players.json`, `teams.json`, and
`matches_*.json` / `events_*.json` for England, France, Germany, Italy, and
Spain) into `data/raw/wyscout/`, and the six Transfermarkt tables into
`data/raw/transfermarkt/`. It needs only the Python standard library.

- Files already downloaded are skipped, so re-run it to retry any failures.
- Use `--force` to re-download everything.
- Transfermarkt is pinned to the snapshot published 2026-09-05 (data current
  to 2026-07-06). Every file is checked against a stored MD5, so everyone
  works on identical data. If a check fails, upstream has changed the file:
  talk to the team before updating `TM_MD5` in the script.
