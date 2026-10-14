# Architecture

## Objective

For a candidate player and destination team, estimate the player's expected
normalized playing-time share during the first 12 months after a transfer and
use that estimate to rank candidates.

## Data flow

```text
Wyscout events ─────┐
                     ├─> player identity matching ─> transfer examples
Transfermarkt data ─┘                                  │
                                                       ├─> target: post-transfer
                                                       │    12-month playing-time share
                                                       ├─> player-role features/clusters
                                                       ├─> team-style features/clusters
                                                       └─> team weakness vector
                                                                  │
                                                                  v
                                              baseline ─> neural fit model
                                                                  │
                                                                  v
                                                   ranking and case studies
```

## Dataset boundaries

The Wyscout data primarily covers the 2017/18 season in the top five leagues,
while Transfermarkt supplies transfer and appearance records over a broader
scope. Because the sources use different identifiers, identity matching is a
first-class pipeline step and must produce an auditable match-rate report.

## Core entities

- `player`: source identifiers, name, position, age, and other stable
  attributes
- `club`: club identifier, season, team style, and weakness features
- `transfer`: player, origin club, destination club, transfer date, and
  pre-transfer information available at prediction time
- `appearance`: player-match minutes and match context
- `player_features`: event-derived and per-90 performance features
- `team_features`: style and weakness vectors
- `transfer_example`: joined model row containing input features and the
  observed 12-month target

## Target definition

For each eligible historical transfer, calculate:

```text
playing_time_share = minutes_played_by_player_in_first_12_months
                      / available_team_minutes_in_first_12_months
```

The exact availability rule and treatment of missing or postponed matches
should be documented when the outcome table is implemented.

## Modelling stages

1. Constant mean predictor as a sanity check.
2. Linear regression using age, position, market value, previous playing time,
   and basic performance features.
3. Player-role and team-style clustering for interpretability.
4. Neural player–team fit model using player strengths, team style, and team
   weaknesses.

Evaluation should use held-out transfers and mean squared error (MSE), with a
time-aware split preferred where the data volume permits it.
