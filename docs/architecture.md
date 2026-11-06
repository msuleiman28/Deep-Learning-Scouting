# Architecture

## Objective

For a candidate player and destination team, estimate the player's expected
normalized playing-time share during the first 12 months after a transfer and
use that estimate to rank candidates.

## Data flow

```text
Wyscout events (2017/18) ─────────┐
EA FC ratings (2015/16–2022/23) ──┤
Understat stats (2014/15–2020/21) ┼─> player identity matching ─> transfer examples
Transfermarkt data ───────────────┘                                     │
ClubElo ratings ─────> club context (Elo, manager tenure) ──────────────┤
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
scope. Milestone 1 uses Wyscout 2017/18 (276 transfers), with EA FC ratings
as a light add-on that no deliverable depends on (`Plan.md`, decision D12).

After milestone 1, a multi-season dataset uses Transfermarkt player-season
features plus Understat and EA FC, both approved by the professor (decisions
D7, D8, D11):
- Understat (2014/15–2020/21 feature seasons): about 1,950 transfers.
- EA FC (2015/16–2022/23): about 2,800 transfers from big-5 selling clubs, and
  up to about 5,800 if selling clubs outside the big 5 are allowed.

Wyscout stays the rich, single-season layer. EA FC ratings are expert
assessments, not observed events, and use each player's latest update on or
before 15 May before the transfer window.

Because the sources use different identifiers, identity matching is a
first-class pipeline step and must produce an auditable match-rate report:
- Wyscout and EA FC are matched to Transfermarkt by date of birth and name.
- Understat is matched by name, club and season, because it has no dates of
  birth.

## Core entities

- `player`: source identifiers, name, position, age, and other stable
  attributes
- `club`: club identifier, season, team style, and weakness features
- `transfer`: player, origin club, destination club, transfer date, and
  pre-transfer information available at prediction time
- `appearance`: player-match minutes and match context
- `player_features`: event-derived and per-90 performance features
  (Wyscout), player-season stats (Transfermarkt, Understat), and EA FC
  attribute ratings
- `club_context`: Elo team strength and destination-manager tenure before the
  transfer
- `team_features`: style and weakness vectors
- `transfer_example`: joined model row containing input features and the
  observed 12-month target

## Target definition

For each eligible historical transfer, calculate:

```text
playing_time_share = minutes_played_by_player_in_first_12_months
                      / available_team_minutes_in_first_12_months
```

Implemented in `notebooks/side_work/01_join_and_baselines.ipynb` (`time_share`):
the player's league minutes for the new club ÷ (90 × the club's league matches
recorded by Transfermarkt in [transfer date, transfer date + 365 days)),
capped at 1. Only league matches count (Transfermarkt
`competition_type == "domestic_league"`; decision D2 in `Plan.md`), so
players aren't penalised for cup or European games they weren't registered
for. Clubs with fewer than 10 recorded league matches in the window get no
label and are dropped rather than given a fake 0.

## Modelling stages

1. Constant mean predictor as a sanity check.
2. Linear regression using age, position, market value, previous playing time,
   and basic performance features.
3. Player-role clustering (milestone 1), then team-style clustering, for
   interpretation and as inputs to team weakness features.
4. Neural player–team fit model using player strengths, player roles, team
   style, and team weaknesses.

Evaluation uses mean squared error (MSE). Test data is used once, for the
final evaluation; all model choices are made on training/validation data.
- **Milestone 1:** the fixed held-out split in `data/processed/split.csv`,
  with cross-validation on the training rows.
- **Later milestones:** a time-based split (decision D10 in `Plan.md`). Test
  on transfers after the 2020/21 season (Understat and EA FC features), or
  after 2022/23 (EA FC only, since Understat stops at 2020/21), and
  train/validate on earlier seasons.
