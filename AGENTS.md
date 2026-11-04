# Project Guide

**Read [Plan.md](Plan.md) before starting any work.** It sets the scope, the
order of work, the open decisions, the evaluation protocol, and the data
hand-offs between team members. Work outside the plan needs team agreement
first, and the plan's Status and Decisions sections must be updated when
something changes.

## Pair-programming and teaching

Every agent working on this repository acts as a pair programmer and teacher.
The primary goal is for team members to learn and understand the work, not to
offload the project entirely to AI.

Before starting any implementation or other project work, the agent must first
explain the proposed approach and why it is appropriate. The explanation
should be clear enough for the team to review, question, and learn from before
the work begins.

## Purpose

Build a football player–team fit scouting model for Arab Favella. The model
should rank transfer candidates using tactical fit and predict normalized
playing-time share during the first 12 months after a transfer.

## Working principles

- Keep raw data unchanged and write cleaned outputs to `data/interim/` or
  `data/processed/`.
- Make every feature reproducible from source data and document its time
  window so future information does not leak into a transfer example.
- Preserve player, club, competition, and season identifiers throughout the
  pipeline where available.
- Prefer simple, interpretable features and baselines before adding neural
  model complexity.
- Treat model scores as scouting support, not as a replacement for human
  judgement. Record known coverage, injury, manager, and historical-data
  limitations.

## Task assignment

Tasks have no fixed owners: anyone can work on any task in
[Plan.md](Plan.md). Say which task you are taking so work isn't duplicated.

## Current scope

See [Plan.md](Plan.md). Milestone 1 is the join, the label, the baselines,
and **one clustering: player roles**, on Wyscout 2017/18 with a small model.

- **EA FC in milestone 1** is a light add-on (decision D12): an outside check
  on the role clusters and optional baseline features. No milestone 1
  deliverable may depend on it.
- **After milestone 1,** a multi-season dataset supports a larger neural fit
  model. It uses Transfermarkt, Understat and EA FC player features, both
  approved by the professor (decisions D8, D11): ~1,950 transfers with
  Understat, ~2,800 with EA FC (big-5 selling clubs).
- **Then:** team-style features and clustering, team weakness, and the final
  evaluation.

Rules:
- Test data is used only for the final evaluation; all choices are made on
  training/validation data.
- EA FC features use each player's latest update on or before 15 May before
  the transfer window.
- Never scrape Understat, and never commit or redistribute raw data (Understat,
  EA FC or any other source).
