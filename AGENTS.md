# Project Guide

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

## Data ownership

- Wyscout events and player features: Kareem
- Transfermarkt transfers, appearances, outcomes, and cross-dataset matching:
  Hussein
- Player-role features and clusters: Mohammed
- Team-style and weakness features: Yuri

## Current scope

This is the initial project structure. Training code, experiments, and
additional project documentation will be added incrementally.
