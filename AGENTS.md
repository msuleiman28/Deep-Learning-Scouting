# Project Guide

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
