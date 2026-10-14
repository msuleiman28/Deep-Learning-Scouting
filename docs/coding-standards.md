# Coding Standards

## General

- Use Python for data preparation, modelling, and analysis.
- Keep functions small and focused on one transformation.
- Use descriptive `snake_case` names for variables, functions, and files.
- Add type hints to reusable functions and docstrings where inputs or outputs
  are not obvious.
- Keep notebooks exploratory; move reusable logic into `src/`.

## Data and features

- Never modify files in `data/raw/`.
- Make feature names explicit about units and windows, for example
  `minutes_last_365d` or `passes_completed_per90`.
- Normalize per-90 features only when the denominator is meaningful, and keep
  the underlying minutes available for filtering and diagnostics.
- Fit preprocessing steps on the training split only.
- Attach a feature description and source column to any new model feature.

## Modelling

- Keep the constant mean predictor and linear regression available as named
  baselines.
- Set random seeds for clustering, training, and sampling when applicable.
- Store model inputs, target definition, split rule, and metrics with each
  experiment.
- Avoid using post-transfer information when constructing pre-transfer
  features.

## Documentation

- Update `README.md` when setup or project scope changes.
- Record assumptions about identity matching, target calculation, and data
  coverage near the relevant pipeline code.
