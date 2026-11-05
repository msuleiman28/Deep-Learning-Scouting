"""Named baselines for the playing-time label, and one way to score them.

Every model must beat these on the same split (Plan.md section 4):
- ``MeanBaseline``: predicts the training mean of y for everyone (does no work).
- ``PreviousShareBaseline``: "he'll play as much as last year", i.e. y = the
  player's league playing-time share in the 365 days before the transfer
  (``prev_share_tm``); the training mean where that is unknown.
- ``GroupMeanBaseline``: the training mean of y within a group (e.g. player
  role); the overall training mean for unseen groups.
- ``linear_regression``: the main baseline model.

All are scikit-learn estimators on DataFrames, so cross_val_score works on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BIG_MISS = 0.25  # predicted playing-time share at least this much above the real one


class MeanBaseline(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class PreviousShareBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, column: str = "prev_share_tm"):
        self.column = column

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X[self.column].fillna(self.mean_).clip(0, 1).to_numpy()


class GroupMeanBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, column: str = "role"):
        self.column = column

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.group_means_ = pd.Series(np.asarray(y), index=X.index).groupby(X[self.column]).mean()
        return self

    def predict(self, X):
        return X[self.column].map(self.group_means_).fillna(self.mean_).to_numpy()


def linear_regression(num: list[str], cat: list[str]) -> Pipeline:
    """Median imputation (+ a 'was missing' flag) and z-scoring for numeric columns, one-hot for
    categorical ones, then ordinary least squares."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, LinearRegression())


def n_parameters(model: Pipeline) -> int:
    """Trainable parameters of a fitted linear pipeline: one weight per input column + the intercept."""
    lr = model[-1]
    return int(np.size(lr.coef_) + np.size(lr.intercept_))


def evaluate(models: dict[str, tuple], train: pd.DataFrame, test: pd.DataFrame, cv,
             target: str = "y") -> pd.DataFrame:
    """Score each (estimator, columns) on the same data: cross-validated MSE on the training rows,
    then MSE, RMSE and big overestimates on the test rows after fitting on all training rows."""
    rows = []
    for name, (est, cols) in models.items():
        cv_mse = -cross_val_score(est, train[cols], train[target], cv=cv, scoring="neg_mean_squared_error")
        est.fit(train[cols], train[target])
        pred = est.predict(test[cols])
        mse = mean_squared_error(test[target], pred)
        big = int((pred - test[target] >= BIG_MISS).sum())
        rows.append({"model": name, "cv_mse_mean": cv_mse.mean(), "cv_mse_std": cv_mse.std(),
                     "test_mse": mse, "test_rmse": np.sqrt(mse),
                     f"overestimated by >= {BIG_MISS}": f"{big} of {len(test)}"})
    return pd.DataFrame(rows).set_index("model")
