"""
pipeline/models.py
Models used inside each walk-forward fold.

Inputs are peer ranks in [0, 1] (missing -> 0.5, the peer median) and 0/1 flags (missing -> 0),
filled the same way in training and scoring. Interaction columns are centred-rank products
(a - .5)(b - .5) of the pairs that survived the IC combination search.

Direction models output raw scores; Calibrator maps them to probabilities with isotonic regression
fit on a purged calibration slice at the end of each training fold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


# ---------------------------------------------------------------- design matrix
def design(df: pd.DataFrame, cols: Sequence[str], pairs: Sequence[Tuple[str, str]] = ()) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in cols:
        x = df[c].astype(np.float32)
        X[c] = x.fillna(0.5 if c.endswith("_pr") else 0.0)
    for a, b in pairs:
        xa = df[a].astype(np.float32).fillna(0.5) if a.endswith("_pr") else df[a].astype(np.float32).fillna(0)
        xb = df[b].astype(np.float32).fillna(0.5) if b.endswith("_pr") else df[b].astype(np.float32).fillna(0)
        X[f"{a}__x__{b}"] = (xa - 0.5) * (xb - 0.5)
    return X


# ---------------------------------------------------------------- direction models
class LinearDir:
    """Standardized L2 logistic regression with explicit interaction columns. Exact SHAP via coef."""
    name = "linear"

    def __init__(self, C: float = 0.5):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.cols = list(X.columns)
        Z = self.scaler.fit_transform(X.to_numpy(np.float64))
        self.clf.fit(Z, y.to_numpy())
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X[self.cols].to_numpy(np.float64))
        return self.clf.predict_proba(Z)[:, 1]

    def shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact SHAP on the logit scale for a linear model with (assumed) independent inputs."""
        Z = self.scaler.transform(X[self.cols].to_numpy(np.float64))
        return pd.DataFrame(Z * self.clf.coef_[0], index=X.index, columns=self.cols)


class ForestDir:
    name = "rf"

    def __init__(self, seed: int = 42):
        self.clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=200,
                                          max_features="sqrt", max_samples=0.3, n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.clf.fit(X.to_numpy(np.float32), y.to_numpy())
        return self

    def score(self, X):
        return self.clf.predict_proba(X[self.cols].to_numpy(np.float32))[:, 1]

    @property
    def estimator(self):
        return self.clf


class BoostDir:
    name = "gbm"

    def __init__(self, seed: int = 42):
        if lgb is not None:
            self.clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                          min_child_samples=200, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      min_samples_leaf=200, random_state=seed)

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.clf.fit(X[self.cols], y.to_numpy())
        return self

    def score(self, X):
        return self.clf.predict_proba(X[self.cols])[:, 1]

    @property
    def estimator(self):
        return self.clf


class BinIsotonicDir:
    """Peer-percentile bins of the single best IC feature -> isotonic map (direction chosen by data)."""
    name = "bin_iso"

    def __init__(self, feature: str, n_bins: int = 10):
        self.feature, self.n_bins = feature, n_bins

    def fit(self, X, y):
        s = X[self.feature].to_numpy(float)
        self.edges = np.unique(np.quantile(s, np.linspace(0, 1, self.n_bins + 1)))
        b = np.digitize(s, self.edges[1:-1]).astype(float)
        self.iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(b, y.to_numpy(float))
        return self

    def score(self, X):
        b = np.digitize(X[self.feature].to_numpy(float), self.edges[1:-1]).astype(float)
        return self.iso.predict(b)


class Calibrator:
    """Isotonic map from raw score to probability, fit on the purged calibration slice."""

    def fit(self, s: np.ndarray, y: np.ndarray):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return self.iso.predict(s)


# ---------------------------------------------------------------- quantiles
class QuantileSet:
    """Gradient-boosted quantile regressions. Quantiles are re-sorted per row (rearrangement) so
    they never cross."""

    def __init__(self, quantiles: Sequence[float], seed: int = 42):
        self.quantiles = sorted(quantiles)
        self.seed = seed
        self.models: Dict[float, object] = {}

    def _make(self, q):
        if lgb is not None:
            return lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, learning_rate=0.03,
                                     num_leaves=31, min_child_samples=300, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, random_state=self.seed, verbose=-1)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                             learning_rate=0.05, min_samples_leaf=300, random_state=self.seed)

    def fit(self, X, y):
        self.cols = list(X.columns)
        for q in self.quantiles:
            self.models[q] = self._make(q).fit(X[self.cols], y.to_numpy(float))
        return self

    def predict(self, X) -> pd.DataFrame:
        P = np.column_stack([self.models[q].predict(X[self.cols]) for q in self.quantiles])
        P.sort(axis=1)  # rearrangement: fixes quantile crossing
        return pd.DataFrame(P, index=X.index, columns=[f"q{int(round(q * 100))}" for q in self.quantiles])


# ---------------------------------------------------------------- confidence
class AnomalyFlag:
    def __init__(self, contamination: float = 0.05, seed: int = 42, max_rows: int = 100_000):
        self.c, self.seed, self.max_rows = contamination, seed, max_rows

    def fit(self, X: pd.DataFrame):
        self.cols = list(X.columns)
        Xs = X.sample(min(len(X), self.max_rows), random_state=self.seed) if len(X) > self.max_rows else X
        self.clf = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(Xs.to_numpy(np.float32))
        sc = -self.clf.score_samples(Xs.to_numpy(np.float32))
        self.threshold = float(np.quantile(sc, 1 - self.c))
        return self

    def score(self, X) -> np.ndarray:
        return -self.clf.score_samples(X[self.cols].to_numpy(np.float32))

    def flag(self, X) -> np.ndarray:
        return np.where(self.score(X) > self.threshold, "low", "normal")
